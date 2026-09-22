import json
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


class TuningCliTests(unittest.TestCase):
    def test_partial_dates_are_rejected_before_loading_data(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for dates in (("2025", "2025", "2026", "2026"),
                          ("2025-07", "2025-12", "2026-01", "2026-07")):
                with self.subTest(dates=dates):
                    output = directory / dates[-1]
                    run = subprocess.run(
                        [sys.executable, str(root / "run_tuning.py"), "--csv", str(directory / "absent.csv"),
                         "--screen-start", dates[0], "--screen-end", dates[1],
                         "--validation-start", dates[2], "--validation-end", dates[3],
                         "--output", str(output)], cwd=root, capture_output=True, text=True, timeout=30,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})
                    self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
                    self.assertIn("YYYY-MM-DD", run.stderr)
                    self.assertNotIn("FileNotFoundError", run.stderr)
                    self.assertFalse(output.exists())

    def test_reserved_final_test_dates_cannot_enter_parameter_selection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dates = pd.bdate_range("2026-07-01", periods=64)
            frame = pd.DataFrame(np.random.default_rng(73).normal(.05, .8, (64, 3)),
                                 index=dates.strftime("%Y%m%d"), columns=list("ABC"))
            csv = directory / "input.csv"
            csv.write_text("Average Equal Weighted Returns -- Daily\n" + frame.to_csv() + "\n")
            for start, end in (("2026-07-30", "2026-08-01"),
                               ("2026-08-03", "2026-08-07"),
                               ("2026-09-01", "2026-09-04")):
                with self.subTest(start=start, end=end):
                    output = directory / end
                    run = subprocess.run(
                        [sys.executable, str(root / "run_tuning.py"), "--csv", str(csv),
                         "--window", "4", "--screen-start", "2026-07-13", "--screen-end", "2026-07-17",
                         "--validation-start", start, "--validation-end", end,
                         "--max-candidates", "9", "--output", str(output)],
                        cwd=root, capture_output=True, text=True, timeout=90,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})
                    self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
                    self.assertIn("reserved final test", run.stderr)
                    self.assertFalse(output.exists(), "Invalid periods must fail before fitting or writing results")

    def test_all_failed_candidates_are_logged_and_failed_fits_are_not_retried(self):
        import run_tuning
        from portfolio.selection import GrowthParameters, SearchConfig, search_growth
        with tempfile.TemporaryDirectory() as temporary:
            dates = pd.bdate_range("2026-01-01", periods=8)
            returns = pd.DataFrame(np.zeros((8, 2)), index=dates, columns=["A", "B"])
            evaluator = run_tuning.PhaseEvaluator(returns, start=str(dates[4].date()),
                end=str(dates[-1].date()), window=4, output=Path(temporary),
                constraints=run_tuning.PortfolioConstraints())
            candidates = [GrowthParameters(0, 0), GrowthParameters(.01, 0)]
            with patch("portfolio.engine.growth_search_evaluation.fit_portfolio_growth", side_effect=RuntimeError("numerical failure")) as fit:
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "All initial"):
                    search_growth(evaluator, config=SearchConfig(max_candidates=2), initial_parameters=candidates)
                attempts_path = Path(temporary) / "candidate_attempts.csv"
                self.assertTrue(attempts_path.exists(), "Failed search must preserve candidate errors")
                attempts = pd.read_csv(attempts_path)
                self.assertEqual(list(attempts["status"]), ["failed", "failed"])
                self.assertTrue(attempts["error"].str.contains("numerical failure").all())
                with self.assertRaises(RuntimeError):
                    evaluator.fit(candidates[0])
                self.assertEqual(fit.call_count, 2, "An excluded failed fit must not be retried during comparisons")

    def test_failed_control_does_not_discard_successful_selection(self):
        import run_tuning
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dates = pd.bdate_range("2025-12-01", periods=28)
            frame = pd.DataFrame(np.random.default_rng(246).normal(.05, .8, (28, 3)),
                                 index=dates.strftime("%Y%m%d"), columns=list("ABC"))
            csv = directory / "input.csv"
            csv.write_text("Average Equal Weighted Returns -- Daily\n" + frame.to_csv() + "\n")
            output = directory / "output"
            argv = ["run_tuning.py", "--csv", str(csv), "--window", "10",
                    "--screen-start", str(dates[10].date()), "--screen-end", str(dates[13].date()),
                    "--validation-start", str(dates[14].date()), "--validation-end", str(dates[17].date()),
                    "--max-candidates", "9", "--output", str(output)]
            real_fit = run_tuning.fit_portfolio_growth
            failed_calls = []

            def fail_control(history, **options):
                loss = options["loss"]
                if loss.reward_strength == 0 and loss.exposure_strength == 0 and history.index[-1] >= dates[13]:
                    failed_calls.append(history.index[-1])
                    raise RuntimeError("control solver failed")
                return real_fit(history, **options)

            with patch.object(sys, "argv", argv), patch("portfolio.engine.growth_search_evaluation.fit_portfolio_growth", side_effect=fail_control), redirect_stdout(io.StringIO()):
                run_tuning.main()
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertIn("LASSO_Constrained", metadata["comparison_failures"])
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            self.assertEqual(summary.loc["LASSO_Constrained", "status"], "failed")
            self.assertEqual(summary.loc["LASSO_Selected", "status"], "ok")
            self.assertEqual(len(failed_calls), 1)

    def test_complete_search_exports_and_future_data_cannot_change_selection(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "run_tuning.py"
        self.assertTrue(script.exists(), "The parameterized LASSO experiment entry point is missing")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dates = pd.bdate_range("2025-12-01", periods=28)
            frame = pd.DataFrame(np.random.default_rng(246).normal(.05, .8, (28, 3)),
                                 index=dates.strftime("%Y%m%d"), columns=list("ABC"))
            csv = directory / "input.csv"

            def write_csv(data):
                csv.write_text("Average Equal Weighted Returns -- Daily\n" + data.to_csv() + "\n")

            write_csv(frame)
            options = [sys.executable, str(script), "--csv", str(csv), "--window", "10",
                       "--screen-start", str(dates[10].date()), "--screen-end", str(dates[13].date()),
                       "--validation-start", str(dates[14].date()),
                       "--validation-end", str(dates[17].date()), "--max-candidates", "9"]

            def run(output, extra=()):
                return subprocess.run(options + ["--output", str(output), *extra], cwd=root,
                                      capture_output=True, text=True, timeout=90,
                                      env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

            output = directory / "first"
            first = run(output)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["lasso_strength"], 1e-5)
            self.assertEqual(metadata["window"], 10)
            self.assertLessEqual(metadata["distinct_search_configurations"], 9)
            self.assertEqual(metadata["screen"]["evaluations"], 9)
            self.assertEqual(metadata["validation"]["evaluation_days"], 4)
            self.assertEqual(len(metadata["data_sha256"]), 64)
            self.assertEqual(metadata["evaluation_role"], "development_validation")
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            self.assertEqual(set(summary.index), {"EW", "LASSO", "LASSO_Constrained",
                "LASSO_LogGrowth", "Selected_RewardOnly", "Selected_ExposureOnly", "LASSO_Selected"})
            selected_returns = pd.read_csv(output / "LASSO_Selected_returns.csv", index_col=0)
            weights = pd.read_csv(output / "LASSO_Selected_weights.csv", index_col=0)
            diagnostics = pd.read_csv(output / "LASSO_Selected_diagnostics.csv", index_col=0)
            self.assertEqual(list(weights.index), list(dates[14:18].strftime("%Y-%m-%d")))
            assert_allclose(weights.sum(axis=1), 1, atol=1e-8)
            self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.5 + 1e-7)
            actual = (frame.iloc[14:18].to_numpy() / 100 * weights.to_numpy()).sum(axis=1)
            assert_allclose(selected_returns.iloc[:, 0], actual, atol=1e-10)
            self.assertAlmostEqual(summary.loc["LASSO_Selected", "cumulative_return"],
                                   float(np.prod(1 + actual) - 1))
            self.assertTrue((pd.to_datetime(diagnostics["history_end"]) < pd.to_datetime(diagnostics.index)).all())
            self.assertIn("exposure_penalty", diagnostics)
            for phase in ("screen", "validation"):
                history = pd.read_csv(output / f"{phase}_search.csv")
                for row in history.loc[history["status"] == "ok"].itertuples():
                    candidate_returns = pd.read_csv(output / phase / f"candidate_{row.candidate_id:02d}_returns.csv")
                    self.assertEqual(len(candidate_returns), 4)

            snapshot = {str(path.relative_to(output)): path.read_bytes()
                        for path in output.rglob("*") if path.is_file()}
            repeated = run(output)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(snapshot, {str(path.relative_to(output)): path.read_bytes()
                                       for path in output.rglob("*") if path.is_file()})

            changed = frame.copy()
            changed.iloc[18:] += 500
            write_csv(changed)
            future_output = directory / "future_changed"
            second = run(future_output)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            future_metadata = json.loads((future_output / "run_metadata.json").read_text())
            self.assertEqual(metadata["selected_parameters"], future_metadata["selected_parameters"])
            for name in ("screen_search.csv", "validation_search.csv", "LASSO_Selected_weights.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(output / name), pd.read_csv(future_output / name))

            write_csv(frame)
            weekend_output = directory / "weekend_end"
            weekend = run(weekend_output, ["--screen-end", str((dates[14] + pd.Timedelta(days=2)).date()),
                                          "--validation-start", str(dates[15].date()),
                                          "--validation-end", str((dates[19] + pd.Timedelta(days=2)).date())])
            self.assertEqual(weekend.returncode, 0, weekend.stdout + weekend.stderr)
            weekend_returns = pd.read_csv(weekend_output / "LASSO_Selected_returns.csv", index_col=0)
            self.assertEqual(list(weekend_returns.index), list(dates[15:20].strftime("%Y-%m-%d")))

            invalid_output = directory / "invalid_dates"
            invalid = run(invalid_output, ["--screen-end", str(dates[14].date())])
            self.assertEqual(invalid.returncode, 2)
            self.assertFalse(invalid_output.exists())


if __name__ == "__main__":
    unittest.main()
