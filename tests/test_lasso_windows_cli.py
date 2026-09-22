import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.data import load_returns
from portfolio.evaluation import backtest
from portfolio.regularization import Regularization
from portfolio.training import fit_portfolio


ROOT = Path(__file__).resolve().parents[1]


class LassoWindowsCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_lasso_windows.py"), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})

    def test_fixed_lasso_presets_export_reproducible_history_only_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dates = pd.bdate_range("2020-01-01", "2026-08-31")
            raw = pd.DataFrame(np.random.default_rng(12).normal(.04, 1, (len(dates), 3)),
                               index=dates, columns=list("ABC"))
            # Data outside the five-year history must never reach training or evaluation.
            raw.loc[:"2021-07-31"] = -99.99
            raw.loc["2026-08-01":] = -99.99
            raw.index = raw.index.strftime("%Y%m%d")
            csv = directory / "input.csv"
            csv.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")
            output = directory / "output"
            args = ["--csv", csv, "--start", "2026-01-02", "--end", "2026-06-30",
                    "--presets", "126d", "252d", "3y", "3y_weighted", "--output", output]
            run = self.run_cli(*args)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            self.assertEqual(set(summary.index), {"LASSO_126d", "LASSO_252d", "LASSO_3y", "LASSO_3y_weighted"})
            self.assertTrue((output / "validation_summary.csv").exists())
            data = load_returns(csv, start="2021-08-01", end="2026-07-31")
            strategy = lambda history: fit_portfolio(history, regularization=Regularization("lasso", 1e-5)).weights
            baseline = backtest(data, strategy, start="2026-01-02", end="2026-06-30", window=252)
            stored = pd.read_csv(output / "LASSO_252d_returns.csv", index_col=0)
            assert_allclose(stored.iloc[:, 0], baseline.returns, rtol=1e-10, atol=1e-14)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertFalse(metadata["final_test"]["evaluated"])
            self.assertFalse(metadata["model"]["hyperparameter_search"])
            self.assertEqual(metadata["model"]["lasso_strength"], 1e-5)
            self.assertEqual(metadata["history"]["actual_start"], "2021-08-02")
            self.assertEqual(metadata["history"]["actual_end"], "2026-07-31")
            self.assertEqual(len(metadata["data_sha256"]), 64)
            diagnostics = pd.read_csv(output / "LASSO_3y_weighted_diagnostics.csv", index_col=0, parse_dates=True)
            self.assertTrue((pd.to_datetime(diagnostics["history_end"]) < diagnostics.index).all())
            self.assertEqual(diagnostics.iloc[0]["history_start"], "2023-01-02")
            self.assertTrue((diagnostics["effective_observations"] < diagnostics["observations"]).all())
            holdings = pd.read_csv(output / "LASSO_3y_weighted_weights.csv", index_col=0, parse_dates=True)
            realized = (holdings * data.loc[holdings.index]).sum(axis=1)
            self.assertAlmostEqual(summary.loc["LASSO_3y_weighted", "cumulative_return"],
                                   (1 + realized).prod() - 1, places=12)
            snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
            repeat = self.run_cli(*args)
            self.assertEqual(repeat.returncode, 2, repeat.stdout + repeat.stderr)
            self.assertEqual(snapshot, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_constrained_variants_use_existing_limits_without_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dates = pd.bdate_range("2021-08-02", "2026-01-05")
            z = np.random.default_rng(71).normal(0, 2, len(dates))
            # B=2*A: unconstrained MinVar needs w_A near 2, while gross<=1.5
            # forces the constrained optimum to [1.25, -.25].
            raw = pd.DataFrame({"A": z, "B": 2 * z}, index=dates.strftime("%Y%m%d"))
            csv = directory / "input.csv"
            csv.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")
            output = directory / "output"
            run = self.run_cli("--csv", csv, "--start", "2026-01-02", "--end", "2026-01-05",
                               "--presets", "252d", "3y", "--constrained", "--output", output)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            self.assertEqual(set(summary.index), {"LASSO_252d", "LASSO_3y",
                                                  "LASSO_252d_Constrained", "LASSO_3y_Constrained"})
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertFalse(metadata["final_test"]["evaluated"])
            self.assertEqual(metadata["model"]["reward_strength"], 0)
            self.assertEqual(metadata["model"]["exposure_strength"], 0)
            data = load_returns(csv)
            for preset in ("252d", "3y"):
                plain, limited = f"LASSO_{preset}", f"LASSO_{preset}_Constrained"
                self.assertGreater(summary.loc[plain, "average_gross_leverage"], 2)
                weights = pd.read_csv(output / f"{limited}_weights.csv", index_col=0, parse_dates=True)
                assert_allclose(weights.to_numpy(), [[1.25, -.25], [1.25, -.25]], atol=2e-5)
                diagnostics = pd.read_csv(output / f"{limited}_diagnostics.csv", index_col=0, parse_dates=True)
                self.assertTrue((diagnostics["solver_status"] == "optimal").all())
                self.assertTrue((diagnostics["reward"] == 0).all())
                self.assertTrue((diagnostics["exposure_penalty"] == 0).all())
                self.assertIsNone(metadata["training"][plain]["constraints"])
                self.assertEqual(metadata["training"][limited]["constraints"]["leverage_limit"], 1.5)
                for prediction, row in diagnostics.iterrows():
                    history = data.loc[row.history_start:row.history_end]
                    self.assertLess(pd.Timestamp(row.history_end), prediction)
                    w = weights.loc[prediction].to_numpy()
                    realized_history = history.to_numpy() @ w
                    expected_variance = np.var(realized_history)
                    expected_loss = expected_variance + 1e-5 * abs(.5 - w[0])
                    self.assertAlmostEqual(row.objective, expected_loss, places=10)
                    self.assertLessEqual(expected_variance, np.var(history.mean(axis=1)) + 1e-10)
                    self.assertGreaterEqual(np.min(1 + realized_history), 1e-6 - 1e-8)


    def test_test_dates_and_partial_dates_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for start, end in (("2026-01-02", "2026-08-03"), ("2026-01", "2026-06-30"),
                               ("2026-01-02", "2026-08"), ("2026-06-30", "2026-01-02")):
                run = self.run_cli("--csv", "/nonexistent.csv", "--start", start, "--end", end,
                                   "--output", output)
                self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
                self.assertNotIn("FileNotFoundError", run.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
