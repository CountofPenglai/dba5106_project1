import importlib
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


ROOT = Path(__file__).resolve().parents[1]


class WeightSelectionTests(unittest.TestCase):
    def test_selects_highest_return_and_prefers_slower_decay_on_a_tie(self):
        select = importlib.import_module("run_weight_tuning").select_half_life
        self.assertEqual(select({126: .12, 252: .15, 504: .11}), 252)
        self.assertEqual(select({126: .12, 252: .12, 504: .12}), 504)

    def test_rejects_invalid_scores_instead_of_silently_selecting(self):
        select = importlib.import_module("run_weight_tuning").select_half_life
        for scores in ({}, {0: .1}, {True: .1}, {126: np.nan}, {126: np.inf}, {126: True}, {126.5: .1}):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select(scores)


class WeightTuningCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_weight_tuning.py"), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"})

    def test_validation_selects_weights_and_excludes_later_data(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dates = pd.bdate_range("2020-01-01", "2026-08-31")
            z = np.random.default_rng(83).normal(size=(len(dates), 2))
            scale = np.where((dates >= "2025-01-01")[:, None], [.5, 2.0], [2.0, .5])
            raw = pd.DataFrame(z * scale, index=dates, columns=["A", "B"])
            raw.loc[:"2021-07-31"] = -99.99
            raw.loc["2026-08-01":] = -99.99
            csv = directory / "input.csv"
            def write_csv():
                export = raw.copy()
                export.index = export.index.strftime("%Y%m%d")
                csv.write_text("Average Equal Weighted Returns -- Daily\n" + export.to_csv() + "\n")
            write_csv()
            output = directory / "output"
            args = ["--csv", csv, "--validation-start", "2026-01-02", "--validation-end", "2026-01-06",
                    "--half-lives", "126", "504"]
            run = self.run_cli(*args, "--output", output)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            candidates = {126: "LASSO_3y_HL126_Constrained", 504: "LASSO_3y_HL504_Constrained"}
            self.assertEqual(set(summary.index), {"LASSO_252d_Constrained", "LASSO_3y_Constrained", *candidates.values()})
            selection = json.loads((output / "selection.json").read_text())
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["candidate_half_lives"], [126, 504])
            self.assertFalse(metadata["final_test"]["evaluated"])
            self.assertEqual(selection["selection_metric"], "cumulative_return")
            self.assertEqual(selection["evaluation_role"], "validation_used_for_selection")
            data = load_returns(csv)
            independently_computed_scores = {}
            observed = {}
            for half_life, name in candidates.items():
                weights = pd.read_csv(output / f"{name}_weights.csv", index_col=0, parse_dates=True)
                observed[name] = weights
                diagnostics = pd.read_csv(output / f"{name}_diagnostics.csv", index_col=0, parse_dates=True)
                for prediction, row in diagnostics.iterrows():
                    history = data.loc[(data.index >= prediction-pd.DateOffset(years=3)) & (data.index < prediction)]
                    self.assertEqual(row.history_start, str(history.index[0].date()))
                    self.assertEqual(row.history_end, str(history.index[-1].date()))
                    q = 2.0 ** (-np.arange(len(history)-1, -1, -1) / half_life)
                    q /= q.sum()
                    # Independent two-asset weighted LASSO optimum, then gross cap.
                    difference = history.A.to_numpy() - history.B.to_numpy()
                    equal = history.mean(axis=1).to_numpy()
                    x = difference - q @ difference
                    y = equal - q @ equal
                    cov, var = q @ (x*y), q @ (x*x)
                    beta = np.sign(cov) * max(abs(cov)-1e-5/2, 0) / var
                    first = np.clip(.5-beta, -.25, 1.25)
                    assert_allclose(weights.loc[prediction], [first, 1-first], atol=2e-5)
                    self.assertAlmostEqual(row.effective_observations, 1/(q@q), places=8)
                    w = weights.loc[prediction].to_numpy()
                    r = history.to_numpy() @ w
                    variance = q @ (r-q@r)**2
                    ew_variance = q @ (equal-q@equal)**2
                    self.assertAlmostEqual(row.variance_limit, ew_variance, places=12)
                    self.assertAlmostEqual(row.objective, variance + 1e-5*abs(.5-w[0]), places=10)
                    self.assertEqual(row.reward, 0)
                    self.assertEqual(row.exposure_penalty, 0)
                    self.assertLessEqual(np.abs(w).sum(), 1.5+1e-7)
                realized = (data.loc[weights.index]*weights).sum(axis=1)
                score = (1+realized).prod()-1
                independently_computed_scores[half_life] = score
                self.assertAlmostEqual(summary.loc[name, "cumulative_return"], score, places=12)
            self.assertFalse(np.allclose(observed[candidates[126]], observed[candidates[504]], atol=1e-3))
            expected = max(independently_computed_scores, key=lambda h: (independently_computed_scores[h], h))
            self.assertEqual(selection["selected_half_life"], expected)
            self.assertEqual(selection["training"]["half_life"], expected)
            self.assertEqual(selection["training"]["window"]["years"], 3)
            self.assertEqual(selection["constraints"]["leverage_limit"], 1.5)
            self.assertEqual(selection["reward_strength"], 0)
            self.assertEqual(selection["exposure_strength"], 0)
            self.assertEqual(int(summary["selected"].sum()), 1)
            snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
            repeat = self.run_cli(*args, "--output", output)
            self.assertEqual(repeat.returncode, 2, repeat.stdout+repeat.stderr)
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in output.iterdir()})
            # A different July/August or any day after validation cannot alter selection or fits.
            raw.loc["2026-01-07":] = -99.99
            write_csv()
            changed_output = directory / "future_changed"
            changed = self.run_cli(*args, "--output", changed_output)
            self.assertEqual(changed.returncode, 0, changed.stdout+changed.stderr)
            changed_selection = json.loads((changed_output / "selection.json").read_text())
            self.assertEqual(selection, changed_selection)
            for name, before in observed.items():
                after = pd.read_csv(changed_output / f"{name}_weights.csv", index_col=0, parse_dates=True)
                assert_allclose(before, after, atol=1e-12, rtol=0)

    def test_bad_dates_and_grids_are_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for extra in (["--validation-end", "2026-08-03"], ["--validation-end", "2026-07-31"],
                          ["--validation-end", "2026-06"], ["--validation-start", "2026-06-30", "--validation-end", "2026-01-02"],
                          ["--half-lives", "0"], ["--half-lives", "126", "126"], ["--half-lives", "nan"]):
                with self.subTest(extra=extra):
                    run = self.run_cli("--csv", "/nonexistent.csv", "--output", output, *extra)
                    self.assertEqual(run.returncode, 2, run.stdout+run.stderr)
                    self.assertNotIn("FileNotFoundError", run.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
