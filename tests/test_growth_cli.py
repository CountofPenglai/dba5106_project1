import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class GrowthCliTests(unittest.TestCase):
    def test_growth_experiment_exports_matched_controls_and_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dates = pd.bdate_range("2026-01-01", periods=16)
            frame = pd.DataFrame(np.random.default_rng(35).normal(.05, 1, (16, 3)),
                                 index=dates.strftime("%Y%m%d"), columns=list("ABC"))
            csv = directory / "input.csv"
            csv.write_text("Average Equal Weighted Returns -- Daily\n" + frame.to_csv() + "\n")
            output = directory / "output"
            run = subprocess.run(
                [sys.executable, str(root / "run_example.py"), "--csv", str(csv),
                 "--window", "10", "--start", str(dates[10].date()),
                 "--end", str(dates[-1].date()), "--weighted", "--growth", "--output", str(output)],
                capture_output=True, text=True, cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"},
                timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = pd.read_csv(output / "summary.csv", index_col=0)
            self.assertEqual(len(summary), 11)
            self.assertIn("LASSO_Constrained", summary.index)
            self.assertIn("LASSO_LogGrowth", summary.index)
            for name in ("LASSO_Constrained", "LASSO_LogGrowth"):
                returns = pd.read_csv(output / f"{name}_returns.csv", index_col=0)
                diagnostics = pd.read_csv(output / f"{name}_diagnostics.csv", index_col=0)
                weights = pd.read_csv(output / f"{name}_weights.csv", index_col=0)
                self.assertEqual(list(returns.index), list(diagnostics.index))
                self.assertEqual(len(diagnostics), 6)
                self.assertTrue((pd.to_datetime(diagnostics["history_end"]) < pd.to_datetime(diagnostics.index)).all())
                self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.5 + 1e-7)
                self.assertTrue((diagnostics["variance"] <= diagnostics["variance_limit"] + 1e-9).all())
                self.assertTrue((diagnostics["solver_status"] == "optimal").all())
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["growth"]["lasso_strength"], 1e-5)
            self.assertEqual(metadata["growth"]["reward_strength"], .01)
            self.assertFalse(metadata["growth"]["hyperparameter_search"])
            self.assertEqual(metadata["evaluation_days"], 6)
            self.assertEqual(len(metadata["data_sha256"]), 64)
            snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
            again = subprocess.run(run.args + ["--window", "11", "--start", str(dates[11].date())],
                                   capture_output=True, text=True, cwd=root,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"},
                                   timeout=60)
            self.assertEqual(again.returncode, 2, again.stdout + again.stderr)
            self.assertIn("empty or new directory", again.stderr)
            self.assertEqual(snapshot, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
