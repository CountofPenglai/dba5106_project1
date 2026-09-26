"""Baseline mathematics, training-window causality, and public CLI coverage."""
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

from portfolio.config import TRAINING_PRESETS, TrainingConfig
from portfolio.data.history_windows import TrainingWindow
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.artifact_store import artifact_path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["EW", "MinVar", "LASSO_CoefficientConstraint", "Ridge_CoefficientConstraint",
         "LASSO_Penalty", "Ridge_Penalty"]


def sample_returns():
    rng = np.random.default_rng(20260926)
    values = rng.normal(.0003, .01, (280, 4))
    values[:, 1] += .6 * values[:, 0]
    return pd.DataFrame(values, index=pd.bdate_range("2024-01-02", periods=280),
                        columns=list("ABCD"))


class BaselineStrategyTests(unittest.TestCase):
    def test_fixed_window_fits_satisfy_independent_optimality_checks(self):
        from portfolio.engine.baseline_strategies import baseline_strategies
        frame = sample_returns()
        for preset in ("126d", "252d"):
            config = TRAINING_PRESETS[preset]
            history = frame.iloc[-config.window.observations:]
            strategies = baseline_strategies(config)
            self.assertEqual(list(strategies), NAMES)
            fitted = {name: strategy(history).to_numpy() for name, strategy in strategies.items()}
            assert_allclose(fitted["EW"], np.full(4, .25))
            covariance = np.cov(history, rowvar=False)
            inv_one = np.linalg.solve(covariance, np.ones(4))
            assert_allclose(fitted["MinVar"], inv_one / inv_one.sum(), atol=1e-10)
            # Independent first-order conditions for variance + penalty on beta.
            X = history.to_numpy()[:, :-1] - history.to_numpy()[:, [-1]]
            X -= X.mean(axis=0)
            y = history.mean(axis=1).to_numpy()
            y = y - y.mean()
            for name, strength in (("LASSO", 1e-5), ("Ridge", 1e-4)):
                with self.subTest(preset=preset, model=name):
                    weights = fitted[name + "_Penalty"]
                    self.assertAlmostEqual(weights.sum(), 1)
                    beta = .25 - weights[:-1]
                    gradient = 2 * X.T @ (X @ beta - y) / len(y)
                    if name == "Ridge":
                        assert_allclose(gradient + 2 * strength * beta, 0, atol=1e-12)
                    else:
                        active = np.abs(beta) > 1e-8
                        assert_allclose(gradient[active] + strength * np.sign(beta[active]),
                                        0, atol=1e-10)
                        self.assertTrue(np.all(np.abs(gradient[~active]) <= strength + 1e-10))

    def test_window_selection_and_future_isolation_for_every_baseline(self):
        from portfolio.engine.baseline_strategies import baseline_strategies
        frame = sample_returns()
        start, changed_date, end = frame.index[-3:]
        altered = frame.copy()
        altered.loc[changed_date:] += np.array([.03, -.02, .04, -.01])
        for count in (63, 126, 252):
            config = TrainingConfig(TrainingWindow(observations=count))
            for name, strategy in baseline_strategies(config).items():
                seen = []
                def recorded(history):
                    seen.append((len(history), history.index[0], history.index[-1]))
                    return strategy(history)
                before = backtest(frame, recorded, start=start, end=end, window=config.window)
                after = backtest(altered, strategy, start=start, end=end, window=config.window)
                with self.subTest(window=count, model=name):
                    for offset, (length, first, last) in enumerate(seen):
                        self.assertEqual(length, count)
                        self.assertEqual(first, frame.index[len(frame) - 3 + offset - count])
                        self.assertEqual(last, frame.index[len(frame) - 4 + offset])
                    assert_allclose(before.weights.loc[:changed_date], after.weights.loc[:changed_date])
                    self.assertEqual(before.returns.loc[start], after.returns.loc[start])

    def test_training_config_observation_weights_reach_minvar_fit(self):
        from portfolio.engine.baseline_strategies import baseline_strategies
        history = sample_returns()
        config = TrainingConfig(TrainingWindow(years=3, min_observations=252), half_life=63)
        weights = baseline_strategies(config)["MinVar"](history)
        q = 2 ** (-np.arange(len(history) - 1, -1, -1) / 63)
        centered = history.to_numpy() - np.average(history, axis=0, weights=q)
        cov = centered.T @ (q[:, None] * centered) / q.sum()
        inv_one = np.linalg.solve(cov, np.ones(4))
        assert_allclose(weights, inv_one / inv_one.sum(), atol=1e-10)


class BaselineCLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "experiments.regularized_baselines", *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=45,
                              env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

    def test_six_model_cli_runs_both_windows_and_records_settings(self):
        frame = sample_returns()
        raw = frame * 100
        raw.index = frame.index.strftime("%Y%m%d")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "returns.csv"
            source.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")
            for count in (126, 252):
                output = root / str(count)
                run = self.run_cli("--baselines-only", "--csv", str(source), "--window", str(count),
                                   "--start", str(frame.index[-3].date()), "--end", str(frame.index[-1].date()),
                                   "--output", str(output))
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                summary = pd.read_csv(artifact_path(output, "summary.csv"), index_col=0)
                self.assertEqual(summary.index.tolist(), NAMES)
                metadata = json.loads(artifact_path(output, "run_metadata.json").read_text())
                self.assertEqual(metadata["models"], NAMES)
                self.assertEqual(metadata["window"], count)
                self.assertEqual(metadata["training"]["window"]["observations"], count)
                self.assertIsNone(metadata["training"]["half_life"])
                self.assertEqual(metadata["evaluation_days"], 3)
                self.assertEqual(metadata["coefficient_constraints"],
                                 {"lasso": {"kind": "lasso", "budget": 1.0},
                                  "ridge": {"kind": "ridge", "budget": 1.0}})
                for name in NAMES:
                    holdings = pd.read_csv(artifact_path(output, name + "_weights.csv"), index_col=0)
                    daily = pd.read_csv(artifact_path(output, name + "_returns.csv"), index_col=0).iloc[:, 0]
                    assert_allclose(holdings.sum(axis=1), 1, atol=1e-12)
                    if name.endswith("CoefficientConstraint"):
                        beta = .25 - holdings.to_numpy()[:, :-1]
                        measure = np.abs(beta).sum(axis=1) if name.startswith("LASSO") else np.square(beta).sum(axis=1)
                        self.assertTrue(np.all(measure <= 1 + 1e-8))
                        diag = pd.read_csv(artifact_path(output, name + "_diagnostics.csv"))
                        assert_allclose(diag["coefficient_measure"], measure, atol=1e-10)
                        self.assertTrue((pd.to_datetime(diag["history_end"]) < pd.to_datetime(diag["date"])).all())
                    assert_allclose(daily, (holdings.to_numpy() * frame.iloc[-3:].to_numpy()).sum(axis=1),
                                    atol=1e-14)

    def test_root_entry_runs_six_models_and_configurable_zero_budgets(self):
        frame = sample_returns()
        raw = frame * 100
        raw.index = frame.index.strftime("%Y%m%d")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "input.csv"; output = root / "run"
            source.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")
            run = subprocess.run([sys.executable, "run_baselines.py", "--csv", str(source),
                                  "--window", "126", "--start", str(frame.index[-3].date()),
                                  "--end", str(frame.index[-1].date()), "--output", str(output),
                                  "--lasso-coefficient-budget", "0", "--ridge-coefficient-budget", "0"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=45,
                                 env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = pd.read_csv(artifact_path(output, "summary.csv"), index_col=0)
            self.assertEqual(summary.index.tolist(), NAMES)
            for name in ("LASSO_CoefficientConstraint", "Ridge_CoefficientConstraint"):
                weights = pd.read_csv(artifact_path(output, name + "_weights.csv"), index_col=0)
                assert_allclose(weights, .25, atol=0)
                assert_allclose(summary.loc[name], summary.loc["EW"], atol=1e-12)

    def test_invalid_window_or_mixed_experiment_flags_fail_before_loading(self):
        for options in (("--window", "1"), ("--baselines-only", "--weighted"),
                        ("--baselines-only", "--growth"),
                        ("--baselines-only", "--lasso-coefficient-budget", "-1")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "output"
                run = self.run_cli(*options, "--csv", str(Path(temp) / "absent.csv"),
                                   "--output", str(output))
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertNotIn("Traceback", run.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
