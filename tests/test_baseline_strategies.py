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



def baseline_strategies(config):
    from portfolio.model_catalog import baseline_models, fit_model
    return {spec.name: (lambda history, spec=spec: fit_model(history, spec).weights)
            for spec in baseline_models()}
