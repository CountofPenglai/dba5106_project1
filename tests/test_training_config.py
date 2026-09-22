import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.evaluation import backtest
from portfolio.training import fit_portfolio


class TrainingConfigTests(unittest.TestCase):
    def test_weighted_preset_reaches_the_actual_fit(self):
        from portfolio.config import TRAINING_PRESETS
        config = TRAINING_PRESETS["3y_weighted"]
        dates = pd.bdate_range("2022-01-03", periods=800)
        frame = pd.DataFrame(np.random.default_rng(21).normal(0, .01, (800, 3)),
                             index=dates, columns=list("ABC"))
        def strategy(history):
            return fit_portfolio(history, sample_weights=config.sample_weights).weights
        result = backtest(frame, strategy, start=dates[-1], end=dates[-1], window=config.window)
        history = frame.loc[(frame.index >= dates[-1] - pd.DateOffset(years=3)) & (frame.index < dates[-1])]
        # Independent weighted covariance solution, with no regularization.
        q = 2.0 ** (-np.arange(len(history) - 1, -1, -1) / 252)
        centered = history.to_numpy() - np.average(history, axis=0, weights=q)
        cov = centered.T @ (q[:, None] * centered) / q.sum()
        expected = np.linalg.solve(cov, np.ones(3))
        assert_allclose(result.weights.iloc[0], expected / expected.sum(), atol=1e-10)
        weights = config.sample_weights(757)
        assert_allclose(weights[[0, 252, 504, 756]], [.125, .25, .5, 1])

    def test_uniform_presets_preserve_unweighted_fit(self):
        from portfolio.config import TRAINING_PRESETS
        for name in ("126d", "252d", "3y"):
            assert_allclose(TRAINING_PRESETS[name].sample_weights(4), np.ones(4))

    def test_invalid_config_fails_before_training(self):
        from portfolio.config import TrainingConfig
        from portfolio.splits import TrainingWindow
        for half_life in (0, -1, np.nan, np.inf, True, "252"):
            with self.subTest(half_life=half_life), self.assertRaises(ValueError):
                TrainingConfig(TrainingWindow(years=3), half_life=half_life)
        with self.assertRaises(ValueError):
            TrainingConfig("three years")


if __name__ == "__main__":
    unittest.main()
