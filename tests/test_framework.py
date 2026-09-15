import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.data import load_returns
from portfolio.splits import split_returns, rolling_windows
from portfolio.model import fit_linear
from portfolio.losses import weighted_mse
from portfolio.regularization import Regularization
from portfolio.weights import exponential, shrink_to_equal
from portfolio.training import fit_portfolio
from portfolio.evaluation import backtest, summarize


class DataTests(unittest.TestCase):
    def test_section_selection_units_and_missing(self):
        # Moving the table must not select value-weighted returns or firm counts.
        content = """File metadata can have variable length.

Average Value Weighted Returns -- Daily
,A,ME9 BM10,B
20260101,90,90,90

Average Equal Weighted Returns -- Daily
,A,ME9 BM10,B
20260101,1,-999,-2
20260102,-99.99,-999,3

Number of Firms in Portfolios
,A,ME9 BM10,B
20260101,100,200,300
"""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "returns.csv"
            path.write_text(content)
            result = load_returns(path)
        self.assertEqual(list(result.columns), ["A", "B"])
        assert_allclose(result.iloc[0], [0.01, -0.02])
        self.assertTrue(np.isnan(result.iloc[1, 0]))
        self.assertEqual(result.index[-1], pd.Timestamp("2026-01-02"))

    def test_split_preserves_gap_and_empty_test(self):
        r = pd.DataFrame({"A": np.arange(10)}, index=pd.date_range("2026-01-01", periods=10))
        parts = split_returns(r, train_end="2026-01-03", validation_end="2026-01-06",
                              test_start="2026-01-08", test_end="2026-01-10")
        self.assertEqual(parts.train.index[-1], pd.Timestamp("2026-01-03"))
        self.assertEqual(len(parts.validation), 3)
        self.assertEqual(parts.test.index[0], pd.Timestamp("2026-01-08"))
        absent = split_returns(r, train_end="2026-01-03", validation_end="2026-01-06",
                               test_start="2026-08-01", test_end="2026-08-30")
        self.assertTrue(absent.test.empty)

    def test_rolling_history_excludes_current_day_and_requires_coverage(self):
        r = pd.DataFrame({"A": range(5)}, index=pd.date_range("2026-01-01", periods=5))
        windows = list(rolling_windows(r, start="2026-01-04", end="2026-01-05", window=3))
        self.assertEqual(list(windows[0][1]["A"]), [0, 1, 2])
        self.assertEqual(list(windows[1][1]["A"]), [1, 2, 3])
        with self.assertRaises(ValueError):
            list(rolling_windows(r, start="2026-01-03", end="2026-01-05", window=3))
        with self.assertRaises(ValueError):
            list(rolling_windows(r, start="2026-08-01", end="2026-08-30", window=3))


class ModelTests(unittest.TestCase):
    def test_regularized_weighted_solutions_match_analytic_minima(self):
        X = np.array([[-1.0], [0.0], [1.0]])
        y = np.array([-2.0, 0.0, 2.0])
        # Weighted variance(X)=1/2. Objectives yield ridge=1, lasso=1.5.
        for kind, expected in [("ridge", 1.0), ("lasso", 1.5)]:
            for scale in [1, 17]:
                result = fit_linear(X, y + 7, regularization=Regularization(kind, 0.5),
                                    sample_weight=np.array([1, 2, 1]) * scale)
                assert_allclose(result.coefficients, [expected], atol=1e-7)
                self.assertAlmostEqual(result.intercept, 7)

    def test_ols_portfolio_matches_covariance_solution(self):
        r = pd.DataFrame(np.random.default_rng(7).normal(size=(80, 4)) + [1, 3, 2, 4],
                         columns=list("ABCD"))
        result = fit_portfolio(r)
        inv_one = np.linalg.solve(np.cov(r, rowvar=False), np.ones(4))
        assert_allclose(result.weights, inv_one / inv_one.sum(), atol=1e-10)

    def test_sample_weighting_changes_the_fit_and_callbacks_compose(self):
        r = pd.DataFrame([[0.1, 0], [-0.1, 0.05], [0.3, -0.1], [0.02, 0.1]], columns=["A", "B"])
        s = np.array([0.1, 0.2, 0.5, 1.0])
        fitted = fit_portfolio(r, sample_weights=lambda n: s,
                               adjust_weights=partial(shrink_to_equal, strength=0.25))
        # Independent weighted covariance solution in asset coordinates.
        centered = r.to_numpy() - np.average(r, axis=0, weights=s)
        cov = centered.T @ (s[:, None] * centered) / s.sum()
        inv_one = np.linalg.solve(cov, np.ones(2))
        expected = inv_one / inv_one.sum()
        assert_allclose(fitted.raw_weights, expected)
        assert_allclose(fitted.weights, 0.75 * expected + 0.125)
        self.assertFalse(np.allclose(fitted.raw_weights, fit_portfolio(r).weights))

    def test_loss_weighting_and_decay(self):
        self.assertAlmostEqual(weighted_mse([1, 3], [0, 0], [3, 1]), 3)
        assert_allclose(exponential(3, half_life=1), [0.25, 0.5, 1])
        assert_allclose(shrink_to_equal(np.array([-0.2, 1.2]), 0.5), [0.15, 0.85])

    def test_bad_data_and_weight_adjustment_are_rejected(self):
        for value in [np.nan, np.inf]:
            with self.assertRaises(ValueError):
                fit_portfolio(pd.DataFrame([[1, value], [2, 3]], columns=["A", "B"]))
        r = pd.DataFrame([[1, 2], [3, 1], [2, 4]], columns=["A", "B"])
        with self.assertRaises(ValueError):
            fit_portfolio(r, sample_weights=lambda n: np.zeros(n))
        with self.assertRaises(ValueError):
            fit_portfolio(r, adjust_weights=lambda w: w * 2)
        with self.assertRaises(ValueError):
            fit_linear(np.ones((3, 1)), np.ones(3), sample_weight=[1, -1, 1])


class EvaluationTests(unittest.TestCase):
    def test_future_returns_cannot_affect_earlier_weights(self):
        dates = pd.bdate_range("2026-01-01", periods=12)
        r = pd.DataFrame(np.random.default_rng(5).normal(0, 0.01, (12, 3)), index=dates, columns=list("ABC"))
        strategy = lambda history: fit_portfolio(history, regularization=Regularization("ridge", 0.001)).weights
        before = backtest(r, strategy, start=dates[5], end=dates[-1], window=5)
        changed = r.copy()
        changed.loc[dates[8]:] += 10
        after = backtest(changed, strategy, start=dates[5], end=dates[-1], window=5)
        assert_allclose(before.weights.loc[:dates[8]], after.weights.loc[:dates[8]])
        assert_allclose(before.returns.loc[:dates[7]], after.returns.loc[:dates[7]])

    def test_metrics_use_decimal_returns_and_initial_wealth(self):
        m = summarize(pd.Series([0.1, -0.1]))
        self.assertAlmostEqual(m["cumulative_return"], -0.01)
        self.assertAlmostEqual(m["max_drawdown"], -0.1)
        self.assertAlmostEqual(m["volatility"], np.sqrt(0.02))
        self.assertAlmostEqual(summarize(pd.Series([-0.1, 0]))["max_drawdown"], -0.1)
        self.assertTrue(np.isnan(summarize(pd.Series([0, 0]))["sharpe"]))

    def test_constant_nonzero_returns_have_undefined_sharpe(self):
        metrics = summarize(pd.Series([0.1, 0.1, 0.1]))
        self.assertEqual(metrics["volatility"], 0.0)
        self.assertTrue(np.isnan(metrics["sharpe"]))
        self.assertTrue(np.isnan(metrics["negative_sharpe"]))

    def test_small_nonzero_volatility_is_not_treated_as_constant(self):
        metrics = summarize(pd.Series([0.1, 0.1 + 1e-12, 0.1 - 1e-12]))
        self.assertGreater(metrics["volatility"], 0.0)
        self.assertTrue(np.isfinite(metrics["sharpe"]))

    def test_custom_strategy_assets_align_and_invalid_weights_fail(self):
        r = pd.DataFrame([[0, 0], [0.1, 0.2], [0.02, -0.01]],
                         columns=["A", "B"], index=pd.date_range("2026-01-01", periods=3))
        result = backtest(r, lambda h: pd.Series([0.75, 0.25], index=["B", "A"]),
                          start="2026-01-03", end="2026-01-03", window=2)
        self.assertAlmostEqual(result.returns.iloc[0], -0.0025)
        with self.assertRaises(ValueError):
            backtest(r, lambda h: pd.Series([1, 1], index=["A", "B"]),
                     start="2026-01-03", end="2026-01-03", window=2)


if __name__ == "__main__":
    unittest.main()
