import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.data import load_returns, preprocess_returns
from portfolio.data.history_windows import rolling_windows
from portfolio.engine.portfolio_fit import fit_portfolio
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.portfolio_metrics import summarize


class DataTests(unittest.TestCase):
    def test_section_selection_units_and_missing(self):
        # Moving the table must not select value-weighted returns or firm counts.
        content = """File metadata can have variable length.

Average Value Weighted Returns -- Daily
,A,ME9 BM10,B,BIG HiBM
20260101,90,90,90,90

Average Equal Weighted Returns -- Daily
,A,ME9 BM10,B,BIG HiBM
20260101,1,-999,-2,-99.99
20260102,-99.99,-999,3,-999

Number of Firms in Portfolios
,A,ME9 BM10,B,BIG HiBM
20260101,100,200,300,400
"""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "returns.csv"
            path.write_text(content)
            result = load_returns(path)
        self.assertEqual(list(result.columns), ["A", "B"])
        assert_allclose(result.iloc[0], [0.01, -0.02])
        self.assertTrue(np.isnan(result.iloc[1, 0]))
        self.assertEqual(result.index[-1], pd.Timestamp("2026-01-02"))

    def test_preprocessing_drops_named_assets_before_numeric_conversion(self):
        # Non-numeric excluded values must never enter the numeric cleaner.
        raw = pd.DataFrame({" BIG HiBM ": ["missing", "missing"],
                            "A": [1.0, 2.0], "ME9 BM10": ["bad", "bad"],
                            "B": [-2.0, 3.0]}, index=[20260701, 20260702])
        original = raw.copy(deep=True)
        result = preprocess_returns(raw)
        self.assertEqual(list(result.columns), ["A", "B"])
        assert_allclose(result.to_numpy(), [[0.01, -0.02], [0.02, 0.03]])
        pd.testing.assert_frame_equal(raw, original)

    def test_preprocessing_accepts_already_removed_assets_and_custom_exclusions(self):
        raw = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=[20260701])
        self.assertEqual(list(preprocess_returns(raw).columns), ["A", "B"])
        self.assertEqual(list(preprocess_returns(raw, drop_assets=("B",)).columns), ["A"])


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
    def test_ols_portfolio_matches_covariance_solution(self):
        r = pd.DataFrame(np.random.default_rng(7).normal(size=(80, 4)) + [1, 3, 2, 4],
                         columns=list("ABCD"))
        result = fit_portfolio(r)
        inv_one = np.linalg.solve(np.cov(r, rowvar=False), np.ones(4))
        assert_allclose(result.weights, inv_one / inv_one.sum(), atol=1e-10)


    def test_nonfinite_returns_are_rejected(self):
        for value in [np.nan, np.inf]:
            with self.assertRaises(ValueError):
                fit_portfolio(pd.DataFrame([[1, value], [2, 3]], columns=["A", "B"]))
class EvaluationTests(unittest.TestCase):
    def test_report_concentration_uses_absolute_weight_shares(self):
        from portfolio.evaluation import portfolio_metrics as metrics
        weights = pd.DataFrame([[1.2, -.2], [.5, .5]])
        result = metrics.summarize_holdings(weights)
        self.assertAlmostEqual(result['average_effective_assets'], (1.4**2/1.48 + 2)/2)
        self.assertAlmostEqual(result['average_max_abs_weight'], .85)
        self.assertAlmostEqual(result['average_gross_leverage'], 1.2)


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
