import json
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.decisions.linear_program_allocation import (
    LPConstraints,
    LinearPortfolioError,
    build_lp_context,
    check_lp_weights,
    solve_linear_portfolio,
)


class LinearDecisionTests(unittest.TestCase):
    @staticmethod
    def manual_metrics(history, weights):
        returns = np.asarray(history, dtype=float) @ np.asarray(weights, dtype=float)
        centered = returns - returns.mean()
        cumulative = np.cumsum(returns)
        peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
        return {
            "budget": float(np.sum(weights)),
            "gross": float(np.sum(np.abs(weights))),
            "max_position": float(np.max(np.abs(weights))),
            "mad": float(np.mean(np.abs(centered))),
            "drawdown": float(np.max(peaks - cumulative)),
            "min_wealth": float(np.min(1.0 + returns)),
        }

    @staticmethod
    def enumerate_two_asset(history, expected_returns, constraints, points=120_001):
        first = np.linspace(-1.0, 2.0, points)
        weights = np.column_stack([first, 1.0 - first])
        realized = weights @ np.asarray(history, dtype=float).T
        centered = realized - realized.mean(axis=1, keepdims=True)
        cumulative = np.cumsum(realized, axis=1)
        peaks = np.maximum.accumulate(
            np.column_stack([np.zeros(len(first)), cumulative]), axis=1
        )[:, 1:]
        equal = np.asarray(history, dtype=float).mean(axis=1)
        equal_centered = equal - equal.mean()
        equal_cumulative = np.cumsum(equal)
        equal_peaks = np.maximum.accumulate(np.r_[0.0, equal_cumulative])[1:]
        mad_limit = constraints.mad_ratio * np.mean(np.abs(equal_centered))
        drawdown_limit = constraints.drawdown_ratio * np.max(
            equal_peaks - equal_cumulative
        )
        feasible = (
            (np.abs(weights).sum(axis=1) <= constraints.leverage_limit + 1e-14)
            & (np.abs(weights).max(axis=1) <= constraints.position_limit + 1e-14)
            & (np.abs(centered).mean(axis=1) <= mad_limit + 1e-14)
            & ((peaks - cumulative).max(axis=1) <= drawdown_limit + 1e-14)
            & ((1.0 + realized).min(axis=1) >= constraints.wealth_floor - 1e-14)
        )
        objective = weights @ np.asarray(expected_returns, dtype=float)
        return float(objective[feasible].max())

    def test_matches_independent_two_asset_enumeration(self):
        history = np.array([[-0.10, 0.0], [0.10, 0.0]])
        constraints = LPConstraints(position_limit=2.0)
        expected = np.array([0.03, 0.01])

        context = build_lp_context(history, constraints)
        weights, diagnostics = solve_linear_portfolio(context, expected)
        grid_best = self.enumerate_two_asset(history, expected, constraints)

        self.assertGreaterEqual(float(expected @ weights), grid_best - 1e-10)
        assert_allclose(weights, [0.5, 0.5], atol=2e-8)
        self.assertAlmostEqual(diagnostics["mad"], context.mad_limit, places=10)
        self.assertAlmostEqual(
            diagnostics["drawdown"], context.drawdown_limit, places=10
        )

    def test_short_position_is_allowed_and_leverage_limit_is_active(self):
        history = np.full((4, 2), 0.01)
        constraints = LPConstraints(position_limit=2.0)

        context = build_lp_context(history, constraints)
        weights, diagnostics = solve_linear_portfolio(context, [0.02, 0.0])

        assert_allclose(weights, [1.25, -0.25], atol=2e-8)
        self.assertLess(weights[1], 0)
        self.assertGreaterEqual(diagnostics["gross"], 1.5 - 2e-8)
        self.assertLessEqual(diagnostics["gross"], 1.5 + 1e-10)
        self.assertEqual(context.mad_limit, 0.0)
        self.assertEqual(context.drawdown_limit, 0.0)

    def test_position_limit_is_active(self):
        context = build_lp_context(
            np.full((3, 2), 0.01),
            LPConstraints(leverage_limit=3.0, position_limit=0.7),
        )

        weights, diagnostics = solve_linear_portfolio(context, [0.02, 0.0])

        assert_allclose(weights, [0.7, 0.3], atol=2e-8)
        self.assertGreaterEqual(diagnostics["max_position"], 0.7 - 1e-8)
        self.assertLessEqual(diagnostics["max_position"], 0.7 + 1e-10)

    def test_mad_limit_can_bind_without_drawdown_limit(self):
        history = np.array([[-0.10, 0.0], [0.10, 0.0]])
        context = build_lp_context(
            history,
            LPConstraints(position_limit=2.0, drawdown_ratio=10.0),
        )

        weights, diagnostics = solve_linear_portfolio(context, [0.02, 0.0])

        assert_allclose(weights, [0.5, 0.5], atol=2e-8)
        self.assertAlmostEqual(diagnostics["mad"], 0.05, places=10)
        self.assertAlmostEqual(diagnostics["mad"], context.mad_limit, places=10)
        self.assertLess(diagnostics["drawdown"], context.drawdown_limit)

    def test_drawdown_limit_can_bind_without_mad_limit(self):
        history = np.array([[-0.10, 0.0], [0.10, 0.0]])
        context = build_lp_context(
            history,
            LPConstraints(position_limit=2.0, mad_ratio=10.0),
        )

        weights, diagnostics = solve_linear_portfolio(context, [0.02, 0.0])

        assert_allclose(weights, [0.5, 0.5], atol=2e-8)
        self.assertAlmostEqual(diagnostics["drawdown"], 0.05, places=10)
        self.assertAlmostEqual(
            diagnostics["drawdown"], context.drawdown_limit, places=10
        )
        self.assertLess(diagnostics["mad"], context.mad_limit)

    def test_drawdown_uses_zero_as_the_initial_peak(self):
        history = np.array([[-0.10, 0.0], [-0.10, 0.0], [0.05, 0.0]])
        context = build_lp_context(
            history,
            LPConstraints(position_limit=2.0, mad_ratio=10.0, drawdown_ratio=10.0),
        )

        actual = check_lp_weights(context, [1.0, 0.0])

        self.assertAlmostEqual(actual["drawdown"], 0.20)

    def test_equal_forecasts_tie_break_exactly_to_equal_weights(self):
        history = np.array(
            [[-0.04, 0.03, 0.01], [0.02, -0.01, 0.01], [0.03, 0.02, -0.02]]
        )
        context = build_lp_context(
            history, LPConstraints(position_limit=0.8, mad_ratio=2, drawdown_ratio=2)
        )

        weights, diagnostics = solve_linear_portfolio(context, [0.01, 0.01, 0.01])

        assert_allclose(weights, np.full(3, 1 / 3), atol=2e-10)
        self.assertLessEqual(abs(diagnostics["secondary_gap"]), 1e-10 + 1e-14)
        self.assertEqual(diagnostics["solver_status"], "optimal")
        json.dumps(diagnostics, allow_nan=False)

    def test_zero_and_tiny_equal_weight_risk_caps_are_enforced(self):
        histories = [
            np.array([[0.02, 0.0], [0.0, 0.02]]),
            np.array([[0.02, 0.0], [0.0, 0.02]]) * 1e-9,
            np.array([[0.02, 0.0], [0.0, 0.020000000004]]),
        ]
        for history in histories:
            with self.subTest(history=history.tolist()):
                context = build_lp_context(
                    history,
                    LPConstraints(position_limit=2.0, drawdown_ratio=10.0),
                )
                weights, diagnostics = solve_linear_portfolio(context, [0.02, 0.0])
                manual = self.manual_metrics(history, weights)

                self.assertLessEqual(
                    manual["mad"], context.mad_limit + max(1e-14, context.mad_limit * 1e-5)
                )
                assert_allclose(weights, [0.5, 0.5], atol=2e-9)
                self.assertTrue(diagnostics["feasible"])

    def test_tiny_drawdown_cap_is_enforced(self):
        base = np.array([[-0.02, 0.02], [0.02, -0.020000000004]])
        for history in (base, base * 1e-9):
            with self.subTest(scale=float(np.abs(history).max())):
                context = build_lp_context(
                    history,
                    LPConstraints(position_limit=2.0, mad_ratio=1e12),
                )

                weights, diagnostics = solve_linear_portfolio(context, [0.02, 0.0])
                manual = self.manual_metrics(history, weights)

                self.assertLessEqual(
                    manual["drawdown"],
                    context.drawdown_limit
                    + max(1e-25, context.drawdown_limit * 1e-5),
                )
                assert_allclose(weights, [0.5, 0.5], atol=2e-9)
                self.assertTrue(diagnostics["feasible"])

    def test_infeasible_history_raises_instead_of_falling_back(self):
        context = build_lp_context(
            np.full((3, 2), -2.0), LPConstraints(position_limit=1.0)
        )

        with self.assertRaisesRegex(LinearPortfolioError, "primary.*infeasible"):
            solve_linear_portfolio(context, [0.01, 0.0])

    def test_invalid_history_and_configuration_are_rejected(self):
        bad_histories = [
            np.ones(3),
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.array([[0.0, np.nan], [0.0, 0.0]]),
            [[0.0, 0.0], [0.0, 0.0]],
            pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], columns=["A", "A"]),
        ]
        for history in bad_histories:
            with self.subTest(history=type(history).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    build_lp_context(history)

        bad_constraints = [
            {"leverage_limit": 0.99},
            {"leverage_limit": np.inf},
            {"position_limit": 0.0},
            {"position_limit": np.nan},
            {"mad_ratio": -1.0},
            {"drawdown_ratio": np.inf},
            {"wealth_floor": 0.0},
            {"wealth_floor": 1.01},
        ]
        for values in bad_constraints:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    LPConstraints(**values)

        with self.assertRaises(ValueError):
            build_lp_context(np.zeros((2, 3)), LPConstraints(position_limit=0.3))

    def test_invalid_expected_returns_and_weights_are_rejected(self):
        context = build_lp_context(
            np.zeros((3, 2)), LPConstraints(position_limit=1.0)
        )
        for expected in ([0.1], [0.1, np.nan], [[0.1, 0.2]]):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    solve_linear_portfolio(context, expected)
        for weights in ([1.0], [np.nan, 0.0], [[0.5, 0.5]]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    check_lp_weights(context, weights)


if __name__ == "__main__":
    unittest.main()
