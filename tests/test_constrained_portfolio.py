import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.engine import portfolio_fit as training
from portfolio.models.constrained_portfolio import fit_constrained_portfolio
from portfolio.engine.rolling_backtest import backtest
from portfolio.regularization import Regularization


class GrowthTests(unittest.TestCase):
    def api(self):
        from portfolio.constraints import PortfolioConstraints
        from portfolio.losses import PortfolioLoss
        return fit_constrained_portfolio, PortfolioLoss, PortfolioConstraints

    def test_loss_uses_actual_returns_and_normalized_sample_weights(self):
        _, Loss, _ = self.api()
        # Mean=.05, variance=.0075; centering before log would change the reward.
        expected_growth = 0.75 * np.log(1.1) + 0.25 * np.log(0.9)
        for sample_weights in ([3, 1], [30, 10]):
            terms = Loss(reward_strength=0.2).components([0.1, -0.1], sample_weights)
            self.assertAlmostEqual(terms["mean"], 0.05)
            self.assertAlmostEqual(terms["variance"], 0.0075)
            self.assertAlmostEqual(terms["log_growth"], expected_growth)
            self.assertAlmostEqual(terms["loss"], 0.0075 - 0.2 * expected_growth)

    def test_invalid_loss_and_constraints_are_rejected(self):
        fit, Loss, Constraints = self.api()
        for eta in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                Loss(reward_strength=eta)
        for values in ([0.1, -1], [0.1, np.nan], [0.1, np.inf]):
            with self.assertRaises(ValueError):
                Loss().components(values)
        for args in ({"leverage_limit": 0.9}, {"volatility_limit": -1},
                     {"wealth_floor": 0}, {"wealth_floor": np.nan}):
            with self.assertRaises(ValueError):
                Constraints(**args)
        history = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]])
        with self.assertRaises(ValueError):
            fit(history, sample_weight=[1, 0])

    def test_zero_reward_with_inactive_constraints_matches_existing_fit(self):
        fit, Loss, Constraints = self.api()
        rng = np.random.default_rng(18)
        history = pd.DataFrame(rng.normal(0.001, 0.03, (40, 3)), columns=list("ABC"))
        sample_weights = np.linspace(0.25, 1, len(history))
        for kind in ("lasso", "ridge"):
            regularization = Regularization(kind, 1e-5)
            baseline = training.fit_portfolio(history, regularization=regularization,
                                              sample_weights=lambda n: sample_weights)
            result = fit(history, regularization=regularization,
                         loss=Loss(reward_strength=0),
                         constraints=Constraints(leverage_limit=10, volatility_limit=10),
                         sample_weight=sample_weights * 17)
            assert_allclose(result.weights, baseline.weights, atol=2e-5)
            self.assertAlmostEqual(result.objective, baseline.objective, places=9)

    def test_solution_matches_independent_two_asset_grid_with_risk_cap(self):
        fit, Loss, Constraints = self.api()
        returns = np.array([[.02, -.01], [-.01, .01], [.03, .005],
                            [-.02, .01], [.025, -.005], [.005, .015]])
        history = pd.DataFrame(returns, columns=["A", "B"])
        eta, lam, vol_limit = .03, 1e-5, .8
        result = fit(history, regularization=Regularization("lasso", lam),
                     loss=Loss(reward_strength=eta),
                     constraints=Constraints(volatility_limit=vol_limit))
        first = np.linspace(-.25, 1.25, 100001)
        grid_returns = returns[:, :1] * first + returns[:, 1:] * (1 - first)
        variances = np.var(grid_returns, axis=0)
        risk_cap = np.var(returns.mean(axis=1)) * vol_limit**2
        losses = variances - eta * np.log1p(grid_returns).mean(axis=0) + lam * np.abs(.5 - first)
        losses[variances > risk_cap] = np.inf
        best = np.argmin(losses)
        assert_allclose(result.weights.iloc[0], first[best], atol=3e-5)
        self.assertLessEqual(np.var(returns @ result.weights), risk_cap + 1e-10)
        self.assertAlmostEqual(result.objective, losses[best], delta=1e-8)

    def test_reward_changes_holdings_and_leverage_limit_binds(self):
        fit, Loss, Constraints = self.api()
        # A dominates B by a constant .01, so risk is identical for any weights.
        history = pd.DataFrame([[.011, .001], [.009, -.001],
                                [.012, .002], [.008, -.002]], columns=["A", "B"])
        baseline = fit(history, loss=Loss(reward_strength=0))
        rewarded = fit(history, loss=Loss(reward_strength=.01))
        assert_allclose(baseline.weights, [.5, .5], atol=1e-5)
        assert_allclose(rewarded.weights, [1.25, -.25], atol=2e-5)
        self.assertLessEqual(np.abs(rewarded.weights).sum(), 1.5 + 1e-7)
        self.assertGreater(rewarded.diagnostics["log_growth"], baseline.diagnostics["log_growth"])
        self.assertAlmostEqual(rewarded.model.intercept,
                               np.mean(history.to_numpy() @ rewarded.weights))

    def test_wealth_floor_is_enforced_without_clipping(self):
        fit, Loss, Constraints = self.api()
        history = pd.DataFrame([[-2, .1], [10, .1]], columns=["A", "B"])
        result = fit(history, loss=Loss(reward_strength=100),
                     constraints=Constraints(leverage_limit=2, volatility_limit=2, wealth_floor=.8))
        # First scenario: 1.1 - 2.1*w_A >= .8, hence w_A <= 1/7.
        assert_allclose(result.weights.iloc[0], 1 / 7, atol=1e-5)
        self.assertGreaterEqual(np.min(1 + history.to_numpy() @ result.weights), .8 - 1e-7)

    def test_infeasible_history_fails_instead_of_substituting_ew(self):
        fit, _, _ = self.api()
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            fit(pd.DataFrame([[-1.1, -1.1], [-1.1, -1.1]], columns=["A", "B"]))

    def test_rolling_growth_fit_does_not_read_future_returns(self):
        fit, _, _ = self.api()
        dates = pd.bdate_range("2026-01-01", periods=10)
        history = pd.DataFrame(np.random.default_rng(42).normal(0, .01, (10, 3)),
                               index=dates, columns=list("ABC"))
        strategy = lambda frame: fit(frame).weights
        before = backtest(history, strategy, start=dates[5], end=dates[7], window=5)
        changed = history.copy()
        changed.loc[dates[7]:] += .5
        after = backtest(changed, strategy, start=dates[5], end=dates[7], window=5)
        assert_allclose(before.weights, after.weights, atol=1e-8)
        assert_allclose(before.returns.iloc[:2], after.returns.iloc[:2], atol=1e-10)

    def test_default_settings_solve_a_feasible_252_day_window(self):
        fit, _, _ = self.api()
        rng = np.random.default_rng(94598)
        for _ in range(15):
            returns = rng.normal(0, .015, (252, 5)) + rng.normal(0, .003, 5)
        result = fit(pd.DataFrame(returns, columns=list("ABCDE")))
        self.assertEqual(result.diagnostics["solver_status"], "optimal")
        self.assertLessEqual(result.diagnostics["gross_leverage"], 1.5 + 1e-7)

    def test_primary_solver_error_recovers_with_a_real_secondary_solve(self):
        # A numerical solver error must not turn a feasible problem into EW.
        import cvxpy as cp
        fit, Loss, _ = self.api()
        solve = cp.Problem.solve

        def fail_primary(problem, *args, **kwargs):
            if kwargs.get("solver") == "CLARABEL":
                raise cp.error.SolverError("Simulated numerical breakdown")
            return solve(problem, *args, **kwargs)

        history = pd.DataFrame([[.011, .001], [.009, -.001],
                                [.012, .002], [.008, -.002]], columns=["A", "B"])
        with patch.object(cp.Problem, "solve", fail_primary):
            result = fit(history, loss=Loss(reward_strength=.01))
        assert_allclose(result.weights, [1.25, -.25], atol=2e-5)
        self.assertEqual(result.diagnostics["solver"], "SCS")
        self.assertEqual(result.diagnostics["solver_status"], "optimal")


if __name__ == "__main__":
    unittest.main()
