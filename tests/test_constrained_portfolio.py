import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.engine import portfolio_fit as training
from portfolio.models.constrained_portfolio import fit_constrained_portfolio
from portfolio.regularization import Regularization


class ConstrainedPortfolioTests(unittest.TestCase):
    def api(self):
        from portfolio.constraints import PortfolioConstraints
        from portfolio.losses import PortfolioLoss
        return fit_constrained_portfolio, PortfolioLoss, PortfolioConstraints


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
