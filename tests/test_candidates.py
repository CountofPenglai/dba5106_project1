import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from portfolio.candidates import estimate_risk, candidate_portfolio, check_candidate


class CandidateTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame([[.02, -.01], [-.01, .01], [.03, .005], [-.02, .01], [.025, -.005], [.005, .015]], columns=['A', 'B'])

    def test_candidates_match_independent_two_asset_feasible_grid(self):
        h = self.history()
        ctx = estimate_risk(h)
        first = np.linspace(-.25, 1.25, 150001)
        grid = np.column_stack([first, 1-first])
        var = np.einsum('ij,jk,ik->i', grid, ctx.covariance, grid)
        feasible = var <= ctx.variance_limit + 1e-14
        grid, var = grid[feasible], var[feasible]
        mu = np.array([.001, .0002])
        for objective in ['return', 'risk']:
            w, diag = candidate_portfolio(ctx, objective=objective, expected_returns=mu if objective=='return' else None)
            best = np.argmax(grid@mu) if objective=='return' else np.argmin(var)
            assert_allclose(w, grid[best], atol=8e-4)
            self.assertEqual(diag['solver_status'], 'optimal')
            check_candidate(ctx, w)

    def test_equal_forecasts_tie_break_to_ew_and_mixtures_feasible(self):
        ctx = estimate_risk(self.history())
        w, _ = candidate_portfolio(ctx, objective='return', expected_returns=[.001, .001])
        assert_allclose(w, [.5, .5], atol=1e-5)
        u, _ = candidate_portfolio(ctx, objective='return', expected_returns=[.001, -.001])
        v, _ = candidate_portfolio(ctx, objective='risk')
        check_candidate(ctx, .6*u+.25*v+.15*ctx.equal_weights)

    def test_invalid_history_and_infeasible_equal_weight_raise(self):
        for h in [pd.DataFrame([[np.nan, .1], [.2, .1]]), pd.DataFrame([[-2., -2.], [-1.5, -1.5]])]:
            with self.assertRaises(ValueError): estimate_risk(h)
        ctx = estimate_risk(self.history())
        with self.assertRaises(ValueError): check_candidate(ctx, [2, -1])
        with self.assertRaises(ValueError): candidate_portfolio(ctx, objective='return', expected_returns=[1])

    def test_risk_cap_does_not_disappear_for_small_returns(self):
        ctx = estimate_risk(self.history()*1e-4)
        with self.assertRaisesRegex(ValueError, 'variance'):
            check_candidate(ctx, [1.25, -.25])

    def test_isotropic_risk_budget_has_unique_equal_weight_solution(self):
        h = pd.DataFrame(np.random.default_rng(12).normal(0, .01, (100, 3)))
        ctx = estimate_risk(h)
        self.assertEqual(ctx.shrinkage, 1.)
        for objective in ['return', 'risk']:
            w, diag = candidate_portfolio(ctx, objective=objective, expected_returns=[.01, .001, -.01])
            assert_allclose(w, [1/3]*3, atol=1e-10)
            self.assertEqual(diag['solver_status'], 'optimal')

    def test_positive_risk_multiplier_certifies_unique_return_optimum(self):
        ctx = estimate_risk(self.history())
        w, diag = candidate_portfolio(ctx, objective='return', expected_returns=[.001, .0002])
        self.assertGreater(diag['risk_dual'], 1e-6)
        self.assertTrue(diag['unique_optimum'])
        self.assertEqual(diag['secondary_gap_scaled'], 0.)
        self.assertAlmostEqual(float(w @ ctx.covariance @ w) / ctx.variance_limit, 1., places=6)
