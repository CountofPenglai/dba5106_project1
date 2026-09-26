"""Analytical tests distinguish explicit coefficient budgets from penalties."""
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


class CoefficientConstraintTests(unittest.TestCase):
    def fit(self, X, y, kind, budget, **options):
        from portfolio.constraints import CoefficientConstraint
        from portfolio.models.coefficient_constrained_regression import fit_coefficient_constrained
        return fit_coefficient_constrained(X, y, constraint=CoefficientConstraint(kind, budget), **options)

    def test_active_budgets_match_orthogonal_analytic_solutions(self):
        # Orthogonal unit-variance columns, OLS beta=(2, 1), free intercept=7.
        X = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
        y = X @ np.array([2., 1.]) + 7
        cases = [('lasso', 1., [1., 0.]), ('ridge', .25, np.array([2., 1.]) / np.sqrt(20))]
        for kind, budget, expected in cases:
            for scale in (1., .001):
                with self.subTest(kind=kind, scale=scale):
                    model, diag = self.fit(X * scale, y * scale, kind, budget)
                    assert_allclose(model.coefficients, expected, atol=3e-5)
                    self.assertAlmostEqual(model.intercept, 7 * scale, places=8)
                    measure = np.abs(model.coefficients).sum() if kind == 'lasso' else model.coefficients @ model.coefficients
                    self.assertLessEqual(measure, budget + 1e-8)
                    self.assertAlmostEqual(diag['coefficient_measure'], measure)
                    self.assertAlmostEqual(diag['objective'], diag['mse'])
                    self.assertEqual(diag['solver_status'], 'optimal')

    def test_zero_budget_is_zero_coefficients_with_free_weighted_intercept(self):
        X = np.array([[1., 2.], [3., 4.], [6., 5.]])
        y = np.array([2., 8., 12.])
        for kind in ('lasso', 'ridge'):
            model, diag = self.fit(X, y, kind, 0., sample_weight=[1, 2, 1])
            assert_allclose(model.coefficients, [0, 0], atol=0)
            self.assertEqual(model.intercept, 7.5)
            self.assertEqual(diag['coefficient_measure'], 0)

    def test_loose_budget_recovers_weighted_ols_and_excludes_zero_weight_rows(self):
        rng = np.random.default_rng(18)
        X = rng.normal(size=(30, 3))
        y = X @ [2., -.5, .2] + 4 + rng.normal(0, .2, 30)
        q = np.arange(30, dtype=float)
        design = np.column_stack([X, np.ones(len(X))])
        expected = np.linalg.lstsq(design * np.sqrt(q)[:, None], y * np.sqrt(q), rcond=None)[0]
        for kind in ('lasso', 'ridge'):
            model, _ = self.fit(X, y, kind, 100, sample_weight=q)
            assert_allclose(np.r_[model.coefficients, model.intercept], expected, atol=2e-5)
            modified = y.copy(); modified[0] = 1e5
            second, _ = self.fit(X, modified, kind, 100, sample_weight=17 * q)
            assert_allclose(second.coefficients, model.coefficients, atol=1e-7)

    def test_portfolio_adapter_preserves_budget_and_reports_no_penalty(self):
        from portfolio.constraints import CoefficientConstraint
        from portfolio.engine.portfolio_fit import fit_portfolio_coefficient_constrained
        rng = np.random.default_rng(29)
        frame = pd.DataFrame(rng.normal(0, .01, (126, 8)))
        for kind in ('lasso', 'ridge'):
            result = fit_portfolio_coefficient_constrained(frame, constraint=CoefficientConstraint(kind, .05))
            beta = result.model.coefficients
            measure = np.abs(beta).sum() if kind == 'lasso' else beta @ beta
            self.assertLessEqual(measure, .05 + 1e-8)
            self.assertAlmostEqual(result.weights.sum(), 1)
            assert_allclose(result.weights.iloc[:-1], 1/8-beta)
            self.assertEqual(result.objective, result.training_loss)
            self.assertAlmostEqual(result.training_loss, result.diagnostics['mse'])

    def test_invalid_budgets_and_sample_weights_are_rejected(self):
        from portfolio.constraints import CoefficientConstraint
        for budget in (-1, np.nan, np.inf, True, '1'):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                CoefficientConstraint('lasso', budget)
        with self.assertRaises(ValueError):
            CoefficientConstraint('none', 1)
        for q in ([0, 0, 0], [0, 0, 1], [1, -1, 1]):
            with self.subTest(weights=q), self.assertRaises(ValueError):
                self.fit([[1], [2], [3]], [1, 2, 3], 'lasso', 1, sample_weight=q)


if __name__ == '__main__':
    unittest.main()
