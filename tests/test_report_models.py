import unittest
import numpy as np
import pandas as pd


class ReportModelTests(unittest.TestCase):
    def history(self, n=252, p=24):
        rng = np.random.default_rng(412)
        market = rng.normal(.0004, .009, (n, 1))
        return pd.DataFrame(market * np.linspace(.3, 1.7, p) + rng.normal(0, .006, (n, p)),
                            index=pd.bdate_range('2024-01-01', periods=n),
                            columns=[f'a{i}' for i in range(p)])

    def test_catalog_keeps_all_report_configurations(self):
        from portfolio.model_catalog import MODELS, BASELINE_NAMES
        self.assertEqual(len(BASELINE_NAMES), 6)
        self.assertEqual(len(MODELS), 19)
        self.assertEqual(MODELS['LASSO_ExposurePenalty'].loss.exposure_strength, 2)
        self.assertEqual(MODELS['LASSO_ExposurePenalty'].loss.reward_strength, 0)
        self.assertEqual(MODELS['LP_TradingPenalty'].transaction_cost, .0005)
        self.assertEqual(MODELS['LASSO_WeightCap_10pct'].constraints.weight_limit, .1)

    def test_all_single_stage_report_models_are_feasible(self):
        from portfolio.model_catalog import MODELS, fit_model
        history = self.history()
        for name, spec in MODELS.items():
            if spec.family == 'lp':
                continue
            with self.subTest(name=name):
                fitted = fit_model(history, spec)
                w = fitted.weights.to_numpy()
                self.assertAlmostEqual(w.sum(), 1, places=8)
                self.assertTrue(np.isfinite(w).all())
                if spec.family == 'constrained':
                    realized = history.to_numpy() @ w
                    ew = history.mean(axis=1).to_numpy()
                    self.assertLessEqual(np.abs(w).sum(), 1.5 + 1e-6)
                    self.assertLessEqual(realized.var(), ew.var() + 1e-8)
                    if spec.constraints.weight_limit:
                        self.assertLessEqual(np.abs(w).max(), .1 + 1e-7)
                    if spec.constraints.beta_bounds:
                        lower, upper = spec.constraints.beta_bounds
                        beta = np.cov(realized, ew, ddof=0)[0, 1] / ew.var()
                        self.assertGreaterEqual(beta, lower - 1e-6)
                        self.assertLessEqual(beta, upper + 1e-6)
                if spec.coefficient_constraint:
                    self.assertLessEqual(spec.coefficient_constraint.measure(fitted.model.coefficients), 1 + 1e-7)

    def test_turnover_penalty_uses_previous_target_and_discourages_changes(self):
        from portfolio.model_catalog import MODELS, fit_model
        history = self.history()
        previous = pd.Series(1/history.shape[1], index=history.columns)
        ordinary = fit_model(history, MODELS['LASSO_ExposurePenalty'], previous_weights=previous)
        penalized = fit_model(history, MODELS['LASSO_Turnover_1'], previous_weights=previous)
        self.assertLessEqual((penalized.weights-previous).abs().sum(),
                             (ordinary.weights-previous).abs().sum()+1e-7)
        self.assertAlmostEqual(penalized.diagnostics['target_turnover'],
                               .5*(penalized.weights-previous).abs().sum())

    def test_weight_cap_and_beta_configuration_validation(self):
        from portfolio.constraints import PortfolioConstraints
        for options in ({'weight_limit': 0}, {'beta_bounds': (1, .5)}, {'beta_bounds': (0, np.nan)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PortfolioConstraints(**options)


if __name__ == '__main__':
    unittest.main()
