import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.covariance import LedoitWolf

from portfolio.engine.portfolio_fit import fit_portfolio_growth as fit_selected
from portfolio.models.portfolio_variants import (
    ExperimentLoss, ExperimentConstraints, fit_portfolio_variant, size_bm_grid_edges,
)
from portfolio.losses import LogGrowthLoss
from portfolio.constraints import PortfolioConstraints
from portfolio.regularization import Regularization


class PortfolioVariantTests(unittest.TestCase):
    @staticmethod
    def history():
        rng = np.random.default_rng(121)
        common = rng.normal(0, .008, (36, 1))
        noise = rng.normal(0, .006, (36, 4))
        return pd.DataFrame(common + noise * [3, 1, .6, .4], columns=list('ABCD'))

    def test_zero_overlays_match_selected_lasso(self):
        history = self.history()
        args = dict(regularization=Regularization('lasso', 1e-5))
        original = fit_selected(history, loss=LogGrowthLoss(0, 2), constraints=PortfolioConstraints(), **args)
        variant = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2),
                                        constraints=ExperimentConstraints(), **args)
        assert_allclose(variant.weights, original.weights, atol=1e-9)
        self.assertAlmostEqual(variant.objective, original.objective, places=10)

    def test_turnover_penalty_reduces_distance_to_previous_holdings(self):
        history = self.history()
        previous = pd.Series([.25] * 4, index=history.columns)
        args = dict(loss=ExperimentLoss(0, 2), previous_weights=previous)
        plain = fit_portfolio_variant(history, **args)
        penalized = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2, turnover_strength=1),
                                          previous_weights=previous)
        movement = lambda fit: .5 * np.abs(fit.weights.to_numpy() - previous.to_numpy()).sum()
        self.assertLess(movement(penalized), movement(plain) - 1e-5)
        self.assertAlmostEqual(penalized.diagnostics['turnover'], movement(penalized), places=8)

    def test_weight_cap_applies_to_fitted_portfolio(self):
        history = self.history()
        baseline = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2))
        cap = .3
        capped = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2),
                                       constraints=ExperimentConstraints(weight_limit=cap))
        self.assertGreater(np.abs(baseline.weights).max(), cap)
        self.assertLessEqual(np.abs(capped.weights).max(), cap + 1e-7)
        self.assertAlmostEqual(capped.weights.sum(), 1, places=8)
        self.assertAlmostEqual(capped.diagnostics['max_abs_weight'], np.abs(capped.weights).max(), places=8)

    def test_ledoit_wolf_uses_training_covariance(self):
        history = self.history()
        fitted = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2),
                                       covariance_estimator='ledoit_wolf')
        expected = LedoitWolf().fit(history.to_numpy())
        self.assertAlmostEqual(fitted.diagnostics['ledoit_wolf_shrinkage'], expected.shrinkage_, places=8)
        w = fitted.weights.to_numpy()
        risk = float(w @ expected.covariance_ @ w)
        ew = np.full(4, .25)
        ew_risk = float(ew @ expected.covariance_ @ ew)
        self.assertLessEqual(risk, ew_risk + 1e-9)

    def test_grid_edges_penalty_reduces_neighbor_gap(self):
        history = self.history()
        edges = [(0, 1), (1, 2), (2, 3)]
        plain = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2), neighbor_edges=edges)
        smooth = fit_portfolio_variant(history, loss=ExperimentLoss(0, 2, smoothness_strength=10),
                                       neighbor_edges=edges)
        gap = lambda fit: np.mean([abs(fit.weights.iloc[i] - fit.weights.iloc[j]) for i, j in edges])
        self.assertLess(gap(smooth), gap(plain))
        self.assertAlmostEqual(smooth.diagnostics['neighbor_smoothness'], gap(smooth), places=8)

    def test_size_bm_grid_edges_keeps_only_adjacent_available_assets(self):
        edges = size_bm_grid_edges(['SMALL LoBM', 'ME1 BM2', 'ME2 BM1'])
        self.assertEqual(set(map(tuple, edges)), {(0, 1), (0, 2)})


if __name__ == '__main__':
    unittest.main()
