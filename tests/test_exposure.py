import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.constraints import PortfolioConstraints
from portfolio.losses import PortfolioLoss
from portfolio.regularization import Regularization
from portfolio.models.constrained_portfolio import fit_constrained_portfolio


class ExposurePenaltyTests(unittest.TestCase):
    @staticmethod
    def history():
        # A exceeds B by .01 in every scenario. Changing the allocation changes
        # growth and gross exposure while leaving training variance unchanged.
        second = np.array([-.10, .10, -.08, .08, -.05, .05])
        return pd.DataFrame({"A": second + .01, "B": second})

    def test_invalid_exposure_strength_is_rejected(self):
        for gamma in (-1, np.nan, np.inf):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    PortfolioLoss(exposure_strength=gamma)

    def test_zero_exposure_strength_preserves_the_existing_objective(self):
        history = self.history()
        baseline = fit_constrained_portfolio(
            history,
            loss=PortfolioLoss(reward_strength=.03),
        )
        explicit_zero = fit_constrained_portfolio(
            history,
            loss=PortfolioLoss(reward_strength=.03, exposure_strength=0),
        )

        assert_allclose(explicit_zero.weights, baseline.weights, atol=1e-9)
        self.assertAlmostEqual(explicit_zero.training_loss, baseline.training_loss)
        self.assertAlmostEqual(explicit_zero.objective, baseline.objective)
        expected_unit = max(0, (explicit_zero.weights.abs().sum() - 1) / .5) ** 2
        self.assertAlmostEqual(
            explicit_zero.diagnostics["exposure_unit_penalty"], expected_unit
        )
        self.assertEqual(explicit_zero.diagnostics["exposure_penalty"], 0)

    def test_positive_exposure_strength_matches_independent_scalar_grid(self):
        history = self.history()
        returns = history.to_numpy()
        eta, gamma, lam = .03, .10, 1e-5
        result = fit_constrained_portfolio(
            history,
            regularization=Regularization("lasso", lam),
            loss=PortfolioLoss(reward_strength=eta, exposure_strength=gamma),
        )

        first_weight = np.linspace(-.25, 1.25, 100001)
        grid_returns = (returns[:, :1] * first_weight
                        + returns[:, 1:] * (1 - first_weight))
        variances = np.var(grid_returns, axis=0)
        ew_variance = float(np.var(returns.mean(axis=1)))
        gross = np.abs(first_weight) + np.abs(1 - first_weight)
        unit_penalty = np.maximum(0, (gross - 1) / .5) ** 2
        losses = (variances - eta * np.log1p(grid_returns).mean(axis=0)
                  + gamma * ew_variance * unit_penalty
                  + lam * np.abs(.5 - first_weight))
        losses[gross > 1.5] = np.inf
        best = int(np.argmin(losses))

        assert_allclose(result.weights.iloc[0], first_weight[best], atol=3e-5)
        self.assertAlmostEqual(result.objective, losses[best], delta=1e-8)
        expected_unit = max(0, (result.weights.abs().sum() - 1) / .5) ** 2
        self.assertAlmostEqual(result.diagnostics["exposure_unit_penalty"], expected_unit)
        self.assertAlmostEqual(result.diagnostics["exposure_scale"], ew_variance)
        self.assertAlmostEqual(
            result.diagnostics["exposure_penalty"], gamma * ew_variance * expected_unit
        )
        self.assertAlmostEqual(
            result.training_loss,
            result.diagnostics["variance"] - result.diagnostics["reward"]
            + result.diagnostics["exposure_penalty"],
        )
        self.assertAlmostEqual(
            result.objective,
            result.training_loss + result.diagnostics["penalty"],
        )

    def test_higher_exposure_strength_reduces_gross_exposure(self):
        history = self.history()
        low = fit_constrained_portfolio(
            history,
            loss=PortfolioLoss(reward_strength=.03, exposure_strength=0),
        )
        high = fit_constrained_portfolio(
            history,
            loss=PortfolioLoss(reward_strength=.03, exposure_strength=1),
        )

        self.assertGreater(low.weights.abs().sum(), high.weights.abs().sum() + .01)

    def test_unit_gross_limit_defines_zero_exposure_penalty(self):
        history = self.history()
        result = fit_constrained_portfolio(
            history,
            loss=PortfolioLoss(reward_strength=.03, exposure_strength=3),
            constraints=PortfolioConstraints(leverage_limit=1),
        )

        self.assertTrue(np.isfinite(result.objective))
        self.assertLessEqual(result.weights.abs().sum(), 1 + 1e-7)
        self.assertEqual(result.diagnostics["exposure_unit_penalty"], 0)
        self.assertEqual(result.diagnostics["exposure_penalty"], 0)


if __name__ == "__main__":
    unittest.main()
