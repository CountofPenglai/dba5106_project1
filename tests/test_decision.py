import unittest
import numpy as np
from numpy.testing import assert_allclose

from portfolio.decision import fit_mixture, mix_weights


class DecisionTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(75)
        z = rng.normal(size=(90, 3))
        z -= z.mean(axis=0)
        return z * [.015, .004, .009] + [.0015, .0003, .0009]

    def grid(self, z, objective, gamma=0):
        theta = np.array([[a/200, b/200, 1-(a+b)/200]
                          for a in range(201) for b in range(201-a)])
        mean = theta @ z.mean(axis=0)
        sd = np.sqrt(np.einsum('ij,jk,ik->i', theta, np.cov(z, rowvar=False), theta))
        scores = mean/sd if objective == 'sharpe' else mean-gamma*sd
        return float(scores.max())

    def test_two_objectives_match_independent_simplex_grid(self):
        z = self.fixture()
        for objective, gamma in [('sharpe', 0), ('return_volatility', .12), ('return_volatility', 0)]:
            fit = fit_mixture(z, objective=objective, gamma=gamma)
            self.assertTrue((fit.theta >= 0).all())
            self.assertAlmostEqual(fit.theta.sum(), 1)
            r = z @ fit.theta
            actual = r.mean()/r.std(ddof=1) if objective == 'sharpe' else r.mean()-gamma*r.std(ddof=1)
            self.assertGreaterEqual(actual, self.grid(z, objective, gamma)-2e-7)
            self.assertAlmostEqual(actual, fit.diagnostics['score'], places=10)

    def test_nonpositive_means_choose_best_vertex(self):
        z = self.fixture()
        z = z-z.mean(axis=0)-[.001, .002, .003]
        fit = fit_mixture(z, objective='sharpe')
        best = np.argmax(z.mean(axis=0)/z.std(axis=0, ddof=1))
        assert_allclose(fit.theta, np.eye(3)[best])
        self.assertEqual(fit.diagnostics['branch'], 'nonpositive_means')

    def test_duplicate_candidates_prefer_equal_weight_component(self):
        r = np.array([.01, -.005, .02, -.008, .012])
        z = np.column_stack([r, r, r])
        for objective in ['sharpe', 'return_volatility']:
            fit = fit_mixture(z, objective=objective, gamma=.1)
            assert_allclose(fit.theta, [0, 0, 1], atol=2e-4)

    def test_invalid_and_zero_risk_are_explicit(self):
        for z in [np.ones((10, 3))*.01, np.zeros((10, 3))]:
            with self.assertRaisesRegex(ValueError, 'risk|volatility'):
                fit_mixture(z, objective='sharpe')
        for z in [np.ones((2, 2)), np.full((10, 3), np.nan)]:
            with self.assertRaises(ValueError): fit_mixture(z, objective='sharpe')
        with self.assertRaises(ValueError): fit_mixture(self.fixture(), objective='unknown')
        with self.assertRaises(ValueError): fit_mixture(self.fixture(), objective='return_volatility', gamma=-1)

    def test_weight_mixing_reconstructs_realized_returns(self):
        weights = np.array([[[.8, .2], [.4, .6], [.5, .5]], [[1.1, -.1], [.3, .7], [.5, .5]]])
        theta = np.array([.2, .3, .5])
        r = np.array([[.02, -.01], [-.015, .01]])
        actual = mix_weights(weights, theta)
        assert_allclose((actual*r).sum(axis=1), np.einsum('tjp,tp->tj', weights, r) @ theta)
        with self.assertRaises(ValueError): mix_weights(weights, [-.1, .3, .8])

    def test_sharpe_scale_invariance_above_degenerate_threshold(self):
        z = (np.random.default_rng(3).normal(0, .01, (90, 3)) + [.001, .0005, .0001]) * 1e-4
        small = fit_mixture(z, objective='sharpe')
        large = fit_mixture(z*1e4, objective='sharpe')
        assert_allclose(small.theta, large.theta, atol=3e-5)
        self.assertAlmostEqual(small.diagnostics['score'], large.diagnostics['score'], places=7)
