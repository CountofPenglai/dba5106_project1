import importlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.decisions import linear_program_allocation as lp


class RiskObjectiveTests(unittest.TestCase):
    def objective(self, **kwargs):
        self.assertTrue(hasattr(lp, 'LPObjective'), 'risk/cost objective is not implemented')
        return lp.LPObjective(**kwargs)

    def context(self):
        return lp.build_lp_context(np.array([[.02, 0], [-.02, 0]]),
            lp.LPConstraints(leverage_limit=1, position_limit=1,
                             mad_ratio=2, drawdown_ratio=10))

    def test_mad_penalty_selects_the_less_volatile_asset(self):
        context = self.context()
        original, _ = lp.solve_linear_portfolio(context, [.001, 0])
        cautious, diag = lp.solve_linear_portfolio(context, [.001, 0],
            objective=self.objective(mad_penalty=.1))
        self.assertGreater(original[0], .999)
        self.assertLess(cautious[0], 1e-6)
        self.assertAlmostEqual(diag['utility'], 0, places=8)
        self.assertLessEqual(diag['secondary_gap'], 1.05e-10)

    def test_cost_discourages_small_trades_and_charges_both_sides(self):
        context = self.context()
        previous = np.array([.25, .75])
        same, _ = lp.solve_linear_portfolio(context, [.001, 0],
            objective=self.objective(transaction_cost=.01), previous_weights=previous)
        assert_allclose(same, previous, atol=1e-7)
        traded, diag = lp.solve_linear_portfolio(context, [.001, 0],
            objective=self.objective(transaction_cost=.0001), previous_weights=previous)
        assert_allclose(traded, [1, 0], atol=2e-7)
        self.assertAlmostEqual(diag['transaction_cost'], .00015, places=9)
        self.assertAlmostEqual(diag['utility'], .00085, places=9)

    def test_zero_penalties_preserve_the_original_solution(self):
        old, old_diag = lp.solve_linear_portfolio(self.context(), [.001, 0])
        new, diag = lp.solve_linear_portfolio(self.context(), [.001, 0],
            objective=self.objective())
        assert_allclose(old, new, atol=1e-12)
        self.assertAlmostEqual(old_diag['predicted_return'], diag['predicted_return'], places=12)

    def test_mad_is_measured_on_the_combined_portfolio(self):
        context = lp.build_lp_context(np.array([[.02, -.02], [-.02, .02]]),
            lp.LPConstraints(position_limit=1))
        weights, diag = lp.solve_linear_portfolio(context, [.001, 0],
            objective=self.objective(mad_penalty=1))
        assert_allclose(weights, [.5, .5], atol=1e-8)
        self.assertAlmostEqual(diag['mad'], 0, places=10)

    def test_invalid_penalties_and_missing_pretrade_holdings_fail(self):
        for key in ('mad_penalty', 'transaction_cost'):
            for value in (-1, np.nan, np.inf, True):
                with self.assertRaises(ValueError):
                    self.objective(**{key: value})
        with self.assertRaisesRegex(ValueError, 'previous'):
            lp.solve_linear_portfolio(self.context(), [.001, 0],
                objective=self.objective(transaction_cost=.001))


class TuningWorkflowTests(unittest.TestCase):
    def module(self):
        name = 'portfolio.engine.lp_risk_tuning_workflow'
        self.assertIsNotNone(importlib.util.find_spec(name), 'training-only tuning is not implemented')
        return importlib.import_module(name)

    def fixture(self):
        workflow = self.module()
        dates = pd.bdate_range('2023-01-02', periods=155)
        returns = pd.DataFrame(np.random.default_rng(972).normal(.0008, .009, (155, 3)),
                               index=dates, columns=list('ABC'))
        d = lambda i: str(dates[i].date())
        config = workflow.LPRiskTuningConfig(history_start=d(0), calibration_end=d(100),
            tuning_start=d(110), tuning_end=d(119), validation_start=d(120),
            validation_end=d(129), window=25, gammas=(0., .1, 1.))
        return returns, config

    def run_small(self, returns, config):
        return self.module().run_lp_risk_tuning(returns, config=config, forecast_alpha=.0025,
            constraints=lp.LPConstraints(position_limit=.6))

    def test_validation_does_not_select_gamma_and_each_decision_is_causal(self):
        r, config = self.fixture()
        result = self.run_small(r, config)
        changed = r.copy()
        changed.loc[config.validation_start:] += .007
        altered = self.run_small(changed, config)
        self.assertEqual(result.frozen['mad_penalty'], altered.frozen['mad_penalty'])
        pd.testing.assert_frame_equal(result.frames['gamma_search'], altered.frames['gamma_search'])
        date = pd.Timestamp(config.validation_start)
        for name in ('LP_MAD_Selected', 'LP_Cost_Only', 'LP_Return_Only', 'EW'):
            assert_allclose(result.frames[name+'_weights'].loc[date],
                            altered.frames[name+'_weights'].loc[date], atol=1e-10)
        self.assertFalse(result.metadata['final_test_evaluated'])
        self.assertEqual(result.metadata['validation_candidate_count'], 1)
        diag = result.frames['forecast_diagnostics']
        self.assertTrue((pd.to_datetime(diag.train_end) < diag.index).all())

    def test_future_data_are_excluded_and_net_returns_use_drifted_holdings(self):
        r, config = self.fixture()
        result = self.run_small(r, config)
        future = r.copy()
        future.loc[future.index > config.validation_end] = np.nan
        stable = self.run_small(future, config)
        pd.testing.assert_frame_equal(result.frames['validation_summary'], stable.frames['validation_summary'])
        for name in ('LP_MAD_Selected', 'LP_Cost_Only', 'LP_Return_Only', 'EW'):
            weights = result.frames[name+'_weights']
            actual = r.loc[weights.index].to_numpy()
            w = weights.to_numpy()
            gross = np.sum(w * actual, axis=1)
            previous = np.vstack([np.full(3, 1/3),
                w[:-1]*(1+actual[:-1])/(1+gross[:-1,None])])
            costs = .0005*np.abs(w-previous).sum(axis=1)
            saved = result.frames[name+'_returns']
            assert_allclose(saved['return'], gross-costs, atol=1e-12)
            assert_allclose(saved['cost'], costs, atol=1e-12)
            self.assertLessEqual(saved.index.max(), pd.Timestamp(config.validation_end))

    def test_tuning_validation_overlap_and_test_period_are_rejected(self):
        _, config = self.fixture()
        for changes in ({'tuning_end': config.validation_start},
                        {'validation_end': '2026-07-01'},
                        {'gammas': (0., 0.)}, {'transaction_cost': -.01}):
            with self.assertRaises(ValueError):
                replace(config, **changes).validate()


class ForecastReuseTests(unittest.TestCase):
    def test_reference_requires_matching_data_forecaster_and_early_calibration(self):
        name = 'experiments.lasso_forecast_lp_risk'
        self.assertIsNotNone(importlib.util.find_spec(name), 'reuse CLI is not implemented')
        cli = importlib.import_module(name)
        from portfolio.engine.lp_risk_tuning_workflow import LPRiskTuningConfig
        from portfolio.evaluation.provenance import frame_digest, code_hashes
        config = LPRiskTuningConfig()
        r = pd.DataFrame([[.001, -.002], [-.003, .004]],
            index=pd.to_datetime(['2021-08-02', '2026-06-30']), columns=['A', 'B'])
        frozen = dict(kind='direct_asset_lp', schema_version=1, assets=['A', 'B'],
            config=dict(history_start=config.history_start, calibration_end=config.calibration_end,
                        validation_end=config.validation_end),
            forecast_configs={'252d': {'window': {'observations': 252, 'years': None, 'min_observations': 2},
                                      'half_life': None}},
            alphas={'252d': .0025}, development_data_digest=frame_digest(r), code_sha256=code_hashes())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frozen_decisions.json'
            path.write_text(json.dumps(frozen))
            self.assertEqual(cli.reuse_forecast_alpha(Path(directory), r, config), .0025)
            for mistake in ('data', 'code', 'calibration', 'window'):
                changed = json.loads(json.dumps(frozen))
                if mistake == 'data': changed['development_data_digest'] = 'bad'
                if mistake == 'code': changed['code_sha256']['portfolio/models/lasso_return_forecast.py'] = 'bad'
                if mistake == 'calibration': changed['config']['calibration_end'] = '2025-08-01'
                if mistake == 'window': changed['forecast_configs']['252d']['window']['observations'] = 126
                path.write_text(json.dumps(changed))
                with self.subTest(mistake=mistake), self.assertRaises(ValueError):
                    cli.reuse_forecast_alpha(Path(directory), r, config)


if __name__ == '__main__':
    unittest.main()
