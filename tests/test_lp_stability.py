import importlib
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.decisions.linear_program_allocation import (
    LPConstraints, LPObjective, build_lp_context, solve_linear_portfolio,
)
from portfolio.engine.lp_risk_tuning_workflow import LPRiskTuningConfig, _rollout


class EWPenaltyTests(unittest.TestCase):
    def objective(self, **kwargs):
        self.assertIn('ew_penalty', LPObjective.__dataclass_fields__, 'EW penalty is missing')
        return LPObjective(**kwargs)

    def context(self):
        return build_lp_context(np.array([[.01, .01], [-.01, -.01]]),
            LPConstraints(leverage_limit=1.5, position_limit=1.5))

    def test_primary_ew_penalty_changes_the_optimum_not_only_a_tie(self):
        old, _ = solve_linear_portfolio(self.context(), [.001, 0])
        new, diag = solve_linear_portfolio(self.context(), [.001, 0],
            objective=self.objective(ew_penalty=.001))
        assert_allclose(old, [1.25, -.25], atol=1e-6)
        assert_allclose(new, [.5, .5], atol=1e-6)
        self.assertAlmostEqual(diag['utility'], .0005, places=9)

    def test_ew_penalty_is_distinct_from_cost_and_preserves_combined_utility(self):
        previous = np.array([1., 0.])
        w, diag = solve_linear_portfolio(self.context(), [.001, 0], previous_weights=previous,
            objective=self.objective(transaction_cost=.0001, ew_penalty=.001))
        assert_allclose(w, [.5, .5], atol=1e-6)
        self.assertAlmostEqual(diag['transaction_cost'], .0001, places=9)
        self.assertAlmostEqual(diag['utility'], .0004, places=9)
        self.assertLessEqual(diag['secondary_gap'], 1.05e-10)

    def test_invalid_ew_penalties_are_rejected(self):
        for value in (-1, np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                self.objective(ew_penalty=value)

    def test_shared_rollout_applies_each_models_own_leverage_constraint(self):
        dates = pd.bdate_range('2025-01-02', periods=15)
        r = pd.DataFrame(np.tile(np.array([.01, -.005, .002]), 5)[:, None]*np.ones((1, 2)),
                         index=dates, columns=['A', 'B'])
        predictions = pd.DataFrame([[.003, -.003]]*3, index=dates[-3:], columns=r.columns)
        config = LPRiskTuningConfig(window=10)
        objectives = {name: LPObjective(transaction_cost=.0005) for name in ('Current', 'Lower')}
        limits = dict(Current=LPConstraints(leverage_limit=1.5, position_limit=1.5),
                      Lower=LPConstraints(leverage_limit=1.2, position_limit=1.5))
        try:
            result = _rollout(r, predictions, objectives, config, limits, lambda _: None)
        except TypeError as error:
            self.fail('Per-model constraints are not implemented: '+str(error))
        assert_allclose(result['Current_weights'].abs().sum(axis=1), 1.5, atol=1e-6)
        assert_allclose(result['Lower_weights'].abs().sum(axis=1), 1.2, atol=1e-6)


class StabilityWorkflowTests(unittest.TestCase):
    def module(self):
        name = 'portfolio.engine.lp_stability_workflow'
        self.assertIsNotNone(importlib.util.find_spec(name), 'stability workflow is missing')
        return importlib.import_module(name)

    def test_selection_favors_lower_drawdown_with_return_and_volatility_guardrails(self):
        m = self.module()
        scores = pd.DataFrame([
            [0., .1, .02, .1, .1],
            [.0001, .085, .019, .08, .09],
            [.0005, .15, .021, .07, .2],
            [.001, .07, .01, .04, .12],
        ], index=['LP_Current', 'Safe', 'Volatile', 'LowReturn'],
        columns=['ew_penalty', 'cumulative_return', 'worst_block_volatility',
                 'worst_block_drawdown', 'sharpe'])
        selected, search = m.select_ew_penalty(scores, return_retention=.8)
        self.assertEqual(selected, 'Safe')
        self.assertFalse(search.loc['Volatile', 'eligible'])
        self.assertFalse(search.loc['LowReturn', 'eligible'])
        failed = scores.drop(index='Safe')
        selected, _ = m.select_ew_penalty(failed, return_retention=.8)
        self.assertEqual(selected, 'LP_Current')

    def fixture(self):
        m = self.module()
        dates = pd.bdate_range('2023-01-02', periods=150)
        r = pd.DataFrame(np.random.default_rng(924).normal(.001, .007, (150, 3)),
                         index=dates, columns=list('ABC'))
        d = lambda i: str(dates[i].date())
        periods = LPRiskTuningConfig(history_start=d(0), calibration_end=d(100),
            tuning_start=d(105), tuning_end=d(119), validation_start=d(120),
            validation_end=d(134), window=25, gammas=(0.,))
        config = m.LPStabilityConfig(periods=periods, ew_penalties=(.0001, .0005), perturbations=2)
        return r, config

    def run_small(self, returns, config):
        return self.module().run_lp_stability(returns, config=config, forecast_alpha=.0025,
            constraints=LPConstraints(position_limit=.6))

    def test_selection_is_frozen_before_validation_and_future_is_excluded(self):
        r, config = self.fixture()
        output = self.run_small(r, config)
        altered = r.copy()
        altered.loc[config.periods.validation_start:] += .003
        changed = self.run_small(altered, config)
        self.assertEqual(output.frozen, changed.frozen)
        pd.testing.assert_frame_equal(output.frames['ew_penalty_search'], changed.frames['ew_penalty_search'])
        first = pd.Timestamp(config.periods.validation_start)
        for name in ('LP_Current', 'LP_Lower_Leverage', 'LP_EW_Anchored'):
            assert_allclose(output.frames[name+'_weights'].loc[first],
                            changed.frames[name+'_weights'].loc[first], atol=1e-10)
        future = r.copy()
        future.loc[future.index > config.periods.validation_end] = np.nan
        stable = self.run_small(future, config)
        pd.testing.assert_frame_equal(output.frames['validation_summary'], stable.frames['validation_summary'])
        self.assertFalse(output.metadata['final_test_evaluated'])
        self.assertEqual(len(output.frames['validation_summary']), 4)  # Three versions plus EW reference.

    def test_stress_fees_and_predictions_do_not_change_main_holdings(self):
        r, config = self.fixture()
        result = self.run_small(r, config)
        for name in ('LP_Current', 'LP_Lower_Leverage', 'LP_EW_Anchored', 'EW'):
            daily = result.frames[name+'_returns']
            expected = np.prod(1+daily.gross_return-2*daily.cost)-1
            actual = result.frames['fee_stress_summary'].loc[name, 'cumulative_return']
            self.assertAlmostEqual(expected, actual, places=12)
        perturb = result.frames['prediction_stress_diagnostics']
        self.assertTrue((perturb['holdings_change'] >= 0).all())
        self.assertTrue(perturb['feasible'].all())
        weights = result.frames['LP_Lower_Leverage_weights']
        self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.2+1e-8)


if __name__ == '__main__':
    unittest.main()
