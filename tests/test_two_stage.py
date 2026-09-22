import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from portfolio.config import TrainingConfig
from portfolio.splits import TrainingWindow
from portfolio.two_stage import ExperimentConfig, run_experiment, run_frozen


class TwoStageTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range('2023-01-02', periods=200)
        rng = np.random.default_rng(919)
        r = pd.DataFrame(rng.normal(.0008, .008, (200, 3)), index=dates, columns=list('ABC'))
        d = lambda i: str(dates[i].date())
        config = ExperimentConfig(history_start=d(0), calibration_end=d(140), decision_start=d(141),
                                 decision_end=d(164), validation_start=d(165), validation_end=d(175),
                                 alphas=(.001, .01), n_splits=3, test_size=3, gammas=(0., .1))
        windows = {'small': TrainingConfig(TrainingWindow(observations=25))}
        risk = TrainingWindow(observations=40)
        return r, config, windows, risk

    def test_chronology_freezing_and_realized_reconstruction(self):
        r, config, windows, risk = self.fixture()
        result = run_experiment(r, config=config, forecast_configs=windows, risk_window=risk)
        self.assertEqual(len(result.frames['validation_summary']), 4)
        self.assertFalse(result.metadata['final_test_evaluated'])
        for name, model in result.frozen['models'].items():
            weights = result.frames[name+'_weights']
            daily = result.frames[name+'_returns']['return']
            assert_allclose((weights*r.loc[weights.index]).sum(axis=1), daily, atol=1e-12)
            theta = np.asarray(model['theta'])
            self.assertAlmostEqual(theta.sum(), 1)
            self.assertTrue((theta >= 0).all())
            self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.5+1e-7)
        diagnostics = result.frames['LASSO_small_forecast_diagnostics']
        self.assertTrue((pd.to_datetime(diagnostics['train_end']).to_numpy() < diagnostics.index.to_numpy()).all())
        altered = r.copy()
        altered.loc[config.validation_start:config.validation_end] += .002
        again = run_experiment(altered, config=config, forecast_configs=windows, risk_window=risk)
        before = result.frames['decision_fits']
        after = again.frames['decision_fits']
        assert_allclose(before[['a','b','c']], after[['a','b','c']], atol=1e-10)
        future = r.copy()
        future.loc[future.index > pd.Timestamp(config.validation_end)] = np.nan
        stable = run_experiment(future, config=config, forecast_configs=windows, risk_window=risk)
        for name in result.frozen['models']:
            assert_allclose(result.frames[name+'_weights'], stable.frames[name+'_weights'])
        self.assertEqual(result.frozen['alphas'], stable.frozen['alphas'])
        with self.assertRaisesRegex(ValueError, 'coverage|[Aa]vailable|test'):
            run_frozen(r, result.frozen)

    def test_invalid_boundaries_and_missing_values_are_rejected(self):
        r, config, windows, risk = self.fixture()
        with self.assertRaises(ValueError):
            run_experiment(r, config=replace(config, calibration_end=config.decision_end), forecast_configs=windows, risk_window=risk)
        bad = r.copy(); bad.iloc[100,0] = np.nan
        with self.assertRaises(ValueError):
            run_experiment(bad, config=config, forecast_configs=windows, risk_window=risk)
        with self.assertRaises(ValueError):
            run_experiment(r, config=replace(config, validation_end='2026-08-03'), forecast_configs=windows, risk_window=risk)

    def test_reserved_test_uses_frozen_parameters_without_fitting_decisions(self):
        from unittest.mock import patch
        r, config, windows, risk = self.fixture()
        development = run_experiment(r, config=config, forecast_configs=windows, risk_window=risk)
        dates = pd.bdate_range(r.index[-1]+pd.offsets.BDay(), '2026-08-28')
        extra = pd.DataFrame(np.random.default_rng(15).normal(.0005, .008, (len(dates),3)), index=dates, columns=r.columns)
        extended = pd.concat([r, extra])
        with patch('portfolio.two_stage.calibrate_alpha', side_effect=AssertionError('must not recalibrate')), patch('portfolio.two_stage.fit_mixture', side_effect=AssertionError('must not refit')):
            final = run_frozen(extended, development.frozen)
        self.assertTrue(final.metadata['final_test_evaluated'])
        self.assertEqual(final.metadata['first_date'], '2026-08-03')
        self.assertEqual(final.metadata['last_date'], '2026-08-28')
        self.assertEqual(final.frozen['models'], development.frozen['models'])
        self.assertEqual(len(final.frames['test_summary']), 4)
        for missing in [pd.Timestamp('2026-08-03'), pd.Timestamp('2026-08-12')]:
            with self.assertRaisesRegex(ValueError, 'coverage'):
                run_frozen(extended.drop(index=missing), development.frozen)
        modified = extended.copy(); modified.iloc[10,0] += .0001
        with self.assertRaisesRegex(ValueError, 'history changed'):
            run_frozen(modified, development.frozen)
        with self.assertRaisesRegex(ValueError, 'cover'):
            run_frozen(extended.iloc[:-1], development.frozen)
        with self.assertRaisesRegex(ValueError, 'assets'):
            run_frozen(extended[['B','A','C']], development.frozen)
