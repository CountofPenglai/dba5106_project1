import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from portfolio.config import TrainingConfig
from portfolio.splits import TrainingWindow
from portfolio.direct_lp import DirectLPConfig, run_direct_lp, run_frozen_lp
from portfolio.linear_decision import LPConstraints


class DirectLPTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range('2023-01-02', periods=200)
        returns = pd.DataFrame(np.random.default_rng(182).normal(.0008, .01, (200, 3)),
                               index=dates, columns=list('ABC'))
        d = lambda i: str(dates[i].date())
        config = DirectLPConfig(history_start=d(0), calibration_end=d(140),
                               development_start=d(141), development_end=d(164),
                               validation_start=d(165), validation_end=d(175),
                               alphas=(.001, .01), n_splits=3, test_size=3)
        windows = {'small': TrainingConfig(TrainingWindow(observations=25))}
        return returns, config, windows, TrainingWindow(observations=40)

    def run_small(self, returns=None, **kwargs):
        r, config, windows, risk = self.fixture()
        return run_direct_lp(r if returns is None else returns, config=config,
                             forecast_configs=windows, risk_window=risk,
                             constraints=LPConstraints(position_limit=.6), **kwargs)

    def test_default_evaluates_only_validation_and_optional_history_preserves_weights(self):
        r, config, _, _ = self.fixture()
        output = self.run_small()
        expected = r.loc[config.validation_start:config.validation_end].index
        self.assertTrue(output.frames['LP_LASSO_small_weights'].index.equals(expected))
        self.assertNotIn('development_summary', output.frames)
        self.assertEqual(output.metadata['development_days'], 0)
        self.assertFalse(output.metadata['development_backtest_included'])
        full = self.run_small(include_development=True)
        self.assertIn('development_summary', full.frames)
        self.assertGreater(full.metadata['development_days'], 0)
        for key in ['LP_LASSO_small', 'LP_Mean_small', 'EW']:
            assert_allclose(output.frames[key+'_weights'], full.frames[key+'_weights'].loc[expected], atol=1e-10)
        self.assertEqual(output.frozen['alphas'], full.frozen['alphas'])

    def test_direct_weights_are_causal_and_returns_are_realized(self):
        r, config, _, _ = self.fixture()
        output = self.run_small()
        self.assertEqual(set(output.frames['validation_summary'].index),
                         {'LP_LASSO_small', 'LP_Mean_small', 'EW'})
        self.assertFalse(output.metadata['decision_model_trained'])
        self.assertFalse(output.metadata['final_test_evaluated'])
        for key in ['LP_LASSO_small', 'LP_Mean_small', 'EW']:
            w = output.frames[key+'_weights']
            assert_allclose(output.frames[key+'_returns']['return'],
                            (w*r.loc[w.index]).sum(axis=1), atol=1e-12)
            self.assertLessEqual(w.abs().sum(axis=1).max(), 1.5+1e-7)
            self.assertLessEqual(w.abs().to_numpy().max(), .6+1e-8)
        diagnostic = output.frames['LP_LASSO_small_diagnostics']
        self.assertTrue((pd.to_datetime(diagnostic['history_end']) < diagnostic.index).all())
        date = pd.Timestamp(config.validation_start)
        changed = r.copy(); changed.loc[date:] += .003
        altered = self.run_small(changed)
        for key in ['LP_LASSO_small', 'LP_Mean_small']:
            assert_allclose(output.frames[key+'_weights'].loc[:date],
                            altered.frames[key+'_weights'].loc[:date], atol=1e-10)
        self.assertEqual(output.frozen['alphas'], altered.frozen['alphas'])

    def test_future_exclusion_and_frozen_final_test(self):
        r, config, _, _ = self.fixture()
        output = self.run_small()
        future = r.copy(); future.loc[future.index > config.validation_end] = np.nan
        stable = self.run_small(future)
        assert_allclose(output.frames['LP_LASSO_small_weights'], stable.frames['LP_LASSO_small_weights'])
        dates = pd.bdate_range(r.index[-1]+pd.offsets.BDay(), '2026-08-28')
        extra = pd.DataFrame(np.random.default_rng(13).normal(.0005, .008, (len(dates),3)), index=dates, columns=r.columns)
        extended = pd.concat([r,extra])
        with patch('portfolio.direct_lp.calibrate_alpha', side_effect=AssertionError('must stay frozen')):
            final = run_frozen_lp(extended, output.frozen)
        self.assertEqual(final.metadata['evaluation_days'], 20)
        self.assertFalse(final.metadata['alpha_reselected'])
        self.assertEqual(final.frozen, output.frozen)
        from copy import deepcopy
        changed_code = deepcopy(output.frozen)
        changed_code['code_sha256']['portfolio/forecasting.py'] = 'different source'
        with self.assertRaisesRegex(ValueError, 'code'):
            run_frozen_lp(extended, changed_code)
        for date in ['2026-08-03', '2026-08-12', '2026-08-28']:
            with self.assertRaises(ValueError):
                run_frozen_lp(extended.drop(index=pd.Timestamp(date)), output.frozen)
        changed = extended.copy(); changed.iloc[100,0] += .002
        with self.assertRaisesRegex(ValueError, 'history'):
            run_frozen_lp(changed, output.frozen)
        with self.assertRaisesRegex(ValueError, 'asset'):
            run_frozen_lp(extended[['C','B','A']], output.frozen)

    def test_invalid_config_and_missing_history_fail(self):
        from dataclasses import replace
        r, config, windows, risk = self.fixture()
        with self.assertRaises(ValueError):
            run_direct_lp(r, config=replace(config, validation_end='2026-08-03'))
        with self.assertRaises(ValueError):
            run_direct_lp(r, config=replace(config, validation_end='2023'))
        bad = r.copy(); bad.iloc[100,0] = np.nan
        with self.assertRaises(ValueError):
            self.run_small(bad)
        with self.assertRaises(ValueError):
            self.run_small(r.iloc[:-25])


if __name__ == '__main__':
    unittest.main()
