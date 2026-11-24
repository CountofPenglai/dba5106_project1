import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


class ReportRunnerTests(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(194)
        return pd.DataFrame(rng.normal(.0003, .009, (335, 24)),
                            index=pd.bdate_range('2024-01-02', periods=335),
                            columns=[f'a{i}' for i in range(24)])

    def test_lp_and_lasso_do_not_use_current_or_future_returns(self):
        from portfolio.engine.report_runner import run_models
        from portfolio.model_catalog import MODELS
        frame = self.frame()
        start, end = frame.index[-2:]
        names = ['LASSO_ExposurePenalty', 'LASSO_Turnover_1', 'LP_Original', 'LP_TradingPenalty']
        options = dict(models=[MODELS[n] for n in names], start=start, end=end,
                       window=252, forecast_alpha=.002511886431509582)
        before = run_models(frame, **options)
        changed = frame.copy()
        changed.loc[start:] += np.linspace(-.02, .02, frame.shape[1])
        after = run_models(changed, **options)
        for name in names:
            assert_allclose(before.weights[name].loc[start], after.weights[name].loc[start], atol=1e-10)
            self.assertEqual(before.diagnostics[name].loc[start, 'history_count'], 252)
            self.assertLess(before.diagnostics[name].loc[start, 'history_end'], start)

    def test_gross_return_is_primary_and_trade_turnover_accounts_for_drift(self):
        from portfolio.engine.report_runner import run_models
        from portfolio.model_catalog import MODELS
        frame = self.frame()
        result = run_models(frame, [MODELS['EW']], start=frame.index[-3], end=frame.index[-1], window=126)
        daily, weights, diag = result.returns['EW'], result.weights['EW'], result.diagnostics['EW']
        assert_allclose(daily['return'], frame.loc[daily.index].mean(axis=1))
        assert_allclose(daily['net_return'], daily['return']-daily['trading_cost'])
        self.assertTrue((diag.target_turnover == 0).all())
        w = weights.iloc[0].to_numpy()
        first_returns = frame.loc[daily.index[0]].to_numpy()
        pretrade = w*(1+first_returns)/(1+w@first_returns)
        expected = .5*np.abs(weights.iloc[1]-pretrade).sum()
        self.assertAlmostEqual(diag.trade_turnover.iloc[1], expected)
        self.assertAlmostEqual(daily.trading_cost.iloc[1], .0005*2*expected)
        self.assertGreater(expected, 0)

    def test_baseline_cli_supports_both_windows_and_refuses_overwrite(self):
        frame = self.frame()
        raw = frame*100
        raw.index = frame.index.strftime('%Y%m%d')
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'data.csv'
            source.write_text('Average Equal Weighted Returns -- Daily\n'+raw.to_csv()+'\n')
            for count in (126, 252):
                output = Path(temp)/str(count)
                command = [sys.executable, str(ROOT/'run_baselines.py'), '--csv', str(source),
                           *(['--window', str(count)] if count != 126 else []), '--start', str(frame.index[-2].date()),
                           '--end', str(frame.index[-1].date()), '--output', str(output)]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                summary = pd.read_csv(output/'metrics/summary.csv')
                self.assertEqual(summary.model.tolist(), ['EW', 'MinVar', 'LASSO_CV', 'Ridge_CV'])
                for name in ('LASSO_CV', 'Ridge_CV'):
                    diagnostics = pd.read_csv(output / 'diagnostics' / f'{name}.csv')
                    self.assertTrue((diagnostics.history_count == count).all())
                    self.assertTrue((diagnostics.history_end < diagnostics.date).all())
                    self.assertTrue((diagnostics.selected_sklearn_alpha > 0).all())
                metadata = json.loads((output/'config/run_metadata.json').read_text())
                self.assertEqual(metadata['window'], count)
                self.assertEqual(metadata['metric_convention']['primary_returns'], 'gross')
                repeat = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(repeat.returncode, 0)
                self.assertIn('empty or new', repeat.stderr)

    def test_course_cv_matches_explicit_chronological_search(self):
        from sklearn.linear_model import Lasso, Ridge
        from portfolio.model_catalog import COURSE_MODELS, fit_model
        from portfolio.models.course_cross_validation import COURSE_ALPHAS
        from portfolio.models.min_variance_regression import RegressionProblem
        history = self.frame().iloc[:126, :5]
        problem = RegressionProblem.from_returns(history)
        for model, estimator in [('LASSO_CV', Lasso), ('Ridge_CV', Ridge)]:
            def fit(alpha, n):
                kwargs = {'tol': 1e-9, 'max_iter': 100000} if model == 'LASSO_CV' else {}
                return estimator(alpha=alpha, **kwargs).fit(problem.X[:n], problem.y[:n])
            errors = []
            for alpha in COURSE_ALPHAS:
                mse = []
                for end in (63, 84, 105):
                    pred = fit(alpha, end).predict(problem.X[end:end+21])
                    mse.append(np.mean((pred - problem.y[end:end+21])**2))
                errors.append(np.mean(mse))
            fitted = fit_model(history, COURSE_MODELS[model])
            alpha = fitted.diagnostics['selected_sklearn_alpha']
            self.assertAlmostEqual(errors[list(COURSE_ALPHAS).index(alpha)], min(errors), places=11)
            assert_allclose(fitted.weights, problem.portfolio_weights(fit(alpha, 126).coef_), atol=2e-5)

    def test_report_defaults_and_frozen_alpha_do_not_depend_on_results(self):
        from portfolio.engine.report_runner import load_forecast_settings
        from portfolio.config import EVALUATION_PERIODS
        settings = load_forecast_settings()
        self.assertEqual(settings['alpha'], .002511886431509582)
        self.assertEqual(settings['window'], 252)
        self.assertEqual(settings['reference_end'], settings['calibration_end'])
        self.assertEqual(EVALUATION_PERIODS['validation'], ('2026-01-02', '2026-06-30'))
        self.assertEqual(EVALUATION_PERIODS['july_test'], ('2026-07-01', '2026-07-31'))
        listed = subprocess.run([sys.executable, str(ROOT/'run_report.py'), '--list'],
                                cwd='/private/tmp', capture_output=True, text=True)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn('LP_TradingPenalty', listed.stdout)

    def test_frozen_forecast_cannot_be_used_before_its_calibration(self):
        from portfolio.engine.report_runner import verify_forecast_reference, load_forecast_settings
        with self.assertRaisesRegex(ValueError, 'after.*calibration'):
            verify_forecast_reference(self.frame(), load_forecast_settings(), start='2024-07-01')


if __name__ == '__main__':
    unittest.main()
