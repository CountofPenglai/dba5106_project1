from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

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
        names = ['LASSO_ExposurePenalty', 'LASSO_WeightCap_10pct', 'HistoricalTop20_EW', 'LP_Return']
        options = dict(models=[MODELS[n] for n in names], start=start, end=end,
                       forecast_alpha=.01)
        before = run_models(frame, **options)
        changed = frame.copy()
        changed.loc[start:] += np.linspace(-.02, .02, frame.shape[1])
        after = run_models(changed, **options)
        for name in names:
            assert_allclose(before.weights[name].loc[start], after.weights[name].loc[start], atol=1e-10)
            self.assertEqual(before.diagnostics[name].loc[start, 'history_count'], 126)
            self.assertLess(before.diagnostics[name].loc[start, 'history_end'], start)

    def test_equal_weight_and_top20_score_only_realized_gross_returns(self):
        from portfolio.engine.report_runner import run_models
        from portfolio.model_catalog import MODELS
        frame = self.frame()
        start, end = frame.index[-3], frame.index[-1]
        result = run_models(frame, [MODELS['EW'], MODELS['HistoricalTop20_EW']], start=start, end=end)
        assert_allclose(result.returns['EW']['return'], frame.loc[start:end].mean(axis=1))
        for day, weights in result.weights['HistoricalTop20_EW'].iterrows():
            prior = frame.loc[frame.index < day].tail(126)
            leaders = prior.mean().round(12).nlargest(20).index
            assert_allclose(weights.loc[leaders], .05)
            self.assertEqual((weights > 0).sum(), 20)
            self.assertAlmostEqual(result.returns['HistoricalTop20_EW'].at[day, 'return'],
                                   frame.loc[day, leaders].mean())

    def test_cli_refuses_to_change_holdout_or_reuse_a_result_directory(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from portfolio.cli import main
        from portfolio.evaluation.artifact_store import fresh_run
        with redirect_stderr(StringIO()):
            for args in (['--window', '252'], ['--period', 'final_test', '--models', 'LP_Return'],
                         ['--period', 'final_test', '--start', '2026-09-01']):
                with self.assertRaises(SystemExit):
                    main(args)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'result'
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                with fresh_run(target) as stage:
                    (stage/'data').write_text('partial')
                    raise RuntimeError('incomplete')
            self.assertFalse(target.exists())
            with fresh_run(target) as stage:
                (stage/'data').write_text('complete')
            with self.assertRaisesRegex(ValueError, 'empty or new'):
                with fresh_run(target):
                    pass
            self.assertEqual((target/'data').read_text(), 'complete')

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
