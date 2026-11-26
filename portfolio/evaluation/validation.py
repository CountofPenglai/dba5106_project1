"""January–June validation and the report's attribution and forecast tables."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from portfolio.config import COURSE_WINDOW, COURSE_TRAINING, FORECAST_CALIBRATION_END, HISTORY_START, VALIDATION
from portfolio.data.returns_loader import load_returns
from portfolio.engine.report_runner import run_models
from portfolio.evaluation.artifact_store import save_run
from portfolio.evaluation.attribution import attribute_returns
from portfolio.evaluation.forecast_metrics import evaluate_method
from portfolio.evaluation.provenance import run_provenance
from portfolio.paths import ORIGINAL_SHA256
from portfolio.models.course_cross_validation import COURSE_ALPHAS, CV_DAYS, CV_SPLITS
from portfolio.models.lasso_return_forecast import build_supervised, calibrate_alpha


def validate(output, source, models, start, end):
    if not pd.Timestamp(VALIDATION[0]) <= pd.Timestamp(start) <= pd.Timestamp(end) <= pd.Timestamp(VALIDATION[1]):
        raise ValueError('Validation dates must lie in January–June 2026')
    meta = run_provenance(source)
    if meta['data_sha256'] != ORIGINAL_SHA256:
        raise ValueError('Validation requires the original course release')
    panel = load_returns(source, start=HISTORY_START, end=end)
    if len(panel.columns) != 98:
        raise ValueError('Expected 98 retained asset portfolios')
    actual = panel.loc[start:end]
    if len(actual) < 2:
        raise ValueError('Choose at least two validation trading days to compute sample SD and Sharpe')
    alpha = None
    if any(s.family == 'lp' for s in models):
        training = build_supervised(panel.loc[:FORECAST_CALIBRATION_END])
        alpha, folds = calibrate_alpha(training, COURSE_TRAINING, end=FORECAST_CALIBRATION_END,
                                       n_splits=CV_SPLITS, test_size=CV_DAYS)
    meta.update(window=COURSE_WINDOW, start=start, end=end, assets=list(panel.columns),
                models=[asdict(s) for s in models], forecast_alpha=alpha,
                forecast_calibration_end=FORECAST_CALIBRATION_END if alpha is not None else None,
                cv=dict(alphas=COURSE_ALPHAS.tolist(), folds=CV_SPLITS, days_per_fold=CV_DAYS),
                metric_convention='Gross daily returns, sample SD, unannualized zero-rate Sharpe')
    result = run_models(panel, models, start=start, end=end, forecast_alpha=alpha, progress=print)
    save_run(result, output, meta)
    winners = actual.mean().nlargest(20).index
    attribution = {n: attribute_returns(result.weights[n], actual, winners)
                   for n in ('MinVar', 'LASSO_CV') if n in result.weights}
    if attribution:
        (pd.DataFrame(attribution) * 10000).rename_axis('average_contribution_bp_per_day').to_csv(
            output / 'metrics/attribution.csv')
        pd.DataFrame({n: result.weights[n].mean().loc[winners] for n in attribution},
                     index=winners).rename_axis('asset').to_csv(output/'diagnostics/retrospective_top20_weights.csv')
    if result.predictions is not None:
        trailing = panel.rolling(COURSE_WINDOW).mean().shift(1).loc[actual.index]
        stats, details = {}, {}
        for method, prediction in [('TrailingMean_126', trailing), ('LASSO_Forecast_126', result.predictions)]:
            stats[method], details[method] = evaluate_method(actual, prediction, trailing)
        if 'HistoricalTop20_EW' in result.returns:
            np.testing.assert_allclose(details['TrailingMean_126'].top_return,
                                       result.returns['HistoricalTop20_EW']['return'], rtol=0, atol=1e-12)
        pd.DataFrame.from_dict(stats, orient='index').rename_axis('method').to_csv(output/'metrics/forecasts.csv')
        pd.concat(details, names=['method','date']).to_csv(output/'diagnostics/daily_top20.csv')
        folds.to_csv(output/'diagnostics/forecast_calibration.csv', index=False)
        trailing.to_csv(output/'predictions/trailing_mean.csv')
    return result.summary
