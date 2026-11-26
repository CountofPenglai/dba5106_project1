"""Shared daily fit, allocation and gross scoring for validation and final testing."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio.config import COURSE_WINDOW, COURSE_TRAINING
from portfolio.data.history_windows import rolling_windows
from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.decisions.linear_program_allocation import build_lp_context, solve_linear_portfolio, LPConstraints
from portfolio.evaluation.portfolio_metrics import summarize, summarize_holdings
from portfolio.model_catalog import fit_model
from portfolio.models.lasso_return_forecast import build_supervised, forecast_at


@dataclass
class ReportRun:
    summary: pd.DataFrame
    returns: dict[str, pd.DataFrame]
    weights: dict[str, pd.DataFrame]
    diagnostics: dict[str, pd.DataFrame]
    predictions: pd.DataFrame | None
    forecast_diagnostics: pd.DataFrame | None


def run_models(panel, models, *, start, end, forecast_alpha=None, progress=None):
    names = [s.name for s in models]
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose a nonempty list of unique portfolios')
    panel = panel.loc[:end]
    if panel.columns.has_duplicates or not np.isfinite(panel.to_numpy()).all():
        raise ValueError('History requires unique assets and complete returns')
    has_lp = any(s.family == 'lp' for s in models)
    if has_lp and forecast_alpha is None:
        raise ValueError('The LP requires the historically calibrated LASSO penalty')
    supervised = build_supervised(panel) if has_lp else None
    weights, daily, diagnostics = ({n: [] for n in names} for _ in range(3))
    dates, forecasts, forecast_info = [], [], []
    for day, history in rolling_windows(panel, start=start, end=end, window=COURSE_WINDOW):
        dates.append(day)
        if has_lp:
            forecast, info = forecast_at(supervised, day, COURSE_TRAINING, forecast_alpha)
            if (info['train_start'], info['train_end'], info['train_count']) != (
                    history.index[0], history.index[-1], len(history)):
                raise ValueError(f'Forecast and allocation windows differ on {day.date()}')
            info.pop('constant_feature_counts')
            forecasts.append(forecast)
            forecast_info.append(info)
        for spec in models:
            if spec.family == 'lp':
                context = build_lp_context(history, LPConstraints())
                w, diag = solve_linear_portfolio(context, forecast.to_numpy())
            elif spec.family == 'ranking':
                if len(history.columns) < 20:
                    raise ValueError('The historical top-20 portfolio needs at least 20 assets')
                rank = history.mean().round(12).rank(ascending=False, method='first')
                w, diag = (rank <= 20).to_numpy(dtype=float)/20, {}
            else:
                fit = fit_model(history, spec)
                w, diag = fit.weights.to_numpy(), fit.diagnostics or {}
            w = validate_portfolio_weights(w, len(history.columns))
            gross = float(w @ panel.loc[day].to_numpy())
            if gross <= -1:
                raise ValueError(f'Nonpositive realized wealth for {spec.name} on {day.date()}')
            weights[spec.name].append(w.copy())
            daily[spec.name].append(gross)
            diagnostics[spec.name].append({**diag, 'history_start': history.index[0],
                                          'history_end': history.index[-1], 'history_count': len(history)})
        if progress and (len(dates) == 1 or len(dates) % 21 == 0):
            progress(f'Evaluated {len(dates)} trading days through {day.date()}')
    index = pd.DatetimeIndex(dates, name='date')
    streams = {n: pd.DataFrame({'return': daily[n]}, index=index) for n in names}
    holdings = {n: pd.DataFrame(weights[n], columns=panel.columns, index=index) for n in names}
    checks = {n: pd.DataFrame(diagnostics[n], index=index) for n in names}
    actual = panel.loc[start:end]
    positions = panel.index.get_indexer(actual.index)
    for name in names:
        pd.testing.assert_index_equal(holdings[name].index, actual.index, check_names=False)
        pd.testing.assert_index_equal(holdings[name].columns, actual.columns)
        np.testing.assert_array_equal(checks[name].history_start, panel.index[positions-COURSE_WINDOW])
        np.testing.assert_array_equal(checks[name].history_end, panel.index[positions-1])
        np.testing.assert_array_equal(checks[name].history_count, np.full(len(index), COURSE_WINDOW))
        np.testing.assert_allclose(holdings[name].sum(axis=1), 1, atol=1e-7, rtol=0)
        np.testing.assert_allclose((holdings[name] * actual).sum(axis=1), daily[name], atol=1e-12, rtol=0)
    rows = {n: {**summarize(daily[n]), **summarize_holdings(holdings[n]), 'evaluation_days': len(index)}
            for n in names}
    return ReportRun(pd.DataFrame.from_dict(rows, orient='index').rename_axis('model'), streams,
                     holdings, checks, pd.DataFrame(forecasts, index=index) if has_lp else None,
                     pd.DataFrame(forecast_info).set_index('date') if has_lp else None)
