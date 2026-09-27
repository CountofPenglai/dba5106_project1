"""One causal daily loop for report portfolios and the forecast/LP pipeline."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

from portfolio.config import TrainingConfig
from portfolio.data.history_windows import TrainingWindow, rolling_windows
from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.decisions.linear_program_allocation import LPConstraints, LPObjective, build_lp_context, solve_linear_portfolio
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.provenance import frame_digest
from portfolio.model_catalog import fit_model
from portfolio.models.lasso_return_forecast import build_supervised, forecast_at
from portfolio.paths import PROJECT_ROOT


@dataclass
class ReportRun:
    summary: pd.DataFrame
    returns: dict[str, pd.DataFrame]
    weights: dict[str, pd.DataFrame]
    diagnostics: dict[str, pd.DataFrame]
    predictions: pd.DataFrame | None
    forecast_diagnostics: pd.DataFrame | None


def load_forecast_settings():
    return json.loads((PROJECT_ROOT/'configs/lasso_forecast.json').read_text())


def verify_forecast_reference(returns, settings, *, start=None):
    """Check historical calibration inputs, not deleted code/result directories."""
    if start is not None and pd.Timestamp(start) <= pd.Timestamp(settings['calibration_end']):
        raise ValueError('LP evaluation must start after forecast calibration')
    if list(returns.columns) != settings['assets']:
        raise ValueError('Forecast reference asset names/order differ')
    reference = returns.loc[settings['history_start']:settings['reference_end']]
    if frame_digest(reference) != settings['reference_data_digest']:
        raise ValueError('Forecast reference data differ; do not reuse the frozen calibration')


def run_models(returns, models, *, start, end, window=252, forecast_alpha=None,
               lp_constraints=LPConstraints(), cost_rate=.0005, progress=None):
    """Refit on previous rows, allocate, then observe the day's realized returns.

    All strategies start each evaluation period from EW. Single-stage turnover
    penalties use previous target weights; LP penalties use drifted pretrade
    weights. Gross returns are primary. Net returns use the same illustrative
    one-way cost rate for all models and never drive model selection here.
    """
    training = TrainingConfig(TrainingWindow(observations=window))
    if not models or len({model.name for model in models}) != len(models):
        raise ValueError('Choose a nonempty list of uniquely named models')
    if not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError('Trading cost rate must be finite and nonnegative')
    used = returns.loc[:end].copy()
    if used.columns.has_duplicates or not np.isfinite(used.to_numpy()).all():
        raise ValueError('Used return history must be complete with unique assets')
    has_lp = any(model.family == 'lp' for model in models)
    if has_lp and forecast_alpha is None:
        raise ValueError('LP requires a frozen forecast alpha')
    supervised = build_supervised(used) if has_lp else None
    notify = progress or (lambda message: None)
    equal = np.full(used.shape[1], 1/used.shape[1])
    targets = {spec.name: equal.copy() for spec in models}
    pretrade = {spec.name: equal.copy() for spec in models}
    weights, daily, diagnostics = ({spec.name: [] for spec in models} for _ in range(3))
    dates, forecasts, forecast_diagnostics = [], [], []
    for number, (day, history) in enumerate(rolling_windows(used, start=start, end=end,
                                                          window=training.window), 1):
        context = build_lp_context(history, lp_constraints) if has_lp else None
        if has_lp:
            forecast, forecast_diag = forecast_at(supervised, day, training, forecast_alpha)
            forecasts.append(forecast)
            forecast_diag['constant_features'] = sum(forecast_diag.pop('constant_feature_counts').values())
            forecast_diagnostics.append(forecast_diag)
        actual = used.loc[day].to_numpy()
        for spec in models:
            name = spec.name
            if spec.family == 'lp':
                w, diag = solve_linear_portfolio(context, forecast.to_numpy(),
                    objective=LPObjective(transaction_cost=spec.transaction_cost), previous_weights=pretrade[name])
            else:
                fitted = fit_model(history, spec, previous_weights=pd.Series(targets[name], index=used.columns))
                w, diag = fitted.weights.to_numpy(), fitted.diagnostics or {}
            w = validate_portfolio_weights(w, len(equal))
            trade = float(.5*np.abs(w-pretrade[name]).sum())
            target_turnover = float(.5*np.abs(w-targets[name]).sum())
            gross = float(w @ actual)
            cost = cost_rate * 2 * trade
            if 1+gross <= 0 or 1+gross-cost <= 0:
                raise ValueError(f'Nonpositive realized wealth for {name} on {day.date()}')
            weights[name].append(w.copy())
            daily[name].append({'return': gross, 'net_return': gross-cost, 'trading_cost': cost})
            diagnostics[name].append({**diag, 'history_start': history.index[0],
                'history_end': history.index[-1], 'history_count': len(history),
                'target_turnover': target_turnover, 'trade_turnover': trade})
            targets[name] = w.copy()
            pretrade[name] = w*(1+actual)/(1+gross)
        dates.append(day)
        if number == 1 or number % 21 == 0:
            notify(f'Evaluated {number} days through {day.date()}')
    index = pd.DatetimeIndex(dates, name='date')
    daily = {name: pd.DataFrame(rows, index=index) for name, rows in daily.items()}
    weights = {name: pd.DataFrame(rows, index=index, columns=used.columns) for name, rows in weights.items()}
    diagnostics = {name: pd.DataFrame(rows, index=index) for name, rows in diagnostics.items()}
    rows = []
    for spec in models:
        name = spec.name
        gross = weights[name].abs().sum(axis=1)
        rows.append({'model': name, **summarize(daily[name]['return']),
            **{'net_'+key: value for key, value in summarize(daily[name]['net_return']).items()},
            'average_gross_leverage': float(gross.mean()), 'max_gross_leverage': float(gross.max()),
            'average_target_turnover': float(diagnostics[name].target_turnover.mean()),
            'average_trade_turnover': float(diagnostics[name].trade_turnover.mean()),
            'evaluation_days': len(index), 'start': str(index[0].date()), 'end': str(index[-1].date())})
    return ReportRun(pd.DataFrame(rows).set_index('model'), daily, weights, diagnostics,
                     pd.DataFrame(forecasts, index=index) if has_lp else None,
                     pd.DataFrame(forecast_diagnostics).set_index('date') if has_lp else None)
