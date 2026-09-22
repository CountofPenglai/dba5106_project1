"""Causal LASSO forecasts followed by daily direct-asset linear programming."""
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
import re

import numpy as np
import pandas as pd

from portfolio.config import TRAINING_PRESETS, TrainingConfig
from portfolio.models.lasso_return_forecast import build_supervised, calibrate_alpha, forecast_at
from portfolio.decisions.linear_program_allocation import LPConstraints, build_lp_context, solve_linear_portfolio
from portfolio.data.history_windows import TrainingWindow, rolling_windows, validate_dates
from portfolio.evaluation.experiment_result import ExperimentResult
from portfolio.evaluation.provenance import frame_digest, code_hashes
from portfolio.evaluation.portfolio_metrics import portfolio_metrics
from portfolio.evaluation.result_export import file_digest


def _solver_versions():
    return {name:version(name) for name in ('numpy','pandas','scipy','scikit-learn')}


@dataclass(frozen=True)
class DirectLPConfig:
    history_start: str = '2021-08-01'
    calibration_end: str = '2024-07-31'
    development_start: str = '2024-08-01'
    development_end: str = '2025-12-31'
    validation_start: str = '2026-01-02'
    validation_end: str = '2026-06-30'
    alphas: tuple = tuple(np.logspace(-6, -2, 21).tolist())
    n_splits: int = 3
    test_size: int = 21

    def validate(self):
        fields = [self.history_start, self.calibration_end, self.development_start,
                  self.development_end, self.validation_start, self.validation_end]
        for value in fields:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError('Dates must use complete YYYY-MM-DD format')
        h, c, ds, de, vs, ve = map(pd.Timestamp, fields)
        if not h < c < ds <= de < vs <= ve < pd.Timestamp('2026-08-01'):
            raise ValueError('Require chronological calibration/development/validation before August')


def _prepare(returns, start, end):
    validate_dates(returns)
    if returns.empty or returns.columns.has_duplicates or returns.shape[1] < 2:
        raise ValueError('Need complete history with at least two unique assets')
    if returns.index[0] > pd.bdate_range(start=start, periods=1)[0]:
        raise ValueError('History does not cover the configured start')
    if returns.index[-1] < pd.bdate_range(end=end, periods=1)[0]:
        raise ValueError('Data do not cover the requested end')
    used = returns.loc[start:end].copy()
    if used.empty or not np.isfinite(used.to_numpy(dtype=float)).all():
        raise ValueError('Used history must be finite and complete')
    return used


def _windows(values):
    configs = values if values is not None else {name: TRAINING_PRESETS[name] for name in ('252d', '3y')}
    if not configs:
        raise ValueError('Need at least one forecast configuration')
    for name, config in configs.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name)
                or not isinstance(config, TrainingConfig) or config.half_life is not None):
            raise ValueError('Forecast windows need safe names and equal observation weights')
    return configs


def _generate(returns, configs, alphas, constraints, risk_window, start, end, notify):
    data = build_supervised(returns)
    names = ['LP_'+kind+'_'+name for name in configs for kind in ('LASSO', 'Mean')]
    weights = {key: [] for key in names+['EW']}
    diagnostics = {key: [] for key in names+['EW']}
    predictions = {key: [] for key in names}
    forecast_rows = {name: [] for name in configs}
    dates = []
    for number, (day, history) in enumerate(rolling_windows(returns, start=start, end=end, window=risk_window), 1):
        context = build_lp_context(history, constraints)
        shared = dict(date=day, history_start=history.index[0], history_end=history.index[-1],
                      history_count=len(history), mad_limit=context.mad_limit,
                      drawdown_limit=context.drawdown_limit)
        cached = []
        for name, training in configs.items():
            predicted, forecast_diag = forecast_at(data, day, training, alphas[name])
            forecast_diag = dict(forecast_diag)
            # Per-asset constant-feature counts are already available in the
            # forecasting API; aggregate here to keep scalar artifact columns.
            forecast_diag['constant_features'] = sum(forecast_diag.pop('constant_feature_counts').values())
            forecast_rows[name].append(forecast_diag)
            mean = data.targets.loc[forecast_diag['train_start']:forecast_diag['train_end']].mean().to_numpy()
            for kind, mu in [('LASSO', predicted.to_numpy()), ('Mean', mean)]:
                key = 'LP_'+kind+'_'+name
                reuse = next(((w, d) for prior, w, d in cached
                              if np.allclose(mu, prior, atol=1e-15, rtol=0)), None)
                if reuse is None:
                    try:
                        w, diag = solve_linear_portfolio(context, mu)
                    except RuntimeError as error:
                        raise RuntimeError(f'{key} on {day.date()}: {error}') from error
                    cached.append((mu.copy(), w, diag))
                else:
                    w, diag = reuse
                weights[key].append(w.copy())
                predictions[key].append(mu.copy())
                diagnostics[key].append({**shared, **diag, 'reused_forecast': reuse is not None})
        weights['EW'].append(context.equal_weights.copy())
        # EW is only a performance reference when a deliberately tighter custom
        # limit excludes it. Do not silently use it as an LP recovery portfolio.
        diagnostics['EW'].append(shared)
        dates.append(day)
        if number == 1 or number % 21 == 0:
            notify(f'Solved direct LP portfolios through {day.date()} ({number} days)')
    index = pd.DatetimeIndex(dates, name='date')
    if len(index) < 3:
        raise ValueError('Need at least three evaluation dates')
    frames = {}
    actual = returns.loc[index].to_numpy()
    for key, rows in weights.items():
        w = np.asarray(rows)
        realized = np.sum(w*actual, axis=1)
        if np.any(1+realized <= 0):
            raise ValueError('Realized portfolio wealth becomes nonpositive')
        previous = np.empty_like(w)
        previous[0] = 1/w.shape[1]
        previous[1:] = w[:-1]*(1+actual[:-1])/(1+realized[:-1,None])
        diag = pd.DataFrame(diagnostics[key]).set_index('date')
        diag['turnover'] = .5*np.abs(w-previous).sum(axis=1)
        frames[key+'_weights'] = pd.DataFrame(w, index=index, columns=returns.columns)
        frames[key+'_returns'] = pd.DataFrame({'return': realized}, index=index)
        frames[key+'_diagnostics'] = diag
        if key in predictions:
            frames[key+'_predictions'] = pd.DataFrame(predictions[key], index=index, columns=returns.columns)
    for name, rows in forecast_rows.items():
        frames[name+'_forecast_diagnostics'] = pd.DataFrame(rows).set_index('date')
    return frames, names+['EW']


def _summary(frames, names, returns, start, end):
    rows = []
    for name in names:
        w = frames[name+'_weights'].loc[start:end]
        if len(w) < 3:
            raise ValueError('Every evaluation period needs at least three observations')
        role = 'equal_weight' if name == 'EW' else ('main' if name.startswith('LP_LASSO') else 'historical_mean_control')
        rows.append(dict(model=name, role=role, **portfolio_metrics(w, returns.loc[w.index]),
                         average_turnover=float(frames[name+'_diagnostics'].loc[w.index,'turnover'].mean()),
                         max_absolute_position=float(w.abs().to_numpy().max())))
    return pd.DataFrame(rows).set_index('model')


def _metadata(config, constraints, risk_window, configs, alphas):
    return dict(config=asdict(config), lp_constraints=asdict(constraints),
                risk_window=asdict(risk_window), forecast_configs={k:asdict(v) for k,v in configs.items()},
                alphas=alphas, decision_model_trained=False, decision_parameter_search=False,
                solver='scipy.optimize.linprog/HiGHS', scipy_version=version('scipy'),
                risk_free_rate=0., transaction_costs=0.,
                risk_constraint_measure='mean absolute deviation of centered historical simple returns',
                drawdown_constraint_measure='uncompounded absolute cumulative-return drawdown from initial zero',
                evaluation_measures='daily Sharpe and sample std ddof=1; compounded return and relative max drawdown; drifted turnover',
                standard_deviation_constrained=False, sharpe_constrained=False,
                compounded_drawdown_constrained=False)


def run_direct_lp(returns, *, config=DirectLPConfig(), constraints=LPConstraints(),
                  forecast_configs=None, risk_window=None, include_development=False, progress=None):
    config.validate()
    notify = progress or (lambda message: None)
    used = _prepare(returns, config.history_start, config.validation_end)
    configs = _windows(forecast_configs)
    risk_window = risk_window or TrainingWindow(years=3, min_observations=252)
    data = build_supervised(used)
    frames, alphas = {}, {}
    for name, training in configs.items():
        notify('Calibrating forecast alpha: '+name)
        alpha, cv = calibrate_alpha(data, training, end=config.calibration_end,
                                    alphas=config.alphas, n_splits=config.n_splits, test_size=config.test_size)
        alphas[name] = alpha
        frames[name+'_alpha_cv'] = cv
        notify(f'{name}: alpha={alpha:.9g}')
    evaluation_start = config.development_start if include_development else config.validation_start
    generated, names = _generate(used, configs, alphas, constraints, risk_window,
                                 evaluation_start, config.validation_end, notify)
    frames.update(generated)
    if include_development:
        frames['development_summary'] = _summary(frames, names, used, config.development_start, config.development_end)
    frames['validation_summary'] = _summary(frames, names, used, config.validation_start, config.validation_end)
    frames['comparison'] = frames['validation_summary'].copy()
    quality = []
    for name in names:
        if name == 'EW': continue
        pred = frames[name+'_predictions'].loc[config.validation_start:config.validation_end]
        actual = used.loc[pred.index].to_numpy()
        quality.append(dict(model=name, mse=float(np.mean((pred.to_numpy()-actual)**2)),
                            zero_forecast_mse=float(np.mean(actual**2))))
    frames['prediction_validation'] = pd.DataFrame(quality).set_index('model')
    metadata = _metadata(config, constraints, risk_window, configs, alphas)
    metadata.update(phase='development', final_test_evaluated=False, assets=list(used.columns),
                    development_data_digest=frame_digest(used),
                    development_backtest_included=bool(include_development), evaluation_start=evaluation_start,
                    development_days=len(used.loc[config.development_start:config.development_end]) if include_development else 0,
                    validation_days=len(used.loc[config.validation_start:config.validation_end]),
                    evaluation_role='development_validation; no LP parameter search; not an untouched final test')
    frozen = dict(schema_version=1, kind='direct_asset_lp', config=asdict(config),
                  assets=list(used.columns), forecast_configs=metadata['forecast_configs'], alphas=alphas,
                  lp_constraints=asdict(constraints), risk_window=asdict(risk_window),
                  development_data_digest=metadata['development_data_digest'],
                  code_sha256=code_hashes(), solver_versions=_solver_versions())
    return ExperimentResult(frames, metadata, frozen)


def run_frozen_lp(returns, frozen, *, progress=None):
    if frozen.get('schema_version') != 1 or frozen.get('kind') != 'direct_asset_lp':
        raise ValueError('Unsupported direct-LP frozen schema')
    config = DirectLPConfig(**frozen['config']); config.validate()
    if frozen.get('code_sha256') != code_hashes():
        raise ValueError('Frozen source code differs from current implementation; reproduce this artifact with its original repository revision')
    if frozen.get('solver_versions') != _solver_versions():
        raise ValueError('Frozen numerical package versions differ')
    used = _prepare(returns, config.history_start, '2026-08-30')
    if list(used.columns) != frozen['assets']:
        raise ValueError('Frozen asset order differs from input')
    if frame_digest(used.loc[:config.validation_end]) != frozen['development_data_digest']:
        raise ValueError('Development history changed after freezing')
    dates = used.loc['2026-08-01':'2026-08-30'].index
    if not dates.equals(pd.bdate_range('2026-08-01','2026-08-30')):
        raise ValueError('Final test requires all 20 August 3–28 trading dates')
    configs = _windows({k:TrainingConfig(TrainingWindow(**v['window']), v['half_life'])
                        for k,v in frozen['forecast_configs'].items()})
    if set(configs) != set(frozen['alphas']):
        raise ValueError('Frozen forecast windows and alphas differ')
    constraints = LPConstraints(**frozen['lp_constraints'])
    risk_window = TrainingWindow(**frozen['risk_window'])
    frames, names = _generate(used, configs, frozen['alphas'], constraints, risk_window,
                              dates[0], dates[-1], progress or (lambda message: None))
    frames['test_summary'] = _summary(frames, names, used, dates[0], dates[-1])
    metadata = _metadata(config, constraints, risk_window, configs, frozen['alphas'])
    metadata.update(phase='final_test', final_test_evaluated=True, alpha_reselected=False,
                    evaluation_days=len(dates), assets=frozen['assets'],
                    evaluation_role='frozen_final_test', first_date=str(dates[0].date()),
                    last_date=str(dates[-1].date()), development_data_digest=frozen['development_data_digest'])
    return ExperimentResult(frames, metadata, frozen)
