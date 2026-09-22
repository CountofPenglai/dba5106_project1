"""Chronological, separately trained forecast and fixed-mixture experiments."""
from dataclasses import asdict, dataclass
from datetime import date as calendar_date
import json
import re

import numpy as np
import pandas as pd

from portfolio.decisions.candidate_portfolios import candidate_portfolio, check_candidate, estimate_risk
from portfolio.config import TRAINING_PRESETS, TrainingConfig
from portfolio.constraints import PortfolioConstraints
from portfolio.decisions.portfolio_mixture import fit_mixture, mix_weights
from portfolio.evaluation.portfolio_metrics import summarize, portfolio_metrics
from portfolio.evaluation.experiment_result import ExperimentResult
from portfolio.evaluation.provenance import frame_digest
from portfolio.models.lasso_return_forecast import build_supervised, calibrate_alpha, forecast_at
from portfolio.data.history_windows import TrainingWindow, rolling_windows, validate_dates

FINAL_TEST_START, FINAL_TEST_END = '2026-08-01', '2026-08-30'


@dataclass(frozen=True)
class ExperimentConfig:
    history_start: str = '2021-08-01'
    calibration_end: str = '2024-07-31'
    decision_start: str = '2024-08-01'
    decision_end: str = '2025-12-31'
    validation_start: str = '2026-01-02'
    validation_end: str = '2026-06-30'
    alphas: tuple = tuple(np.logspace(-6, -2, 21).tolist())
    n_splits: int = 3
    test_size: int = 21
    gammas: tuple = (0., .025, .05, .1, .2)

    def validate(self):
        dates = [self.history_start, self.calibration_end, self.decision_start,
                 self.decision_end, self.validation_start, self.validation_end]
        for value in dates:
            if not isinstance(value, str) or calendar_date.fromisoformat(value).isoformat() != value:
                raise ValueError('Dates must use complete YYYY-MM-DD format')
        h, c, ds, de, vs, ve = map(pd.Timestamp, dates)
        if not h < c < ds <= de < vs <= ve < pd.Timestamp(FINAL_TEST_START):
            raise ValueError('Require chronological calibration, decision training, validation before reserved final test')
        g = np.asarray(self.gammas, dtype=float)
        if g.ndim != 1 or not len(g) or not np.isfinite(g).all() or (g < 0).any() or len(np.unique(g)) != len(g):
            raise ValueError('Gammas must be unique finite nonnegative values')


@dataclass
class CandidateHistory:
    dates: pd.DatetimeIndex
    asset_returns: pd.DataFrame
    weights: dict
    component_returns: dict
    contexts: list
    frames: dict


def _prepare(returns, start, end):
    validate_dates(returns)
    if returns.columns.has_duplicates or returns.shape[1] < 2:
        raise ValueError('Need at least two uniquely named assets')
    # Weekend period ends mean the preceding weekday, never silently shorten
    # a missing weekday at the end of the reserved August test.
    required_end = pd.bdate_range(end=end, periods=1)[0]
    if returns.empty or returns.index[-1] < required_end:
        raise ValueError(f'Available data do not cover requested end {end}')
    cropped = returns.loc[start:end].copy()
    if cropped.empty or not np.isfinite(cropped.to_numpy(dtype=float)).all():
        raise ValueError('Used history must be finite and complete')
    return cropped


def _forecast_configs(configs):
    result = configs or {key: TRAINING_PRESETS[key] for key in ('252d', '3y')}
    for key, value in result.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+', key) or not isinstance(value, TrainingConfig) or value.half_life is not None:
            raise ValueError('Forecast configs need safe unique names and equal-weight TrainingConfig values')
    return result


def _mean_forecast(data, prediction_date, config):
    history = data.targets.loc[data.targets.index < prediction_date]
    if config.window.observations is not None:
        history = history.iloc[-config.window.observations:]
    else:
        history = history.loc[prediction_date-pd.DateOffset(years=config.window.years):]
    return history.mean().to_numpy()


def generate_candidates(returns, configs, alphas, *, start, end, risk_window, progress=None):
    """Each day's target returns are read only after its portfolios are chosen."""
    notify = progress or (lambda message: None)
    data = build_supervised(returns)
    keys = [f'{kind}_{name}' for name in configs for kind in ('LASSO', 'Mean')]
    weights = {key: [] for key in keys}
    predictions = {key: [] for key in keys}
    component_returns = {key: [] for key in keys}
    forecast_diagnostics = {name: [] for name in configs}
    candidate_diagnostics, risk_rows, contexts, dates = [], [], [], []
    for number, (day, history) in enumerate(rolling_windows(returns, start=start, end=end, window=risk_window), 1):
        context = estimate_risk(history)
        minimum, min_diag = candidate_portfolio(context, objective='risk')
        equal = context.equal_weights
        solved_means = []
        daily_weights = {}
        for name, config in configs.items():
            prediction, diagnostics = forecast_at(data, day, config, alphas[name])
            forecast_diagnostics[name].append(diagnostics)
            for kind, mu in [('LASSO', prediction.to_numpy()), ('Mean', _mean_forecast(data, day, config))]:
                key = f'{kind}_{name}'
                # Identical forecasts (e.g. intercept-only LASSO) must not
                # manufacture different holdings due solely to solver noise.
                cached = next(((w, d) for prior, w, d in solved_means
                               if np.allclose(mu, prior, atol=1e-15, rtol=0)), None)
                if cached is None:
                    maximum, diag = candidate_portfolio(context, objective='return', expected_returns=mu)
                    solved_means.append((mu.copy(), maximum, diag))
                else:
                    maximum, diag = cached
                w = np.stack([maximum, minimum, equal])
                daily_weights[key] = w
                weights[key].append(w)
                predictions[key].append(mu)
                candidate_diagnostics.append(dict(date=day, forecast=key, component='U', reused_forecast=cached is not None, **{k:v for k,v in diag.items() if k != 'expected_return'}))
        # Only now reveal the target day.
        realized = returns.loc[day].to_numpy(dtype=float)
        for key, w in daily_weights.items():
            component_returns[key].append(w @ realized)
        candidate_diagnostics.append(dict(date=day, forecast='shared', component='V', reused_forecast=False, **min_diag))
        risk_rows.append(dict(date=day, history_start=history.index[0], history_end=history.index[-1],
                              observations=len(history), shrinkage=context.shrinkage, variance_limit=context.variance_limit))
        contexts.append(context)
        dates.append(day)
        if number == 1 or number % 21 == 0:
            notify(f'Generated candidates through {day.date()} ({number} days)')
    index = pd.DatetimeIndex(dates, name='date')
    if len(index) < 2:
        raise ValueError('Need at least two evaluation dates')
    frames = {'risk_diagnostics': pd.DataFrame(risk_rows).set_index('date'),
              'candidate_diagnostics': pd.DataFrame(candidate_diagnostics).set_index('date')}
    for name, records in forecast_diagnostics.items():
        d = pd.DataFrame(records).set_index('date')
        d['constant_feature_counts'] = d['constant_feature_counts'].map(lambda x: json.dumps(x, sort_keys=True))
        frames[f'LASSO_{name}_forecast_diagnostics'] = d
    for key in keys:
        weights[key] = np.asarray(weights[key])
        component_returns[key] = pd.DataFrame(component_returns[key], index=index, columns=['U', 'V', 'E'])
        frames[key+'_predictions'] = pd.DataFrame(predictions[key], index=index, columns=returns.columns)
        frames[key+'_components'] = component_returns[key]
        for j, component in enumerate(['U', 'V', 'E']):
            frames[key+'_'+component+'_weights'] = pd.DataFrame(weights[key][:, j], index=index, columns=returns.columns)
    return CandidateHistory(index, returns.loc[index], weights, component_returns, contexts, frames)


def _record_strategy(frames, generated, name, values):
    rows = []
    for context, w in zip(generated.contexts, values):
        rows.append(check_candidate(context, w))
    weights = pd.DataFrame(values, index=generated.dates, columns=generated.asset_returns.columns)
    daily = np.sum(values*generated.asset_returns.to_numpy(), axis=1)
    if np.any(1+daily <= 0):
        raise ValueError('Realized portfolio wealth becomes nonpositive')
    before = np.empty_like(values)
    before[0] = 1/values.shape[1]
    before[1:] = values[:-1]*(1+generated.asset_returns.to_numpy()[:-1])/(1+daily[:-1,None])
    diagnostics = pd.DataFrame(rows, index=generated.dates)
    diagnostics['turnover'] = .5*np.abs(values-before).sum(axis=1)
    frames[name+'_weights'] = weights
    frames[name+'_returns'] = pd.DataFrame({'return': daily}, index=generated.dates)
    frames[name+'_diagnostics'] = diagnostics
    return weights


def _score_period(frames, generated, name, start, end):
    w = frames[name+'_weights'].loc[start:end]
    row = portfolio_metrics(w, generated.asset_returns.loc[w.index])
    row['average_turnover'] = float(frames[name+'_diagnostics'].loc[w.index, 'turnover'].mean())
    return row


def _evaluate_models(generated, models, frames, *, start, end):
    rows = []
    for name, model in models.items():
        key, theta = model['forecast_key'], np.asarray(model['theta'], dtype=float)
        values = mix_weights(generated.weights[key], theta)
        _record_strategy(frames, generated, name, values)
        reconstructed = generated.component_returns[key].to_numpy() @ theta
        np.testing.assert_allclose(frames[name+'_returns']['return'], reconstructed, rtol=0, atol=1e-12)
        rows.append(dict(model=name, role='main' if key.startswith('LASSO_') else 'historical_mean_control',
                         objective=model['objective'], gamma=model['gamma'], a=theta[0], b=theta[1], c=theta[2],
                         **_score_period(frames, generated, name, start, end)))
    return pd.DataFrame(rows).set_index('model')


def _baselines(returns, start, end, progress):
    from portfolio.engine.rolling_backtest import backtest
    from portfolio.losses import LogGrowthLoss
    from portfolio.regularization import Regularization
    from portfolio.engine.portfolio_fit import fit_portfolio, fit_portfolio_growth
    result, rows = {}, []
    for preset in ('252d', '3y'):
        for constrained in (False, True):
            name = 'Baseline_'+preset+('_Constrained' if constrained else '')
            if progress: progress('Evaluating '+name)
            def strategy(history):
                if constrained:
                    return fit_portfolio_growth(history, regularization=Regularization('lasso', 1e-5),
                                                loss=LogGrowthLoss(reward_strength=0, exposure_strength=0),
                                                constraints=PortfolioConstraints()).weights
                return fit_portfolio(history, regularization=Regularization('lasso', 1e-5)).weights
            bt = backtest(returns, strategy, start=start, end=end, window=TRAINING_PRESETS[preset].window)
            result[name+'_weights'], result[name+'_returns'] = bt.weights, bt.returns.to_frame('return')
            rows.append(dict(model=name, role='legacy_baseline', **portfolio_metrics(bt.weights, returns.loc[bt.weights.index])))
    return result, pd.DataFrame(rows).set_index('model')


def run_experiment(returns, *, config=ExperimentConfig(), forecast_configs=None,
                   risk_window=None, progress=None, include_baselines=False):
    config.validate()
    notify = progress or (lambda message: None)
    returns = _prepare(returns, config.history_start, config.validation_end)
    configs = _forecast_configs(forecast_configs)
    risk_window = risk_window or TrainingWindow(years=3, min_observations=252)
    data, frames, alphas = build_supervised(returns), {}, {}
    for name, training in configs.items():
        notify('Calibrating shared forecast alpha: '+name)
        alpha, scores = calibrate_alpha(data, training, end=config.calibration_end, alphas=config.alphas,
                                        n_splits=config.n_splits, test_size=config.test_size)
        alphas[name] = alpha
        frames[name+'_alpha_cv'] = scores
        notify(f'{name}: selected alpha={alpha:.8g}')
    generated = generate_candidates(returns, configs, alphas, start=config.decision_start,
                                    end=config.validation_end, risk_window=risk_window, progress=notify)
    frames.update(generated.frames)
    models, fit_rows, gamma_rows = {}, [], []
    for key, components in generated.component_returns.items():
        train = components.loc[config.decision_start:config.decision_end]
        validation = components.loc[config.validation_start:config.validation_end]
        if min(len(train), len(validation)) < 3:
            raise ValueError('Decision training and validation each need at least three days')
        sharpe = fit_mixture(train, objective='sharpe')
        models[key+'_Sharpe'] = dict(forecast_key=key, objective='sharpe', gamma=0., theta=sharpe.theta.tolist())
        fit_rows.append(dict(forecast=key, objective='sharpe', gamma=0., a=sharpe.theta[0], b=sharpe.theta[1], c=sharpe.theta[2], **{k:v for k,v in sharpe.diagnostics.items() if k not in ('objective','gamma')}))
        fitted = {}
        for gamma in config.gammas:
            fit = fit_mixture(train, objective='return_volatility', gamma=gamma)
            fitted[gamma] = fit
            fit_rows.append(dict(forecast=key, objective='return_volatility', gamma=gamma, a=fit.theta[0], b=fit.theta[1], c=fit.theta[2], **{k:v for k,v in fit.diagnostics.items() if k not in ('objective','gamma')}))
            score = summarize(validation.to_numpy() @ fit.theta)
            gamma_rows.append(dict(forecast=key, gamma=gamma, a=fit.theta[0], b=fit.theta[1], c=fit.theta[2], **score))
        candidates = [row for row in gamma_rows if row['forecast'] == key]
        best = max(row['cumulative_return'] for row in candidates)
        selected = max(row['gamma'] for row in candidates if np.isclose(row['cumulative_return'], best, atol=1e-10, rtol=1e-8))
        for row in candidates: row['selected'] = row['gamma'] == selected
        models[key+'_ReturnVolatility'] = dict(forecast_key=key, objective='return_volatility', gamma=float(selected), theta=fitted[selected].theta.tolist())
        notify(f'{key}: Sharpe theta={np.round(sharpe.theta,4).tolist()}; utility gamma={selected}, theta={np.round(fitted[selected].theta,4).tolist()}')
    frames['decision_fits'] = pd.DataFrame(fit_rows)
    frames['gamma_validation'] = pd.DataFrame(gamma_rows)
    frames['validation_summary'] = _evaluate_models(generated, models, frames, start=config.validation_start, end=config.validation_end)
    frames['decision_training_summary'] = pd.DataFrame([dict(model=name, **_score_period(frames, generated, name, config.decision_start, config.decision_end)) for name in models]).set_index('model')
    component_rows = []
    for key in generated.weights:
        for j, component in enumerate(('U','V','E')):
            name = key+'_'+component
            _record_strategy(frames, generated, name, generated.weights[key][:,j])
            component_rows.append(dict(model=name, role='component', **_score_period(frames, generated, name, config.validation_start, config.validation_end)))
    frames['component_summary'] = pd.DataFrame(component_rows).set_index('model')
    comparison = [frames['validation_summary'], frames['component_summary']]
    if include_baselines:
        extra, table = _baselines(returns, config.validation_start, config.validation_end, notify)
        frames.update(extra); frames['baseline_summary'] = table; comparison.append(table)
    comparison_columns = ['role', 'mean', 'volatility', 'sharpe', 'negative_sharpe', 'cumulative_return', 'max_drawdown', 'average_gross_leverage', 'max_gross_leverage', 'evaluation_days', 'start', 'end']
    frames['comparison'] = pd.concat([table[comparison_columns] for table in comparison])
    predicted_quality = []
    for key in generated.weights:
        pred = frames[key+'_predictions'].loc[config.validation_start:config.validation_end]
        actual = returns.loc[pred.index]
        predicted_quality.append(dict(forecast=key, mse=float(np.mean((pred.to_numpy()-actual.to_numpy())**2)),
                                      zero_forecast_mse=float(np.mean(actual.to_numpy()**2))))
    frames['prediction_validation'] = pd.DataFrame(predicted_quality).set_index('forecast')
    frozen = dict(schema_version=1, config=asdict(config), assets=list(returns.columns),
                  forecast_configs={k:asdict(v) for k,v in configs.items()}, alphas=alphas,
                  risk_window=asdict(risk_window), constraints=asdict(PortfolioConstraints()), models=models,
                  development_data_digest=frame_digest(returns), decision_fit_end=config.decision_end,
                  evaluation_role='development_validation_used_for_selection')
    metadata = dict(phase='development', config=asdict(config), assets=list(returns.columns),
                    alphas=alphas, forecast_configs=frozen['forecast_configs'], risk_window=frozen['risk_window'],
                    decision_training_days=len(generated.asset_returns.loc[config.decision_start:config.decision_end]),
                    validation_days=len(generated.asset_returns.loc[config.validation_start:config.validation_end]),
                    final_test_evaluated=False, development_data_digest=frozen['development_data_digest'],
                    frozen_decision_weights=True, risk_free_rate=0, transaction_costs=0,
                    metric_convention='daily Sharpe, sample std ddof=1; compounded realized returns; turnover relative to drifted holdings',
                    training_returns_role='in-sample for decision mixture despite out-of-sample component generation',
                    evaluation_role='development_validation_used_for_selection')
    return ExperimentResult(frames, metadata, frozen)


def run_frozen(returns, frozen, *, start=FINAL_TEST_START, end=FINAL_TEST_END, progress=None):
    """Explicit reserved-test path: no alpha, gamma or mixture fitting."""
    if (start, end) != (FINAL_TEST_START, FINAL_TEST_END):
        raise ValueError('Final test must cover the complete reserved August period')
    if frozen.get('schema_version') != 1:
        raise ValueError('Unsupported frozen decision schema')
    config = ExperimentConfig(**frozen['config']); config.validate()
    returns = _prepare(returns, config.history_start, end)
    if list(returns.columns) != frozen['assets']:
        raise ValueError('Frozen model assets do not match input data')
    if frame_digest(returns.loc[:config.validation_end]) != frozen['development_data_digest']:
        raise ValueError('Development history changed since decisions were frozen')
    if frozen.get('constraints') != asdict(PortfolioConstraints()):
        raise ValueError('Frozen constraints are incompatible with this implementation')
    configs = {key:TrainingConfig(TrainingWindow(**value['window']), value['half_life'])
               for key,value in frozen['forecast_configs'].items()}
    configs = _forecast_configs(configs)
    # Calendar bounds include weekends; execution uses all actually observed
    # market dates, while _prepare has already verified final weekday coverage.
    dates = returns.loc[start:end].index
    # The reserved August 2026 interval contains 20 US trading weekdays
    # (no scheduled exchange holiday). Require every one, not merely its end.
    expected_dates = pd.bdate_range(start, end)
    if not dates.equals(expected_dates):
        raise ValueError('Final test coverage must contain all 20 trading dates from August 3 to 28')
    if len(dates) < 2:
        raise ValueError('Final test data are unavailable')
    generated = generate_candidates(returns, configs, frozen['alphas'], start=dates[0], end=dates[-1],
                                    risk_window=TrainingWindow(**frozen['risk_window']), progress=progress)
    frames = generated.frames.copy()
    frames['test_summary'] = _evaluate_models(generated, frozen['models'], frames, start=start, end=end)
    metadata = dict(phase='final_test', final_test_evaluated=True, evaluation_days=len(dates),
                    first_date=str(dates[0].date()), last_date=str(dates[-1].date()),
                    frozen_decision_weights=True, alpha_refitted=False, gamma_reselected=False,
                    development_data_digest=frozen['development_data_digest'], assets=frozen['assets'])
    return ExperimentResult(frames, metadata, frozen)
