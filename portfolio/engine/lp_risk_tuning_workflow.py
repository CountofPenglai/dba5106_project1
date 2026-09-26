"""Select one LP MAD penalty on past development data, then freeze for validation."""

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from portfolio.config import TrainingConfig
from portfolio.data.history_windows import TrainingWindow, rolling_windows
from portfolio.decisions.linear_program_allocation import (
    LPConstraints, LPObjective, build_lp_context, solve_linear_portfolio,
)
from portfolio.engine.forecast_lp_workflow import _prepare
from portfolio.evaluation.experiment_result import ExperimentResult
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.provenance import frame_digest
from portfolio.models.lasso_return_forecast import build_supervised, forecast_at


@dataclass(frozen=True)
class LPRiskTuningConfig:
    history_start: str = '2021-08-01'
    calibration_end: str = '2024-07-31'
    tuning_start: str = '2025-07-01'
    tuning_end: str = '2025-12-31'
    validation_start: str = '2026-01-02'
    validation_end: str = '2026-06-30'
    window: int = 252
    gammas: tuple = (0., .05, .1, .25, .5, 1.)
    transaction_cost: float = .0005

    def validate(self):
        fields = [self.history_start, self.calibration_end, self.tuning_start,
                  self.tuning_end, self.validation_start, self.validation_end]
        for value in fields:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError('Dates must use complete YYYY-MM-DD format')
        h, c, ts, te, vs, ve = map(pd.Timestamp, fields)
        if not h < c < ts < te < vs < ve < pd.Timestamp('2026-07-01'):
            raise ValueError('Require calibration < tuning < validation; July test is excluded')
        TrainingWindow(observations=self.window)
        if not self.gammas or len(set(self.gammas)) != len(self.gammas):
            raise ValueError('Gamma candidates must be nonempty and unique')
        for gamma in self.gammas:
            LPObjective(mad_penalty=gamma, transaction_cost=self.transaction_cost)


def _forecasts(returns, config, alpha, start, end, notify):
    """Fit once per date; share exactly the same predictions across all candidates."""
    data = build_supervised(returns.loc[:end])
    training = TrainingConfig(TrainingWindow(observations=config.window))
    predictions, diagnostics = [], []
    dates = returns.loc[start:end].index
    for number, day in enumerate(dates, 1):
        predicted, diag = forecast_at(data, day, training, alpha)
        diag['constant_features'] = sum(diag.pop('constant_feature_counts').values())
        predictions.append(predicted)
        diagnostics.append(diag)
        if number == 1 or number % 21 == 0:
            notify(f'Forecasts through {day.date()} ({number}/{len(dates)})')
    return pd.DataFrame(predictions), pd.DataFrame(diagnostics).set_index('date')


def _rollout(returns, predictions, objectives, config, constraints, notify):
    """All strategies start from EW; each then carries its own drifted holdings.

    Return accounting uses the conventional first-order deduction
    r_net = r_gross - cost_rate * L1(target - drifted_previous).
    Holdings drift on gross returns; this does not model fee-financed trades,
    market impact, short borrow fees or funding costs.
    """
    names = list(objectives)
    previous = {name: np.full(returns.shape[1], 1/returns.shape[1]) for name in names}
    weights, daily, diagnostics = ({name: [] for name in names} for _ in range(3))
    start, end = predictions.index[[0, -1]]
    for number, (day, history) in enumerate(rolling_windows(
            returns, start=start, end=end, window=config.window), 1):
        contexts = {}
        mu = predictions.loc[day].to_numpy()
        actual = returns.loc[day].to_numpy()
        for name, objective in objectives.items():
            limits = constraints[name] if isinstance(constraints, dict) else constraints
            if limits not in contexts:
                contexts[limits] = build_lp_context(history, limits)
            context = contexts[limits]
            if objective is None:
                w, diag = context.equal_weights.copy(), {}
            else:
                w, diag = solve_linear_portfolio(context, mu, objective=objective,
                                                 previous_weights=previous[name])
            traded = float(np.abs(w-previous[name]).sum())
            cost = config.transaction_cost * traded
            gross = float(w @ actual)
            net = gross - cost
            if min(1+gross, 1+net) <= 0:
                raise ValueError(f'Nonpositive realized wealth: {name} on {day.date()}')
            weights[name].append(w)
            daily[name].append(dict(gross_return=gross, cost=cost, return_value=net))
            diagnostics[name].append(dict(date=day, **diag,
                history_start=history.index[0], history_end=history.index[-1],
                history_count=len(history), turnover=.5*traded, charged_cost=cost))
            previous[name] = w*(1+actual)/(1+gross)
        if number == 1 or number % 21 == 0:
            notify(f'LP decisions through {day.date()} ({number}/{len(predictions)})')
    frames = {}
    for name in names:
        frames[name+'_weights'] = pd.DataFrame(weights[name], index=predictions.index, columns=returns.columns)
        frames[name+'_returns'] = pd.DataFrame(daily[name], index=predictions.index).rename(columns={'return_value': 'return'})
        frames[name+'_diagnostics'] = pd.DataFrame(diagnostics[name]).set_index('date')
    return frames


def _summary(frames, names):
    rows = []
    for name in names:
        daily = frames[name+'_returns']
        w = frames[name+'_weights']
        diag = frames[name+'_diagnostics']
        net, gross = summarize(daily['return']), summarize(daily['gross_return'])
        if not np.isfinite([*net.values(), *gross.values()]).all():
            raise ValueError('Performance is nonfinite or Sharpe is undefined')
        rows.append(dict(model=name, **net, **{'gross_'+k: v for k, v in gross.items()},
            average_turnover=float(diag.turnover.mean()),
            average_gross_leverage=float(w.abs().sum(axis=1).mean()),
            max_gross_leverage=float(w.abs().sum(axis=1).max()),
            mean_daily_cost=float(daily.cost.mean()), evaluation_days=len(daily),
            start=str(daily.index[0].date()), end=str(daily.index[-1].date())))
    return pd.DataFrame(rows).set_index('model')


def run_lp_risk_tuning(returns, *, forecast_alpha, config=LPRiskTuningConfig(),
                       constraints=LPConstraints(), progress=None):
    """Tune gamma on training history only; evaluate one selected gamma and controls.

    The LASSO alpha must have been fixed by calibration_end. It is supplied by
    the caller, never selected using tuning/validation outcomes in this workflow.
    Validation is walk-forward: refits may use already observed validation days.
    """
    config.validate()
    notify = progress or (lambda message: None)
    used = _prepare(returns, config.history_start, config.validation_end)
    training = used.loc[:config.tuning_end]
    predictions, forecast_diag = _forecasts(training, config, forecast_alpha,
        config.tuning_start, config.tuning_end, notify)
    objectives = {f'gamma_{i}': LPObjective(gamma, config.transaction_cost)
                  for i, gamma in enumerate(config.gammas)}
    tuning = _rollout(training, predictions, objectives, config, constraints, notify)
    search = _summary(tuning, objectives)
    search.insert(0, 'mad_penalty', list(config.gammas))
    # Course metric on net returns; ties favor smaller drawdown, then more shrinkage.
    tied = search[np.isclose(search.sharpe, search.sharpe.max(), atol=1e-8, rtol=1e-6)]
    selected = tied.sort_values(['max_drawdown', 'mad_penalty'], ascending=False).iloc[0]
    gamma = float(selected.mad_penalty)
    search['selected'] = search.index == selected.name
    frozen = dict(schema_version=1, kind='mad_cost_direct_lp', mad_penalty=gamma,
        forecast_alpha=float(forecast_alpha), config=asdict(config),
        constraints=asdict(constraints), assets=list(used.columns),
        selection_rule='maximum training net daily Sharpe; ties: smaller drawdown, larger gamma',
        tuning_data_digest=frame_digest(training), selection_end=config.tuning_end)
    notify(f'Frozen MAD penalty: {gamma:g}; training net daily Sharpe: {selected.sharpe:.6f}')

    # No validation candidate sweep: only the frozen choice and predeclared controls.
    validation_predictions, validation_diag = _forecasts(used, config, forecast_alpha,
        config.validation_start, config.validation_end, notify)
    controls = dict(LP_MAD_Selected=LPObjective(gamma, config.transaction_cost),
                    LP_Cost_Only=LPObjective(0., config.transaction_cost),
                    LP_Return_Only=LPObjective(), EW=None)
    frames = _rollout(used, validation_predictions, controls, config, constraints, notify)
    frames.update({'tuning_'+name: frame for name, frame in tuning.items()})
    frames['gamma_search'] = search
    frames['training_predictions'] = predictions
    frames['validation_predictions'] = validation_predictions
    frames['forecast_diagnostics'] = pd.concat([forecast_diag, validation_diag])
    frames['validation_summary'] = _summary(frames, controls)
    metadata = dict(phase='training_tuning_then_validation', config=asdict(config),
        forecast_alpha=float(forecast_alpha), alpha_reselected=False,
        selected_mad_penalty=gamma, lp_constraints=asdict(constraints),
        selection_rule=frozen['selection_rule'], validation_candidate_count=1,
        final_test_evaluated=False, data_end=config.validation_end,
        tuning_days=len(predictions), validation_days=len(validation_predictions),
        assets=list(used.columns), used_data_digest=frame_digest(used),
        risk_free_rate=0., decision_model_trained=False, decision_parameter_search=True,
        transaction_cost_rate=config.transaction_cost,
        cost_convention='one-way rate times L1(target minus drifted holdings); first-order net=gross-cost',
        holdings_initialization='EW reset independently at tuning and validation starts',
        validation_protocol='walk-forward refits using prior observations; alpha/gamma frozen',
        risk_measure='historical portfolio mean absolute deviation; not standard deviation',
        drawdown_constraint='historical uncompounded cumulative-return drawdown; no future guarantee',
        turnover_budget=False, evaluation_role='reused development validation, not an untouched test')
    return ExperimentResult(frames, metadata, frozen)
