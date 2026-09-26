"""Compare the existing cost-aware LP with lower leverage and an EW anchor."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from portfolio.decisions.linear_program_allocation import (
    LPConstraints, LPObjective, build_lp_context, solve_linear_portfolio,
)
from portfolio.engine.forecast_lp_workflow import _prepare
from portfolio.engine.lp_risk_tuning_workflow import LPRiskTuningConfig, _forecasts, _rollout, _summary
from portfolio.evaluation.experiment_result import ExperimentResult
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.provenance import frame_digest


@dataclass(frozen=True)
class LPStabilityConfig:
    periods: LPRiskTuningConfig = LPRiskTuningConfig(tuning_start='2025-01-02', gammas=(0.,))
    lower_leverage: float = 1.2
    ew_penalties: tuple = (.0001, .00025, .0005)
    return_retention: float = .8
    blocks: int = 3
    stressed_cost: float = .001
    perturbation_size: float = .0001
    perturbations: int = 5
    random_seed: int = 20260924

    def validate(self):
        self.periods.validate()
        if self.periods.gammas != (0.,):
            raise ValueError('MAD penalty must remain fixed at zero for this comparison')
        LPConstraints(leverage_limit=self.lower_leverage)
        if not self.ew_penalties or len(set(self.ew_penalties)) != len(self.ew_penalties):
            raise ValueError('EW candidates must be nonempty and unique')
        for tau in self.ew_penalties:
            if LPObjective(ew_penalty=tau).ew_penalty <= 0:
                raise ValueError('EW candidates must be positive; zero is the current-LP control')
        if not np.isfinite(self.return_retention) or not 0 < self.return_retention <= 1:
            raise ValueError('Return retention must be in (0, 1]')
        for name in ('blocks', 'perturbations'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name+' must be a positive integer')
        for value in (self.stressed_cost, self.perturbation_size):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Stress magnitudes must be finite and positive')


def block_summary(frames, names, blocks):
    """Partition an existing continuous path; never reset holdings at block boundaries."""
    rows = []
    for name in names:
        daily = frames[name+'_returns']
        if len(daily) < 3*blocks:
            raise ValueError('Need at least three observations per evaluation block')
        for number, positions in enumerate(np.array_split(np.arange(len(daily)), blocks), 1):
            subset = daily.iloc[positions]
            rows.append(dict(model=name, block=number, start=str(subset.index[0].date()),
                end=str(subset.index[-1].date()), observations=len(subset),
                **summarize(subset['return'])))
    return pd.DataFrame(rows).set_index(['model', 'block'])


def select_ew_penalty(scores, *, return_retention):
    """Prefer lower worst-block drawdown, subject to explicit training guardrails.

    If no positive penalty qualifies, retain the current LP and mark the fallback.
    For a negative baseline return, the floor allows the same proportional loss
    deterioration, rather than accidentally demanding a profit improvement.
    """
    search = scores.copy()
    base = search.loc['LP_Current']
    floor = base.cumulative_return - (1-return_retention)*abs(base.cumulative_return)
    search['return_floor'] = floor
    search['eligible'] = ((search.ew_penalty > 0)
        & (search.cumulative_return >= floor-1e-12)
        & (search.worst_block_volatility <= base.worst_block_volatility+1e-12)
        & (search.worst_block_drawdown <= base.worst_block_drawdown+1e-12))
    qualified = search.loc[search.eligible]
    selected = ('LP_Current' if qualified.empty else qualified.sort_values(
        ['worst_block_drawdown', 'sharpe', 'ew_penalty'],
        ascending=[True, False, True]).index[0])
    search['selected'] = search.index == selected
    return selected, search


def prediction_stress(returns, predictions, frames, objectives, limits, config, notify):
    """One-step sensitivity probes with fixed pretrade holdings, not a noisy backtest.

    Probe the first validation date of each month with common uniform +/-1 bp
    forecast shocks. No outcomes or probe scores enter parameter selection.
    """
    rng = np.random.default_rng(config.random_seed)
    sample_dates = predictions.index[~predictions.index.to_period('M').duplicated()]
    rows = []
    for day in sample_dates:
        history = returns.loc[returns.index < day].tail(config.periods.window)
        shocks = rng.uniform(-config.perturbation_size, config.perturbation_size,
                             (config.perturbations, returns.shape[1]))
        contexts = {limit: build_lp_context(history, limit) for limit in set(limits.values())}
        for name, objective in objectives.items():
            if objective is None:
                continue
            weights = frames[name+'_weights']
            position = weights.index.get_loc(day)
            if position == 0:
                previous = np.full(returns.shape[1], 1/returns.shape[1])
            else:
                prior_day = weights.index[position-1]
                prior_w = weights.iloc[position-1].to_numpy()
                prior_r = returns.loc[prior_day].to_numpy()
                previous = prior_w*(1+prior_r)/(1+prior_w@prior_r)
            base = weights.loc[day].to_numpy()
            for sample, shock in enumerate(shocks, 1):
                w, diag = solve_linear_portfolio(contexts[limits[name]],
                    predictions.loc[day].to_numpy()+shock, objective=objective,
                    previous_weights=previous)
                rows.append(dict(date=day, model=name, sample=sample,
                    holdings_change=float(.5*np.abs(w-base).sum()),
                    max_asset_change=float(np.abs(w-base).max()), feasible=diag['feasible']))
        notify(f'Prediction sensitivity checked: {day.date()}')
    return pd.DataFrame(rows).set_index('date')


def run_lp_stability(returns, *, forecast_alpha, config=LPStabilityConfig(),
                     constraints=LPConstraints(), progress=None):
    config.validate()
    if config.lower_leverage >= constraints.leverage_limit:
        raise ValueError('Lower-leverage control must have a smaller gross limit')
    notify = progress or (lambda message: None)
    periods = config.periods
    used = _prepare(returns, periods.history_start, periods.validation_end)
    training = used.loc[:periods.tuning_end]
    predictions, forecast_diag = _forecasts(training, periods, forecast_alpha,
        periods.tuning_start, periods.tuning_end, notify)
    cost_objective = LPObjective(transaction_cost=periods.transaction_cost)
    low_limit = replace(constraints, leverage_limit=config.lower_leverage)
    objectives = dict(LP_Current=cost_objective, LP_Lower_Leverage=cost_objective, EW=None)
    objectives.update({f'EW_Candidate_{i}': replace(cost_objective, ew_penalty=tau)
                       for i, tau in enumerate(config.ew_penalties)})
    limits = {name: low_limit if name == 'LP_Lower_Leverage' else constraints for name in objectives}
    training_frames = _rollout(training, predictions, objectives, periods, limits, notify)
    training_summary = _summary(training_frames, objectives)
    blocks = block_summary(training_frames, objectives, config.blocks)
    training_summary['worst_block_volatility'] = blocks.volatility.groupby('model').max()
    training_summary['worst_block_drawdown'] = -blocks.max_drawdown.groupby('model').min()
    training_summary['ew_penalty'] = [obj.ew_penalty if obj else 0. for obj in objectives.values()]
    selected, search = select_ew_penalty(training_summary, return_retention=config.return_retention)
    chosen = objectives[selected]
    frozen = dict(schema_version=1, kind='lp_stability_comparison', config=asdict(config),
        forecast_alpha=float(forecast_alpha), ew_penalty=chosen.ew_penalty,
        ew_candidate_qualified=bool(search.eligible.any()), training_choice=selected,
        base_constraints=asdict(constraints), lower_constraints=asdict(low_limit),
        assets=list(used.columns), selection_end=periods.tuning_end,
        tuning_data_digest=frame_digest(training))
    notify(f'Frozen EW penalty: {chosen.ew_penalty:g}; qualified={frozen["ew_candidate_qualified"]}')

    validation_predictions, validation_diag = _forecasts(used, periods, forecast_alpha,
        periods.validation_start, periods.validation_end, notify)
    validation_objectives = dict(LP_Current=cost_objective, LP_Lower_Leverage=cost_objective,
                                LP_EW_Anchored=chosen, EW=None)
    validation_limits = {name: low_limit if name == 'LP_Lower_Leverage' else constraints
                         for name in validation_objectives}
    frames = _rollout(used, validation_predictions, validation_objectives, periods, validation_limits, notify)
    frames['validation_summary'] = _summary(frames, validation_objectives)
    frames['validation_block_summary'] = block_summary(frames, validation_objectives, config.blocks)
    fee_rows = []
    for name in validation_objectives:
        daily, diag = frames[name+'_returns'], frames[name+'_diagnostics']
        fee_rows.append(dict(model=name, cost_rate=config.stressed_cost,
            **summarize(daily.gross_return-2*config.stressed_cost*diag.turnover)))
        w = frames[name+'_weights']
        frames['validation_summary'].loc[name, 'average_ew_distance'] = float(
            np.abs(w.to_numpy()-1/w.shape[1]).sum(axis=1).mean())
    frames['fee_stress_summary'] = pd.DataFrame(fee_rows).set_index('model')
    probes = prediction_stress(used, validation_predictions, frames, validation_objectives,
                               validation_limits, config, notify)
    frames['prediction_stress_diagnostics'] = probes
    frames['prediction_stress_summary'] = probes.groupby('model').agg(
        mean_holdings_change=('holdings_change', 'mean'),
        median_holdings_change=('holdings_change', 'median'),
        max_holdings_change=('holdings_change', 'max'), probes=('holdings_change', 'size'))
    frames.update({'tuning_'+name: frame for name, frame in training_frames.items()})
    frames['training_summary'] = training_summary
    frames['training_block_summary'] = blocks
    frames['ew_penalty_search'] = search
    frames['training_predictions'] = predictions
    frames['validation_predictions'] = validation_predictions
    frames['forecast_diagnostics'] = pd.concat([forecast_diag, validation_diag])
    metadata = dict(config=asdict(config), phase='training_selection_then_validation',
        final_test_evaluated=False, validation_candidate_count=1, alpha_reselected=False,
        forecast_alpha=float(forecast_alpha), selected_ew_penalty=chosen.ew_penalty,
        ew_candidate_qualified=frozen['ew_candidate_qualified'],
        base_constraints=asdict(constraints), lower_constraints=asdict(low_limit),
        used_data_digest=frame_digest(used), data_end=periods.validation_end,
        assets=list(used.columns), training_days=len(predictions), validation_days=len(validation_predictions),
        selection_rule='minimize worst-block drawdown subject to return retention and no worse worst-block volatility/drawdown; ties: Sharpe, smaller tau; fallback: current LP',
        cost_convention='one-way rate times full L1 trades; first-order net=gross-cost; gross holdings drift',
        fee_stress='fixed validation holdings; higher execution costs, no reoptimization',
        prediction_stress='common uniform shocks on first monthly validation dates; fixed original pretrade holdings; no reselection',
        blocks='equal-count chronological blocks; holdings continuous, drawdown resets for block metrics only',
        validation_protocol='walk-forward, frozen alpha/tau; independent EW initialization for each phase',
        evaluation_role='reused development validation and synthetic sensitivity checks, not a new final test')
    return ExperimentResult(frames, metadata, frozen)
