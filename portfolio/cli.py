"""Shared command-line options for the baseline and report entrypoints."""
import argparse
from dataclasses import asdict
from pathlib import Path

from portfolio.config import EVALUATION_PERIODS, HISTORY_START
from portfolio.data.returns_loader import load_returns
from portfolio.decisions.linear_program_allocation import LPConstraints
from portfolio.engine.report_runner import load_forecast_settings, run_models, verify_forecast_reference
from portfolio.evaluation.artifact_store import default_output, ensure_fresh_output, save_run
from portfolio.evaluation.provenance import run_provenance
from portfolio.model_catalog import BASELINE_NAMES, COURSE_MODELS, JULY_MODELS, MODELS, baseline_models
from portfolio.models.course_cross_validation import COURSE_ALPHAS, CV_DAYS, CV_SPLITS
from portfolio.paths import DEFAULT_CSV


def main(argv=None, *, baselines_only=False):
    archived = {name: MODELS[name] for name in BASELINE_NAMES} if baselines_only else MODELS
    available = {**archived, **COURSE_MODELS}
    parser = argparse.ArgumentParser(description='Run course baselines or reproduce report portfolios.')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--window', type=int,
                        help='Prior trading observations; course default 126, archived report default 252.')
    parser.add_argument('--period', choices=tuple(EVALUATION_PERIODS), default='validation')
    parser.add_argument('--start', help='Optional evaluation start date.')
    parser.add_argument('--end', help='Optional evaluation end.')
    parser.add_argument('--models', nargs='+', choices=tuple(available))
    parser.add_argument('--lasso-coefficient-budget', type=float, default=1.)
    parser.add_argument('--ridge-coefficient-budget', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--list', action='store_true', help='List retained model names without running them.')
    args = parser.parse_args(argv)
    if args.list:
        for name in available:
            print(f'{name}: {available[name].title} ({available[name].section})')
        return
    start, end = EVALUATION_PERIODS[args.period]
    start, end = args.start or start, args.end or end
    defaults = tuple(COURSE_MODELS) if baselines_only else (JULY_MODELS if args.period == 'july_test' else tuple(MODELS))
    names = args.models or defaults
    catalog = {**available, **{spec.name: spec for spec in baseline_models(
        lasso_budget=args.lasso_coefficient_budget, ridge_budget=args.ridge_coefficient_budget)}}
    selected = [catalog[name] for name in names]
    uses_cv = any(spec.family == 'cv' for spec in selected)
    if args.window is None:
        args.window = 126 if baselines_only or uses_cv else 252
    output = args.output or default_output('baselines' if baselines_only else 'report')
    try:
        ensure_fresh_output(output)
        returns = load_returns(args.csv, start=HISTORY_START, end=end)
        settings = None
        if any(spec.family == 'lp' for spec in selected):
            settings = load_forecast_settings()
            if args.window != settings['window']:
                raise ValueError('The frozen LP forecast was calibrated for 252 days; select non-LP models for other windows')
            verify_forecast_reference(returns, settings, start=start)
        result = run_models(returns, selected, start=start, end=end, window=args.window,
                            forecast_alpha=None if settings is None else settings['alpha'],
                            lp_constraints=LPConstraints() if settings is None else LPConstraints(**settings['lp_constraints']),
                            progress=print)
        metadata = {**run_provenance(args.csv), 'window': args.window, 'period': args.period,
            'start': start, 'end': end, 'history_start': HISTORY_START,
            'assets': list(returns.columns), 'models': [asdict(spec) for spec in selected],
            'forecast_settings': settings, 'parameters_reselected': uses_cv,
            'window_cv': {'models': [s.name for s in selected if s.family == 'cv'],
                'criterion': 'mean squared error', 'alphas_sklearn': COURSE_ALPHAS.tolist(),
                'expanding_folds': CV_SPLITS, 'days_per_fold': CV_DAYS} if uses_cv else None,
            'independent_untouched_test': False if args.period == 'july_test' else None,
            'metric_convention': {'primary_returns': 'gross', 'sharpe': 'daily mean/std(ddof=1), risk-free rate=0',
                'target_turnover': '0.5 * L1(target - previous target)',
                'trade_turnover': '0.5 * L1(target - drifted pretrade)',
                'supplementary_cost_rate': .0005, 'initial_holdings': 'EW at each evaluation start'}}
        save_run(result, output, metadata)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(result.summary[['mean', 'volatility', 'sharpe', 'cumulative_return', 'max_drawdown']].to_string())
    print(f'Results: {output}')
