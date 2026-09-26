"""Compare rolling portfolios, with optional weighting and log-growth experiments."""

from portfolio.paths import PROJECT_ROOT
from portfolio.evaluation.artifact_store import configure_output
from portfolio.evaluation.provenance import run_provenance

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

import pandas as pd

from portfolio import Regularization, backtest, fit_portfolio_cv, load_returns, summarize
from portfolio import fit_portfolio_growth, LogGrowthLoss, PortfolioConstraints, CoefficientConstraint
from portfolio.config import TrainingConfig
from portfolio.data.history_windows import TrainingWindow
from portfolio.engine.baseline_strategies import (
    BASELINE_REGULARIZATION, baseline_strategies, penalty_baseline_strategies, equal_weight, portfolio_strategy,
)
from portfolio.data.observation_weights import exponential
from portfolio.decisions.holding_weights import shrink_to_equal


def cv_portfolio_strategy(kind, **options):
    """Same idea, but pick alpha by cross-validation fresh in every window."""
    def strategy(history):
        return fit_portfolio_cv(history, kind=kind, **options).weights
    return strategy


def growth_portfolio_strategy(*, records, **options):
    """Keep fit diagnostics alongside the same history -> weights interface."""
    def strategy(history):
        fitted = fit_portfolio_growth(history, **options)
        records.append({"history_end": history.index[-1], **fitted.diagnostics})
        return fitted.weights
    return strategy


def write_metadata(args, returns, evaluation_days, constraints, training, models):
    """Record the dataset, code, versions, and fixed settings used in this run."""
    metadata = {
        **run_provenance(args.csv),
        "artifact_layout": args.layout, "artifact_layout_version": 1,
        "data_first_date": str(returns.index[0].date()),
        "data_last_date": str(returns.index[-1].date()), "assets": list(returns.columns),
        "start": args.start, "end": args.end, "window": args.window,
        "evaluation_days": evaluation_days, "weighted": args.weighted,
        "models": models, "baselines_only": args.baselines_only, "training": asdict(training),
        "coefficient_constraints": ({
            "lasso": asdict(CoefficientConstraint("lasso", args.lasso_coefficient_budget)),
            "ridge": asdict(CoefficientConstraint("ridge", args.ridge_coefficient_budget)),
        } if args.baselines_only else None),
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "baseline_cv_scoring": {"lasso": "MSE", "ridge": "R2 (existing sklearn default with cv=5)"},
        "baseline_parameters": {
            "fixed_strengths": {reg.kind: reg.strength for reg in BASELINE_REGULARIZATION.values()
                                if reg.kind != "none"},
            "cv": {"folds": 5, "shuffle": False, "alpha_scale": "sklearn native",
                   "alpha_grid": {"type": "logspace", "start": -8, "stop": 8, "count": 21}},
            "weighted": {"half_life": 63, "shrink_to_equal": 0.25},
        },
        "growth": {"enabled": args.growth, "lasso_strength": 1e-5,
                   "reward_strength": args.reward_strength, "control_reward_strength": 0.0,
                   "hyperparameter_search": False, "sample_weights": "uniform",
                   "post_fit_adjustment": None, **asdict(constraints)},
    }
    (args.artifacts.path("run_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv=None, *, default_layout="flat", baselines_only=False):
    parser = argparse.ArgumentParser(description=("Six basic portfolios: EW, MinVar, coefficient-constrained and penalty LASSO/Ridge."
                                                  if baselines_only else __doc__))
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--window", type=int, default=252,
                        help="Prior trading observations per fit (e.g. 126 or 252; default: 252)")
    parser.add_argument("--baselines-only", action="store_true", default=baselines_only,
                        help="Run the six basic models (coefficient constraints and penalties); excludes CV and overlays")
    parser.add_argument("--lasso-coefficient-budget", type=float, default=1.0,
                        help="Six-model suite: sum(abs(beta)) <= budget (default: 1; no tuning)")
    parser.add_argument("--ridge-coefficient-budget", type=float, default=1.0,
                        help="Six-model suite: sum(beta**2) <= budget (default: 1; no tuning)")
    parser.add_argument("--weighted", action="store_true", help="Add fixed Ridge and CV LASSO/Ridge with both weighting hooks")
    parser.add_argument("--growth", action="store_true", help="Add fixed LASSO with constraints, with and without a log-growth reward")
    parser.add_argument("--reward-strength", type=float, default=0.01, help="Fixed log-growth reward coefficient; no search")
    parser.add_argument("--leverage-limit", type=float, default=1.5, help="Growth experiments: maximum sum(abs(weights))")
    parser.add_argument("--volatility-limit", type=float, default=1.0, help="Growth experiments: training volatility / EW volatility limit")
    parser.add_argument("--output", type=Path, help="Optional directory for returns, weights and summary CSVs")
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout,
                        help="Categorized artifact folders or historical flat filenames")
    args = parser.parse_args(argv)
    if args.baselines_only and (args.weighted or args.growth):
        parser.error("--baselines-only cannot be combined with --weighted or --growth")
    try:
        training = TrainingConfig(TrainingWindow(observations=args.window))
        CoefficientConstraint("lasso", args.lasso_coefficient_budget)
        CoefficientConstraint("ridge", args.ridge_coefficient_budget)
    except ValueError as error:
        parser.error(str(error))
    configure_output(parser, args, "baselines" if args.baselines_only else "regularized_baselines",
                     optional_flat=True)
    loss = LogGrowthLoss(reward_strength=args.reward_strength)
    constraints = PortfolioConstraints(leverage_limit=args.leverage_limit,
                                       volatility_limit=args.volatility_limit)

    returns = load_returns(args.csv)
    diagnostics = {}
    strategies = (baseline_strategies(training, lasso_budget=args.lasso_coefficient_budget,
                                     ridge_budget=args.ridge_coefficient_budget, diagnostics=diagnostics)
                  if args.baselines_only else penalty_baseline_strategies(training))
    if not args.baselines_only:
        strategies.update({
            "LASSO_CV": cv_portfolio_strategy("lasso"),
            "Ridge_CV": cv_portfolio_strategy("ridge"),
        })
    if args.weighted:
        strategies["RidgeWeighted"] = portfolio_strategy(
            regularization=Regularization("ridge", 1e-4),
            sample_weights=partial(exponential, half_life=63),
            adjust_weights=partial(shrink_to_equal, strength=0.25),
        )
        strategies["LASSO_CV_Weighted"] = cv_portfolio_strategy(
            "lasso",
            sample_weights=partial(exponential, half_life=63),
            adjust_weights=partial(shrink_to_equal, strength=0.25),
        )
        strategies["Ridge_CV_Weighted"] = cv_portfolio_strategy(
            "ridge",
            sample_weights=partial(exponential, half_life=63),
            adjust_weights=partial(shrink_to_equal, strength=0.25),
        )

    if args.growth:
        for name, objective in [("LASSO_Constrained", LogGrowthLoss(reward_strength=0)),
                                ("LASSO_LogGrowth", loss)]:
            diagnostics[name] = []
            strategies[name] = growth_portfolio_strategy(
                records=diagnostics[name], loss=objective, constraints=constraints,
                regularization=Regularization("lasso", 1e-5))

    print(f"Assets: {returns.shape[1]}; evaluation: {args.start} to {args.end}; window: {args.window}")
    print("Penalty strengths: LASSO=1e-5, Ridge=1e-4. Sharpe is daily.")
    if args.baselines_only:
        print(f"Coefficient budgets: LASSO sum(abs(beta))<={args.lasso_coefficient_budget:g}; "
              f"Ridge sum(beta**2)<={args.ridge_coefficient_budget:g}. No tuning.")
    if not args.baselines_only:
        print("CV variants reselect alpha per window.")
    if args.growth:
        print(f"Growth experiment: fixed LASSO=1e-5, eta={loss.reward_strength:g}; "
              f"gross<={constraints.leverage_limit:g}, training volatility<={constraints.volatility_limit:g}*EW. No search.")
    rows = {}
    for name, strategy in strategies.items():
        print(f"Running {name}...", flush=True)
        result = backtest(returns, strategy, start=args.start, end=args.end, window=training.window)
        rows[name] = summarize(result.returns)
        gross = result.weights.abs().sum(axis=1)
        rows[name].update(average_gross_leverage=float(gross.mean()), max_gross_leverage=float(gross.max()))
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            result.returns.to_csv(args.artifacts.path(f"{name}_returns.csv"))
            result.weights.to_csv(args.artifacts.path(f"{name}_weights.csv"))
            if name in diagnostics:
                pd.DataFrame(diagnostics[name], index=result.returns.index).to_csv(
                    args.artifacts.path(f"{name}_diagnostics.csv"))
    summary = pd.DataFrame.from_dict(rows, orient="index")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    if args.output is not None:
        summary.to_csv(args.artifacts.path("summary.csv"))
        write_metadata(args, returns, len(result.returns), constraints, training, list(rows))

    if args.artifacts is not None:
        args.artifacts.finalize()
        print(f"Saved artifacts to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main(default_layout="categorized")
