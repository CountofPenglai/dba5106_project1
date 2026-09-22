"""Compare a Size-BM grid-smoothness penalty against the selected fixed-LASSO portfolio.

The 98 retained portfolios sit on a 10x10 Size/Book-to-Market grid. This adds
one structural assumption to the existing constrained, fixed-strength LASSO
model: portfolios adjacent on that grid should not receive very different
weights unless the data provide enough evidence. rho=0 reproduces the
selected model (eta=0, gamma=2) exactly; rho>0 adds the grid penalty.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio import PortfolioConstraints, Regularization, backtest, fit_portfolio, load_returns
from portfolio.data.asset_graph import build_size_bm_incidence
from portfolio.engine.grid_smoothness_workflow import fit_portfolio_growth_smooth
from portfolio.evaluation.artifact_store import configure_output
from portfolio.evaluation.portfolio_metrics import backtest_metrics
from portfolio.evaluation.provenance import run_provenance
from portfolio.losses import LogGrowthLoss
from portfolio.paths import PROJECT_ROOT

VALIDATION_START, VALIDATION_END, WINDOW = "2026-01-02", "2026-06-30", 252
LASSO = Regularization("lasso", 1e-5)
SELECTED_LOSS = LogGrowthLoss(reward_strength=0, exposure_strength=2)


def _neighbor_gap_metrics(result, incidence):
    daily_gap = result.weights.to_numpy() @ incidence.T
    return {"average_neighbor_gap": float(np.abs(daily_gap).mean())}


def _run_rho(returns, incidence, rho, *, start, end, window):
    records = []

    def strategy(history):
        fitted = fit_portfolio_growth_smooth(
            history, regularization=LASSO, loss=SELECTED_LOSS, constraints=PortfolioConstraints(),
            rho=rho, incidence=incidence)
        records.append({"history_end": str(history.index[-1].date()), **fitted.diagnostics})
        return fitted.weights

    result = backtest(returns, strategy, start=start, end=end, window=window)
    diagnostics = pd.DataFrame(records, index=result.returns.index)
    return result, diagnostics


def main(argv=None, *, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--start", default=VALIDATION_START)
    parser.add_argument("--end", default=VALIDATION_END)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout)
    args = parser.parse_args(argv)
    configure_output(parser, args, "grid_smoothness", optional_flat=False)
    try:
        if args.window <= 0:
            raise ValueError("window must be positive")
        if any(rho < 0 or not np.isfinite(rho) for rho in args.rho):
            raise ValueError("rho values must be finite and nonnegative")
        if len(set(args.rho)) != len(args.rho):
            raise ValueError("rho values must be unique")
        returns = load_returns(args.csv)
        if len(returns.loc[args.start:args.end]) < 2:
            raise ValueError("Evaluation needs at least two trading days")
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    incidence, edges = build_size_bm_incidence(returns.columns)
    print(f"Assets: {returns.shape[1]}; grid edges: {len(edges)}; "
          f"evaluation: {args.start} to {args.end}, window={args.window}", flush=True)

    rows, gap_rows = {}, {}

    print("Running EW...", flush=True)
    ew_strategy = lambda history: pd.Series(1 / history.shape[1], index=history.columns)
    ew_result = backtest(returns, ew_strategy, start=args.start, end=args.end, window=args.window)
    rows["EW"] = backtest_metrics(ew_result)
    gap_rows["EW"] = _neighbor_gap_metrics(ew_result, incidence)
    ew_result.returns.to_csv(args.artifacts.path("EW_returns.csv"))
    ew_result.weights.to_csv(args.artifacts.path("EW_weights.csv"))

    print("Running plain LASSO (no constraints, no reward/exposure/smoothness)...", flush=True)
    lasso_strategy = lambda history: fit_portfolio(history, regularization=LASSO).weights
    lasso_result = backtest(returns, lasso_strategy, start=args.start, end=args.end, window=args.window)
    rows["LASSO"] = backtest_metrics(lasso_result)
    gap_rows["LASSO"] = _neighbor_gap_metrics(lasso_result, incidence)
    lasso_result.returns.to_csv(args.artifacts.path("LASSO_returns.csv"))
    lasso_result.weights.to_csv(args.artifacts.path("LASSO_weights.csv"))

    for rho in args.rho:
        name = "Selected_LASSO_rho0" if rho == 0 else f"Smooth_LASSO_rho{rho:g}".replace(".", "_")
        print(f"Running {name} (rho={rho:g})...", flush=True)
        result, diagnostics = _run_rho(returns, incidence, rho, start=args.start, end=args.end, window=args.window)
        rows[name] = backtest_metrics(result)
        gap_rows[name] = _neighbor_gap_metrics(result, incidence)
        result.returns.to_csv(args.artifacts.path(f"{name}_returns.csv"))
        result.weights.to_csv(args.artifacts.path(f"{name}_weights.csv"))
        diagnostics.to_csv(args.artifacts.path(f"{name}_diagnostics.csv"))

    summary = pd.DataFrame.from_dict(rows, orient="index")
    gaps = pd.DataFrame.from_dict(gap_rows, orient="index")
    summary = summary.join(gaps)
    summary.to_csv(args.artifacts.path("summary.csv"))
    print("\n" + summary.to_string(float_format=lambda value: f"{value:.6f}"))

    metadata = {
        **run_provenance(args.csv),
        "artifact_layout": args.layout, "artifact_layout_version": 1,
        "evaluation_role": "development_validation",
        "start": args.start, "end": args.end, "window": args.window,
        "assets": list(returns.columns), "grid_edges": len(edges),
        "lasso_strength": LASSO.strength, "selected_loss": asdict(SELECTED_LOSS),
        "constraints": asdict(PortfolioConstraints()), "rho_values": args.rho,
        "grid_gap_definition": "mean(|w_i - w_j|) over horizontal/vertical Size-BM neighbor edges",
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "final_test": {"start": "2026-08-01", "end": "2026-08-30", "evaluated": False},
    }
    args.artifacts.path("run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    args.artifacts.finalize()
    print(f"\nSaved artifacts to {args.output.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    main(default_layout="categorized")
