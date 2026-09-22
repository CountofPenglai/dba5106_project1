"""Compare constrained regularized portfolios with market-beta bounds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio.data.returns_loader import load_returns as _load_returns
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.artifact_store import (DEFAULT_CSV, RunArtifacts, default_output,
                                                 ensure_fresh_output)
from portfolio.evaluation.portfolio_metrics import summarize as _summarize
from portfolio.evaluation.provenance import run_provenance
from portfolio.models.market_beta_portfolio import calculate_market_betas
from portfolio.models.market_beta_portfolio import fit_portfolio as _fit_portfolio

DATA = DEFAULT_CSV
START_DATE, END_DATE, WINDOW = "2026-01-02", "2026-06-30", 252
LASSO_STRENGTH, RIDGE_STRENGTH = 1e-5, 1e-4
GROSS_LIMIT, WEALTH_FLOOR = 1.5, 1e-6
EXPERIMENTS = {
    "LASSO_unconstrained": ("LASSO", None), "Ridge_unconstrained": ("Ridge", None),
    "LASSO_market_like_0.8_1.2": ("LASSO", (0.8, 1.2)),
    "Ridge_market_like_0.8_1.2": ("Ridge", (0.8, 1.2)),
    "LASSO_conservative_0.5_0.9": ("LASSO", (0.5, 0.9)),
    "Ridge_conservative_0.5_0.9": ("Ridge", (0.5, 0.9)),
}

def load_returns(path):
    returns = _load_returns(path).dropna()
    if returns.shape[1] != 98:
        raise ValueError(f"Expected 98 assets; found {returns.shape[1]}.")
    return returns

def _fit(history, kind, bounds):
    return fit_portfolio(history, kind, bounds)

def fit_portfolio(history, model_type, beta_bounds=None):
    return _fit_portfolio(history, model_type, beta_bounds, lasso_strength=LASSO_STRENGTH,
                         ridge_strength=RIDGE_STRENGTH, gross_limit=GROSS_LIMIT,
                         wealth_floor=WEALTH_FLOOR)

def summarize(values):
    metrics = _summarize(values)
    return {"mean_daily_return": metrics["mean"], "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"], "cumulative_return": metrics["cumulative_return"],
            "maximum_drawdown": metrics["max_drawdown"]}

def _backtest(returns, kind, bounds, start, end, window):
    betas = []
    def strategy(history):
        weights, beta = _fit(history, kind, bounds)
        betas.append(beta)
        return pd.Series(weights, index=history.columns)
    return backtest(returns, strategy, start=start, end=end, window=window), betas

def _metrics(result, betas, bounds):
    values = summarize(result.returns)
    binding = [] if bounds is None else [abs(beta-bounds[0]) < 1e-5 or abs(beta-bounds[1]) < 1e-5 for beta in betas]
    return {"mean_daily_return": values["mean_daily_return"], "volatility": values["volatility"],
            "sharpe": values["sharpe"], "cumulative_return": values["cumulative_return"],
            "maximum_drawdown": values["maximum_drawdown"], "average_beta": np.mean(betas),
            "minimum_beta": np.min(betas), "maximum_beta": np.max(betas),
            "beta_constraint_binding_pct": np.nan if bounds is None else 100*np.mean(binding)}

def run_experiment(returns, model_type, beta_bounds, *, start=None, end=None, window=None):
    result, betas = _backtest(returns, model_type, beta_bounds, start or START_DATE,
                              end or END_DATE, WINDOW if window is None else window)
    return _metrics(result, betas, beta_bounds)

def main(argv=None, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout)
    args = parser.parse_args(argv)
    try:
        store = RunArtifacts(args.output or default_output("market_beta_constraints"), args.layout)
        ensure_fresh_output(store.root)
        returns, rows = load_returns(args.csv), {}
        if args.window <= 0 or len(returns.loc[args.start:args.end]) < 2:
            raise ValueError("window must be positive and evaluation needs at least two complete rows")
        store.prepare()
        for name, (kind, bounds) in EXPERIMENTS.items():
            print(f"Running {name}...", flush=True)
            result, betas = _backtest(returns, kind, bounds, args.start, args.end, args.window)
            rows[name] = _metrics(result, betas, bounds)
            store.write_backtest(name, result)
            pd.Series(betas, index=result.returns.index, name="fitted_beta").to_csv(store.path(f"{name}_diagnostics.csv"))
        table = pd.DataFrame(rows).T
        table.to_csv(store.path("market_beta_results.csv"))
        store.path("run_metadata.json").write_text(json.dumps({**run_provenance(args.csv),
            "start": args.start, "end": args.end, "window": args.window,
            "lasso_strength": LASSO_STRENGTH, "ridge_strength": RIDGE_STRENGTH,
            "gross_limit": GROSS_LIMIT, "wealth_floor": WEALTH_FLOOR,
            "artifact_layout": args.layout, "artifact_layout_version": 1,
            "missing_row_policy": "drop every row missing any retained asset",
            "solver_policy": "CLARABEL; retry SCS only on cvxpy SolverError; accept optimal or optimal_inaccurate"}, indent=2))
        store.finalize()
        print("\nFinal results:\n\n" + table.to_string(float_format=lambda value: f"{value:.8f}"))
        print(f"\nSaved artifacts to {store.root.resolve()}")
        return table
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main(default_layout="categorized")
