"""Run four baselines, optionally adding sample and portfolio weighting."""

import argparse
from functools import partial
from pathlib import Path

import pandas as pd

from portfolio import Regularization, backtest, fit_portfolio, load_returns, summarize
from portfolio.weights import exponential, shrink_to_equal


def equal_weight(history):
    return pd.Series(1 / history.shape[1], index=history.columns)


def portfolio_strategy(**options):
    """Adapt the richer FitResult to backtest's history -> weights interface."""
    def strategy(history):
        return fit_portfolio(history, **options).weights
    return strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--window", type=int, default=126)
    parser.add_argument("--weighted", action="store_true", help="Also run Ridge with both weighting hooks")
    parser.add_argument("--output", type=Path, help="Optional directory for returns, weights and summary CSVs")
    args = parser.parse_args()

    returns = load_returns(args.csv)
    strategies = {
        "EW": equal_weight,
        "MinVar": portfolio_strategy(),
        "LASSO": portfolio_strategy(regularization=Regularization("lasso", 1e-5)),
        "Ridge": portfolio_strategy(regularization=Regularization("ridge", 1e-4)),
    }
    if args.weighted:
        strategies["RidgeWeighted"] = portfolio_strategy(
            regularization=Regularization("ridge", 1e-4),
            sample_weights=partial(exponential, half_life=63),
            adjust_weights=partial(shrink_to_equal, strength=0.25),
        )

    print(f"Assets: {returns.shape[1]}; evaluation: {args.start} to {args.end}; window: {args.window}")
    print("Fixed example strengths; no hyperparameter search. Returns are decimal fractions; Sharpe is daily.")
    rows = {}
    for name, strategy in strategies.items():
        result = backtest(returns, strategy, start=args.start, end=args.end, window=args.window)
        rows[name] = summarize(result.returns)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            result.returns.to_csv(args.output / f"{name}_returns.csv")
            result.weights.to_csv(args.output / f"{name}_weights.csv")
    summary = pd.DataFrame.from_dict(rows, orient="index")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    if args.output is not None:
        summary.to_csv(args.output / "summary.csv")


if __name__ == "__main__":
    main()
