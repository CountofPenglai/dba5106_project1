"""Tony's Ledoit-Wolf Shrinkage Model: compare two covariance estimators."""

import argparse
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio import Regularization, backtest, load_returns, summarize
from portfolio.splits import rolling_windows
from portfolio.tony_models import (LogGrowthLoss, PortfolioConstraints,
                                   fit_portfolio_growth)


FINAL_TEST_START = "2026-08-01"
LASSO_STRENGTH = 1e-5
REWARD_STRENGTH = 0.0
EXPOSURE_STRENGTH = 2.0


def turnover_series(weights):
    """Half-L1 target-weight turnover, with EW before the first evaluation day."""
    initial = np.full(weights.shape[1], 1 / weights.shape[1])
    previous = np.vstack([initial, weights.to_numpy(dtype=float)[:-1]])
    values = 0.5 * np.abs(weights.to_numpy(dtype=float) - previous).sum(axis=1)
    return pd.Series(values, index=weights.index, name="turnover")


def metrics(result):
    gross = result.weights.abs().sum(axis=1)
    turnover = turnover_series(result.weights)
    return {
        **summarize(result.returns),
        "average_gross_leverage": float(gross.mean()),
        "max_gross_leverage": float(gross.max()),
        "average_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
    }


def run_regularized(returns, *, start, end, window, covariance_estimator,
                    constraints=PortfolioConstraints()):
    records = []

    def strategy(history):
        fitted = fit_portfolio_growth(
            history,
            regularization=Regularization("lasso", LASSO_STRENGTH),
            loss=LogGrowthLoss(
                reward_strength=REWARD_STRENGTH,
                exposure_strength=EXPOSURE_STRENGTH,
            ),
            constraints=constraints,
            covariance_estimator=covariance_estimator,
        )
        records.append({"history_end": history.index[-1], **fitted.diagnostics})
        return fitted.weights

    result = backtest(returns, strategy, start=start, end=end, window=window)
    diagnostics = pd.DataFrame(records, index=result.returns.index)
    return result, diagnostics


def save_result(output, name, result, diagnostics=None):
    result.returns.to_csv(output / f"{name}_returns.csv")
    result.weights.to_csv(output / f"{name}_weights.csv")
    turnover_series(result.weights).to_csv(output / f"{name}_turnover.csv")
    if diagnostics is not None:
        diagnostics.to_csv(output / f"{name}_diagnostics.csv")


def plot_comparison(output, results, summary):
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    color_map = dict(zip(results, colors))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for name, result in results.items():
        cumulative = (1 + result.returns).cumprod() - 1
        axes[0, 0].plot(cumulative.index, 100 * cumulative,
                        label=name, color=color_map[name], linewidth=1.8)
        rolling_volatility = result.returns.rolling(20).std(ddof=1)
        axes[0, 1].plot(rolling_volatility.index, 100 * rolling_volatility,
                        label=name, color=color_map[name], linewidth=1.6)
    axes[0, 0].set_title("Cumulative return")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 1].set_title("Realized volatility (20-day rolling)")
    axes[0, 1].set_ylabel("Percent")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)

    names = list(results)
    x = np.arange(len(names))
    sharpe_bars = axes[1, 0].bar(x, summary.loc[names, "sharpe"],
                                 color=[color_map[name] for name in names])
    axes[1, 0].bar_label(sharpe_bars, fmt="%.4f", padding=3, fontsize=9)
    axes[1, 0].set_title("Daily Sharpe ratio")
    axes[1, 0].set_xticks(x, names, rotation=20, ha="right")
    average_turnover = 100 * summary.loc[names, "average_turnover"]
    turnover_bars = axes[1, 1].bar(x, average_turnover,
                                   color=[color_map[name] for name in names])
    axes[1, 1].bar_label(turnover_bars, fmt="%.2f%%", padding=3, fontsize=9)
    axes[1, 1].set_title("Average daily turnover (including initial rebalance)")
    axes[1, 1].set_ylabel("Percent")
    axes[1, 1].set_xticks(x, names, rotation=20, ha="right")
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.suptitle("Covariance ablation: sample vs Ledoit-Wolf", fontsize=14)
    fig.savefig(output / "shrinkage_comparison.png", dpi=180)
    plt.close(fig)


def write_metadata(args, returns, constraints, diagnostics, summary):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    with args.csv.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    code = [root / "run_shrinkage.py", root / "requirements.txt",
            *sorted((root / "portfolio").glob("*.py"))]
    metadata = {
        "evaluation_role": "prespecified_validation_ablation",
        "selection_metric": None,
        "course_metric": "negative_daily_sharpe",
        "comparison": ["sample", "ledoit_wolf"],
        "window": args.window,
        "lasso_strength": LASSO_STRENGTH,
        "reward_strength": REWARD_STRENGTH,
        "exposure_strength": EXPOSURE_STRENGTH,
        "ledoit_wolf_shrinkage_mean": float(
            diagnostics["LASSO_LedoitWolf"]["ledoit_wolf_shrinkage"].mean()
        ),
        "ledoit_wolf_shrinkage_range": [
            float(diagnostics["LASSO_LedoitWolf"]["ledoit_wolf_shrinkage"].min()),
            float(diagnostics["LASSO_LedoitWolf"]["ledoit_wolf_shrinkage"].max()),
        ],
        "constraints": vars(constraints),
        "requested_start": args.start,
        "requested_end": args.end,
        "actual_start": str(summary.attrs["actual_start"]),
        "actual_end": str(summary.attrs["actual_end"]),
        "csv": str(args.csv.resolve()),
        "data_sha256": digest.hexdigest(),
        "data_first_date": str(returns.index[0].date()),
        "data_last_date": str(returns.index[-1].date()),
        "assets": list(returns.columns),
        "code_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code
        },
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "cvxpy",
                            "clarabel", "scs", "matplotlib")
        },
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "final_test": {
            "start": FINAL_TEST_START,
            "evaluated": False,
            "note": "This command is restricted to dates before the reserved final test.",
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path(__file__).parent / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be an empty or new directory")
    try:
        if args.window < 2:
            raise ValueError("--window must be at least two")
        for value in (args.start, args.end):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Dates must use complete YYYY-MM-DD format")
        if not pd.Timestamp(args.start) <= pd.Timestamp(args.end):
            raise ValueError("Require start <= end")
        if pd.Timestamp(args.end) >= pd.Timestamp(FINAL_TEST_START):
            raise ValueError(f"Evaluation must end before {FINAL_TEST_START}")
        returns = load_returns(args.csv)
        days = returns.loc[args.start:args.end].index
        if len(days) < 2:
            raise ValueError("Evaluation needs at least two trading days")
        _, first_history = next(rolling_windows(
            returns, start=args.start, end=args.end, window=args.window
        ))
        relevant = returns.loc[first_history.index[0]:args.end].to_numpy(dtype=float)
        if not np.isfinite(relevant).all():
            raise ValueError("Training/evaluation returns must be finite")
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    constraints = PortfolioConstraints()
    actual_start, actual_end = str(days[0].date()), str(days[-1].date())
    evaluation_returns = returns.loc[:actual_end].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    diagnostics = {}
    rows = {}

    ew_strategy = lambda history: pd.Series(1 / history.shape[1], index=history.columns)
    ew_result = backtest(evaluation_returns, ew_strategy, start=actual_start,
                         end=actual_end, window=args.window)
    results["EW"] = ew_result
    rows["EW"] = metrics(ew_result)
    save_result(args.output, "EW", ew_result)

    for name, estimator in (("LASSO_Selected", "sample"),
                            ("LASSO_LedoitWolf", "ledoit_wolf")):
        print(f"Running {name}: covariance={estimator}", flush=True)
        result, fit_diagnostics = run_regularized(
            evaluation_returns, start=actual_start, end=actual_end,
            window=args.window, covariance_estimator=estimator,
            constraints=constraints,
        )
        results[name] = result
        diagnostics[name] = fit_diagnostics
        rows[name] = metrics(result)
        save_result(args.output, name, result, fit_diagnostics)

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.attrs["actual_start"] = actual_start
    summary.attrs["actual_end"] = actual_end
    summary.to_csv(args.output / "summary.csv")
    plot_comparison(args.output, results, summary)
    write_metadata(args, returns, constraints, diagnostics, summary)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
