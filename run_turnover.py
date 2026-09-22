"""Tony's Turnover Penalty Model: compare prespecified turnover penalties."""

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
    """Half-L1 daily turnover, with EW as the initial pre-experiment holding."""
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("Weights must be a nonempty DataFrame")
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
        "max_turnover": float(turnover.max()),
        "total_turnover": float(turnover.sum()),
    }


def strength_label(strength):
    if strength == 0:
        return "LASSO_Selected"
    value = f"{strength:g}".replace(".", "p")
    return f"LASSO_Turnover_{value}"


def run_regularized(returns, *, start, end, window, turnover_strength,
                    constraints=PortfolioConstraints()):
    """Run one causal path; each fit sees the preceding realized holding only."""
    previous = None
    records = []

    def strategy(history):
        nonlocal previous
        fitted = fit_portfolio_growth(
            history,
            regularization=Regularization("lasso", LASSO_STRENGTH),
            loss=LogGrowthLoss(
                reward_strength=REWARD_STRENGTH,
                exposure_strength=EXPOSURE_STRENGTH,
                turnover_strength=turnover_strength,
            ),
            constraints=constraints,
            previous_weights=previous,
        )
        previous = fitted.weights.copy()
        records.append({"history_end": history.index[-1], **fitted.diagnostics})
        return fitted.weights

    result = backtest(returns, strategy, start=start, end=end, window=window)
    diagnostics = pd.DataFrame(records, index=result.returns.index)
    observed = turnover_series(result.weights)
    if not np.allclose(diagnostics["turnover"], observed, atol=1e-9, rtol=0):
        raise RuntimeError("Recorded turnover does not match exported weights")
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
        # The initial EW-to-model rebalance is reported in the summary, but it
        # would dominate the scale and hide subsequent day-to-day turnover.
        turnover = turnover_series(result.weights).iloc[1:].rolling(10, min_periods=1).mean()
        axes[0, 1].plot(turnover.index, turnover, label=name,
                        color=color_map[name], linewidth=1.5)
    axes[0, 0].set_title("Cumulative return")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 1].set_title("Ongoing turnover (10-day moving average)")
    axes[0, 1].set_ylabel("Half-L1 turnover")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)

    names = list(results)
    x = np.arange(len(names))
    axes[1, 0].bar(x, summary.loc[names, "sharpe"],
                   color=[color_map[name] for name in names])
    axes[1, 0].set_title("Daily Sharpe ratio")
    axes[1, 0].set_xticks(x, names, rotation=25, ha="right")
    average_turnover = 100 * summary.loc[names, "average_turnover"]
    bars = axes[1, 1].bar(x, average_turnover,
                          color=[color_map[name] for name in names])
    axes[1, 1].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    axes[1, 1].set_title("Average daily turnover (including initial rebalance)")
    axes[1, 1].set_ylabel("Percent")
    axes[1, 1].set_xticks(x, names, rotation=25, ha="right")
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.suptitle("Turnover regularization sensitivity: Jan-Jun 2026", fontsize=14)
    fig.savefig(output / "turnover_comparison.png", dpi=180)
    plt.close(fig)


def write_metadata(args, returns, constraints, summary):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    with args.csv.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    code = [root / "run_turnover.py", root / "requirements.txt",
            *sorted((root / "portfolio").glob("*.py"))]
    metadata = {
        "evaluation_role": "prespecified_validation_sensitivity",
        "selection_metric": None,
        "course_metric": "negative_daily_sharpe",
        "window": args.window,
        "lasso_strength": LASSO_STRENGTH,
        "reward_strength": REWARD_STRENGTH,
        "exposure_strength": EXPOSURE_STRENGTH,
        "turnover_strengths": args.turnover_strengths,
        "turnover_definition": "0.5 * L1(current_weights - previous_weights); initial weights are EW",
        "turnover_scale": "training EW variance (normalized sample weights, ddof=0)",
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
    parser.add_argument("--turnover-strengths", type=float, nargs="+", default=[0, .1, 1])
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
        strengths = [float(value) for value in args.turnover_strengths]
        if (not strengths or len(strengths) != len(set(strengths))
                or any(not np.isfinite(value) or value < 0 for value in strengths)):
            raise ValueError("Turnover strengths must be unique finite nonnegative values")
        args.turnover_strengths = strengths
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
    rows = {}

    ew_name = "EW"
    ew_strategy = lambda history: pd.Series(1 / history.shape[1], index=history.columns)
    ew_result = backtest(evaluation_returns, ew_strategy, start=actual_start,
                         end=actual_end, window=args.window)
    results[ew_name] = ew_result
    rows[ew_name] = metrics(ew_result)
    save_result(args.output, ew_name, ew_result)

    for strength in strengths:
        name = strength_label(strength)
        print(f"Running {name}: tau={strength:g}", flush=True)
        result, diagnostics = run_regularized(
            evaluation_returns, start=actual_start, end=actual_end,
            window=args.window, turnover_strength=strength, constraints=constraints
        )
        results[name] = result
        rows[name] = metrics(result)
        save_result(args.output, name, result, diagnostics)

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.attrs["actual_start"] = actual_start
    summary.attrs["actual_end"] = actual_end
    summary.to_csv(args.output / "summary.csv")
    plot_comparison(args.output, results, summary)
    write_metadata(args, returns, constraints, summary)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
