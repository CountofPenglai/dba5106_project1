"""Tony's Concentration Constraint Model: test one asset-weight cap."""

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


def concentration_frame(weights):
    """Daily concentration statistics computed from exported target weights."""
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("Weights must be a nonempty DataFrame")
    values = weights.to_numpy(dtype=float)
    squared = np.square(values).sum(axis=1)
    top_count = min(10, values.shape[1])
    return pd.DataFrame({
        "max_abs_weight": np.abs(values).max(axis=1),
        "effective_assets": 1 / squared,
        "top10_weight": np.sort(values, axis=1)[:, -top_count:].sum(axis=1),
    }, index=weights.index)


def metrics(result):
    gross = result.weights.abs().sum(axis=1)
    concentration = concentration_frame(result.weights)
    return {
        **summarize(result.returns),
        "average_gross_leverage": float(gross.mean()),
        "max_gross_leverage": float(gross.max()),
        "average_max_abs_weight": float(concentration["max_abs_weight"].mean()),
        "max_abs_weight": float(concentration["max_abs_weight"].max()),
        "average_effective_assets": float(concentration["effective_assets"].mean()),
        "average_top10_weight": float(concentration["top10_weight"].mean()),
    }


def capped_label(weight_limit):
    value = f"{100 * weight_limit:g}".replace(".", "p")
    return f"LASSO_WeightCap_{value}pct"


def run_regularized(returns, *, start, end, window, constraints):
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
        )
        records.append({"history_end": history.index[-1], **fitted.diagnostics})
        return fitted.weights

    result = backtest(returns, strategy, start=start, end=end, window=window)
    diagnostics = pd.DataFrame(records, index=result.returns.index)
    observed = concentration_frame(result.weights)["max_abs_weight"]
    if not np.allclose(diagnostics["max_abs_weight"], observed, atol=1e-9, rtol=0):
        raise RuntimeError("Recorded concentration does not match exported weights")
    return result, diagnostics


def save_result(output, name, result, diagnostics=None):
    result.returns.to_csv(output / f"{name}_returns.csv")
    result.weights.to_csv(output / f"{name}_weights.csv")
    concentration_frame(result.weights).to_csv(output / f"{name}_concentration.csv")
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
        maximum = 100 * concentration_frame(result.weights)["max_abs_weight"]
        axes[0, 1].plot(maximum.index, maximum, label=name,
                        color=color_map[name], linewidth=1.5)
    axes[0, 0].set_title("Cumulative return")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 1].set_title("Largest absolute asset weight")
    axes[0, 1].set_ylabel("Percent")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)

    names = list(results)
    x = np.arange(len(names))
    axes[1, 0].bar(x, summary.loc[names, "sharpe"],
                   color=[color_map[name] for name in names])
    axes[1, 0].set_title("Daily Sharpe ratio")
    axes[1, 0].set_xticks(x, names, rotation=20, ha="right")
    effective = summary.loc[names, "average_effective_assets"]
    bars = axes[1, 1].bar(x, effective,
                          color=[color_map[name] for name in names])
    axes[1, 1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1, 1].set_title("Average effective number of assets")
    axes[1, 1].set_xticks(x, names, rotation=20, ha="right")
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.suptitle("Single-asset concentration constraint: Jan-Jun 2026", fontsize=14)
    fig.savefig(output / "concentration_comparison.png", dpi=180)
    plt.close(fig)


def write_metadata(args, returns, baseline_constraints, capped_constraints, summary):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    with args.csv.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    code = [root / "run_concentration.py", root / "requirements.txt",
            *sorted((root / "portfolio").glob("*.py"))]
    metadata = {
        "evaluation_role": "prespecified_validation_sensitivity",
        "selection_metric": None,
        "course_metric": "negative_daily_sharpe",
        "window": args.window,
        "lasso_strength": LASSO_STRENGTH,
        "reward_strength": REWARD_STRENGTH,
        "exposure_strength": EXPOSURE_STRENGTH,
        "weight_limit": args.weight_limit,
        "weight_limit_definition": "absolute upper bound applied to every fitted asset weight",
        "baseline_constraints": vars(baseline_constraints),
        "capped_constraints": vars(capped_constraints),
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
    parser.add_argument("--weight-limit", type=float, default=.10)
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
        if not np.isfinite(args.weight_limit) or args.weight_limit <= 0:
            raise ValueError("Weight limit must be a positive finite number")
        returns = load_returns(args.csv)
        if args.weight_limit * returns.shape[1] < 1 - 1e-12:
            raise ValueError("Weight limit is infeasible for the number of assets")
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

    baseline_constraints = PortfolioConstraints()
    capped_constraints = PortfolioConstraints(weight_limit=args.weight_limit)
    actual_start, actual_end = str(days[0].date()), str(days[-1].date())
    evaluation_returns = returns.loc[:actual_end].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    rows = {}

    ew_strategy = lambda history: pd.Series(1 / history.shape[1], index=history.columns)
    ew_result = backtest(evaluation_returns, ew_strategy, start=actual_start,
                         end=actual_end, window=args.window)
    results["EW"] = ew_result
    rows["EW"] = metrics(ew_result)
    save_result(args.output, "EW", ew_result)

    capped_name = capped_label(args.weight_limit)
    for name, constraints in (("LASSO_Selected", baseline_constraints),
                              (capped_name, capped_constraints)):
        print(f"Running {name}", flush=True)
        result, diagnostics = run_regularized(
            evaluation_returns, start=actual_start, end=actual_end,
            window=args.window, constraints=constraints,
        )
        results[name] = result
        rows[name] = metrics(result)
        save_result(args.output, name, result, diagnostics)

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.attrs["actual_start"] = actual_start
    summary.attrs["actual_end"] = actual_end
    summary.to_csv(args.output / "summary.csv")
    plot_comparison(args.output, results, summary)
    write_metadata(args, returns, baseline_constraints, capped_constraints, summary)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
