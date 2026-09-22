"""Compare fixed LASSO windows, optionally adding the existing hard constraints."""

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from portfolio import (Regularization, PortfolioConstraints, LogGrowthLoss, backtest,
                       fit_portfolio, fit_portfolio_growth, load_returns, summarize)
from portfolio.config import TRAINING_PRESETS
from portfolio.splits import rolling_windows
from portfolio.weights import normalize_sample_weights


FINAL_TEST_START, FINAL_TEST_END = "2026-08-01", "2026-08-30"
HISTORY_START, HISTORY_END = "2021-08-01", "2026-07-31"
VALIDATION_START, VALIDATION_END = "2026-01-02", "2026-06-30"
LASSO = Regularization("lasso", 1e-5)


def lasso_strategy(config, records, *, constraints=None):
    """Connect an ordinary training configuration to the shared fit function."""
    def strategy(history):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            if constraints is None:
                fitted = fit_portfolio(history, regularization=LASSO,
                                       sample_weights=config.sample_weights)
            else:
                fitted = fit_portfolio_growth(
                    history, regularization=LASSO, sample_weights=config.sample_weights,
                    loss=LogGrowthLoss(reward_strength=0, exposure_strength=0), constraints=constraints)
        q = normalize_sample_weights(config.sample_weights(len(history)), len(history))
        records.append({"history_start": str(history.index[0].date()),
                        "history_end": str(history.index[-1].date()),
                        "observations": len(history),
                        "effective_observations": float(1 / (q @ q)),
                        "training_loss": fitted.training_loss, "objective": fitted.objective,
                        **(fitted.diagnostics or {})})
        if constraints is not None and len(records) % 63 == 0:
            print(f"  Fitted {len(records)} days; history through {history.index[-1].date()}", flush=True)
        return fitted.weights
    return strategy


def metrics(result, start=None, end=None):
    daily = result.returns.loc[start:end]
    gross = result.weights.loc[daily.index].abs().sum(axis=1)
    return {**summarize(daily), "average_gross_leverage": float(gross.mean()),
            "max_gross_leverage": float(gross.max()), "evaluation_days": len(daily),
            "start": str(daily.index[0].date()), "end": str(daily.index[-1].date())}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--start", default="2024-08-01", help="Evaluation begins after three years of warm-up")
    parser.add_argument("--end", default=VALIDATION_END)
    parser.add_argument("--presets", nargs="+", choices=TRAINING_PRESETS,
                        default=["252d", "3y", "3y_weighted"],
                        help="Choose training settings here; the model and backtest are unchanged")
    parser.add_argument("--constrained", action="store_true",
                        help="Also fit each preset with existing hard limits: gross<=1.5, training volatility<=EW, wealth floor=1e-6")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be an empty or new directory")
    try:
        for value in (args.start, args.end):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Dates must use complete YYYY-MM-DD format")
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if not pd.Timestamp(HISTORY_START) <= start <= end < pd.Timestamp(FINAL_TEST_START):
            raise ValueError("Evaluation must be ordered, inside the five-year history, and before the reserved final test (2026-08-01)")
        if len(set(args.presets)) != len(args.presets):
            raise ValueError("Training presets must be unique")
        returns = load_returns(args.csv, start=HISTORY_START, end=HISTORY_END)
        # Match the shared splitter's coverage check; do not silently truncate.
        if start < returns.index[0] or end > returns.index[-1]:
            raise ValueError("Evaluation range is outside available data")
        days = returns.loc[start:end].index
        if len(days) < 2:
            raise ValueError("Evaluation needs at least two trading days")
        start, end = days[0], days[-1]
        for preset in args.presets:
            _, first_history = next(rolling_windows(returns, start=start, end=end,
                                                    window=TRAINING_PRESETS[preset].window))
            if not np.isfinite(returns.loc[first_history.index[0]:end].to_numpy()).all():
                raise ValueError("Training/evaluation returns must be finite")
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"History: {HISTORY_START} to {HISTORY_END}; assets: {returns.shape[1]}; "
          f"evaluation: {start.date()} to {end.date()} ({len(days)} days)", flush=True)
    print("Fixed LASSO strength=1e-5; no CV, growth reward, exposure penalty or EW mixing. Daily Sharpe.", flush=True)
    if args.constrained:
        print("Adding constrained counterparts: gross<=1.5, training volatility<=EW, wealth floor=1e-6.", flush=True)
    rows, validation_rows, training_metadata = {}, {}, {}
    includes_validation = start <= pd.Timestamp(VALIDATION_START) and end >= pd.Timestamp(VALIDATION_END)
    variants = []
    for preset in args.presets:
        variants.append((f"LASSO_{preset}", TRAINING_PRESETS[preset], None))
        if args.constrained:
            variants.append((f"LASSO_{preset}_Constrained", TRAINING_PRESETS[preset], PortfolioConstraints()))
    for name, config, constraints in variants:
        records = []
        print(f"Running {name}...", flush=True)
        result = backtest(returns, lasso_strategy(config, records, constraints=constraints),
                          start=start, end=end, window=config.window)
        result.returns.to_csv(args.output / f"{name}_returns.csv")
        result.weights.to_csv(args.output / f"{name}_weights.csv")
        diagnostics = pd.DataFrame(records, index=result.returns.index)
        diagnostics.to_csv(args.output / f"{name}_diagnostics.csv")
        rows[name] = metrics(result)
        if includes_validation:
            validation_rows[name] = metrics(result, VALIDATION_START, VALIDATION_END)
        training_metadata[name] = {
            **asdict(config), "constraints": None if constraints is None else asdict(constraints),
            "minimum_actual_observations": int(diagnostics.observations.min()),
            "maximum_actual_observations": int(diagnostics.observations.max()),
            "first_fit": records[0], "last_fit": records[-1],
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.to_csv(args.output / "summary.csv")
    print("\nRequested evaluation period:\n" + summary.to_string(float_format=lambda x: f"{x:.6f}"))
    if validation_rows:
        validation = pd.DataFrame.from_dict(validation_rows, orient="index")
        validation.to_csv(args.output / "validation_summary.csv")
        print("\nOriginal January-June validation:\n" + validation.to_string(float_format=lambda x: f"{x:.6f}"))
    root = Path(__file__).resolve().parent
    code = [Path(__file__).resolve(), root / "requirements.txt", *sorted((root / "portfolio").glob("*.py"))]
    metadata = {
        "csv": str(args.csv.resolve()), "data_sha256": file_digest(args.csv),
        "history": {"start": HISTORY_START, "end": HISTORY_END,
                    "actual_start": str(returns.index[0].date()), "actual_end": str(returns.index[-1].date()),
                    "observations": len(returns)},
        "assets": list(returns.columns), "requested_start": args.start, "requested_end": args.end,
        "evaluation_start": str(start.date()), "evaluation_end": str(end.date()), "evaluation_days": len(days),
        "model": {"kind": "lasso", "lasso_strength": LASSO.strength, "sklearn_alpha": LASSO.strength / 2,
                  "hyperparameter_search": False, "budget": "sum(weights)=1",
                  "reward_strength": 0, "exposure_strength": 0, "post_fit_adjustment": None},
        "constrained_variants": args.constrained,
        "training": training_metadata,
        "weighting": "Exponential training-observation weights, normalized before fitting; evaluation returns are unweighted",
        "validation": {"start": VALIDATION_START, "end": VALIDATION_END,
                       "fully_evaluated": includes_validation},
        "final_test": {"start": FINAL_TEST_START, "end": FINAL_TEST_END, "evaluated": False},
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs; cumulative return compounds realized daily returns",
        "python": platform.python_version(),
        "packages": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "cvxpy", "clarabel", "scs")},
        "code_sha256": {str(path.relative_to(root)): file_digest(path) for path in code},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
