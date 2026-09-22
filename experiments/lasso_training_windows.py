"""Compare fixed LASSO windows, optionally adding the existing hard constraints."""

from portfolio.paths import PROJECT_ROOT
from portfolio.evaluation.artifact_store import configure_output
from portfolio.evaluation.provenance import file_digest, run_provenance

import argparse
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio import PortfolioConstraints, backtest, fit_portfolio_growth, load_returns
from portfolio.config import TRAINING_PRESETS
from portfolio.data.history_windows import rolling_windows


from portfolio.engine.lasso_window_strategy import (
    FINAL_TEST_START, FINAL_TEST_END, HISTORY_START, HISTORY_END,
    VALIDATION_START, VALIDATION_END, LASSO, lasso_strategy, metrics,
)
from portfolio.evaluation.provenance import file_digest


def main(argv=None, *, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--start", default="2024-08-01", help="Evaluation begins after three years of warm-up")
    parser.add_argument("--end", default=VALIDATION_END)
    parser.add_argument("--presets", nargs="+", choices=TRAINING_PRESETS,
                        default=["252d", "3y", "3y_weighted"],
                        help="Choose training settings here; the model and backtest are unchanged")
    parser.add_argument("--constrained", action="store_true",
                        help="Also fit each preset with existing hard limits: gross<=1.5, training volatility<=EW, wealth floor=1e-6")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout,
                        help="Categorized artifact folders or historical flat filenames")
    args = parser.parse_args(argv)
    configure_output(parser, args, "lasso_training_windows", optional_flat=False)
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
        result.returns.to_csv(args.artifacts.path(f"{name}_returns.csv"))
        result.weights.to_csv(args.artifacts.path(f"{name}_weights.csv"))
        diagnostics = pd.DataFrame(records, index=result.returns.index)
        diagnostics.to_csv(args.artifacts.path(f"{name}_diagnostics.csv"))
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
    summary.to_csv(args.artifacts.path("summary.csv"))
    print("\nRequested evaluation period:\n" + summary.to_string(float_format=lambda x: f"{x:.6f}"))
    if validation_rows:
        validation = pd.DataFrame.from_dict(validation_rows, orient="index")
        validation.to_csv(args.artifacts.path("validation_summary.csv"))
        print("\nOriginal January-June validation:\n" + validation.to_string(float_format=lambda x: f"{x:.6f}"))
    metadata = {
        **run_provenance(args.csv),
        "artifact_layout": args.layout, "artifact_layout_version": 1,
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
    }
    (args.artifacts.path("run_metadata.json")).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    if args.artifacts is not None:
        args.artifacts.finalize()
        print(f"Saved artifacts to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main(default_layout="categorized")
