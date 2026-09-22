"""Select the recency half-life for three-year constrained LASSO on validation."""

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import date
from importlib.metadata import version
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from portfolio import PortfolioConstraints, backtest, load_returns
from portfolio.config import TRAINING_PRESETS
from portfolio.splits import rolling_windows
from run_lasso_windows import (
    FINAL_TEST_START, FINAL_TEST_END, HISTORY_START, HISTORY_END,
    VALIDATION_START, VALIDATION_END, LASSO, file_digest, lasso_strategy, metrics,
)


def select_half_life(scores):
    """Maximize finite validation cumulative return; exact ties prefer slower decay.

    Only candidate scores belong here. Controls are reported but not searched.
    The returned half-life is chosen AFTER validation, never per-day using
    returns that have not yet occurred.
    """
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("Need at least one candidate score")
    for half_life, score in scores.items():
        if isinstance(half_life, bool) or not isinstance(half_life, Integral) or half_life <= 0:
            raise ValueError("Half-lives must be positive integers in trading observations")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
            raise ValueError("Every candidate must have a finite cumulative return")
    return int(max(scores, key=lambda half_life: (scores[half_life], half_life)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--validation-start", default=VALIDATION_START)
    parser.add_argument("--validation-end", default=VALIDATION_END)
    parser.add_argument("--half-lives", type=int, nargs="+", default=[126, 252, 504],
                        help="Predeclared positive trading-observation half-lives; no adaptive grid expansion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be an empty or new directory")
    try:
        for value in (args.validation_start, args.validation_end):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Dates must use complete YYYY-MM-DD format")
        start, end = pd.Timestamp(args.validation_start), pd.Timestamp(args.validation_end)
        if not pd.Timestamp(VALIDATION_START) <= start <= end <= pd.Timestamp(VALIDATION_END):
            raise ValueError(f"Selection must stay within validation ({VALIDATION_START} to {VALIDATION_END}); July and final testing are excluded")
        if len(set(args.half_lives)) != len(args.half_lives) or any(h <= 0 for h in args.half_lives):
            raise ValueError("Half-lives must be unique positive integers")
        candidates = {h: replace(TRAINING_PRESETS["3y"], half_life=h) for h in args.half_lives}
        # Controls are outside selection; they cannot change the chosen half-life.
        variants = [
            ("LASSO_252d_Constrained", TRAINING_PRESETS["252d"], None),
            ("LASSO_3y_Constrained", TRAINING_PRESETS["3y"], None),
            *[(f"LASSO_3y_HL{h}_Constrained", config, h) for h, config in candidates.items()],
        ]
        returns = load_returns(args.csv, start=HISTORY_START, end=HISTORY_END)
        if start < returns.index[0] or end > returns.index[-1]:
            raise ValueError("Validation range is outside available data")
        days = returns.loc[start:end].index
        if len(days) < 2:
            raise ValueError("Validation needs at least two trading days")
        start, end = days[0], days[-1]
        for _, config, _ in variants:
            _, first_history = next(rolling_windows(returns, start=start, end=end, window=config.window))
            if not np.isfinite(returns.loc[first_history.index[0]:end].to_numpy()).all():
                raise ValueError("Training/validation returns must be finite")
    except (ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))

    constraints = PortfolioConstraints()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Validation: {start.date()} to {end.date()} ({len(days)} days); candidates: {args.half_lives}", flush=True)
    print("Three calendar years, fixed LASSO=1e-5, existing hard constraints; reward=0, exposure penalty=0.", flush=True)
    print("Select by validation cumulative return. These are selection-period scores, not final test results.", flush=True)
    rows, training_metadata, scores = {}, {}, {}
    for name, config, half_life in variants:
        records = []
        print(f"Running {name}...", flush=True)
        result = backtest(returns, lasso_strategy(config, records, constraints=constraints),
                          start=start, end=end, window=config.window)
        result.returns.to_csv(args.output / f"{name}_returns.csv")
        result.weights.to_csv(args.output / f"{name}_weights.csv")
        diagnostics = pd.DataFrame(records, index=result.returns.index)
        diagnostics.to_csv(args.output / f"{name}_diagnostics.csv")
        rows[name] = {**metrics(result), "role": "control" if half_life is None else "candidate",
                      "half_life": half_life}
        training_metadata[name] = {
            **asdict(config), "minimum_actual_observations": int(diagnostics.observations.min()),
            "maximum_actual_observations": int(diagnostics.observations.max()),
            "mean_effective_observations": float(diagnostics.effective_observations.mean()),
            "first_fit": records[0], "last_fit": records[-1],
        }
        if half_life is not None:
            scores[half_life] = rows[name]["cumulative_return"]

    # Every candidate has completed the same period before any selection occurs.
    selected_half_life = select_half_life(scores)
    selected_name = f"LASSO_3y_HL{selected_half_life}_Constrained"
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary["selected"] = summary.index == selected_name
    selection = {
        "selected_strategy": selected_name, "selected_half_life": selected_half_life,
        "training": asdict(candidates[selected_half_life]), "constraints": asdict(constraints),
        "lasso_strength": LASSO.strength, "reward_strength": 0, "exposure_strength": 0,
        "selection_metric": "cumulative_return", "selected_score": scores[selected_half_life],
        "tie_break": "longer half-life on exactly equal cumulative returns",
        "validation_start": str(start.date()), "validation_end": str(end.date()),
        "evaluation_role": "validation_used_for_selection",
        "deployment_note": "Freeze this setting after validation; its validation returns are not an independent test.",
    }
    root = Path(__file__).resolve().parent
    code = [Path(__file__).resolve(), root / "run_lasso_windows.py", root / "requirements.txt",
            *sorted((root / "portfolio").glob("*.py"))]
    metadata = {
        "csv": str(args.csv.resolve()), "data_sha256": file_digest(args.csv),
        "history": {"start": HISTORY_START, "end": HISTORY_END,
                    "actual_start": str(returns.index[0].date()), "actual_end": str(returns.index[-1].date()),
                    "observations": len(returns)},
        "assets": list(returns.columns), "validation_days": len(days),
        "requested_validation_start": args.validation_start, "requested_validation_end": args.validation_end,
        "candidate_half_lives": args.half_lives, "selection": selection,
        "model": {"kind": "lasso", "lasso_strength": LASSO.strength,
                  "reward_strength": 0, "exposure_strength": 0, "post_fit_adjustment": None},
        "constraints": asdict(constraints), "training": training_metadata,
        "weighting": "Candidates: exponential sample weights with the stated trading-observation half-life; controls: uniform. Training loss and EW risk limit use the same normalized weights.",
        "evaluation_weighting": "Unweighted realized daily portfolio returns; cumulative return compounds them",
        "search_scope": "Only half-life; fixed candidate grid, no early stopping or adaptive expansion; controls excluded from selection",
        "final_test": {"start": FINAL_TEST_START, "end": FINAL_TEST_END, "evaluated": False},
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "python": platform.python_version(),
        "packages": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "cvxpy", "clarabel", "scs")},
        "code_sha256": {str(path.relative_to(root)): file_digest(path) for path in code},
    }
    summary.to_csv(args.output / "summary.csv")
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False)+"\n")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"Selected half-life: {selected_half_life} trading observations; validation cumulative return: {scores[selected_half_life]:.4%}")


if __name__ == "__main__":
    main()
