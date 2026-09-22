#!/usr/bin/env python3
"""Run guarded development or explicitly frozen final-test experiments."""

from portfolio.paths import PROJECT_ROOT
from portfolio.evaluation.artifact_store import configure_output, ensure_fresh_output
from portfolio.evaluation.provenance import file_digest

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.data.returns_loader import load_returns
from portfolio.evaluation.result_export import export_result


ROOT = PROJECT_ROOT
DEFAULT_CSV = ROOT / "100_Portfolios_10x10_Daily.csv"
HISTORY_START = "2021-08-01"
VALIDATION_END = "2026-06-30"
FINAL_TEST_START = "2026-08-01"
FINAL_TEST_END = "2026-08-30"
FINAL_REQUIRED_LAST_RETURN = "2026-08-28"


def _progress(message):
    print(message, flush=True)


def run_experiment(*args, **kwargs):
    from portfolio.engine.forecast_mixture_workflow import run_experiment as implementation
    return implementation(*args, **kwargs)


def run_frozen(*args, **kwargs):
    from portfolio.engine.forecast_mixture_workflow import run_frozen as implementation
    return implementation(*args, **kwargs)


def _guard_fresh_output(path):
    ensure_fresh_output(path)


def _load_complete(path, *, end, required_last):
    returns = load_returns(path, start=HISTORY_START, end=end)
    if returns.index[0] > pd.Timestamp("2021-08-02"):
        raise ValueError("history must begin by 2021-08-02")
    if returns.index[-1] < pd.Timestamp(required_last):
        raise ValueError(f"returns do not cover the required period through {required_last}")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("the required return history contains missing or infinite values")
    return returns


def _read_frozen(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen decisions: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("frozen decisions must be a JSON object")
    return value


def main(argv=None, *, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--frozen", type=Path,
        help="Explicitly evaluate August using previously exported frozen_decisions.json",
    )
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout,
                        help="Categorized artifact folders or historical flat filenames")
    args = parser.parse_args(argv)
    configure_output(parser, args, "lasso_forecast_mixture", optional_flat=False)
    try:
        _guard_fresh_output(args.output)
        if not args.csv.is_file():
            raise ValueError(f"CSV file does not exist: {args.csv}")
        if args.frozen is None:
            returns = _load_complete(
                args.csv, end=VALIDATION_END, required_last=VALIDATION_END
            )
            print("Running development experiment through 2026-06-30; August is excluded.", flush=True)
            result = run_experiment(returns, progress=_progress, include_baselines=True)
            result.metadata["phase"] = "development"
            result.metadata["final_test"] = {
                "start": FINAL_TEST_START, "end": FINAL_TEST_END, "evaluated": False,
            }
        else:
            frozen = _read_frozen(args.frozen)
            returns = _load_complete(
                args.csv, end=FINAL_TEST_END, required_last=FINAL_REQUIRED_LAST_RETURN
            )
            print("Running explicit frozen August final-test evaluation.", flush=True)
            result = run_frozen(
                returns, frozen, start=FINAL_TEST_START, end=FINAL_TEST_END, progress=_progress
            )
            result.metadata["phase"] = "final_test"
            result.metadata["final_test"] = {
                "start": FINAL_TEST_START, "end": FINAL_TEST_END, "evaluated": True,
            }
        export_result(result, args.output, source_csv=args.csv, layout=args.layout)
        print(f"Saved artifacts to {args.output.resolve()}", flush=True)
        return result
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main(default_layout="categorized")
