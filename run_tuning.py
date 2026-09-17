"""Select two loss coefficients for a constrained, fixed-LASSO portfolio."""

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from portfolio import (LogGrowthLoss, PortfolioConstraints, Regularization,
                       backtest, fit_portfolio, fit_portfolio_growth, load_returns, summarize)
from portfolio.selection import COARSE_PARAMETERS, GrowthParameters, SearchConfig, search_growth
from portfolio.splits import rolling_windows


FINAL_TEST_START = "2026-08-01"
FINAL_TEST_END = "2026-08-30"


def metrics(result):
    gross = result.weights.abs().sum(axis=1)
    return {**summarize(result.returns), "average_gross_leverage": float(gross.mean()),
            "max_gross_leverage": float(gross.max())}


def save_fit(directory, name, result, diagnostics=None):
    directory.mkdir(parents=True, exist_ok=True)
    result.returns.to_csv(directory / f"{name}_returns.csv")
    result.weights.to_csv(directory / f"{name}_weights.csv")
    if diagnostics is not None:
        diagnostics.to_csv(directory / f"{name}_diagnostics.csv")


class PhaseEvaluator:
    """Fit only past windows; cache complete candidate backtests within one phase."""

    def __init__(self, returns, *, start, end, window, output, constraints):
        self.returns = returns.loc[:end].copy()
        self.requested_start, self.requested_end = start, end
        days = self.returns.loc[start:end].index
        self.start, self.end = str(days[0].date()), str(days[-1].date())
        self.window = window
        self.output, self.constraints = output, constraints
        self.cache = {}
        self.failures = {}
        self.attempts = []
        self.calls = 0

    def fit(self, parameters):
        if parameters in self.cache:
            return self.cache[parameters]
        if parameters in self.failures:
            raise RuntimeError(self.failures[parameters])
        records = []

        def strategy(history):
            fitted = fit_portfolio_growth(history, regularization=Regularization("lasso", 1e-5),
                loss=LogGrowthLoss(**asdict(parameters)), constraints=self.constraints)
            records.append({"history_end": history.index[-1], **fitted.diagnostics})
            return fitted.weights

        try:
            result = backtest(self.returns, strategy, start=self.start, end=self.end, window=self.window)
            if (result.returns <= -1).any():
                raise ValueError("Candidate has nonpositive realized wealth in the evaluation period")
            if not np.isfinite(metrics(result)["cumulative_return"]):
                raise ValueError("Candidate has a nonfinite cumulative return")
        except (RuntimeError, ValueError) as error:
            self.failures[parameters] = f"{type(error).__name__}: {error}"
            raise
        diagnostics = pd.DataFrame(records, index=result.returns.index)
        self.cache[parameters] = (result, diagnostics)
        return result, diagnostics

    def __call__(self, parameters):
        self.calls += 1
        print(f"{self.output.name} candidate {self.calls}: eta={parameters.reward_strength:g}, "
              f"gamma={parameters.exposure_strength:g}", flush=True)
        record = {"candidate_id": self.calls, **asdict(parameters), "status": "ok", "error": ""}
        try:
            result, diagnostics = self.fit(parameters)
            values = metrics(result)
            record.update(values)
            save_fit(self.output, f"candidate_{self.calls:02d}", result, diagnostics)
            return values
        except (RuntimeError, ValueError) as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            # Preserve failures even if no candidate succeeds and search raises.
            self.attempts.append(record)
            self.output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.attempts).to_csv(self.output / "candidate_attempts.csv", index=False)


def unique_parameters(values):
    return list(dict.fromkeys(values))


def phase_metadata(result, evaluator, config):
    return {"start": evaluator.start, "end": evaluator.end,
            "requested_start": evaluator.requested_start, "requested_end": evaluator.requested_end,
            "evaluation_days": len(evaluator.returns.loc[evaluator.start:evaluator.end]),
            "evaluations": len(result.history), "settings": asdict(config),
            "selected_parameters": asdict(result.selected),
            "raw_best_parameters": asdict(result.raw_best),
            "stop_reason": result.stop_reason, "completed_rounds": result.rounds}


def write_metadata(args, returns, constraints, screen, validation, screen_eval, validation_eval,
                   screen_config, validation_config, comparison_parameters, comparison_failures):
    root = Path(__file__).resolve().parent
    searched = set()
    for result in (screen, validation):
        searched.update(zip(result.history["reward_strength"], result.history["exposure_strength"]))
    digest = hashlib.sha256()
    with args.csv.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    code = [root / "run_tuning.py", root / "requirements.txt", *sorted((root / "portfolio").glob("*.py"))]
    metadata = {
        "evaluation_role": "development_validation", "selection_metric": "cumulative_return",
        "course_metric": "negative_daily_sharpe", "window": args.window, "lasso_strength": 1e-5,
        "sample_weights": "uniform", "post_fit_adjustment": None,
        "search_space": {"coarse_parameters": [asdict(value) for value in COARSE_PARAMETERS],
                         "reward_max": .03, "exposure_max": 3,
                         "neighbors": "halve/double one positive coordinate; from zero use half the smallest positive coarse value",
                         "exposure_scale": "training EW variance (normalized sample weights, ddof=0)"},
        "constraints": asdict(constraints), "selected_parameters": asdict(validation.selected),
        "raw_best_parameters": asdict(validation.raw_best),
        "screen": phase_metadata(screen, screen_eval, screen_config),
        "validation": phase_metadata(validation, validation_eval, validation_config),
        "distinct_search_configurations": len(searched), "distinct_search_budget": args.max_candidates,
        "total_search_evaluations": len(screen.history) + len(validation.history),
        "comparison_parameters": comparison_parameters,
        "comparison_failures": comparison_failures,
        "comparison_ablations_used_for_selection": False,
        "csv": str(args.csv.resolve()), "data_sha256": digest.hexdigest(),
        "data_first_date": str(returns.index[0].date()), "data_last_date": str(returns.index[-1].date()),
        "assets": list(returns.columns),
        "code_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in code},
        "python": platform.python_version(),
        "packages": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "cvxpy", "clarabel", "scs")},
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "final_test": {"start": FINAL_TEST_START, "end": FINAL_TEST_END, "evaluated": False,
                       "note": "Final testing requires complete data and frozen settings; this command only selects on development periods."},
    }
    if len(searched) > args.max_candidates:
        raise RuntimeError("Distinct search budget exceeded")
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--screen-start", default="2025-07-01")
    parser.add_argument("--screen-end", default="2025-12-31")
    parser.add_argument("--validation-start", default="2026-01-02")
    parser.add_argument("--validation-end", default="2026-06-30")
    parser.add_argument("--max-candidates", type=int, default=15,
                        help="Maximum distinct searched pairs across both phases (9 to 15)")
    parser.add_argument("--min-improvement", type=float, default=.002,
                        help="Required cumulative-return improvement; .002 means 0.2 percentage points")
    parser.add_argument("--patience", type=int, default=1,
                        help="Stop after this many complete refinement batches without significant improvement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be an empty or new directory")
    try:
        if not 9 <= args.max_candidates <= 15:
            raise ValueError("--max-candidates must be between 9 and 15")
        date_strings = [args.screen_start, args.screen_end, args.validation_start, args.validation_end]
        try:
            if any(date.fromisoformat(value).isoformat() != value for value in date_strings):
                raise ValueError
        except ValueError as error:
            raise ValueError("Dates must use complete YYYY-MM-DD format") from error
        dates = pd.to_datetime(date_strings)
        if dates.isna().any() or not dates[0] <= dates[1] < dates[2] <= dates[3]:
            raise ValueError("Require screen_start <= screen_end < validation_start <= validation_end")
        if dates[3] >= pd.Timestamp(FINAL_TEST_START):
            raise ValueError(f"Validation must end before the reserved final test begins ({FINAL_TEST_START})")
        screen_config = SearchConfig(max_candidates=9, min_improvement=args.min_improvement, patience=args.patience)
        returns = load_returns(args.csv)
        for start, end in ((args.screen_start, args.screen_end), (args.validation_start, args.validation_end)):
            evaluation = returns.loc[start:end]
            if len(evaluation) < 2:
                raise ValueError("Each evaluation phase needs at least two trading days")
            # Validate the splitter before starting a run: coverage and window errors
            # must not leave a directory that resembles a finished experiment.
            _, first_history = next(rolling_windows(returns, start=start, end=end, window=args.window))
            relevant = returns.loc[first_history.index[0]:end].to_numpy(dtype=float)
            if not np.isfinite(relevant).all():
                raise ValueError("Training/evaluation returns must be finite")
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    constraints = PortfolioConstraints()
    args.output.mkdir(parents=True, exist_ok=True)
    screen_eval = PhaseEvaluator(returns, start=args.screen_start, end=args.screen_end,
        window=args.window, output=args.output / "screen", constraints=constraints)
    screen = search_growth(screen_eval, config=screen_config)
    screen.history.to_csv(args.output / "screen_search.csv", index=False)
    chosen = screen.selected
    initial = unique_parameters([GrowthParameters(0, 0), GrowthParameters(.01, 0), chosen,
                                 GrowthParameters(chosen.reward_strength, 0),
                                 GrowthParameters(0, chosen.exposure_strength)])
    validation_config = SearchConfig(max_candidates=len(initial) + args.max_candidates - len(COARSE_PARAMETERS),
                                    min_improvement=args.min_improvement, patience=args.patience)
    validation_eval = PhaseEvaluator(returns, start=args.validation_start, end=args.validation_end,
        window=args.window, output=args.output / "validation", constraints=constraints)
    validation = search_growth(validation_eval, config=validation_config, initial_parameters=initial)
    validation.history.to_csv(args.output / "validation_search.csv", index=False)
    selected = validation.selected  # Freeze before evaluating descriptive ablations.
    print(f"Selected eta={selected.reward_strength:g}, gamma={selected.exposure_strength:g}; "
          f"stop={validation.stop_reason}", flush=True)
    comparisons = {
        "LASSO_Constrained": GrowthParameters(0, 0),
        "LASSO_LogGrowth": GrowthParameters(.01, 0),
        "Selected_RewardOnly": GrowthParameters(selected.reward_strength, 0),
        "Selected_ExposureOnly": GrowthParameters(0, selected.exposure_strength),
        "LASSO_Selected": selected,
    }
    rows = {}
    comparison_failures = {}
    plain = {"EW": lambda history: pd.Series(1 / history.shape[1], index=history.columns),
             "LASSO": lambda history: fit_portfolio(history, regularization=Regularization("lasso", 1e-5)).weights}
    for name, strategy in plain.items():
        result = backtest(validation_eval.returns, strategy, start=validation_eval.start,
                          end=validation_eval.end, window=args.window)
        rows[name] = {"status": "ok", "error": "", **metrics(result)}
        save_fit(args.output, name, result)
    for name, parameters in comparisons.items():
        try:
            result, diagnostics = validation_eval.fit(parameters)
        except (RuntimeError, ValueError) as error:
            comparison_failures[name] = f"{type(error).__name__}: {error}"
            rows[name] = {"status": "failed", "error": comparison_failures[name]}
        else:
            rows[name] = {"status": "ok", "error": "", **metrics(result)}
            save_fit(args.output, name, result, diagnostics)
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.to_csv(args.output / "summary.csv")
    write_metadata(args, returns, constraints, screen, validation, screen_eval, validation_eval,
                   screen_config, validation_config, {name: asdict(value) for name, value in comparisons.items()},
                   comparison_failures)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
