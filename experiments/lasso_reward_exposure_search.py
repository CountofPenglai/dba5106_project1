"""Select two loss coefficients for a constrained, fixed-LASSO portfolio."""

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

from portfolio import PortfolioConstraints, Regularization, backtest, fit_portfolio, fit_portfolio_growth, load_returns
from portfolio.engine.parameter_search import COARSE_PARAMETERS, GrowthParameters, SearchConfig, search_growth
from portfolio.data.history_windows import rolling_windows


from portfolio.engine.growth_search_evaluation import PhaseEvaluator, save_fit, metrics

FINAL_TEST_START = "2026-08-01"
FINAL_TEST_END = "2026-08-30"


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
    searched = set()
    for result in (screen, validation):
        searched.update(zip(result.history["reward_strength"], result.history["exposure_strength"]))
    metadata = {
        **run_provenance(args.csv),
        "artifact_layout": args.layout, "artifact_layout_version": 1,
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
        "data_first_date": str(returns.index[0].date()), "data_last_date": str(returns.index[-1].date()),
        "assets": list(returns.columns),
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "final_test": {"start": FINAL_TEST_START, "end": FINAL_TEST_END, "evaluated": False,
                       "note": "Final testing requires complete data and frozen settings; this command only selects on development periods."},
    }
    if len(searched) > args.max_candidates:
        raise RuntimeError("Distinct search budget exceeded")
    (args.artifacts.path("run_metadata.json")).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main(argv=None, *, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv")
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout,
                        help="Categorized artifact folders or historical flat filenames")
    args = parser.parse_args(argv)
    configure_output(parser, args, "lasso_reward_exposure_search", optional_flat=False)
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
        window=args.window, output=(args.output / "search" / "screen" if args.layout == "categorized" else args.output / "screen"), constraints=constraints)
    screen = search_growth(screen_eval, config=screen_config)
    screen.history.to_csv(args.artifacts.path("screen_search.csv"), index=False)
    chosen = screen.selected
    initial = unique_parameters([GrowthParameters(0, 0), GrowthParameters(.01, 0), chosen,
                                 GrowthParameters(chosen.reward_strength, 0),
                                 GrowthParameters(0, chosen.exposure_strength)])
    validation_config = SearchConfig(max_candidates=len(initial) + args.max_candidates - len(COARSE_PARAMETERS),
                                    min_improvement=args.min_improvement, patience=args.patience)
    validation_eval = PhaseEvaluator(returns, start=args.validation_start, end=args.validation_end,
        window=args.window, output=(args.output / "search" / "validation" if args.layout == "categorized" else args.output / "validation"), constraints=constraints)
    validation = search_growth(validation_eval, config=validation_config, initial_parameters=initial)
    validation.history.to_csv(args.artifacts.path("validation_search.csv"), index=False)
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
        save_fit(args.output, name, result, layout=args.layout)
    for name, parameters in comparisons.items():
        try:
            result, diagnostics = validation_eval.fit(parameters)
        except (RuntimeError, ValueError) as error:
            comparison_failures[name] = f"{type(error).__name__}: {error}"
            rows[name] = {"status": "failed", "error": comparison_failures[name]}
        else:
            rows[name] = {"status": "ok", "error": "", **metrics(result)}
            save_fit(args.output, name, result, diagnostics, layout=args.layout)
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.to_csv(args.artifacts.path("summary.csv"))
    write_metadata(args, returns, constraints, screen, validation, screen_eval, validation_eval,
                   screen_config, validation_config, {name: asdict(value) for name, value in comparisons.items()},
                   comparison_failures)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

    if args.artifacts is not None:
        args.artifacts.finalize()
        print(f"Saved artifacts to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main(default_layout="categorized")
