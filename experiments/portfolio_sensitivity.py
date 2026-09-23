"""Validation-only comparisons of four prespecified constrained-LASSO variants.

The model math lives in portfolio.models.portfolio_variants. This module owns
rolling evaluation, comparable metrics, output layout and provenance.
"""
import argparse
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio import Regularization, backtest, load_returns, summarize
from portfolio.data.history_windows import rolling_windows
from portfolio.evaluation.artifact_store import configure_output
from portfolio.evaluation.provenance import run_provenance
from portfolio.models.portfolio_variants import (
    ExperimentConstraints, ExperimentLoss, fit_portfolio_variant, size_bm_grid_edges,
)
from portfolio.paths import PROJECT_ROOT

FINAL_TEST_START = "2026-08-01"
LASSO = Regularization("lasso", 1e-5)
SELECTED_LOSS = ExperimentLoss(reward_strength=0, exposure_strength=2)


def _label(prefix, value):
    return f"{prefix}_{value:g}".replace(".", "p")


def _turnover(weights):
    initial = np.full(weights.shape[1], 1 / weights.shape[1])
    before = np.vstack([initial, weights.to_numpy()[:-1]])
    return pd.Series(.5 * np.abs(weights.to_numpy() - before).sum(axis=1),
                     index=weights.index, name="turnover")


def _neighbor_gap(weights, edges):
    pairs = np.asarray(edges, dtype=int)
    values = weights.to_numpy()
    return pd.Series(np.abs(values[:, pairs[:, 0]] - values[:, pairs[:, 1]]).mean(axis=1),
                     index=weights.index, name="neighbor_gap")


def _concentration(weights):
    values = weights.to_numpy()
    return pd.DataFrame({
        "max_abs_weight": np.abs(values).max(axis=1),
        "effective_assets": 1 / np.square(values).sum(axis=1),
        "top10_weight": np.sort(values, axis=1)[:, -min(10, values.shape[1]):].sum(axis=1),
    }, index=weights.index)


def _metrics(result, mode, edges):
    gross = result.weights.abs().sum(axis=1)
    row = {**summarize(result.returns), "average_gross_leverage": float(gross.mean()),
           "max_gross_leverage": float(gross.max())}
    if mode in ("turnover", "shrinkage"):
        movement = _turnover(result.weights)
        row.update(average_turnover=float(movement.mean()),
                   max_turnover=float(movement.max()), total_turnover=float(movement.sum()))
    if mode == "smoothness":
        gaps = _neighbor_gap(result.weights, edges)
        row.update(average_neighbor_gap=float(gaps.mean()), max_neighbor_gap=float(gaps.max()))
    if mode == "concentration":
        concentration = _concentration(result.weights)
        row.update(average_max_abs_weight=float(concentration.max_abs_weight.mean()),
                   max_abs_weight=float(concentration.max_abs_weight.max()),
                   average_effective_assets=float(concentration.effective_assets.mean()),
                   average_top10_weight=float(concentration.top10_weight.mean()))
    return row


def _run_variant(returns, *, start, end, window, loss, constraints, estimator, edges,
                 carry_previous):
    records = []
    previous = None

    def strategy(history):
        nonlocal previous
        fitted = fit_portfolio_variant(
            history, regularization=LASSO, loss=loss, constraints=constraints,
            previous_weights=previous if carry_previous else None,
            neighbor_edges=edges if edges else None, covariance_estimator=estimator)
        if carry_previous:
            previous = fitted.weights.copy()
        records.append({"history_end": history.index[-1], **fitted.diagnostics})
        return fitted.weights

    result = backtest(returns, strategy, start=start, end=end, window=window)
    diagnostics = pd.DataFrame(records, index=result.returns.index)
    return result, diagnostics


def _save(store, name, result, mode, edges, diagnostics=None):
    store.write_backtest(name, result, diagnostics)
    if mode in ("turnover", "shrinkage"):
        _turnover(result.weights).to_csv(store.path(f"{name}_turnover.csv"))
    if mode == "smoothness":
        _neighbor_gap(result.weights, edges).to_csv(store.path(f"{name}_neighbor_gap.csv"))
    if mode == "concentration":
        _concentration(result.weights).to_csv(store.path(f"{name}_concentration.csv"))


def _plot(store, mode, results, summary):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, result in results.items():
        axes[0].plot(result.returns.index, 100 * ((1 + result.returns).cumprod() - 1),
                     label=name, linewidth=1.6)
    axes[0].set(title="Cumulative validation return", ylabel="Percent")
    axes[0].legend(fontsize=7)
    axes[1].bar(range(len(summary)), summary.sharpe)
    axes[1].set_xticks(range(len(summary)), summary.index, rotation=35, ha="right", fontsize=8)
    axes[1].set(title="Daily Sharpe ratio")
    for axis in axes:
        axis.grid(alpha=.25)
    figure.suptitle(mode.replace('_', ' ').title() + ": 252-day sensitivity")
    figure.savefig(store.path(f"{mode}_comparison.png"), dpi=160)
    plt.close(figure)


def _settings(mode, args):
    basic = ExperimentConstraints()
    if mode == "turnover":
        return [("LASSO_Selected" if strength == 0 else _label("LASSO_Turnover", strength),
                 ExperimentLoss(0, 2, turnover_strength=strength), basic, "sample", True)
                for strength in args.turnover_strengths]
    if mode == "smoothness":
        return [("LASSO_Selected" if strength == 0 else _label("LASSO_Smooth", strength),
                 ExperimentLoss(0, 2, smoothness_strength=strength), basic, "sample", False)
                for strength in args.smoothness_strengths]
    if mode == "shrinkage":
        return [("LASSO_Selected", SELECTED_LOSS, basic, "sample", False),
                ("LASSO_LedoitWolf", SELECTED_LOSS, basic, "ledoit_wolf", False)]
    label = _label("LASSO_WeightCap", args.weight_limit * 100) + "pct"
    return [("LASSO_Selected", SELECTED_LOSS, basic, "sample", False),
            (label, SELECTED_LOSS, ExperimentConstraints(weight_limit=args.weight_limit),
             "sample", False)]


def main(mode, argv=None, *, default_layout="categorized"):
    if mode not in ("turnover", "smoothness", "shrinkage", "concentration"):
        raise ValueError("Unknown sensitivity direction")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--turnover-strengths", nargs="+", type=float, default=[0, .1, 1])
    parser.add_argument("--smoothness-strengths", nargs="+", type=float, default=[0, .1, 1])
    parser.add_argument("--weight-limit", type=float, default=.10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout)
    args = parser.parse_args(argv)
    configure_output(parser, args, mode, optional_flat=False)
    try:
        if args.window < 2:
            raise ValueError("Window must contain at least two observations")
        for value in (args.start, args.end):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Dates must be complete YYYY-MM-DD values")
        if args.start > args.end or args.end >= FINAL_TEST_START:
            raise ValueError("Evaluation dates must be ordered and end before 2026-08-01")
        for strengths in (args.turnover_strengths, args.smoothness_strengths):
            if (not strengths or len(strengths) != len(set(strengths))
                    or any(not np.isfinite(value) or value < 0 for value in strengths)):
                raise ValueError("Penalty strengths must be unique, finite and nonnegative")
        if not np.isfinite(args.weight_limit) or args.weight_limit <= 0:
            raise ValueError("Weight limit must be finite and positive")
        returns = load_returns(args.csv)
        days = returns.loc[args.start:args.end].index
        if len(days) < 2:
            raise ValueError("Evaluation needs at least two trading days")
        _, first_history = next(rolling_windows(returns, start=args.start, end=args.end,
                                                window=args.window))
        relevant = returns.loc[first_history.index[0]:args.end].to_numpy()
        if not np.isfinite(relevant).all():
            raise ValueError("Training and validation returns must be finite")
        if mode == "concentration" and args.weight_limit * returns.shape[1] < 1 - 1e-12:
            raise ValueError("Weight limit is infeasible for full investment")
        edges = size_bm_grid_edges(returns.columns) if mode == "smoothness" else None
    except (ValueError, TypeError, StopIteration) as error:
        parser.error(str(error))

    start, end = str(days[0].date()), str(days[-1].date())
    # The fitting/backtest layer receives no post-validation observations.
    evaluation_returns = returns.loc[:end].copy()
    store = args.artifacts.prepare()
    results, rows, fit_diagnostics = {}, {}, {}
    equal_weight = lambda history: pd.Series(1 / history.shape[1], index=history.columns)
    ew = backtest(evaluation_returns, equal_weight, start=start, end=end, window=args.window)
    results["EW"], rows["EW"] = ew, _metrics(ew, mode, edges)
    _save(store, "EW", ew, mode, edges)

    for name, loss, constraints, estimator, carry_previous in _settings(mode, args):
        print(f"Running {mode}: {name}", flush=True)
        result, diagnostics = _run_variant(
            evaluation_returns, start=start, end=end, window=args.window,
            loss=loss, constraints=constraints, estimator=estimator,
            edges=edges, carry_previous=carry_previous)
        results[name], rows[name], fit_diagnostics[name] = result, _metrics(result, mode, edges), diagnostics
        if mode == "turnover" and not np.allclose(diagnostics.turnover, _turnover(result.weights), atol=1e-8):
            raise RuntimeError("Recorded turnover differs from saved weights")
        if mode == "smoothness" and not np.allclose(diagnostics.neighbor_smoothness,
                                                     _neighbor_gap(result.weights, edges), atol=1e-8):
            raise RuntimeError("Recorded grid gap differs from saved weights")
        _save(store, name, result, mode, edges, diagnostics)
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.to_csv(store.path("summary.csv"))
    _plot(store, mode, results, summary)
    metadata = {
        **run_provenance(args.csv),
        "experiment": mode, "artifact_layout": args.layout, "artifact_layout_version": 1,
        "evaluation_role": "development_validation_sensitivity",
        "selection_metric": None, "course_metric": "negative_daily_sharpe",
        "window": args.window, "requested_start": args.start, "requested_end": args.end,
        "actual_start": start, "actual_end": end, "evaluation_days": len(days),
        "assets": list(returns.columns), "grid_edges": len(edges) if edges else None,
        "model": "fully invested EW-minus-basis LASSO", "lasso_strength": LASSO.strength,
        "selected_loss": asdict(SELECTED_LOSS),
        "variants": {name: {"loss": asdict(loss), "constraints": asdict(constraints),
                            "covariance_estimator": estimator, "previous_target_used": carry_previous}
                     for name, loss, constraints, estimator, carry_previous in _settings(mode, args)},
        "metric_convention": "Daily Sharpe=mean/std(ddof=1); no risk-free rate or transaction costs",
        "final_test": {"start": FINAL_TEST_START, "evaluated": False},
    }
    if mode == "shrinkage":
        metadata["ledoit_wolf_shrinkage_mean"] = float(
            fit_diagnostics["LASSO_LedoitWolf"].ledoit_wolf_shrinkage.mean())
    store.path("run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    store.finalize()
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)
    print(f"Saved artifacts to {store.root.resolve()}", flush=True)
    return summary
