#!/usr/bin/env python3
"""Independent numerical audit of exported direct-asset LP results.

This script intentionally does not import portfolio.linear_decision (or any
other portfolio module).  It rebuilds the source-return frame, risk histories,
LP proxy values, realized returns, turnover, and performance summaries from
the exported artifacts.  A small deterministic sample of primary programs is
also re-solved with CVXPY/CLARABEL using a separate affine formulation.
"""

from __future__ import annotations

import argparse
import hashlib
from io import StringIO
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "run_direct_lp.py").is_file() and (candidate / "portfolio").is_dir():
            return candidate
    raise RuntimeError("cannot locate project root containing run_direct_lp.py")


ROOT = find_project_root()
DEFAULT_RESULTS = ROOT / "results" / "direct_lp_20260922"
DEFAULT_CSV = ROOT / "100_Portfolios_10x10_Daily.csv"
DEFAULT_REPORT = ROOT / ".lp-work" / "audit-report.md"
REFERENCE_MODELS = (
    "Baseline_252d",
    "Baseline_252d_Constrained",
    "Baseline_3y",
    "Baseline_3y_Constrained",
    "LASSO_252d_U",
    "LASSO_3y_U",
    "LASSO_252d_Sharpe",
    "LASSO_252d_ReturnVolatility",
    "LASSO_3y_Sharpe",
    "LASSO_3y_ReturnVolatility",
)
EXPECTED_DAYS = {"validation": 123}
EXPECTED_ASSETS = 98
RETURN_ATOL = 1e-12
DIAGNOSTIC_ATOL = 2e-11
FEASIBILITY_ATOL = 2e-8
SECONDARY_TOLERANCE = 1e-10
INDEPENDENT_OBJECTIVE_ATOL = 5e-10


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_frame(frame: pd.DataFrame) -> str:
    encoded = frame.to_csv(float_format="%.17g", date_format="%Y-%m-%d").encode()
    return hashlib.sha256(encoded).hexdigest()


def load_returns_independently(path: Path, *, start: str, end: str) -> pd.DataFrame:
    heading = "Average Equal Weighted Returns -- Daily"
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip() == heading:
                break
        else:
            raise ValueError(f"source CSV has no {heading!r} section")
        lines = []
        for line in source:
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line)
    if len(lines) < 2:
        raise ValueError("source equal-weighted table is empty")
    frame = pd.read_csv(StringIO("".join(lines)), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame = frame.drop(columns=["ME9 BM10", "BIG HiBM"], errors="ignore")
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame = frame.sort_index().loc[start:end]
    frame = frame.astype(float).replace([-99.99, -999], np.nan) / 100.0
    frame.index.name = "date"
    if (frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates
            or frame.isna().to_numpy().any()
            or not np.isfinite(frame.to_numpy(dtype=float)).all()):
        raise ValueError("independently parsed source returns are invalid")
    return frame


def read_frame(directory: Path, name: str, *, dated: bool = False) -> pd.DataFrame:
    path = Path(directory) / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"missing artifact: {path.name}")
    frame = pd.read_csv(path, index_col=0)
    if dated:
        frame.index = pd.to_datetime(frame.index)
    numeric = frame.select_dtypes(include=[np.number])
    if (frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates
            or frame.isna().to_numpy().any()
            or (numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all())):
        raise ValueError(f"invalid artifact: {path.name}")
    return frame


def assert_close(label: str, actual, expected, *, atol=RETURN_ATOL, rtol=0.0) -> float:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape mismatch {actual.shape} != {expected.shape}")
    maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"{label}: maximum absolute error {maximum:.12g}")
    return maximum


def assert_scalar(label: str, actual, expected, *, atol=DIAGNOSTIC_ATOL, rtol=1e-9) -> float:
    return assert_close(label, [actual], [expected], atol=atol, rtol=rtol)


def true_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def independent_metrics(daily) -> dict[str, float]:
    values = np.asarray(daily, dtype=float)
    mean = float(values.mean())
    volatility = 0.0 if np.all(values == values[0]) else float(values.std(ddof=1))
    sharpe = mean / volatility if volatility > 0 else float("nan")
    wealth = np.r_[1.0, np.cumprod(1.0 + values)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "mean": mean,
        "volatility": volatility,
        "sharpe": sharpe,
        "negative_sharpe": -sharpe,
        "cumulative_return": float(wealth[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def risk_history(returns: pd.DataFrame, day: pd.Timestamp, spec: dict) -> pd.DataFrame:
    before = returns.loc[returns.index < day]
    observations = spec.get("observations")
    years = spec.get("years")
    if observations is not None:
        history = before.iloc[-int(observations):]
    elif years is not None:
        history = before.loc[day - pd.DateOffset(years=int(years)):]
    else:
        raise ValueError("risk window has neither observations nor years")
    if len(history) < int(spec["min_observations"]):
        raise ValueError(f"risk history before {day.date()} is too short")
    return history


def proxy_values(history: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    realized = history @ weights
    centered = realized - realized.mean()
    cumulative = np.cumsum(realized)
    peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    return {
        "budget": float(weights.sum()),
        "gross": float(np.abs(weights).sum()),
        "max_position": float(np.abs(weights).max()),
        "mad": float(np.abs(centered).mean()),
        "drawdown": float(np.max(peaks - cumulative)),
        "min_wealth": float(np.min(1.0 + realized)),
    }


def equal_weight_limits(history: np.ndarray, constraints: dict) -> tuple[float, float]:
    equal = np.full(history.shape[1], 1.0 / history.shape[1])
    proxies = proxy_values(history, equal)
    return (float(constraints["mad_ratio"]) * proxies["mad"],
            float(constraints["drawdown_ratio"]) * proxies["drawdown"])


def audit_provenance(results: Path, csv: Path, metadata: dict, frozen: dict,
                     returns: pd.DataFrame) -> dict:
    source_hash = digest_file(csv)
    recorded = {
        "data_sha256": metadata.get("data_sha256"),
        "source.sha256": metadata.get("source", {}).get("sha256"),
    }
    for label, value in recorded.items():
        if value != source_hash:
            raise ValueError(f"raw source hash differs from {label}")
    recorded_path = Path(metadata.get("source", {}).get("path", "")).resolve()
    if recorded_path != csv.resolve():
        raise ValueError("metadata source path differs from audited CSV")
    prefix_hash = digest_frame(returns)
    for label, value in (("metadata", metadata.get("development_data_digest")),
                         ("frozen", frozen.get("development_data_digest"))):
        if value != prefix_hash:
            raise ValueError(f"development frame digest differs from {label}")

    metadata_hashes = metadata.get("code_sha256")
    frozen_hashes = frozen.get("code_sha256")
    if not isinstance(metadata_hashes, dict) or not metadata_hashes:
        raise ValueError("metadata code hashes are missing")
    if not isinstance(frozen_hashes, dict) or not frozen_hashes:
        raise ValueError("frozen code hashes are missing")
    checked = {}
    for recorded_name, expected in metadata_hashes.items():
        path = Path(recorded_name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"metadata code hash mismatch: {recorded_name}")
        checked[str(path.resolve())] = expected
    for recorded_name, expected in frozen_hashes.items():
        path = ROOT / recorded_name
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"frozen code hash mismatch: {recorded_name}")
        matching = [value for name, value in metadata_hashes.items()
                    if Path(name).name == Path(recorded_name).name]
        if matching and any(value != expected for value in matching):
            raise ValueError(f"metadata/frozen code hashes disagree: {recorded_name}")
        checked[str(path.resolve())] = expected

    start_path = results / "run-start-code-sha256.json"
    start_checked = 0
    if start_path.is_file():
        start_hashes = json.loads(start_path.read_text())
        if not isinstance(start_hashes, dict) or not start_hashes:
            raise ValueError("run-start code hash artifact is invalid")
        for recorded_name, expected in start_hashes.items():
            path = ROOT / recorded_name
            if not path.is_file() or digest_file(path) != expected:
                raise ValueError(f"run-start code hash mismatch: {recorded_name}")
            candidates = [value for name, value in metadata_hashes.items()
                          if Path(name).name == Path(recorded_name).name]
            if candidates and any(value != expected for value in candidates):
                raise ValueError(f"run-start/export code hashes disagree: {recorded_name}")
            start_checked += 1
    return {
        "source_sha256": source_hash,
        "development_frame_sha256": prefix_hash,
        "current_code_files_checked": len(checked),
        "run_start_code_files_checked": start_checked,
    }


def validate_artifact_axes(name: str, frame: pd.DataFrame, dates: pd.DatetimeIndex,
                           assets: list[str] | None = None) -> None:
    if not frame.index.equals(dates):
        raise ValueError(f"{name}: dates differ from complete evaluation index")
    if assets is not None and list(frame.columns) != assets:
        raise ValueError(f"{name}: asset columns/order differ")


def audit_daily_artifacts(results: Path, returns: pd.DataFrame, metadata: dict,
                          frozen: dict, strategies: list[str]) -> tuple[dict, dict, dict]:
    config = frozen["config"]
    dates = returns.loc[config["validation_start"]:config["validation_end"]].index
    if len(dates) != EXPECTED_DAYS["validation"]:
        raise ValueError(
            f"expected {EXPECTED_DAYS['validation']} validation dates, found {len(dates)}")
    assets = list(returns.columns)
    if len(assets) != EXPECTED_ASSETS or assets != frozen.get("assets") or assets != metadata.get("assets"):
        raise ValueError("expected the same ordered 98 assets in source, metadata, and frozen inputs")

    weights, saved_returns, diagnostics, predictions = {}, {}, {}, {}
    return_errors = {}
    for name in strategies:
        weights[name] = read_frame(results, name + "_weights", dated=True)
        saved_returns[name] = read_frame(results, name + "_returns", dated=True)
        diagnostics[name] = read_frame(results, name + "_diagnostics", dated=True)
        validate_artifact_axes(name + " weights", weights[name], dates, assets)
        validate_artifact_axes(name + " returns", saved_returns[name], dates)
        validate_artifact_axes(name + " diagnostics", diagnostics[name], dates)
        if list(saved_returns[name].columns) != ["return"]:
            raise ValueError(f"{name}: saved return artifact must have one 'return' column")
        realized = np.sum(weights[name].to_numpy(dtype=float)
                          * returns.loc[dates].to_numpy(dtype=float), axis=1)
        return_errors[name] = assert_close(
            name + " realized returns", saved_returns[name]["return"], realized)

    for name in strategies:
        if name == "EW":
            equal = np.full((len(dates), len(assets)), 1.0 / len(assets))
            assert_close("EW weights", weights[name], equal, atol=1e-15)
        else:
            predictions[name] = read_frame(results, name + "_predictions", dated=True)
            validate_artifact_axes(name + " predictions", predictions[name], dates, assets)

    expected_history = frozen["risk_window"]
    if (expected_history.get("observations") is not None
            or int(expected_history.get("years", 0)) != 3
            or int(expected_history.get("min_observations", 0)) != 252):
        raise ValueError("real audit requires the frozen prior-three-year/252-minimum risk window")
    constraints = frozen["lp_constraints"]
    expected_constraints = {
        "leverage_limit": 1.5,
        "position_limit": 0.05,
        "mad_ratio": 1.0,
        "drawdown_ratio": 1.0,
        "wealth_floor": 1e-6,
    }
    for key, value in expected_constraints.items():
        if not np.isclose(float(constraints[key]), value, atol=1e-15, rtol=0):
            raise ValueError(f"unexpected frozen LP constraint {key}={constraints[key]!r}")

    maxima = {
        "budget_error": 0.0,
        "gross_excess": 0.0,
        "position_excess": 0.0,
        "mad_excess": 0.0,
        "drawdown_excess": 0.0,
        "wealth_shortfall": 0.0,
        "saved_diagnostic_error": 0.0,
        "projected_return_error": 0.0,
        "secondary_gap_error": 0.0,
        "secondary_gap_maximum": 0.0,
    }
    history_cache = {}
    for day in dates:
        history_frame = risk_history(returns, day, expected_history)
        history = history_frame.to_numpy(dtype=float)
        mad_limit, drawdown_limit = equal_weight_limits(history, constraints)
        history_cache[day] = (history, mad_limit, drawdown_limit)
        for name in strategies:
            row = diagnostics[name].loc[day]
            for column, expected in (("history_start", history_frame.index[0]),
                                     ("history_end", history_frame.index[-1])):
                if column not in row or pd.Timestamp(row[column]) != expected:
                    raise ValueError(f"{name} {day.date()}: {column} does not match prior history")
            if int(row["history_count"]) != len(history):
                raise ValueError(f"{name} {day.date()}: history_count differs")
            maxima["saved_diagnostic_error"] = max(
                maxima["saved_diagnostic_error"],
                assert_scalar(f"{name} {day.date()} mad_limit", row["mad_limit"], mad_limit),
                assert_scalar(f"{name} {day.date()} drawdown_limit", row["drawdown_limit"], drawdown_limit),
            )
            w = weights[name].loc[day].to_numpy(dtype=float)
            proxies = proxy_values(history, w)
            maxima["budget_error"] = max(maxima["budget_error"], abs(proxies["budget"] - 1.0))
            maxima["gross_excess"] = max(maxima["gross_excess"],
                                          proxies["gross"] - float(constraints["leverage_limit"]))
            maxima["position_excess"] = max(maxima["position_excess"],
                                             proxies["max_position"] - float(constraints["position_limit"]))
            maxima["mad_excess"] = max(maxima["mad_excess"], proxies["mad"] - mad_limit)
            maxima["drawdown_excess"] = max(maxima["drawdown_excess"],
                                             proxies["drawdown"] - drawdown_limit)
            maxima["wealth_shortfall"] = max(maxima["wealth_shortfall"],
                                              float(constraints["wealth_floor"]) - proxies["min_wealth"])
            if name == "EW":
                continue
            required = {"budget", "gross", "max_position", "mad", "drawdown", "min_wealth",
                        "solver_status", "feasible", "primary_return", "predicted_return",
                        "secondary_gap"}
            missing = required.difference(diagnostics[name].columns)
            if missing:
                raise ValueError(f"{name}: diagnostics missing {sorted(missing)}")
            for key, value in proxies.items():
                maxima["saved_diagnostic_error"] = max(
                    maxima["saved_diagnostic_error"],
                    assert_scalar(f"{name} {day.date()} {key}", row[key], value),
                )
            if str(row["solver_status"]) != "optimal" or not true_value(row["feasible"]):
                raise ValueError(f"{name} {day.date()}: saved solve is not feasible/optimal")
            mu = predictions[name].loc[day].to_numpy(dtype=float)
            projected = float(mu @ w)
            maxima["projected_return_error"] = max(
                maxima["projected_return_error"],
                assert_scalar(f"{name} {day.date()} predicted return",
                              row["predicted_return"], projected, atol=5e-13, rtol=1e-10),
            )
            gap = float(row["primary_return"]) - projected
            maxima["secondary_gap_error"] = max(
                maxima["secondary_gap_error"],
                assert_scalar(f"{name} {day.date()} secondary gap", row["secondary_gap"], gap,
                              atol=5e-13, rtol=1e-10),
            )
            maxima["secondary_gap_maximum"] = max(maxima["secondary_gap_maximum"], gap)
            if gap > SECONDARY_TOLERANCE + 5e-12 or gap < -FEASIBILITY_ATOL:
                raise ValueError(f"{name} {day.date()}: final objective is outside secondary tolerance")

    if maxima["budget_error"] > FEASIBILITY_ATOL:
        raise ValueError(f"budget feasibility failed: {maxima['budget_error']:.3g}")
    if maxima["gross_excess"] > FEASIBILITY_ATOL:
        raise ValueError(f"gross feasibility failed: {maxima['gross_excess']:.3g}")
    if maxima["position_excess"] > FEASIBILITY_ATOL:
        raise ValueError(f"position feasibility failed: {maxima['position_excess']:.3g}")
    if maxima["mad_excess"] > FEASIBILITY_ATOL:
        raise ValueError(f"MAD feasibility failed: {maxima['mad_excess']:.3g}")
    if maxima["drawdown_excess"] > FEASIBILITY_ATOL:
        raise ValueError(f"drawdown feasibility failed: {maxima['drawdown_excess']:.3g}")
    if maxima["wealth_shortfall"] > FEASIBILITY_ATOL:
        raise ValueError(f"wealth-floor feasibility failed: {maxima['wealth_shortfall']:.3g}")
    return ({"dates": len(dates), "assets": len(assets), "return_max_errors": return_errors,
             **maxima},
            {"weights": weights, "returns": saved_returns, "diagnostics": diagnostics,
             "predictions": predictions, "dates": dates},
            history_cache)


def audit_turnover(returns: pd.DataFrame, artifacts: dict, strategies: list[str]) -> dict:
    maximum = 0.0
    assets = returns.columns
    for name in strategies:
        weights = artifacts["weights"][name]
        diagnostics = artifacts["diagnostics"][name]
        actual = returns.loc[weights.index].to_numpy(dtype=float)
        values = weights.to_numpy(dtype=float)
        realized = np.sum(values * actual, axis=1)
        pretrade = np.empty_like(values)
        pretrade[0] = 1.0 / len(assets)
        pretrade[1:] = values[:-1] * (1.0 + actual[:-1]) / (1.0 + realized[:-1, None])
        turnover = 0.5 * np.abs(values - pretrade).sum(axis=1)
        maximum = max(maximum, assert_close(name + " drifted turnover",
                                             diagnostics["turnover"], turnover))
    return {"strategies": len(strategies), "maximum_absolute_error": maximum}


def audit_summaries(results: Path, artifacts: dict, frozen: dict,
                    strategies: list[str]) -> dict:
    config = frozen["config"]
    periods = {"validation": (config["validation_start"], config["validation_end"])}
    report = {}
    for period, (start, end) in periods.items():
        summary = read_frame(results, period + "_summary")
        if set(summary.index) != set(strategies):
            raise ValueError(f"{period}_summary has unexpected strategies")
        period_maximum = 0.0
        period_days = None
        for name in strategies:
            daily = artifacts["returns"][name].loc[start:end, "return"]
            weights = artifacts["weights"][name].loc[daily.index]
            diagnostics = artifacts["diagnostics"][name].loc[daily.index]
            expected = independent_metrics(daily)
            gross = weights.abs().sum(axis=1)
            expected.update(
                average_gross_leverage=float(gross.mean()),
                max_gross_leverage=float(gross.max()),
                evaluation_days=len(daily),
                average_turnover=float(diagnostics["turnover"].mean()),
                max_absolute_position=float(weights.abs().to_numpy().max()),
            )
            expected_strings = {"start": str(daily.index[0].date()), "end": str(daily.index[-1].date())}
            row = summary.loc[name]
            for key, value in expected.items():
                period_maximum = max(period_maximum, assert_scalar(
                    f"{period} {name} {key}", row[key], value, atol=2e-12, rtol=1e-9))
            for key, value in expected_strings.items():
                if str(row[key]) != value:
                    raise ValueError(f"{period} {name} {key} differs")
            period_days = len(daily)
        if period_days != EXPECTED_DAYS[period]:
            raise ValueError(f"{period}: expected {EXPECTED_DAYS[period]} days, found {period_days}")
        report[period] = {"days": period_days, "strategies": len(strategies),
                          "maximum_absolute_metric_error": period_maximum}
    return report


def validation_cost_sensitivity(artifacts: dict, frozen: dict,
                                strategies: list[str]) -> dict:
    """Illustrative after-the-fact costs; these rates do not select a model."""
    start = frozen["config"]["validation_start"]
    end = frozen["config"]["validation_end"]
    rates = {"5bps": 0.0005, "10bps": 0.0010}
    rows = {}
    for name in strategies:
        gross = artifacts["returns"][name].loc[start:end, "return"].to_numpy(dtype=float)
        turnover = artifacts["diagnostics"][name].loc[start:end, "turnover"].to_numpy(dtype=float)
        values = {"gross_cumulative_return": float(np.prod(1.0 + gross) - 1.0)}
        for label, one_way_rate in rates.items():
            net = gross - 2.0 * turnover * one_way_rate
            if np.any(1.0 + net <= 0.0):
                raise ValueError(f"{name}: illustrative {label} net wealth becomes nonpositive")
            values[label + "_cumulative_return"] = float(np.prod(1.0 + net) - 1.0)
        rows[name] = values
    return {
        "method": "daily gross return - 2 * drifted turnover * one-way rate; then compound",
        "one_way_rates": rates,
        "used_for_selection": False,
        "strategies": rows,
    }


def reference_frame_match(label: str, left: pd.DataFrame, right: pd.DataFrame) -> float:
    if (not left.index.equals(right.index) or list(left.columns) != list(right.columns)
            or left.shape != right.shape):
        raise ValueError(f"{label}: axes differ")
    left_numeric = left.select_dtypes(include=[np.number])
    right_numeric = right[left_numeric.columns]
    maximum = assert_close(label, left_numeric, right_numeric, atol=1e-12, rtol=1e-9)
    for column in left.columns.difference(left_numeric.columns):
        if not left[column].astype(str).equals(right[column].astype(str)):
            raise ValueError(f"{label}: column {column} differs")
    return maximum


def audit_reference(results: Path, metadata: dict, frozen: dict) -> dict:
    reference_info = metadata.get("reference_comparison")
    if not isinstance(reference_info, dict):
        raise ValueError("reference comparison metadata is missing")
    reference = Path(reference_info["directory"])
    if not reference.is_dir():
        raise ValueError("saved reference result directory is unavailable")
    metadata_path = reference / "run_metadata.json"
    if digest_file(metadata_path) != reference_info.get("metadata_sha256"):
        raise ValueError("reference metadata hash differs")
    start, end = frozen["config"]["validation_start"], frozen["config"]["validation_end"]
    checked, maximum = 0, 0.0
    for name in REFERENCE_MODELS:
        for suffix in ("weights", "returns"):
            imported = read_frame(results, f"reference_{name}_{suffix}", dated=True)
            original = read_frame(reference, f"{name}_{suffix}", dated=True).loc[start:end]
            maximum = max(maximum, reference_frame_match(
                f"reference {name} {suffix}", imported, original))
            checked += 1
    if checked != 2 * len(REFERENCE_MODELS):
        raise ValueError("not all expected reference artifacts were checked")
    if int(reference_info.get("strategies", -1)) != len(REFERENCE_MODELS):
        raise ValueError("reference strategy count differs")

    comparison = read_frame(results, "comparison")
    original_comparison = read_frame(reference, "comparison")
    common_columns = [column for column in comparison.columns
                      if column in original_comparison.columns and column != "role"]
    for name in REFERENCE_MODELS:
        if name not in comparison.index or name not in original_comparison.index:
            raise ValueError(f"comparison is missing reference row {name}")
        for column in common_columns:
            left, right = comparison.loc[name, column], original_comparison.loc[name, column]
            try:
                maximum = max(maximum, assert_scalar(
                    f"comparison {name} {column}", left, right, atol=1e-12, rtol=1e-9))
            except (TypeError, ValueError):
                if str(left) != str(right):
                    raise ValueError(f"comparison {name} {column} differs")
    return {"directory": str(reference.resolve()), "strategies": len(REFERENCE_MODELS),
            "artifacts_checked": checked, "maximum_absolute_error": maximum,
            "numeric_tolerance": {"atol": 1e-12, "rtol": 1e-9}}


def solve_primary_clarabel(history: np.ndarray, mu: np.ndarray, constraints: dict,
                            mad_limit: float, drawdown_limit: float) -> tuple[np.ndarray, float, str]:
    count, assets = history.shape
    weights = cp.Variable(assets)
    peaks = cp.Variable(count)
    centered = history - history.mean(axis=0, keepdims=True)
    cumulative = np.cumsum(history, axis=0)
    cumulative_returns = cumulative @ weights
    restrictions = [
        cp.sum(weights) == 1.0,
        cp.norm1(weights) <= float(constraints["leverage_limit"]),
        cp.abs(weights) <= float(constraints["position_limit"]),
        cp.norm1(centered @ weights) <= count * mad_limit,
        history @ weights >= float(constraints["wealth_floor"]) - 1.0,
        peaks >= 0.0,
        peaks >= cumulative_returns,
        peaks - cumulative_returns <= drawdown_limit,
    ]
    if count > 1:
        restrictions.append(peaks[1:] >= peaks[:-1])
    problem = cp.Problem(cp.Maximize(mu @ weights), restrictions)
    value = problem.solve(
        solver=cp.CLARABEL,
        max_iter=1000,
        tol_gap_abs=1e-11,
        tol_gap_rel=1e-11,
        tol_feas=1e-11,
    )
    if problem.status != cp.OPTIMAL or weights.value is None or value is None:
        raise ValueError(f"independent CLARABEL primary solve status is {problem.status!r}")
    return np.asarray(weights.value, dtype=float), float(value), str(problem.status)


def audit_sampled_primary_solves(artifacts: dict, history_cache: dict, frozen: dict,
                                 strategies: list[str]) -> dict:
    lp_strategies = [name for name in strategies if name != "EW"]
    dates = artifacts["dates"]
    positions = np.linspace(0, len(dates) - 1, 6, dtype=int)
    samples = []
    for number, position in enumerate(positions):
        name = lp_strategies[number % len(lp_strategies)]
        day = dates[position]
        history, mad_limit, drawdown_limit = history_cache[day]
        mu = artifacts["predictions"][name].loc[day].to_numpy(dtype=float)
        independent_weights, objective, status = solve_primary_clarabel(
            history, mu, frozen["lp_constraints"], mad_limit, drawdown_limit)
        proxies = proxy_values(history, independent_weights)
        limits = frozen["lp_constraints"]
        if (abs(proxies["budget"] - 1.0) > FEASIBILITY_ATOL
                or proxies["gross"] > float(limits["leverage_limit"]) + FEASIBILITY_ATOL
                or proxies["max_position"] > float(limits["position_limit"]) + FEASIBILITY_ATOL
                or proxies["mad"] > mad_limit + FEASIBILITY_ATOL
                or proxies["drawdown"] > drawdown_limit + FEASIBILITY_ATOL
                or proxies["min_wealth"] < float(limits["wealth_floor"]) - FEASIBILITY_ATOL):
            raise ValueError(f"independent CLARABEL solution is infeasible: {name} {day.date()}")
        diagnostic = artifacts["diagnostics"][name].loc[day]
        primary = float(diagnostic["primary_return"])
        final = float(diagnostic["predicted_return"])
        primary_error = abs(objective - primary)
        if primary_error > INDEPENDENT_OBJECTIVE_ATOL:
            raise ValueError(
                f"{name} {day.date()}: independent primary objective differs by {primary_error:.3g}")
        if primary - final > SECONDARY_TOLERANCE + 5e-12:
            raise ValueError(f"{name} {day.date()}: saved final is outside secondary tolerance")
        if final > objective + INDEPENDENT_OBJECTIVE_ATOL:
            raise ValueError(f"{name} {day.date()}: saved final exceeds independent optimum")
        samples.append({
            "strategy": name,
            "date": str(day.date()),
            "status": status,
            "independent_objective": objective,
            "saved_primary_return": primary,
            "saved_final_return": final,
            "primary_absolute_error": primary_error,
            "saved_primary_minus_final": primary - final,
        })
    return {"solver": "CVXPY CLARABEL", "formulation": "affine/absolute/running-peak LP",
            "sample_count": len(samples), "objective_tolerance": INDEPENDENT_OBJECTIVE_ATOL,
            "secondary_tolerance": SECONDARY_TOLERANCE, "samples": samples}


def audit(results: Path, csv: Path) -> dict:
    results, csv = Path(results).resolve(), Path(csv).resolve()
    metadata_path, frozen_path = results / "run_metadata.json", results / "frozen_decisions.json"
    if not metadata_path.is_file() or not frozen_path.is_file():
        raise ValueError("result metadata/frozen artifacts are unavailable")
    metadata = json.loads(metadata_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if metadata.get("phase") != "development" or metadata.get("final_test_evaluated") is not False:
        raise ValueError("audit accepts only development results with the final test untouched")
    if (metadata.get("development_backtest_included") is not False
            or int(metadata.get("development_days", -1)) != 0
            or int(metadata.get("validation_days", -1)) != EXPECTED_DAYS["validation"]
            or metadata.get("evaluation_start") != frozen["config"]["validation_start"]):
        raise ValueError("audit requires the validation-only 123-day direct-LP run")
    if (results / "development_summary.csv").exists():
        raise ValueError("validation-only result unexpectedly contains development_summary.csv")
    if frozen.get("schema_version") != 1 or frozen.get("kind") != "direct_asset_lp":
        raise ValueError("unexpected frozen direct-LP schema")
    config = frozen["config"]
    returns = load_returns_independently(
        csv, start=config["history_start"], end=config["validation_end"])
    forecast_names = list(frozen["forecast_configs"])
    if set(forecast_names) != {"252d", "3y"}:
        raise ValueError("expected exactly the 252d and 3y forecast windows")
    strategies = [f"LP_{kind}_{window}" for window in forecast_names
                  for kind in ("LASSO", "Mean")] + ["EW"]

    report = {"status": "passed", "scope": "validation_only", "provenance": {},
              "daily_artifacts": {},
              "turnover": {}, "period_summaries": {}, "reference": {},
              "validation_cost_sensitivity": {}, "independent_primary_solves": {}}
    report["provenance"] = audit_provenance(results, csv, metadata, frozen, returns)
    daily, artifacts, history_cache = audit_daily_artifacts(
        results, returns, metadata, frozen, strategies)
    report["daily_artifacts"] = daily
    report["turnover"] = audit_turnover(returns, artifacts, strategies)
    report["period_summaries"] = audit_summaries(results, artifacts, frozen, strategies)
    report["validation_cost_sensitivity"] = validation_cost_sensitivity(
        artifacts, frozen, strategies)
    report["reference"] = audit_reference(results, metadata, frozen)
    report["independent_primary_solves"] = audit_sampled_primary_solves(
        artifacts, history_cache, frozen, strategies)
    return report


def render_report(report: dict, command: str) -> str:
    daily = report["daily_artifacts"]
    summaries = report["period_summaries"]
    solved = report["independent_primary_solves"]
    max_primary_error = max(sample["primary_absolute_error"] for sample in solved["samples"])
    max_final_gap = max(sample["saved_primary_minus_final"] for sample in solved["samples"])
    cost_rows = ["| Strategy | Gross | 5 bps one-way | 10 bps one-way |",
                 "|---|---:|---:|---:|"]
    cost_strategies = report["validation_cost_sensitivity"]["strategies"]
    display_order = ("LP_LASSO_252d", "LP_Mean_252d", "LP_LASSO_3y", "LP_Mean_3y", "EW")
    if set(cost_strategies) != set(display_order):
        raise ValueError("cost-sensitivity strategies differ from expected validation portfolios")
    for name in display_order:
        values = cost_strategies[name]
        cost_rows.append(
            f"| {name} | {values['gross_cumulative_return']:.2%} | "
            f"{values['5bps_cumulative_return']:.2%} | "
            f"{values['10bps_cumulative_return']:.2%} |")
    cost_table = "\n".join(cost_rows)
    return f"""# Independent direct-LP artifact audit

Status: **passed**.

Command:

`{command}`

The audit wrote `results/direct_lp_20260922/verification.json` only after every check completed.

## Evidence

- The raw source SHA-256, independently reconstructed development-frame digest, {report['provenance']['current_code_files_checked']} current code hashes, and {report['provenance']['run_start_code_files_checked']} optional run-start code hashes matched saved provenance.
- Reconstructed all five 98-asset weight and realized-return streams over all {daily['dates']} validation dates. The saved validation summary covered {summaries['validation']['days']} dates; no development backtest or development summary was produced.
- For every date and strategy, independently rebuilt the prior-three-calendar-year history, EW-relative centered-simple-return MAD cap, zero-initialized uncompounded absolute drawdown cap, budget, gross leverage, absolute-position, and historical one-day wealth-floor checks.
- Recomputed every saved diagnostic and every forecast projection `mu @ w`; maximum diagnostic error was `{daily['saved_diagnostic_error']:.3g}` and maximum projection error was `{daily['projected_return_error']:.3g}`.
- Reconstructed drifted pre-trade turnover for all {report['turnover']['strategies']} strategies; maximum absolute error was `{report['turnover']['maximum_absolute_error']:.3g}`.
- Independently recomputed validation compounded return, sample daily volatility, daily Sharpe, compounded relative drawdown, leverage, position, and turnover metrics; maximum metric error was `{summaries['validation']['maximum_absolute_metric_error']:.3g}`.
- Calculated illustrative validation cost sensitivity at 5 and 10 bps one-way using daily gross return minus `2 * drifted turnover * rate`, then compounding. These rates were not used for model or parameter selection.
- Matched {report['reference']['artifacts_checked']} imported weight/return artifacts for {report['reference']['strategies']} existing reference strategies and their shared numeric comparison fields; maximum absolute error was `{report['reference']['maximum_absolute_error']:.3g}`.
- Independently solved {solved['sample_count']} real primary LPs with CVXPY/CLARABEL using affine budget/wealth/running-peak constraints and absolute-value gross/MAD/position constraints. All statuses were optimal and solutions were independently feasible. Maximum primary-objective difference was `{max_primary_error:.3g}`; maximum saved primary-to-final gap was `{max_final_gap:.3g}` against the `1e-10` secondary tolerance.

## Illustrative validation cost sensitivity

{cost_table}

The LP risk measures are proxies: MAD is the mean absolute deviation of centered historical simple returns, and drawdown is the largest decline in uncompounded cumulative simple return from a running peak that includes initial zero. Reported performance drawdown is separately computed from compounded wealth. No August 2026 final-test return was read or evaluated.
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    verification = args.results / "verification.json"
    if verification.exists():
        parser.error(f"refusing to overwrite {verification}")
    try:
        report = audit(args.results, args.csv)
        command = ("PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
                   ".venv/bin/python .lp-work/audit_direct_lp.py")
        report_text = render_report(report, command)
        encoded = json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text)
        verification.write_text(encoded)
    except (cp.error.SolverError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(encoded, end="")


if __name__ == "__main__":
    main()
