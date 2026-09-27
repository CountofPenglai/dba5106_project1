#!/usr/bin/env python3
"""Independently audit exported two-stage portfolio artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.data import load_returns


DEFAULT_RESULTS = ROOT / "results" / "two_stage_lasso_20260922"
DEFAULT_CSV = ROOT / "100_Portfolios_10x10_Daily.csv"
DEFAULT_PRIOR = Path("/Users/zhengce/projects/fundation_ba/project1/results/lasso_windows_constrained_5y_20260922")
ATOL = 1e-10


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frame(directory, name):
    path = Path(directory) / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"missing artifact: {path.name}")
    frame = pd.read_csv(path, index_col=0)
    if frame.empty or frame.isna().to_numpy().any():
        raise ValueError(f"empty or missing values in {path.name}")
    return frame


def dated(frame):
    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    return result


def assert_close(label, actual, expected, *, atol=ATOL, rtol=1e-8):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape mismatch {actual.shape} != {expected.shape}")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"{label}: maximum absolute error {error:.3g}")
    return error


def metrics(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    volatility = 0.0 if np.all(values == values[0]) else float(values.std(ddof=1))
    sharpe = mean / volatility if volatility > 0 else np.nan
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {
        "mean": mean,
        "volatility": volatility,
        "sharpe": sharpe,
        "negative_sharpe": -sharpe,
        "cumulative_return": float(wealth[-1] - 1),
        "max_drawdown": float(drawdown.min()),
    }


def risk_history(returns, day, risk_window):
    before = returns.loc[returns.index < day]
    if risk_window["observations"] is not None:
        history = before.iloc[-int(risk_window["observations"]):]
    else:
        cutoff = day - pd.DateOffset(years=int(risk_window["years"]))
        history = before.loc[cutoff:]
    minimum = int(risk_window["min_observations"])
    if len(history) < minimum:
        raise ValueError(f"risk history before {day.date()} has only {len(history)} rows")
    return history


def independent_features(returns):
    values = returns.to_numpy(dtype=float)
    market = values.mean(axis=1)
    features = np.empty((len(returns) - 63, returns.shape[1], 6))
    for row, target in enumerate(range(63, len(returns))):
        h = values[:target]
        features[row, :, 0] = h[-1]
        features[row, :, 1] = np.prod(1 + h[-5:], axis=0) - 1
        features[row, :, 2] = np.prod(1 + h[-21:], axis=0) - 1
        features[row, :, 3] = np.prod(1 + h[-63:], axis=0) - 1
        features[row, :, 4] = np.std(h[-21:], axis=0, ddof=1)
        features[row, :, 5] = np.prod(1 + market[target-21:target]) - 1
    return features, returns.iloc[63:]


def audit_forecast(results, returns, frozen, report):
    features, targets = independent_features(returns)
    first_validation = pd.Timestamp(frozen["config"]["validation_start"])
    position = targets.index.get_loc(first_validation)
    for name, config in frozen["forecast_configs"].items():
        window = config["window"]
        if window["observations"] is not None:
            positions = np.arange(position-int(window["observations"]), position)
        else:
            cutoff = first_validation - pd.DateOffset(years=int(window["years"]))
            positions = np.arange(targets.index.searchsorted(cutoff), position)
        asset = 0
        scaler = StandardScaler().fit(features[positions, asset])
        model = Lasso(alpha=float(frozen["alphas"][name]), max_iter=100000, tol=1e-8)
        model.fit(scaler.transform(features[positions, asset]), targets.iloc[positions, asset])
        expected = model.predict(scaler.transform(features[position, asset][None]))[0]
        saved = dated(read_frame(results, f"LASSO_{name}_predictions"))
        actual = saved.loc[first_validation, targets.columns[asset]]
        report["forecast_checks"][name] = {
            "date": str(first_validation.date()), "asset": str(targets.columns[asset]),
            "absolute_error": abs(float(actual) - float(expected)),
        }
        assert_close(f"{name} first validation forecast", [actual], [expected], atol=1e-13, rtol=1e-10)


def audit_decision_fits(results, frozen, report):
    fits = read_frame(results, "decision_fits")
    start, end = frozen["config"]["decision_start"], frozen["config"]["decision_end"]
    maximum = 0.0
    for row_number, row in fits.iterrows():
        components = dated(read_frame(results, row["forecast"] + "_components")).loc[start:end]
        theta = row[["a", "b", "c"]].to_numpy(dtype=float)
        realized = components.to_numpy() @ theta
        mean, volatility = float(realized.mean()), float(realized.std(ddof=1))
        if row["objective"] == "sharpe":
            score = mean / volatility
        elif row["objective"] == "return_volatility":
            score = mean - float(row["gamma"]) * volatility
        else:
            raise ValueError(f"decision_fits row {row_number} has unknown objective")
        maximum = max(maximum,
                      assert_close(f"decision fit {row_number} mean", [row["mean"]], [mean], atol=1e-12),
                      assert_close(f"decision fit {row_number} volatility", [row["volatility"]], [volatility], atol=1e-12),
                      assert_close(f"decision fit {row_number} score", [row["score"]], [score], atol=1e-12))
    report["decision_objective_checks"] = {"rows": len(fits), "maximum_absolute_error": maximum}


def audit_turnover(results, returns, strategy_names, report):
    maximum = 0.0
    for name in sorted(set(strategy_names)):
        weights = dated(read_frame(results, name + "_weights"))
        diagnostics = dated(read_frame(results, name + "_diagnostics"))
        asset_returns = returns.loc[weights.index].to_numpy(dtype=float)
        realized = np.sum(weights.to_numpy() * asset_returns, axis=1)
        before = np.empty_like(weights.to_numpy())
        before[0] = 1 / weights.shape[1]
        before[1:] = (weights.to_numpy()[:-1] * (1 + asset_returns[:-1])
                      / (1 + realized[:-1, None]))
        turnover = 0.5 * np.abs(weights.to_numpy() - before).sum(axis=1)
        maximum = max(maximum, assert_close(
            name + " drifted turnover", diagnostics.loc[weights.index, "turnover"], turnover, atol=1e-12
        ))
    report["turnover_checks"] = {"strategies": len(set(strategy_names)),
                                 "maximum_absolute_error": maximum}


def audit_constraints(results, returns, frozen, model_weights, report):
    constraints = frozen["constraints"]
    limit = float(constraints["leverage_limit"])
    floor = float(constraints["wealth_floor"])
    volatility_limit = float(constraints["volatility_limit"])
    maxima = {"budget_error": 0.0, "gross_excess": 0.0, "risk_ratio": 0.0,
              "wealth_shortfall": 0.0}
    candidate_names = []
    for key in sorted({model["forecast_key"] for model in frozen["models"].values()}):
        candidate_names.extend((f"{key}_U_weights", f"{key}_V_weights", f"{key}_E_weights"))
    frames = {name: dated(read_frame(results, name)) for name in sorted(set(candidate_names))}
    frames.update(model_weights)
    risk_by_date = {}
    for day in sorted(set().union(*(set(frame.index) for frame in frames.values()))):
        history = risk_history(returns, day, frozen["risk_window"])
        covariance = LedoitWolf().fit(history.to_numpy()).covariance_
        equal = np.full(returns.shape[1], 1 / returns.shape[1])
        variance_limit = float(equal @ covariance @ equal) * volatility_limit ** 2
        risk_by_date[day] = (history.to_numpy(), covariance, variance_limit)
    for name, weights in frames.items():
        for day, row in weights.iterrows():
            history, covariance, variance_limit = risk_by_date[day]
            w = row.to_numpy(dtype=float)
            maxima["budget_error"] = max(maxima["budget_error"], abs(float(w.sum()) - 1))
            maxima["gross_excess"] = max(maxima["gross_excess"], float(np.abs(w).sum()) - limit)
            maxima["risk_ratio"] = max(maxima["risk_ratio"], float(w @ covariance @ w) / variance_limit)
            maxima["wealth_shortfall"] = max(
                maxima["wealth_shortfall"], floor - float(np.min(1 + history @ w))
            )
    if maxima["budget_error"] > 1e-7 or maxima["gross_excess"] > 1e-7:
        raise ValueError(f"budget/gross constraint audit failed: {maxima}")
    if maxima["risk_ratio"] > 1 + 1e-6 or maxima["wealth_shortfall"] > 1e-8:
        raise ValueError(f"risk/wealth constraint audit failed: {maxima}")
    report["constraint_checks"] = {**maxima, "frames": len(frames)}


def audit_baselines(results, prior, report):
    if not Path(prior).is_dir():
        report["baseline_comparison"] = {"available": False}
        return
    mapping = {
        "Baseline_252d": "LASSO_252d",
        "Baseline_252d_Constrained": "LASSO_252d_Constrained",
        "Baseline_3y": "LASSO_3y",
        "Baseline_3y_Constrained": "LASSO_3y_Constrained",
    }
    checked, maximum = 0, 0.0
    for new, old in mapping.items():
        for suffix in ("weights", "returns"):
            new_path, old_path = Path(results) / f"{new}_{suffix}.csv", Path(prior) / f"{old}_{suffix}.csv"
            if not new_path.is_file() or not old_path.is_file():
                continue
            left, right = dated(read_frame(results, f"{new}_{suffix}")), dated(read_frame(prior, f"{old}_{suffix}"))
            dates = left.index.intersection(right.index)
            dates = dates[(dates >= "2026-01-01") & (dates <= "2026-06-30")]
            if len(dates):
                maximum = max(maximum, assert_close(f"baseline {new} {suffix}", left.loc[dates], right.loc[dates]))
                checked += 1
    report["baseline_comparison"] = {"available": checked > 0, "artifacts_checked": checked,
                                     "maximum_absolute_error": maximum}


def audit(results, csv, prior):
    results, csv = Path(results), Path(csv)
    metadata = json.loads((results / "run_metadata.json").read_text())
    frozen = json.loads((results / "frozen_decisions.json").read_text())
    returns = load_returns(csv, start=frozen["config"]["history_start"],
                           end=frozen["config"]["validation_end"])
    report = {"status": "passed", "source_checks": {}, "return_checks": {},
              "mixture_checks": {}, "metric_checks": {}, "gamma_checks": {},
              "forecast_checks": {}}

    current_hash = digest_file(csv)
    if current_hash != metadata["data_sha256"]:
        raise ValueError("current source CSV hash differs from run metadata")
    prefix_hash = hashlib.sha256(returns.to_csv(float_format="%.17g", date_format="%Y-%m-%d").encode()).hexdigest()
    if prefix_hash != frozen["development_data_digest"]:
        raise ValueError("current development prefix differs from frozen decisions")
    report["source_checks"] = {"file_sha256": current_hash, "development_prefix_sha256": prefix_hash}

    model_weights = {}
    validation_summary = read_frame(results, "validation_summary")
    for name, model in frozen["models"].items():
        weights = dated(read_frame(results, name + "_weights"))
        saved_returns = dated(read_frame(results, name + "_returns"))["return"]
        realized = (weights * returns.loc[weights.index]).sum(axis=1)
        report["return_checks"][name] = assert_close(name + " raw return", saved_returns, realized, atol=1e-12)
        components = dated(read_frame(results, model["forecast_key"] + "_components"))
        theta = np.asarray(model["theta"], dtype=float)
        report["mixture_checks"][name] = {
            "theta_sum_error": abs(float(theta.sum()) - 1),
            "minimum_theta": float(theta.min()),
            "return_error": assert_close(name + " component mixture", saved_returns, components @ theta, atol=1e-12),
        }
        candidate_weights = np.stack([
            dated(read_frame(results, f"{model['forecast_key']}_{component}_weights")).loc[weights.index].to_numpy()
            for component in ("U", "V", "E")
        ], axis=1)
        report["mixture_checks"][name]["weight_error"] = assert_close(
            name + " candidate weight mixture", weights,
            np.einsum("ncp,c->np", candidate_weights, theta), atol=1e-12
        )
        if abs(theta.sum() - 1) > 1e-10 or theta.min() < -1e-10:
            raise ValueError(f"invalid frozen theta for {name}")
        validation = saved_returns.loc[frozen["config"]["validation_start"]:frozen["config"]["validation_end"]]
        computed = metrics(validation)
        stored = validation_summary.loc[name]
        report["metric_checks"][name] = max(
            assert_close(f"{name} metric {key}", [stored[key]], [value], atol=1e-12)
            for key, value in computed.items()
        )
        model_weights[name] = weights

    gamma = read_frame(results, "gamma_validation")
    original_gammas = sorted(float(value) for value in frozen["config"]["gammas"])
    for forecast, rows in gamma.groupby("forecast"):
        observed = sorted(rows["gamma"].astype(float).tolist())
        assert_close(f"{forecast} gamma grid", observed, original_gammas, atol=0, rtol=0)
        best = float(rows["cumulative_return"].max())
        tied = rows[np.isclose(rows["cumulative_return"], best, atol=1e-10, rtol=1e-8)]
        selected = float(tied["gamma"].max())
        model = frozen["models"][forecast + "_ReturnVolatility"]
        if float(model["gamma"]) != selected:
            raise ValueError(f"{forecast}: frozen gamma does not match validation selection")
        saved_selected = rows.loc[rows["selected"].astype(str).str.lower().eq("true"), "gamma"].astype(float).tolist()
        if saved_selected != [selected]:
            raise ValueError(f"{forecast}: saved selected-gamma flags are inconsistent")
        chosen = rows[np.isclose(rows["gamma"], selected, atol=0, rtol=0)].iloc[0]
        assert_close(f"{forecast} selected theta", model["theta"], chosen[["a", "b", "c"]])
        report["gamma_checks"][forecast] = {"selected": selected, "cumulative_return": best}

    audit_decision_fits(results, frozen, report)
    strategy_names = list(frozen["models"])
    for key in sorted({model["forecast_key"] for model in frozen["models"].values()}):
        strategy_names.extend(f"{key}_{component}" for component in ("U", "V", "E"))
    audit_turnover(results, returns, strategy_names, report)
    audit_constraints(results, returns, frozen, model_weights, report)
    audit_forecast(results, returns, frozen, report)
    audit_baselines(results, prior, report)
    report["counts"] = {"models": len(frozen["models"]), "return_days": int(sum(len(x) for x in model_weights.values()))}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args(argv)
    output = args.results / "verification.json"
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    try:
        report = audit(args.results, args.csv, args.prior)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    output.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
