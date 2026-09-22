"""Bounded callback-based search for portfolio growth coefficients."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real

import pandas as pd


MAX_REWARD_STRENGTH = 0.03
MAX_EXPOSURE_STRENGTH = 3.0
_ZERO_REWARD_NEIGHBOR = 0.003 / 2
_ZERO_EXPOSURE_NEIGHBOR = 0.1 / 2


@dataclass(frozen=True)
class GrowthParameters:
    """The two nonnegative coefficients varied by the validation search."""

    reward_strength: float
    exposure_strength: float

    def __post_init__(self):
        values = (
            ("reward_strength", self.reward_strength, MAX_REWARD_STRENGTH),
            ("exposure_strength", self.exposure_strength, MAX_EXPOSURE_STRENGTH),
        )
        for name, value, upper in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite number")
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f"{name} must be finite and between 0 and {upper:g}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SearchConfig:
    """Budget and stopping rule for a complete-batch parameter search."""

    max_candidates: int = 15
    min_improvement: float = 0.002
    patience: int = 1

    def __post_init__(self):
        if (isinstance(self.max_candidates, bool)
                or not isinstance(self.max_candidates, Integral)):
            raise TypeError("max_candidates must be a positive integer")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        if isinstance(self.min_improvement, bool) or not isinstance(self.min_improvement, Real):
            raise TypeError("min_improvement must be a finite nonnegative number")
        improvement = float(self.min_improvement)
        if not math.isfinite(improvement) or improvement < 0:
            raise ValueError("min_improvement must be a finite nonnegative number")
        if isinstance(self.patience, bool) or not isinstance(self.patience, Integral):
            raise TypeError("patience must be a positive integer")
        if self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        object.__setattr__(self, "max_candidates", int(self.max_candidates))
        object.__setattr__(self, "min_improvement", improvement)
        object.__setattr__(self, "patience", int(self.patience))


@dataclass(frozen=True)
class SearchResult:
    selected: GrowthParameters
    raw_best: GrowthParameters
    history: pd.DataFrame
    stop_reason: str
    rounds: int
    selected_score: float
    raw_best_score: float


COARSE_PARAMETERS = tuple(
    GrowthParameters(reward, exposure)
    for reward in (0, 0.003, 0.01)
    for exposure in (0, 0.1, 1)
)

_REQUIRED_METRICS = ("cumulative_return", "average_gross_leverage")
_RECORD_FIELDS = {
    "candidate_id", "reward_strength", "exposure_strength", "round", "status", "error"
}


def _validated_metrics(value):
    if not isinstance(value, Mapping):
        raise ValueError("Evaluation must return a metrics mapping")
    result = dict(value)
    missing = [name for name in _REQUIRED_METRICS if name not in result]
    if missing:
        raise ValueError("Missing required metric(s): " + ", ".join(missing))
    overlap = _RECORD_FIELDS.intersection(result)
    if overlap:
        raise ValueError("Metrics use reserved field(s): " + ", ".join(sorted(overlap)))
    for name in _REQUIRED_METRICS:
        metric = result[name]
        if isinstance(metric, bool) or not isinstance(metric, Real):
            raise ValueError(f"{name} must be a finite number")
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(f"{name} must be finite")
        if name == "average_gross_leverage" and metric < 0:
            raise ValueError("average_gross_leverage must be nonnegative")
        result[name] = metric
    return result


def _neighbors(center):
    reward_values = [_ZERO_REWARD_NEIGHBOR] if center.reward_strength == 0 else [
        center.reward_strength / 2,
        center.reward_strength * 2,
    ]
    exposure_values = [_ZERO_EXPOSURE_NEIGHBOR] if center.exposure_strength == 0 else [
        center.exposure_strength / 2,
        center.exposure_strength * 2,
    ]
    candidates = []
    for reward in reward_values:
        if reward <= MAX_REWARD_STRENGTH:
            candidates.append(GrowthParameters(reward, center.exposure_strength))
    for exposure in exposure_values:
        if exposure <= MAX_EXPOSURE_STRENGTH:
            candidates.append(GrowthParameters(center.reward_strength, exposure))
    return list(dict.fromkeys(candidates))


def _raw_best(evaluations):
    successful = [item for item in evaluations if item[1]["status"] == "ok"]
    return max(successful, key=lambda item: item[1]["cumulative_return"])


def _evaluate_batch(evaluate, candidates, round_number, evaluations):
    for parameters in candidates:
        record = {
            "candidate_id": len(evaluations) + 1,
            "reward_strength": parameters.reward_strength,
            "exposure_strength": parameters.exposure_strength,
            "round": round_number,
            "status": "ok",
            "error": "",
        }
        try:
            returned = evaluate(parameters)
        except (RuntimeError, ValueError) as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        else:
            try:
                returned = _validated_metrics(returned)
            except ValueError as error:
                record.update(status="failed", error=f"ValueError: {error}")
            else:
                record.update(returned)
        evaluations.append((parameters, record))


def search_growth(evaluate, *, config=SearchConfig(), initial_parameters=None):
    """Evaluate complete batches and return a deterministic conservative choice."""

    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be a SearchConfig")
    if initial_parameters is None:
        initial = list(COARSE_PARAMETERS)
    else:
        initial = list(initial_parameters)
        if not initial:
            raise ValueError("initial_parameters must not be empty")
    if any(not isinstance(parameters, GrowthParameters) for parameters in initial):
        raise TypeError("initial_parameters must contain GrowthParameters")
    initial = list(dict.fromkeys(initial))
    if len(initial) > config.max_candidates:
        raise ValueError("Initial batch exceeds max_candidates")

    evaluations = []
    visited = set(initial)
    _evaluate_batch(evaluate, initial, 0, evaluations)
    if not any(record["status"] == "ok" for _, record in evaluations):
        raise RuntimeError("All initial candidates failed")

    _, initial_best_record = _raw_best(evaluations)
    improvement_anchor = initial_best_record["cumulative_return"]
    stagnant_rounds = 0
    rounds = 0

    while True:
        if len(evaluations) >= config.max_candidates:
            stop_reason = "budget"
            break
        center, _ = _raw_best(evaluations)
        batch = [candidate for candidate in _neighbors(center) if candidate not in visited]
        if not batch:
            stop_reason = "no_neighbors"
            break
        if len(evaluations) + len(batch) > config.max_candidates:
            stop_reason = "budget"
            break
        rounds += 1
        visited.update(batch)
        _evaluate_batch(evaluate, batch, rounds, evaluations)
        _, best_record = _raw_best(evaluations)
        best_score = best_record["cumulative_return"]
        improvement = best_score - improvement_anchor
        if (improvement > 0
                and improvement + 1e-15 >= config.min_improvement):
            improvement_anchor = best_score
            stagnant_rounds = 0
        else:
            stagnant_rounds += 1
            if stagnant_rounds >= config.patience:
                stop_reason = "patience"
                break

    raw_parameters, raw_record = _raw_best(evaluations)
    raw_score = raw_record["cumulative_return"]
    tolerance = config.min_improvement + 1e-15
    eligible = [
        item for item in evaluations
        if (item[1]["status"] == "ok"
            and raw_score - item[1]["cumulative_return"] <= tolerance)
    ]
    selected_parameters, selected_record = min(
        eligible,
        key=lambda item: (
            item[0].reward_strength,
            item[1]["average_gross_leverage"],
            item[0].exposure_strength,
            item[1]["candidate_id"],
        ),
    )
    history = pd.DataFrame(record for _, record in evaluations)
    return SearchResult(
        selected=selected_parameters,
        raw_best=raw_parameters,
        history=history,
        stop_reason=stop_reason,
        rounds=rounds,
        selected_score=selected_record["cumulative_return"],
        raw_best_score=raw_score,
    )
