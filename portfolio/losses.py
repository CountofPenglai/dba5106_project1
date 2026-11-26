"""Training loss definitions, independent of fitting and financial metrics."""

from dataclasses import dataclass

import numpy as np

from portfolio.data.observation_weights import normalize_sample_weights


@dataclass(frozen=True)
class PortfolioLoss:
    exposure_strength: float = 0

    def __post_init__(self):
        if not np.isfinite(self.exposure_strength) or self.exposure_strength < 0:
            raise ValueError("Exposure strength must be finite and nonnegative")

    def components(self, returns, sample_weight=None):
        r = np.asarray(returns, dtype=float)
        if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
            raise ValueError("Need at least two finite portfolio returns")
        if np.any(r <= -1):
            raise ValueError("Each historical scenario must preserve positive wealth")
        weights = normalize_sample_weights(sample_weight, len(r))
        if np.count_nonzero(weights) < 2:
            raise ValueError("At least two observations need positive sample weight")
        mean = float(weights @ r)
        variance = float(weights @ np.square(r - mean))
        return {"mean": mean, "variance": variance, "loss": variance}
