"""Training loss definitions, independent of fitting and financial metrics."""

from dataclasses import dataclass

import numpy as np

from portfolio.data.observation_weights import normalize_sample_weights


def weighted_mse(y, prediction, sample_weight=None):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (y.ndim != 1 or len(y) == 0 or prediction.shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()):
        raise ValueError("Targets and predictions must be aligned finite vectors")
    weights = normalize_sample_weights(sample_weight, len(y))
    return float(weights @ np.square(y - prediction))


@dataclass(frozen=True)
class LogGrowthLoss:
    """Portfolio variance minus a reward for average log growth.

    Inputs are actual decimal portfolio returns R@w, not centered residuals.
    Regularization is added separately. Sample weights sum to one.
    """

    reward_strength: float = 0.01
    exposure_strength: float = 0

    def __post_init__(self):
        if not np.isfinite(self.reward_strength) or self.reward_strength < 0:
            raise ValueError("Reward strength must be finite and nonnegative")
        if not np.isfinite(self.exposure_strength) or self.exposure_strength < 0:
            raise ValueError("Exposure strength must be finite and nonnegative")

    def components(self, returns, sample_weight=None):
        r = np.asarray(returns, dtype=float)
        if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
            raise ValueError("Need at least two finite portfolio returns")
        if np.any(r <= -1):
            raise ValueError("Log growth requires every portfolio return to exceed -1")
        weights = normalize_sample_weights(sample_weight, len(r))
        if np.count_nonzero(weights) < 2:
            raise ValueError("At least two observations need positive sample weight")
        mean = float(weights @ r)
        variance = float(weights @ np.square(r - mean))
        log_growth = float(weights @ np.log1p(r))
        reward = self.reward_strength * log_growth
        return {"mean": mean, "variance": variance, "log_growth": log_growth,
                "reward": reward, "loss": variance - reward}
