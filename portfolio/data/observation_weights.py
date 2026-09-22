"""Weights on historical observations, independent of asset holdings."""
import numpy as np

def normalize_sample_weights(sample_weight, n):
    """Validate and normalize sample weights to sum to one."""
    weights = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    total = weights.sum()
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isfinite(total) or total <= 0):
        raise ValueError("Sample weights must be finite, nonnegative, aligned, and have positive sum")
    return weights / total

def uniform(n):
    """One weight per training observation, in chronological order."""
    return np.ones(n)

def exponential(n, half_life=63):
    """Trading-observation half-life; newest observation has weight one."""
    if not np.isfinite(half_life) or half_life <= 0 or n < 1:
        raise ValueError("half_life and observation count must be positive")
    return np.exp2((np.arange(n) - n + 1) / half_life)
