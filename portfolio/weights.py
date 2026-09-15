"""Independent sample-weight and portfolio-weight functions."""

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


def validate_portfolio_weights(weights, n):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all():
        raise ValueError("Portfolio weights must be finite and aligned with assets")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Portfolio weights must sum to one")
    return weights


def shrink_to_equal(weights, strength=0.0):
    """Mix a fully invested portfolio with EW; negative weights remain allowed."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("Need a vector of at least two asset weights")
    validate_portfolio_weights(weights, len(weights))
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Shrinkage strength must be between zero and one")
    return (1 - strength) * weights + strength / len(weights)
