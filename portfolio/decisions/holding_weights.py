"""Validate holdings and optionally shrink a portfolio toward equal weight."""
import numpy as np

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
