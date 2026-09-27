"""Validate fully invested asset holdings."""
import numpy as np

def validate_portfolio_weights(weights, n):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all():
        raise ValueError("Portfolio weights must be finite and aligned with assets")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Portfolio weights must sum to one")
    return weights
