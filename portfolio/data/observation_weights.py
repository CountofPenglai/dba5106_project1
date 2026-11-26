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
