"""Training loss definitions, independent of fitting and financial metrics."""

import numpy as np

from .weights import normalize_sample_weights


def weighted_mse(y, prediction, sample_weight=None):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (y.ndim != 1 or len(y) == 0 or prediction.shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()):
        raise ValueError("Targets and predictions must be aligned finite vectors")
    weights = normalize_sample_weights(sample_weight, len(y))
    return float(weights @ np.square(y - prediction))
