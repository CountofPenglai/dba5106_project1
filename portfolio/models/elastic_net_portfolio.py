"""Elastic Net portfolio fitting under the full-investment constraint."""

import numpy as np
from sklearn.linear_model import ElasticNet


STRENGTH = 1e-5


def fit_weights(history, l1_ratio, *, strength=STRENGTH):
    """Fit the legacy Elastic Net regression and return asset weights."""
    returns = history.to_numpy()
    n_assets = returns.shape[1]
    equal_weights = np.full(n_assets, 1 / n_assets)
    null_basis = np.vstack([np.eye(n_assets - 1), -np.ones(n_assets - 1)])
    model = ElasticNet(
        alpha=strength / 2,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=100000,
        tol=1e-8,
    )
    model.fit(returns @ null_basis, returns @ equal_weights)
    return equal_weights - null_basis @ model.coef_
