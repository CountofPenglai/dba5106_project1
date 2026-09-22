"""PCA-reduced LASSO and Ridge portfolio fitting."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge


LASSO_STRENGTH = 1e-5
RIDGE_STRENGTH = 1e-4


def fit_weights(history, n_components, model_type, *,
                lasso_strength=LASSO_STRENGTH, ridge_strength=RIDGE_STRENGTH):
    """Fit PCA followed by the selected legacy regularized regression."""
    returns = history.to_numpy()
    n_assets = returns.shape[1]
    equal_weights = np.full(n_assets, 1 / n_assets)
    null_basis = np.vstack([np.eye(n_assets - 1), -np.ones(n_assets - 1)])
    features = returns @ null_basis
    target = returns @ equal_weights

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features)
    if model_type == "LASSO":
        model = Lasso(alpha=lasso_strength / 2, fit_intercept=True,
                      max_iter=100000, tol=1e-8)
    elif model_type == "Ridge":
        model = Ridge(alpha=ridge_strength / 2, fit_intercept=True)
    else:
        raise ValueError("model_type must be 'LASSO' or 'Ridge'")
    model.fit(components, target)
    beta = pca.components_.T @ model.coef_
    return equal_weights - null_basis @ beta
