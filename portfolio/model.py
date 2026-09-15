"""Linear regression architecture and the portfolio parameterization."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from .regularization import Regularization
from .weights import normalize_sample_weights


@dataclass
class LinearModel:
    coefficients: np.ndarray
    intercept: float

    def predict(self, X):
        return np.asarray(X) @ self.coefficients + self.intercept


@dataclass
class RegressionProblem:
    X: np.ndarray
    y: np.ndarray
    basis: np.ndarray
    equal_weights: np.ndarray

    @classmethod
    def from_returns(cls, returns):
        R = np.asarray(returns, dtype=float)
        if R.ndim != 2 or min(R.shape) < 2 or not np.isfinite(R).all():
            raise ValueError("Training needs at least two rows/assets and no missing or infinite returns")
        p = R.shape[1]
        equal = np.full(p, 1 / p)
        basis = np.vstack([np.eye(p - 1), -np.ones(p - 1)])
        return cls(R @ basis, R @ equal, basis, equal)

    def portfolio_weights(self, coefficients):
        return self.equal_weights - self.basis @ coefficients


def fit_linear(X, y, *, regularization=Regularization(), sample_weight=None):
    """Minimize weighted MSE + strength*penalty, with an unpenalized intercept.

    sklearn's native objective conventions differ. Normalizing sample weights
    to sum n requires Lasso alpha=strength/2 and Ridge alpha=n*strength.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1
            or y.shape != (X.shape[0],) or not np.isfinite(X).all() or not np.isfinite(y).all()):
        raise ValueError("X and y must be finite and aligned, with at least two observations")
    n = len(y)
    weights = normalize_sample_weights(sample_weight, n) * n
    if np.count_nonzero(weights) < 2:
        raise ValueError("At least two observations need positive sample weight")
    if regularization.kind == "none" or regularization.strength == 0:
        estimator = LinearRegression()
    elif regularization.kind == "lasso":
        estimator = Lasso(alpha=regularization.strength / 2, max_iter=100000, tol=1e-8)
    else:
        estimator = Ridge(alpha=n * regularization.strength)
    estimator.fit(X, y, sample_weight=weights)
    return LinearModel(np.asarray(estimator.coef_), float(estimator.intercept_))
