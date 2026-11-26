"""Linear regression architecture and the portfolio parameterization."""

from dataclasses import dataclass

import numpy as np


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
