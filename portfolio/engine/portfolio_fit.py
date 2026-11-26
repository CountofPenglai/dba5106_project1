"""Fit the course minimum-variance portfolio regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.models.fit_result import FitResult
from portfolio.models.min_variance_regression import LinearModel, RegressionProblem


def fit_portfolio(history):
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History requires unique asset names")
    problem = RegressionProblem.from_returns(history)
    estimator = LinearRegression().fit(problem.X, problem.y)
    model = LinearModel(estimator.coef_, float(estimator.intercept_))
    weights = validate_portfolio_weights(problem.portfolio_weights(model.coefficients), history.shape[1])
    variance = float(np.square(problem.y - model.predict(problem.X)).mean())
    return FitResult(model, pd.Series(weights, index=history.columns, name="weight"), variance, variance)
