"""Time-ordered selection of the native sklearn alpha in each course window."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from portfolio.models.min_variance_regression import LinearModel, RegressionProblem
from portfolio.models.fit_result import FitResult
from portfolio.decisions.holding_weights import validate_portfolio_weights


COURSE_ALPHAS = np.logspace(-8, 8, 21)
CV_SPLITS = 3
CV_DAYS = 21


def fit_course_cv(history, kind):
    problem = RegressionProblem.from_returns(history)
    if len(history) < CV_SPLITS * CV_DAYS + 2:
        raise ValueError("Course CV needs three 21-day folds and at least two initial training days")
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_DAYS)
    if kind == "lasso":
        estimator = LassoCV(alphas=COURSE_ALPHAS, cv=cv, max_iter=1000000, tol=1e-5)
    elif kind == "ridge":
        estimator = RidgeCV(alphas=COURSE_ALPHAS, cv=cv, scoring="neg_mean_squared_error")
    else:
        raise ValueError("Course CV supports LASSO and Ridge")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(problem.X, problem.y)
    alpha = float(estimator.alpha_)
    diagnostics = {"selected_sklearn_alpha": alpha}
    model = LinearModel(estimator.coef_, float(estimator.intercept_))
    weights = validate_portfolio_weights(problem.portfolio_weights(model.coefficients), history.shape[1])
    mse = float(np.mean((problem.y - model.predict(problem.X)) ** 2))
    # Express the native estimator's penalty on the project's mean-squared-error scale.
    penalty = (2 * alpha * np.abs(model.coefficients).sum() if kind == "lasso"
               else alpha * (model.coefficients @ model.coefficients) / len(history))
    return FitResult(model, pd.Series(weights, index=history.columns, name="weight"), mse, mse + penalty,
                     diagnostics)
