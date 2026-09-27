"""Fit the fixed-penalty and explicit coefficient-budget baselines."""
import pandas as pd
from portfolio.constraints import CoefficientConstraint
from portfolio.data.observation_weights import uniform
from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.losses import weighted_mse
from portfolio.models.coefficient_constrained_regression import fit_coefficient_constrained
from portfolio.models.fit_result import FitResult
from portfolio.models.min_variance_regression import RegressionProblem, fit_linear
from portfolio.regularization import Regularization


def _prepare_history(history, sample_weights):
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must be a DataFrame with unique asset names")
    return RegressionProblem.from_returns(history), sample_weights(len(history))


def _result(history, problem, model, observation_weights, penalty=0., diagnostics=None):
    weights = validate_portfolio_weights(problem.portfolio_weights(model.coefficients), history.shape[1])
    loss = weighted_mse(problem.y, model.predict(problem.X), observation_weights)
    return FitResult(model, pd.Series(weights, index=history.columns, name="weight"),
                     loss, loss + penalty, diagnostics)


def fit_portfolio(history, *, regularization=Regularization(), sample_weights=uniform):
    problem, observation_weights = _prepare_history(history, sample_weights)
    model = fit_linear(problem.X, problem.y, regularization=regularization,
                       sample_weight=observation_weights)
    return _result(history, problem, model, observation_weights,
                   penalty=regularization.penalty(model.coefficients))


def fit_portfolio_coefficient_constrained(history, *, constraint: CoefficientConstraint,
                                          sample_weights=uniform):
    problem, observation_weights = _prepare_history(history, sample_weights)
    model, diagnostics = fit_coefficient_constrained(
        problem.X, problem.y, constraint=constraint, sample_weight=observation_weights)
    return _result(history, problem, model, observation_weights, diagnostics=diagnostics)
