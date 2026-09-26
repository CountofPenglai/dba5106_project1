"""Compose preprocessing outputs, linear fitting, and portfolio adjustment."""

from dataclasses import dataclass

import pandas as pd

from portfolio.losses import weighted_mse
from portfolio.models.min_variance_regression import LinearModel, RegressionProblem, fit_linear, fit_linear_cv
from portfolio.regularization import Regularization
from portfolio.data.observation_weights import uniform
from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.constraints import CoefficientConstraint, PortfolioConstraints
from portfolio.models.coefficient_constrained_regression import fit_coefficient_constrained
from portfolio.losses import LogGrowthLoss
from portfolio.models.constrained_regularized_portfolio import fit_growth


@dataclass
class FitResult:
    model: LinearModel
    raw_weights: pd.Series
    weights: pd.Series
    training_loss: float
    objective: float
    alpha: float | None = None
    diagnostics: dict | None = None


def _prepare_history(history, sample_weights):
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must be a DataFrame with unique asset names")
    return RegressionProblem.from_returns(history), sample_weights(len(history))


def _fit_result(history, problem, model, observation_weights, *,
                adjust_weights=None, penalty=0.0, alpha=None):
    raw = validate_portfolio_weights(problem.portfolio_weights(model.coefficients), history.shape[1])
    adjusted = raw.copy()
    if adjust_weights is not None:
        adjusted = validate_portfolio_weights(adjust_weights(adjusted), history.shape[1])
    loss = weighted_mse(problem.y, model.predict(problem.X), observation_weights)
    return FitResult(
        model=model,
        raw_weights=pd.Series(raw, index=history.columns, name="weight"),
        weights=pd.Series(adjusted, index=history.columns, name="weight"),
        training_loss=loss, objective=loss + penalty, alpha=alpha,
    )


def fit_portfolio(history, *, regularization=Regularization(), sample_weights=uniform,
                  adjust_weights=None, model_fit=fit_linear):
    """Fit a fixed-penalty min-variance regression, then optionally adjust holdings.

    sample_weights(n) weights historical observations; adjust_weights(w) acts on
    the fitted portfolio. model_fit is an ordinary regression-fitting callable.
    """
    problem, observation_weights = _prepare_history(history, sample_weights)
    model = model_fit(problem.X, problem.y, regularization=regularization,
                      sample_weight=observation_weights)
    return _fit_result(history, problem, model, observation_weights,
                       adjust_weights=adjust_weights,
                       penalty=regularization.penalty(model.coefficients))


def fit_portfolio_cv(history, *, kind, alphas=None, cv=5, sample_weights=uniform, adjust_weights=None):
    """Reselect sklearn alpha inside this window; keep legacy CV scoring intact.

    As before, objective records prediction loss alone for CV variants. Fixed
    variants record prediction loss plus the explicitly configured penalty.
    """
    problem, observation_weights = _prepare_history(history, sample_weights)
    model, alpha = fit_linear_cv(problem.X, problem.y, kind=kind, alphas=alphas, cv=cv,
                                sample_weight=observation_weights)
    return _fit_result(history, problem, model, observation_weights,
                       adjust_weights=adjust_weights, alpha=alpha)


def fit_portfolio_growth(history, *, regularization=Regularization("lasso", 1e-5),
                         loss=LogGrowthLoss(), constraints=PortfolioConstraints(),
                         sample_weights=uniform):
    """Fit fixed-strength regularization with a log-growth reward and limits.

    No CV or post-fit EW mixing is performed. Both loss and constraints act on
    the portfolio that is returned. All estimates use the supplied history.
    """
    problem, observation_weights = _prepare_history(history, sample_weights)
    model, diagnostics = fit_growth(problem, regularization=regularization, loss=loss,
                                   constraints=constraints, sample_weight=observation_weights)
    weights = pd.Series(problem.portfolio_weights(model.coefficients),
                        index=history.columns, name="weight")
    return FitResult(model=model, raw_weights=weights.copy(), weights=weights,
                     training_loss=diagnostics["loss"],
                     objective=diagnostics["loss"] + diagnostics["penalty"],
                     diagnostics=diagnostics)


def fit_portfolio_coefficient_constrained(history, *, constraint: CoefficientConstraint,
                                          sample_weights=uniform):
    """Fit minimum variance with an explicit coefficient budget, no penalty."""
    problem, observation_weights = _prepare_history(history, sample_weights)
    model, diagnostics = fit_coefficient_constrained(
        problem.X, problem.y, constraint=constraint, sample_weight=observation_weights)
    result = _fit_result(history, problem, model, observation_weights)
    result.diagnostics = diagnostics
    return result
