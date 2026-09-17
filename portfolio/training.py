"""Compose preprocessing outputs, linear fitting, and portfolio adjustment."""

from dataclasses import dataclass

import pandas as pd

from .losses import weighted_mse
from .model import LinearModel, RegressionProblem, fit_linear, fit_linear_cv
from .regularization import Regularization
from .weights import uniform, validate_portfolio_weights
from .constraints import PortfolioConstraints
from .losses import LogGrowthLoss
from .optimization import fit_growth


@dataclass
class FitResult:
    model: LinearModel
    raw_weights: pd.Series
    weights: pd.Series
    training_loss: float
    objective: float
    alpha: float | None = None
    diagnostics: dict | None = None


def fit_portfolio(history, *, regularization=Regularization(), sample_weights=uniform,
                  adjust_weights=None, model_fit=fit_linear):
    """Fit one historical window; each extension point is an ordinary callable.

    sample_weights(n) -> vector in chronological order.
    adjust_weights(raw_vector) -> final vector in the same asset order.
    model_fit(X, y, regularization=..., sample_weight=...) -> LinearModel.
    """
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must be a DataFrame with unique asset names")
    problem = RegressionProblem.from_returns(history)
    observation_weights = sample_weights(len(history))
    model = model_fit(problem.X, problem.y, regularization=regularization,
                      sample_weight=observation_weights)
    raw = problem.portfolio_weights(model.coefficients)
    validate_portfolio_weights(raw, history.shape[1])
    adjusted = raw.copy() if adjust_weights is None else adjust_weights(raw.copy())
    adjusted = validate_portfolio_weights(adjusted, history.shape[1])
    loss = weighted_mse(problem.y, model.predict(problem.X), observation_weights)
    return FitResult(
        model=model,
        raw_weights=pd.Series(raw, index=history.columns, name="weight"),
        weights=pd.Series(adjusted, index=history.columns, name="weight"),
        training_loss=loss,
        objective=loss + regularization.penalty(model.coefficients),
    )


def fit_portfolio_cv(history, *, kind, alphas=None, cv=5, sample_weights=uniform, adjust_weights=None):
    """Like fit_portfolio, but select alpha by cross-validation on this window.

    Re-run inside every rolling window so lambda tracks the current data
    instead of staying fixed at one guessed value for the whole backtest.
    """
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must be a DataFrame with unique asset names")
    problem = RegressionProblem.from_returns(history)
    observation_weights = sample_weights(len(history))
    model, alpha = fit_linear_cv(problem.X, problem.y, kind=kind, alphas=alphas, cv=cv,
                                 sample_weight=observation_weights)
    raw = problem.portfolio_weights(model.coefficients)
    validate_portfolio_weights(raw, history.shape[1])
    adjusted = raw.copy() if adjust_weights is None else adjust_weights(raw.copy())
    adjusted = validate_portfolio_weights(adjusted, history.shape[1])
    loss = weighted_mse(problem.y, model.predict(problem.X), observation_weights)
    return FitResult(
        model=model,
        raw_weights=pd.Series(raw, index=history.columns, name="weight"),
        weights=pd.Series(adjusted, index=history.columns, name="weight"),
        training_loss=loss,
        objective=loss,
        alpha=alpha,
    )


def fit_portfolio_growth(history, *, regularization=Regularization("lasso", 1e-5),
                         loss=LogGrowthLoss(), constraints=PortfolioConstraints(),
                         sample_weights=uniform):
    """Fit fixed-strength regularization with a log-growth reward and limits.

    No CV or post-fit EW mixing is performed. Both loss and constraints act on
    the portfolio that is returned. All estimates use the supplied history.
    """
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must be a DataFrame with unique asset names")
    problem = RegressionProblem.from_returns(history)
    model, diagnostics = fit_growth(problem, regularization=regularization, loss=loss,
                                   constraints=constraints, sample_weight=sample_weights(len(history)))
    weights = pd.Series(problem.portfolio_weights(model.coefficients),
                        index=history.columns, name="weight")
    return FitResult(model=model, raw_weights=weights.copy(), weights=weights,
                     training_loss=diagnostics["loss"],
                     objective=diagnostics["loss"] + diagnostics["penalty"],
                     diagnostics=diagnostics)
