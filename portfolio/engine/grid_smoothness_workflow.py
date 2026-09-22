"""Compose preprocessing, the grid-smoothness fit, and portfolio export."""
import pandas as pd

from portfolio.constraints import PortfolioConstraints
from portfolio.data.observation_weights import uniform
from portfolio.engine.portfolio_fit import FitResult
from portfolio.losses import LogGrowthLoss
from portfolio.models.grid_smoothness_portfolio import fit_growth_smooth
from portfolio.models.min_variance_regression import RegressionProblem
from portfolio.regularization import Regularization


def fit_portfolio_growth_smooth(history, *, regularization=Regularization("lasso", 1e-5),
                                loss=LogGrowthLoss(), constraints=PortfolioConstraints(),
                                rho=0.0, incidence=None, sample_weights=uniform):
    """Fit fixed-strength regularization plus a grid-smoothness penalty.

    Same conventions as fit_portfolio_growth: no CV or post-fit EW mixing;
    incidence columns must be aligned with history.columns.
    """
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must be a DataFrame with unique asset names")
    problem = RegressionProblem.from_returns(history)
    observation_weights = sample_weights(len(history))
    model, diagnostics = fit_growth_smooth(
        problem, regularization=regularization, loss=loss, constraints=constraints,
        rho=rho, incidence=incidence, sample_weight=observation_weights)
    weights = pd.Series(problem.portfolio_weights(model.coefficients),
                        index=history.columns, name="weight")
    return FitResult(model=model, raw_weights=weights.copy(), weights=weights,
                     training_loss=diagnostics["loss"],
                     objective=diagnostics["loss"] + diagnostics["penalty"],
                     diagnostics=diagnostics)
