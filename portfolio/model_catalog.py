"""Named report configurations: six baselines and thirteen report comparisons.

Configuration differences share fitting code. Report sections use the numbering
in DBA5106_Final_Report_Draft 2.docx. No model selection happens in this catalog.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from portfolio.constraints import CoefficientConstraint, PortfolioConstraints
from portfolio.engine.portfolio_fit import fit_portfolio, fit_portfolio_coefficient_constrained
from portfolio.losses import PortfolioLoss
from portfolio.models.constrained_portfolio import fit_constrained_portfolio
from portfolio.models.fit_result import FitResult
from portfolio.models.min_variance_regression import LinearModel, RegressionProblem
from portfolio.regularization import Regularization


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    section: str
    family: str = 'baseline'
    regularization: Regularization = Regularization()
    coefficient_constraint: CoefficientConstraint | None = None
    constraints: PortfolioConstraints = PortfolioConstraints()
    loss: PortfolioLoss = PortfolioLoss()
    market_beta: bool = False
    transaction_cost: float = 0.


LASSO = Regularization('lasso', 1e-5)
RIDGE = Regularization('ridge', 1e-4)
_specs = [
    ModelSpec('EW', 'Equal Weight', '2'),
    ModelSpec('MinVar', 'Minimum Variance', '2'),
    ModelSpec('LASSO_CoefficientConstraint', 'LASSO (coefficient constraint)', '2',
              coefficient_constraint=CoefficientConstraint('lasso', 1)),
    ModelSpec('Ridge_CoefficientConstraint', 'Ridge (coefficient constraint)', '2',
              coefficient_constraint=CoefficientConstraint('ridge', 1)),
    ModelSpec('LASSO_Penalty', 'LASSO (penalty)', '2', regularization=LASSO),
    ModelSpec('Ridge_Penalty', 'Ridge (penalty)', '2', regularization=RIDGE),
    ModelSpec('LASSO_HardConstraints', 'Hard-Constrained LASSO', '3.1', 'constrained', LASSO),
    ModelSpec('LASSO_ExposurePenalty', 'LASSO with Investment Size Control', '3.2', 'constrained', LASSO,
              loss=PortfolioLoss(exposure_strength=2)),
    ModelSpec('LASSO_Turnover_0p1', 'Turnover-Penalized LASSO (tau=0.1)', '3.3', 'constrained', LASSO,
              loss=PortfolioLoss(exposure_strength=2, turnover_strength=.1)),
    ModelSpec('LASSO_Turnover_1', 'Turnover-Penalized LASSO (tau=1)', '3.3', 'constrained', LASSO,
              loss=PortfolioLoss(exposure_strength=2, turnover_strength=1)),
    ModelSpec('LASSO_WeightCap_10pct', 'LASSO with a 10% Single-Asset Weight Cap', '5.1', 'constrained', LASSO,
              constraints=PortfolioConstraints(weight_limit=.1), loss=PortfolioLoss(exposure_strength=2)),
    ModelSpec('LP_Original', 'LASSO Forecasting + Original LP', '4.1', 'lp'),
    ModelSpec('LP_TradingPenalty', 'LASSO Forecasting + Trading-Penalized LP', '4.1', 'lp', transaction_cost=.0005),
]
for kind, regularization in (('LASSO', LASSO), ('Ridge', RIDGE)):
    for suffix, bounds in (('Unbounded', None), ('Market', (.8, 1.2)), ('Conservative', (.5, .9))):
        _specs.append(ModelSpec(f'{kind}_Beta{suffix}', f'{kind} with Market Beta Control ({suffix})',
                               '5.2', 'constrained', regularization,
                               constraints=PortfolioConstraints(beta_bounds=bounds), market_beta=True))
MODELS = {spec.name: spec for spec in _specs}
BASELINE_NAMES = tuple(spec.name for spec in _specs if spec.family == 'baseline')
JULY_MODELS = ('EW', 'LASSO_ExposurePenalty', 'LASSO_WeightCap_10pct',
               'Ridge_BetaConservative', 'LASSO_BetaMarket', 'LP_Original', 'LP_TradingPenalty')


def baseline_models(*, lasso_budget=1., ridge_budget=1.):
    models = [MODELS[name] for name in BASELINE_NAMES]
    return [replace(spec, coefficient_constraint=CoefficientConstraint(spec.coefficient_constraint.kind,
                    lasso_budget if spec.coefficient_constraint.kind == 'lasso' else ridge_budget))
            if spec.coefficient_constraint else spec for spec in models]


def fit_model(history, spec, *, previous_weights=None):
    """Fit one single-stage model; the LP runner owns forecasting and allocation."""
    if spec.name == 'EW':
        problem = RegressionProblem.from_returns(history)
        model = LinearModel(np.zeros(history.shape[1]-1), float(problem.y.mean()))
        variance = float(problem.y.var())
        return FitResult(model, pd.Series(problem.equal_weights, index=history.columns, name='weight'),
                         variance, variance, {})
    if spec.family == 'constrained':
        return fit_constrained_portfolio(history, regularization=spec.regularization, loss=spec.loss,
                                         constraints=spec.constraints, previous_weights=previous_weights,
                                         market_beta=spec.market_beta)
    if spec.family == 'lp':
        raise ValueError('LP models require the forecasting and allocation runner')
    if spec.coefficient_constraint:
        return fit_portfolio_coefficient_constrained(history, constraint=spec.coefficient_constraint)
    return fit_portfolio(history, regularization=spec.regularization)
