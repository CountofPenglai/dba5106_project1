"""Portfolio specifications used in the final report."""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from portfolio.constraints import PortfolioConstraints
from portfolio.engine.portfolio_fit import fit_portfolio
from portfolio.losses import PortfolioLoss
from portfolio.models.constrained_portfolio import fit_constrained_portfolio
from portfolio.models.course_cross_validation import fit_course_cv
from portfolio.models.fit_result import FitResult
from portfolio.models.min_variance_regression import LinearModel, RegressionProblem
from portfolio.regularization import Regularization


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    family: str = 'baseline'
    regularization: Regularization = Regularization()
    constraints: PortfolioConstraints = PortfolioConstraints()
    loss: PortfolioLoss = PortfolioLoss()


COURSE_MODELS = {spec.name: spec for spec in (
    ModelSpec('EW', 'Equal Weight'),
    ModelSpec('MinVar', 'Minimum Variance'),
    ModelSpec('LASSO_CV', 'Course LASSO', 'cv', Regularization('lasso')),
    ModelSpec('Ridge_CV', 'Course Ridge', 'cv', Regularization('ridge')),
)}
EXPOSURE = ModelSpec('LASSO_ExposurePenalty', 'Exposure-penalized LASSO', 'constrained',
                     Regularization('lasso', 1e-5), loss=PortfolioLoss(exposure_strength=2))
CAPPED = replace(EXPOSURE, name='LASSO_WeightCap_10pct', title='LASSO with a 10% asset cap',
                 constraints=PortfolioConstraints(weight_limit=.1))
MODELS = {**COURSE_MODELS, **{s.name: s for s in (
    EXPOSURE, CAPPED, ModelSpec('HistoricalTop20_EW', 'Historical top 20 in equal weights', 'ranking'),
    ModelSpec('LP_Return', 'LASSO forecast with risk-constrained LP', 'lp'),
)}}


def fit_model(history, spec):
    if spec.family == 'cv':
        return fit_course_cv(history, spec.regularization.kind)
    if spec.family == 'constrained':
        return fit_constrained_portfolio(history, regularization=spec.regularization,
                                         loss=spec.loss, constraints=spec.constraints)
    if spec.name == 'MinVar':
        return fit_portfolio(history)
    if spec.name == 'EW':
        problem = RegressionProblem.from_returns(history)
        model = LinearModel(np.zeros(history.shape[1]-1), float(problem.y.mean()))
        variance = float(problem.y.var())
        return FitResult(model, pd.Series(problem.equal_weights, index=history.columns, name='weight'),
                         variance, variance, {})
    raise ValueError(f'{spec.name} requires a ranking or forecast allocation rule')
