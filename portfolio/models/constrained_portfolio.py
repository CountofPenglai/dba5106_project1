"""Exposure-penalized LASSO, optionally with an absolute per-asset weight cap."""
import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio.constraints import PortfolioConstraints
from portfolio.data.observation_weights import normalize_sample_weights
from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.engine.convex_solver import solve_checked
from portfolio.losses import PortfolioLoss
from portfolio.models.fit_result import FitResult
from portfolio.models.min_variance_regression import LinearModel, RegressionProblem
from portfolio.regularization import Regularization


def fit_constrained_portfolio(history, *, regularization=Regularization('lasso', 1e-5),
                              loss=PortfolioLoss(), constraints=PortfolioConstraints(),
                              sample_weight=None):
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError('History must have unique asset columns')
    problem = RegressionProblem.from_returns(history)
    n, p = history.shape
    if constraints.weight_limit is not None and constraints.weight_limit * p < 1 - 1e-12:
        raise ValueError('Weight limit is infeasible for full investment')
    a = normalize_sample_weights(sample_weight, n)
    if np.count_nonzero(a) < 2:
        raise ValueError('At least two observations need positive weight')
    X, y = problem.X, problem.y
    beta = cp.Variable(p - 1)
    holdings = problem.equal_weights - problem.basis @ beta
    scenario_returns = y - X @ beta
    centered_y, centered_X = y - a @ y, X - a @ X
    deviations = cp.multiply(np.sqrt(a), centered_y - centered_X @ beta)
    variance = cp.sum_squares(deviations)
    ew_variance = float(a @ centered_y**2)
    risk_limit = constraints.volatility_limit**2 * ew_variance
    if regularization.kind != 'lasso':
        raise ValueError('The exposure-penalized portfolio uses LASSO')
    objective = variance + regularization.strength * cp.norm1(beta)
    if loss.exposure_strength and constraints.leverage_limit > 1 and ew_variance > 0:
        excess = cp.pos((cp.norm1(holdings)-1)/(constraints.leverage_limit-1))
        objective += loss.exposure_strength * ew_variance * cp.square(excess)
    restrictions = [cp.norm1(holdings) <= constraints.leverage_limit,
                    1 + scenario_returns >= constraints.wealth_floor]
    restrictions.append(deviations == 0 if risk_limit == 0 else
                            cp.norm(100*deviations, 2) <= 100*np.sqrt(risk_limit))
    if constraints.weight_limit is not None:
        restrictions.append(cp.abs(holdings) <= constraints.weight_limit)
    optimization = cp.Problem(cp.Minimize(objective), restrictions)

    def checked(coefficients):
        weights = validate_portfolio_weights(problem.portfolio_weights(coefficients), p)
        actual = history.to_numpy() @ weights
        terms = loss.components(actual, a)
        gross = float(np.abs(weights).sum())
        min_wealth = float(np.min(1 + actual))
        if (gross > constraints.leverage_limit + 1e-7
                or terms['variance'] > risk_limit + (1e-10 + 1e-6*risk_limit)
                or min_wealth < constraints.wealth_floor - 1e-7
                or (constraints.weight_limit is not None
                    and np.abs(weights).max() > constraints.weight_limit + 1e-7)):
            raise ValueError('Portfolio failed independent constraint checks')
        excess = max(0., (gross-1)/(constraints.leverage_limit-1))**2 if constraints.leverage_limit > 1 else 0.
        exposure_penalty = loss.exposure_strength * ew_variance * excess
        fitted_loss = terms['loss'] + exposure_penalty
        return {**terms, 'loss': fitted_loss, 'penalty': regularization.penalty(coefficients),
                'exposure_unit_penalty': excess, 'exposure_scale': ew_variance,
                'exposure_penalty': exposure_penalty,
                'gross_leverage': gross,
                'max_abs_weight': float(np.abs(weights).max()),
                'ew_variance': ew_variance, 'variance_limit': risk_limit, 'min_wealth': min_wealth}

    solvers = [
        ('CLARABEL', dict(max_iter=300, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)),
        ('SCS', dict(eps=1e-8, max_iters=100000)),
    ]
    coefficients, solver = solve_checked(optimization, beta, checked, solvers=solvers,
                                         stop_on_infeasible=True, error_label='Constrained portfolio failed')
    diagnostics = {**checked(coefficients), **solver}
    weights = pd.Series(problem.portfolio_weights(coefficients), index=history.columns, name='weight')
    return FitResult(LinearModel(coefficients, diagnostics['mean']), weights,
                     diagnostics['loss'], diagnostics['loss']+diagnostics['penalty'], diagnostics)
