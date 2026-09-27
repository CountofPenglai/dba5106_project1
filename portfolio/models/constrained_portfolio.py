"""One constrained regression for hard limits, exposure, turnover, caps and beta.

The coefficient vector maps to fully invested holdings as w = EW - N @ beta.
The market-beta experiments retain their original CVXPY variance formulation
and solver defaults so that their report records remain reproducible.
"""
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


def market_betas(history):
    values = history.to_numpy(dtype=float)
    market = values.mean(axis=1)
    centered_market = market - market.mean()
    denominator = float(centered_market @ centered_market)
    if denominator <= 0:
        raise ValueError("Market beta requires a nonconstant EW benchmark")
    return (values - values.mean(axis=0)).T @ centered_market / denominator


def fit_constrained_portfolio(history, *, regularization=Regularization('lasso', 1e-5),
                              loss=PortfolioLoss(), constraints=PortfolioConstraints(),
                              previous_weights=None, market_beta=False, sample_weight=None):
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError('History must have unique asset columns')
    problem = RegressionProblem.from_returns(history)
    n, p = history.shape
    if constraints.weight_limit is not None and constraints.weight_limit * p < 1 - 1e-12:
        raise ValueError('Weight limit is infeasible for full investment')
    previous = problem.equal_weights
    if previous_weights is not None:
        if (not isinstance(previous_weights, pd.Series) or previous_weights.index.has_duplicates
                or set(previous_weights.index) != set(history.columns)):
            raise ValueError('Previous target weights must have the same unique assets')
        previous = validate_portfolio_weights(previous_weights.reindex(history.columns), p)
    a = normalize_sample_weights(sample_weight, n)
    if np.count_nonzero(a) < 2:
        raise ValueError('At least two observations need positive weight')
    if market_beta and not np.allclose(a, 1/n):
        raise ValueError('Report market-beta fitting uses equal observation weights')

    X, y = problem.X, problem.y
    beta = cp.Variable(p - 1)
    holdings = problem.equal_weights - problem.basis @ beta
    scenario_returns = y - X @ beta
    centered_y, centered_X = y - a @ y, X - a @ X
    deviations = cp.multiply(np.sqrt(a), centered_y - centered_X @ beta)
    variance = cp.sum_squares(deviations)
    ew_variance = float(a @ centered_y**2)
    risk_limit = constraints.volatility_limit**2 * ew_variance
    asset_betas = market_betas(history) if market_beta or constraints.beta_bounds else None
    if market_beta:
        # Preserve the formulation underlying Table 7 (not a different model).
        scenario_returns = history.to_numpy() @ holdings
        variance = cp.sum_squares(scenario_returns - cp.sum(scenario_returns)/n) / n
        ew_variance = float(np.mean((y-y.mean())**2))
        risk_limit = constraints.volatility_limit**2 * ew_variance

    coefficient_penalty = 0
    if regularization.kind == 'lasso':
        coefficient_penalty = regularization.strength * cp.norm1(beta)
    elif regularization.kind == 'ridge':
        coefficient_penalty = regularization.strength * cp.sum_squares(beta)
    objective = variance + coefficient_penalty
    if loss.reward_strength:
        objective -= loss.reward_strength * (a @ cp.log(1 + scenario_returns))
    if loss.exposure_strength and constraints.leverage_limit > 1 and ew_variance > 0:
        excess = cp.pos((cp.norm1(holdings)-1)/(constraints.leverage_limit-1))
        objective += loss.exposure_strength * ew_variance * cp.square(excess)
    if loss.turnover_strength:
        objective += loss.turnover_strength * ew_variance * .5 * cp.norm1(holdings-previous)
    restrictions = [cp.norm1(holdings) <= constraints.leverage_limit,
                    1 + scenario_returns >= constraints.wealth_floor]
    if market_beta:
        restrictions.append(variance <= risk_limit)
    else:
        restrictions.append(deviations == 0 if risk_limit == 0 else
                            cp.norm(100*deviations, 2) <= 100*np.sqrt(risk_limit))
    if constraints.weight_limit is not None:
        restrictions.append(cp.abs(holdings) <= constraints.weight_limit)
    if constraints.beta_bounds is not None:
        lower, upper = constraints.beta_bounds
        restrictions.extend([asset_betas @ holdings >= lower, asset_betas @ holdings <= upper])
    optimization = cp.Problem(cp.Minimize(objective), restrictions)

    def checked(coefficients):
        weights = validate_portfolio_weights(problem.portfolio_weights(coefficients), p)
        actual = history.to_numpy() @ weights
        terms = loss.components(actual, a)
        gross = float(np.abs(weights).sum())
        min_wealth = float(np.min(1 + actual))
        fitted_beta = None if asset_betas is None else float(asset_betas @ weights)
        # The published beta runs use solver defaults; allow their documented
        # numerical residual while still checking actual holdings independently.
        tolerance = 2e-6 if market_beta else 1e-7
        if (gross > constraints.leverage_limit + tolerance
                or terms['variance'] > risk_limit + (1e-8 if market_beta else 1e-10 + 1e-6*risk_limit)
                or min_wealth < constraints.wealth_floor - tolerance
                or (constraints.weight_limit is not None
                    and np.abs(weights).max() > constraints.weight_limit + tolerance)
                or (constraints.beta_bounds is not None and not
                    constraints.beta_bounds[0]-tolerance <= fitted_beta <= constraints.beta_bounds[1]+tolerance)):
            raise ValueError('Portfolio failed independent constraint checks')
        excess = max(0., (gross-1)/(constraints.leverage_limit-1))**2 if constraints.leverage_limit > 1 else 0.
        movement = float(.5*np.abs(weights-previous).sum())
        exposure_penalty = loss.exposure_strength * ew_variance * excess
        turnover_penalty = loss.turnover_strength * ew_variance * movement
        fitted_loss = terms['loss'] + exposure_penalty + turnover_penalty
        return {**terms, 'loss': fitted_loss, 'penalty': regularization.penalty(coefficients),
                'exposure_unit_penalty': excess, 'exposure_scale': ew_variance,
                'exposure_penalty': exposure_penalty, 'turnover_penalty': turnover_penalty,
                'target_turnover': movement, 'gross_leverage': gross,
                'max_abs_weight': float(np.abs(weights).max()), 'market_beta': fitted_beta,
                'ew_variance': ew_variance, 'variance_limit': risk_limit, 'min_wealth': min_wealth}

    solvers = [('CLARABEL', {}), ('SCS', {})] if market_beta else [
        ('CLARABEL', dict(max_iter=300, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)),
        ('SCS', dict(eps=1e-8, max_iters=100000)),
    ]
    coefficients, solver = solve_checked(optimization, beta, checked, solvers=solvers,
                                         stop_on_infeasible=True, error_label='Constrained portfolio failed')
    diagnostics = {**checked(coefficients), **solver}
    weights = pd.Series(problem.portfolio_weights(coefficients), index=history.columns, name='weight')
    return FitResult(LinearModel(coefficients, diagnostics['mean']), weights,
                     diagnostics['loss'], diagnostics['loss']+diagnostics['penalty'], diagnostics)
