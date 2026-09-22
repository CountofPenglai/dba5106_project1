"""History-only risk estimates and feasible, interpretable candidate portfolios."""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio.constraints import PortfolioConstraints
from portfolio.engine.convex_solver import covariance_factor, solve_checked
from portfolio.decisions.holding_weights import validate_portfolio_weights


@dataclass
class RiskContext:
    covariance: np.ndarray
    factor: np.ndarray
    history: np.ndarray
    constraints: PortfolioConstraints
    equal_weights: np.ndarray
    variance_limit: float
    shrinkage: float


def check_candidate(context, weights):
    w = validate_portfolio_weights(weights, len(context.equal_weights))
    gross = float(np.abs(w).sum())
    variance = float(w @ context.covariance @ w)
    wealth = float(np.min(1 + context.history @ w))
    if gross > context.constraints.leverage_limit + 1e-7:
        raise ValueError('Candidate violates gross leverage limit')
    if variance > context.variance_limit * (1 + 1e-6):
        raise ValueError('Candidate violates estimated variance limit')
    if wealth < context.constraints.wealth_floor - 1e-8:
        raise ValueError('Candidate violates historical wealth floor')
    return dict(gross_leverage=gross, estimated_variance=variance,
                variance_limit=context.variance_limit, min_wealth=wealth,
                budget_error=float(abs(w.sum()-1)))


def estimate_risk(history, constraints=PortfolioConstraints()):
    if isinstance(history, pd.DataFrame) and history.columns.has_duplicates:
        raise ValueError('Risk history has duplicate assets')
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError('Risk history needs finite aligned rows and at least two assets/days')
    risk = LedoitWolf().fit(values)
    covariance = np.asarray(risk.covariance_)
    equal = np.full(values.shape[1], 1/values.shape[1])
    limit = float(equal @ covariance @ equal)*constraints.volatility_limit**2
    if limit <= 1e-16:
        raise ValueError('Risk estimate is degenerate')
    result = RiskContext(covariance, covariance_factor(covariance), values.copy(),
                         constraints, equal, limit, float(risk.shrinkage_))
    check_candidate(result, equal)
    return result


def candidate_portfolio(context, *, objective, expected_returns=None):
    if objective not in {'return', 'risk'}:
        raise ValueError('Candidate objective must be return or risk')
    n = len(context.equal_weights)
    mu = None
    if objective == 'return':
        mu = np.asarray(expected_returns, dtype=float)
        if mu.shape != (n,) or not np.isfinite(mu).all():
            raise ValueError('Expected returns must be finite and aligned with all assets')
    # Full covariance shrinkage can make Sigma=s*I. At the EW risk cap,
    # Cauchy-Schwarz proves EW is the ONLY feasible fully invested portfolio.
    # This is an analytic optimum, not a fallback after numerical failure.
    isotropic = np.array_equal(context.covariance, np.eye(n)*context.covariance[0, 0])
    ew_variance = float(context.equal_weights @ context.covariance @ context.equal_weights)
    if isotropic and context.variance_limit <= ew_variance*(1+1e-14):
        values = context.equal_weights.copy()
        info = dict(solver='analytic', solver_status='optimal',
                    solver_attempts='EW is uniquely feasible at isotropic minimum-risk budget',
                    primary_solver='analytic', primary_solver_attempts='unique feasible portfolio',
                    primary_score=float(mu @ values*1e4) if objective == 'return' else -1.,
                    secondary_gap_scaled=0., risk_shrinkage=context.shrinkage, objective=objective,
                    risk_dual=0., unique_optimum=True)
        if objective == 'return': info['expected_return'] = float(mu @ values)
        return values, {**check_candidate(context, values), **info}
    w = cp.Variable(n)
    factor = context.factor / np.sqrt(context.variance_limit)
    restrictions = [cp.sum(w) == 1,
                    cp.norm1(w) <= context.constraints.leverage_limit,
                    cp.norm(factor @ w) <= 1,
                    1 + context.history @ w >= context.constraints.wealth_floor]
    criterion = cp.sum_squares(factor @ w) if objective == 'risk' else -(mu*1e4) @ w
    primary = cp.Problem(cp.Minimize(criterion), restrictions)
    check = lambda values: check_candidate(context, values)
    values, primary_info = solve_checked(primary, w, check)
    best = float(values @ context.covariance @ values / context.variance_limit
                 if objective == 'risk' else -(mu*1e4) @ values)
    # Positive-definite minimum variance has a unique solution. A linear return
    # objective can have a face of ties, which we resolve by closeness to EW.
    gap = 0.
    info = primary_info
    risk_dual = float(restrictions[2].dual_value)
    positive_definite = np.linalg.eigvalsh(context.covariance).min() > 1e-16
    # With a positive risk multiplier, the Lagrangian is strictly convex on
    # sum(w)=1 for PD Sigma; consequently its feasible return optimum is unique.
    # No secondary selection among ties is needed in that case.
    unique = positive_definite and (objective == 'risk' or risk_dual > 1e-6)
    if not unique:
        tolerance = 1e-5 if objective == 'return' else 1e-7
        secondary = cp.Problem(cp.Minimize(cp.sum_squares(w-context.equal_weights)),
                               restrictions + [criterion <= best+tolerance])
        def check_secondary(x):
            check(x)
            val = float(x @ context.covariance @ x / context.variance_limit
                        if objective == 'risk' else -(mu*1e4) @ x)
            if val > best+tolerance+1e-7:
                raise ValueError('Secondary candidate changes the primary objective')
        equal_score = float(context.equal_weights @ context.covariance @ context.equal_weights / context.variance_limit
                            if objective == 'risk' else -(mu*1e4) @ context.equal_weights)
        if equal_score <= best+tolerance:
            values = context.equal_weights.copy()
            info = dict(solver='analytic', solver_status='optimal', solver_attempts='EW solves secondary tie-break')
        else:
            values, info = solve_checked(secondary, w, check_secondary)
        gap = float(values @ context.covariance @ values / context.variance_limit-best
                    if objective == 'risk' else -(mu*1e4) @ values-best)
    diagnostics = {**check(values), **info, 'objective': objective,
                   'primary_solver': primary_info['solver'],
                   'primary_solver_attempts': primary_info['solver_attempts'],
                   'primary_score': -best, 'secondary_gap_scaled': gap,
                   'risk_shrinkage': context.shrinkage, 'risk_dual': risk_dual, 'unique_optimum': unique}
    if objective == 'return':
        diagnostics['expected_return'] = float(mu @ values)
    return values, diagnostics
