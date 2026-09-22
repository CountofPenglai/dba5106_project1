"""Constrained regularized portfolios with optional market-beta bounds."""

import cvxpy as cp
import numpy as np


LASSO_STRENGTH = 1e-5
RIDGE_STRENGTH = 1e-4
GROSS_LIMIT = 1.5
WEALTH_FLOOR = 1e-6


def calculate_market_betas(history):
    """Calculate each asset's beta to the equal-weighted benchmark."""
    returns = history.to_numpy()
    n_assets = returns.shape[1]
    equal_weights = np.full(n_assets, 1 / n_assets)
    benchmark_returns = returns @ equal_weights
    centered_benchmark = benchmark_returns - benchmark_returns.mean()
    centered_assets = returns - returns.mean(axis=0)
    denominator = np.sum(centered_benchmark ** 2)
    if denominator <= 0:
        raise ValueError("Equal-weighted benchmark has zero variance.")
    return (centered_assets.T @ centered_benchmark) / denominator, benchmark_returns


def fit_portfolio(history, model_type, beta_bounds=None, *,
                  lasso_strength=LASSO_STRENGTH, ridge_strength=RIDGE_STRENGTH,
                  gross_limit=GROSS_LIMIT, wealth_floor=WEALTH_FLOOR):
    """Fit the legacy constrained LASSO or Ridge market-beta portfolio."""
    returns = history.to_numpy()
    n_obs, n_assets = returns.shape
    equal_weights = np.full(n_assets, 1 / n_assets)
    null_basis = np.vstack([np.eye(n_assets - 1), -np.ones(n_assets - 1)])
    theta = cp.Variable(n_assets - 1)
    weights = equal_weights - null_basis @ theta
    portfolio_returns = returns @ weights
    benchmark_returns = returns @ equal_weights
    market_betas, _ = calculate_market_betas(history)
    portfolio_beta = market_betas @ weights
    portfolio_mean = cp.sum(portfolio_returns) / n_obs
    variance_loss = cp.sum_squares(portfolio_returns - portfolio_mean) / n_obs
    benchmark_variance = np.mean((benchmark_returns - benchmark_returns.mean()) ** 2)
    constraints = [
        cp.norm1(weights) <= gross_limit,
        variance_loss <= benchmark_variance,
        1 + portfolio_returns >= wealth_floor,
    ]
    if beta_bounds is not None:
        lower, upper = beta_bounds
        constraints += [portfolio_beta >= lower, portfolio_beta <= upper]
    if model_type == "LASSO":
        objective = cp.Minimize(variance_loss + lasso_strength * cp.norm1(theta))
    elif model_type == "Ridge":
        objective = cp.Minimize(variance_loss + ridge_strength * cp.sum_squares(theta))
    else:
        raise ValueError("model_type must be LASSO or Ridge.")
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS, verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Optimization failed: {problem.status}")
    fitted_weights = np.asarray(weights.value).reshape(-1)
    return fitted_weights, float(market_betas @ fitted_weights)
