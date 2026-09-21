from io import StringIO
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


# Place the CSV in the same folder as this script, or replace this with
# another relative path within the project. No computer-specific path is used.
DATA = Path("100_Portfolios_10x10_Daily(1).csv")
START_DATE, END_DATE, WINDOW = "2026-01-02", "2026-06-30", 252
LASSO_STRENGTH, RIDGE_STRENGTH = 1e-5, 1e-4
GROSS_LIMIT, WEALTH_FLOOR = 1.5, 1e-6


def load_returns(path):
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip() == "Average Equal Weighted Returns -- Daily":
                break
        else:
            raise ValueError("Required equal-weighted section was not found.")

        lines = []
        for line in file:
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line)

    returns = pd.read_csv(StringIO("".join(lines)), index_col=0)
    returns.columns = returns.columns.str.strip()
    returns = returns.drop(columns=["ME9 BM10", "BIG HiBM"], errors="ignore")
    returns.index = pd.to_datetime(returns.index.astype(str), format="%Y%m%d")
    returns = returns.sort_index().astype(float)
    returns = returns.replace([-99.99, -999], np.nan).dropna()
    returns = returns / 100

    if returns.shape[1] != 98:
        raise ValueError(f"Expected 98 assets; found {returns.shape[1]}.")

    return returns


def calculate_market_betas(history):
    R = history.to_numpy()
    n_assets = R.shape[1]
    ew = np.full(n_assets, 1 / n_assets)
    ew_returns = R @ ew

    centered_ew = ew_returns - ew_returns.mean()
    centered_assets = R - R.mean(axis=0)
    denominator = np.sum(centered_ew ** 2)

    if denominator <= 0:
        raise ValueError("Equal-weighted benchmark has zero variance.")

    market_betas = (centered_assets.T @ centered_ew) / denominator
    return market_betas, ew_returns


def fit_portfolio(history, model_type, beta_bounds=None):
    R = history.to_numpy()
    n_obs, n_assets = R.shape
    ew = np.full(n_assets, 1 / n_assets)

    # 98 portfolio weights, 97 independent adjustment coefficients.
    N = np.vstack([np.eye(n_assets - 1), -np.ones(n_assets - 1)])
    theta = cp.Variable(n_assets - 1)
    weights = ew - N @ theta

    portfolio_returns = R @ weights
    ew_returns = R @ ew
    market_betas, _ = calculate_market_betas(history)
    portfolio_beta = market_betas @ weights

    portfolio_mean = cp.sum(portfolio_returns) / n_obs
    variance_loss = cp.sum_squares(
        portfolio_returns - portfolio_mean
    ) / n_obs

    ew_variance = np.mean((ew_returns - ew_returns.mean()) ** 2)

    constraints = [
        cp.norm1(weights) <= GROSS_LIMIT,
        variance_loss <= ew_variance,
        1 + portfolio_returns >= WEALTH_FLOOR,
    ]

    if beta_bounds is not None:
        beta_lower, beta_upper = beta_bounds
        constraints += [
            portfolio_beta >= beta_lower,
            portfolio_beta <= beta_upper,
        ]

    if model_type == "LASSO":
        objective = cp.Minimize(
            variance_loss + LASSO_STRENGTH * cp.norm1(theta)
        )
    elif model_type == "Ridge":
        objective = cp.Minimize(
            variance_loss + RIDGE_STRENGTH * cp.sum_squares(theta)
        )
    else:
        raise ValueError("model_type must be LASSO or Ridge.")

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        problem.solve(solver=cp.SCS, verbose=False)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Optimization failed: {problem.status}")

    fitted_weights = np.asarray(weights.value).reshape(-1)
    fitted_beta = float(market_betas @ fitted_weights)
    return fitted_weights, fitted_beta


def summarize(values):
    values = np.asarray(values)
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1
    mean_return = values.mean()
    volatility = values.std(ddof=1)

    return {
        "mean_daily_return": mean_return,
        "volatility": volatility,
        "sharpe": mean_return / volatility,
        "cumulative_return": wealth[-1] - 1,
        "maximum_drawdown": drawdowns.min(),
    }


def run_experiment(returns, model_type, beta_bounds):
    realized_returns = []
    fitted_betas = []
    binding_days = []

    for date in returns.loc[START_DATE:END_DATE].index:
        history = returns.loc[:date].iloc[:-1].tail(WINDOW)
        weights, fitted_beta = fit_portfolio(
            history, model_type, beta_bounds
        )

        realized_returns.append(
            float(weights @ returns.loc[date].to_numpy())
        )
        fitted_betas.append(fitted_beta)

        if beta_bounds is None:
            binding_days.append(False)
        else:
            lower, upper = beta_bounds
            binding_days.append(
                abs(fitted_beta - lower) < 1e-5
                or abs(fitted_beta - upper) < 1e-5
            )

    result = summarize(realized_returns)
    result["average_beta"] = np.mean(fitted_betas)
    result["minimum_beta"] = np.min(fitted_betas)
    result["maximum_beta"] = np.max(fitted_betas)
    result["beta_constraint_binding_pct"] = (
        np.nan if beta_bounds is None else 100 * np.mean(binding_days)
    )
    return result


if __name__ == "__main__":
    returns = load_returns(DATA)

    experiments = {
        "LASSO_unconstrained": ("LASSO", None),
        "Ridge_unconstrained": ("Ridge", None),
        "LASSO_market_like_0.8_1.2": ("LASSO", (0.8, 1.2)),
        "Ridge_market_like_0.8_1.2": ("Ridge", (0.8, 1.2)),
        "LASSO_conservative_0.5_0.9": ("LASSO", (0.5, 0.9)),
        "Ridge_conservative_0.5_0.9": ("Ridge", (0.5, 0.9)),
    }

    results = {}

    for name, (model_type, beta_bounds) in experiments.items():
        print(f"Running {name}...")
        results[name] = run_experiment(
            returns, model_type, beta_bounds
        )

    table = pd.DataFrame(results).T
    print("\nFinal results:\n")
    print(table.to_string(float_format=lambda x: f"{x:.8f}"))
    table.to_csv("market_beta_results.csv")
    print("\nSaved results to market_beta_results.csv")

