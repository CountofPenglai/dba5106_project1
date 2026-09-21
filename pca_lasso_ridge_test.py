"""Rolling PCA + LASSO/Ridge portfolio experiment.

Follows the repository README:
- reads only ``Average Equal Weighted Returns -- Daily``;
- removes ``ME9 BM10`` and ``BIG HiBM``;
- converts percentage returns to decimals;
- uses 252 previous trading days for one-day-ahead validation;
- evaluates 2026-01-02 through 2026-06-30.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge


DATA = Path("upload/100_Portfolios_10x10_Daily(1).csv")
START, END, WINDOW = "2026-01-02", "2026-06-30", 252
LASSO_STRENGTH, RIDGE_STRENGTH = 1e-5, 1e-4


def load_returns(path):
    """Load and preprocess only the required dataset section."""
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip() == "Average Equal Weighted Returns -- Daily":
                break
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
    return returns / 100


def fit_weights(history, n_components, model_type):
    """Fit PCA followed by LASSO or Ridge and return portfolio weights."""
    r = history.to_numpy()
    n_assets = r.shape[1]

    ew = np.full(n_assets, 1 / n_assets)
    N = np.vstack([np.eye(n_assets - 1), -np.ones(n_assets - 1)])

    # README model structure: X = R @ N and y = R @ EW.
    X = r @ N
    y = r @ ew

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)

    if model_type == "LASSO":
        model = Lasso(
            alpha=LASSO_STRENGTH / 2,
            fit_intercept=True,
            max_iter=100000,
            tol=1e-8,
        )
    elif model_type == "Ridge":
        model = Ridge(alpha=RIDGE_STRENGTH / 2, fit_intercept=True)
    else:
        raise ValueError("model_type must be 'LASSO' or 'Ridge'")

    model.fit(components, y)

    # Map component coefficients back to the 97-dimensional beta space.
    beta = pca.components_.T @ model.coef_
    return ew - N @ beta


def summarize(portfolio_returns):
    x = np.asarray(portfolio_returns)
    wealth = np.r_[1.0, np.cumprod(1 + x)]
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1
    mean = x.mean()
    volatility = x.std(ddof=1)
    return {
        "mean_daily": mean,
        "volatility": volatility,
        "sharpe": mean / volatility,
        "cumulative_return": wealth[-1] - 1,
        "max_drawdown": drawdowns.min(),
    }


def run(n_components, model_type):
    returns = load_returns(DATA)
    portfolio_returns = []

    for date in returns.loc[START:END].index:
        history = returns.loc[:date].iloc[:-1].tail(WINDOW)
        weights = fit_weights(history, n_components, model_type)
        portfolio_returns.append(weights @ returns.loc[date].to_numpy())

    return summarize(portfolio_returns)


if __name__ == "__main__":
    results = {}
    for model_type in ("LASSO", "Ridge"):
        for k in (3, 5, 10, 15, 20, 30):
            results[f"{model_type}_PCA_{k}"] = run(k, model_type)

    print(pd.DataFrame(results).T.to_string(float_format=lambda x: f"{x:.8f}"))

