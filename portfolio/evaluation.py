"""Walk-forward execution and daily-return evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .splits import rolling_windows
from .weights import validate_portfolio_weights


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame


def backtest(returns, strategy, *, start, end, window=126):
    """strategy(history) returns an asset-indexed Series of portfolio weights.

    Weights are held for one day; costs and a risk-free rate are not included.
    Current-day returns are read only after the strategy produces its weights.
    """
    if returns.columns.has_duplicates:
        raise ValueError("Asset names must be unique")
    dates, daily, holdings = [], [], []
    for date, history in rolling_windows(returns, start=start, end=end, window=window):
        if not np.isfinite(history.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or infinite training returns before {date.date()}")
        weights = strategy(history)
        if (not isinstance(weights, pd.Series) or weights.index.has_duplicates
                or set(weights.index) != set(returns.columns)):
            raise ValueError("Strategy must return one weight for every asset, indexed by asset name")
        values = validate_portfolio_weights(weights.reindex(returns.columns), len(returns.columns))
        realized = returns.loc[date].to_numpy(dtype=float)
        if not np.isfinite(realized).all():
            raise ValueError(f"Missing or infinite evaluation returns on {date.date()}")
        dates.append(date)
        daily.append(float(values @ realized))
        holdings.append(values.copy())
    index = pd.DatetimeIndex(dates, name="date")
    return BacktestResult(pd.Series(daily, index=index, name="return"),
                          pd.DataFrame(holdings, index=index, columns=returns.columns))


def summarize(returns):
    """Daily Sharpe = mean/std, ddof=1; max_drawdown is a negative fraction."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("Metrics need at least two finite daily returns")
    mean = float(r.mean())
    # Floating-point averaging can give identical nonzero values a tiny std.
    std = 0.0 if np.all(r == r[0]) else float(r.std(ddof=1))
    sharpe = mean / std if std > 0 else float("nan")
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {"mean": mean, "volatility": std, "sharpe": sharpe,
            "negative_sharpe": -sharpe, "cumulative_return": float(wealth[-1] - 1),
            "max_drawdown": float(drawdown.min())}
