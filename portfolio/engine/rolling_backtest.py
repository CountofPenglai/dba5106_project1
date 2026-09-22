"""Causal daily execution of history-to-holdings strategies."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from portfolio.data.history_windows import rolling_windows
from portfolio.decisions.holding_weights import validate_portfolio_weights

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
