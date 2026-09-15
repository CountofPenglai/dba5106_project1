"""Chronological holdouts and history-only rolling windows."""

from dataclasses import dataclass

import pandas as pd


def validate_dates(returns):
    index = returns.index
    if (not isinstance(index, pd.DatetimeIndex) or index.has_duplicates
            or index.hasnans or not index.is_monotonic_increasing):
        raise ValueError("Returns need a sorted, unique DatetimeIndex")


@dataclass
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_returns(returns, *, train_end, validation_end, test_start, test_end):
    """Keep calendar boundaries explicit; a gap and an empty test are allowed."""
    validate_dates(returns)
    train_end, validation_end, test_start, test_end = map(
        pd.Timestamp, (train_end, validation_end, test_start, test_end))
    if not train_end < validation_end < test_start <= test_end:
        raise ValueError("Require train_end < validation_end < test_start <= test_end")
    dates = returns.index
    return DataSplit(
        returns.loc[dates <= train_end].copy(),
        returns.loc[(dates > train_end) & (dates <= validation_end)].copy(),
        returns.loc[(dates >= test_start) & (dates <= test_end)].copy(),
    )


def rolling_windows(returns, *, start, end, window=126):
    """Yield (evaluation date, preceding observations), never the current row.

    Use actual available trading dates for start/end. Partial trailing coverage
    is rejected rather than silently reporting a shorter evaluation period.
    """
    validate_dates(returns)
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError("window must be an integer >= 2")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or returns.empty:
        raise ValueError("Evaluation range must be nonempty and ordered")
    if start < returns.index[0] or end > returns.index[-1]:
        raise ValueError("Evaluation range is outside available data; use available trading dates")
    positions = [i for i, date in enumerate(returns.index) if start <= date <= end]
    if not positions:
        raise ValueError("No evaluation dates in the requested range")
    if positions[0] < window:
        raise ValueError(f"Need {window} observations before the first evaluation date")
    for position in positions:
        yield returns.index[position], returns.iloc[position - window:position].copy()
