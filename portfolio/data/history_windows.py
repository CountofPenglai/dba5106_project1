"""Chronological holdouts and history-only rolling windows."""

from dataclasses import dataclass

import pandas as pd


def validate_dates(returns):
    index = returns.index
    if (not isinstance(index, pd.DatetimeIndex) or index.has_duplicates
            or index.hasnans or not index.is_monotonic_increasing):
        raise ValueError("Returns need a sorted, unique DatetimeIndex")


@dataclass(frozen=True)
class TrainingWindow:
    """Number of preceding trading observations in each fit."""
    observations: int

    def __post_init__(self):
        if type(self.observations) is not int or self.observations < 2:
            raise ValueError("Training needs at least two trading observations")


def rolling_windows(returns, *, start, end, window=126):
    """Yield (evaluation date, preceding observations), never the current row.

    window accepts the legacy integer (default 126) or a TrainingWindow.
    Use actual available trading dates for start/end. Partial trailing coverage
    is rejected rather than silently reporting a shorter evaluation period.
    """
    validate_dates(returns)
    if not isinstance(window, TrainingWindow):
        window = TrainingWindow(observations=window)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or returns.empty:
        raise ValueError("Evaluation range must be nonempty and ordered")
    if start < returns.index[0] or end > returns.index[-1]:
        raise ValueError("Evaluation range is outside available data; use available trading dates")
    positions = [i for i, date in enumerate(returns.index) if start <= date <= end]
    if not positions:
        raise ValueError("No evaluation dates in the requested range")
    for position in positions:
        date = returns.index[position]
        left = position - window.observations
        if left < 0:
            raise ValueError(f"Need {window.observations} observations before {date.date()}")
        yield date, returns.iloc[left:position].copy()
