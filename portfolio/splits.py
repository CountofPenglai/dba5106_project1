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
    """Select a fixed observation count OR at most a number of calendar years.

    Calendar windows include the anniversary and exclude the prediction day.
    They may be shorter when history is unavailable, but must satisfy
    min_observations. The experiment's evaluation start controls its warm-up.
    """
    observations: int | None = None
    years: int | None = None
    min_observations: int = 2

    def __post_init__(self):
        if (self.observations is None) == (self.years is None):
            raise ValueError("Set exactly one of observations or years")
        for name, value, minimum in (("observations", self.observations, 2),
                                     ("years", self.years, 1),
                                     ("min_observations", self.min_observations, 2)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                      or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.min_observations is None:
            raise ValueError("min_observations must be an integer >= 2")
        if self.observations is not None and self.min_observations > self.observations:
            raise ValueError("min_observations cannot exceed observations")


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
        if window.observations is not None:
            left = max(0, position - window.observations)
            required = window.observations
        else:
            cutoff = date - pd.DateOffset(years=window.years)
            left = returns.index.searchsorted(cutoff, side="left")
            required = window.min_observations
        if position - left < required:
            raise ValueError(f"Need {required} observations before {date.date()}")
        yield date, returns.iloc[left:position].copy()
