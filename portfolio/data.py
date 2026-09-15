"""Read the equal-weighted section of a French daily portfolio CSV."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_returns(raw, *, drop_assets=("ME9 BM10",), start=None, end=None):
    """Convert percentage points to decimal returns, preserving missing values.

    This function accepts RAW percentage-point returns. Apply it once, before
    training. It does not estimate statistics or fill missing observations.
    """
    frame = raw.copy()
    frame.columns = frame.columns.str.strip()
    if frame.columns.has_duplicates:
        raise ValueError("Asset names must be unique")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError("Dates must be unique and valid")
    frame = frame.sort_index().drop(columns=list(drop_assets), errors="ignore")
    frame = frame.loc[start:end].astype(float).replace([-99.99, -999], np.nan) / 100.0
    if frame.empty:
        raise ValueError("No returns in the requested date range")
    frame.index.name = "date"
    return frame


def load_returns(path, *, drop_assets=("ME9 BM10",), start=None, end=None):
    """Find the table by its heading, so updated files need no row offsets."""
    heading = "Average Equal Weighted Returns -- Daily"
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip() == heading:
                break
        else:
            raise ValueError(f"Missing CSV section: {heading}")
        lines = []
        for line in source:
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line)
    if len(lines) < 2:
        raise ValueError("Equal-weighted returns table is empty")
    names = [name.strip() for name in lines[0].strip().split(",")[1:]]
    if len(set(names)) != len(names):
        raise ValueError("Asset names must be unique")
    raw = pd.read_csv(StringIO("".join(lines)), index_col=0)
    return preprocess_returns(raw, drop_assets=drop_assets, start=start, end=end)
