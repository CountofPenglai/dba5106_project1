"""Raw percentage returns to the explicitly filtered decimal-return panel."""
import numpy as np
import pandas as pd

DEFAULT_DROP_ASSETS = ("ME9 BM10", "BIG HiBM")

def preprocess_returns(raw, *, drop_assets=DEFAULT_DROP_ASSETS, start=None, end=None):
    """Convert percentage points to decimal returns, preserving missing values.

    This function accepts RAW percentage-point returns. Apply it once, before
    training. Excluded assets are removed before date and numeric processing.
    It does not estimate statistics or fill missing observations.
    """
    frame = raw.copy()
    frame.columns = frame.columns.str.strip()
    frame = frame.drop(columns=list(drop_assets), errors="ignore")
    if frame.columns.has_duplicates:
        raise ValueError("Asset names must be unique")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError("Dates must be unique and valid")
    frame = frame.sort_index()
    frame = frame.loc[start:end].astype(float).replace([-99.99, -999], np.nan) / 100.0
    if frame.empty:
        raise ValueError("No returns in the requested date range")
    frame.index.name = "date"
    return frame
