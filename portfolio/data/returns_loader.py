"""Read the equal-weighted section of a French daily portfolio CSV."""

from io import StringIO
from pathlib import Path

import pandas as pd


from portfolio.data.preprocessing import DEFAULT_DROP_ASSETS, preprocess_returns


MODEL_SECTION = "Average Equal Weighted Returns -- Daily"


def load_returns(path, *, drop_assets=DEFAULT_DROP_ASSETS, start=None, end=None):
    """Find the table by its heading, so updated files need no row offsets."""
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip() == MODEL_SECTION:
                break
        else:
            raise ValueError(f"Missing CSV section: {MODEL_SECTION}")
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
