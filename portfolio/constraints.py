"""Portfolio risk limits for the exposure-penalized LASSO."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioConstraints:
    """Limits apply to each historical fit, not to unknown future returns.

    volatility_limit is a multiple of EW volatility in the same training window.
    wealth_floor bounds 1 + daily portfolio return in every training scenario.
    """

    leverage_limit: float = 1.5
    volatility_limit: float = 1.0
    wealth_floor: float = 1e-6
    weight_limit: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.leverage_limit) or self.leverage_limit < 1:
            raise ValueError("Leverage limit must be finite and at least one")
        if not np.isfinite(self.volatility_limit) or self.volatility_limit < 0:
            raise ValueError("Volatility limit must be finite and nonnegative")
        if not np.isfinite(self.wealth_floor) or not 0 < self.wealth_floor <= 1:
            raise ValueError("Wealth floor must be in (0, 1]")
        if self.weight_limit is not None and (not np.isfinite(self.weight_limit) or self.weight_limit <= 0):
            raise ValueError("Weight limit must be positive and finite")
