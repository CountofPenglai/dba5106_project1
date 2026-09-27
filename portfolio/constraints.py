"""Explicit coefficient and portfolio limits, separate from loss penalties."""

from dataclasses import dataclass
from numbers import Real

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
    beta_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        if not np.isfinite(self.leverage_limit) or self.leverage_limit < 1:
            raise ValueError("Leverage limit must be finite and at least one")
        if not np.isfinite(self.volatility_limit) or self.volatility_limit < 0:
            raise ValueError("Volatility limit must be finite and nonnegative")
        if not np.isfinite(self.wealth_floor) or not 0 < self.wealth_floor <= 1:
            raise ValueError("Wealth floor must be in (0, 1]")
        if self.weight_limit is not None and (not np.isfinite(self.weight_limit) or self.weight_limit <= 0):
            raise ValueError("Weight limit must be positive and finite")
        if self.beta_bounds is not None:
            if (len(self.beta_bounds) != 2 or not np.isfinite(self.beta_bounds).all()
                    or self.beta_bounds[0] > self.beta_bounds[1]):
                raise ValueError("Beta bounds must be finite and ordered")


@dataclass(frozen=True)
class CoefficientConstraint:
    """Bound sum(abs(beta)) for LASSO or sum(beta**2) for Ridge.

    The intercept is free. A zero budget forces beta=0; the portfolio mapping
    then returns EW. This is separate from PortfolioConstraints on holdings.
    """
    kind: str
    budget: float = 1.0

    def __post_init__(self):
        if self.kind not in {"lasso", "ridge"}:
            raise ValueError("Coefficient constraint must be lasso or ridge")
        if (isinstance(self.budget, bool) or not isinstance(self.budget, Real)
                or not np.isfinite(self.budget) or self.budget < 0):
            raise ValueError("Coefficient budget must be finite and nonnegative")

    def measure(self, coefficients):
        beta = np.asarray(coefficients, dtype=float)
        return float(np.abs(beta).sum() if self.kind == "lasso" else beta @ beta)
