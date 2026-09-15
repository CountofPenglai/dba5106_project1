"""Penalty configuration shared by the solver and objective reporting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Regularization:
    kind: str = "none"
    strength: float = 0.0

    def __post_init__(self):
        if self.kind not in {"none", "lasso", "ridge"}:
            raise ValueError("Regularization must be none, lasso, or ridge")
        if not np.isfinite(self.strength) or self.strength < 0:
            raise ValueError("Regularization strength must be finite and nonnegative")
        if self.kind == "none" and self.strength != 0:
            raise ValueError("Unregularized models require strength=0")

    def penalty(self, coefficients):
        beta = np.asarray(coefficients, dtype=float)
        if self.kind == "lasso":
            return float(self.strength * np.abs(beta).sum())
        if self.kind == "ridge":
            return float(self.strength * (beta @ beta))
        return 0.0
