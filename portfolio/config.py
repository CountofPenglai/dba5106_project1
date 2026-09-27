"""Reusable training-window and sample-weight settings, independent of models."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from portfolio.data.history_windows import TrainingWindow
from portfolio.data.observation_weights import exponential, uniform


@dataclass(frozen=True)
class TrainingConfig:
    """Change one configuration without changing the fit/backtest pipeline.

    half_life is measured in trading observations; None means equal weights.
    Calendar history length and observation-weight decay remain independent.
    """
    window: TrainingWindow
    half_life: float | None = None

    def __post_init__(self):
        if not isinstance(self.window, TrainingWindow):
            raise ValueError("window must be a TrainingWindow")
        if self.half_life is not None and (
                isinstance(self.half_life, bool) or not isinstance(self.half_life, Real)
                or not np.isfinite(self.half_life) or self.half_life <= 0):
            raise ValueError("half_life must be a finite positive number or None")

    def sample_weights(self, n):
        """The ordinary sample_weights(n) callback accepted by fit_portfolio."""
        return uniform(n) if self.half_life is None else exponential(n, self.half_life)


TRAINING_PRESETS = {
    "126d": TrainingConfig(TrainingWindow(observations=126)),
    "252d": TrainingConfig(TrainingWindow(observations=252)),
}

# The July period is an explicit reproduction of previously evaluated data.
EVALUATION_PERIODS = {
    'validation': ('2026-01-02', '2026-06-30'),
    'july_test': ('2026-07-01', '2026-07-31'),
}
HISTORY_START = '2021-08-01'
