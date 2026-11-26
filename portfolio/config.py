"""Shared training and evaluation settings for the final report."""
from dataclasses import dataclass
from portfolio.data.history_windows import TrainingWindow


@dataclass(frozen=True)
class TrainingConfig:
    window: TrainingWindow


COURSE_WINDOW = 126
COURSE_TRAINING = TrainingConfig(TrainingWindow(observations=COURSE_WINDOW))
VALIDATION = ('2026-01-02', '2026-06-30')
HISTORY_START = '2021-08-01'
FORECAST_CALIBRATION_END = '2024-07-31'
