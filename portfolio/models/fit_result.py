"""The common result of fitting a single-stage portfolio model."""
from dataclasses import dataclass
import pandas as pd
from portfolio.models.min_variance_regression import LinearModel


@dataclass
class FitResult:
    model: LinearModel
    weights: pd.Series
    training_loss: float
    objective: float
    diagnostics: dict | None = None
