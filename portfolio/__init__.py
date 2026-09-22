"""Small, composable tools for the DBA5106 portfolio experiments."""

from .data import load_returns
from .evaluation import backtest, summarize
from .regularization import Regularization
from .splits import TrainingWindow, split_returns
from .config import TrainingConfig, TRAINING_PRESETS
from .training import fit_portfolio, fit_portfolio_cv, fit_portfolio_growth
from .constraints import PortfolioConstraints
from .losses import LogGrowthLoss
from .selection import GrowthParameters, SearchConfig, search_growth

__all__ = [
    "load_returns", "split_returns", "Regularization",
    "TrainingWindow", "TrainingConfig", "TRAINING_PRESETS",
    "fit_portfolio", "fit_portfolio_cv", "fit_portfolio_growth", "backtest", "summarize",
    "LogGrowthLoss", "PortfolioConstraints",
    "GrowthParameters", "SearchConfig", "search_growth",
]
