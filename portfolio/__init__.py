"""Small, composable tools for the DBA5106 portfolio experiments."""

from portfolio.data.returns_loader import load_returns
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.regularization import Regularization
from portfolio.data.history_windows import TrainingWindow, split_returns
from portfolio.config import TrainingConfig, TRAINING_PRESETS
from portfolio.engine.portfolio_fit import fit_portfolio, fit_portfolio_cv, fit_portfolio_growth
from portfolio.constraints import PortfolioConstraints
from portfolio.losses import LogGrowthLoss
from portfolio.engine.parameter_search import GrowthParameters, SearchConfig, search_growth

__all__ = [
    "load_returns", "split_returns", "Regularization",
    "TrainingWindow", "TrainingConfig", "TRAINING_PRESETS",
    "fit_portfolio", "fit_portfolio_cv", "fit_portfolio_growth", "backtest", "summarize",
    "LogGrowthLoss", "PortfolioConstraints",
    "GrowthParameters", "SearchConfig", "search_growth",
]
