"""Report-aligned portfolio research: data, models, execution and evaluation."""
from portfolio.data.returns_loader import load_returns
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.portfolio_metrics import summarize

__all__ = ["load_returns", "backtest", "summarize"]
