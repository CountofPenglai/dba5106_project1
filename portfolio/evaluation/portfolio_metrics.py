"""Daily return and portfolio performance measurements."""
import numpy as np

def summarize(returns):
    """Daily Sharpe = mean/std, ddof=1; max_drawdown is a negative fraction."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("Metrics need at least two finite daily returns")
    mean = float(r.mean())
    # Floating-point averaging can give identical nonzero values a tiny std.
    std = 0.0 if np.all(r == r[0]) else float(r.std(ddof=1))
    sharpe = mean / std if std > 0 else float("nan")
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {"mean": mean, "volatility": std, "sharpe": sharpe,
            "negative_sharpe": -sharpe, "cumulative_return": float(wealth[-1] - 1),
            "max_drawdown": float(drawdown.min())}


def portfolio_metrics(weights, returns):
    daily = np.sum(weights.to_numpy()*returns.to_numpy(), axis=1)
    result = summarize(daily)
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Performance has undefined Sharpe or nonfinite values')
    gross = weights.abs().sum(axis=1)
    return {**result, 'average_gross_leverage': float(gross.mean()), 'max_gross_leverage': float(gross.max()),
            'evaluation_days': len(daily), 'start': str(weights.index[0].date()), 'end': str(weights.index[-1].date())}


def backtest_metrics(result, start=None, end=None):
    daily = result.returns.loc[start:end]
    gross = result.weights.loc[daily.index].abs().sum(axis=1)
    return {**summarize(daily), "average_gross_leverage": float(gross.mean()),
            "max_gross_leverage": float(gross.max()), "evaluation_days": len(daily),
            "start": str(daily.index[0].date()), "end": str(daily.index[-1].date())}
