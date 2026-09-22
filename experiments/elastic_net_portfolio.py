"""Compare rolling Elastic Net portfolios across L1 ratios."""
import argparse
import json
from pathlib import Path
import pandas as pd
from portfolio.data.returns_loader import load_returns as _load_returns
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.artifact_store import (DEFAULT_CSV, RunArtifacts, default_output,
                                                 ensure_fresh_output)
from portfolio.evaluation.portfolio_metrics import summarize as _summarize
from portfolio.evaluation.provenance import run_provenance
from portfolio.models.elastic_net_portfolio import fit_weights as _fit_weights

DATA = DEFAULT_CSV
START, END, WINDOW = "2026-01-02", "2026-06-30", 252
STRENGTH = 1e-5
L1_RATIOS = (0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0)

def load_returns(path):
    return _load_returns(path).dropna()

def _strategy(ratio):
    return lambda history: pd.Series(fit_weights(history, ratio), index=history.columns)

def fit_weights(history, l1_ratio):
    return _fit_weights(history, l1_ratio, strength=STRENGTH)

def summarize(values):
    metrics = _summarize(values)
    return {"mean_daily": metrics["mean"], "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"], "negative_sharpe": metrics["negative_sharpe"],
            "cumulative_return": metrics["cumulative_return"],
            "max_drawdown": metrics["max_drawdown"]}

def _metrics(result):
    return summarize(result.returns)

def run(l1_ratio, *, returns=None, csv=None, start=None, end=None, window=None):
    data = load_returns(DATA if csv is None else csv) if returns is None else returns
    return _metrics(backtest(data, _strategy(l1_ratio), start=start or START, end=end or END,
                             window=WINDOW if window is None else window))

def main(argv=None, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout)
    args = parser.parse_args(argv)
    try:
        store = RunArtifacts(args.output or default_output("elastic_net_portfolio"), args.layout)
        ensure_fresh_output(store.root)
        returns, rows = load_returns(args.csv), {}
        if args.window <= 0 or len(returns.loc[args.start:args.end]) < 2:
            raise ValueError("window must be positive and evaluation needs at least two complete rows")
        store.prepare()
        for ratio in L1_RATIOS:
            name = f"ElasticNet_l1ratio_{ratio:g}"
            result = backtest(returns, _strategy(ratio), start=args.start, end=args.end, window=args.window)
            store.write_backtest(name, result)
            rows[name] = _metrics(result)
        table = pd.DataFrame(rows).T
        table.to_csv(store.path("summary.csv"))
        store.path("run_metadata.json").write_text(json.dumps({**run_provenance(args.csv),
            "start": args.start, "end": args.end, "window": args.window, "strength": STRENGTH,
            "l1_ratios": list(L1_RATIOS), "artifact_layout": args.layout,
            "artifact_layout_version": 1,
            "missing_row_policy": "drop every row missing any retained asset"}, indent=2))
        store.finalize()
        print(table.to_string(float_format=lambda value: f"{value:.8f}"))
        print(f"\nSaved artifacts to {store.root.resolve()}")
        return table
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main(default_layout="categorized")
