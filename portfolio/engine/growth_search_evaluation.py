"""Cached candidate evaluation and failure recording for growth/exposure search."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from portfolio.engine.rolling_backtest import backtest
from portfolio.engine.portfolio_fit import fit_portfolio_growth
from portfolio.regularization import Regularization
from portfolio.losses import LogGrowthLoss
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.artifact_store import RunArtifacts

def metrics(result):
    gross = result.weights.abs().sum(axis=1)
    return {**summarize(result.returns), "average_gross_leverage": float(gross.mean()),
            "max_gross_leverage": float(gross.max())}

def save_fit(directory, name, result, diagnostics=None, *, layout="flat"):
    RunArtifacts(directory, layout).write_backtest(name, result, diagnostics)

class PhaseEvaluator:
    """Fit only past windows; cache complete candidate backtests within one phase."""

    def __init__(self, returns, *, start, end, window, output, constraints):
        self.returns = returns.loc[:end].copy()
        self.requested_start, self.requested_end = start, end
        days = self.returns.loc[start:end].index
        self.start, self.end = str(days[0].date()), str(days[-1].date())
        self.window = window
        self.output, self.constraints = output, constraints
        self.cache = {}
        self.failures = {}
        self.attempts = []
        self.calls = 0

    def fit(self, parameters):
        if parameters in self.cache:
            return self.cache[parameters]
        if parameters in self.failures:
            raise RuntimeError(self.failures[parameters])
        records = []

        def strategy(history):
            fitted = fit_portfolio_growth(history, regularization=Regularization("lasso", 1e-5),
                loss=LogGrowthLoss(**asdict(parameters)), constraints=self.constraints)
            records.append({"history_end": history.index[-1], **fitted.diagnostics})
            return fitted.weights

        try:
            result = backtest(self.returns, strategy, start=self.start, end=self.end, window=self.window)
            if (result.returns <= -1).any():
                raise ValueError("Candidate has nonpositive realized wealth in the evaluation period")
            if not np.isfinite(metrics(result)["cumulative_return"]):
                raise ValueError("Candidate has a nonfinite cumulative return")
        except (RuntimeError, ValueError) as error:
            self.failures[parameters] = f"{type(error).__name__}: {error}"
            raise
        diagnostics = pd.DataFrame(records, index=result.returns.index)
        self.cache[parameters] = (result, diagnostics)
        return result, diagnostics

    def __call__(self, parameters):
        self.calls += 1
        print(f"{self.output.name} candidate {self.calls}: eta={parameters.reward_strength:g}, "
              f"gamma={parameters.exposure_strength:g}", flush=True)
        record = {"candidate_id": self.calls, **asdict(parameters), "status": "ok", "error": ""}
        try:
            result, diagnostics = self.fit(parameters)
            values = metrics(result)
            record.update(values)
            save_fit(self.output, f"candidate_{self.calls:02d}", result, diagnostics)
            return values
        except (RuntimeError, ValueError) as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            # Preserve failures even if no candidate succeeds and search raises.
            self.attempts.append(record)
            self.output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.attempts).to_csv(self.output / "candidate_attempts.csv", index=False)
