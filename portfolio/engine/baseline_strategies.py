"""Six basic portfolios: EW, MinVar, coefficient constraints and penalties.

History selection stays in the shared backtest. Constraints bind regression
coefficients, not portfolio leverage or volatility. Legacy names stay available
through penalty_baseline_strategies for earlier experiments.
"""
import pandas as pd

from portfolio.config import TRAINING_PRESETS, TrainingConfig
from portfolio.constraints import CoefficientConstraint
from portfolio.engine.portfolio_fit import fit_portfolio, fit_portfolio_coefficient_constrained
from portfolio.regularization import Regularization


BASELINE_REGULARIZATION = {
    "MinVar": Regularization(),
    "LASSO": Regularization("lasso", 1e-5),
    "Ridge": Regularization("ridge", 1e-4),
}


def equal_weight(history):
    """Equal capital in each asset; history length does not affect holdings."""
    return pd.Series(1 / history.shape[1], index=history.columns)


def portfolio_strategy(**options):
    """Adapt fit_portfolio's FitResult to the history -> weights interface."""
    def strategy(history):
        return fit_portfolio(history, **options).weights
    return strategy


def penalty_baseline_strategies(training: TrainingConfig = TRAINING_PRESETS["252d"]):
    """The historical four-model suite, retaining LASSO/Ridge output names."""
    return {
        "EW": equal_weight,
        **{name: portfolio_strategy(regularization=regularization,
                                   sample_weights=training.sample_weights)
           for name, regularization in BASELINE_REGULARIZATION.items()},
    }


def coefficient_constraint_strategy(constraint, training, records=None):
    """Fit a coefficient budget and optionally retain per-window diagnostics."""
    def strategy(history):
        fitted = fit_portfolio_coefficient_constrained(
            history, constraint=constraint, sample_weights=training.sample_weights)
        if records is not None:
            records.append({"history_start": history.index[0], "history_end": history.index[-1],
                            "observations": len(history), **fitted.diagnostics})
        return fitted.weights
    return strategy


def baseline_strategies(training: TrainingConfig = TRAINING_PRESETS["252d"], *,
                        lasso_budget=1.0, ridge_budget=1.0, diagnostics=None):
    """Create all six baselines; pass training.window to the shared backtest.

    Budgets bound sum(abs(beta)) and sum(beta**2), respectively. They are fixed
    configuration values, not penalty lambdas and not selected by validation.
    An optional diagnostics dictionary receives the two daily record lists.
    """
    penalty = penalty_baseline_strategies(training)
    strategies = {"EW": penalty["EW"], "MinVar": penalty["MinVar"]}
    for kind, budget, name in (("lasso", lasso_budget, "LASSO_CoefficientConstraint"),
                              ("ridge", ridge_budget, "Ridge_CoefficientConstraint")):
        records = None if diagnostics is None else diagnostics.setdefault(name, [])
        strategies[name] = coefficient_constraint_strategy(CoefficientConstraint(kind, budget), training, records)
    strategies.update(LASSO_Penalty=penalty["LASSO"], Ridge_Penalty=penalty["Ridge"])
    return strategies
