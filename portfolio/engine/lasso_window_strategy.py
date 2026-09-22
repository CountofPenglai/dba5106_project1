"""Fixed-LASSO fit adapter shared by window and recency-weight experiments."""
import warnings
from sklearn.exceptions import ConvergenceWarning
from portfolio.regularization import Regularization
from portfolio.losses import LogGrowthLoss
from portfolio.engine.portfolio_fit import fit_portfolio, fit_portfolio_growth
from portfolio.data.observation_weights import normalize_sample_weights
from portfolio.evaluation.portfolio_metrics import backtest_metrics as metrics
from portfolio.evaluation.provenance import file_digest

FINAL_TEST_START, FINAL_TEST_END = "2026-08-01", "2026-08-30"
HISTORY_START, HISTORY_END = "2021-08-01", "2026-07-31"
VALIDATION_START, VALIDATION_END = "2026-01-02", "2026-06-30"
LASSO = Regularization("lasso", 1e-5)

def lasso_strategy(config, records, *, constraints=None):
    """Connect an ordinary training configuration to the shared fit function."""
    def strategy(history):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            if constraints is None:
                fitted = fit_portfolio(history, regularization=LASSO,
                                       sample_weights=config.sample_weights)
            else:
                fitted = fit_portfolio_growth(
                    history, regularization=LASSO, sample_weights=config.sample_weights,
                    loss=LogGrowthLoss(reward_strength=0, exposure_strength=0), constraints=constraints)
        q = normalize_sample_weights(config.sample_weights(len(history)), len(history))
        records.append({"history_start": str(history.index[0].date()),
                        "history_end": str(history.index[-1].date()),
                        "observations": len(history),
                        "effective_observations": float(1 / (q @ q)),
                        "training_loss": fitted.training_loss, "objective": fitted.objective,
                        **(fitted.diagnostics or {})})
        if constraints is not None and len(records) % 63 == 0:
            print(f"  Fitted {len(records)} days; history through {history.index[-1].date()}", flush=True)
        return fitted.weights
    return strategy
