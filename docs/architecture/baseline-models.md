# Basic portfolio models: coefficient constraints and penalties

## Quick start

The primary entrypoint is **`run_baselines.py` in the project root**. It runs six
models using the same data, training window, evaluation dates and metrics:

```bash
python run_baselines.py --window 126
python run_baselines.py --window 252
python run_baselines.py --window 252 --lasso-coefficient-budget 0.5 --ridge-coefficient-budget 0.25
```

Defaults: 252 prior trading observations, validation from January 2 to June 30,
2026, LASSO coefficient budget 1 and Ridge squared-coefficient budget 1. These
budgets are explicit initial settings, not tuned values. The third command is a
configuration example, not a recommended or tested performance improvement.

| Output name | Training objective | Explicit coefficient constraint |
|---|---|---|
| EW | No estimation; equal capital in all retained assets | None |
| MinVar | Minimize MSE, equivalent to historical portfolio variance | None |
| LASSO_CoefficientConstraint | Minimize MSE only | Σ|βⱼ| ≤ t₁; default t₁ = 1 |
| Ridge_CoefficientConstraint | Minimize MSE only | Σβⱼ² ≤ t₂; default t₂ = 1 |
| LASSO_Penalty | Minimize MSE + λΣ|βⱼ|; λ = 0.00001 | None |
| Ridge_Penalty | Minimize MSE + λΣβⱼ²; λ = 0.0001 | None |

The constraints apply directly to regression coefficients **β**, not holdings w.
The intercept is free. Ridge's budget bounds the **squared L2 norm**: a budget of
0.25 means an L2 norm of at most 0.5. Budgets must be finite and nonnegative.
A zero budget forces β = 0 and hence EW holdings; a sufficiently loose budget
recovers an unconstrained minimum-variance solution.

## Shared architecture

`CSV → preprocessing → prior history → baseline fit → holdings → realized return → metrics`

| File | Responsibility |
|---|---|
| `run_baselines.py` | Easily accessible six-model entrypoint; forwards to the shared experiment runner |
| `portfolio/engine/baseline_strategies.py` | Central six-model definitions and historical penalty-only recipes |
| `portfolio/constraints.py` | `CoefficientConstraint`: L1/squared-L2 budget; distinct from `PortfolioConstraints` on holdings |
| `portfolio/models/coefficient_constrained_regression.py` | Minimize MSE under a coefficient constraint, using the shared checked convex solver |
| `portfolio/models/min_variance_regression.py` | Regression parameterization and original OLS/LASSO/Ridge penalty solvers |
| `portfolio/engine/portfolio_fit.py` | Convert historical returns to a regression problem, then recover holdings and diagnostics |
| `portfolio/config.py`, `data/history_windows.py` | Training configuration, sample weights and causal history selection |
| `portfolio/engine/rolling_backtest.py` | Daily fitting and realized-return calculation |
| `experiments/regularized_baselines.py` | Shared CLI configuration, execution, metrics and artifact export |

Preprocessing selects the daily equal-weighted source table, removes `ME9 BM10`
and `BIG HiBM` before numerical processing, converts percentage points to decimal
returns, and preserves missing observations. The retained 98 assets are Size–BM
portfolios, not individual stocks. Missing or nonfinite values in a used training
or evaluation row cause an error; they are not silently imputed.

For a history matrix R with n observations and p assets, let e = (1/p, …, 1/p),
N = [I; −1ᵀ], X = RN and y = Re. Fit p−1 coefficients β with a free intercept a:

`MSE = weighted_mean((y − a − Xβ)²)`

Recover holdings with `w = e − Nβ`. This automatically ensures `Σwᵢ = 1`.
The residual is `Rw − a`; optimizing the intercept centers portfolio returns,
so MSE represents historical portfolio variance. These are direct allocation
models, not next-day asset-return forecasting models.

The constrained solver centers X and y using training observation weights,
optimizes MSE with the L1 or squared-L2 inequality, and recovers the intercept.
It scales the entire objective by one positive constant for numerical stability;
this does not change the optimum. It independently checks the returned
coefficient budget. There is no penalty term, post-fit clipping, additional
portfolio-risk constraint or EW blending in these two versions.

The penalty versions preserve the original implementation. Inputs are
unstandardized decimal-return contrasts. To keep `MSE + λ × penalty` comparable
across window lengths, sklearn uses `Lasso(alpha=λ/2)` and `Ridge(alpha=nλ)`.
Changing the window does not change the project-level λ.

Both constraint and penalty formulations regularize β in the chosen regression
basis. A zero coefficient keeps the corresponding asset's EW weight; it does not
remove that asset. Short positions remain possible. Neither formulation adds a
separate leverage or volatility limit. The existing `LASSO_Constrained` experiment
is different: it retains an L1 penalty and imposes limits on portfolio holdings,
training volatility and historical wealth. Its implementation and name remain
unchanged.

The constrained and penalized convex formulations can have corresponding
solutions for suitable budgets and Lagrange multipliers. **A budget t is not the
same parameter as penalty λ**, and their relationship depends on the training
data. The fixed budget 1 does not claim to reproduce the old fixed-λ results.

## Training configuration and compatibility

`--window` accepts an integer of at least two prior trading observations, provided
enough complete history exists. `--start` and `--end` select evaluation dates,
not training-data boundaries. Every fit excludes the evaluation day. Already
observed validation returns may enter later rolling training windows.

In Python, change one configuration and reuse the same six-model loop:

```python
from portfolio.config import TRAINING_PRESETS
from portfolio.data.returns_loader import load_returns
from portfolio.engine.baseline_strategies import baseline_strategies
from portfolio.engine.rolling_backtest import backtest
from portfolio.paths import DEFAULT_CSV

training = TRAINING_PRESETS["126d"]  # or "252d", "3y", "3y_weighted"
returns = load_returns(DEFAULT_CSV)
results = {
    name: backtest(returns, strategy, start="2026-01-02", end="2026-06-30",
                   window=training.window)
    for name, strategy in baseline_strategies(
        training, lasso_budget=1.0, ridge_budget=1.0).items()
}
```

Custom `TrainingConfig(TrainingWindow(observations=504))` is supported.
Weighted configurations use `TrainingConfig.half_life`; constraints themselves
still act on β, not on observation weights. EW holdings are independent of both
settings, but use the same coverage checks and evaluation dates.

`python -m experiments.regularized_baselines --baselines-only` also runs the six
basic models. Without this flag, that module and legacy `run_example.py` retain
the previous EW/MinVar/LASSO/Ridge/LASSO_CV/Ridge_CV default and optional weighted
or growth experiments. In historical files, **LASSO and Ridge mean the penalty
versions**. The new suite gives them explicit `_Penalty` names. Direct
`backtest` and `rolling_windows` calls retain their original 126-observation
default; both CLI entrypoints default to 252.

## Outputs and verification

New baseline runs save to `results/baselines/<UTC_timestamp>/`:

- `metrics/summary.csv`: six models and their gross performance metrics.
- `returns/` and `weights/`: daily realized returns and holdings.
- `diagnostics/*CoefficientConstraint_diagnostics.csv`: history dates, observation count, configured/actual coefficient budget, L1/squared-L2 measures, MSE and solver status.
- `config/run_metadata.json`: model names, window, budgets, fixed penalties, data identity and source fingerprints.

`--output` selects a new or empty directory; `--layout flat` retains historical
file placement. Returns and metrics are before transaction costs. Sharpe is
daily and unannualized, with no risk-free-rate subtraction.

The preceding four-model window check is preserved in
`results/baseline_window_checks/20260926/`. The six-model check is saved separately
in `results/baselines/coefficient_constraints_20260926/{126d,252d}/`.
Both use the same 123 validation days (January 2–June 30, 2026), with no July
final-test evaluation. Both used panels have zero missing/nonfinite values.
Initial history intervals are 2025-07-03–2025-12-31 for 126 observations and
2024-12-30–2025-12-31 for 252 observations.

The six-model results below use coefficient budgets t₁ = t₂ = 1 and the existing
fixed penalty strengths. They verify the new formulations; no parameter search
or replacement of the published final-test results is involved.

| Window | Model | Cumulative return | Daily Sharpe | Daily volatility | Maximum drawdown | Average gross exposure |
|---|---|---:|---:|---:|---:|---:|
| 126 | EW | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| 126 | MinVar | -2.297% | -0.01719 | 0.8770% | -8.988% | 11.977 |
| 126 | LASSO_CoefficientConstraint | 9.375% | 0.12473 | 0.5985% | -4.563% | 1.962 |
| 126 | Ridge_CoefficientConstraint | -2.533% | -0.02869 | 0.6532% | -7.427% | 7.800 |
| 126 | LASSO_Penalty | 4.458% | 0.06934 | 0.5318% | -5.022% | 2.579 |
| 126 | Ridge_Penalty | 1.417% | 0.02273 | 0.5754% | -6.135% | 2.767 |
| 252 | EW | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| 252 | MinVar | 2.936% | 0.03677 | 0.7074% | -4.612% | 8.921 |
| 252 | LASSO_CoefficientConstraint | 7.993% | 0.10241 | 0.6299% | -6.325% | 1.946 |
| 252 | Ridge_CoefficientConstraint | 3.435% | 0.04586 | 0.6436% | -4.610% | 7.429 |
| 252 | LASSO_Penalty | 6.979% | 0.09951 | 0.5673% | -4.828% | 3.036 |
| 252 | Ridge_Penalty | 5.104% | 0.07778 | 0.5389% | -4.933% | 3.125 |

All 182 tests passed. Independent review found no important issues. Tests cover
analytical constrained solutions, squared-L2 units, objective scaling, a free
weighted intercept, zero/loose budgets, invalid inputs, time causality and both
CLI entrypoints. A separate artifact audit reconstructed coefficient norms,
training MSE and realized returns from saved holdings. All constrained fits were
optimal and satisfied their budgets; original EW/MinVar/penalty holdings and
returns matched the preceding window check exactly. See the run’s
`verification.json` and `tests.log`.
