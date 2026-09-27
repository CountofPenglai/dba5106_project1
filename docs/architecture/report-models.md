# Report models and architecture

## Responsibilities

| Location | Responsibility |
|---|---|
| `run_baselines.py`, `run_report.py` | Two public evaluation commands |
| `portfolio/model_catalog.py` | Model names, report sections and fixed parameters |
| `portfolio/config.py` | Training-window presets and evaluation dates |
| `portfolio/data/` | CSV section selection, preprocessing and causal windows |
| `portfolio/models/min_variance_regression.py` | Linear regression parameterization and fixed penalties |
| `portfolio/models/coefficient_constrained_regression.py` | Explicit L1/squared-L2 coefficient budgets |
| `portfolio/models/constrained_portfolio.py` | Shared holding constraints and loss overlays |
| `portfolio/models/lasso_return_forecast.py` | Causal return features, chronological calibration, rolling LASSO forecasts |
| `portfolio/decisions/linear_program_allocation.py` | Direct asset-weight LP and independent feasibility checks |
| `portfolio/engine/report_runner.py` | Shared daily fit → allocate → observe loop |
| `portfolio/evaluation/` | Metrics, immutable output destinations and provenance |
| `configs/lasso_forecast.json` | Portable frozen LP forecast settings |

## Data and dates

The loader reads **Average Equal Weighted Returns -- Daily** from the French
100 Size–BM portfolio file. These columns are portfolio return series, not
individual stock prices. It removes `ME9 BM10` and `BIG HiBM` before numeric
processing, converts `-99.99`/`-999` to missing values, and divides percentages
by 100. It neither fills missing values nor silently removes incomplete dates.

Commands retain history from **August 2, 2021**, the first available trading day
on/after August 1. The current panel has **1,255 dates × 98 assets**, ending
July 31, 2026, with **zero missing cells**. The entire retained history is not
used in each daily fit: for the first validation date (January 2, 2026), a
252-day fit uses **December 30, 2024–December 31, 2025**; a 126-day fit uses
**July 3–December 31, 2025**. The forecast layer additionally needs 63 feature
warm-up rows before its 252 complete training targets.

Validation is January 2–June 30, 2026 (123 observations); July reproduction is
July 1–31 (22 observations). Daily refits can use earlier observations from the
same evaluation period, but never the current or a later return.

## Six baselines: report Section 2

Let e be equal weights, N = [I; -1ᵀ], X = R N and y = R e. Fit the coefficients
β and a free intercept, then map them to fully invested weights **w = e − Nβ**.
The free intercept makes minimum regression MSE equivalent to minimum sample
portfolio variance. The intercept does not enter the holdings.

| Model ID | Optimization |
|---|---|
| `EW` | w = e; no estimation |
| `MinVar` | Minimize MSE, without a coefficient penalty/budget |
| `LASSO_CoefficientConstraint` | Minimize MSE subject to Σ|β| ≤ 1 |
| `Ridge_CoefficientConstraint` | Minimize MSE subject to Σβ² ≤ 1 |
| `LASSO_Penalty` | Minimize MSE + 1e-5 × Σ|β| |
| `Ridge_Penalty` | Minimize MSE + 1e-4 × Σβ² |

Coefficient budgets are adjustable through `--lasso-coefficient-budget` and
`--ridge-coefficient-budget`; default 1 is an untuned reference. These budgets
constrain regression coefficients, not portfolio gross exposure. The penalty
conventions remain unchanged: sklearn LASSO alpha = λ/2; Ridge alpha = nλ.

## Shared constrained regression: Sections 3 and 5

All configurations retain net investment 1, gross exposure ≤ 1.5, training
volatility ≤ EW training volatility and historical wealth ≥ 1e-6. Only the
following settings differ; they are not separately copied model implementations.

| Report | Model ID | Difference from the hard-constrained model |
|---|---|---|
| 3.1 | `LASSO_HardConstraints` | Fixed LASSO penalty; no reward/holding penalty |
| 3.2 | `LASSO_ExposurePenalty` | γ=2, η=0 investment-size penalty |
| 3.3 | `LASSO_Turnover_0p1`, `LASSO_Turnover_1` | Add τ=0.1 or 1 target-turnover penalty to γ=2 |
| 5.1 | `LASSO_WeightCap_10pct` | Add |wᵢ| ≤ 0.1 to γ=2 |
| 5.2 | `LASSO_BetaUnbounded`, `Ridge_BetaUnbounded` | No beta bounds; γ=0 |
| 5.2 | `LASSO_BetaMarket`, `Ridge_BetaMarket` | Beta in [0.8, 1.2]; γ=0 |
| 5.2 | `LASSO_BetaConservative`, `Ridge_BetaConservative` | Beta in [0.5, 0.9]; γ=0 |

Investment-size penalty = γ × EW variance × [max(0, (Σ|w|−1)/0.5)]².
Target-turnover penalty = τ × EW variance × 0.5Σ|w − previous target|.
The τ=0 control is `LASSO_ExposurePenalty`; it is stored only once. Each
single-stage strategy starts its evaluation period from EW.

Market beta is estimated relative to the EW return series in the same training
window. The beta models build on hard-constrained regression, **without** the
γ=2 penalty or the 10% cap. The historical beta formulation and solver defaults
are retained within the common fitter, with added independent feasibility
checks. This preserves the slight numerical distinction between Table 7's
unbounded LASSO and Section 3.1's mathematically equivalent hard LASSO.

The selected η=0, γ=2 values are fixed for reproduction. Historical screening
and limited refinement records, budgets, stopping rules and selected values
remain in `results/report/selection/` and its immutable provenance. The obsolete
search launchers are recoverable from the backup, rather than part of the
normal evaluation flow. The common loss retains the small reward term needed
to explain that parameterization; no positive-reward model is selected here.

## Forecasting and LP: Section 4.1

The forecaster uses six causal features (lagged daily, 5/21/63-day cumulative
returns, 21-day volatility and 21-day EW return). Each asset's features are
standardized on its own prior training rows before LASSO fitting. One shared
alpha, **0.002511886431509582**, was calibrated using three chronological
21-day folds ending July 2024 and is fixed in `configs/lasso_forecast.json`.
The chronological calibration function remains available; normal runs reuse
the frozen value and verify the calibration data fingerprint.

The LP directly optimizes 98 weights with full investment, gross exposure ≤1.5,
absolute weight ≤5%, historical MAD and uncompounded drawdown no larger than
EW, and positive wealth in every historical scenario. Forecast and risk windows
are both 252 observations. The primary-optimum-preserving EW tie-break and
independent feasibility checks are retained.

- `LP_Original`: maximize predicted portfolio return.
- `LP_TradingPenalty`: maximize predicted return − 0.0005Σ|w − pretrade weights|.

Pretrade weights include the previous day's price drift. The coefficient
represents a one-way 5 bps trading-cost assumption. Main reported performance
uses gross returns; it does not subtract this cost. Supplementary net returns
subtract 0.0005 times full L1 traded weight for every model in a new run.

No surviving model needs a historical result directory to train or evaluate.
