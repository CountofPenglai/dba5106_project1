# DBA5106 Portfolio Optimization

The course benchmark uses four portfolios and a 126-trading-day window. The same framework reproduces the original 252-day report. Data preparation, rolling evaluation, metrics and result storage are reused across models; model names and fixed settings are defined in [`portfolio/model_catalog.py`](portfolio/model_catalog.py).

## Project Framework

The daily workflow is:

```text
CSV returns → preprocessing → prior training window → fit and allocate
            → evaluate on the next trading day → save metrics and diagnostics
```

There are two modeling paths:

- **Portfolio regression:** MinVar, LASSO and Ridge estimate regression coefficients and convert them into fully invested portfolio weights. Variants add coefficient budgets, loss penalties or portfolio constraints. EW uses equal weights without fitting.
- **Forecasting + LP:** LASSO predicts each asset's next-day return, then linear programming chooses portfolio weights subject to allocation and historical risk limits.

Each daily fit uses only observations available before that day's return. Course LASSO and Ridge select their penalty inside each historical window. The archived report configurations reuse their fixed settings.

| Location | Responsibility |
| --- | --- |
| `run_baselines.py`, `run_report.py` | Public entry points, with shared options in `portfolio/cli.py` |
| `portfolio/model_catalog.py`, `portfolio/config.py` | Model configurations, training presets and evaluation dates |
| `portfolio/data/` | CSV loading, preprocessing, history windows and observation weights |
| `portfolio/models/min_variance_regression.py` | MinVar and penalty-based LASSO/Ridge |
| `portfolio/models/coefficient_constrained_regression.py` | LASSO/Ridge with explicit coefficient budgets |
| `portfolio/models/constrained_portfolio.py` | Shared optimizer for portfolio constraints and loss penalties |
| `portfolio/models/lasso_return_forecast.py` | Return features, chronological calibration and rolling forecasts |
| `portfolio/decisions/` | Coefficient-to-weight mapping and direct LP allocation |
| `portfolio/engine/` | Model dispatch, rolling execution and solver handling |
| `portfolio/evaluation/` | Performance metrics, output files and reproducibility metadata |
| `portfolio/losses.py`, `portfolio/constraints.py`, `portfolio/regularization.py` | Loss, constraint and regularization settings |
| `configs/lasso_forecast.json` | Frozen LP forecast calibration and risk settings |
| `tests/`, `tools/`, `docs/` | Tests, data/result audits and detailed documentation |

To change a run, use the command-line options below. To add a configuration of an existing model, register it in the model catalog; a new model family also needs implementation and dispatch in `portfolio/engine/report_runner.py`.

## Setup and Data

Run all commands from the repository root. Python 3.12 is used by the current development environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Place `100_Portfolios_10x10_Daily.csv` in the repository root, or supply its path with `--csv`. If it is missing in a fresh clone, recover the archived input file with:

```bash
git fetch origin backup/pre-report-refactor-20260927
git restore --source=FETCH_HEAD -- 100_Portfolios_10x10_Daily.csv
```

The loader selects **Average Equal Weighted Returns -- Daily**, drops `ME9 BM10` and `BIG HiBM`, converts percentage returns to decimals, and treats `-99.99` and `-999` as missing values without filling them. The remaining 98 columns are portfolio return series, not stock prices. History starts on the first trading day on or after August 1, 2021; each fit uses only its configured rolling window.

## Running Models

### Course benchmark

Run the following from the project root:

```bash
.venv/bin/python -m tools.verify_course_benchmark
.venv/bin/python run_baselines.py
```

The course run uses EW, MinVar, LASSO_CV and Ridge_CV. January 2 through June 30, 2026 supplies 123 validation returns. The target weights for a trading day use exactly the previous **126 trading observations**, equally weighted. The first window runs from July 3 through December 31, 2025. Short positions are allowed and the portfolio weights sum to one. The daily gross return is the dot product of that day's target weights and realized returns. The reported daily Sharpe divides the arithmetic mean by the sample standard deviation (ddof=1), using a zero risk-free rate.

All four portfolios use the same 98 columns in source order. The file contains several tables; the equal-weighted daily table describes how stocks are combined **within** each asset. EW also weights the 98 resulting assets equally. Missing markers and blank cells stay missing. An exact raw value of -99 represents -99%, while -99.99 and -999 are the file's missing markers. The July 2026 data release has zero missing cells in the retained panel from August 2, 2021 through June 30, 2026.

For each day, course LASSO and Ridge use the handout's 21-point native sklearn alpha grid from 1e-8 to 1e8. Both regress EW returns on the return differences against the final retained asset, with an intercept and decimal-return inputs. We chose three expanding chronological folds with 21 validation days each. At a 126-day window the fold training lengths are 63, 84 and 105. The criterion is mean squared error. The winning estimator is refitted on the full window before forming the next day's holdings. Each selected alpha is saved in the daily diagnostics.

**Reference checks.** January 2, 2026 is observation 26,152 in the full equal-weighted panel. On the first six assets, the six MinVar weights match the handout to three decimals and the first return is **1.393062%**. The handout also quotes full-period 100-asset Sharpe ratios of 0.111 (EW) and 0.021 (MinVar). On this file and the stated dates, the 100-asset diagnostic gives 0.13607 and 0.01147; the required 98-asset panel gives 0.13253 and −0.01719. An independent covariance solve gives the same daily MinVar returns. The cause of the remaining reference discrepancy is unresolved. Keep the 100-asset figures as a diagnostic only.

See the [handout checks and input fingerprint](results/course_126/handout_checks.json) and the [four-model validation results](results/course_126/validation/metrics/summary.csv).

### Archived baseline alternatives

The fixed-penalty and coefficient-budget baselines used in the submitted report can still be selected by name. Use an explicit window to reproduce that report or compare history lengths.

```bash
.venv/bin/python run_baselines.py --window 252 --models EW MinVar LASSO_Penalty Ridge_Penalty LASSO_CoefficientConstraint Ridge_CoefficientConstraint
```

Coefficient-constrained LASSO uses `sum(abs(beta)) <= budget`; coefficient-constrained Ridge uses `sum(beta**2) <= budget`. Both budgets default to 1 and can be changed with `--lasso-coefficient-budget` and `--ridge-coefficient-budget`. They constrain model coefficients, rather than portfolio gross exposure. The penalty versions retain fixed strengths of `1e-5` and `1e-4`, respectively.

### Report Models

```bash
# List all model IDs, titles and report sections without running them.
.venv/bin/python run_report.py --list

# Run all 19 retained configurations on validation, using 252 days of history.
.venv/bin/python run_report.py

# Run selected models.
.venv/bin/python run_report.py --models LASSO_HardConstraints LASSO_ExposurePenalty
.venv/bin/python run_report.py --models LP_Original LP_TradingPenalty

# Reproduce the seven previously evaluated July portfolios.
.venv/bin/python run_report.py --period july_test
```

The report models cover hard constraints, exposure and turnover penalties, a single-asset weight cap, market-beta constraints, and the two LP objectives. Their implementations share the framework above.

| Option | Behavior |
| --- | --- |
| `--window N` | Prior trading observations. Course baselines default to `126`. Archived report runs default to `252`. Including a course CV model in a mixed run defaults it to `126`. |
| `--period validation` | Default evaluation period: January 2–June 30, 2026. |
| `--period july_test` | July 1–31, 2026. `run_report.py` defaults to the seven selected portfolios; `run_baselines.py` runs the four course baselines by default. |
| `--models ID ...` | Override the default model selection; use `--list` to see available IDs. |
| `--start YYYY-MM-DD`, `--end YYYY-MM-DD` | Override the evaluation dates while retaining earlier data for training. |
| `--csv PATH` | Use an input CSV at another location. |
| `--output PATH` | Write to a new or empty directory instead of the default timestamped directory. |

**LP runs require `--window 252`**, matching the frozen forecast calibration. To use another window with `run_report.py`, select only non-LP models. For LP, enough earlier data must also be available to construct the forecast features; the saved calibration data fingerprint is checked before execution.

For a quick two-day smoke run:

```bash
.venv/bin/python run_baselines.py --window 126 --start 2026-01-02 --end 2026-01-05
```

July is the project's already-used final evaluation period. The July command reproduces those results; use validation for model comparisons and selection.

## Outputs

New runs are saved under `results/runs/baselines/<timestamp>/` or `results/runs/report/<timestamp>/`:

```text
metrics/summary.csv          Aggregate performance and holdings statistics
returns/<model>.csv          Daily gross returns, supplementary net returns and costs
weights/<model>.csv          Daily portfolio weights
diagnostics/<model>.csv      Fitting, risk and trading diagnostics
config/run_metadata.json     Run settings, data/source hashes and package versions
predictions/lasso_forecast.csv  Asset return forecasts, for LP runs
```

LP runs also save `diagnostics/lasso_forecast.csv`. Main performance metrics use **gross returns before transaction costs**, daily unannualized Sharpe with zero risk-free rate, sample volatility and compounded maximum drawdown.

The 126-day benchmark is stored in `results/course_126/`. Curated 252-day report results are stored separately in `results/report/validation/` and `results/report/july_test/`; historical parameter-selection evidence is in `results/report/selection/`. Normal runs leave these records unchanged and reject nonempty output directories.

## Checks and Further Reading

```bash
# Unit and integration tests.
.venv/bin/python -m unittest discover -s tests -v

# Check saved results and reconstruct their gross returns from the input data.
.venv/bin/python -m tools.verify_report_results --csv 100_Portfolios_10x10_Daily.csv

# Inspect missing values in the retained history.
.venv/bin/python audit_missing_values.py --start 2021-08-01 --end 2026-07-31
```

The result verifier compares report figures with the stored `results/report/report_reference.json` snapshot; it does not read the latest Word document.

- [Model formulas and report mapping](docs/architecture/report-models.md)
- [Validation and July results](docs/experiments/final-model-summary.md)
- [Saved-result provenance and recovery](docs/results.md)
