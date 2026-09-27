# DBA5106 Portfolio Optimization

A shared framework for the six baseline portfolios and the models retained in the final report. Data preparation, rolling evaluation, metrics and result storage are reused across models; model names and fixed settings are defined in [`portfolio/model_catalog.py`](portfolio/model_catalog.py).

## Project Framework

The daily workflow is:

```text
CSV returns → preprocessing → prior training window → fit and allocate
            → evaluate on the next trading day → save metrics and diagnostics
```

There are two modeling paths:

- **Portfolio regression:** MinVar, LASSO and Ridge estimate regression coefficients and convert them into fully invested portfolio weights. Variants add coefficient budgets, loss penalties or portfolio constraints. EW uses equal weights without fitting.
- **Forecasting + LP:** LASSO predicts each asset's next-day return, then linear programming chooses portfolio weights subject to allocation and historical risk limits.

Each daily fit uses only observations available before that day's return. The run commands reuse fixed hyperparameters; they refit coefficients and weights without performing a new parameter search.

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

### Baselines

The six baseline IDs are `EW`, `MinVar`, `LASSO_CoefficientConstraint`, `Ridge_CoefficientConstraint`, `LASSO_Penalty` and `Ridge_Penalty`.

```bash
# Run all six baselines; only --window changes the training history.
.venv/bin/python run_baselines.py --window 126
.venv/bin/python run_baselines.py --window 252

# Run only the penalty-based LASSO and Ridge baselines.
.venv/bin/python run_baselines.py --window 252 --models LASSO_Penalty Ridge_Penalty
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
| `--window N` | Number of prior trading observations; default `252`. Non-LP models also support `126` and other feasible lengths. |
| `--period validation` | Default evaluation period: January 2–June 30, 2026. |
| `--period july_test` | July 1–31, 2026. `run_report.py` defaults to the seven selected portfolios; `run_baselines.py` still runs all six baselines. |
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

Curated report results are stored separately in `results/report/validation/` and `results/report/july_test/`; historical parameter-selection evidence is in `results/report/selection/`. Normal runs leave these records unchanged and reject nonempty output directories.

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
