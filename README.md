# DBA5106 Portfolio Optimization

A compact implementation of the models in **DBA5106_Final_Report_Draft 2**.
Six baselines and all report comparisons share data handling, rolling execution
and performance reporting. Model settings live in [the model catalog](portfolio/model_catalog.py).

## Run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Six baselines; change one option to switch the training window.
.venv/bin/python run_baselines.py --window 126
.venv/bin/python run_baselines.py --window 252

# Report validation: January 2–June 30, 2026; 252 prior trading observations.
.venv/bin/python run_report.py
.venv/bin/python run_report.py --list
.venv/bin/python run_report.py --models LASSO_HardConstraints LASSO_ExposurePenalty
.venv/bin/python run_report.py --models LP_Original LP_TradingPenalty

# Explicitly reproduce the seven already-evaluated July portfolios.
.venv/bin/python run_report.py --period july_test
```

Place `100_Portfolios_10x10_Daily.csv` in the repository root or pass `--csv`.
The complete input file is preserved on branch
[`backup/pre-report-refactor-20260927`](https://github.com/CountofPenglai/dba5106_project1/tree/backup/pre-report-refactor-20260927).
To recover that exact file from a clone, fetch the branch and restore **only the
CSV** from it; do not check out the entire backup over current work.

New runs go to `results/runs/<suite>/<timestamp>/`. `--output` may select another
empty directory. No command overwrites the curated report records or searches
for new hyperparameters. The LP's frozen calibration supports 252 days; choose
non-LP models to experiment with other window lengths.

## Read

- [Model architecture, formulas and report mapping](docs/architecture/report-models.md)
- [Report validation and July results](docs/experiments/final-model-summary.md)
- [Saved results, provenance and recovery](docs/results.md)
- [Report reproduction discrepancies awaiting review](docs/report-reproduction-check.md)
- [Missing-value audit](docs/data-quality/missing-values-2026-09-22.md)

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m tools.verify_report_results
.venv/bin/python -m tools.verify_report_results --csv 100_Portfolios_10x10_Daily.csv
.venv/bin/python audit_missing_values.py --start 2021-08-01 --end 2026-07-31
```

The main metrics use **gross returns**, daily unannualized Sharpe (zero risk-free
rate), sample volatility and compounded maximum drawdown. Trading-adjusted net
returns are supplementary. July was already evaluated; reproducing it is not a
new untouched test, and it must not be used for further parameter selection.
