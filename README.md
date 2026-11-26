# DBA5106 Portfolio Optimization

The project reproduces the [final report](docs/DBA5106_Project1_Final_Report_Outline.docx) using 98 asset portfolios and the preceding 126 trading days for each decision. The eight validation portfolios and the four prespecified test portfolios use one daily backtest and one gross-return scoring convention.

## Run the report

From the repository root, use Python 3.12 and install the dependencies.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Reproduce January–June portfolio performance, attribution and forecast tables.
.venv/bin/python run_report.py

# Reproduce the fixed August–September holdout and monthly rank check.
.venv/bin/python run_report.py --period final_test

# Run the four course benchmarks or list the portfolios in the report.
.venv/bin/python run_baselines.py
.venv/bin/python run_report.py --list

# Verify the curated results against the actual Word tables and run the tests.
.venv/bin/python -m tools.verify_report_results
.venv/bin/python -m unittest discover -s tests
```

Commands write to fresh directories under `results/runs/`. Use `--output PATH` to choose a new or empty directory. For a shorter validation example, run `run_baselines.py --start 2026-01-02 --end 2026-01-06` with the same interpreter. Validation accepts a subset via `--models`. The final-test command takes its dates, releases and portfolios from the [protocol fixed before testing](configs/final_test_126.json).

## Models and evaluation

| Stage | Models | Data and output |
| --- | --- | --- |
| Validation | EW, MinVar, course LASSO and Ridge, exposure-penalized LASSO, its 10% single-asset cap variant, historical top 20 in equal weights, and risk-constrained LP | 123 days from January 2 to June 30, 2026. [Performance](results/course_126/validation/metrics/summary.csv) and [attribution](results/course_126/validation/metrics/attribution.csv) |
| Forecast comparison | Per-asset LASSO and trailing 126-day mean | The LASSO penalty is calibrated through July 31, 2024. Its daily predictions serve both the LP and the [forecast comparison](results/course_126/validation/metrics/forecasts.csv). |
| Final test | EW, course LASSO, course Ridge, exposure-penalized LASSO | One continuous 42-day run. Report [August, September and pooled performance](results/course_126/final_test/metrics/test_periods.csv) and [monthly rank correlations](results/course_126/final_test/metrics/monthly_rank_correlation.csv). |

The reader takes the equal-weighted daily panel from the two French CSV files at the repository root, drops `ME9 BM10` and `BIG HiBM`, and converts percentage points to decimals. Blanks, −99.99 and −999 are missing. Exact −99 represents −99%. Validation and history through July use the original release. The updated file contributes only August–September observations to the backtest.

Course LASSO and Ridge select their penalties by regression MSE within each training window using three chronological 21-day folds. Table 3 compares portfolios by daily Sharpe. Returns are the sum of opening weights times realized asset returns. Sharpe is the arithmetic daily mean divided by sample daily standard deviation, at a zero risk-free rate.

## Project structure

- `portfolio/data/` reads and filters returns and selects historical windows.
- `portfolio/model_catalog.py` defines the report portfolios. `portfolio/models/` fits the regressions. `portfolio/decisions/` maps coefficients to holdings and solves the risk-constrained LP.
- `portfolio/engine/report_runner.py` fits, allocates, scores and reconciles each portfolio. `portfolio/evaluation/` computes metrics, attribution and forecast scores and writes results.
- `portfolio/evaluation/final_test.py` applies the frozen holdout protocol. Each published run includes its daily holdings, returns, diagnostics and data fingerprints.

The curated holdout also retains the [first evaluation record](results/course_126/final_test/config/first_evaluation.json). Its weights, twelve period rows and monthly rank correlations agree with the cleaned pipeline.

The [evaluation protocol](docs/experiments/course-evaluation-plan.md) maps each question to its data and table. The [handout checks](results/course_126/handout_checks.json) reproduce the six-asset example. The original source release gives different 100-asset reference Sharpes from the handout. The report uses the prescribed 98 assets.
