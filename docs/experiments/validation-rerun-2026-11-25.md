# Train/validation rerun (November 25, 2026)

A rerun of the training and validation stage on `main` at `33d7832`, using only the original `100_Portfolios_10x10_Daily.csv` (SHA-256 `11399a83…`, the course benchmark file). The August–September test file `100_Portfolios_10x10_Daily_Test_Aug_Sep_2026.csv` was **not read**. No date after June 30, 2026 is evaluated, and each daily fit uses only observations before that day.

- **Training:** each day's weights are fitted on the preceding window (126 trading days for the course baselines, 252 for the report models). The first window starts in 2025.
- **Validation:** January 2 – June 30, 2026, 123 trading days.
- No parameters were reselected; all models use their existing fixed or cross-validated settings.

## Commands

```bash
.venv/bin/python run_baselines.py --csv 100_Portfolios_10x10_Daily.csv --period validation
.venv/bin/python run_report.py    --csv 100_Portfolios_10x10_Daily.csv --period validation
```

Outputs: [`course_126/`](../../results/validation_rerun_2026-11-25/course_126/) (EW, MinVar, LASSO_CV, Ridge_CV) and [`report_252/`](../../results/validation_rerun_2026-11-25/report_252/) (all 19 report configurations).

## Results (daily, gross, zero risk-free rate)

| Model | Average return | Std. deviation | Sharpe |
|---|---:|---:|---:|
| EW | 0.132% | 0.997% | 0.133 |
| MinVar (126) | −0.015% | 0.877% | −0.017 |
| LASSO_CV (126) | 0.012% | 0.541% | 0.023 |
| Ridge_CV (126) | 0.000% | 0.594% | 0.000 |
| LASSO with Investment Size Control (252) | 0.089% | 0.601% | 0.147 |
| LASSO with 10% Weight Cap (252) | 0.099% | 0.636% | 0.156 |
| Ridge with Beta Control, Conservative (252) | 0.087% | 0.573% | 0.151 |
| LASSO Forecasting + Trading-Penalized LP (252) | 0.151% | 1.019% | 0.148 |

All 19 report models are in [`report_252/metrics/summary.csv`](../../results/validation_rerun_2026-11-25/report_252/metrics/summary.csv).

## Reproducibility

The rerun matches the saved results: course baselines differ from `results/course_126/validation` by at most 5e-16, and report models from `results/report/validation` by at most 2e-11 in mean and volatility and 3e-9 in Sharpe (solver tolerance).
