# Evaluation protocol for the final report

All portfolio decisions use the preceding 126 trading days. The forecast uses 126 dated targets and 63 earlier observations to initialize lagged features. The course and report portfolios use the same 98 assets, preprocessing and daily gross-return scoring.

| Evidence | Period and rule | Report |
| --- | --- | --- |
| Forecast calibration | Three chronological 21-day folds ending by July 31, 2024. Choose the penalty by pooled prediction MSE. Refit coefficients daily with the selected penalty. | Sections 3–4 and Table 5 |
| Portfolio validation | January 2–June 30, 2026. Compare the eight portfolios in Table 3. Course LASSO and Ridge choose their penalties on three chronological 21-day folds within each 126-day window. | Tables 3 and 4 |
| Forecast validation | Compare next-day per-asset LASSO predictions with trailing arithmetic means. Rank 20 assets before each decision. Break ties in source order after rounding to 12 decimal places. | Table 5 |
| Final evaluation | August and September 2026, 21 trading days each. Use the original release through July 31. Append observed test returns for subsequent decisions. | Table 6 |

The top 20 in Table 4 are ranked retrospectively for attribution. The historical top-20 portfolio in Table 3 ranks assets using only prior returns. The LP maximizes forecast return subject to its investment and historical risk limits, with ties resolved toward EW.

Before opening the test results, the group chose the original exposure-penalized LASSO alongside EW, course LASSO and course Ridge. The [frozen protocol](../../configs/final_test_126.json) records that amendment and the complete settings. Use arithmetic average daily return, sample standard deviation and daily Sharpe at a zero risk-free rate. Recompute the combined figures from all 42 daily returns. The monthly rank check compares prior 126-day asset means with the realized mean in the following month.

[Validation performance](../../results/course_126/validation/metrics/summary.csv) · [Attribution](../../results/course_126/validation/metrics/attribution.csv) · [Forecast scores](../../results/course_126/validation/metrics/forecasts.csv) · [Final test](../../results/course_126/final_test/metrics/test_periods.csv) · [Monthly rank correlations](../../results/course_126/final_test/metrics/monthly_rank_correlation.csv)
