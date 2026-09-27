# Report reproduction check

All 19 validation configurations and seven previously evaluated July portfolios
were rerun on September 27, 2026, using the refactored code, the same 98 assets,
a 252-day window and frozen parameters. No new model selection was performed.
The source Word report has **not been edited**; discrepancies below await review.

## Numeric comparison

The reference is `DBA5106_Final_Report_Draft 2.docx`, identified by SHA-256 in
[the extracted reference](../results/report/report_reference.json). Its seven
tables contribute 138 numeric cells; three additional July figures come from
Section 3.2. Of these **141 comparisons, 139 match at the displayed precision**.
Returns, daily Sharpe, volatility, drawdown, gross exposure, turnover, largest
weight and average beta all match. The two differences concern diversification.

| Location | Model | Report | Current result | Metric |
|---|---|---:|---:|---|
| Table 6 and Section 5.1 | Selected/exposure-penalized LASSO | 3.46 | 3.562933 ≈ 3.56 | Average effective assets |
| Table 6 and Section 5.1 | LASSO with a 10% weight cap | 12.03 | 12.423471 ≈ 12.42 | Average effective assets |

This is a **definition mismatch**, not a failure to reproduce portfolio weights:

- The report uses the daily reciprocal of squared raw weights, `1 / Σwᵢ²`,
  then averages over days. Applying that formula to the fresh holdings gives
  3.455575 and 12.027041, which reproduce 3.46 and 12.03.
- The current project's curated summary uses absolute weight shares
  `qᵢ = |wᵢ| / Σ|wᵢ|`, then averages `1 / Σqᵢ²`. This gives 3.56 and 12.42.
- These definitions agree for fully invested long-only portfolios. With short
  positions, total absolute weights exceed one and the values differ.

No definition is changed merely to match the report. The next review should
choose and state one definition consistently in Table 6 and Section 5.1.
The report's `3.46` appears twice in that section, and `12.03` appears once.

## Wording to review separately

These are disclosure/description issues, distinct from numerical reproduction:

1. **July's role (Sections 1 and 6).** The report describes July as a frozen test
   kept separate from selection, yet identifies the final winner by its July
   performance. The trading-penalized LP was also developed after the original
   July comparison had been viewed. Reproducing those observations does not
   restore an untouched test; the existing project summary already records
   this chronology. Review the wording together rather than changing models.
2. **Market-beta starting point (Section 5.2).** “Building on the above” can imply
   inheritance of the selected model's gamma=2 penalty or the 10% cap. The
   implementation uses hard-constrained LASSO/Ridge with gamma=0 and no such
   individual-weight cap. Its reported numerical results are reproduced.

## Evidence and checks

- Validation: January 2–June 30, 2026, 123 trading days; July: July 1–31, 22 days.
- All 2,491 daily model portfolios independently satisfy the applicable budget,
  coefficient, gross exposure, variance, wealth, concentration, beta and LP
  risk constraints, within solver tolerances.
- Gross returns were rebuilt from daily weights and actual asset returns.
  Target-to-target turnover and drift-adjusted trading were checked separately.
- Compared with the previously saved results, the largest cumulative-return
  difference is 3.18e-9 and the largest daily-Sharpe difference is 8.86e-9.
  These do not change any displayed report return or Sharpe.
- The validation log contains three NumPy invalid-reduce warnings. Follow-up
  checks did not reproduce them; final saved values are finite, all daily
  optimizations were accepted as optimal, and independent feasibility checks
  passed. The warnings remain in the execution log.

See [all report comparisons](../results/report/report_comparison.csv),
[old/new numerical differences](../results/report/historical_comparison.csv),
[constraint verification](../results/report/verification.json), and the fresh
[validation](../results/report/validation/metrics/summary.csv) and
[July](../results/report/july_test/metrics/summary.csv) summaries.

The test suite was reduced from 75 to 59 tests: 18 duplicate or archived-direction
cases removed and two focused checks added for concentration accounting and
report rounding. Core 126/252-day support, causal training, independent optimizer
checks, feasibility failures, forecasting, LP, costs and artifact checks remain.
