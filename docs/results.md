# Saved results and recovery

The active results are a complete fixed-parameter rerun on September 27, 2026:
**26 model/period records**, all using 252 prior trading observations. Nineteen
configurations cover validation and seven cover the previously evaluated July
period. Daily fits, forecasts and decisions were recomputed; parameters were not
reselected. No old performance stream is mixed into the new summaries.

```text
results/
  report/validation/       # 19 models, 123 trading days
  report/july_test/        # Seven models, 22 trading days
  report/selection/        # Small historical selection records, not new searches
  report/logs/             # Full rerun console logs
  report/report_reference.json
  report/report_comparison.csv
  report/historical_comparison.csv
  report/verification.json
  data_audit/recent/       # Relevant historical missing-value audit
  manifest.json           # Model-to-artifact mapping and SHA-256 checksums
  runs/                   # Future timestamped executions; ignored by Git
```

Both evaluation directories contain summary metrics, daily returns, holdings,
diagnostics, predictions and run metadata. Each model is stored once per period;
there is no duplicate baseline directory. The original fitting source fingerprints
remain in each run's metadata. Added summary statistics have separate aggregation
provenance. Historical search metadata are labeled and retained with selection
records, rather than represented as results of this rerun.

Main returns are **gross**. Supplementary net returns subtract the same assumed
5 bps times full L1 traded weight for every model. Sharpe is daily and
unannualized; volatility uses sample standard deviation and drawdown is compounded.
`average_target_turnover` compares consecutive target weights, initially EW.
`average_trade_turnover` compares targets with drift-adjusted pretrade weights.
Both use half the L1 difference; report Tables 4 and 5 use different measures.

Effective asset count uses normalized absolute holdings: average over days of
`(Σ|wᵢ|)² / Σwᵢ²`. The source report uses a different definition for two numbers;
see [the discrepancy record](report-reproduction-check.md). The report has not
been changed pending review. Numeric consistency and artifact integrity are
reported separately, so a valid artifact audit does not mean every report cell
matches.

## Verify

`python -m tools.verify_report_results` checks hashes, coverage, metrics and report
rounding. Add `--csv 100_Portfolios_10x10_Daily.csv` to reconstruct returns and
actual trading from the original data. It currently reports 139/141 matching
report cells and two definition differences, not model-performance failures.

July was already observed. In particular, the trading-penalized LP was developed
after the first July results had been viewed. This run reproduces past experiments
and is not a new independent test.

## Recover earlier work

The complete pre-refactor version is on branch
`backup/pre-report-refactor-20260927`, commit `c0b1cc5`, including the input CSV,
old code and research outputs. It also has a verified external archive at
`../.cleanup-backups/project1/20260927-report-refactor/complete-pre-refactor.tar.gz`.

The curated results immediately preceding this rerun, including 126-day baseline
records and the broader data audits, are archived outside the project under
`../.cleanup-backups/project1/20260927-report-rerun/previous-results/`. Their source
code is available at local commit `2bc5ea5`. Only their active result files were
removed; all six baseline implementations still support 126-day and 252-day
execution through `--window`.
