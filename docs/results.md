# Saved results and recovery

The curated snapshot contains **32 model/period/window records**: six baselines
at 126 and 252 days, thirteen additional report configurations at 252 days,
and seven July evaluations. Existing report numbers are preserved, not rerun
or replaced by new parameter choices.

```text
results/
  baselines/126d/           # Six baseline streams and their summary
  baselines/252d/           # Six baseline streams and their summary
  report/validation/       # 19-row summary; additional model streams
  report/july_test/        # Seven previously evaluated final-test portfolios
  report/selection/        # Exposure search, forecast CV and LP penalty evidence
  report/provenance/       # Unmodified historical settings, hashes and audits
  data_audit/              # Original missing-value checks
  manifest.json           # Record-to-model/source mapping and file checksums
  runs/                   # New timestamped executions; ignored by Git
```

The six baseline streams are shared by the report validation table instead of
being stored twice. The manifest points each row to its canonical daily
holdings and gross-return stream. Repeated τ=0/selected-LASSO and cost-only LP
labels have been consolidated. Mathematically equivalent beta-unbounded and
hard LASSO retain distinct records because their numerical results differ.

Historical LP files originally used unprefixed return/Sharpe fields for **net**
performance. Their migrated `return` column and main summary now consistently
mean **gross**. When available, `net_return` and `trading_cost` retain the
supplementary accounting. Original source files are identified by SHA-256 in
the manifest; original metadata are copied without changing their hashes.

`average_target_turnover` compares consecutive target weights, starting from
EW. `average_trade_turnover` compares each target with drifted pretrade weights.
Both use one-half of L1 change. Report Section 3.3 uses the former; Section 4.1
uses the latter. They must not be interchanged.

Parameter-selection evidence is retained, while daily holdings for discarded
search candidates are in the external backup. The LP forecast freeze is now
tracked under `configs/`; it no longer loads settings from the removed stability
experiment. Its calibration-only fingerprint is derived after verifying the
original complete-data fingerprint, which is also retained separately.

## Historical interpretation

The exposure-penalty model had the strongest observed July return/Sharpe among
the evaluated models. This observation is not a new selection on untouched
data. The trading-penalized LP was developed after the first July results had
already been reviewed. Reproduction commands perform no new hyperparameter
search and do not change these historical limitations.

## Verify and recover

`python -m tools.verify_report_results` checks all curated file checksums,
coverage and report metrics. Add `--csv 100_Portfolios_10x10_Daily.csv` to
independently reconstruct every daily gross return from the saved holdings.

The complete pre-refactor version is on branch
`backup/pre-report-refactor-20260927`, commit
`c0b1cc517dc3edeb205898aa4ff5b590ec74a62c`. It includes the input CSV and all old
code/results. A verified external archive also exists at
`../.cleanup-backups/project1/20260927-report-refactor/complete-pre-refactor.tar.gz`
relative to the original project directory. Its SHA-256 is stored alongside it.
Restore historical experiments into a **separate directory**, using their
recorded package versions; their source fingerprints are intentionally not
rewritten to match this refactor.
