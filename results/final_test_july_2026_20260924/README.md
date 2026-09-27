# July 2026 final-test artifacts

`metrics/report_final_test_summary.csv` is the authoritative gross-only table used in `docs/experiments/final-model-summary.md`. It merges the original six configurations with LP_Current. All reported return, Sharpe, volatility and drawdown metrics exclude transaction costs.

`metrics/combined_final_test_summary.csv` retains explicit `gross_*` and supplemental `net_*` fields for all seven paths. `metrics/final_test_summary.csv` is the original LP_Current/EW exporter output: unprefixed metrics there are net and `gross_*` metrics are gross. Do not use the unprefixed net metrics for the report ranking. Net diagnostics assume 5 bps per unit of one-way traded notional, initial EW holdings, market drift, and first-order fee deduction.

LP_Current settings were frozen before this execution, with no parameter search. July had already been evaluated before this LP refinement was developed, so this is not a newly untouched test. Original six-model holdings and results are unchanged; only LP_Current was newly evaluated.

From the repository root, reproduce the current LP/EW run in a fresh output directory:

```bash
PYTHONPATH=. .venv/bin/python results/final_test_july_2026_20260924/config/july_current_lp_runner.py --output /tmp/current_lp_july_reproduction
```

Recheck this saved run and its merged accounting without model refits:

```bash
PYTHONPATH=. .venv/bin/python results/final_test_july_2026_20260924/config/audit_and_merge.py --result results/final_test_july_2026_20260924
```

`config/pre_run_selection.json` preserves the specification written before the July forecasts and solves. `config/verification.json` records checks and source fingerprints. Code and numerical package checks in the evaluation runner intentionally reject changed production sources or incompatible package versions.
