# Independent two-stage artifact audit

Status: **passed**.

Command:

`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python .two-stage-work/audit_two_stage.py`

The audit wrote `results/two_stage_lasso_20260922/verification.json` only after every check completed.

## Evidence

- Source CSV SHA-256 and the development-prefix digest match the saved provenance.
- Reconstructed raw realized returns for 8 frozen models across 3,832 saved model-days; maximum absolute error `1.10e-16`.
- Reconstructed returns and asset weights from saved U/V/E components and frozen theta; maximum errors `1.11e-16` and `4.44e-16` respectively. Every theta sums to one; the minimum coefficient is nonnegative.
- Independently recomputed all 24 decision-training mean, sample-volatility, and Sharpe/return-minus-volatility objective rows; maximum absolute error `9.20e-17`.
- Recomputed all four gamma selections from the original grid and validation compounded return. Selected gammas are LASSO-252d `0`, LASSO-3y `0.05`, Mean-252d `0`, Mean-3y `0`.
- Independently recomputed validation mean, sample volatility, daily Sharpe, compounded return, and maximum drawdown for all 8 models; maximum absolute error `1.29e-15`.
- Independently rebuilt drifted pre-trade turnover for 20 model/component strategies; maximum absolute error `2.98e-15`.
- Independently refit one Ledoit-Wolf covariance per date from the preceding configured history, then checked 20 candidate/final weight streams. Maximum budget error `4.33e-12`; maximum gross-limit excess `4.68e-09`; maximum risk ratio `1.0000000001174834`; no historical-wealth shortfall.
- Independently rebuilt first-validation lag features, scaler, and sklearn LASSO for `SMALL LoBM` under both 252-day and three-year windows; both forecast errors are exactly zero.
- Compared January–June weights and returns for all four legacy baseline variants against the original repository's prior result directory: 8 artifacts checked, maximum absolute difference exactly zero.
