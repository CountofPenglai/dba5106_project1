# Independent direct-LP artifact audit

Status: **passed**.

Command:

`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python .lp-work/audit_direct_lp.py`

The audit wrote `results/direct_lp_20260922/verification.json` only after every check completed.

## Evidence

- The raw source SHA-256, independently reconstructed development-frame digest, 25 current code hashes, and 11 optional run-start code hashes matched saved provenance.
- Reconstructed all five 98-asset weight and realized-return streams over all 123 validation dates. The saved validation summary covered 123 dates; no development backtest or development summary was produced.
- For every date and strategy, independently rebuilt the prior-three-calendar-year history, EW-relative centered-simple-return MAD cap, zero-initialized uncompounded absolute drawdown cap, budget, gross leverage, absolute-position, and historical one-day wealth-floor checks.
- Recomputed every saved diagnostic and every forecast projection `mu @ w`; maximum diagnostic error was `6.66e-16` and maximum projection error was `7.37e-17`.
- Reconstructed drifted pre-trade turnover for all 5 strategies; maximum absolute error was `2.98e-15`.
- Independently recomputed validation compounded return, sample daily volatility, daily Sharpe, compounded relative drawdown, leverage, position, and turnover metrics; maximum metric error was `8.44e-15`.
- Calculated illustrative validation cost sensitivity at 5 and 10 bps one-way using daily gross return minus `2 * drifted turnover * rate`, then compounding. These rates were not used for model or parameter selection.
- Matched 20 imported weight/return artifacts for 10 existing reference strategies and their shared numeric comparison fields; maximum absolute error was `4.44e-15`.
- Independently solved 6 real primary LPs with CVXPY/CLARABEL using affine budget/wealth/running-peak constraints and absolute-value gross/MAD/position constraints. All statuses were optimal and solutions were independently feasible. Maximum primary-objective difference was `1.27e-12`; maximum saved primary-to-final gap was `1e-10` against the `1e-10` secondary tolerance.

## Illustrative validation cost sensitivity

| Strategy | Gross | 5 bps one-way | 10 bps one-way |
|---|---:|---:|---:|
| LP_LASSO_252d | 16.27% | 13.71% | 11.20% |
| LP_Mean_252d | 16.98% | 15.73% | 14.50% |
| LP_LASSO_3y | 11.89% | 10.77% | 9.67% |
| LP_Mean_3y | 11.47% | 10.44% | 9.41% |
| EW | 16.93% | 16.88% | 16.84% |

The LP risk measures are proxies: MAD is the mean absolute deviation of centered historical simple returns, and drawdown is the largest decline in uncompounded cumulative simple return from a running peak that includes initial zero. Reported performance drawdown is separately computed from compounded wealth. No August 2026 final-test return was read or evaluated.
