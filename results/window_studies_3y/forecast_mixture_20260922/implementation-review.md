# Final bounded review

Status: no remaining actionable defects found in the reviewed new modules. Earlier findings below have been resolved and independently rechecked. This is a bounded implementation review, not an audit of the ongoing full real-data run. No production files were edited or whole experiment rerun.

## Return-optimum uniqueness shortcut

Reviewed the new shortcut in `portfolio/candidates.py`, its targeted test, and the saved `.two-stage-work/real-failure.npz` fixture.

The mathematical argument is valid. Write the normalized risk constraint as `||B w|| <= 1`, with `B.T B = Sigma / variance_limit`. When Sigma is positive definite, B is invertible. For distinct fully invested vectors u and v, equality in the Euclidean norm's convexity inequality would require `B u` and `B v` to be nonnegative collinear. Invertibility would then imply `u = c v`; because both have coordinate sum one, c must equal one, contradicting distinctness. Therefore `||B w||` is strictly convex on the affine budget set `sum(w)=1`.

For an optimal dual solution with positive risk multiplier lambda, the Lagrangian on this affine set is the strictly convex term `lambda * ||B w||`, plus convex leverage terms and affine return/wealth terms. It is strictly convex and has at most one minimizer. Every primal optimum minimizes that same optimal Lagrangian by primal-dual optimality, so the primal return optimum is unique. Consequently no selection between tied primary optima is required. A positive multiplier is essential; positive-definite covariance alone would not establish uniqueness of the linear-return objective.

The implementation uses a positive-definiteness threshold and `risk_dual > 1e-6` after the primary solver reports optimal and independent feasibility checks succeed. This is a numerical certificate subject to the solver's existing tolerances, as are the other optimization certificates; it does not accept a failed solve or relax constraints. Noncertified cases retain the secondary tie-break. The existing normal two-asset grid test continues independently checking the returned objective optimum.

Saved real fixture replay (753 observations, 98 assets):

- Solver: CLARABEL, status `optimal`.
- Risk dual: `6.0339634364437105`.
- Minimum eigenvalue of normalized covariance: `0.03985798478640442`.
- Estimated variance / allowed variance: `1.0000000000339322`.
- Gross leverage: `1.5000000017509367`.
- Minimum historical wealth: `0.9576141804995806`.
- Budget error: `4.440892098500626e-16`.
- Forecast return: `0.0014000147465153781`.
- `unique_optimum=True`; secondary gap zero.
- Repeated independent invocation returned exactly the same weights (maximum difference 0.0).

These values put this particular fixture comfortably within the positive-definiteness and positive-dual conditions, with feasibility discrepancies far below the existing allowed tolerances.

## Earlier findings: resolved

1. **Absolute risk tolerance permitted material small-scale violations (P2).** Replaced the fixed absolute variance slack with a tolerance proportional to the variance limit. The earlier 8.14-times-budget fixture is now rejected; regression passes.
2. **Small but nondegenerate Sharpe training failed from fixed scaling (P2).** Common data-dependent scaling now conditions both primary and secondary problems, while preserving original-unit scores and the `1e-8` degeneracy threshold. Original regression passes. Earlier independent review exercised 120 rescaled cases successfully, with maximum theta difference `9.34e-9` versus their unscaled counterparts.
3. **Frozen test accepted a truncated August sample (P1).** The frozen path now requires exactly the expected 20 dates from August 3–28, 2026. Missing first/interior-date regressions reject those inputs; the complete frozen path succeeds without refitting alpha or theta.

The additional exact-isotropic EW shortcut was also reviewed: under `Sigma=s I`, the unit-budget minimum variance is `s/n`, attained only by EW, so at that cap returning independently checked EW is an analytic unique solution.

## Other reviewed areas

- Forecast features and training labels are strictly prior to prediction; calibration scalers and LASSO fits stay within folds. Independent reconstruction matched all nine calibration MSEs exactly. Post-cutoff perturbation left calibration scores and alpha unchanged.
- Orchestration keeps calibrated alpha and fitted theta fixed in later stages; validates actual mixed returns against realized component-return mixtures; shares risk contexts across forecast windows; and treats historical-mean controls consistently.
- Main/component performance uses daily sample volatility and Sharpe, compounded returns, and turnover against drifted prior holdings. Development validation is labeled as used for selection.
- CLI and export reject nonempty output directories, validate frame/JSON finiteness before writing, preserve frozen decisions, and record source/code hashes plus runtime/package versions.

## Verification

Independently executed in the final pass:

```
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_candidates tests.test_decision tests.test_forecasting tests.test_two_stage tests.test_two_stage_cli -q
```

Result: **32 tests passed**, including the uniqueness regression and complete/partial frozen-test coverage. The root agent separately reported that the complete 98-test suite passed; that broader suite was not rerun during this bounded review. Full real-data run results remain subject to the root agent's completion and artifact checks.
