# Independent direct-LP review

Scope: `portfolio/direct_lp.py`, `portfolio/lp_reference.py`, `run_direct_lp.py`, shared forecast/split/evaluation/export code, and direct-LP tests. LP core implementation was a stub during this first pass and is explicitly excluded until ready.

## Actionable findings (initial)

1. **P1 — Frozen final test does not verify source provenance.** `portfolio/direct_lp.py:193–196` does not include relevant source fingerprints in the frozen payload; `run_frozen_lp` never verifies source hashes before forecasting/solving. Editing the forecast, risk formulation, solver tolerance or runner after development therefore silently changes August decisions while retaining `evaluation_role='frozen_final_test'`. Separate export metadata contains code hashes but `--frozen` does not load/check it. Store targeted source hashes in frozen schema and reject differences before `_generate`.
2. **P2 — Verified reference comparison ignores shared experiment configuration.** `portfolio/lp_reference.py:33–35` compares only validation dates. A reference using a different alpha-calibration cutoff, calibration folds, forecast windows, or development boundaries can pass the same data/code/assets checks and be labeled verified. Verify shared chronology/CV fields and forecast configurations/alphas/risk window where intended to match; legacy-only decision settings need not equal the direct-LP configuration.

## Evidence

- `.venv/bin/python -m unittest discover -s tests -p test_direct_lp_cli.py -v`: 4 tests passed.
- Reviewed forecast construction and historical-mean controls: feature construction excludes target-day returns; forecast/risk histories end before each prediction; development input is cropped before August; frozen evaluation does not invoke calibration.
- Reviewed realized arithmetic: per-day return is dot(weights, actual returns), turnover uses prior weights drifted by realized returns, compounded performance includes initial unit wealth in relative drawdown.
- Deliberate MAD and uncompounded absolute drawdown proxy constraints are not correctness findings; metadata correctly separates them from realized volatility/Sharpe/compounded drawdown.

Pending: finished core LP formulation, independent small-case probes and core tests.

## Follow-up review

Both initial findings have been addressed by root. Frozen payload now carries portable relevant source hashes and numerical package versions, checked before final-test generation. Reference comparison now compares all shared dates/CV fields, mapping development to legacy decision dates, and forecast configurations/selected alphas/risk window. Re-ran CLI/reference suite after the patch: all 4 tests pass, including added mismatch subcases. No open orchestration/CLI/reference findings; core review remains pending.

## Core LP review

- Inspected signed weight bounds, gross absolute-value auxiliaries, centered-return MAD auxiliaries, nonnegative monotone peaks anchored to initial zero, historical wealth floors, and EW-distance secondary LP with preserved objective. Formulation is mathematically consistent with the agreed proxies.
- Ran `.venv/bin/python .lp-work/review_probe.py`: 40 independent random three-asset tests passed against explicit pairwise drawdown constraints; maximum independently computed primary-to-final objective gap `1.0000000306986667e-10`.
- Ran `.venv/bin/python -m unittest discover -s tests -p test_direct_lp.py -v`: all 3 tests passed, including the real frozen roundtrip and source-mismatch rejection.
- **P2 numerical finding sent to implementer:** for `history=np.array([[.02,0],[0,.02]])*1e-9`, `LPConstraints(position_limit=2)`, and `mu=[.02,0]`, EW is feasible but solve raises independent-feasibility failure. Zero-MAD centered equalities are not normalized and their ~1e-11 coefficients are dropped by HiGHS. The check safely rejects the invalid computed weights, but the LP should solve this feasible finite input. Normalize equality rows (and/or scale the zero-MAD formulation) and add regression. No invalid portfolio acceptance found.

## Final reviewed state

Implementer normalized equality rows and added the reproduced tiny zero-MAD fixture after observing its regression failure. Re-ran core tests: all 12 passed. Re-ran independent 40-case pairwise-drawdown probe: all passed with the same maximum objective gap. All three actionable findings are resolved; no open correctness findings in the reviewed code. This is a code/small-case review, not the separate full real-data result audit.

## Final drawdown-conditioning follow-up

Reviewed geometric scaling of peak/cumulative drawdown variables and matching scaled drawdown RHS. It preserves the exact feasible set; the monotone/nonnegative peak constraints stay valid. The tiny positive-drawdown fixture now covers both ordinary and 1e-9-scaled histories with an independently computed cap check. Re-ran all 12 core tests and the independent 40-case pairwise-drawdown probe: all pass. Maximum independent objective gap: `1.0000000220250493e-10`. No open findings.

Reviewed core SHA-256: `72d50f185e95499cf5ccd072c5392fb5c1a00bc845d97857f25e9190496ca95d`.

## Validation-only scope update review

Reviewed latest `run_direct_lp` / CLI / tests after user requested validation-only backtesting. Default generation starts at validation_start, omits the development summary, records zero development evaluation days, and preserves pre-validation history for causal forecast fitting and risk constraints. Explicit include-development mode remains available and is disallowed with frozen-final-test mode. Calibration and frozen configuration remain identical between modes.

Ran `python -m unittest discover -s tests -p 'test_direct_lp*.py' -v`: all 8 tests pass. Independently confirmed frozen payloads match between default/full mode and that only initial validation-day turnover differs because default begins from EW while full mode carries drifted prior holdings; subsequent turnover and all daily holdings/returns agree. This is the expected start-of-backtest convention, not a correctness finding. No open findings on this scope change.
