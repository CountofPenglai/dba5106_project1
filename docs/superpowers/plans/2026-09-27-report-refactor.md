# Report-aligned portfolio refactor implementation plan

> Execution: superpowers:executing-plans; approved by the user on 2026-09-27.

**Goal:** Keep six baselines and every model/configuration in the final report, remove unrelated research and obsolete results, and make each retained result traceable to executable code.

**Architecture:** One named model catalog; shared coefficient regression and constrained-portfolio fitting; separate LASSO forecasting and LP allocation; one causal rolling evaluation and artifact schema. Two public commands: run_baselines.py and run_report.py.

**Tech stack:** Existing Python, NumPy, pandas, scikit-learn, CVXPY and SciPy dependencies.

**Spec:** User-approved discussion and DBA5106_Final_Report_Draft 2.docx. Preserve report comparisons, fixed parameters, 126/252-day baseline support, gross performance metrics, and original saved numerical evidence. No new tuning or model selection using July. Archive removed research outside the project.

## Global constraints

- Six baselines: EW, MinVar, LASSO/Ridge coefficient budgets (1 each), LASSO/Ridge penalties (1e-5/1e-4).
- 252-day report history; Jan 2–Jun 30 validation; July 1–31 evaluation. All training windows exclude the current day.
- Retain hard constraints; exposure gamma=2, eta=0; turnover tau=.1/1 plus the shared tau=0 control; 10% weight cap; six beta configurations; original and cost-only LP (.0005).
- Keep LP's chronological 3x21 forecast calibration and frozen alpha 0.002511886431509582. Preserve EW tie-break and feasibility checks.
- Historical hashes remain unchanged; transformed result tables cite their original files and checksums.
- Preserve distinct target turnover and drift-adjusted trade turnover. Gross returns are the main reported stream; net returns are supplementary.
- Existing main checkout stays usable until validation; no remote push is part of this change.

## Review focus

- Hidden dependency on deleted result directories or untracked July scripts.
- Leakage from current/future return rows or selecting configuration on July.
- Confusing coefficient budgets, portfolio constraints, and penalty scales.
- Numeric changes in market-beta solver formulation; original report records must not be overwritten.
- Migration completeness, gross/net semantics, and loss of original data/source fingerprints.

## Tasks

### Task 1: Model catalog and shared constrained fitting
- [x] Add tests for all report models, coefficient-budget distinction, hard risk/position/beta limits and target-turnover state.
- [x] Implement portfolio/model_catalog.py and portfolio/models/constrained_portfolio.py; simplify baseline fitting and public exports.
- [x] Run targeted model/causality tests; preserve saved parity reference cases before cleanup.

### Task 2: Causal report execution and public commands
- [x] Test LP forecasts/allocation using frozen settings; gross/net and drifted trading accounting; CLI window/period defaults and output protection.
- [x] Implement portfolio/engine/report_runner.py and shared CLI, run_baselines.py/run_report.py. No dependencies on archived output directories.
- [x] Migrate forecast freeze into tracked configs, retain chronological calibration function.
- [x] Run CLI and numerical checks.

### Task 3: Canonical results and removal
- [x] Build a migration manifest mapping report rows to originals and verify daily holdings/returns before replacing outputs.
- [x] Keep six baseline results at 126/252; report validation/July, selection evidence and missing-data audit.
- [x] Remove unrelated implementations, launchers, tests and obsolete docs after verified external archive.
- [x] Update README, model/results guide and final summary links; add a report reproduction table.

### Task 4: Verification, independent review and integration
- [x] Run all retained tests, report-source audit and representative old/new numerical parity checks.
- [x] Independent whole-change review; fix material findings and verify.
- [x] Commit locally with English message, fast-forward original checkout, migrate cleaned ignored results, verify original checkout commands. Do not push.

## Completion record — September 27, 2026

Implemented in commit `b4c1901` and integrated into the original local `main`
checkout. No remote push was performed for this refactor.

- Production Python files: 98 to 37; lines: 6,146 to 2,168.
- Retained: six baselines, both 126/252-day baseline records, 19 report
  configurations, seven previously evaluated July portfolios.
- Original-checkout verification: 75 tests passed; 32 result records and
  159 artifact hashes passed; every saved gross return stream reconstructed
  from holdings and the original dataset.
- Numerical compatibility: 96 representative daily refits; maximum absolute
  holding difference 6.04e-8 and daily return difference 1.43e-9. See
  [the detailed verification record](../../architecture/refactor-verification.json).
- Independent read-only review found no actionable issues. It checked report
  coverage, causal execution, turnover/cost definitions, nine focused tests,
  all saved return streams and artifact hashes, all 123 migration source
  hashes against backup commit `c0b1cc5`, and both LP data fingerprints.
- No new parameter selection or full historical backtest was performed.

### Implementation decisions

- The native worktree tool could not locate the repository from the chat's
  working directory, so a manual isolated worktree was used. The original
  checkout remained unchanged until final integration.
- Hard LASSO and unbounded-beta LASSO share a fitter but retain distinct IDs
  and their historical numerical formulations. Their mathematics agree, but
  solver defaults produce slightly different saved results. Beta feasibility
  tolerances remain explicit; changing them would require numerical revalidation.
- The old LP field named `development_data_digest` actually includes data
  through June 30, 2026. It remains unchanged as historical evidence. The new
  frozen config adds a calibration-only fingerprint for causal reproduction.
- Removed experimental entrypoints intentionally lose compatibility; their
  code and results remain recoverable from the complete backup. No report
  method or baseline was removed.
- Historical selection quality was not reassessed. July was already observed,
  and the later LP's development history is disclosed in the results guide.
- Local integration and an English checkpoint commit implement the approved
  plan; remote publication remains a separate action.

The full snapshot is on `backup/pre-report-refactor-20260927`. Verified external
archives are under `../.cleanup-backups/project1/20260927-report-refactor/`:
`complete-pre-refactor.tar.gz`, `original-results/` (764 original result files),
and `refactor-process/` (migration, review and verification working records).
The user's raw dataset, local environment and untracked assignment PDF were
preserved. The isolated worktree is retained for reuse.
