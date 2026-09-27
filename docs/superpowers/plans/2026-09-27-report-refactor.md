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
- [ ] Add tests for all report models, coefficient-budget distinction, hard risk/position/beta limits and target-turnover state.
- [ ] Implement portfolio/model_catalog.py and portfolio/models/constrained_portfolio.py; simplify baseline fitting and public exports.
- [ ] Run targeted model/causality tests; preserve saved parity reference cases before cleanup.

### Task 2: Causal report execution and public commands
- [ ] Test LP forecasts/allocation using frozen settings; gross/net and drifted trading accounting; CLI window/period defaults and output protection.
- [ ] Implement portfolio/engine/report_runner.py and shared CLI, run_baselines.py/run_report.py. No dependencies on archived output directories.
- [ ] Migrate forecast freeze into tracked configs, retain chronological calibration function.
- [ ] Run CLI and numerical checks.

### Task 3: Canonical results and removal
- [ ] Build a migration manifest mapping report rows to originals and verify daily holdings/returns before replacing outputs.
- [ ] Keep six baseline results at 126/252; report validation/July, selection evidence and missing-data audit.
- [ ] Remove unrelated implementations, launchers, tests and obsolete docs after verified external archive.
- [ ] Update README, model/results guide and final summary links; add a report reproduction table.

### Task 4: Verification, independent review and integration
- [ ] Run all retained tests, report-source audit and representative old/new numerical parity checks.
- [ ] Independent whole-change review; fix material findings and verify.
- [ ] Commit locally with English message, fast-forward original checkout, migrate cleaned ignored results, verify original checkout commands. Do not push.
