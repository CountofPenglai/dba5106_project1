# Parameterized LASSO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement parameterized LASSO with growth reward, exposure penalty and bounded chronological validation search.

**Architecture:** Keep the current CVXPY linear portfolio fit. Extend its loss with a nonnegative exposure coefficient and add a small, independent callback-based search module. A new CLI composes historical screening, validation refinement and frozen comparisons.

**Tech Stack:** Python 3.12, NumPy, pandas, CVXPY, unittest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-parameterized-lasso.md`

## Global Constraints

- Model `w=EW-N@beta`; 252-day default; fixed LASSO strength `1e-5`.
- eta>=0, gamma>=0; gamma defaults to zero and old outputs must remain compatible.
- Hard gross limit 1.5, training volatility limit 1*EW, wealth floor 1e-6.
- Coarse eta=(0,.003,.01), gamma=(0,.1,1); bounds eta<=.03 and gamma<=3.
- At most 15 distinct searched configurations; full candidate batches and full evaluation horizons only.
- min_improvement=.002 cumulative-return fraction; patience=1 complete refinement batch.
- Historical screen ends before validation begins. Test data is never used for selection.
- No external publication or remote push. All changes must be reviewable in the original project at completion.

### Task 1: Extend the convex objective

**Files:** modify portfolio/losses.py, portfolio/optimization.py; add tests/test_exposure.py.

**Interfaces:** `LogGrowthLoss(reward_strength=.01, exposure_strength=0)`; existing fit_portfolio_growth signature remains unchanged. Diagnostics gain exposure_unit_penalty, exposure_scale, exposure_penalty. training_loss includes exposure penalty but excludes LASSO; objective includes both.

- [ ] Add failing tests for invalid gamma, gamma=0 equivalence, independent two-asset scalar-grid optimum with gamma>0, lower gross under higher gamma, and gross_limit=1 with no divide by zero.
- [ ] Run `.venv/bin/python -m unittest discover -s tests -p test_exposure.py -v` and verify missing behavior fails.
- [ ] Add finite/nonnegative exposure_strength validation. Implement `gamma * ew_variance * square(pos((norm1(w)-1)/(limit-1)))` for limit>1, with zero term at limit=1. Keep gamma=0 objective unchanged. Independently recompute diagnostic penalty from returned weights, add to loss, preserve feasibility checks.
- [ ] Run exposure and growth tests. Report proof of scalar optimum, default compatibility and test results. Parent integrates; do not commit shared work.

### Task 2: Bounded search and early stopping

**Files:** create portfolio/selection.py and tests/test_selection.py.

**Interfaces:** frozen `GrowthParameters(reward_strength: float, exposure_strength: float)`; `SearchConfig(max_candidates=15, min_improvement=.002, patience=1)`; `search_growth(evaluate, *, config=SearchConfig(), initial_parameters=None)` returns a result with selected parameters, raw-best parameters, history DataFrame, stop_reason and completed round count. Callback maps GrowthParameters to a metrics dict containing cumulative_return and average_gross_leverage; other metrics are preserved. Constants expose COARSE_PARAMETERS (nine pairs). No dependency on CVXPY.

- [ ] Write failing tests with an explicit tabulated callback landscape for complete initial batch, plateau stopping after a complete neighbor batch, no partial budget batch, cumulative significant improvements, deduplication, conservative tie selection, best restoration and callback failure records. Include nonfinite/invalid metric handling and configuration validation.
- [ ] Execute tests and verify missing feature failures.
- [ ] Implement deterministic two-coordinate search per spec. Each candidate has phase-local id, round, eta, gamma, status, error and returned metrics. Catch numerical RuntimeError/ValueError as failed candidate; if all coarse evaluations fail, raise. Unexpected programming errors propagate. Never compare differently weighted training objectives for selection.
- [ ] Verify tests and report specification/quality concerns. Parent integrates; do not commit shared work.

### Task 3: Experiment runner and artifacts (parent)

**Files:** create run_tuning.py and tests/test_tuning_cli.py; update portfolio/__init__.py and README.md.

**Interfaces:** consumes the loss and search APIs above. CLI options csv, window, screen-start/end, validation-start/end, output; defaults use spec dates. Fixed experiment settings are serialized, with early-stop min-improvement and budget configurable within the global experiment budget. CLI test uses a synthetic two-table Ken French CSV.

- [ ] Write a failing end-to-end CLI test using a small chronological dataset and shorter explicit window. Verify metadata selection, complete phase dates, rolling weights/returns and rejected reused output directory. Include a future-data perturbation regression proving that selection ignores data after validation_end.
- [ ] Run it to verify the entry point is missing, then implement callbacks which crop inputs to phase end, fit every historical window and save actual diagnostics, returns and weights. Validate phase order and coverage before output creation.
- [ ] Run the nine-point historical screen with max_candidates=9. Seed validation from controls and historical-selected ablations. Set validation budget to initial count+6, never over 15. Freeze parameters before the descriptive baseline/ablation comparisons.
- [ ] Write search CSVs, comparison summary, run metadata with data/code hashes, selected/raw-best parameters, search settings, phase evaluation counts, unique configuration count and test availability. No final August evaluation on partial data.
- [ ] Run integration tests and full suite, then actual 98-asset experiment. Independently recompute selected metrics and checks from saved weights and raw returns.

### Task 4: Independent review, report and integration

**Files:** docs/experiments/parameterized-lasso-252.md plus fixes justified by review.

- [ ] Independent reviewers inspect math/compatibility and search/data leakage/CLI. Fix actionable findings with regression tests.
- [ ] Document selected parameters, stop reasons, candidate counts and seven-way comparison. Explicitly distinguish development selection from final testing and search stopping from training early stopping.
- [ ] Run full tests and git diff --check after all changes.
- [ ] Transfer reviewed changes and experiment artifacts into the original project after checking original tracked files still equal the starting commit. Preserve the user-owned PDF. Leave changes uncommitted for review, and report tests, results and limitations.
