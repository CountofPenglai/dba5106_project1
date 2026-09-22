# Portfolio Refactor Implementation Plan

> **For agentic workers:** Use task-scoped implementation and independent review. This plan follows the user-approved architecture; routine details are resolved without another approval cycle.

**Goal:** Make model architecture readable, remove duplicate orchestration, and standardize artifact locations while preserving experiments.

**Architecture:** Canonical responsibility packages plus thin compatibility forwarding modules. Explicit experiment recipes share fitting, backtesting, metrics, solver and artifact infrastructure.

**Tech Stack:** Python, numpy, pandas, sklearn, cvxpy, scipy; no new dependencies.

**Spec:** docs/superpowers/specs/2026-09-22-portfolio-refactor-design.md

## Global Constraints
Preserve objectives, coefficients, dates, sample weights, CV scoring, solver tolerances and failure policies. Existing historical artifacts stay in place. No push. Legacy flat outputs remain supported. Model code imports canonical modules only.

## Review Focus
- Import aliases must preserve patchability and public imports while data/evaluation change from modules to packages.
- New output layouts must round-trip through reference readers and reject nonempty destinations before expensive work.
- Nested source modules must all be fingerprinted; old frozen hashes cannot be silently accepted.
- Independent experiments retain their historical missing-row handling, normalization and metric labels.
- Time windows and constrained solutions must produce the same numeric results after extraction.

## Task 1: Characterize and relocate core
- [x] Capture deterministic weights, CV alphas, constrained results and forecast/decision fixtures at the baseline.
- [x] Move model/data/decision/engine code into explicit packages, rewriting imports to canonical locations.
- [x] Keep legacy aliases; combine fixed/CV fit assembly and separate shared results/metrics from mixture workflow.
- [x] Run all original tests and compare captured numeric fixtures.

## Task 2: Artifact storage and experiment recipes
- [x] Add tests for categorized paths, flat compatibility, rejection of unsafe names/nonempty roots, default root independence, manifest lookup and nested code hashes.
- [x] Implement `RunArtifacts(root, layout='flat')`, `artifact_path(root, name)`, `default_output(direction)`, `ensure_fresh_output(root)` and recursive source inventory.
- [x] Move root CLI implementations to descriptive experiment modules; forwarding scripts call `main()` (flat), module entrypoints call `main(default_layout='categorized')`.
- [x] Update CSV/JSON writing to the artifact store; preserve old output names as logical names; index metadata and frozen decisions under config.
- [x] Update references to read both layouts and verify canonical source paths. Test both paths.

## Task 3: Independent exploratory models
- [x] Move Elastic Net, rolling PCA and market-beta math into explicitly named model files, with experiment recipes that use common loaders, backtests and metrics.
- [x] Preserve legacy complete-row handling and historical sklearn strength conventions; market beta's legacy solver status policy stays explicit.
- [x] Add synthetic numerical equivalence tests, argparse help, `--csv`, `--output`, `--layout` and categorized exports; retain legacy root script names.

## Task 4: Documentation and final review
- [x] Write an architecture/method map and output reference; update README with new canonical commands and compatibility policy.
- [x] Run full suite, new CLI help/synthetic smoke checks, deterministic parity checks and diff hygiene.
- [x] Dispatch independent review covering behavior preservation, output layouts and provenance. Address substantive findings and rerun affected checks.
- [x] Bring reviewed changes into the user's original checkout without touching the untracked PDF or historical outputs; do not push.

## Execution record
- Baseline suite: 118 passed. Expanded suite: 138 passed in 42.605 s.
- Synthetic equivalence: 69 numeric groups, maximum absolute difference 0; real-data equivalence: six 98-asset fits, maximum absolute difference 0.
- Independent exploratory and whole-framework reviews approved; five artifact-routing checks passed after the category correction.
- Ruling: preserve numerical fallback/status behavior while allowing unexpected programming errors to propagate; documented in the spec.
- Ruling: keep existing result directories untouched and organize future output, preserving explicit flat-layout calls.
- Ruling: apply reviewed changes as uncommitted changes to the original checkout; the user requested a refactor, not a commit or push.
