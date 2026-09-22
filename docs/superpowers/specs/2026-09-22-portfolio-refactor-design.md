# Portfolio architecture and artifact layout

## Approved intent
Reduce duplication and boilerplate, make filenames describe model responsibilities or experiment directions, retain every explored method, and organize output paths. The user approved responsibility-based layering and thin legacy entrypoints. This is a structural refactor; preserve objectives, coefficients, dates, sample weights, CV scoring, solver tolerances and numerical fallback/status policies. Existing historical artifacts stay in place.

## Components
- `portfolio/data`: French CSV loading/preprocessing, chronological history windows and observation/holding weights.
- `portfolio/models`: minimum-variance regression, constrained regularized portfolio fitting, LASSO return forecasting, Elastic Net, rolling PCA and market-beta formulations.
- `portfolio/decisions`: candidate portfolios, trained portfolio mixtures, direct asset linear programming.
- `portfolio/engine`: fitting assembly, parameter search, rolling backtests, forecast/mixture and forecast/LP workflows, checked solver execution.
- `portfolio/evaluation`: metrics, common results, artifact layout/export, provenance and reference verification.
- Root `losses.py`, `constraints.py`, `regularization.py`, `config.py` retain clear configuration roles.
- `experiments`: readable recipes; `tools`: data audit; original root scripts remain forwarding entrypoints.

Keep ordinary functions and small dataclasses. Canonical modules must not depend on compatibility wrappers or on root CLI scripts. Separate coefficient fitting from mapping to holdings. Share mechanics without forcing forecasting, traditional regression and LP into one training abstraction.

## Artifact contract
New `python -m experiments.<direction>` commands default to categorized output under repository-root `results/<direction>/<UTC timestamp>/`; explicit `--output` selects the exact run root. Categories are `metrics`, `returns`, `weights`, `predictions`, `diagnostics`, `config`, `search`, `figures`, `logs`. A manifest maps logical old filenames to relative stored paths. Metadata records layout version and source files recursively. Legacy commands and programmatic export retain flat layout by default; `--layout categorized` opts in. New commands support `--layout flat`. Reject nonempty output directories. Do not relocate existing runs or rewrite their metadata/frozen hashes.

References read both flat and manifest-backed layouts, validate source fingerprints using unique repository-relative paths, and independently reconstruct returns. Existing old-code frozen artifacts remain bound to their historical revision; a source mismatch must fail with guidance, never bypass verification.

## Method preservation
Keep EW, MinVar, LASSO/Ridge fixed and legacy CV, EW shrinkage, recency weights, 126/252/3-year presets, constrained/growth/exposure LASSO, early-stopped reward/exposure search, half-life search, Elastic Net, PCA, market beta, predictive LASSO, mean forecasts, candidate mixtures, Sharpe/return-volatility decisions and direct LP. Preserve original assignment demonstration and audit. Turnover/smoothness/shrinkage report-only entries are indexed as missing runnable code, not silently recreated.

## Validation
Baseline: commit 0934d54, 118 unittest tests passed. Compare deterministic pre/post-refactor numeric fixtures, preserve original API/CLI tests, add meaningful categorized-output/readback/overwrite/provenance checks, smoke every new CLI, and obtain independent review. No full multi-year backtest is needed for a structural refactor. Do not push without a new user request.

## Implementation clarifications
Expected market-beta solver errors still fall back from CLARABEL to SCS; arbitrary programming exceptions now propagate instead of being swallowed by a catch-all. This implements the requested exception cleanup. Previously standalone Elastic Net/PCA/market-beta commands now save new runs under their named results directories when no output path is specified; the market-beta command no longer overwrites a fixed CSV in cwd. Legacy APIs and explicitly requested flat run layouts remain supported.
