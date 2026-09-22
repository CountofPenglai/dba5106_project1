# Two-stage LASSO Portfolio Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans task by task. Track verification evidence in the task ledger.

**Goal:** Implement and evaluate independently trained return forecasters and fixed, interpretable portfolio mixtures under Sharpe and mean-minus-volatility objectives.

**Architecture:** Six lagged features feed per-asset LASSO models with shared alpha calibrated before decision training. A common risk model produces feasible return-seeking, minimum-risk and EW portfolios. A three-component simplex model learns fixed mixture coefficients from their historical out-of-sample returns.

**Tech Stack:** Existing Python, NumPy, pandas, scikit-learn, CVXPY; unittest. No added dependencies.

**Spec:** ../specs/2026-09-22-two-stage-lasso-decision-design.md

## Global Constraints

- Preserve all existing files and 126/252/3y entry-point behavior.
- History begins 2021-08-01. Calibration ends 2024-07-31. Decision training: 2024-08-01..2025-12-31. Validation: 2026-01-02..2026-06-30. Reserved final test: 2026-08-01..2026-08-30.
- Equal observation weights; alpha grid logspace(-6,-2,21), last 3x21 chronological folds. Shared alpha per forecast window, frozen after calibration.
- Fixed decision weights; gamma grid [0,.025,.05,.1,.2], selected by validation cumulative return with larger-gamma tie break.
- Gross <=1.5, estimated volatility <=EW, historical wealth floor 1e-6; same risk model across forecasts.
- Daily Sharpe, sample std ddof=1, rf=0, zero transaction costs, turnover uses drifted positions.
- No product code changes in the original checkout until independently verified additions are ready. No remote push.

## Review Focus

1. A future data edit must not affect earlier forecasts, alpha, candidate weights or learned theta: causal feature and runner perturbation tests.
2. Tiny/zero risk and all-negative component means must produce an explicit, correct Sharpe outcome: mixture tests against independent simplex enumeration.
3. Repeated candidate portfolios and near-active leverage constraints must remain deterministic and feasible: tie-break, convex-mixture and solver diagnostic tests.
4. The terminal reserved test must never be evaluated by a development run, including a CSV with future rows: CLI guard and frozen-artifact checks.
5. Validation must never refit theta: repeat fits with identical decision training but changed validation, compare every gamma theta and Sharpe theta.

## Interfaces

- `portfolio/forecasting.py`: `ForecastData(features: ndarray[n,p,6], targets: DataFrame)`; `build_supervised(returns) -> ForecastData`; `calibrate_alpha(data, config, *, end, alphas=None, n_splits=3, test_size=21) -> (float, DataFrame)`; `forecast_at(data, date, config, alpha) -> (Series, dict)`.
- `portfolio/candidates.py`: `RiskContext` stores covariance, factor, history, constraints and EW; `estimate_risk(history, constraints=PortfolioConstraints())`; `candidate_portfolio(context, *, objective, expected_returns=None) -> (ndarray, dict)`; `check_candidate(context, weights) -> dict`.
- `portfolio/decision.py`: `fit_mixture(component_returns, *, objective, gamma=0) -> MixtureFit(theta, diagnostics)`; `mix_weights(candidate_weights, theta)`; shared score helper reports the true daily objectives.
- `portfolio/two_stage.py`: experiment config, chronological generation and frozen decision evaluation; reusable function returning dataframes/dicts, independent of CLI.
- `run_two_stage.py`: guarded development CLI, raw data loading, artifacts and progress output.

## Task 1: Causal features and calibrated LASSO forecasts

Files: forecasting.py and tests/test_forecasting.py. Implementer owns these only.
- [x] Write tests for exact lag/compound features, constant feature scaling, 252/calendar training boundaries and future invariance.
- [x] Run tests and observe missing implementation failure.
- [x] Implement six features and per-asset StandardScaler plus Lasso(alpha), with ConvergenceWarning treated as error.
- [x] Implement 3x21 fold-local preprocessing and equal MSE aggregation. Ties within atol=1e-12, rtol=1e-8 choose larger alpha.
- [x] Test fold diagnostics and recompute predictions independently from sklearn on a small fixture; review.

## Task 2: Feasible candidate portfolios and fixed mixture fitting

Files: candidates.py, decision.py; tests/test_candidates.py, tests/test_decision.py.
- [x] Write analytic two-asset feasibility tests and three-component simplex-enumeration objective tests; observe red.
- [x] Fit LedoitWolf; solve max-return/min-risk with common constraints; use solver scaling and deterministic EW secondary tie break.
- [x] Independently check gross, budget, risk and wealth; record solve status, attempts and objective gaps; no hidden EW fallback.
- [x] Fit mean-minus-standard-deviation on the simplex. Fit positive-mean Sharpe by convex transformation; handle nonpositive means using the best vertex; reject degenerate feasible risk.
- [x] Test repeated components, gamma=0, zero risk, negative means, invalid input and convex-mixture feasibility; review.

## Task 3: Chronological experiment and guarded CLI

Files: two_stage.py, run_two_stage.py; tests/test_two_stage.py.
- [x] Write a small end-to-end fixture testing calibration/train/validation separation, frozen theta and future test-row exclusion; observe red.
- [x] Generate daily forecasts and historical-mean controls, shared risk/min-risk/EW candidates, and real candidate returns.
- [x] Train both objectives only on decision dates; select gamma only on validation and retain all candidate theta/results.
- [x] Export predictions, alpha scores, candidates, theta, validation gamma tables, portfolio weights/returns, metadata, constraints, reconstruction and drifted-turnover diagnostics.
- [x] Support a separate frozen-config final-test evaluation path; development mode rejects final-test dates and incomplete periods.
- [x] Validate against existing 252/3y baseline outputs only when dataset hash and date coverage match; otherwise compute baselines via existing APIs.
- [x] Full suite and independent review.

## Task 4: Real-data evaluation and delivery

Files: README.md appended usage, docs/experiments/two-stage-lasso.md and plot; new results directory (ignored).
- [x] Run the full calibration + 356-day decision training + 123-day validation for four main experiments and historical-mean controls.
- [x] Independently reconstruct daily realized returns, objective values, constraints, frozen theta and selected gamma; record exact tolerances and failures.
- [x] Compare four models, components, EW and existing 252/3y constrained/unconstrained baselines on identical validation dates.
- [x] Write report without claiming independent final-test performance; include prediction quality and mixture weights so signal contribution is interpretable.
- [x] Review all code, address findings, run relevant checks, then copy only verified new/changed task files back without overwriting earlier user work.

## Execution ruling

The user explicitly said “实施当前计划” after reviewing the design. Implementation is authorized; routine implementation details are resolved here without another confirmation loop. The original checkout has necessary uncommitted window modules. The isolated checkout includes an exact copy of that working state; a manifest protects it during final integration. Root implements optimization/orchestration while one bounded forecasting implementer works on disjoint files; independent reviews follow. No simultaneous implementation subagents are dispatched.
