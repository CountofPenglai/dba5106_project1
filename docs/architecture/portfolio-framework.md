# Portfolio framework map

## Read the model first

Two distinct uses of LASSO are intentionally separate:

- `portfolio/models/min_variance_regression.py`: fit the contemporaneous EW-return regression, then map coefficients through `w = EW - N @ beta` to holdings. This is a portfolio variance formulation, not a next-day forecast.
- `portfolio/models/lasso_return_forecast.py`: construct six causal lagged features and predict each asset's next return. The forecasts feed a separate decision method.

Constrained objectives are visible in `portfolio/models/constrained_regularized_portfolio.py`. `portfolio/losses.py`, `regularization.py`, and `constraints.py` define the reward, penalty and feasible region. Solver retries are in `engine/convex_solver.py`; the model retains its independent holdings/variance/wealth checks.

## Responsibilities

| Directory | Responsibility | Main files |
|---|---|---|
| `portfolio/data` | Read raw returns, remove excluded assets, preserve missing values, select history, weight observations | `returns_loader.py`, `preprocessing.py`, `history_windows.py`, `observation_weights.py` |
| `portfolio/models` | Regression, forecasting and exploratory mathematical formulations | `min_variance_regression.py`, `constrained_regularized_portfolio.py`, `lasso_return_forecast.py`, `elastic_net_portfolio.py`, `pca_regularized_portfolio.py`, `market_beta_portfolio.py`, `portfolio_variants.py` |
| `portfolio/decisions` | Turn forecasts into holdings, or adjust existing holdings | `candidate_portfolios.py`, `portfolio_mixture.py`, `linear_program_allocation.py`, `holding_weights.py` |
| `portfolio/engine` | Assemble fitting, iterate causally, search parameters, run decision workflows and solvers | `portfolio_fit.py`, `rolling_backtest.py`, `parameter_search.py`, `growth_search_evaluation.py`, `lasso_window_strategy.py`, `forecast_mixture_workflow.py`, `forecast_lp_workflow.py`, `convex_solver.py` |
| `portfolio/evaluation` | Metrics, result records, artifacts and reproducibility checks | `portfolio_metrics.py`, `experiment_result.py`, `artifact_store.py`, `result_export.py`, `provenance.py`, `reference_comparison.py` |
| `experiments` | Explicit experiment recipes and CLI options | Table below |
| `tools` | Dataset inspection | `audit_missing_values.py` |
| `examples` | Historical assignment source, preserved verbatim | `original_assignment.py` |

`portfolio/config.py` remains the single place for reusable `126d`, `252d`, `3y`, and `3y_weighted` presets. A window experiment changes `--presets`; its fit loop and objective do not change.

## Preserved method inventory

Run canonical recipes as `python -m experiments.<name>`. The old entrypoints remain thin forwarders. Original imports such as `portfolio.training` still resolve to the canonical implementation; new internal code uses canonical paths.

| Direction / preserved methods | Canonical recipe | Legacy command |
|---|---|---|
| EW, MinVar, fixed LASSO/Ridge, legacy CV, exponential observation weighting, shrink-to-EW; optional hard constraints/log growth | `regularized_baselines` | `run_example.py` |
| Fixed LASSO at 126/252 days or three calendar years, with optional weighting/hard constraints | `lasso_training_windows` | `run_lasso_windows.py` |
| Constrained three-year LASSO; validation selects recency half-life | `lasso_recency_weight_search` | `run_weight_tuning.py` |
| Fixed LASSO; reward/exposure coefficient search, coarse screen, validation refinement, batch early stopping and ablations | `lasso_reward_exposure_search` | `run_tuning.py` |
| Elastic Net ratio sweep | `elastic_net_portfolio` | `elastic_net_test.py` |
| Rolling PCA followed by LASSO/Ridge | `pca_regularized_portfolio` | `pca_lasso_ridge_test.py` |
| LASSO/Ridge with historical market-beta bounds | `market_beta_constraints` | `market_beta_cvxpy_test.py` |
| Causal LASSO/mean forecasts, max-return/min-variance/EW candidates, learned Sharpe or return-volatility mixtures | `lasso_forecast_mixture` | `run_two_stage.py` |
| Causal LASSO/mean forecasts, direct asset LP with MAD/drawdown/holding limits and secondary EW tie-break | `lasso_forecast_lp` | `run_direct_lp.py` |
| Turnover, Size-BM smoothness, Ledoit-Wolf shrinkage and concentration-cap sensitivity | `portfolio_sensitivity` | `run_turnover.py`, `run_smoothness.py`, `run_shrinkage.py`, `run_concentration.py` |
| Raw sentinel audit by column and period | `python -m tools.audit_missing_values` | `audit_missing_values.py` |
| Original teaching demonstration (historical CSV path/row offsets/units retained) | `examples/original_assignment.py` | `5106_assignement_01.py` |

The last example is original course material, not an active experiment. Standalone turnover, grid-smoothness and covariance-shrinkage reports/images are retained as historical records. Their new runners share the portfolio-variant model and sensitivity workflow; concentration capping uses the same path. Ledoit-Wolf risk estimation within the two-stage pipeline remains implemented. All runnable method implementations, compatibility entrypoints and 126-day/three-year presets remain available. See [the retained-result inventory](../results-retention.md).

## Conventions deliberately retained

- Baseline fixed penalties keep project strength-to-sklearn scaling. Legacy 5-fold LassoCV uses MSE and RidgeCV uses its existing R-squared default. The forecasting model uses its separate late chronological 3-by-21 CV design.
- PCA's historical Ridge alpha convention is retained, even though it differs from the general regression adapter.
- The four sensitivity runners use the shared loader and reject incomplete training or validation windows; no rows are silently dropped.
- Market-beta results still accept `OPTIMAL_INACCURATE` under the historical policy. Expected CVXPY `SolverError` falls back to SCS; unrelated programming errors are no longer suppressed.
- LP remains direct optimization, without decision-model training, and defaults to validation only. Learned mixtures keep their separate decision-training period. Risk definitions, original test boundaries and frozen-mode checks remain intact.

## Extension points

A traditional strategy is `history -> indexed holdings`; `rolling_backtest.backtest` manages dates and realized returns. `portfolio_fit` assembles the regression, sample weights and result. A new decision algorithm consumes the existing forecasts/risk context and returns holdings; it does not need to copy preprocessing or export code. Shared source inventories recurse into canonical packages, so moving an implementation cannot make it disappear from provenance.
