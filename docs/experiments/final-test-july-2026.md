# July 2026 Final Test: Preselected 252-Day Portfolio Models

Run date: 2026-09-23. The user designated **2026-07-01 to 2026-07-31** as the final test because August observations are not yet available in the project dataset. The test contains **22 trading days and 98 retained Size-BM portfolios**. This is the only post-validation period in the current file that had not been used to select or compare model performance. July had previously been checked for data completeness, but not used for performance ranking. Results below are a one-time evaluation of configurations chosen from the January–June validation results; no settings were changed after seeing July performance.

## Frozen comparison

The six rows were fixed before the July run: EW is the benchmark; `LASSO_WeightCap_10pct` had the highest validation daily Sharpe; `Ridge_conservative_0.5_0.9` was a distinct high-Sharpe Ridge method; `LASSO_market_like_0.8_1.2` combined relatively high validation return with higher Sharpe than EW; `LP_LASSO_252d` came closest to EW's validation cumulative return; and `LASSO_Selected` is the direct uncapped control for the 10% cap. Each is an existing configuration from the [unified validation summary](final-model-summary.md). The validation period contains 123 trading days; the final test contains 22, so their cumulative returns are not horizon-matched.

| Model | Validation return | Validation daily Sharpe | July final return | July daily Sharpe | July daily vol. | July max drawdown | Avg. gross | Avg. turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EW | 16.933% | 0.13253 | -0.841% | -0.05186 | 0.6952% | -2.135% | 1.000 | 0.33% |
| LASSO_Selected | 11.248% | 0.14723 | **2.017%** | **0.16073** | 0.5748% | -1.296% | 1.016 | 5.27% |
| LASSO_WeightCap_10pct | 12.702% | 0.15611 | 1.667% | 0.15531 | **0.4915%** | **-1.103%** | 1.016 | 4.36% |
| Ridge_conservative_0.5_0.9 | 11.001% | 0.15110 | 1.289% | 0.10612 | 0.5629% | -1.331% | 1.500 | 6.26% |
| LASSO_market_like_0.8_1.2 | 14.028% | 0.13901 | 0.451% | 0.03451 | 0.6508% | -1.549% | 1.236 | 2.42% |
| LP_LASSO_252d | 16.719% | 0.12841 | 0.200% | 0.01629 | 0.7043% | -1.926% | 1.500 | 12.71% |

Cumulative return is compounded. Daily Sharpe is mean daily return divided by the sample daily standard deviation (`ddof=1`), with zero risk-free rate and no annualization. Maximum drawdown includes initial wealth of 1 and is shown as a negative percentage. Average gross is the mean sum of absolute portfolio weights. Turnover is half the absolute change from **drifted** previous holdings; its first day starts from EW. Results exclude trading costs and market impact.

## What the final test shows

All five selected alternatives gained during July while EW returned **-0.841%**. The uncapped `LASSO_Selected` led on both cumulative return (**2.017%**) and daily Sharpe (**0.16073**). The capped variant was close in Sharpe (**0.15531**) and had the lowest volatility and shallowest drawdown, but did **not** retain its validation lead over the uncapped model: its July return was **1.667%**. The cap therefore looks useful for controlling concentration and risk here, but its validation Sharpe advantage did not replicate in this short test.

The conservative-beta Ridge remained positive but its daily Sharpe fell from 0.15110 in validation to 0.10612. The market-like-beta LASSO and direct LP also remained positive, yet their July daily Sharpe was only 0.03451 and 0.01629 respectively. In particular, LP's validation return near EW did not persist as a strong July return; its **12.71% average daily turnover** is the highest of these candidates before any trading costs. None of these test observations was used to pick a replacement configuration.

This final test has only 22 observations. Its Sharpe ratios and model ordering have substantial sampling uncertainty, and the different validation/test horizons make compounded returns unsuitable for direct percentage comparisons. The results describe this July 2026 period; they do not prove a persistent ranking.

## Execution and verification

All five fitted methods used 252 prior trading observations for each held day. On July 1 their histories ended June 30; as July progressed, earlier July returns entered later rolling fits, never the same day's fit. The LP used the previously selected forecast alpha **0.002511886431509582**, 252 supervised training targets, 252 risk observations, and unchanged LP limits. Its pre-July frozen data digest and every source file that existed in the LP freeze matched the retained artifact. The LP's original `run_frozen_lp` entry point is hard-coded to August, so this July run used its existing forecast and optimization functions with those frozen settings; the frozen artifact itself was not edited.

All six saved return streams were recomputed independently from their daily weights and the local source CSV. Maximum daily reconstruction error was **9.85e-17** and maximum core-metric difference was **8.33e-15**. Every saved window contained 252 dates strictly before its holding date; the cap stayed at or below 10%; beta bounds held; and all 22 LP solves reported optimal and feasible. Recomputing all five non-EW models on June 30 reproduced their saved validation weights to within **9.89e-17** maximum elementwise difference, confirming the test runner used the same configurations.

[Machine-readable validation and final-test comparison](final-test-july-2026.csv) contains unrounded values. The local detailed artifact directory is `results/final_test_july_2026_20260923/`, with weights, returns, fit diagnostics, frozen selection, run metadata, and the exact one-shot runner. The source file is the project's `100_Portfolios_10x10_Daily.csv` (SHA-256 `11399a830c7b2a192d652b39e866a74c05d7380dd6edf1573ce1c40f9d271c23`); it ends on July 31, 2026. The former August target is unavailable and is not part of this final-test report.
