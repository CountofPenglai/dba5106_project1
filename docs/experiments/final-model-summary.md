# Portfolio Model Validation and Final-Test Summary

Updated: 2026-09-23. The main text shows one representative per modeling idea; the appendix preserves every result from both unified 252-day validation rounds. The one-time July 2026 final test is summarized below and detailed in the [final-test report](final-test-july-2026.md).

## Evaluation basis and consolidation rule

- Both rounds used the same 98 retained Size-BM portfolios and 2026-01-02 to 2026-06-30 validation interval (123 trading days). Rolling model, forecast and risk histories used 252 prior trading observations; the portfolio-mixture decision layer used a separate earlier 252-observation training period.
- Cumulative return is compounded. Daily Sharpe is mean daily return divided by sample daily standard deviation (`ddof=1`), without annualization or a risk-free rate. Max drawdown includes initial wealth 1. Reported performance excludes transaction costs.
- **Within each modeling idea, the representative is the configuration with the highest daily Sharpe**, the course evaluation metric. Exact or near-exact aliases use the clearer configuration name. The number in parentheses after each idea is the number of result rows consolidated, including repeated controls where applicable.
- Fixed LASSO and Ridge, hard constraints, reward/exposure changes, and the two mixture training objectives remain separate ideas. The LASSO and Ridge beta-constrained models also remain separate because their portfolio behavior differs. Candidate-only mixture U/V components are shown in the appendix, not treated as full decision methods.
- The validation period has already informed model development. Choosing a winner from these rows is descriptive and introduces selection bias. The user designated July 2026 as the final test after the six candidates were selected; July outcomes were not used to retune them.

Source records: [Round 1 report](unified-252-2026-09-22.md), [Round 2 report](unified-252-sensitivity-2026-09-23.md), and [complete 71-row CSV](unified-252-expanded-2026-09-23.csv).

## Representative results — Round 1

Round 1 contained 65 rows across eight experiment families. EW had the highest cumulative return (**16.933%**). Conservative-beta Ridge had the highest daily Sharpe in that round (**0.15110**), while direct LP using LASSO forecasts came closest to EW in cumulative return (**16.719%**). The selected exposure-penalized LASSO used zero return reward, so its gain cannot be attributed to the reward term.

| Modeling idea | Representative configuration | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---|---:|---:|---:|---:|---:|
| EW benchmark (5) | EW | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| Unregularized MinVar (1) | MinVar | 2.936% | 0.03677 | 0.7074% | -4.612% | 8.921 |
| Fixed LASSO (2) | LASSO | 6.979% | 0.09951 | 0.5673% | -4.828% | 3.036 |
| Fixed Ridge (1) | Ridge | 5.104% | 0.07778 | 0.5389% | -4.933% | 3.125 |
| CV-selected regularization (2) | LASSO_CV | 4.737% | 0.06589 | 0.5981% | -4.860% | 3.800 |
| Recency + EW overlay (3) | LASSO_CV_Weighted | 6.476% | 0.09629 | 0.5452% | -4.892% | 3.177 |
| Hard-constrained LASSO (3) | LASSO_Constrained | 8.804% | 0.12233 | 0.5743% | -4.400% | 1.500 |
| Log-growth reward (2) | LASSO_LogGrowth | 8.073% | 0.11330 | 0.5716% | -4.512% | 1.500 |
| Soft exposure penalty (2) | LASSO_Selected † | 11.248% | 0.14723 | 0.6010% | -4.191% | 1.015 |
| Recency-weighted LASSO (5) | LASSO_252d_HL504_Constrained † | 8.508% | 0.11976 | 0.5678% | -4.598% | 1.500 |
| Elastic Net (11) | ElasticNet_l1ratio_0.95 | 7.012% | 0.10011 | 0.5664% | -4.811% | 3.069 |
| Rolling PCA (12) | Ridge_PCA_15 | 6.081% | 0.08645 | 0.5742% | -4.931% | 3.607 |
| Market-beta LASSO (3) | LASSO_market_like_0.8_1.2 | 14.028% | 0.13901 | 0.7904% | -6.435% | 1.328 |
| Market-beta Ridge (3) | Ridge_conservative_0.5_0.9 | 11.001% | 0.15110 | 0.5725% | -5.429% | 1.500 |
| Sharpe-trained mixture (2) | Mean_252d_Sharpe | 9.167% | 0.12817 | 0.5690% | -4.803% | 1.500 |
| Return-volatility mixture (2) | LASSO_252d_ReturnVolatility † | 9.167% | 0.12818 | 0.5690% | -4.804% | 1.500 |
| Direct asset-level LP (2) | LP_LASSO_252d | 16.719% | 0.12841 | 1.0195% | -8.820% | 1.500 |

† identifies a validation-selected configuration or a descriptive equivalent of one. The recency row names the explicit HL504 configuration that produced the same holdings as its selected alias. The mean- and LASSO-forecast Sharpe mixtures are numerically indistinguishable; their tiny tie-break difference is not evidence that one forecast is better. The EW-like E components and the U/V mixture controls are omitted from the representative table but remain in Appendix A.

## Representative results — Round 2

Round 2 added six non-control variants under the same validation dates and 252-day window. The four studies changed only turnover penalty, Size-BM grid smoothness, covariance estimator, or a 10% absolute asset-weight cap. Their EW and selected-LASSO controls exactly matched Round 1 holdings.

| Modeling idea | Representative configuration | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---|---:|---:|---:|---:|---:|
| Turnover penalty (2) | LASSO_Turnover_0p1 | 10.380% | 0.13245 | 0.6209% | -5.072% | 1.014 |
| Size-BM grid smoothness (2) | LASSO_Smooth_0p1 | 11.275% | 0.14751 | 0.6012% | -4.191% | 1.015 |
| Covariance shrinkage (1) | LASSO_LedoitWolf | 11.003% | 0.14317 | 0.6057% | -4.376% | 1.015 |
| Single-asset weight cap (1) | LASSO_WeightCap_10pct | 12.702% | 0.15611 | 0.6359% | -5.494% | 1.016 |

The 10% cap achieved the highest daily Sharpe of all 71 rows (**0.15611**) and improved cumulative return over the uncapped selected LASSO (**12.702% versus 11.248%**), but its maximum drawdown was deeper (**-5.494% versus -4.191%**). Grid smoothing at `rho=0.1` changed performance only marginally. Ledoit-Wolf covariance shrinkage and the turnover penalties did not improve Sharpe over the selected LASSO.

## Representative models close to or ahead of EW

No distinct model exceeded EW’s **16.933% cumulative return**. The following screens apply to the 21 representatives above, excluding duplicated EW controls. They are presentation rules, not statistical tests or new training rules.

### Close to EW in cumulative return

Here, “close” means within **3.00 percentage points** of EW’s cumulative return (at least **13.933%**).

| Representative model | Cumulative return | Gap vs EW | Daily Sharpe | Daily volatility | Max drawdown |
|---|---:|---:|---:|---:|---:|
| LP_LASSO_252d | 16.719% | -0.214 pp | 0.12841 | 1.0195% | -8.820% |
| LASSO_market_like_0.8_1.2 | 14.028% | -2.905 pp | 0.13901 | 0.7904% | -6.435% |

`LP_LASSO_252d` was only 0.214 percentage points behind EW in return, but its Sharpe, volatility and drawdown were slightly worse. `LASSO_market_like_0.8_1.2` had a larger return gap but improved on EW’s Sharpe, volatility and drawdown. The mean-forecast LP is in the full appendix: it was close in return but had lower Sharpe than the LASSO-forecast LP, so only the latter represents this idea in the body.

### Better risk-adjusted validation performance than EW

This screen requires **daily Sharpe > EW**, **daily volatility < EW**, and **a shallower max drawdown than EW**. Every model listed here still had lower cumulative return than EW.

| Representative model | Cumulative return | Gap vs EW | Daily Sharpe | Daily volatility | Max drawdown |
|---|---:|---:|---:|---:|---:|
| LASSO_WeightCap_10pct | 12.702% | -4.231 pp | 0.15611 | 0.6359% | -5.494% |
| Ridge_conservative_0.5_0.9 | 11.001% | -5.932 pp | 0.15110 | 0.5725% | -5.429% |
| LASSO_Smooth_0p1 | 11.275% | -5.658 pp | 0.14751 | 0.6012% | -4.191% |
| LASSO_Selected † | 11.248% | -5.685 pp | 0.14723 | 0.6010% | -4.191% |
| LASSO_LedoitWolf | 11.003% | -5.930 pp | 0.14317 | 0.6057% | -4.376% |
| LASSO_market_like_0.8_1.2 | 14.028% | -2.905 pp | 0.13901 | 0.7904% | -6.435% |

The capped LASSO leads on daily Sharpe. The selected exposure-penalized LASSO and smoothed LASSO have shallower drawdowns, while market-like beta LASSO is the only representative that also lies within three percentage points of EW return. These trade-offs should be stated before a final strategy is chosen.

## July 2026 final test (22 trading days)

Six existing configurations were fixed from the validation results before the July test. The table below reports the one-time result; the [full report](final-test-july-2026.md) and [unrounded CSV](final-test-july-2026.csv) include the selection rule, risk measures, turnover, saved artifacts and verification.

| Preselected model | July cumulative return | July daily Sharpe | July daily volatility | July max drawdown |
|---|---:|---:|---:|---:|
| EW | -0.841% | -0.05186 | 0.6952% | -2.135% |
| LASSO_Selected | **2.017%** | **0.16073** | 0.5748% | -1.296% |
| LASSO_WeightCap_10pct | 1.667% | 0.15531 | **0.4915%** | **-1.103%** |
| Ridge_conservative_0.5_0.9 | 1.289% | 0.10612 | 0.5629% | -1.331% |
| LASSO_market_like_0.8_1.2 | 0.451% | 0.03451 | 0.6508% | -1.549% |
| LP_LASSO_252d | 0.200% | 0.01629 | 0.7043% | -1.926% |

The uncapped selected LASSO led on July return and Sharpe; the 10% cap produced the lowest volatility and shallowest drawdown but did not retain its validation Sharpe lead. July contains only 22 observations, so this ranking is descriptive, and it was not used for a further model search. The former August target has no records in the current CSV.

## Appendix A — complete results from both rounds

The tables below preserve all **65 Round 1 rows and six new Round 2 rows**. Repeated EW controls, component-only portfolios, and exact or near-exact aliases are intentionally retained for traceability. A row is a saved configuration or control, not necessarily an independent modeling idea. Values are copied from the combined verified CSV.

### Round 1: all 65 rows

#### Baseline regressions and weighting overlays

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| EW | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| LASSO_Constrained | 8.804% | 0.12233 | 0.5743% | -4.400% | 1.500 |
| LASSO_LogGrowth | 8.073% | 0.11330 | 0.5716% | -4.512% | 1.500 |
| LASSO | 6.979% | 0.09951 | 0.5673% | -4.828% | 3.036 |
| LASSO_CV_Weighted | 6.476% | 0.09629 | 0.5452% | -4.892% | 3.177 |
| RidgeWeighted | 6.599% | 0.09226 | 0.5814% | -5.847% | 2.145 |
| Ridge_CV_Weighted | 5.204% | 0.08050 | 0.5298% | -5.597% | 3.612 |
| Ridge | 5.104% | 0.07778 | 0.5389% | -4.933% | 3.125 |
| LASSO_CV | 4.737% | 0.06589 | 0.5981% | -4.860% | 3.800 |
| Ridge_CV | 3.759% | 0.05681 | 0.5550% | -5.153% | 4.576 |
| MinVar | 2.936% | 0.03677 | 0.7074% | -4.612% | 8.921 |

#### Constrained LASSO and reward/exposure search

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| LASSO_Selected † | 11.248% | 0.14723 | 0.6010% | -4.191% | 1.015 |
| Selected_ExposureOnly † | 11.248% | 0.14723 | 0.6010% | -4.191% | 1.015 |
| EW | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| LASSO_Constrained | 8.804% | 0.12233 | 0.5743% | -4.400% | 1.500 |
| Selected_RewardOnly † | 8.804% | 0.12233 | 0.5743% | -4.400% | 1.500 |
| LASSO_LogGrowth | 8.073% | 0.11330 | 0.5716% | -4.512% | 1.500 |
| LASSO | 6.979% | 0.09951 | 0.5673% | -4.828% | 3.036 |

#### Recency-weighted LASSO

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| LASSO_252d_Weighted_Selected † | 8.508% | 0.11976 | 0.5678% | -4.598% | 1.500 |
| LASSO_252d_HL504_Constrained | 8.508% | 0.11976 | 0.5678% | -4.598% | 1.500 |
| LASSO_252d_HL252_Constrained | 8.277% | 0.11803 | 0.5611% | -4.610% | 1.500 |
| LASSO_252d_HL126_Constrained | 7.733% | 0.11352 | 0.5467% | -4.550% | 1.500 |
| LASSO_252d_Weighted_HL252 | 6.988% | 0.10260 | 0.5500% | -4.785% | 2.958 |

#### Elastic Net

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| ElasticNet_l1ratio_0.95 | 7.012% | 0.10011 | 0.5664% | -4.811% | 3.069 |
| ElasticNet_l1ratio_0.975 | 6.994% | 0.09980 | 0.5668% | -4.818% | 3.052 |
| ElasticNet_l1ratio_0.9 | 6.975% | 0.09974 | 0.5656% | -4.795% | 3.105 |
| ElasticNet_l1ratio_0.99 | 6.985% | 0.09964 | 0.5671% | -4.824% | 3.042 |
| ElasticNet_l1ratio_0.995 | 6.982% | 0.09958 | 0.5672% | -4.826% | 3.039 |
| ElasticNet_l1ratio_1 | 6.979% | 0.09951 | 0.5673% | -4.828% | 3.036 |
| ElasticNet_l1ratio_0.85 | 6.854% | 0.09820 | 0.5651% | -4.786% | 3.144 |
| ElasticNet_l1ratio_0.8 | 6.703% | 0.09612 | 0.5654% | -4.776% | 3.190 |
| ElasticNet_l1ratio_0.75 | 6.669% | 0.09544 | 0.5667% | -4.767% | 3.240 |
| ElasticNet_l1ratio_0.5 | 5.489% | 0.07863 | 0.5733% | -4.583% | 3.588 |
| ElasticNet_l1ratio_0.25 | 4.196% | 0.06049 | 0.5801% | -4.390% | 4.395 |

#### Rolling PCA with LASSO/Ridge

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| Ridge_PCA_15 | 6.081% | 0.08645 | 0.5742% | -4.931% | 3.607 |
| LASSO_PCA_15 | 5.838% | 0.08535 | 0.5587% | -4.591% | 3.192 |
| LASSO_PCA_10 | 5.337% | 0.07795 | 0.5625% | -4.399% | 3.148 |
| Ridge_PCA_10 | 4.697% | 0.06881 | 0.5655% | -4.395% | 3.457 |
| LASSO_PCA_20 | 3.853% | 0.05758 | 0.5610% | -4.645% | 3.295 |
| Ridge_PCA_20 | 3.632% | 0.05117 | 0.6021% | -5.000% | 3.948 |
| LASSO_PCA_5 | 3.174% | 0.04640 | 0.5840% | -5.526% | 2.904 |
| LASSO_PCA_3 | 3.293% | 0.04155 | 0.6911% | -6.833% | 2.616 |
| Ridge_PCA_5 | 2.702% | 0.03979 | 0.5879% | -5.505% | 3.025 |
| Ridge_PCA_3 | 2.928% | 0.03728 | 0.6933% | -6.849% | 2.686 |
| LASSO_PCA_30 | 2.327% | 0.03552 | 0.5723% | -4.857% | 3.375 |
| Ridge_PCA_30 | 0.656% | 0.01189 | 0.5944% | -5.642% | 4.459 |

#### Market-beta constraints

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| Ridge_conservative_0.5_0.9 | 11.001% | 0.15110 | 0.5725% | -5.429% | 1.500 |
| LASSO_market_like_0.8_1.2 | 14.028% | 0.13901 | 0.7904% | -6.435% | 1.328 |
| LASSO_conservative_0.5_0.9 | 10.160% | 0.13725 | 0.5858% | -4.687% | 1.500 |
| Ridge_unconstrained | 9.806% | 0.13619 | 0.5705% | -5.429% | 1.500 |
| Ridge_market_like_0.8_1.2 | 13.438% | 0.13224 | 0.7995% | -6.262% | 1.225 |
| LASSO_unconstrained | 8.807% | 0.12238 | 0.5743% | -4.399% | 1.500 |

#### Two-stage portfolio mixtures and component controls

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| Mean_252d_E | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| LASSO_252d_E | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| LASSO_252d_ReturnVolatility † | 9.167% | 0.12818 | 0.5690% | -4.804% | 1.500 |
| Mean_252d_Sharpe | 9.167% | 0.12817 | 0.5690% | -4.803% | 1.500 |
| LASSO_252d_Sharpe | 9.167% | 0.12817 | 0.5690% | -4.803% | 1.500 |
| LASSO_252d_V | 9.167% | 0.12817 | 0.5690% | -4.803% | 1.500 |
| Mean_252d_V | 9.167% | 0.12817 | 0.5690% | -4.803% | 1.500 |
| Mean_252d_ReturnVolatility † | 10.240% | 0.07870 | 1.0813% | -9.878% | 1.500 |
| Mean_252d_U | 10.240% | 0.07870 | 1.0813% | -9.878% | 1.500 |
| LASSO_252d_U | 8.260% | 0.06460 | 1.0905% | -10.780% | 1.500 |

#### Direct asset-level linear programming

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| EW | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| LP_LASSO_252d | 16.719% | 0.12841 | 1.0195% | -8.820% | 1.500 |
| LP_Mean_252d | 16.102% | 0.12356 | 1.0252% | -9.406% | 1.500 |

### Round 2: all six added variants

#### Turnover penalty

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| LASSO_Turnover_0p1 | 10.380% | 0.13245 | 0.6209% | -5.072% | 1.014 |
| LASSO_Turnover_1 | 11.956% | 0.11317 | 0.8428% | -7.409% | 1.013 |

#### Size-BM grid smoothness

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| LASSO_Smooth_0p1 | 11.275% | 0.14751 | 0.6012% | -4.191% | 1.015 |
| LASSO_Smooth_1 | 11.241% | 0.14723 | 0.6006% | -4.132% | 1.015 |

#### Ledoit-Wolf covariance shrinkage

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| LASSO_LedoitWolf | 11.003% | 0.14317 | 0.6057% | -4.376% | 1.015 |

#### Single-asset weight cap

| Model | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|
| LASSO_WeightCap_10pct | 12.702% | 0.15611 | 0.6359% | -5.494% | 1.016 |

The four Round 2 output directories also contain their rerun EW and selected-LASSO controls; those eight repeated controls are omitted from the 71-row combined table because their holdings match Round 1 exactly. The July final-test period is not included in either validation round.
