# Portfolio Optimization: Final Report Outline

Updated: 2026-09-24. This is an **outline**, not the final report prose. Each modeling idea has one subsection in its primary category. Each subsection gives its location in *DBA5106 GP.docx*, a brief method description focused on overfitting or estimation risk, and its observed result. The document is a source for the experiment history; current figures come from the [unified 252-day validation](unified-252-2026-09-22.md), [sensitivity rerun](unified-252-sensitivity-2026-09-23.md), [cost-aware LP study](lp-mad-cost-2026-09-24.md), [LP stability comparison](lp-stability-2026-09-24.md), and [July final-test results](../../results/final_test_july_2026_20260924/metrics/report_final_test_summary.csv). Appendix A retains all 71 original validation rows; Appendix B preserves every evaluated July configuration on the same gross reporting basis.

## 1. Dataset overview

The equal-weighted daily-return section of the Size–Book-to-Market 10 × 10 portfolio dataset contains 100 portfolio return series and 26,296 dates from 1926-07-01 to 2026-07-31. These series represent characteristic-sorted portfolios, not individual stocks. The report uses 98 retained series after the specified exclusions. **Source in GP document:** the unnumbered “Regarding the dataset” paragraph.

## 2. Data: preprocessing and chronological splits

### 2.1 Preprocessing and missingness

Preprocessing first removes `ME9 BM10` and `BIG HiBM`, converts `-99.99` and `-999` to missing values, and changes percentage-point returns to decimal returns; it does not impute returns. The five-year historical extract begins in **2021**, with its first trading date on **2021-08-02**, and ends on 2026-07-31. It has **1,255 dates**. Before the two columns are removed, this extract has **272 missing cells**: 250 in `ME9 BM10` and 22 in `BIG HiBM`. After exclusion, the **1,255 × 98** panel has **zero missing cells and zero incomplete dates**. Earlier retained assets have historical missing values, so this completeness claim applies to the stated five-year extract, not the full 1926–2026 file. [Data-quality audit](../data-quality/missing-values-2026-09-22.md).

### 2.2 Training, validation, and final-test chronology

The five-year extract is an available history, **not** a five-year fit at every date. Historical studies compared 126 trading observations, 252 trading observations, and up to three calendar years (about 750–755 observations), including recency-weighted variants. The latest common comparison uses **252 preceding trading dates** for rolling fits, forecasts, and risk estimates; the separate mixture decision layer also uses 252 earlier training observations. All fits use information available before their holding date.

The common development validation is **2026-01-02 to 2026-06-30** (123 trading days). Predictor regularization for the two-stage models was calibrated before evaluation using three late, chronological 21-day folds ending by 2024-07-31. The final-test period is **2026-07-01 to 2026-07-31** (22 trading days). The original six configurations were fixed before its first evaluation. The current cost-aware LP was developed after those July results had been reviewed; its settings were frozen before the present run and were not retuned on July. Consequently, the current LP result is not evidence from a newly untouched test. Earlier validation results informed model development, so they are comparative evidence rather than an untouched test. Daily Sharpe is mean daily return divided by sample daily standard deviation (`ddof=1`), with zero risk-free rate and no annualization. The original representatives were chosen using validation performance and control roles; the current cost-aware specification represents the LP family in the main table, with the original LP retained in Appendix B. All reported cumulative returns, daily Sharpe ratios, volatility and drawdown use **gross returns before transaction costs**, consistently across validation and the final test. A trading-cost penalty inside an optimization objective does not change this common reporting basis. Cost-adjusted diagnostics are retained in the saved experiment outputs but are not used for the report ranking. [Window study](lasso-windows-5y.md) · [Final-test protocol](final-test-july-2026.md).

## 3. Baseline models

### 3.1 Equal-Weight Portfolio (EW)

- **GP source:** unnumbered “basemodel” and “Current situation” sections.
- **Method:** Allocate equally to the retained portfolios; this estimation-free benchmark avoids fitting noise.
- **Result:** January–June validation cumulative return **16.933%**, daily Sharpe **0.13253**—the highest validation return among the original representatives, before the cost-aware LP update.

### 3.2 Minimum-Variance Portfolio (MinVar)

- **GP source:** unnumbered “basemodel” and “Current situation” sections.
- **Method:** Fit the original minimum-variance regression without LASSO or Ridge shrinkage, exposing sensitivity to noisy estimates.
- **Result:** Validation return **2.936%**, Sharpe **0.03677**, and average gross exposure **8.921**.

### 3.3 Fixed-Penalty LASSO

- **GP source:** unnumbered “Current situation” and “Adaptation of Lambda” sections.
- **Method:** Penalize coefficients in the contemporaneous minimum-variance regression, then map them to holdings; this is **not** a next-day return forecast.
- **Result:** Validation return **6.979%** and Sharpe **0.09951**, above MinVar but below EW.

### 3.4 Fixed-Penalty Ridge

- **GP source:** unnumbered “Current situation” and “Adaptation of Lambda” sections.
- **Method:** Use L2 coefficient shrinkage in the same minimum-variance portfolio construction to stabilize estimates.
- **Result:** Validation return **5.104%** and Sharpe **0.07778**.

## 4. Loss: objective and soft-penalty changes

### 4.1 Log-Growth-Reward LASSO

- **GP source:** §5, “Improving Cumulative Returns through Loss Function Optimization.”
- **Method:** Reward historical log growth within hard-constrained LASSO; the reward can fit historical return noise, so it is compared with the matched constraints-only model.
- **Result:** At fixed reward strength, validation return **8.073%**, Sharpe **0.11330**, below Hard-Constrained LASSO's **8.804%** and **0.12233**.

### 4.2 Exposure-Penalized LASSO with Tunable Return Reward

- **GP source:** §6, “Parameterized LASSO with Constraints and Return Reward.”
- **Method:** Search a small reward/exposure-penalty set, retaining hard constraints and early stopping in parameter refinement; the added exposure penalty aims to curb unstable long–short holdings.
- **Result:** The selected `LASSO_Selected` uses reward weight **η = 0** and exposure-penalty weight **γ = 2**: validation return **11.248%**, Sharpe **0.14723**, average gross exposure **1.015**. Improvement cannot be credited to a positive reward.

### 4.3 Elastic Net Portfolio

- **GP source:** §12, “Use of Elastic Net Regularization Method.”
- **Method:** Combine L1 and L2 coefficient penalties to reduce dependence on a noisy sparse selection among correlated portfolios.
- **Result:** Best validation configuration, L1 ratio **0.95**, returned **7.012%** with Sharpe **0.10011**; its gain over fixed LASSO was marginal.

### 4.4 Turnover-Penalized LASSO

- **GP source:** §7, “Reducing Portfolio Instability through Turnover Penalty.”
- **Method:** Penalize changes from the previous holdings to reduce sensitivity to small rolling-sample changes.
- **Result:** The representative `LASSO_Turnover_0p1` returned **10.380%**, Sharpe **0.13245**, below the selected unpenalized exposure model's **11.248%** and **0.14723** before trading costs.

### 4.5 Size–BM Grid-Smoothed LASSO

- **GP source:** §8, “Fused LASSO (Size–Book-to-Market Structure).”
- **Method:** Softly penalize weight differences between adjacent Size–BM portfolios to reduce cross-sectional fitting of noise; the penalty acts on **portfolio weights**, not LASSO coefficients.
- **Result:** `LASSO_Smooth_0p1` returned **11.275%**, Sharpe **0.14751**, only slightly above the selected exposure model.

## 5. Structure: information and decision-design changes

### 5.1 Cross-Validated Regularization

- **GP source:** unnumbered “Adaptation of Lambda” section.
- **Method:** Select regularization strength within each rolling window. This addresses arbitrary fixed penalties, but a fold score need not align with portfolio Sharpe; the legacy five-fold LASSO uses MSE and Ridge uses its default R² score.
- **Result:** `LASSO_CV` returned **4.737%**, Sharpe **0.06589**; `Ridge_CV` returned **3.759%**, Sharpe **0.05681**, both below their fixed-penalty counterparts.

### 5.2 Recency Weighting and Shrink-Toward-EW Overlay

- **GP source:** unnumbered “Adaptation of Lambda,” `_Weighted` variants.
- **Method:** Give recent training observations more influence, then blend fitted holdings toward EW to limit unstable deviations; these are overlays, not a third lambda-selection rule.
- **Result:** The representative `LASSO_CV_Weighted` returned **6.476%**, Sharpe **0.09629**, improving on `LASSO_CV` but not fixed LASSO's **0.09951** Sharpe.

### 5.3 Training-Window Length and Recency-Weighted LASSO

- **GP source:** §14, “Three-Year LASSO Training Window”; the 126-to-252-day comparison also appears in unnumbered “Attempts.”
- **Method:** Compare shorter and longer rolling histories, then downweight older observations when using longer histories. More samples can reduce estimation noise, while aggressive recency weighting can overfit recent regimes. The 126-day implementation remains available, but no comparable saved 126-day LASSO/Ridge result is present.
- **Result:** In the historical January–June comparison, three-year equal-weighted LASSO returned **8.46%**, Sharpe **0.1192**, versus 252-day fixed LASSO's **6.98%**, **0.0995**; three-year weighted constrained LASSO returned **8.11%**, **0.1159**. Under the latest uniform 252-day rerun, the representative recency-weighted constrained LASSO returned **8.508%**, **0.11976**. These are distinct window settings, not a single pooled ranking. [Three-year study](lasso-windows-5y.md) · [Weighted constraints](lasso-3y-weight-tuning.md).

### 5.4 Rolling PCA with LASSO or Ridge

- **GP source:** §11, “Use of Principal Component Analysis (PCA) to reduce features.”
- **Method:** Estimate a lower-dimensional return representation within each rolling training window before regularized fitting, aiming to reduce 97-dimensional adjustment noise.
- **Result:** Representative `Ridge_PCA_15` returned **6.081%**, Sharpe **0.08645**, below fixed LASSO and EW in the latest rerun.

### 5.5 Ledoit–Wolf Covariance Shrinkage

- **GP source:** §9, “Covariance Estimation using Ledoit–Wolf Shrinkage.”
- **Method:** Shrink the noisy sample covariance estimate within the selected LASSO formulation; this changes risk estimation, not the asset universe or decision model.
- **Result:** `LASSO_LedoitWolf` returned **11.003%**, Sharpe **0.14317**, below the matched sample-covariance model's **11.248%**, **0.14723**.

### 5.6 Two-Stage Sharpe-Trained Mixture

- **GP source:** no dedicated DOCX chapter; §15 supplies the related prediction/decision separation.
- **Method:** Forecast next-day asset returns, form return-seeking, minimum-risk, and EW candidates, then train a low-dimensional mixture for sample daily Sharpe rather than 98 free decision weights.
- **Result:** Representative `Mean_252d_Sharpe` returned **9.167%**, Sharpe **0.12817**; the LASSO-forecast mixture was numerically indistinguishable, so the results do not establish a forecasting advantage.

### 5.7 Two-Stage Return–Volatility-Trained Mixture

- **GP source:** no dedicated DOCX chapter; §15 supplies the related prediction/decision separation.
- **Method:** Use the same low-dimensional candidate mixture but train its weights on mean return minus a volatility penalty, limiting decision-layer complexity.
- **Result:** Representative `LASSO_252d_ReturnVolatility` returned **9.167%**, Sharpe **0.12818**, effectively the same risk-adjusted outcome as the Sharpe-trained mixture.

### 5.8 LASSO Forecast + Direct Linear Programming

- **GP source:** §15, “Two-Stage Portfolio Construction with Linear Programming.” The cost and stability refinements extend this design.
- **Design:** Separate prediction from allocation. A causal LASSO predicts next-day returns from lagged features, using 252 preceding observations and a shared alpha fixed at **0.0025118864**. A linear program directly chooses 98 holdings. To limit overfitting and unstable exposure, weights sum to one, gross exposure is capped at **1.5**, absolute asset weights at **5%**, and historical portfolio MAD and uncompounded drawdown at their EW levels; historical daily wealth must remain positive. MAD is a linearizable measure of portfolio fluctuations, not standard deviation or a guarantee about future risk.
- **Implementation:** The current objective is `maximize predicted_return(w) − 0.0005 × sum(abs(w − w_pretrade))`, where previous holdings are adjusted for market drift. Absolute-value auxiliary variables preserve linear programming. An EW-distance tie-break selects among nearly equal objective values; there is no turnover budget or trained decision-layer coefficient vector. The cost term guides allocation, while reported performance follows the common gross-return convention.
- **Optimization process:** The original return-only LP had **18.300%** average daily validation turnover. Adding a fixed **5 bps** one-way cost penalty reduced this to **2.434%**. A six-value MAD-penalty search on July–December 2025 selected **gamma = 0**, using the previously specified fee-adjusted training Sharpe criterion. Thus the result supports cost-aware decisions rather than a positive MAD penalty; the hard MAD constraint remains. Subsequent stability checks compared a **1.2** gross cap and a training-selected EW penalty of **0.0001**. The lower cap reduced validation risk with **17.540%** gross return, while EW anchoring returned **18.531%** and worsened drawdown. The **1.5-cap, zero-MAD-penalty, zero-EW-penalty** version remains the current specification; those two alternatives were not evaluated on July.
- **Result:** Current LP validation return **19.659%**, daily Sharpe **0.14834**, compared with the original LP's **16.719%** and **0.12841**. July did not reproduce this improvement: current LP returned **−0.410%** with Sharpe **−0.02264**, versus the original LP's **0.200%** and **0.01629**. It traded less and had a shallower July drawdown, but did not improve July return or Sharpe. Full final-test comparisons appear in §7 and Appendix B. [Cost study](lp-mad-cost-2026-09-24.md) · [Stability study](lp-stability-2026-09-24.md).

## 6. Constraint: feasible-holdings changes

### 6.1 Hard-Constrained LASSO

- **GP source:** §4, “Improving Cumulative Returns through Portfolio Constraints.”
- **Method:** Keep the fixed LASSO loss while limiting gross exposure to 1.5, training-window volatility to EW's level, and historical daily wealth to a positive floor. These limits act on **holdings and their historical returns**, separately from LASSO's coefficient penalty.
- **Result:** `LASSO_Constrained` returned **8.804%**, Sharpe **0.12233**, versus fixed LASSO's **6.979%**, **0.09951**; average gross exposure fell from **3.036** to **1.500**.

### 6.2 Market-Beta-Constrained LASSO

- **GP source:** §13, “Introducing Market Beta to Daily Portfolio Allocation.”
- **Method:** Estimate beta relative to EW and constrain the final portfolio's market sensitivity, controlling an exposure that coefficient regularization alone does not target.
- **Result:** Market-like beta range **0.8–1.2**, `LASSO_market_like_0.8_1.2`, returned **14.028%**, Sharpe **0.13901**; the conservative **0.5–0.9** LASSO variant returned **10.160%**, **0.13725**.

### 6.3 Market-Beta-Constrained Ridge

- **GP source:** §13, “Introducing Market Beta to Daily Portfolio Allocation.”
- **Method:** Apply the same portfolio-beta bounds to the Ridge base, retaining a separate comparison because L1 and L2 fits produce different holdings.
- **Result:** Conservative `Ridge_conservative_0.5_0.9` returned **11.001%**, Sharpe **0.15110**; its market-like variant returned **13.438%**, **0.13224**.

### 6.4 10%-Capped Exposure-Penalized LASSO

- **GP source:** §10, “Applying a 10% Single-Asset Weight Cap Constraint.”
- **Method:** Add `|w_i| ≤ 0.10` to the selected exposure-penalized LASSO to prevent concentration that aggregate leverage control can leave untouched.
- **Result:** `LASSO_WeightCap_10pct` returned **12.702%**, Sharpe **0.15611**—the highest validation Sharpe among the 71 saved rows. The cap's separate July outcome appears in §7.

## 7. Selected methods and July 2026 final test

EW is the benchmark; the remaining methods represent strong validation Sharpe, return, or explicit exposure control. Each idea retains one representative in the main table. January–June validation contains 123 days and July contains 22, so cumulative returns are **not horizon-matched**. Every metric below is calculated from **gross portfolio returns, before transaction costs**, matching the original comparison.

Gross daily return is `r_t = sum(w_i,t × asset_return_i,t)`; cumulative return is `product(1 + r_t) − 1`. No trading-cost deduction enters this table's cumulative return, Sharpe, volatility or drawdown. The current LP still uses its fixed 5 bps trading-cost penalty when choosing holdings. Each July path starts from EW, then uses only earlier observations for subsequent decisions. Original model results are reused from their saved holdings without refitting or reselection.

| Method | Validation return | Validation daily Sharpe | July return | July daily Sharpe | July daily volatility | July max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| EW | 16.933% | 0.13253 | -0.841% | -0.05186 | 0.6952% | -2.135% |
| Exposure-Penalized LASSO | 11.248% | 0.14723 | 2.017% | 0.16073 | 0.5748% | -1.296% |
| 10%-Capped Exposure-Penalized LASSO | 12.702% | 0.15611 | 1.667% | 0.15531 | 0.4915% | -1.103% |
| Conservative-Beta Ridge | 11.001% | 0.15110 | 1.289% | 0.10612 | 0.5629% | -1.331% |
| Market-Like-Beta LASSO | 14.028% | 0.13901 | 0.451% | 0.03451 | 0.6508% | -1.549% |
| LASSO Forecast + Direct LP (cost-aware) | 19.659% | 0.14834 | -0.410% | -0.02264 | 0.7159% | -1.771% |

Exposure-Penalized LASSO retains the best July return and Sharpe, while its 10%-capped counterpart has the lowest July volatility and smallest drawdown. Current LP loses less than EW (**−0.410% vs. −0.841%**), with a smaller drawdown (**−1.771% vs. −2.135%**), but slightly greater daily volatility (**0.7159% vs. 0.6952%**). Its strong validation return therefore does not establish superior out-of-sample performance. The original LP and all saved July configurations remain in Appendix B. [Combined gross results](../../results/final_test_july_2026_20260924/metrics/report_final_test_summary.csv) · [Verification](../../results/final_test_july_2026_20260924/config/verification.json) · [Original selection protocol](final-test-july-2026.md).

## 8. Interpretation to develop in the final prose

- **Overfitting and stability:** MinVar's extreme gross exposure motivates regularization and explicit holdings limits. LASSO, soft exposure control, beta limits, and a position cap each target a different source of estimation or decision instability. A constraint satisfied on training history cannot guarantee the same future risk bound.
- **What held up:** Exposure-Penalized LASSO led July return and Sharpe; the capped variant had the lowest volatility and shallowest drawdown. The selected exposure model had **η = 0**, so these outcomes do not validate the proposed return reward. The cap's validation Sharpe lead over the uncapped model did not repeat in July.
- **What weakened:** Conservative-Beta Ridge remained positive but its July Sharpe fell from validation's **0.15110** to **0.10612**. Market-Like-Beta LASSO had limited gains while EW was negative; exposure is a plausible mechanism, not a proven causal explanation. The current LP's cost penalty reduced July average daily turnover from the original LP's **12.708%** to **3.434%**, but return fell from **0.200%** to **−0.410%**. This difference reflects changed holdings, not a deduction of fees from reported returns. Penalizing trades can retain positions whose subsequent returns disappoint; forecast errors and the flexibility of 98 decision weights remain possible sources of overfitting. Lower turnover and a shallower drawdown need not coincide with higher return.
- **Inference limit:** July contains only 22 returns. The earlier validation period was repeatedly used in development, and the current LP was developed after the original July results were visible, as recorded in §2.2. The results describe this observed period; chronological fitting does not restore an untouched test. Reported performance excludes transaction costs, slippage, market impact, borrowing and financing costs, so it is not an estimate of fully implementable net profit.

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

## Appendix B — complete July final-test results

All seven saved configurations, including both stages of the LP design, are retained below. Gross results for the original six are unchanged. All metrics follow the report's gross-return convention; average daily turnover is half the L1 change from drifted holdings. No original model was retrained for this comparison.

| Configuration | Cumulative return | Daily Sharpe | Daily volatility | Max drawdown | Avg. daily turnover | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|---:|
| EW | -0.841% | -0.05186 | 0.6952% | -2.135% | 0.328% | 1.000 |
| LASSO_Selected | 2.017% | 0.16073 | 0.5748% | -1.296% | 5.272% | 1.016 |
| LASSO_WeightCap_10pct | 1.667% | 0.15531 | 0.4915% | -1.103% | 4.361% | 1.016 |
| Ridge_conservative_0.5_0.9 | 1.289% | 0.10612 | 0.5629% | -1.331% | 6.255% | 1.500 |
| LASSO_market_like_0.8_1.2 | 0.451% | 0.03451 | 0.6508% | -1.549% | 2.425% | 1.236 |
| LP_LASSO_252d | 0.200% | 0.01629 | 0.7043% | -1.926% | 12.708% | 1.500 |
| LP_Current | -0.410% | -0.02264 | 0.7159% | -1.771% | 3.434% | 1.500 |

The [saved run](../../results/final_test_july_2026_20260924/) contains daily returns, current-LP holdings and forecasts, frozen settings, source fingerprints, and evaluation scripts. Its `report_final_test_summary.csv` reproduces the gross figures in this report; the combined CSV also retains explicitly labeled `net_*` diagnostics, which are not used for ranking. The audit checked 22 LP allocations, 22 causal forecast windows and return/cost accounting for 154 portfolio-days across seven paths. Every checked LP allocation met its historical constraints. Original results and Appendix A remain unchanged.
