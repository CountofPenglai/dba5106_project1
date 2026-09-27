# Final model summary

This summary follows the model names and section numbering in **DBA5106_Final_Report_Draft 2**. It contains only retained report methods; previous exploratory results are in the complete backup.

All main figures use gross returns, daily unannualized Sharpe, sample daily volatility and compounded maximum drawdown. Parameters are fixed; this refactor performs no new model selection.

## Validation: January 2–June 30, 2026

All 19 report configurations use the preceding 252 trading observations. The six baselines also support 126-day execution; earlier results are in the external backup.

| Report section | Model | Return | Daily Sharpe | Daily volatility | Max drawdown | Average gross |
|---|---|---:|---:|---:|---:|---:|
| 2 | Equal Weight (`EW`) | 16.933% | 0.13253 | 0.9974% | -8.632% | 1.000 |
| 2 | Minimum Variance (`MinVar`) | 2.936% | 0.03677 | 0.7074% | -4.612% | 8.921 |
| 2 | LASSO (coefficient constraint) (`LASSO_CoefficientConstraint`) | 7.993% | 0.10241 | 0.6299% | -6.325% | 1.946 |
| 2 | Ridge (coefficient constraint) (`Ridge_CoefficientConstraint`) | 3.435% | 0.04586 | 0.6436% | -4.610% | 7.429 |
| 2 | LASSO (penalty) (`LASSO_Penalty`) | 6.979% | 0.09951 | 0.5673% | -4.828% | 3.036 |
| 2 | Ridge (penalty) (`Ridge_Penalty`) | 5.104% | 0.07778 | 0.5389% | -4.933% | 3.125 |
| 3.1 | Hard-Constrained LASSO (`LASSO_HardConstraints`) | 8.804% | 0.12233 | 0.5743% | -4.400% | 1.500 |
| 3.2 | LASSO with Investment Size Control (`LASSO_ExposurePenalty`) | 11.248% | 0.14723 | 0.6010% | -4.191% | 1.015 |
| 3.3 | Turnover-Penalized LASSO (tau=0.1) (`LASSO_Turnover_0p1`) | 10.380% | 0.13245 | 0.6209% | -5.072% | 1.014 |
| 3.3 | Turnover-Penalized LASSO (tau=1) (`LASSO_Turnover_1`) | 11.956% | 0.11317 | 0.8428% | -7.409% | 1.013 |
| 4.1 | LASSO Forecasting + Original LP (`LP_Original`) | 16.719% | 0.12841 | 1.0195% | -8.820% | 1.500 |
| 4.1 | LASSO Forecasting + Trading-Penalized LP (`LP_TradingPenalty`) | 19.659% | 0.14834 | 1.0190% | -9.593% | 1.500 |
| 5.1 | LASSO with a 10% Single-Asset Weight Cap (`LASSO_WeightCap_10pct`) | 12.702% | 0.15611 | 0.6359% | -5.494% | 1.016 |
| 5.2 | LASSO with Market Beta Control (Unbounded) (`LASSO_BetaUnbounded`) | 8.807% | 0.12238 | 0.5743% | -4.399% | 1.500 |
| 5.2 | LASSO with Market Beta Control (Market) (`LASSO_BetaMarket`) | 14.028% | 0.13901 | 0.7904% | -6.435% | 1.328 |
| 5.2 | LASSO with Market Beta Control (Conservative) (`LASSO_BetaConservative`) | 10.160% | 0.13725 | 0.5858% | -4.687% | 1.500 |
| 5.2 | Ridge with Market Beta Control (Unbounded) (`Ridge_BetaUnbounded`) | 9.806% | 0.13619 | 0.5705% | -5.429% | 1.500 |
| 5.2 | Ridge with Market Beta Control (Market) (`Ridge_BetaMarket`) | 13.438% | 0.13224 | 0.7995% | -6.262% | 1.225 |
| 5.2 | Ridge with Market Beta Control (Conservative) (`Ridge_BetaConservative`) | 11.001% | 0.15110 | 0.5725% | -5.429% | 1.500 |

The tau=0 turnover control reuses LASSO with Investment Size Control. Market-beta variants use hard constraints without the gamma=2 investment-size penalty. The unbounded beta-LASSO and Hard-Constrained LASSO are mathematically equivalent but preserve different historical numerical formulations.

## July 1–31, 2026: previously evaluated final period

| Model | Return | Daily Sharpe | Daily volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Equal Weight | -0.841% | -0.05186 | 0.6952% | -2.135% |
| LASSO with Investment Size Control | 2.017% | 0.16073 | 0.5748% | -1.296% |
| LASSO with a 10% Single-Asset Weight Cap | 1.667% | 0.15531 | 0.4915% | -1.103% |
| Ridge with Market Beta Control (Conservative) | 1.289% | 0.10612 | 0.5629% | -1.331% |
| LASSO with Market Beta Control (Market) | 0.451% | 0.03451 | 0.6508% | -1.549% |
| LASSO Forecasting + Original LP | 0.200% | 0.01629 | 0.7043% | -1.926% |
| LASSO Forecasting + Trading-Penalized LP | -0.410% | -0.02264 | 0.7159% | -1.771% |

LASSO with Investment Size Control (eta=0, gamma=2) has the strongest observed July return and Sharpe in this group; the 10% weight-cap variant is also positive. These observations do not constitute new selection on untouched data.

The trading-penalized LP improved validation return and reduced trading, but its July return was negative. It was developed after the first July results had been reviewed, so its July evaluation has a different selection history from the original frozen six-model comparison. A weak forecasting signal and allocation sensitivity remain plausible design limitations; the short test alone does not establish a unique cause.

## Design and selection evidence

- Hard constraints restrict leverage, historical variance and single-day wealth. Investment-size penalties discourage unnecessary long–short allocations within those limits.
- Turnover and individual-weight constraints address trading stability and concentration. Beta bounds control market sensitivity; each has distinct effects on return and risk.
- Both LP configurations reuse the same frozen 252-day LASSO forecast and historical MAD/drawdown constraints; only the trading-penalty coefficient changes from 0 to 0.0005.
- Historical exposure search, chronological 3x21 forecast calibration and LP penalty selection are preserved in [selection evidence](../../results/report/selection/), with unchanged provenance.

## Sources

- [Validation summary](../../results/report/validation/metrics/summary.csv)
- [July summary](../../results/report/july_test/metrics/summary.csv)
- [Full rerun and report discrepancies awaiting review](../report-reproduction-check.md)
- [Model/source/checksum manifest](../../results/manifest.json)
- [Architecture and formulas](../architecture/report-models.md)
- [Result conventions and backup](../results.md)
