# Training-tuned MAD and transaction-cost penalties for direct LP

This historical experiment record preserves the original net-return tuning and transaction-cost diagnostics. Current cross-model report comparisons use **gross returns before transaction costs**; see the [final model summary](final-model-summary.md). Explicitly labeled net results below are supplementary cost diagnostics, not the report's ranking basis.

## Design

The aim is to reduce unstable holdings and the cost of trading while preserving a
simple, interpretable linear program. The existing causal 252-day LASSO return
forecaster is reused, with its original shared alpha of 0.002511886431509582
(calibrated no later than 31 July 2024). Its six lagged features, per-asset scaling
and daily refitting are unchanged. The decision stage optimizes the 98 asset
weights directly; it is not a separately trained prediction or mixture model.

The daily objective is:

    maximize predicted_return(w) - gamma * MAD(w) - 0.0005 * sum(abs(w - w_pretrade))
    MAD(w) = mean(abs(R @ w - mean(R @ w)))

Here R contains the preceding 252 daily return observations, and w_pretrade is
the previous allocation after market drift. MAD measures fluctuations of the
combined portfolio and captures offsets between assets. It is a linearizable
proxy for volatility, not standard deviation. Absolute-value auxiliary variables
keep the optimization in scipy/HiGHS linear programming.

The existing limits remain: weights sum to one; gross exposure is at most 1.5;
absolute asset weights are at most 5%; historical MAD and uncompounded cumulative
P&L drawdown may not exceed their historical EW counterparts; historical daily
wealth must remain positive. The second optimization chooses the closest solution
to EW in L1 while retaining the full primary utility within 1e-10. This is a
tie-break, not an additional shrinkage coefficient. There is no turnover budget.

Transaction costs are an experimental assumption of 5 bps per unit of traded
notional on each side. If half-L1 turnover is 20%, the charge is 0.0005 × 40% =
0.02% of portfolio value. All validation strategies, including EW and the original
LP, pay the same fee for comparison. Net daily return is approximated by gross
return minus this charge; holdings drift using gross returns. This first-order
accounting omits fee-financed trade adjustments, market impact, short borrow fees
and funding costs.

## Training and validation protocol

- Input history begins in August 2021, after the existing two-column exclusion.
  The retained 98 assets have no missing observations in the used period.
- The forecaster and risk context both use 252 preceding trading observations.
  A single `LPRiskTuningConfig.window` controls both lengths.
- Gamma candidates are fixed before evaluation: 0, 0.05, 0.1, 0.25, 0.5 and 1.
- Tuning uses 1 July–31 December 2025: 128 chronological, one-day-ahead decisions
  inside the earlier development/training data. The same forecasts are shared
  across candidates. Selection maximizes net daily Sharpe; numerical ties favor
  smaller realized drawdown, then larger gamma.
- Gamma is frozen before evaluating 2 January–30 June 2026: 123 validation days.
  Only the selected gamma and predeclared controls are evaluated there. As in the
  existing rolling protocol, later forecasts may use already observed validation
  days, while alpha and gamma remain fixed.
- Both periods start independently from EW and charge the initial rebalance.
  No selected training holdings are carried into validation.
- July final-test data are excluded. January–June is a previously used development
  validation period, so this experiment supplies no new untouched-test evidence.

The controls are the original return-only LP, an LP with transaction costs but
gamma fixed at zero, and EW. Original LP artifacts and the July final-test report
remain unchanged. Gross metrics allow comparison to earlier reports; net metrics
provide the fair cost-adjusted comparison for this experiment.


## Results

### Training-period coefficient selection

| Gamma | Net cumulative return | Net daily Sharpe | Net daily volatility | Net maximum drawdown | Selected |
|---:|---:|---:|---:|---:|:---:|
| 0 | 11.668% | 0.08379 | 1.101% | -8.940% | Yes |
| 0.05 | 10.131% | 0.07684 | 1.053% | -9.222% |  |
| 0.1 | 7.683% | 0.06195 | 1.017% | -8.770% |  |
| 0.25 | 5.051% | 0.04987 | 0.843% | -7.864% |  |
| 0.5 | 6.438% | 0.07169 | 0.716% | -5.457% |  |
| 1 | 6.322% | 0.07293 | 0.689% | -5.263% |  |

Training selected **gamma = 0**. Higher penalties reduced realized volatility, but
none surpassed the zero-penalty candidate on the predeclared net Sharpe criterion.
For example, gamma = 1 lowered training daily volatility from 1.101% to 0.689% and
drawdown magnitude from 8.940% to 5.263%, while lowering return from 11.668% to
6.322% and daily Sharpe from 0.08379 to 0.07293. This is evidence of a risk–return
trade-off within training, not validation evidence for gamma = 1.

### Frozen validation comparison

| Strategy | Gross return | Net return | Net daily Sharpe | Net daily volatility | Net maximum drawdown | Average daily turnover |
|---|---:|---:|---:|---:|---:|---:|
| Training-selected LP (gamma = 0) | 19.659% | 19.302% | 0.14598 | 1.019% | -9.654% | 2.434% |
| Cost-only LP control (gamma = 0) | 19.659% | 19.302% | 0.14598 | 1.019% | -9.654% | 2.434% |
| Original return-only LP | 16.719% | 14.123% | 0.11042 | 1.020% | -9.464% | 18.300% |
| Equal weight | 16.933% | 16.885% | 0.13220 | 0.997% | -8.646% | 0.335% |

Daily Sharpe is mean daily return divided by sample standard deviation (ddof = 1),
with zero risk-free rate. Returns are compounded; reported maximum drawdown is
relative compounded-wealth drawdown, which differs from the LP historical proxy.
Average turnover is half the L1 change from drifted holdings.

## Interpretation

The selected model and the cost-only control are identical. The experiment therefore
supports **cost-aware allocation**, not an incremental benefit from a nonzero MAD
penalty under this selection rule. The cost term discourages acting on small forecast
differences and reduces average daily turnover from 18.300% to 2.434%. This lowers
fee drag and changes the holding path; gross return also increases from 16.719% to
19.659%, so the net-return improvement is not solely an accounting effect.

Risk improvement is incomplete. The selected model has slightly higher net daily
volatility than EW (1.019% versus 0.997%) and a larger drawdown magnitude (9.654%
versus 8.646%). Its drawdown also exceeds the original LP after common costs
(9.464%). Gross leverage remains at 1.5. Historical MAD/drawdown feasibility
does not guarantee lower future volatility or compounded drawdown. A net-Sharpe
selection rule does not guarantee the business preference for lower drawdown either.

Anti-overfitting safeguards are a reused forecast specification, a fixed six-value
grid, strictly earlier coefficient selection, a frozen validation choice, and explicit
controls. They reduce opportunities to fit the validation outcome; they do not prove
generalization. This validation period has already informed earlier research, and no
new final-test claim is made. Nonzero candidates were not swept again on validation.

## Implementation and verification

- `portfolio/decisions/linear_program_allocation.py`: shared `LPObjective`, MAD/cost
  linearization, utility-preserving EW tie-break and independent feasibility checks.
- `portfolio/engine/lp_risk_tuning_workflow.py`: training-only selection, shared causal
  forecasts, independent holdings paths, frozen validation and gross/net metrics.
- `experiments/lasso_forecast_lp_risk.py`: reference data/config/source verification,
  one experiment entrypoint and standard categorized output.
- `tests/test_lp_risk_tuning.py`: analytical risk/cost behavior, validation and future
  exclusion, holdings drift, fee accounting and reference integrity.

Verification: all **164** repository tests passed. An independent reviewer found no
actionable issue and checked mixed-objective solutions against twelve two-asset
grids. A separate saved-artifact audit recomputed **1,137 LP allocations**, all net
metrics and **251** forecast date boundaries. Every checked allocation satisfied
the existing hard constraints. The validation forecasts exactly match the previous
unified 252-day run; original-LP holdings differ by at most 3.05e-12.

## Reproduction and artifacts

```bash
python -m experiments.lasso_forecast_lp_risk \
  --forecast-reference results/unified_252/20260922T122949.119118Z/lp
```

Saved run: [results/lasso_forecast_lp_risk/20260924T073448Z](../../results/lasso_forecast_lp_risk/20260924T073448Z/).
The run contains training search scores and paths, validation gross/net metrics,
holdings, forecasts, solver diagnostics, frozen configuration, source fingerprints
and audit/test logs. Existing model implementations and earlier results are retained.
