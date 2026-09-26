# Stability comparison of three cost-aware LP versions

This historical experiment record preserves the original net-return tuning and transaction-cost diagnostics. Current cross-model report comparisons use **gross returns before transaction costs**; see the [final model summary](final-model-summary.md). Explicitly labeled net results below are supplementary cost diagnostics, not the report's ranking basis.

## Purpose and design

This experiment targets more stable behavior under unfamiliar conditions, while
accepting some reduction in return. It compares the current successful cost-aware
LP with two separate, interpretable changes. The forecasting model is unchanged.

| Version | Gross leverage cap | Primary EW-distance penalty |
|---|---:|---:|
| Current cost-aware LP | 1.5 | 0 |
| Lower-leverage LP | 1.2 | 0 |
| EW-anchored LP | 1.5 | Selected on training data |

EW is included as a reference, not as a fourth LP variant. All versions reuse the
same causal 252-day LASSO forecasts and 252-day risk history. The original shared
forecast alpha, 0.002511886431509582, remains fixed from calibration ending on
31 July 2024. Trading costs remain 5 bps per unit of notional traded on each side.
The previously selected MAD objective penalty remains zero; existing hard limits
on MAD, drawdown, asset positions, leverage and historical wealth remain active.

The primary objective is:

    maximize predicted_return(w)
             - 0.0005 * sum(abs(w - w_pretrade))
             - tau * sum(abs(w - w_EW))

The first penalty discourages trades; the second discourages forecast-driven
deviation from the diversified EW reference. They are separate effects. The EW
term participates in the main objective, unlike the existing near-exact tie-break
toward EW. All absolute values use linear auxiliary variables, so the decision
problem remains a linear program. Lower leverage is tested independently of EW
anchoring; there is no combined version or turnover budget in this comparison.

## Predeclared selection and evaluation protocol

The loader uses history from August 2021 through June 2026 and retains the same
98 complete return columns. No July final-test observations are evaluated or used
to select parameters.

- **Training/development:** 2 January–31 December 2025, 250 rolling decisions.
  Divide the continuous realized path into three chronological blocks of 84, 83
  and 83 observations. Holdings continue through block boundaries; only block
  performance metrics restart their wealth index.
- **EW penalty candidates:** 0.0001, 0.00025 and 0.0005. Require total training net
  return to retain at least 80% of the current LP's positive return, and require
  worst-block daily volatility and drawdown magnitude to be no worse than the
  current LP. For a negative baseline, the return floor permits a 20% proportional
  deterioration instead. Among eligible candidates, choose the smallest
  worst-block drawdown; ties favor higher full-period net Sharpe, then smaller tau.
  If none qualifies, retain tau = 0 and explicitly record the fallback.
- **Lower leverage:** fixed at 1.2 before evaluation; no leverage grid search.
- **Validation:** freeze tau before 2 January–30 June 2026, 123 observations.
  Evaluate only the three specified versions plus EW. Forecasts may refit on
  observations already available before each decision; alpha and tau stay fixed.
- **Initialization and fees:** each phase starts independently at EW. First-day
  trades incur costs; later trades use drifted holdings. Net return uses the same
  first-order gross-minus-fees convention as the previous experiment, with gross
  holdings drift. Financing and short borrow fees remain outside this experiment.

Two diagnostic stress checks are fixed in advance. The fee check charges 10 bps
to the original validation holdings without reoptimizing. The forecast check uses
the first validation trading day of each month and five common, seeded shocks per
date: each asset forecast is perturbed uniformly within plus/minus one basis point.
Each strategy is resolved from its own original pretrade holdings; the shocks do
not accumulate into a separate backtest. Sensitivity is half the L1 difference
between the perturbed and original allocations. No stress result feeds back into
selection or changes the held-out validation configuration.

January–June has been used in earlier research. These are development validation
and sensitivity results, not a new independent final test or a guarantee about
future market regimes.


## Results

### Training selection

| Version | Net return | Daily Sharpe | Worst-block daily volatility | Worst-block drawdown magnitude | Eligible EW candidate |
|---|---:|---:|---:|---:|:---:|
| Current LP | 6.936% | 0.02637 | 1.807% | 21.343% | Control |
| Lower leverage (1.2) | 7.486% | 0.02785 | 1.793% | 21.661% | Control |
| Equal weight | 12.912% | 0.04239 | 1.813% | 22.324% | Control |
| EW candidate 0.0001 | 7.064% | 0.02670 | 1.807% | 21.293% | Yes |
| EW candidate 0.00025 | 7.664% | 0.02848 | 1.800% | 21.316% | Yes |
| EW candidate 0.0005 | 5.918% | 0.02366 | 1.794% | 21.532% | No |

The return floor was **5.549%**, based on the current LP's 6.936% training net
return. Tau **0.0001** was selected over the other eligible value, 0.00025, because
its worst-block drawdown was slightly smaller. The training drawdown improvement
over the current LP was only **0.050 percentage points**, so this selection should
not be interpreted as a large or statistically established advantage.

### Frozen validation results

| Version | Net return | Daily Sharpe | Daily volatility | Maximum drawdown | Average turnover | Average gross leverage |
|---|---:|---:|---:|---:|---:|---:|
| Current LP | 19.302% | 0.14598 | 1.019% | -9.654% | 2.434% | 1.50 |
| Lower leverage (1.2) | 17.229% | 0.13363 | 1.005% | -9.068% | 2.157% | 1.20 |
| EW anchor (tau = 0.0001) | 18.184% | 0.13827 | 1.020% | -9.935% | 2.389% | 1.50 |
| Equal weight | 16.885% | 0.13220 | 0.997% | -8.646% | 0.335% | 1.00 |

All metrics in this table include the same 5 bps one-way costs. Sharpe is daily,
uses sample standard deviation and a zero risk-free rate. Drawdown is measured
on compounded net wealth, not the uncompounded proxy constrained inside the LP.

### Stability checks

| Version | Net return at 10 bps | Mean holdings change under forecast shocks | Median change | Largest change |
|---|---:|---:|---:|---:|
| Current LP | 18.946% | 2.909% | 0.503% | 15.867% |
| Lower leverage (1.2) | 16.919% | 2.296% | 0.516% | 12.038% |
| EW anchor (tau = 0.0001) | 17.837% | 2.510% | 0.551% | 14.421% |
| Equal weight | 16.837% | — | — | — |

There are 30 one-step forecast probes per LP (90 in total). Holdings change is
half the L1 allocation difference, not a return or loss. Mean and maximum changes
fall for both modifications, but median changes are slightly higher than the
current LP. These sampled sensitivities do not demonstrate uniformly smoother
decisions or stability under every forecast perturbation.

| Validation interval | Current LP drawdown | Lower-leverage drawdown | EW-anchor drawdown |
|---|---:|---:|---:|
| 2026-01-02 to 2026-03-03 | -3.584% | -3.301% | -3.671% |
| 2026-03-04 to 2026-04-30 | -7.170% | -6.917% | -7.319% |
| 2026-05-01 to 2026-06-30 | -3.892% | -3.878% | -4.052% |

Block drawdowns restart their wealth reference within each interval. They need
not equal the drawdown of the continuous full validation path.

## Interpretation

**Lower leverage is the more promising defensive modification in this validation
period.** It retains about 89.3% of the current LP's net cumulative return, reduces
full-period drawdown magnitude by 0.586 percentage points, and lowers volatility
and drawdown in all three validation blocks. Mean forecast-shock sensitivity falls
by approximately 21.1%. At 10 bps fees it still returns 16.919%, although its
advantage over EW (16.837%) is small.

The improvement is modest and is not consistent across all historical conditions.
In the 2025 training period, lower leverage has slightly worse full-period drawdown
(21.661% versus 21.343%) and slightly higher overall daily volatility. Changing
a gross exposure cap changes the hedges and asset mix; it does not simply scale
down the original portfolio. The model remains fully invested on a net basis.

**The selected EW anchor does not improve realized risk in validation.** Although
it reduces average L1 distance to EW from 1.675 to 1.514 and lowers mean forecast
sensitivity, its validation net return falls to 18.184%, daily volatility rises
slightly, and drawdown worsens to 9.935%. Drawdown is worse in all three validation
blocks. Gross leverage remains at 1.5. Closer holdings to a benchmark do not
guarantee a better realized return path; the slight training-period advantage did
not carry over to this validation period. No alternative tau was selected after
seeing these outcomes.

**Practical conclusion:** retain the current LP as the return-oriented reference
and the 1.2-leverage version as the defensive candidate. The present evidence
does not support replacing either with the selected EW anchor. The favorable
validation and shock results for lower leverage are exploratory evidence, not
proof of superior performance in unknown regimes. No July test was reopened.

## Implementation, verification and artifacts

- `LPObjective.ew_penalty` adds a primary full-L1 distance-to-EW penalty. The same
  auxiliary-variable builder handles trading and EW penalties independently.
- The existing forecast, rollout and metrics functions are reused; rollout now
  supports per-model constraints. `lp_stability_workflow.py` holds the selection
  and diagnostic rules; `lasso_forecast_lp_stability.py` is the experiment entrypoint.
- All **171** repository tests passed. Independent review reported no actionable
  issue, including 25 randomized combined-objective checks against feasible grids.
- A saved-result audit verified **1,619** LP allocations, **373** forecast date
  boundaries, 90 recorded sensitivity probes, fees, objectives, full/block metrics
  and the frozen choice. Current-LP validation holdings and forecasts exactly
  match the preceding cost-aware LP run.

```bash
python -m experiments.lasso_forecast_lp_stability \
  --forecast-reference results/unified_252/20260922T122949.119118Z/lp
```

Saved run: [results/lasso_forecast_lp_stability/20260924T075554Z](../../results/lasso_forecast_lp_stability/20260924T075554Z/).
Training candidate paths, all block metrics, gross/net validation returns, holdings,
forecasts, stress diagnostics, frozen settings, code fingerprints and verification
logs are retained separately. Previous experiment results remain unchanged.
