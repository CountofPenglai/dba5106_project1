# Tony's Portfolio Models

Tony's four focused extensions are classified with the course's **Data + Loss + Structure + Constraint** framework. All four keep the same 252-trading-day causal rolling design and reserve August for final testing.

| Model label | Framework layer | Validation conclusion | Status tags |
|---|---|---|---|
| **Tony's Turnover Penalty Model** | Loss | Turnover falls, but no-cost Sharpe decreases | `loss` `not-selected` |
| **Tony's Size-BM Smoothness Model** | Structure / Loss | Small smoothing effect; Sharpe change is immaterial | `structure` `loss` `candidate-only` |
| **Tony's Ledoit-Wolf Shrinkage Model** | Data / Risk estimation | Turnover falls slightly, but Sharpe decreases | `data` `risk-estimation` `not-selected` |
| **Tony's Concentration Constraint Model** | Constraint | Sharpe rises from 0.1472 to 0.1561 and effective assets rise from 3.46 to 12.03 | `constraint` `recommended-candidate` |

## Tony's Turnover Penalty Model

- Entry point: [`run_turnover.py`](run_turnover.py)
- Tests: [`tests/test_turnover.py`](tests/test_turnover.py), [`tests/test_turnover_cli.py`](tests/test_turnover_cli.py)
- Report and figure: [`turnover-252.md`](turnover-252.md), [`turnover-252.png`](turnover-252.png)
- Tags: `loss`, `not-selected`

## Tony's Size-BM Smoothness Model

- Entry point: [`run_smoothness.py`](run_smoothness.py)
- Tests: [`tests/test_smoothness.py`](tests/test_smoothness.py), [`tests/test_smoothness_cli.py`](tests/test_smoothness_cli.py)
- Report and figure: [`smoothness-252.md`](smoothness-252.md), [`smoothness-252.png`](smoothness-252.png)
- Tags: `structure`, `loss`, `candidate-only`

## Tony's Ledoit-Wolf Shrinkage Model

- Entry point: [`run_shrinkage.py`](run_shrinkage.py)
- Tests: [`tests/test_shrinkage.py`](tests/test_shrinkage.py), [`tests/test_shrinkage_cli.py`](tests/test_shrinkage_cli.py)
- Report and figure: [`shrinkage-252.md`](shrinkage-252.md), [`shrinkage-252.png`](shrinkage-252.png)
- Tags: `data`, `risk-estimation`, `not-selected`

## Tony's Concentration Constraint Model

- Entry point: [`run_concentration.py`](run_concentration.py)
- Tests: [`tests/test_concentration.py`](tests/test_concentration.py), [`tests/test_concentration_cli.py`](tests/test_concentration_cli.py)
- Report and figure: [`concentration-252.md`](concentration-252.md), [`concentration-252.svg`](concentration-252.svg)
- Tags: `constraint`, `recommended-candidate`

All status labels summarize development/validation evidence only. They are not final-test claims.
