# Direct LP core implementation report

Scope: `portfolio/linear_decision.py` and `tests/test_linear_decision.py` only.
No commits or remote operations were made. The only additional file written by
this implementer is this report.

## Interface and formulation

`LPConstraints` defaults to gross leverage 1.5, absolute position 0.05, EW MAD
ratio 1, EW uncompounded drawdown ratio 1, and historical wealth floor `1e-6`.
`build_lp_context`, `check_lp_weights`, and `solve_linear_portfolio` implement
the plan interface. Invalid structural limits and nonfinite/malformed inputs
raise explicitly; nonoptimal, infeasible, nonfinite, or independently
infeasible solver results raise `LinearPortfolioError`. There is no fallback
portfolio.

For `T` historical rows and `N` assets, the primary sparse LP uses weights
`w`, absolute-weight auxiliaries `u`, centered-return absolute auxiliaries `z`,
and running-peak auxiliaries `p`. It maximizes `mu @ w` subject to:

- `sum(w) = 1`, `-cap <= w_i <= cap`;
- `u >= w`, `u >= -w`, `sum(u) <= leverage_limit`;
- `z_t >= A_t @ w`, `z_t >= -A_t @ w`,
  `sum(z) / T <= mad_limit`, where `A = R - column_mean(R)`;
- `p_t >= 0`, `p_t >= C_t @ w`, `p_t >= p_(t-1)`, and
  `p_t - C_t @ w <= drawdown_limit`, where `C = cumsum(R, axis=0)`;
- `1 + R_t @ w >= wealth_floor` for every historical row.

The lower bound `p_t >= 0` makes the initial cumulative P&L zero part of every
running peak. Thus an initial loss and an all-negative path are counted. Weight
bounds are explicitly negative/positive rather than relying on `linprog`'s
default nonnegative bounds.

Rows are normalized before HiGHS. Positive tiny MAD and drawdown limits use
geometric auxiliary-variable scaling, balancing the cap and underlying return
coefficient magnitudes. Exactly zero MAD adds the equivalent equalities
`A @ w = 0`; exactly zero drawdown adds the equivalent historical constraints
`R_t @ w >= 0`. Equality rows are normalized too. Independent feasibility
checks use a cap-relative tolerance plus a floating-point scale term, without
dividing by a zero cap.

The secondary sparse LP adds `q >= abs(w - EW)`, minimizes `sum(q)`, and retains
the primary predicted return within the specified absolute daily tolerance
`1e-10`. Diagnostics contain Python/JSON-compatible finite values for both
solver stages, primary/final predicted return and gap, and independently
recomputed budget, gross, maximum position, MAD, drawdown, and minimum wealth.

## Test-first evidence

The initial test command was run before the module existed:

```text
.venv/bin/python -m unittest tests.test_linear_decision -v
ModuleNotFoundError: No module named 'portfolio.linear_decision'
FAILED (errors=1)
```

A deliberately nonfunctional API stub was then added so behavior tests could
fail at their assertions/calls. The same command produced 11 failed tests
(`NotImplementedError` plus missing validation). Only after that observed red
run was the LP implemented.

Conditioning regressions also followed red-green cycles:

- A positive EW MAD cap of about `1e-12` initially returned actual MAD about
  `2e-12`; independent verification rejected it. Geometric MAD scaling made the
  regression pass.
- The reviewer supplied `[[.02, 0], [0, .02]] * 1e-9`, whose zero-MAD
  equality coefficients were initially small enough for HiGHS to discard.
  The test observed weights `[1, 0]` and independent rejection before equality
  normalization made it pass at EW.
- An analogous positive EW drawdown around `2e-21`, from returns around
  `2e-11`, initially returned drawdown around `3e-11` and was independently
  rejected. Geometric running-peak/cumulative scaling made it pass.

Final focused verification, after all fixes and before freezing source:

```text
.venv/bin/python -m unittest tests.test_linear_decision -v
Ran 12 tests in 0.052s
OK

git diff --check -- portfolio/linear_decision.py tests/test_linear_decision.py
.venv/bin/python -m py_compile portfolio/linear_decision.py tests/test_linear_decision.py
# both exited 0 with no output
```

The runner/CLI integration suite was also run against the core:

```text
.venv/bin/python -m unittest tests.test_direct_lp tests.test_direct_lp_cli -v
Ran 7 tests in 1.890s
OK
```

Root owns and runs the final repository-wide suite.

## Independent mathematical checks

The main two-asset fixture has returns `[-.10, 0]` and `[.10, 0]`. Write the
first weight as `x` and the second as `1-x`. MAD is `.10 * abs(x)` and
uncompounded drawdown is `.10 * max(x, 0)`. EW sets both limits to `.05`, so a
forecast favoring the first asset is maximized at `x=.5`. The test independently
enumerates 120,001 values of `x`, directly recomputes gross, position, MAD,
running-peak drawdown, and wealth, and compares the best feasible objective to
the LP; it does not call the production constraint builder for its expected
result.

A constant-positive-return fixture makes both risk caps exactly zero while
leaving return independent of allocation. With budget one and gross limit 1.5,
the analytic maximum for a forecast favoring asset one is `[1.25, -.25]`; this
verifies short support and explicit negative bounds. Other tests isolate the
position cap, MAD cap, drawdown cap, initial-zero drawdown behavior, equal-
forecast EW tie, infeasibility, malformed data/configuration, and tiny/zero
risk caps.

The read-only reviewer additionally compared the implementation against an
independent all-pairs drawdown formulation on 40 random small cases. All passed;
the largest primary-to-final predicted-return gap was `1.00000003e-10`, the
expected numerical realization of the documented `1e-10` secondary tolerance.

## Actual 98-asset timing probe

The probe loaded the real French 100-portfolio source, removed the two standard
excluded assets through `portfolio.data.load_returns`, selected the three-year
history preceding 2026-01-02, and used the historical asset means as `mu`.
Environment: SciPy 1.18.1, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

```text
prediction date:       2026-01-02
history:               2023-01-03 .. 2025-12-31
observations/assets:   752 / 98
A_ub shape/nonzeros:   (4709, 1700) / 373,992
context build:         0.007459917 seconds
primary+secondary LP:  0.882908041 seconds
total:                 0.890367958 seconds
gross:                 1.4999999999999998
MAD / limit:           .009164177167443487 / .009164177167443482
drawdown / limit:      .2533379312766097 / .26780204081632636
secondary gap:         9.999999981726015e-11
independent feasible:  true
```

At that one-window rate, 1,916 calls to `solve_linear_portfolio` imply about
28.2 minutes for the LP portion. This is an estimate from one actual window,
not a full-run benchmark; solve time varies with the daily active set and
forecast objective.
