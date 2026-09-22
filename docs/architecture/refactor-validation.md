# Refactor verification

Baseline revision: `0934d54e1b9f6c64d877921271e13ba925c6b449`.

| Check | Result |
|---|---|
| Original suite before refactoring | 118 tests passed |
| Expanded suite | 138 tests passed |
| Synthetic before/after snapshots | 69 numeric output groups; maximum absolute difference 0 |
| Actual dataset before/after fits | 6 fits across 98 assets; maximum absolute difference 0 |
| Exploratory model tests | 11 passed, including real categorized CLI executions for Elastic Net, PCA and market beta |
| Independent reviews | Scoped exploratory review and whole-framework review approved; category follow-up approved |

Synthetic comparisons include fixed and cross-validated LASSO/Ridge, constrained/growth/exposure fits, asset forecasts, learned portfolio mixtures and direct LP output frames. Actual-data comparisons use histories ending 2026-06-29 at 252-observation and three-year windows, each with plain LASSO, constrained LASSO and reward/exposure LASSO. Comparisons cover weights, losses and objectives; they do not claim a new full-period performance experiment.

The expanded suite checks flat and categorized CLI output equivalence, source fingerprint coverage/tampering/checkout portability, categorized reference readback, nonempty-output rejection, named artifact placement and missing-row preservation. Existing tests continue to check temporal exclusion, original CV/solver behavior, constraints, early stopping and frozen modes. The two growth-search mocks now target the extracted evaluator module where the dependency is looked up; their assertions are unchanged.

Re-run the supported suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -v
```

This verification is for structural/behavioral equivalence. Historical method comparisons and their reported performance remain in the existing experiment reports.
