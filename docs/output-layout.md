# Experiment output locations

## Defaults and compatibility

New module commands use `results/<module_name>/<UTC_timestamp>/`, rooted at the repository regardless of shell cwd. Example: `python -m experiments.lasso_training_windows --presets 252d 3y --start 2026-01-02` creates a separate run under `results/lasso_training_windows/`.

`--output /chosen/run` always means that exact run root. `--layout categorized` or `--layout flat` selects file placement. Nonempty destinations are rejected before fitting. Automatic timestamps include microseconds; results are never merged into a previous run.

Old root commands keep flat files by default so existing explicit-output commands and scripts still work. `run_example.py` and `audit_missing_values.py` without `--output` retain console-only behavior. Previously standalone Elastic Net/PCA/market-beta commands now save into named run directories; market beta no longer overwrites `market_beta_results.csv` in cwd.

## Categorized run

```text
results/lasso_forecast_lp/20260922T120000.123456Z/
  artifact_manifest.json
  metrics/       # summary.csv, validation_summary.csv, comparison.csv
  returns/       # realized daily returns and candidate component returns
  weights/       # asset holdings, by date and strategy
  predictions/   # per-asset forecasts
  diagnostics/   # training/risk/solver diagnostics and data-audit counts
  config/        # run_metadata.json, frozen_decisions.json, selection.json, audit.json
  search/        # alpha calibration, search scores, screen/validation candidate files
  figures/       # plots, when produced
  logs/          # logs, when produced
```

Only categories actually written by a run are created. This refactor does not add automatic charts or console-log capture. Search candidate files remain grouped by phase under `search/screen/` and `search/validation/`. CSV filenames and their numerical columns remain unchanged. `market_beta_results.csv` is a metrics artifact.

The root manifest records layout version and maps each logical filename to its stored relative path. Failed runs may leave diagnostic files but do not receive a completed manifest. Keep metadata and frozen decisions together with the result directory.

Read files without depending on their physical placement:

```python
from portfolio.evaluation.artifact_store import artifact_path
import pandas as pd

run = "results/lasso_forecast_lp/my_run"
summary = pd.read_csv(artifact_path(run, "validation_summary.csv"), index_col=0)
```

`artifact_path` supports both completed categorized runs and historical flat runs. LP `--reference` uses the same lookup, then verifies data, configuration, code and reconstructed returns.

## Retained results and frozen configurations

The current 252-observation comparison is kept only in `results/unified_252/20260922T122949.119118Z/`. Its entire run directory and published report/CSV/figure remain unchanged.

Three-year training/forecast/risk studies are grouped under `results/window_studies_3y/`. This includes the historical 252-day forecast variants that used a **three-year risk window**. These are different experiments from the all-252-observation comparison. Pure 252-day regression controls in the historical bundles were removed; their current January–June validation counterparts are available in the unified baseline family.

Each retained historical directory remains flat and now has an `artifact_manifest.json` plus `retention_metadata.json`. The latter records removed files, filtered summary rows, replacement locations and the external recovery archive. Original `run_metadata.json` and frozen configurations retain their original bytes and describe the source run. Historical audit scripts/logs describe that original run; the cleanup audit separately verifies the retained subset. Use `artifact_path` to read either layout.

The old growth/search runs and incomplete outputs have been removed from the active project. Missing-value audits are retained separately. See [the retention inventory](results-retention.md) for exact directories and missing historical results.

Source fingerprints cover nested implementations with distinct repository-relative identities. Historical frozen artifacts remain tied to their original source revision; moving or pruning exported files does not authorize a different code version. Use the original Git revision to reproduce an old frozen run. Never edit its hashes merely to make a new checkout accept it.
