"""A single categorized output layout; existing runs are never overwritten."""
from datetime import datetime, timezone
import json
from pathlib import Path
from portfolio.paths import PROJECT_ROOT


def ensure_fresh_output(root):
    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError('output must be an empty or new directory')


def default_output(suite):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    return PROJECT_ROOT/'results/runs'/suite/stamp


def save_run(run, output, metadata):
    """Publish metrics and daily streams, with sufficient metadata to reproduce."""
    output = Path(output)
    ensure_fresh_output(output)
    for group in ('metrics', 'returns', 'weights', 'diagnostics', 'config', 'predictions'):
        (output/group).mkdir(parents=True, exist_ok=True)
    run.summary.to_csv(output/'metrics/summary.csv')
    for name in run.returns:
        run.returns[name].to_csv(output/'returns'/f'{name}.csv')
        run.weights[name].to_csv(output/'weights'/f'{name}.csv')
        run.diagnostics[name].to_csv(output/'diagnostics'/f'{name}.csv')
    if run.predictions is not None:
        run.predictions.to_csv(output/'predictions/lasso_forecast.csv')
        run.forecast_diagnostics.to_csv(output/'diagnostics/lasso_forecast.csv')
    (output/'config/run_metadata.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
