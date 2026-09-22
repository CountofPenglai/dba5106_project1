"""Verify and reconstruct earlier experiment results before comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .two_stage import portfolio_metrics
from .two_stage_io import file_digest


_REFERENCE_MODELS = [
    'Baseline_252d', 'Baseline_252d_Constrained', 'Baseline_3y', 'Baseline_3y_Constrained',
    'LASSO_252d_U', 'LASSO_3y_U', 'LASSO_252d_Sharpe', 'LASSO_252d_ReturnVolatility',
    'LASSO_3y_Sharpe', 'LASSO_3y_ReturnVolatility',
]
_MODULES = ('data', 'config', 'splits', 'forecasting', 'model', 'weights', 'losses',
            'constraints', 'optimization', 'training', 'regularization', 'evaluation',
            'candidates', 'decision', 'convex', 'two_stage')


def add_reference_comparison(result, reference, returns, *, source_csv):
    """Add historical baselines only after provenance and arithmetic checks."""
    reference = Path(reference)
    metadata_path = reference/'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    config = result.metadata['config']
    if (metadata.get('phase') != 'development' or metadata.get('final_test_evaluated') is not False
            or metadata.get('data_sha256') != file_digest(source_csv)
            or metadata.get('development_data_digest') != result.frozen['development_data_digest']
            or metadata.get('assets') != list(returns.columns)):
        raise ValueError('Reference data/provenance do not match this development experiment')
    shared_fields = {name:name for name in ('history_start', 'calibration_end', 'validation_start',
                                           'validation_end', 'alphas', 'n_splits', 'test_size')}
    shared_fields.update(development_start='decision_start', development_end='decision_end')
    normalize = lambda value: json.loads(json.dumps(value))
    for field, prior_field in shared_fields.items():
        if normalize(metadata['config'][prior_field]) != normalize(config[field]):
            raise ValueError('Reference experiment configuration differs: '+field)
    for field in ('forecast_configs', 'alphas', 'risk_window'):
        if normalize(metadata[field]) != normalize(result.metadata[field]):
            raise ValueError('Reference forecast/risk configuration differs: '+field)
    hashes = {Path(path).name:value for path,value in metadata['code_sha256'].items()}
    for module in _MODULES:
        filename = module+'.py'
        if hashes.get(filename) != file_digest(Path(__file__).parent/filename):
            raise ValueError('Reference source code differs: '+filename)
    table = pd.read_csv(reference/'comparison.csv', index_col=0)
    if table.index.has_duplicates:
        raise ValueError('Duplicate reference model names')
    actual = returns.loc[config['validation_start']:config['validation_end']]
    rows = []
    imported = {}
    for name in _REFERENCE_MODELS:
        if name not in table.index: continue
        w = pd.read_csv(reference/(name+'_weights.csv'), index_col=0, parse_dates=True)
        r = pd.read_csv(reference/(name+'_returns.csv'), index_col=0, parse_dates=True)
        if (not w.index.equals(actual.index) or not r.index.equals(actual.index)):
            # Older mixture and U artifacts include their development segment.
            w = w.loc[config['validation_start']:config['validation_end']]
            r = r.loc[config['validation_start']:config['validation_end']]
        if (not w.index.equals(actual.index) or not r.index.equals(actual.index)
                or list(w.columns) != list(actual.columns) or r.shape[1] != 1
                or not np.isfinite(w.to_numpy()).all() or not np.isfinite(r.to_numpy()).all()):
            raise ValueError('Reference dates/assets/values do not align: '+name)
        realized = np.sum(w.to_numpy()*actual.to_numpy(), axis=1)
        if not np.allclose(realized, r.iloc[:,0].to_numpy(), atol=1e-12, rtol=0):
            raise ValueError('Reference returns do not reconstruct: '+name)
        metrics = portfolio_metrics(w, actual)
        for key,value in metrics.items():
            stored = table.loc[name,key]
            if isinstance(value, str):
                valid = str(stored) == value
            else:
                valid = np.isclose(float(stored), value, atol=1e-12, rtol=1e-8)
            if not valid:
                raise ValueError('Reference metric differs: '+name+'/'+key)
        rows.append(dict(model=name, role='previous_experiment', **metrics))
        imported['reference_'+name+'_weights'] = w
        imported['reference_'+name+'_returns'] = r
    if not rows:
        raise ValueError('Reference has no supported baseline strategies')
    comparison = pd.DataFrame(rows).set_index('model')
    common = [c for c in comparison.columns if c in result.frames['comparison'].columns]
    result.frames['comparison'] = pd.concat([result.frames['comparison'][common], comparison[common]])
    result.frames.update(imported)
    result.metadata['reference_comparison'] = dict(directory=str(reference.resolve()), strategies=len(rows),
        metadata_sha256=file_digest(metadata_path), returns_reconstructed=True, source_code_verified=True)
