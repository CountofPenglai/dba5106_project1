"""Reuse a 252-day LASSO forecaster; tune the LP MAD penalty before validation."""

import argparse
import json
from pathlib import Path

from portfolio.data.returns_loader import load_returns
from portfolio.engine.lp_risk_tuning_workflow import LPRiskTuningConfig, run_lp_risk_tuning
from portfolio.evaluation.artifact_store import artifact_path, configure_output
from portfolio.evaluation.provenance import code_hashes, file_digest, frame_digest
from portfolio.evaluation.result_export import export_result
from portfolio.paths import DEFAULT_CSV


def reuse_forecast_alpha(reference, returns, config):
    """Verify the original forecast settings/data; LP source may intentionally change."""
    frozen = json.loads(artifact_path(reference, 'frozen_decisions.json').read_text())
    if frozen.get('kind') != 'direct_asset_lp' or frozen.get('schema_version') != 1:
        raise ValueError('Reference must be an existing direct-LP experiment')
    expected_window = dict(window=dict(observations=config.window, years=None, min_observations=2),
                           half_life=None)
    if frozen.get('forecast_configs', {}).get('252d') != expected_window:
        raise ValueError('Reference forecast window differs from the requested window')
    for field in ('history_start', 'calibration_end', 'validation_end'):
        if frozen['config'][field] != getattr(config, field):
            raise ValueError('Reference forecast dates differ: '+field)
    if frozen['assets'] != list(returns.columns):
        raise ValueError('Reference forecast asset order differs')
    if frame_digest(returns) != frozen['development_data_digest']:
        raise ValueError('Reference development data differ')
    current = code_hashes()
    forecast_sources = ('portfolio/models/lasso_return_forecast.py', 'portfolio/config.py',
        'portfolio/data/history_windows.py', 'portfolio/data/preprocessing.py',
        'portfolio/data/returns_loader.py')
    for source in forecast_sources:
        if frozen['code_sha256'].get(source) != current[source]:
            raise ValueError('Reference forecast implementation differs: '+source)
    alpha = float(frozen['alphas']['252d'])
    if not 0 < alpha < float('inf'):
        raise ValueError('Reference alpha must be finite and positive')
    return alpha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--forecast-reference', type=Path, required=True,
                        help='Existing 252-day LP result directory with the frozen forecast alpha')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--layout', choices=('categorized', 'flat'), default='categorized')
    args = parser.parse_args(argv)
    configure_output(parser, args, 'lasso_forecast_lp_risk')
    config = LPRiskTuningConfig()
    config.validate()
    returns = load_returns(args.csv, start=config.history_start, end=config.validation_end)
    alpha = reuse_forecast_alpha(args.forecast_reference, returns, config)
    print(f'Reusing LASSO alpha={alpha:.9g}; 252-day forecast/risk windows; July excluded.', flush=True)
    result = run_lp_risk_tuning(returns, forecast_alpha=alpha, config=config,
                               progress=lambda message: print(message, flush=True))
    reference_file = artifact_path(args.forecast_reference, 'frozen_decisions.json')
    source = dict(path=str(args.forecast_reference.resolve()), sha256=file_digest(reference_file))
    result.metadata['forecast_reference'] = source
    result.frozen['forecast_reference'] = source
    export_result(result, args.output, source_csv=args.csv, layout=args.layout)
    print(result.frames['validation_summary'].to_string(), flush=True)
    print('Saved artifacts to '+str(args.output.resolve()), flush=True)
    return result


if __name__ == '__main__':
    main()
