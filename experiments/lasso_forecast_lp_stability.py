"""Compare cost-aware LP, lower leverage and training-selected EW anchoring."""

import argparse
from pathlib import Path

from experiments.lasso_forecast_lp_risk import reuse_forecast_alpha
from portfolio.data.returns_loader import load_returns
from portfolio.engine.lp_stability_workflow import LPStabilityConfig, run_lp_stability
from portfolio.evaluation.artifact_store import artifact_path, configure_output
from portfolio.evaluation.provenance import file_digest
from portfolio.evaluation.result_export import export_result
from portfolio.paths import DEFAULT_CSV


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--forecast-reference', type=Path, required=True,
                        help='Original 252-day direct-LP result directory')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--layout', choices=('categorized', 'flat'), default='categorized')
    args = parser.parse_args(argv)
    configure_output(parser, args, 'lasso_forecast_lp_stability')
    config = LPStabilityConfig()
    config.validate()
    periods = config.periods
    returns = load_returns(args.csv, start=periods.history_start, end=periods.validation_end)
    alpha = reuse_forecast_alpha(args.forecast_reference, returns, periods)
    print(f'252-day LP stability comparison; reused forecast alpha={alpha:.9g}; July excluded.', flush=True)
    result = run_lp_stability(returns, forecast_alpha=alpha, config=config,
                             progress=lambda message: print(message, flush=True))
    reference = dict(path=str(args.forecast_reference.resolve()),
                    sha256=file_digest(artifact_path(args.forecast_reference, 'frozen_decisions.json')))
    result.metadata['forecast_reference'] = reference
    result.frozen['forecast_reference'] = reference
    export_result(result, args.output, source_csv=args.csv, layout=args.layout)
    print(result.frames['validation_summary'].to_string(), flush=True)
    print('Saved artifacts to '+str(args.output.resolve()), flush=True)
    return result


if __name__ == '__main__':
    main()
