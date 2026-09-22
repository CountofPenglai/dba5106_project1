#!/usr/bin/env python3
"""Run direct-asset LP allocation from causal LASSO forecasts."""

from portfolio.paths import PROJECT_ROOT
from portfolio.evaluation.artifact_store import configure_output
from portfolio.evaluation.provenance import file_digest
import argparse
import json
from pathlib import Path

from portfolio.data.returns_loader import load_returns
from portfolio.engine.forecast_lp_workflow import run_direct_lp, run_frozen_lp
from portfolio.decisions.linear_program_allocation import LPConstraints
from portfolio.evaluation.reference_comparison import add_reference_comparison
from portfolio.evaluation.result_export import default_code_paths, export_result


ROOT = PROJECT_ROOT


def main(argv=None, *, default_layout="flat"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT/'100_Portfolios_10x10_Daily.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path, help='Verified previous two-stage result directory for comparison')
    parser.add_argument('--frozen', type=Path, help='Explicit frozen August final-test mode')
    parser.add_argument('--include-development', action='store_true',
                        help='Also backtest August 2024–December 2025; default evaluates validation only')
    parser.add_argument('--position-limit', type=float, help='Absolute per-asset cap; development default .05')
    parser.add_argument('--mad-ratio', type=float, help='MAD limit relative to historical EW; default 1')
    parser.add_argument('--drawdown-ratio', type=float, help='Uncompounded absolute drawdown limit relative to EW; default 1')
    parser.add_argument("--layout", choices=("flat", "categorized"), default=default_layout,
                        help="Categorized artifact folders or historical flat filenames")
    args = parser.parse_args(argv)
    configure_output(parser, args, "lasso_forecast_lp", optional_flat=False)
    try:
        if not args.csv.is_file():
            raise ValueError('Source CSV is unavailable')
        overrides = {key:getattr(args,key) for key in ('position_limit','mad_ratio','drawdown_ratio') if getattr(args,key) is not None}
        if args.frozen is not None:
            if overrides or args.reference is not None or args.include_development:
                raise ValueError('Frozen test cannot override limits or import development comparisons')
            returns = load_returns(args.csv, start='2021-08-01', end='2026-08-30')
            if returns.empty or str(returns.index[-1].date()) < '2026-08-28':
                raise ValueError('Complete August final-test data are unavailable')
            frozen = json.loads(args.frozen.read_text())
            if not isinstance(frozen, dict):
                raise ValueError('Frozen configuration must be an object')
            result = run_frozen_lp(returns, frozen, progress=lambda x:print(x,flush=True))
        else:
            constraints = LPConstraints(**overrides)
            returns = load_returns(args.csv, start='2021-08-01', end='2026-06-30')
            scope = 'development and validation' if args.include_development else 'validation only'
            print(f'Direct LP {scope}; August excluded; no decision-model training.',flush=True)
            result = run_direct_lp(returns, constraints=constraints, include_development=args.include_development,
                                   progress=lambda x:print(x,flush=True))
            if args.reference is not None:
                add_reference_comparison(result, args.reference, returns, source_csv=args.csv)
        export_result(result, args.output, source_csv=args.csv, layout=args.layout)
        print('Saved artifacts to '+str(args.output.resolve()),flush=True)
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main(default_layout="categorized")
