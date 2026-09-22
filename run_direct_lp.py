#!/usr/bin/env python3
"""Run direct-asset LP allocation from causal LASSO forecasts."""
import argparse
import json
from pathlib import Path

from portfolio.data import load_returns
from portfolio.direct_lp import run_direct_lp, run_frozen_lp
from portfolio.linear_decision import LPConstraints
from portfolio.lp_reference import add_reference_comparison
from portfolio.two_stage_io import default_code_paths, export_result


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT/'100_Portfolios_10x10_Daily.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Verified previous two-stage result directory for comparison')
    parser.add_argument('--frozen', type=Path, help='Explicit frozen August final-test mode')
    parser.add_argument('--include-development', action='store_true',
                        help='Also backtest August 2024–December 2025; default evaluates validation only')
    parser.add_argument('--position-limit', type=float, help='Absolute per-asset cap; development default .05')
    parser.add_argument('--mad-ratio', type=float, help='MAD limit relative to historical EW; default 1')
    parser.add_argument('--drawdown-ratio', type=float, help='Uncompounded absolute drawdown limit relative to EW; default 1')
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError('--output must be empty or new')
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
        export_result(result, args.output, source_csv=args.csv,
                      code_paths=default_code_paths()+[Path(__file__).resolve()])
        print('Saved artifacts to '+str(args.output.resolve()),flush=True)
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
