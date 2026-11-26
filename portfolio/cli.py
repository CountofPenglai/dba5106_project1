"""Run the report validation or reproduce the preregistered holdout."""
import argparse
from pathlib import Path

from portfolio.config import COURSE_WINDOW, VALIDATION
from portfolio.evaluation.artifact_store import default_output, fresh_run
from portfolio.evaluation.validation import validate
from portfolio.model_catalog import COURSE_MODELS, MODELS
from portfolio.paths import DEFAULT_CSV
from portfolio.evaluation.final_test import evaluate


def main(argv=None, *, baselines_only=False):
    catalog = COURSE_MODELS if baselines_only else MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period', choices=('validation',) if baselines_only else ('validation','final_test'),
                        default='validation')
    parser.add_argument('--models', nargs='+', choices=tuple(catalog))
    parser.add_argument('--window', type=int, choices=(COURSE_WINDOW,), default=COURSE_WINDOW)
    parser.add_argument('--csv', type=Path, help='Location of the original validation release')
    parser.add_argument('--start', help='First validation date')
    parser.add_argument('--end', help='Last validation date')
    parser.add_argument('--output', type=Path, help='A new or empty directory')
    parser.add_argument('--list', action='store_true', help='List validation portfolios')
    args = parser.parse_args(argv)
    if args.list:
        for spec in catalog.values():
            print(f'{spec.name}: {spec.title}')
        return
    output = args.output or default_output(args.period)
    try:
        if args.period == 'final_test':
            if args.models or args.csv or args.start or args.end:
                parser.error('The final test uses its frozen portfolios, data releases and dates')
            evaluate(output)
        else:
            models = [catalog[n] for n in args.models] if args.models else list(catalog.values())
            with fresh_run(output) as stage:
                summary = validate(stage, args.csv or DEFAULT_CSV, models,
                                   args.start or VALIDATION[0], args.end or VALIDATION[1])
            print(summary[['mean', 'volatility', 'sharpe']].to_string())
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f'Results: {output}')
