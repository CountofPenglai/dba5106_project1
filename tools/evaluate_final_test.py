"""Reproduce the frozen holdout via the shared evaluation pipeline."""
import argparse
from pathlib import Path

from portfolio.evaluation.final_test import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    evaluate(parser.parse_args().output)
