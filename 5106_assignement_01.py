"""Compatibility launcher for the archived original assignment example.

The original source and its historical data assumptions are preserved verbatim.
Use experiments.regularized_baselines for the maintained framework.
"""
from runpy import run_module

if __name__ == "__main__":
    run_module("examples.original_assignment", run_name="__main__")
