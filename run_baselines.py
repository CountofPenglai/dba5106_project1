"""Run EW, MinVar, coefficient-constrained and penalty LASSO/Ridge baselines.

Examples:
    python run_baselines.py --window 126
    python run_baselines.py --window 252 --lasso-coefficient-budget 1 --ridge-coefficient-budget 1
"""
from experiments.regularized_baselines import main

if __name__ == "__main__":
    main(default_layout="categorized", baselines_only=True)
