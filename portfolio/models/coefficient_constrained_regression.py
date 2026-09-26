"""Minimum-MSE linear regression with explicit L1 or squared-L2 budgets.

The constraint acts on beta only. There is no coefficient penalty in the loss,
no constraint on the intercept, and no post-fit coefficient clipping.
"""
import numpy as np

from portfolio.constraints import CoefficientConstraint
from portfolio.data.observation_weights import normalize_sample_weights
from portfolio.engine.convex_solver import solve_checked
from portfolio.models.min_variance_regression import LinearModel, _regression_inputs


def fit_coefficient_constrained(X, y, *, constraint: CoefficientConstraint, sample_weight=None):
    """Minimize weighted MSE subject to sum(abs(beta)) or sum(beta**2) <= budget."""
    import cvxpy as cp

    X, y = _regression_inputs(X, y)
    q = normalize_sample_weights(sample_weight, len(y))
    if np.count_nonzero(q) < 2:
        raise ValueError("At least two observations need positive sample weight")
    mean_X, mean_y = q @ X, float(q @ y)
    centered_X, centered_y = X - mean_X, y - mean_y

    def check(coefficients):
        measure = constraint.measure(coefficients)
        if measure > constraint.budget + 1e-8 * (1 + constraint.budget):
            raise ValueError("Coefficient budget exceeded")

    if constraint.budget == 0:
        # Both nonnegative norms vanish only at beta=0; no numerical solve needed.
        coefficients = np.zeros(X.shape[1])
        solver = dict(solver="analytic", solver_status="analytic", solver_attempts="zero budget")
    else:
        beta = cp.Variable(X.shape[1])
        # A positive scalar leaves the minimizer unchanged and stabilizes small
        # decimal-return objectives. No penalty or extra shrinkage is introduced.
        scale = max(float(q @ np.square(centered_y)), 1e-12)
        residual = cp.multiply(np.sqrt(q / scale), centered_y - centered_X @ beta)
        measure = cp.norm1(beta) if constraint.kind == "lasso" else cp.sum_squares(beta)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(residual)), [measure <= constraint.budget])
        coefficients, solver = solve_checked(problem, beta, check,
                                              error_label="Coefficient-constrained regression failed")

    intercept = float(mean_y - mean_X @ coefficients)
    mse = float(q @ np.square(y - X @ coefficients - intercept))
    diagnostics = {
        "constraint_kind": constraint.kind,
        "coefficient_budget": float(constraint.budget),
        "coefficient_measure": constraint.measure(coefficients),
        "coefficient_l1": float(np.abs(coefficients).sum()),
        "coefficient_l2_squared": float(coefficients @ coefficients),
        "mse": mse, "objective": mse, **solver,
    }
    return LinearModel(coefficients, intercept), diagnostics
