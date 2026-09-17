"""Convex fitting for the linear portfolio model with log-growth rewards."""

import numpy as np
import warnings

from .constraints import PortfolioConstraints
from .losses import LogGrowthLoss
from .model import LinearModel
from .regularization import Regularization
from .weights import normalize_sample_weights, validate_portfolio_weights


def fit_growth(problem, *, regularization=Regularization("lasso", 1e-5),
               loss=LogGrowthLoss(), constraints=PortfolioConstraints(), sample_weight=None):
    """Fit beta in w=EW-N@beta with a genuine, constrained convex objective.

    The free intercept is eliminated analytically: b = mean(R@w). The reward
    uses R@w = y-X@beta, so it cannot be absorbed by that intercept. Keep the
    objective in decimal-return units; large multipliers destabilize the
    exponential-cone solve for otherwise feasible windows.
    """
    import cvxpy as cp

    X, y = problem.X, problem.y
    a = normalize_sample_weights(sample_weight, len(y))
    if np.count_nonzero(a) < 2:
        raise ValueError("At least two observations need positive sample weight")
    beta = cp.Variable(X.shape[1])
    portfolio = problem.equal_weights - problem.basis @ beta
    returns = y - X @ beta
    centered_y, centered_X = y - a @ y, X - a @ X
    centered = centered_y - centered_X @ beta
    deviations = cp.multiply(np.sqrt(a), centered)
    variance = cp.sum_squares(deviations)
    ew_variance = float(a @ np.square(centered_y))
    variance_limit = constraints.volatility_limit**2 * ew_variance

    penalty = 0
    if regularization.kind == "lasso":
        penalty = regularization.strength * cp.norm1(beta)
    elif regularization.kind == "ridge":
        penalty = regularization.strength * cp.sum_squares(beta)
    objective = variance + penalty
    if loss.reward_strength > 0:
        objective -= loss.reward_strength * (a @ cp.log(1 + returns))
    if (loss.exposure_strength > 0 and constraints.leverage_limit > 1
            and ew_variance > 0):
        normalized_exposure = cp.pos(
            (cp.norm1(portfolio) - 1) / (constraints.leverage_limit - 1)
        )
        objective += (loss.exposure_strength * ew_variance
                      * cp.square(normalized_exposure))
    restrictions = [cp.norm1(portfolio) <= constraints.leverage_limit,
                    1 + returns >= constraints.wealth_floor]
    if variance_limit == 0:
        restrictions.append(deviations == 0)
    else:
        restrictions.append(cp.norm(100 * deviations, 2) <= 100 * np.sqrt(variance_limit))
    optimization = cp.Problem(cp.Minimize(objective), restrictions)
    attempts = []
    solvers = [
        ("CLARABEL", {"max_iter": 300, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}),
        ("SCS", {"eps": 1e-8, "max_iters": 100000}),
    ]
    for solver, options in solvers:
        try:
            with warnings.catch_warnings():
                # Handle this status explicitly below and expose each attempt
                # in diagnostics; never return an inaccurate solution silently.
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.*", category=UserWarning)
                optimization.solve(solver=solver, warm_start=False, **options)
        except cp.error.SolverError as error:
            attempts.append(f"{solver}: solver_error ({error})")
            continue
        attempts.append(f"{solver}: {optimization.status}")
        if optimization.status in {cp.INFEASIBLE, cp.UNBOUNDED}:
            break
        if optimization.status != cp.OPTIMAL or beta.value is None:
            continue

        coefficients = np.asarray(beta.value, dtype=float)
        try:
            values = validate_portfolio_weights(problem.portfolio_weights(coefficients), len(problem.equal_weights))
            actual = y - X @ coefficients
            terms = loss.components(actual, a)
        except ValueError:
            attempts[-1] += " (invalid returned values)"
            continue
        gross = float(np.abs(values).sum())
        min_wealth = float(np.min(1 + actual))
        # Independently verify holdings; do not project, clip returns, or
        # substitute EW. Both solvers solve precisely the same objective.
        if (gross > constraints.leverage_limit + 1e-7
                or terms["variance"] > variance_limit + 1e-10 + 1e-6 * variance_limit
                or min_wealth < constraints.wealth_floor - 1e-8):
            attempts[-1] += " (constraint check failed)"
            continue
        if constraints.leverage_limit == 1:
            exposure_unit_penalty = 0.0
        else:
            normalized_exposure = max(
                0.0, (gross - 1) / (constraints.leverage_limit - 1)
            )
            exposure_unit_penalty = normalized_exposure**2
        exposure_penalty = (loss.exposure_strength * ew_variance
                            * exposure_unit_penalty)
        training_loss = terms["loss"] + exposure_penalty
        diagnostics = {**terms, "loss": training_loss,
                       "exposure_unit_penalty": exposure_unit_penalty,
                       "exposure_scale": ew_variance,
                       "exposure_penalty": exposure_penalty,
                       "penalty": regularization.penalty(coefficients),
                       "gross_leverage": gross, "ew_variance": ew_variance,
                       "variance_limit": variance_limit, "min_wealth": min_wealth,
                       "solver": solver, "solver_status": optimization.status,
                       "solver_attempts": " | ".join(attempts)}
        return LinearModel(coefficients, terms["mean"]), diagnostics
    raise RuntimeError("Growth optimization failed: " + " | ".join(attempts))
