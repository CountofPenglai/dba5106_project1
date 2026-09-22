"""Convex fitting for the linear portfolio model with a Size-BM grid-smoothness penalty."""

import numpy as np
from portfolio.engine.convex_solver import solve_checked

from portfolio.constraints import PortfolioConstraints
from portfolio.losses import LogGrowthLoss
from portfolio.models.min_variance_regression import LinearModel
from portfolio.regularization import Regularization
from portfolio.data.observation_weights import normalize_sample_weights
from portfolio.decisions.holding_weights import validate_portfolio_weights


def fit_growth_smooth(problem, *, regularization=Regularization("lasso", 1e-5),
                      loss=LogGrowthLoss(), constraints=PortfolioConstraints(),
                      rho=0.0, incidence=None, sample_weight=None):
    """Fit beta in w=EW-N@beta, adding an adjacent-weight (grid) penalty.

    This mirrors fit_growth exactly and adds one term: rho times training EW
    variance times the mean absolute weight difference across grid edges in
    `incidence` (a signed (num_edges, num_assets) matrix, e.g. from
    build_size_bm_incidence). rho=0 reproduces fit_growth's objective and
    result exactly; incidence is unused in that case.
    """
    import cvxpy as cp

    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    if rho > 0 and incidence is None:
        raise ValueError("incidence is required when rho > 0")

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
    if rho > 0:
        grid_gap = cp.norm1(incidence @ portfolio) / incidence.shape[0]
        objective += rho * ew_variance * grid_gap
    restrictions = [cp.norm1(portfolio) <= constraints.leverage_limit,
                    1 + returns >= constraints.wealth_floor]
    if variance_limit == 0:
        restrictions.append(deviations == 0)
    else:
        restrictions.append(cp.norm(100 * deviations, 2) <= 100 * np.sqrt(variance_limit))
    optimization = cp.Problem(cp.Minimize(objective), restrictions)

    def checked_components(coefficients):
        values = validate_portfolio_weights(problem.portfolio_weights(coefficients), len(problem.equal_weights))
        actual = y - X @ coefficients
        terms = loss.components(actual, a)
        gross = float(np.abs(values).sum())
        min_wealth = float(np.min(1 + actual))
        # Recompute feasibility from actual holdings, independently of solver status.
        if (gross > constraints.leverage_limit + 1e-7
                or terms["variance"] > variance_limit + 1e-10 + 1e-6 * variance_limit
                or min_wealth < constraints.wealth_floor - 1e-8):
            raise ValueError("constraint check failed")
        exposure_unit_penalty = 0.0 if constraints.leverage_limit == 1 else max(
            0.0, (gross - 1) / (constraints.leverage_limit - 1)) ** 2
        exposure_penalty = loss.exposure_strength * ew_variance * exposure_unit_penalty
        neighbor_gap = float(np.abs(incidence @ values).mean()) if incidence is not None else 0.0
        smoothness_penalty = rho * ew_variance * neighbor_gap
        return {**terms, "loss": terms["loss"] + exposure_penalty + smoothness_penalty,
                "exposure_unit_penalty": exposure_unit_penalty, "exposure_scale": ew_variance,
                "exposure_penalty": exposure_penalty, "penalty": regularization.penalty(coefficients),
                "gross_leverage": gross, "ew_variance": ew_variance,
                "variance_limit": variance_limit, "min_wealth": min_wealth,
                "rho": rho, "neighbor_gap": neighbor_gap, "smoothness_penalty": smoothness_penalty}

    solvers = [
        ("CLARABEL", {"max_iter": 300, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}),
        ("SCS", {"eps": 1e-8, "max_iters": 100000}),
    ]
    coefficients, solver_diagnostics = solve_checked(
        optimization, beta, checked_components, solvers=solvers,
        stop_on_infeasible=True, error_label="Smooth growth optimization failed")
    diagnostics = {**checked_components(coefficients), **solver_diagnostics}
    return LinearModel(coefficients, diagnostics["mean"]), diagnostics
