"""Direct asset-weight linear programs with historical risk constraints.

The risk limits here are linear historical proxies. MAD is the mean absolute
deviation from the historical portfolio-return mean. Drawdown is the maximum
decline of uncompounded cumulative P&L from a running peak that starts at zero.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog


_PRIMARY_RETURN_TOLERANCE = 1e-10
_HIGHS_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
    "ipm_optimality_tolerance": 1e-12,
}


class LinearPortfolioError(RuntimeError):
    """Raised when an LP is nonoptimal or fails independent verification."""


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class LPConstraints:
    leverage_limit: float = 1.5
    position_limit: float = 0.05
    mad_ratio: float = 1.0
    drawdown_ratio: float = 1.0
    wealth_floor: float = 1e-6

    def __post_init__(self):
        leverage = _finite_number(self.leverage_limit, "Leverage limit")
        position = _finite_number(self.position_limit, "Position limit")
        mad = _finite_number(self.mad_ratio, "MAD ratio")
        drawdown = _finite_number(self.drawdown_ratio, "Drawdown ratio")
        floor = _finite_number(self.wealth_floor, "Wealth floor")
        if leverage < 1:
            raise ValueError("Leverage limit must be at least one")
        if position <= 0:
            raise ValueError("Position limit must be positive")
        if mad < 0:
            raise ValueError("MAD ratio must be nonnegative")
        if drawdown < 0:
            raise ValueError("Drawdown ratio must be nonnegative")
        if not 0 < floor <= 1:
            raise ValueError("Wealth floor must be in (0, 1]")
        object.__setattr__(self, "leverage_limit", leverage)
        object.__setattr__(self, "position_limit", position)
        object.__setattr__(self, "mad_ratio", mad)
        object.__setattr__(self, "drawdown_ratio", drawdown)
        object.__setattr__(self, "wealth_floor", floor)


@dataclass(frozen=True)
class LPObjective:
    """Daily return less MAD, trading-cost and distance-from-EW penalties."""

    mad_penalty: float = 0.0
    transaction_cost: float = 0.0
    ew_penalty: float = 0.0

    def __post_init__(self):
        for name in ('mad_penalty', 'transaction_cost', 'ew_penalty'):
            value = _finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f'{name} must be nonnegative')
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LPContext:
    history: np.ndarray
    equal_weights: np.ndarray
    constraints: LPConstraints
    mad_limit: float
    drawdown_limit: float
    assets: tuple | None = None
    _centered: np.ndarray = field(repr=False, compare=False, default=None)
    _cumulative: np.ndarray = field(repr=False, compare=False, default=None)
    _a_ub: sparse.csr_matrix = field(repr=False, compare=False, default=None)
    _b_ub: np.ndarray = field(repr=False, compare=False, default=None)
    _a_eq: sparse.csr_matrix = field(repr=False, compare=False, default=None)
    _b_eq: np.ndarray = field(repr=False, compare=False, default=None)
    _bounds: tuple = field(repr=False, compare=False, default=())


def _drawdown(portfolio_returns):
    cumulative = np.cumsum(np.asarray(portfolio_returns, dtype=float))
    peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    return float(np.max(peaks - cumulative))


def _normalize_rows(matrix, right_hand_side):
    matrix = sparse.csr_matrix(matrix, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    magnitudes = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    scales = np.ones_like(magnitudes)
    nonzero = magnitudes > 0
    scales[nonzero] = 1.0 / magnitudes[nonzero]
    return sparse.diags(scales, format="csr") @ matrix, rhs * scales


def _history_values(history):
    assets = None
    if isinstance(history, pd.DataFrame):
        if not history.columns.is_unique:
            raise ValueError("History assets must be unique")
        assets = tuple(history.columns.tolist())
        try:
            values = history.to_numpy(dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("History must contain finite numeric returns") from error
    elif isinstance(history, np.ndarray):
        values = np.array(history, dtype=float, copy=True)
    else:
        raise TypeError("History must be a pandas DataFrame or NumPy ndarray")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("History needs at least two dates and two assets")
    if not np.isfinite(values).all():
        raise ValueError("History must contain only finite returns")
    return values, assets


def _mad_scale(centered, mad_limit):
    scale = float(np.max(np.abs(centered)))
    return float(np.sqrt(mad_limit * scale)) if mad_limit > 0 and scale > 0 else max(scale, 1.0)


def _primary_problem(history, centered, cumulative, constraints, mad_limit, drawdown_limit):
    dates, assets = history.shape
    eye_assets = sparse.eye(assets, format="csr")
    eye_dates = sparse.eye(dates, format="csr")
    zeros_nt = sparse.csr_matrix((assets, dates))
    zeros_tn = sparse.csr_matrix((dates, assets))
    zeros_tt = sparse.csr_matrix((dates, dates))

    rows = [
        sparse.hstack([eye_assets, -eye_assets, zeros_nt, zeros_nt]),
        sparse.hstack([-eye_assets, -eye_assets, zeros_nt, zeros_nt]),
    ]
    rhs = [np.zeros(assets), np.zeros(assets)]
    rows.append(
        sparse.hstack(
            [
                sparse.csr_matrix((1, assets)),
                sparse.csr_matrix(np.ones((1, assets))),
                sparse.csr_matrix((1, dates)),
                sparse.csr_matrix((1, dates)),
            ]
        )
    )
    rhs.append(np.array([constraints.leverage_limit]))

    mad_variable_scale = _mad_scale(centered, mad_limit)
    centered_matrix = sparse.csr_matrix(centered / mad_variable_scale)
    rows.extend(
        [
            sparse.hstack([centered_matrix, zeros_tn, -eye_dates, zeros_tt]),
            sparse.hstack([-centered_matrix, zeros_tn, -eye_dates, zeros_tt]),
        ]
    )
    rhs.extend([np.zeros(dates), np.zeros(dates)])
    rows.append(
        sparse.hstack(
            [
                sparse.csr_matrix((1, assets)),
                sparse.csr_matrix((1, assets)),
                sparse.csr_matrix(np.ones((1, dates))),
                sparse.csr_matrix((1, dates)),
            ]
        )
    )
    rhs.append(np.array([dates * mad_limit / mad_variable_scale]))

    cumulative_scale = float(np.max(np.abs(cumulative)))
    if drawdown_limit > 0.0 and cumulative_scale > 0.0:
        drawdown_variable_scale = float(
            np.sqrt(drawdown_limit * cumulative_scale)
        )
    else:
        drawdown_variable_scale = max(cumulative_scale, 1.0)
    cumulative_matrix = sparse.csr_matrix(cumulative / drawdown_variable_scale)
    rows.append(sparse.hstack([cumulative_matrix, zeros_tn, zeros_tt, -eye_dates]))
    rhs.append(np.zeros(dates))
    if dates > 1:
        monotone_peak = sparse.diags(
            [np.ones(dates - 1), -np.ones(dates - 1)],
            [0, 1],
            shape=(dates - 1, dates),
            format="csr",
        )
        rows.append(
            sparse.hstack(
                [
                    sparse.csr_matrix((dates - 1, assets)),
                    sparse.csr_matrix((dates - 1, assets)),
                    sparse.csr_matrix((dates - 1, dates)),
                    monotone_peak,
                ]
            )
        )
        rhs.append(np.zeros(dates - 1))
    rows.append(sparse.hstack([-cumulative_matrix, zeros_tn, zeros_tt, eye_dates]))
    rhs.append(np.full(dates, drawdown_limit / drawdown_variable_scale))
    rows.append(
        sparse.hstack(
            [sparse.csr_matrix(-history), zeros_tn, zeros_tt, zeros_tt]
        )
    )
    rhs.append(np.full(dates, 1.0 - constraints.wealth_floor))

    a_ub, b_ub = _normalize_rows(
        sparse.vstack(rows, format="csr"), np.concatenate(rhs)
    )
    equality_rows = [
        sparse.hstack(
            [
                sparse.csr_matrix(np.ones((1, assets))),
                sparse.csr_matrix((1, assets)),
                sparse.csr_matrix((1, dates)),
                sparse.csr_matrix((1, dates)),
            ]
        )
    ]
    equality_rhs = [np.array([1.0])]
    if mad_limit == 0.0:
        equality_rows.append(
            sparse.hstack(
                [sparse.csr_matrix(centered), zeros_tn, zeros_tt, zeros_tt]
            )
        )
        equality_rhs.append(np.zeros(dates))
    if drawdown_limit == 0.0:
        zero_drawdown = sparse.hstack(
            [sparse.csr_matrix(-history), zeros_tn, zeros_tt, zeros_tt]
        )
        extra_a, extra_b = _normalize_rows(zero_drawdown, np.zeros(dates))
        a_ub = sparse.vstack([a_ub, extra_a], format="csr")
        b_ub = np.r_[b_ub, extra_b]
    a_eq, b_eq = _normalize_rows(
        sparse.vstack(equality_rows, format="csr"), np.concatenate(equality_rhs)
    )
    bounds = (
        *((-constraints.position_limit, constraints.position_limit),) * assets,
        *((0.0, constraints.position_limit),) * assets,
        *((0.0, None),) * dates,
        *((0.0, None),) * dates,
    )
    return a_ub, b_ub, a_eq, b_eq, bounds


def build_lp_context(history, constraints=LPConstraints()):
    """Validate history and build a reusable sparse LP constraint context."""
    if not isinstance(constraints, LPConstraints):
        raise TypeError("constraints must be an LPConstraints instance")
    values, assets = _history_values(history)
    _, asset_count = values.shape
    if asset_count * constraints.position_limit < 1.0 - 1e-12:
        raise ValueError("Position limit is incompatible with a unit budget")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = values - values.mean(axis=0)
        cumulative = np.cumsum(values, axis=0)
        equal_weights = np.full(asset_count, 1.0 / asset_count)
        equal_returns = values @ equal_weights
        mad_limit = constraints.mad_ratio * float(
            np.mean(np.abs(equal_returns - equal_returns.mean()))
        )
        drawdown_limit = constraints.drawdown_ratio * _drawdown(equal_returns)
    if not (
        np.isfinite(centered).all()
        and np.isfinite(cumulative).all()
        and np.isfinite(mad_limit)
        and np.isfinite(drawdown_limit)
    ):
        raise ValueError("History or risk configuration produces nonfinite LP values")
    a_ub, b_ub, a_eq, b_eq, bounds = _primary_problem(
        values, centered, cumulative, constraints, mad_limit, drawdown_limit
    )
    for array in (values, centered, cumulative, equal_weights, b_ub, b_eq):
        array.setflags(write=False)
    return LPContext(
        history=values,
        equal_weights=equal_weights,
        constraints=constraints,
        mad_limit=float(mad_limit),
        drawdown_limit=float(drawdown_limit),
        assets=assets,
        _centered=centered,
        _cumulative=cumulative,
        _a_ub=a_ub,
        _b_ub=b_ub,
        _a_eq=a_eq,
        _b_eq=b_eq,
        _bounds=bounds,
    )


def _weight_values(context, weights):
    if not isinstance(context, LPContext):
        raise TypeError("context must be an LPContext")
    values = np.asarray(weights, dtype=float)
    expected_shape = (context.history.shape[1],)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"Weights must be finite with shape {expected_shape}")
    return values


def _feasibility_tolerances(context):
    history_scale = max(float(np.max(np.abs(context.history))), 1e-6)
    cumulative_scale = max(float(np.max(np.abs(context._cumulative))), 1e-6)
    return {
        "budget": 5e-9,
        "exposure": 5e-9,
        "mad": max(5e-13 * history_scale, 5e-8 * context.mad_limit),
        "drawdown": max(5e-13 * cumulative_scale, 5e-8 * context.drawdown_limit),
        "wealth": max(5e-10, 5e-13 * history_scale),
    }


def check_lp_weights(context, weights):
    """Independently recompute the actual constraint values for weights."""
    values = _weight_values(context, weights)
    realized = context.history @ values
    budget = float(values.sum())
    gross = float(np.abs(values).sum())
    max_position = float(np.abs(values).max())
    mad = float(np.abs(realized - realized.mean()).mean())
    drawdown = _drawdown(realized)
    min_wealth = float(np.min(1.0 + realized))
    tolerances = _feasibility_tolerances(context)
    feasible = bool(
        abs(budget - 1.0) <= tolerances["budget"]
        and gross <= context.constraints.leverage_limit + tolerances["exposure"]
        and max_position <= context.constraints.position_limit + tolerances["exposure"]
        and mad <= context.mad_limit + tolerances["mad"]
        and drawdown <= context.drawdown_limit + tolerances["drawdown"]
        and min_wealth >= context.constraints.wealth_floor - tolerances["wealth"]
    )
    return {
        "budget": budget,
        "budget_error": float(abs(budget - 1.0)),
        "gross": gross,
        "max_position": max_position,
        "mad": mad,
        "drawdown": drawdown,
        "min_wealth": min_wealth,
        "mad_limit": float(context.mad_limit),
        "drawdown_limit": float(context.drawdown_limit),
        "feasible": feasible,
    }


def _solve(c, *, a_ub, b_ub, a_eq, b_eq, bounds, stage):
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options=_HIGHS_OPTIONS,
    )
    if result.status != 0 or not result.success:
        condition = "infeasible" if result.status == 2 else "nonoptimal"
        raise LinearPortfolioError(
            f"{stage} LP is {condition} (status {result.status}): {result.message}"
        )
    if result.x is None or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise LinearPortfolioError(f"{stage} LP returned nonfinite values")
    return result


def _verified_weights(context, result, stage):
    weights = np.asarray(result.x[: context.history.shape[1]], dtype=float)
    actual = check_lp_weights(context, weights)
    if not actual["feasible"]:
        raise LinearPortfolioError(
            f"{stage} LP solution failed independent feasibility checks: {actual}"
        )
    return weights, actual


def solve_linear_portfolio(context, expected_returns, *, objective=LPObjective(),
                           previous_weights=None):
    """Maximize return less MAD/cost/EW penalties; break utility ties toward EW.

    Cost is a one-way rate times sum(abs(target - pretrade weights)). The
    caller supplies drifted pretrade holdings; no turnover budget is imposed.
    Zero penalties retain the original return-only LP.
    """
    if not isinstance(context, LPContext):
        raise TypeError("context must be an LPContext")
    expected = np.asarray(expected_returns, dtype=float)
    asset_count = context.history.shape[1]
    if expected.shape != (asset_count,) or not np.isfinite(expected).all():
        raise ValueError(f"Expected returns must be finite with shape {(asset_count,)}")
    if not isinstance(objective, LPObjective):
        raise TypeError('objective must be an LPObjective')
    if previous_weights is None and objective.transaction_cost > 0:
        raise ValueError('previous_weights are required for transaction costs')
    previous = (context.equal_weights if previous_weights is None
                else _weight_values(context, previous_weights))
    if abs(previous.sum() - 1) > 1e-8:
        raise ValueError('previous_weights must sum to one')
    variable_count = context._a_ub.shape[1]
    cost_vector = np.zeros(variable_count)
    cost_vector[:asset_count] = -expected
    dates = len(context.history)
    cost_vector[2*asset_count:2*asset_count+dates] = (
        objective.mad_penalty * _mad_scale(context._centered, context.mad_limit) / dates)
    a_ub, b_ub, a_eq, bounds = context._a_ub, context._b_ub, context._a_eq, context._bounds
    for rate, center in ((objective.transaction_cost, previous),
                         (objective.ew_penalty, context.equal_weights)):
        if rate == 0:
            continue
        # Auxiliary q >= abs(w - center), shared by trading and EW penalties.
        identity = sparse.eye(asset_count, format='csr')
        padding = sparse.csr_matrix((asset_count, variable_count - asset_count))
        a_ub = sparse.vstack([
            sparse.hstack([a_ub, sparse.csr_matrix((a_ub.shape[0], asset_count))]),
            sparse.hstack([identity, padding, -identity]),
            sparse.hstack([-identity, padding, -identity]),
        ], format='csr')
        b_ub = np.r_[b_ub, center, -center]
        a_eq = sparse.hstack([a_eq, sparse.csr_matrix((a_eq.shape[0], asset_count))], format='csr')
        bounds = (*bounds, *((0.0, None),) * asset_count)
        cost_vector = np.r_[cost_vector, np.full(asset_count, rate)]
        variable_count += asset_count

    def utility(weights, actual):
        cost = objective.transaction_cost * float(np.abs(weights - previous).sum())
        ew_cost = objective.ew_penalty * float(np.abs(weights-context.equal_weights).sum())
        return float(expected @ weights - objective.mad_penalty * actual['mad'] - cost - ew_cost), cost

    primary = _solve(
        cost_vector,
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=a_eq,
        b_eq=context._b_eq,
        bounds=bounds,
        stage="primary",
    )
    primary_weights, primary_actual = _verified_weights(context, primary, "primary")
    primary_return = float(expected @ primary_weights)
    primary_utility, _ = utility(primary_weights, primary_actual)
    if abs(primary_utility + primary.fun) > 5e-10:
        raise LinearPortfolioError('primary LP utility failed independent verification')

    secondary_a = sparse.hstack(
        [
            a_ub,
            sparse.csr_matrix((a_ub.shape[0], asset_count)),
        ],
        format="csr",
    )
    return_row = sparse.hstack(
        [
            sparse.csr_matrix(cost_vector.reshape(1, -1)),
            sparse.csr_matrix((1, asset_count)),
        ],
        format="csr",
    )
    identity = sparse.eye(asset_count, format="csr")
    distance_positive = sparse.hstack(
        [
            identity,
            sparse.csr_matrix((asset_count, variable_count - asset_count)),
            -identity,
        ],
        format="csr",
    )
    distance_negative = sparse.hstack(
        [
            -identity,
            sparse.csr_matrix((asset_count, variable_count - asset_count)),
            -identity,
        ],
        format="csr",
    )
    new_rows, new_rhs = _normalize_rows(
        sparse.vstack([return_row, distance_positive, distance_negative], format="csr"),
        np.r_[
            -(primary_utility - _PRIMARY_RETURN_TOLERANCE),
            context.equal_weights,
            -context.equal_weights,
        ],
    )
    secondary_a = sparse.vstack([secondary_a, new_rows], format="csr")
    secondary_b = np.r_[b_ub, new_rhs]
    secondary_eq = sparse.hstack(
        [
            a_eq,
            sparse.csr_matrix((a_eq.shape[0], asset_count)),
        ],
        format="csr",
    )
    secondary_objective = np.r_[np.zeros(variable_count), np.ones(asset_count)]
    secondary = _solve(
        secondary_objective,
        a_ub=secondary_a,
        b_ub=secondary_b,
        a_eq=secondary_eq,
        b_eq=context._b_eq,
        bounds=(*bounds, *((0.0, None),) * asset_count),
        stage="secondary",
    )
    weights, actual = _verified_weights(context, secondary, "secondary")
    predicted_return = float(expected @ weights)
    final_utility, transaction_cost = utility(weights, actual)
    secondary_gap = float(primary_utility - final_utility)
    if secondary_gap > _PRIMARY_RETURN_TOLERANCE + 5e-12:
        raise LinearPortfolioError(
            "secondary LP degraded utility beyond the 1e-10 tolerance"
        )
    diagnostics = {
        "solver": "scipy.optimize.linprog(highs)",
        "solver_status": "optimal",
        "primary_solver_status": "optimal",
        "secondary_solver_status": "optimal",
        "primary_return": primary_return,
        "predicted_return": predicted_return,
        "primary_utility": primary_utility,
        "utility": final_utility,
        "mad_penalty": objective.mad_penalty,
        "transaction_cost": transaction_cost,
        "ew_penalty": objective.ew_penalty,
        "ew_distance": float(np.abs(weights-context.equal_weights).sum()),
        "secondary_gap": secondary_gap,
        "primary_iterations": int(primary.nit),
        "secondary_iterations": int(secondary.nit),
        **actual,
    }
    return weights.copy(), diagnostics
