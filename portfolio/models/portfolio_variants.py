"""Constrained LASSO variants used by the four 252-day sensitivity experiments.

The selected model is delegated unchanged when every overlay is zero. Each
nonzero overlay acts on the same fully invested portfolio w = EW - N @ beta.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio.constraints import PortfolioConstraints
from portfolio.data.asset_graph import build_size_bm_incidence
from portfolio.decisions.holding_weights import validate_portfolio_weights
from portfolio.engine.convex_solver import covariance_factor, solve_checked
from portfolio.engine.portfolio_fit import FitResult, fit_portfolio_growth
from portfolio.losses import LogGrowthLoss
from portfolio.models.min_variance_regression import LinearModel, RegressionProblem
from portfolio.regularization import Regularization


@dataclass(frozen=True)
class ExperimentLoss(LogGrowthLoss):
    turnover_strength: float = 0.0
    smoothness_strength: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        for value in (self.turnover_strength, self.smoothness_strength):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Overlay strengths must be finite and nonnegative")


@dataclass(frozen=True)
class ExperimentConstraints(PortfolioConstraints):
    weight_limit: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.weight_limit is not None and (not np.isfinite(self.weight_limit)
                                               or self.weight_limit <= 0):
            raise ValueError("Weight limit must be positive and finite")


def size_bm_grid_edges(columns):
    """Return available horizontal/vertical Size-BM neighbor index pairs."""
    _, edges = build_size_bm_incidence(columns)
    return edges


def _incidence(edges, asset_count):
    if edges is None:
        return None
    pairs = np.asarray(edges, dtype=int)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or np.any(pairs < 0) or np.any(pairs >= asset_count)
            or np.any(pairs[:, 0] == pairs[:, 1])):
        raise ValueError("Neighbor edges must be valid pairs of distinct asset indices")
    matrix = np.zeros((len(pairs), asset_count))
    matrix[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    matrix[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    return matrix


def fit_portfolio_variant(history, *, regularization=Regularization("lasso", 1e-5),
                          loss=ExperimentLoss(), constraints=ExperimentConstraints(),
                          previous_weights=None, neighbor_edges=None,
                          covariance_estimator="sample"):
    """Fit the selected LASSO with optional turnover, grid, risk or weight-cap change.

    Training risk and every hard limit use only the supplied historical rows.
    The previous target weights, if supplied, are the preceding day's output.
    """
    if covariance_estimator not in ("sample", "ledoit_wolf"):
        raise ValueError("Unknown covariance estimator")
    if not isinstance(history, pd.DataFrame) or history.columns.has_duplicates:
        raise ValueError("History must have distinct asset columns")
    problem = RegressionProblem.from_returns(history)
    n_assets = history.shape[1]
    incidence = _incidence(neighbor_edges, n_assets)
    if loss.smoothness_strength and incidence is None:
        raise ValueError("Smoothness needs neighbor edges")
    if constraints.weight_limit is not None and constraints.weight_limit * n_assets < 1 - 1e-12:
        raise ValueError("Weight limit is infeasible for full investment")
    if previous_weights is None:
        previous = problem.equal_weights
    else:
        if not isinstance(previous_weights, pd.Series) or set(previous_weights.index) != set(history.columns):
            raise ValueError("Previous weights must be indexed by the same assets")
        previous = validate_portfolio_weights(previous_weights.reindex(history.columns), n_assets)

    def overlay_diagnostics(weights):
        values = weights.to_numpy(dtype=float)
        return {
            "turnover": float(.5 * np.abs(values - previous).sum()),
            "neighbor_smoothness": float(np.abs(incidence @ values).mean()) if incidence is not None else 0.0,
            "max_abs_weight": float(np.abs(values).max()),
        }

    if (covariance_estimator == "sample" and loss.turnover_strength == 0
            and loss.smoothness_strength == 0 and constraints.weight_limit is None):
        selected = fit_portfolio_growth(history, regularization=regularization,
                                        loss=LogGrowthLoss(loss.reward_strength, loss.exposure_strength),
                                        constraints=PortfolioConstraints(
                                            constraints.leverage_limit, constraints.volatility_limit,
                                            constraints.wealth_floor))
        selected.diagnostics = {**selected.diagnostics, **overlay_diagnostics(selected.weights)}
        return selected

    X, y = problem.X, problem.y
    sample_weights = np.full(len(history), 1 / len(history))
    centered_y, centered_X = y - y.mean(), X - X.mean(axis=0)
    ew_variance_sample = float(np.mean(centered_y**2))
    beta = cp.Variable(X.shape[1])
    portfolio = problem.equal_weights - problem.basis @ beta
    scenario_returns = y - X @ beta
    if covariance_estimator == "sample":
        deviations = cp.multiply(np.sqrt(sample_weights), centered_y - centered_X @ beta)
        risk = cp.sum_squares(deviations)
        risk_vector = deviations
        ew_risk = ew_variance_sample
        covariance = None
        shrinkage = None
    else:
        estimator = LedoitWolf().fit(history.to_numpy(dtype=float))
        covariance = estimator.covariance_
        factor = covariance_factor(covariance)
        risk_vector = factor @ portfolio
        risk = cp.sum_squares(risk_vector)
        ew_risk = float(problem.equal_weights @ covariance @ problem.equal_weights)
        shrinkage = float(estimator.shrinkage_)
    risk_limit = constraints.volatility_limit**2 * ew_risk

    objective = risk
    if regularization.kind == "lasso":
        objective += regularization.strength * cp.norm1(beta)
    elif regularization.kind == "ridge":
        objective += regularization.strength * cp.sum_squares(beta)
    if loss.reward_strength:
        objective -= loss.reward_strength * (sample_weights @ cp.log(1 + scenario_returns))
    if loss.exposure_strength and constraints.leverage_limit > 1 and ew_risk > 0:
        exposure = cp.pos((cp.norm1(portfolio) - 1) / (constraints.leverage_limit - 1))
        objective += loss.exposure_strength * ew_risk * cp.square(exposure)
    if loss.turnover_strength:
        objective += loss.turnover_strength * ew_risk * .5 * cp.norm1(portfolio - previous)
    if loss.smoothness_strength:
        objective += loss.smoothness_strength * ew_risk * cp.norm1(incidence @ portfolio) / len(incidence)

    restrictions = [cp.norm1(portfolio) <= constraints.leverage_limit,
                    1 + scenario_returns >= constraints.wealth_floor]
    if risk_limit == 0:
        restrictions.append(risk_vector == 0)
    else:
        restrictions.append(cp.norm(100 * risk_vector, 2) <= 100 * np.sqrt(risk_limit))
    if constraints.weight_limit is not None:
        restrictions.append(cp.abs(portfolio) <= constraints.weight_limit)
    optimization = cp.Problem(cp.Minimize(objective), restrictions)

    def checked(coefficients):
        values = validate_portfolio_weights(problem.portfolio_weights(coefficients), n_assets)
        actual_returns = history.to_numpy(dtype=float) @ values
        terms = loss.components(actual_returns)
        gross = float(np.abs(values).sum())
        fitted_risk = (terms["variance"] if covariance is None
                       else float(values @ covariance @ values))
        if (gross > constraints.leverage_limit + 1e-7
                or fitted_risk > risk_limit + 1e-10 + 1e-6 * risk_limit
                or np.min(1 + actual_returns) < constraints.wealth_floor - 1e-8
                or (constraints.weight_limit is not None
                    and np.abs(values).max() > constraints.weight_limit + 1e-7)):
            raise ValueError("variant constraint check failed")
        exposure_unit = (max(0, (gross - 1) / (constraints.leverage_limit - 1))**2
                         if constraints.leverage_limit > 1 else 0.0)
        movement = float(.5 * np.abs(values - previous).sum())
        gap = float(np.abs(incidence @ values).mean()) if incidence is not None else 0.0
        exposure_penalty = loss.exposure_strength * ew_risk * exposure_unit
        turnover_penalty = loss.turnover_strength * ew_risk * movement
        smoothness_penalty = loss.smoothness_strength * ew_risk * gap
        fitted_loss = (fitted_risk - terms["reward"] + exposure_penalty
                       + turnover_penalty + smoothness_penalty)
        return {**terms, "variance": fitted_risk, "loss": fitted_loss,
                "exposure_unit_penalty": exposure_unit, "exposure_scale": ew_risk,
                "exposure_penalty": exposure_penalty,
                "turnover": movement, "turnover_penalty": turnover_penalty,
                "neighbor_smoothness": gap, "smoothness_penalty": smoothness_penalty,
                "max_abs_weight": float(np.abs(values).max()),
                "penalty": regularization.penalty(coefficients),
                "gross_leverage": gross, "ew_variance": ew_risk,
                "variance_limit": risk_limit, "min_wealth": float(np.min(1 + actual_returns)),
                "ledoit_wolf_shrinkage": shrinkage}

    coefficients, solver_info = solve_checked(
        optimization, beta, checked,
        solvers=[("CLARABEL", {"max_iter": 300, "tol_gap_abs": 1e-9,
                              "tol_gap_rel": 1e-9, "tol_feas": 1e-9}),
                 ("SCS", {"eps": 1e-8, "max_iters": 100000})],
        stop_on_infeasible=True, error_label="Variant optimization failed")
    diagnostics = {**checked(coefficients), **solver_info}
    weights = pd.Series(problem.portfolio_weights(coefficients), index=history.columns, name="weight")
    model = LinearModel(coefficients, diagnostics["mean"])
    return FitResult(model=model, raw_weights=weights.copy(), weights=weights,
                     training_loss=diagnostics["loss"],
                     objective=diagnostics["loss"] + diagnostics["penalty"],
                     diagnostics=diagnostics)
