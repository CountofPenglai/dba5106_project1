"""Learn fixed, nonnegative mixtures from realized out-of-sample strategy returns."""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .convex import covariance_factor, solve_checked


@dataclass
class MixtureFit:
    theta: np.ndarray
    diagnostics: dict


def _simplex(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or x.min() < -1e-7 or abs(x.sum()-1) > 1e-7:
        raise ValueError('Mixture weights must lie on the three-component simplex')
    # Only remove solver roundoff after validation. Recompute all scores after
    # this explicitly recorded correction, never clip asset returns/weights.
    x = np.maximum(x, 0)
    return x/x.sum()


def mix_weights(candidate_weights, theta):
    x = np.asarray(theta, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or (x < 0).any() or abs(x.sum()-1) > 1e-8:
        raise ValueError('Mixture weights must be nonnegative and sum to one')
    w = np.asarray(candidate_weights, dtype=float)
    if w.ndim not in {2, 3} or w.shape[-2] != 3 or not np.isfinite(w).all():
        raise ValueError('Candidate weights need shape (3, assets) or (dates, 3, assets)')
    return np.einsum('...jp,j->...p', w, x)


def fit_mixture(component_returns, *, objective, gamma=0):
    z = np.asarray(component_returns, dtype=float)
    if z.ndim != 2 or z.shape[1] != 3 or len(z) < 3 or not np.isfinite(z).all():
        raise ValueError('Mixture training needs three finite component series and at least three days')
    if objective not in {'sharpe', 'return_volatility'}:
        raise ValueError('Unknown decision objective')
    if isinstance(gamma, bool) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError('Gamma must be finite and nonnegative')
    # A common data-dependent scale preserves both optimizers and keeps the
    # conic problems well conditioned even for small, nondegenerate returns.
    scale = 1/max(float(np.std(z, axis=0, ddof=1).max()), float(np.abs(z.mean(axis=0)).max()), 1e-12)
    scaled = z*scale
    mean = scaled.mean(axis=0)
    factor = covariance_factor(np.cov(scaled, rowvar=False, ddof=1))
    theta = cp.Variable(3)
    simplex = [theta >= 0, cp.sum(theta) == 1]
    sd = cp.norm(factor @ theta)
    target = np.array([0., 0., 1.])
    primary_info = {}
    branch = 'mean_volatility'
    if objective == 'sharpe':
        minimum_risk = cp.Problem(cp.Minimize(sd), simplex)
        low, _ = solve_checked(minimum_risk, theta, _simplex)
        if np.std(z @ _simplex(low), ddof=1) <= 1e-8:
            raise ValueError('Feasible mixture has degenerate volatility (<=1e-8)')
        if mean.max() <= 0:
            scores = z.mean(axis=0)/z.std(axis=0, ddof=1)
            best = scores.max()
            tied = np.flatnonzero(np.isclose(scores, best, atol=1e-10, rtol=1e-8))
            index = 2 if 2 in tied else int(tied[-1])
            values = np.eye(3)[index]
            primary_info = dict(solver='analytic', solver_status='optimal', solver_attempts='best nonpositive vertex')
            branch = 'nonpositive_means'
            score = float(scores[index])
            return MixtureFit(values, {**primary_info, 'branch': branch, 'score': score,
                                      'objective': objective, 'gamma': float(gamma),
                                      'mean': float((z@values).mean()), 'volatility': float((z@values).std(ddof=1)),
                                      'simplex_correction': 0., 'primary_score': score,
                                      'primary_solver': 'analytic', 'primary_solver_attempts': 'best nonpositive vertex'})
        y = cp.Variable(3)
        transformed = cp.Problem(cp.Minimize(cp.norm(factor @ y)), [y >= 0, mean @ y == 1])
        def check_transform(x):
            if x.min() < -1e-7 or x.sum() <= 0 or abs(mean @ x - 1) > 1e-7:
                raise ValueError('Invalid positive-Sharpe transformation')
        raw, primary_info = solve_checked(transformed, y, check_transform)
        initial = _simplex(raw/raw.sum())
        primary_score = float(mean @ initial/np.linalg.norm(factor @ initial))
        floor = max(0., primary_score-1e-8)
        tie_restriction = floor*sd <= mean @ theta
        branch = 'positive_mean_transform'
    else:
        loss = gamma*sd-mean @ theta
        problem = cp.Problem(cp.Minimize(loss), simplex)
        initial, primary_info = solve_checked(problem, theta, _simplex)
        initial = _simplex(initial)
        primary_score = float(mean @ initial-gamma*np.linalg.norm(factor @ initial)) / scale
        tie_restriction = loss <= -primary_score*scale+1e-8
    secondary = cp.Problem(cp.Minimize(cp.sum_squares(theta-target)), simplex+[tie_restriction])
    def check_secondary(x):
        t = _simplex(x)
        sigma = np.linalg.norm(factor @ t)
        score = float(mean @ t/sigma if objective == 'sharpe' else (mean @ t-gamma*sigma)/scale)
        tolerance = 1e-7 if objective == 'sharpe' else 2e-8/scale
        if score < primary_score-tolerance:
            raise ValueError('Secondary mixture degrades primary objective')
    target_sd = float(np.linalg.norm(factor @ target))
    target_score = float(mean @ target/target_sd if objective == 'sharpe'
                         else (mean @ target-gamma*target_sd)/scale)
    tie_tolerance = 1e-8 if objective == 'sharpe' else 1e-8/scale
    if target_score >= primary_score-tie_tolerance:
        # EW is feasible and achieves zero secondary distance: exact solution
        # of the tie-break, including rank-deficient duplicate components.
        raw = target.copy()
        info = dict(solver='analytic', solver_status='optimal', solver_attempts='EW solves secondary tie-break')
    else:
        raw, info = solve_checked(secondary, theta, check_secondary)
    values = _simplex(raw)
    realized = z @ values
    avg, volatility = float(realized.mean()), float(realized.std(ddof=1))
    score = avg/volatility if objective == 'sharpe' else avg-gamma*volatility
    return MixtureFit(values, {**info, 'primary_solver': primary_info['solver'],
                              'primary_solver_attempts': primary_info['solver_attempts'],
                              'branch': branch, 'score': float(score), 'primary_score': primary_score,
                              'mean': avg, 'volatility': volatility, 'objective': objective,
                              'gamma': float(gamma), 'simplex_correction': float(np.max(np.abs(values-raw)))})
