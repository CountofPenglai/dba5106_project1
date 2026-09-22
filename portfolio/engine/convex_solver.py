"""Checked CVXPY execution shared by constrained models and decision layers."""
import warnings
import cvxpy as cp
import numpy as np


def solve_checked(problem, variable, check, *, solvers=None, stop_on_infeasible=False,
                  error_label="Optimization failed"):
    """Retry a numerical failure, but never substitute another objective/strategy."""
    attempts = []
    solvers = solvers or [
        ('CLARABEL', dict(max_iter=400, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)),
        ('SCS', dict(eps=1e-8, max_iters=100000)),
    ]
    for solver, options in solvers:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Solution may be inaccurate.*', category=UserWarning)
                problem.solve(solver=solver, warm_start=False, **options)
            attempts.append(f'{solver}: {problem.status}')
            if stop_on_infeasible and problem.status in (cp.INFEASIBLE, cp.UNBOUNDED):
                break
            if problem.status != cp.OPTIMAL or variable.value is None:
                continue
            values = np.asarray(variable.value, dtype=float).copy()
            if not np.isfinite(values).all():
                raise ValueError('nonfinite solution')
            check(values)
            return values, dict(solver=solver, solver_status=problem.status,
                                solver_attempts=' | '.join(attempts))
        except (cp.error.SolverError, ValueError) as exc:
            attempts.append(f'{solver}: {type(exc).__name__}: {exc}')
    raise RuntimeError(error_label + ': ' + ' | '.join(attempts))


def covariance_factor(covariance):
    """Return B with B.T@B=covariance; tolerate roundoff, not a non-PSD input."""
    cov = np.asarray(covariance, dtype=float)
    eigenvalues, vectors = np.linalg.eigh((cov+cov.T)/2)
    if eigenvalues.min() < -1e-10*max(1., np.max(np.abs(cov))):
        raise ValueError('Covariance must be positive semidefinite')
    return np.sqrt(np.maximum(eigenvalues, 0))[:, None]*vectors.T
