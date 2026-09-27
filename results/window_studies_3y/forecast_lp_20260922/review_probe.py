"""Independent pairwise-drawdown LP verification, owned by read-only reviewer."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent/'run_direct_lp.py').is_file())))
import numpy as np
from scipy.optimize import linprog
from portfolio.linear_decision import LPConstraints, build_lp_context, solve_linear_portfolio


def independent(history, mu, constraints):
    n, p = history.shape
    # Variables: signed asset weights, gross auxiliaries, MAD auxiliaries.
    count = 2 * p + n
    rows, rhs = [], []
    def row(w=None, g=None, m=None, bound=0.):
        values = np.zeros(count)
        if w is not None: values[:p] = w
        if g is not None: values[p:2*p] = g
        if m is not None: values[2*p:] = m
        rows.append(values); rhs.append(bound)
    basis = np.eye(p)
    for a in range(p):
        row(w=basis[a], g=-basis[a]); row(w=-basis[a], g=-basis[a])
    row(g=np.ones(p), bound=constraints.leverage_limit)
    centered = history - history.mean(axis=0)
    for t in range(n):
        m = np.zeros(n); m[t] = -1
        row(w=centered[t], m=m); row(w=-centered[t], m=m)
        row(w=-history[t], bound=1-constraints.wealth_floor)
    ew_daily = history.mean(axis=1)
    mad = np.abs(ew_daily - ew_daily.mean()).mean() * constraints.mad_ratio
    row(m=np.ones(n)/n, bound=mad)
    cumulative = np.vstack([np.zeros(p), history.cumsum(axis=0)])
    ew_cumulative = np.r_[0., ew_daily.cumsum()]
    dd = (np.maximum.accumulate(ew_cumulative)-ew_cumulative).max()*constraints.drawdown_ratio
    for t in range(1,n+1):
        for s in range(t): row(w=cumulative[s]-cumulative[t], bound=dd)
    equality = np.zeros((1,count)); equality[0,:p] = 1
    result=linprog(np.r_[-mu, np.zeros(p+n)], A_ub=rows, b_ub=rhs,
                   A_eq=equality, b_eq=[1.], bounds=[(-constraints.position_limit,constraints.position_limit)]*p+[(0,None)]*(p+n),
                   method='highs', options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not result.success: raise AssertionError(result.message)
    return -result.fun


rng = np.random.default_rng(9924)
gaps=[]
for case in range(40):
    n,p=8,3
    history=rng.normal(.0004,.02,(n,p))
    mu=rng.normal(.001,.004,p)
    constraints=LPConstraints(position_limit=.8, mad_ratio=1.+.5*(case%3),drawdown_ratio=1.+.5*((case//3)%3))
    context=build_lp_context(history,constraints)
    weights,diag=solve_linear_portfolio(context,mu)
    optimum=independent(history,mu,constraints)
    gap=optimum-mu@weights
    gaps.append(gap)
    if not (-2e-9<=gap<=1.2e-10): raise AssertionError((case,gap,weights,diag))
    realized=history@weights
    cum=np.r_[0.,realized.cumsum()]
    dd=np.max(np.maximum.accumulate(cum)-cum)
    mad=np.mean(np.abs(realized-realized.mean()))
    if mad>context.mad_limit+1e-10 or dd>context.drawdown_limit+1e-10: raise AssertionError((case,mad,dd,diag))
print('40 independent random pairwise-drawdown LP comparisons passed; maximum primary-to-final gap:',max(gaps))
