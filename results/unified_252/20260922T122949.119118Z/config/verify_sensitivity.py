"""Verify four appended 252-day studies and merge only their novel variants."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio.data.returns_loader import load_returns
from portfolio.evaluation.artifact_store import artifact_path
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.provenance import verify_code_hashes

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
START, END, WINDOW = '2026-01-02', '2026-06-30', 252
EXPECTED = {
    'turnover': ['EW', 'LASSO_Selected', 'LASSO_Turnover_0p1', 'LASSO_Turnover_1'],
    'smoothness': ['EW', 'LASSO_Selected', 'LASSO_Smooth_0p1', 'LASSO_Smooth_1'],
    'shrinkage': ['EW', 'LASSO_Selected', 'LASSO_LedoitWolf'],
    'concentration': ['EW', 'LASSO_Selected', 'LASSO_WeightCap_10pct'],
}
CSV = PROJECT / '100_Portfolios_10x10_Daily.csv'
source = load_returns(CSV).loc[:END]
actual = source.loc[START:END]
old = pd.read_csv(ROOT / 'metrics/leaderboard_by_sharpe.csv')
old_selected = pd.read_csv(artifact_path(ROOT/'loss_search','LASSO_Selected_weights.csv'),
                           index_col=0, parse_dates=True).loc[START:END]
old_ew = pd.read_csv(artifact_path(ROOT/'baselines','EW_weights.csv'),
                     index_col=0, parse_dates=True).loc[START:END]
old_selected_returns = pd.read_csv(artifact_path(ROOT/'loss_search','LASSO_Selected_returns.csv'),
                                   index_col=0, parse_dates=True).loc[START:END].iloc[:,0]
assert len(old) == 65 and len(actual) == 123 and len(actual.columns) == 98
assert actual.index.max() == pd.Timestamp(END)
sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
assert sha == '11399a830c7b2a192d652b39e866a74c05d7380dd6edf1573ce1c40f9d271c23'
checks = {'original_rows_preserved':len(old), 'new_families':4, 'new_variants':6,
          'validation_days':len(actual), 'assets':len(actual.columns), 'window':WINDOW,
          'data_sha256':sha, 'final_test_evaluated':False,
          'max_return_reconstruction_error':0., 'max_metric_reconstruction_error':0.,
          'max_budget_error':0., 'max_baseline_weight_error':0.,
          'maximum_training_risk_ratio':0., 'solver_statuses':{}, 'families':{}}
rows=[]
flat_code_hash=None
for family,names in EXPECTED.items():
    run = ROOT / family
    meta = json.loads(artifact_path(run,'run_metadata.json').read_text())
    assert meta['window']==WINDOW and meta['actual_start']==START and meta['actual_end']==END
    assert meta['evaluation_days']==len(actual) and meta['data_sha256']==sha
    assert not meta['final_test']['evaluated'] and meta['final_test']['start']=='2026-08-01'
    assert meta['artifact_layout']=='categorized' and len(meta['assets'])==98
    verify_code_hashes(meta['code_sha256'])
    if flat_code_hash is None: flat_code_hash=meta['code_sha256']
    assert flat_code_hash==meta['code_sha256']
    summary = pd.read_csv(artifact_path(run,'summary.csv'),index_col=0)
    assert list(summary.index)==names, (family,list(summary.index))
    fcheck={'models':len(names), 'new_variants':len(names)-2,
            'maximum_training_risk_ratio':0., 'maximum_weight_error':0.}
    for name in names:
        weights_path=artifact_path(run,f'{name}_weights.csv')
        returns_path=artifact_path(run,f'{name}_returns.csv')
        w=pd.read_csv(weights_path,index_col=0,parse_dates=True)
        r_saved=pd.read_csv(returns_path,index_col=0,parse_dates=True).iloc[:,0]
        assert w.index.equals(actual.index) and r_saved.index.equals(actual.index)
        assert w.columns.equals(actual.columns)
        wa=w.to_numpy(dtype=float)
        assert np.isfinite(wa).all()
        budget_error=float(np.max(np.abs(wa.sum(axis=1)-1)))
        checks['max_budget_error']=max(checks['max_budget_error'],budget_error)
        assert budget_error<1e-8
        daily=(wa*actual.to_numpy()).sum(axis=1)
        return_error=float(np.max(np.abs(daily-r_saved.to_numpy())))
        checks['max_return_reconstruction_error']=max(checks['max_return_reconstruction_error'],return_error)
        assert return_error<1e-12
        computed=summarize(daily)
        metric_error=max(abs(computed[key]-float(summary.loc[name,key])) for key in computed)
        checks['max_metric_reconstruction_error']=max(checks['max_metric_reconstruction_error'],metric_error)
        assert metric_error<1e-10
        if name=='EW':
            baseline_error=float(np.max(np.abs(wa-old_ew.to_numpy())))
            assert baseline_error<1e-10
        if name=='LASSO_Selected':
            baseline_error=float(np.max(np.abs(wa-old_selected.to_numpy())))
            assert baseline_error<1e-8,(family,baseline_error)
            assert np.max(np.abs(daily-old_selected_returns.to_numpy()))<1e-10
        if name in ('EW','LASSO_Selected'):
            checks['max_baseline_weight_error']=max(checks['max_baseline_weight_error'],baseline_error)
        if name != 'EW':
            assert np.abs(wa).sum(axis=1).max()<=1.5+1e-7
            diag=pd.read_csv(artifact_path(run,f'{name}_diagnostics.csv'),index_col=0,parse_dates=True)
            assert diag.index.equals(actual.index) and len(diag)==123
            assert diag.solver_status.eq('optimal').all()
            for solver,count in diag.solver.value_counts().items():
                checks['solver_statuses'][solver]=checks['solver_statuses'].get(solver,0)+int(count)
            assert diag.min_wealth.min()>=1e-6-1e-8
            if name=='LASSO_WeightCap_10pct':
                assert np.abs(wa).max()<=.1+1e-7
            for day,values in zip(actual.index,wa):
                hist=source.loc[source.index<day].tail(WINDOW)
                assert len(hist)==WINDOW and hist.index[-1]<day
                if name=='LASSO_LedoitWolf':
                    cov=LedoitWolf().fit(hist.to_numpy()).covariance_
                    risk=float(values@cov@values)
                    ew=np.full(len(values),1/len(values))
                    ew_risk=float(ew@cov@ew)
                else:
                    returns=hist.to_numpy()@values
                    risk=float(np.var(returns,ddof=0))
                    ew_risk=float(np.var(hist.to_numpy().mean(axis=1),ddof=0))
                ratio=risk/ew_risk
                fcheck['maximum_training_risk_ratio']=max(fcheck['maximum_training_risk_ratio'],ratio)
                assert ratio <= 1+1e-6,(family,name,day,ratio)
            checks['maximum_training_risk_ratio']=max(checks['maximum_training_risk_ratio'],fcheck['maximum_training_risk_ratio'])
        if name in ('EW','LASSO_Selected'):
            continue
        gross=np.abs(wa).sum(axis=1)
        previous=np.empty_like(wa)
        previous[0]=np.full(wa.shape[1],1/wa.shape[1])
        previous[1:]=wa[:-1]*(1+actual.to_numpy()[:-1])/(1+daily[:-1,None])
        turnover=.5*np.abs(wa-previous).sum(axis=1)
        beta=np.cov(daily,actual.mean(axis=1),ddof=1)[0,1]/actual.mean(axis=1).var(ddof=1)
        row=dict(key=f'{family}/{name}',family=family,model=name,
                 selection='prespecified_validation_sensitivity',equivalent_holdings='',
                 **computed,average_gross_leverage=float(gross.mean()),
                 max_gross_leverage=float(gross.max()),max_absolute_position=float(np.abs(wa).max()),
                 average_daily_turnover=float(turnover.mean()),realized_beta_to_EW=float(beta),
                 evaluation_days=len(daily),window=WINDOW,start=START,end=END,
                 weights_path=str(weights_path.relative_to(ROOT)),
                 returns_path=str(returns_path.relative_to(ROOT)))
        rows.append(row)
    checks['families'][family]=fcheck
assert len(rows)==6
combined=pd.concat([old,pd.DataFrame(rows)[old.columns]],ignore_index=True)
assert len(combined)==71 and combined.key.is_unique
combined=combined.sort_values('sharpe',ascending=False,kind='stable').reset_index(drop=True)
new=pd.DataFrame(rows).sort_values('sharpe',ascending=False)
combined.to_csv(ROOT/'metrics/expanded_leaderboard_by_sharpe.csv',index=False)
new.to_csv(ROOT/'metrics/sensitivity_variants.csv',index=False)
(ROOT/'config/sensitivity_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(new[['family','model','cumulative_return','sharpe','volatility','max_drawdown','average_gross_leverage']].to_string(index=False))
print(json.dumps(checks,indent=2))
