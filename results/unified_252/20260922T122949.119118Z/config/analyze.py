"""Independently reconstruct uniform-window validation results from holdings."""
from pathlib import Path
import argparse
import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,default=Path('/Users/zhengce/projects/fundation_ba/project1'))
parser.add_argument('--run-root',type=Path)
parser.add_argument('--output',type=Path,default=Path('/private/tmp/project1-unified252/analysis'))
args=parser.parse_args()
PROJECT = args.project.resolve()
sys.path.insert(0, str(PROJECT))
from portfolio.data.returns_loader import load_returns
from portfolio.evaluation.artifact_store import artifact_path
from portfolio.evaluation.provenance import file_digest, verify_code_hashes

ROOT = args.run_root or Path(Path('/private/tmp/project1-unified252/current_run.txt').read_text())
DEST = args.output
DEST.mkdir(parents=True,exist_ok=True)
START, END = '2026-01-02', '2026-06-30'
returns = load_returns(PROJECT/'100_Portfolios_10x10_Daily.csv', start='2021-08-01', end=END)
actual = returns.loc[START:END]
assert actual.shape == (123,98) and np.isfinite(returns.to_numpy()).all()
expected = dict(baselines=11, loss_search=7, recency=5, elastic_net=11, pca=12, market_beta=6, mixture=10, lp=3)
status = json.loads((ROOT/'config/run_status.json').read_text())
assert all(s['status']=='complete' for s in status.values()), status
rows, checks, return_series, weight_arrays, window_checks = [], [], {}, {}, []
current_data_hash = file_digest(PROJECT/'100_Portfolios_10x10_Daily.csv')
assert current_data_hash == json.loads((ROOT/'config/data_audit.json').read_text())['sha256']
history = {day:returns.loc[returns.index<day].tail(252).to_numpy() for day in actual.index}
covariances = {day:LedoitWolf().fit(h).covariance_ for day,h in history.items()}
equal = np.ones(98)/98

def summarize(r):
    mean, vol = float(r.mean()), float(r.std(ddof=1))
    wealth = np.r_[1.,np.cumprod(1+r)]
    return dict(mean=mean, volatility=vol, sharpe=mean/vol, negative_sharpe=-mean/vol,
                cumulative_return=float(wealth[-1]-1), max_drawdown=float((wealth/np.maximum.accumulate(wealth)-1).min()))

def selection(family,name):
    if family=='loss_search' and (name.startswith('Selected_') or name=='LASSO_Selected'):
        return 'validation_selected' if name=='LASSO_Selected' else 'selected_parameter_ablation'
    if family=='recency' and name.endswith('Selected'):
        return 'validation_selected'
    if family=='mixture':
        if name.endswith('ReturnVolatility'): return 'validation_selected'
        if name.endswith('Sharpe'): return 'trained_on_prior_252_days'
        return 'component_control'
    if family in ('elastic_net','pca') or (family=='recency' and '_HL' in name and 'Constrained' in name):
        return 'prespecified_candidate_comparison'
    if '_CV' in name: return 'legacy_5fold_selection_inside_training_window'
    if family=='lp': return 'historically_calibrated' if 'LASSO' in name else 'fixed_settings'
    return 'fixed_settings'

for family,count in expected.items():
    run = ROOT/family
    meta = json.loads(artifact_path(run,'run_metadata.json').read_text())
    assert meta['data_sha256']==current_data_hash, family
    verify_code_hashes(meta['code_sha256'])
    marker=json.loads(artifact_path(run,'benchmark_runner.json').read_text())
    assert marker['window']==252 and marker['validation_start']==START and marker['validation_end']==END
    assert marker['runner_sha256']==file_digest(ROOT/'config/run_family.py')
    if family in ('mixture','lp'):
        assert meta['risk_window']['observations']==252 and meta['risk_window']['years'] is None
        assert all(c['window']['observations']==252 and c['window']['years'] is None for c in meta['forecast_configs'].values())
        if family=='mixture': assert meta['decision_training_days']==252 and meta['validation_days']==123
    else: assert meta['window']==252
    summary_name = 'comparison.csv' if family in ('mixture','lp') else ('market_beta_results.csv' if family=='market_beta' else 'summary.csv')
    exported = pd.read_csv(artifact_path(run,summary_name),index_col=0)
    assert len(exported)==count, (family,len(exported),count)
    if family=='loss_search':
        assert meta['comparison_failures']=={} and (exported.status=='ok').all()
        for phase in ('screen','validation'):
            searched=pd.read_csv(artifact_path(run,phase+'_search.csv'))
            assert (searched.status=='ok').all(),(phase,searched)
    manifest = json.loads((run/'artifact_manifest.json').read_text())['artifacts']
    for name,original in exported.iterrows():
        key=family+'/'+name
        full=pd.read_csv(artifact_path(run,name+'_weights.csv'),index_col=0,parse_dates=True)
        w=full.loc[START:END]
        assert w.index.is_unique and w.columns.is_unique and w.index.equals(actual.index) and w.columns.equals(actual.columns),key
        wa=w.to_numpy()
        assert np.isfinite(wa).all() and np.max(np.abs(wa.sum(axis=1)-1))<=1e-8,key
        r=np.sum(wa*actual.to_numpy(),axis=1)
        assert np.min(1+r)>0,key
        saved=pd.read_csv(artifact_path(run,name+'_returns.csv'),index_col=0,parse_dates=True).loc[START:END].iloc[:,0]
        np.testing.assert_allclose(r,saved.to_numpy(),rtol=0,atol=1e-12,err_msg=key)
        metrics=summarize(r)
        errors=[]
        for metric,value in metrics.items():
            original_name={'mean':'mean_daily_return' if family=='market_beta' else ('mean_daily' if family in ('pca','elastic_net') else 'mean'),
                           'max_drawdown':'maximum_drawdown' if family=='market_beta' else 'max_drawdown'}.get(metric,metric)
            if original_name in original:
                error=abs(value-float(original[original_name]));errors.append(error)
                assert error<1e-10,(key,metric,error)
        gross=np.abs(wa).sum(axis=1)
        before=np.empty_like(wa);before[0]=equal
        before[1:]=wa[:-1]*(1+actual.to_numpy()[:-1])/(1+r[:-1,None])
        turnover=.5*np.abs(wa-before).sum(axis=1)
        beta=np.cov(r,actual.mean(axis=1),ddof=1)[0,1]/actual.mean(axis=1).var(ddof=1)
        alias=next((prior for prior,values in weight_arrays.items() if np.max(np.abs(values-wa))<1e-10),'')
        weight_arrays[key]=wa
        return_series[key]=pd.Series(r,index=actual.index)
        row=dict(family=family,model=name,key=key,selection=selection(family,name),equivalent_holdings=alias,
                 **metrics,average_gross_leverage=float(gross.mean()),max_gross_leverage=float(gross.max()),
                 max_absolute_position=float(np.abs(wa).max()),average_daily_turnover=float(turnover.mean()),
                 realized_beta_to_EW=float(beta),evaluation_days=len(r),window=252,start=START,end=END,
                 weights_path=str(artifact_path(run,name+'_weights.csv').relative_to(ROOT)),
                 returns_path=str(artifact_path(run,name+'_returns.csv').relative_to(ROOT)))
        rows.append(row)
        constrained = family in ('market_beta','loss_search','mixture','lp') or 'Constrained' in name or name=='LASSO_LogGrowth' or (family=='recency' and name.endswith('Selected'))
        if family=='loss_search' and name in ('EW','LASSO'):constrained=False
        checks.append(dict(family=family,model=name,return_reconstruction_error=float(np.max(np.abs(r-saved.to_numpy()))),
                           metric_reconstruction_error=max(errors),max_budget_error=float(np.max(np.abs(wa.sum(axis=1)-1))),constrained=constrained))
        if constrained:
            assert gross.max()<=1.5+1e-7,(key,'gross',gross.max())
            half=None
            if family=='recency':
                if name.endswith('Selected'):half=meta['selected_half_life']
                else:half=int(re.search(r'HL(\d+)',name).group(1))
            max_risk_ratio=0.;min_wealth=10.;max_beta_error=0.;max_dd_ratio=0.
            for day,weights in zip(w.index,wa):
                h=history[day];hr=h@weights;er=h@equal
                assert len(h)==252
                min_wealth=min(min_wealth,float(np.min(1+hr)))
                if family=='mixture':
                    cov=covariances[day];ratio=float(weights@cov@weights/(equal@cov@equal))
                elif family=='lp':
                    ratio=float(np.abs(hr-hr.mean()).mean()/np.abs(er-er.mean()).mean())
                    def dd(values):
                        c=np.r_[0.,np.cumsum(values)]
                        return float(np.max(np.maximum.accumulate(c)-c))
                    dd_ratio=dd(hr)/dd(er);max_dd_ratio=max(max_dd_ratio,dd_ratio)
                    assert dd_ratio<=1+1e-6,(key,day,'historical dd',dd_ratio)
                else:
                    q=np.ones(252) if half is None else np.exp2((np.arange(252)-251)/half)
                    q=q/q.sum()
                    ratio=float(np.sum(q*(hr-q@hr)**2)/np.sum(q*(er-q@er)**2))
                max_risk_ratio=max(max_risk_ratio,ratio)
                assert ratio<=1+1e-6,(key,day,'historical risk',ratio)
                if family=='market_beta':
                    b=float((er-er.mean())@(hr-hr.mean())/np.sum((er-er.mean())**2))
                    bounds=(.8,1.2) if 'market_like' in name else ((.5,.9) if 'conservative' in name else None)
                    if bounds:
                        error=max(0.,bounds[0]-b,b-bounds[1]);max_beta_error=max(max_beta_error,error)
                        assert error<=1e-6,(key,day,'beta',b)
            assert min_wealth>=1e-6-1e-8,(key,'wealth',min_wealth)
            if family=='lp':assert np.abs(wa).max()<=.05+1e-7,key
            checks[-1].update(max_historical_risk_ratio=max_risk_ratio,min_historical_wealth=min_wealth,
                              max_beta_bound_error=max_beta_error,max_lp_historical_drawdown_ratio=max_dd_ratio)
    for logical in manifest:
        if logical.endswith('_diagnostics.csv') and not logical.startswith('search/'):
            diag=pd.read_csv(artifact_path(run,logical),index_col=0)
            for col in ('observations','history_count','train_count'):
                if col in diag:
                    assert (diag[col]==252).all(),(family,logical,col)
                    window_checks.append(dict(family=family,artifact=logical,column=col,rows=len(diag),minimum=int(diag[col].min()),maximum=int(diag[col].max())))
            for col in ('history_end','train_end'):
                if col in diag:
                    assert (pd.to_datetime(diag[col])<pd.to_datetime(diag.index)).all(),(family,logical,col)
        if logical.endswith('_alpha_cv.csv'):
            cv=pd.read_csv(artifact_path(run,logical),index_col=0)
            assert len(cv)==63 and (cv.train_count==252).all() and (cv.validation_count==21).all()
            assert (pd.to_datetime(cv.train_end)<pd.to_datetime(cv.validation_start)).all()
            window_checks.append(dict(family=family,artifact=logical,column='train_count',rows=len(cv),minimum=252,maximum=252))

table=pd.DataFrame(rows).set_index('key')
table.sort_values('sharpe',ascending=False).to_csv(DEST/'leaderboard_by_sharpe.csv')
table.sort_values('cumulative_return',ascending=False).to_csv(DEST/'leaderboard_by_return.csv')
unique=table[table.equivalent_holdings==''].sort_values('sharpe',ascending=False)
unique.to_csv(DEST/'unique_portfolios.csv')
pd.DataFrame(checks).to_csv(DEST/'reconstruction_checks.csv',index=False)
pd.DataFrame(window_checks).to_csv(DEST/'window_checks.csv',index=False)
pd.DataFrame(return_series).to_csv(DEST/'validation_returns.csv')
best=pd.concat([table.loc[table.groupby('family')[metric].idxmax()].assign(selected_by=metric) for metric in ('sharpe','cumulative_return')])
best.to_csv(DEST/'family_highlights.csv')
audit=dict(model_rows=len(table),unique_holdings=len(unique),same_holdings_tolerance=1e-10,
           families=len(expected),validation_days=123,assets=98,data_sha256=current_data_hash,
           max_return_reconstruction_error=max(x['return_reconstruction_error'] for x in checks),
           max_metric_reconstruction_error=max(x['metric_reconstruction_error'] for x in checks),
           max_budget_error=max(x['max_budget_error'] for x in checks),source_hashes_verified=True,
           decision_training_days=252,first_validation_day_initial_holdings='EW for all turnover calculations',
           checks='dates, columns, finite holdings/returns, full investment, saved returns/metrics, source/data identity, training counts, chronology, hard constraints',
           final_test_evaluated=False)
(DEST/'verification.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
print(unique[['family','model','cumulative_return','sharpe','volatility','max_drawdown','average_gross_leverage']].head(18).to_string(index=False))
