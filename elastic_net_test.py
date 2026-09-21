"""Rolling Elastic Net portfolio experiment."""
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

DATA=Path('upload/100_Portfolios_10x10_Daily(1).csv')
START,END,WINDOW='2026-01-02','2026-06-30',252
STRENGTH=1e-5

def load_returns(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip()=='Average Equal Weighted Returns -- Daily': break
        lines=[]
        for line in f:
            if not line.strip():
                if lines: break
                continue
            lines.append(line)
    raw=pd.read_csv(StringIO(''.join(lines)),index_col=0)
    raw.columns=raw.columns.str.strip()
    raw=raw.drop(columns=['ME9 BM10','BIG HiBM'],errors='ignore')
    raw.index=pd.to_datetime(raw.index.astype(str),format='%Y%m%d')
    return raw.sort_index().astype(float).replace([-99.99,-999],np.nan).dropna()/100

def fit_weights(history,l1_ratio):
    r=history.to_numpy(); n,p=r.shape
    ew=np.full(p,1/p); N=np.vstack([np.eye(p-1),-np.ones(p-1)])
    X=r@N; y=r@ew
    # sklearn uses 1/(2n)||residual||^2. STRENGTH/2 matches the
    # repository's fixed LASSO strength convention for the L1 term.
    model=ElasticNet(alpha=STRENGTH/2,l1_ratio=l1_ratio,
                     fit_intercept=True,max_iter=100000,tol=1e-8)
    model.fit(X,y)
    return ew-N@model.coef_

def summarize(x):
    x=np.asarray(x); wealth=np.r_[1.,np.cumprod(1+x)]
    dd=wealth/np.maximum.accumulate(wealth)-1
    vol=x.std(ddof=1); mean=x.mean()
    return {'mean_daily':mean,'volatility':vol,'sharpe':mean/vol,
            'negative_sharpe':-mean/vol,
            'cumulative_return':wealth[-1]-1,'max_drawdown':dd.min()}

def run(l1_ratio):
    ret=load_returns(DATA); values=[]
    for date in ret.loc[START:END].index:
        hist=ret.loc[:date].iloc[:-1].tail(WINDOW)
        values.append(float(fit_weights(hist,l1_ratio)@ret.loc[date].to_numpy()))
    return summarize(values)

if __name__=='__main__':
    rows={}
    for ratio in (0.25,0.5,0.75):
        label=f'ElasticNet_l1ratio_{ratio:g}'
        rows[label]=run(ratio)
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x:f'{x:.8f}'))
