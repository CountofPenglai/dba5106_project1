"""Audit frozen July LP artifacts and cost-adjust existing July holdings without refits."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from portfolio.data.returns_loader import load_returns
from portfolio.paths import DEFAULT_CSV, PROJECT_ROOT
from portfolio.evaluation.provenance import file_digest
from portfolio.evaluation.artifact_store import RunArtifacts

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--original', type=Path, default=PROJECT_ROOT/'results/final_test_july_2026_20260923')
args = parser.parse_args()
ROOT, OLD = args.result, args.original
data = load_returns(DEFAULT_CSV, start='2021-08-01', end='2026-07-31')
old_summary = pd.read_csv(OLD/'final_test_summary.csv', index_col=0)
new_summary = pd.read_csv(ROOT/'metrics/final_test_summary.csv', index_col=0)
expected_dates = pd.bdate_range('2026-07-01', '2026-07-31').difference(pd.DatetimeIndex(['2026-07-03']))

def read(path):
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')

def stats(r):
    r = np.asarray(r)
    wealth = np.r_[1., np.cumprod(1+r)]
    std = r.std(ddof=1)
    return dict(mean=r.mean(), volatility=std, sharpe=r.mean()/std,
                negative_sharpe=-r.mean()/std, cumulative_return=wealth[-1]-1,
                max_drawdown=(wealth/np.maximum.accumulate(wealth)-1).min())

sources = {}
rows = []
max_errors = dict(gross_return=0., net_return=0., cost=0., metrics=0.)
for model in [*old_summary.index, 'LP_Current']:
    current = model == 'LP_Current'
    weight_path = ROOT/'weights/LP_Current_weights.csv' if current else OLD/'weights'/f'{model}.csv'
    daily_path = ROOT/'returns/LP_Current_returns.csv' if current else OLD/'returns'/f'{model}.csv'
    w = read(weight_path)
    saved = read(daily_path)
    assert w.index.equals(expected_dates) and saved.index.equals(expected_dates)
    assert list(w.columns) == list(data.columns) and w.shape == (22, 98)
    assert np.isfinite(w.to_numpy()).all()
    previous = np.full(98, 1/98)
    daily = []
    for day, weights in w.iterrows():
        actual = data.loc[day].to_numpy()
        allocation = weights.to_numpy()
        gross = allocation @ actual
        turnover = np.abs(allocation-previous).sum()/2
        cost = .0005 * 2 * turnover
        daily.append(dict(gross_return=gross, cost=cost, net_return=gross-cost, turnover=turnover))
        previous = allocation*(1+actual)/(1+gross)
    frame = pd.DataFrame(daily, index=w.index)
    gross_saved = saved['gross_return'] if current else saved['return']
    err = float(np.max(np.abs(frame.gross_return-gross_saved)))
    assert err < 1e-12
    max_errors['gross_return'] = max(max_errors['gross_return'], err)
    if current:
        for key, col in [('net_return','return'), ('cost','cost')]:
            err = float(np.max(np.abs(frame[key]-saved[col])))
            assert err < 1e-12
            max_errors[key] = err
    gross, net = stats(frame.gross_return), stats(frame.net_return)
    reference = new_summary.loc[model] if current else old_summary.loc[model]
    for key, value in gross.items():
        err = abs(float(reference[('gross_' if current else '')+key])-value)
        assert err < 1e-12
        max_errors['metrics'] = max(max_errors['metrics'], err)
    if current or model == 'EW':
        for key, value in net.items():
            assert abs(float(new_summary.loc[model, key])-value) < 1e-12
    if not current:
        assert abs(reference.average_daily_turnover-frame.turnover.mean()) < 1e-12
    row = dict(model=model, **{'gross_'+k:v for k,v in gross.items()},
               **{'net_'+k:v for k,v in net.items()}, average_turnover=frame.turnover.mean(),
               average_gross_leverage=w.abs().sum(axis=1).mean(), evaluation_days=22,
               start='2026-07-01', end='2026-07-31', transaction_cost_rate=.0005,
               result_origin='frozen_current_lp_evaluation' if current else 'existing_holdings_cost_recalculation')
    rows.append(row)
    frame.to_csv(ROOT/'returns'/f'combined_{model}_returns.csv')
    for path in (weight_path,daily_path):
        if not current:
            sources[str(path)] = file_digest(path)

# Independently check all 22 current-LP allocations against actual prior observations.
w = read(ROOT/'weights/LP_Current_weights.csv')
diag = read(ROOT/'diagnostics/LP_Current_diagnostics.csv')
forecast_diag = read(ROOT/'diagnostics/forecast_diagnostics.csv')
max_gross, max_position, max_budget_error = 0., 0., 0.
for day, weights in w.iterrows():
    history = data.loc[data.index < day].tail(252)
    assert len(history) == 252
    for table, begin, end, count in [(diag,'history_start','history_end','history_count'),
                                     (forecast_diag,'train_start','train_end','train_count')]:
        assert pd.Timestamp(table.loc[day, begin]) == history.index[0]
        assert pd.Timestamp(table.loc[day, end]) == history.index[-1] < day
        assert table.loc[day, count] == 252
    assert forecast_diag.loc[day, 'alpha'] == 0.002511886431509582
    r = history.to_numpy() @ weights.to_numpy()
    ew = history.mean(axis=1).to_numpy()
    mad = np.mean(np.abs(r-r.mean()))
    mad_limit = np.mean(np.abs(ew-ew.mean()))
    def dd_proxy(x):
        path = np.r_[0., np.cumsum(x)]
        return (np.maximum.accumulate(path)-path).max()
    assert abs(weights.sum()-1) < 1e-9
    assert weights.abs().sum() <= 1.5 + 1e-9
    assert weights.abs().max() <= .05 + 1e-9
    assert mad <= mad_limit + 1e-9
    assert dd_proxy(r) <= dd_proxy(ew) + 1e-9
    assert np.min(1+r) >= 1e-6-1e-9
    assert diag.loc[day,'solver_status'] == 'optimal' and diag.loc[day,'feasible']
    max_gross = max(max_gross, weights.abs().sum())
    max_position = max(max_position, weights.abs().max())
    max_budget_error = max(max_budget_error, abs(weights.sum()-1))

selection = json.loads((ROOT/'config/frozen_decisions.json').read_text())
assert selection == json.loads((ROOT/'config/pre_run_selection.json').read_text())
assert selection['parameter_search'] is False
assert selection['objective'] == dict(mad_penalty=0., transaction_cost=.0005, ew_penalty=0.)
assert selection['test_period_previously_reviewed'] and not selection['independent_untouched_test']
pd.DataFrame(rows).set_index('model').to_csv(ROOT/'metrics/combined_final_test_summary.csv')
report = dict(passed=True, checked_lp_allocations=22, checked_forecast_boundaries=22,
              checked_cost_accounting_paths=7, checked_portfolio_days=154,
              all_assets=98, forecast_and_risk_window=252, parameter_search=False,
              maximum_errors=max_errors, max_gross=float(max_gross), max_position=float(max_position),
              max_budget_error=float(max_budget_error), original_result_sha256=sources,
              old_models_refitted=False, accounting='gross minus .0005 times full L1 trades; start from EW; gross holdings drift',
              interpretation='July previously reviewed; no new untouched-test claim')
(ROOT/'config/verification.json').write_text(json.dumps(report, indent=2)+'\n')
RunArtifacts(ROOT,'categorized').finalize()
print(pd.DataFrame(rows).set_index('model').to_string())
print(json.dumps({k:v for k,v in report.items() if k!='original_result_sha256'},indent=2))
