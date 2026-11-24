"""Reproduce the handout checks using the archived daily equal-weighted panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.config import EVALUATION_PERIODS, HISTORY_START
from portfolio.data.history_windows import rolling_windows
from portfolio.data.preprocessing import DEFAULT_DROP_ASSETS
from portfolio.data.returns_loader import MODEL_SECTION, load_returns
from portfolio.engine.rolling_backtest import backtest
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.provenance import file_digest
from portfolio.model_catalog import MODELS, fit_model
from portfolio.paths import DEFAULT_CSV


def verify(csv):
    start, end = EVALUATION_PERIODS['validation']
    all_assets = load_returns(csv, drop_assets=(), end=end)
    official = load_returns(csv, start=HISTORY_START, end=end)
    if len(all_assets.columns) != 100 or tuple(all_assets.columns[[89, 99]]) != DEFAULT_DROP_ASSETS:
        raise ValueError('Expected the original 100-asset French table with the two documented exclusions')
    pd.testing.assert_frame_equal(official, all_assets.drop(columns=list(DEFAULT_DROP_ASSETS)).loc[HISTORY_START:])
    date, history = next(rolling_windows(official, start=start, end=end, window=126))
    first = history.iloc[:, :6]
    w = fit_model(first, MODELS['MinVar']).weights
    realized_pct = float(w @ official.loc[date, first.columns] * 100)
    expected = [-.111, -.199, .348, .460, .006, .496]
    checks = {
        'row_number_is_26152': int(all_assets.index.get_loc(date)) + 1 == 26152,
        'first_window': (str(history.index[0].date()), str(history.index[-1].date()), len(history)) ==
                        ('2025-07-03', '2025-12-31', 126),
        'six_weights_match_handout_at_3_decimals': w.round(3).tolist() == expected,
        'return_matches_handout_at_3_decimals_percent': round(realized_pct, 3) == 1.393,
    }
    ref = {'EW': .111, 'MinVar': .021}
    comparisons = []
    for panel in (all_assets, official):
        # Independent analytic solution of the minimum-variance problem.
        ones = np.ones(panel.shape[1])
        analytic_returns = []
        for day, prior in rolling_windows(panel, start=start, end=end, window=126):
            inv_one = np.linalg.solve(prior.cov().to_numpy(), ones)
            analytic_returns.append((inv_one / inv_one.sum()) @ panel.loc[day].to_numpy())
        independent = {'MinVar': analytic_returns, 'EW': panel.loc[start:end].mean(axis=1)}
        for name, target in ref.items():
            run = backtest(panel, lambda h: fit_model(h, MODELS[name]).weights,
                           start=start, end=end, window=126)
            np.testing.assert_allclose(run.returns, independent[name], rtol=0, atol=1e-10)
            sharpe = summarize(run.returns)['sharpe']
            comparisons.append(dict(assets=panel.shape[1], model=name, sharpe=sharpe,
                                    handout_100_asset_reference=target if panel.shape[1] == 100 else None,
                                    matches_at_3_decimals=(round(sharpe, 3) == target)
                                    if panel.shape[1] == 100 else None))
    return dict(sha256=file_digest(csv), section=MODEL_SECTION, excluded=DEFAULT_DROP_ASSETS,
                history_first=str(official.index[0].date()), history_last=end, eval_days=len(official.loc[start:end]),
                retained_assets=official.shape[1], missing_retained_cells=int(official.isna().sum().sum()),
                row_number=int(all_assets.index.get_loc(date)) + 1, first_window_dates=[str(x.date())
                for x in (history.index[0], history.index[-1])], first_window_days=len(history),
                first_six_weights=w.to_dict(), first_day_return_pct=realized_pct, first_day_checks=checks,
                sharpe_comparison=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    result = verify(parser.parse_args().csv)
    print(json.dumps(result, indent=2))
    if not all(result['first_day_checks'].values()) or result['missing_retained_cells']:
        raise SystemExit('Core course benchmark check failed; inspect the published data version and panel.')
