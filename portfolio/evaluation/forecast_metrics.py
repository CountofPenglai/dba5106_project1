"""Score next-day return forecasts and corresponding top-20 rankings."""
import json

import numpy as np
import pandas as pd

from portfolio.evaluation.portfolio_metrics import summarize

TOP = 20


def top_mask(frame, count):
    # Resolve ties in source column order.
    return frame.round(12).rank(axis=1, ascending=False, method='first') <= count


def evaluate_method(actual, predictions, trailing_mean, count=TOP):
    for frame in (predictions, trailing_mean):
        pd.testing.assert_index_equal(actual.index, frame.index)
        pd.testing.assert_index_equal(actual.columns, frame.columns)
    if not all(np.isfinite(frame).all().all() for frame in (actual, predictions, trailing_mean)):
        raise ValueError('Incomplete forecast evaluation grid')
    if not 0 < count < actual.shape[1]:
        raise ValueError('Need at least one selected and one unselected asset')

    sse = np.square(actual - predictions).to_numpy().sum()
    within_asset_ss = np.square(actual - actual.mean()).to_numpy().sum()
    mean_sse = np.square(actual - trailing_mean).to_numpy().sum()
    if within_asset_ss == 0 or mean_sse == 0:
        raise ValueError('A reference has zero squared error; relative R² is undefined')
    selected = top_mask(predictions, count)
    realized_top = top_mask(actual, count)
    baseline_top = top_mask(trailing_mean, count)
    daily = pd.DataFrame({
        'top_return': actual.where(selected).mean(axis=1),
        'rest_return': actual.where(~selected).mean(axis=1),
        'ew_return': actual.mean(axis=1),
        'hits': (selected & realized_top).sum(axis=1),
        'overlap_with_trailing_mean': (selected & baseline_top).sum(axis=1),
        'forecasts_matching_mean': np.isclose(predictions, trailing_mean, atol=1e-12, rtol=0).sum(axis=1),
    })
    daily['selected_assets'] = [json.dumps(actual.columns[row].tolist()) for row in selected.to_numpy()]
    perf = summarize(daily.top_return)
    stats = {
        'pooled_r2': 1 - sse / within_asset_ss,
        'skill_vs_trailing_mean': 1 - sse / mean_sse,
        'mse': sse / actual.size, 'days': len(daily), 'asset_days': actual.size,
        'mean_hits': daily.hits.mean(), 'hit_rate': daily.hits.mean() / count,
        'mean_top_return': perf['mean'], 'top_volatility': perf['volatility'],
        'top_sharpe': perf['sharpe'], 'top_cumulative_return': perf['cumulative_return'],
        'mean_rest_return': daily.rest_return.mean(), 'mean_top_minus_rest': (daily.top_return-daily.rest_return).mean(),
        'days_top_beats_rest': int((daily.top_return > daily.rest_return).sum()),
        'mean_pick_overlap': daily.overlap_with_trailing_mean.mean(),
        'identical_pick_days': int((daily.overlap_with_trailing_mean == count).sum()),
        'forecasts_matching_mean': int(daily.forecasts_matching_mean.sum()),
    }
    return stats, daily
