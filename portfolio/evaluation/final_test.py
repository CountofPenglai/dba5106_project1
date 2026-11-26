"""Evaluate the four prespecified portfolios on the continuous August–September holdout."""
from dataclasses import asdict
from copy import deepcopy
import json

import pandas as pd

from portfolio.config import COURSE_WINDOW
from portfolio.data.returns_loader import load_returns
from portfolio.engine.report_runner import run_models
from portfolio.evaluation.artifact_store import fresh_run, save_run
from portfolio.evaluation.portfolio_metrics import summarize
from portfolio.evaluation.provenance import file_digest, run_provenance
from portfolio.model_catalog import MODELS
from portfolio.models.course_cross_validation import COURSE_ALPHAS, CV_DAYS, CV_SPLITS
from portfolio.paths import ORIGINAL_SHA256, PROJECT_ROOT, TEST_SHA256

PROTOCOL = PROJECT_ROOT / 'configs/final_test_126.json'


def monthly_rank_check(panel, holdout, window):
    records, detail = [], []
    for month, actual in holdout.groupby(holdout.index.strftime('%Y-%m')):
        prior = panel.loc[panel.index < actual.index[0]].tail(window)
        means = pd.DataFrame({'prior_mean': prior.mean(), 'realized_mean': actual.mean()})
        ranks = means.round(12).rank(ascending=False, method='average')
        records.append({'month': month, 'as_of': prior.index[-1].date(),
                        'history_days': len(prior), 'realized_days': len(actual),
                        'assets': len(means), 'spearman': ranks.prior_mean.corr(ranks.realized_mean)})
        detail.append(means.join(ranks.add_suffix('_rank')).rename_axis('asset')
                      .reset_index().assign(month=month))
    return pd.DataFrame(records), pd.concat(detail, ignore_index=True)


def evaluate(output):
    with fresh_run(output) as stage:
        _evaluate(stage)


def _evaluate(output):
    rules = json.loads(PROTOCOL.read_text())
    models = [MODELS[entry['name']] for entry in rules['models']]
    cv = rules['course_cv']
    # Adapt the frozen protocol's original schema without rewriting the pre-test record.
    frozen = deepcopy(rules['models'])
    for entry in frozen:
        for key in ('title', 'section'):
            entry.pop(key)
        inactive = [entry.pop(key) for key in ('market_beta', 'coefficient_constraint',
                                               'transaction_cost', 'short_budget', 'winner_floor')]
        inactive += [entry['constraints'].pop('beta_bounds'), entry['loss'].pop('reward_strength'),
                     entry['loss'].pop('turnover_strength')]
        if any(value not in (None, False, 0) for value in inactive):
            raise ValueError('Frozen protocol requires an unsupported model feature')
    settings = [asdict(model) for model in models]
    for spec in settings:
        spec.pop('title')
    if (rules['window'] != COURSE_WINDOW or settings != frozen
            or cv['alphas'] != COURSE_ALPHAS.tolist()
            or (cv['folds'], cv['days_per_fold']) != (CV_SPLITS, CV_DAYS)):
        raise ValueError('Model settings have changed since the test rules were frozen')

    historical_file = PROJECT_ROOT / rules['history_file']
    updated_file = PROJECT_ROOT / rules['test_file']
    provenance = run_provenance(historical_file)
    updated_digest = file_digest(updated_file)
    if provenance['data_sha256'] != ORIGINAL_SHA256 or updated_digest != TEST_SHA256:
        raise ValueError('The data releases differ from the original holdout evaluation')
    old = load_returns(historical_file, start=rules['history_start'], end=rules['history_end'])
    update = load_returns(updated_file, start=rules['history_start'], end=rules['test_end'])
    for data in (old, update):
        if data.columns.tolist() != rules['assets']:
            raise ValueError('Asset universe differs from the frozen test protocol')
    actual = update.loc[rules['test_start']:rules['test_end']]
    counts = actual.groupby(actual.index.strftime('%Y-%m')).size().to_dict()
    if (counts != rules['expected_month_counts'] or old.index[-1] != pd.Timestamp(rules['history_end'])
            or actual.index[0] != pd.Timestamp(rules['test_start'])
            or actual.index[-1] != pd.Timestamp(rules['test_end'])):
        raise ValueError('Incomplete dates in the historical or holdout panel')
    overlap = old.index.intersection(update.index)
    revised_cells = int(old.loc[overlap].ne(update.loc[overlap]).to_numpy().sum())
    panel = pd.concat([old, actual], verify_integrity=True)
    window = rules['window']
    run = run_models(panel, models, start=rules['test_start'], end=rules['test_end'],
                     progress=print)

    days = pd.Series(actual.index.strftime('%Y-%m'), index=actual.index)
    portfolio_results = []
    for model in models:
        name = model.name
        returns = run.returns[name]['return']
        for period, sample in [*returns.groupby(days), ('combined', returns)]:
            metric = summarize(sample)
            portfolio_results.append({'model': name, 'period': period, 'days': len(sample),
                                      'mean': metric['mean'], 'sd': metric['volatility'],
                                      'sharpe': metric['sharpe']})
    rank_summary, asset_ranks = monthly_rank_check(panel, actual, window)
    metadata = {**provenance, 'updated_csv': updated_file.name,
                'updated_sha256': updated_digest, 'protocol': rules,
                'protocol_sha256': file_digest(PROTOCOL), 'overlapping_dates': len(overlap),
                'overlapping_revised_cells_ignored': revised_cells,
                'old_dates_absent_in_update': len(old.index.difference(overlap))}
    save_run(run, output, metadata)
    pd.DataFrame(portfolio_results).to_csv(output/'metrics/test_periods.csv', index=False)
    rank_summary.to_csv(output/'metrics/monthly_rank_correlation.csv', index=False)
    asset_ranks.to_csv(output/'diagnostics/monthly_asset_ranks.csv', index=False)
