"""Verify curated result checksums, report figures and (optionally) realized returns."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from portfolio.data.returns_loader import load_returns
from portfolio.evaluation.portfolio_metrics import summarize, summarize_holdings
from portfolio.model_catalog import JULY_MODELS, MODELS
from portfolio.paths import PROJECT_ROOT


def compare_report(root=PROJECT_ROOT):
    """Check the reference report's numeric cells at their stated precision."""
    report = Path(root)/'results/report'
    reference = json.loads((report/'report_reference.json').read_text())
    summaries = {period: pd.read_csv(report/period/'metrics/summary.csv', index_col=0)
                 for period in {item['period'] for item in reference['checks']}}
    rows = []
    for item in reference['checks']:
        actual = float(summaries[item['period']].loc[item['model'], item['metric']])
        rows.append({**item, 'actual': actual,
                     'matches_report': abs(actual-item['expected']) <= item['tolerance']+1e-12})
    return pd.DataFrame(rows)


def audit_results(root=PROJECT_ROOT, source_csv=None):
    root = Path(root)
    manifest = json.loads((root/'results/manifest.json').read_text())
    for name, expected in manifest['artifact_sha256'].items():
        path = root/name
        if not path.resolve().is_relative_to((root/'results').resolve()):
            raise ValueError('Result path escapes results directory')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Artifact checksum mismatch: '+name)
    data = None if source_csv is None else load_returns(source_csv, start='2021-08-01')
    keys = set()
    for record in manifest['records']:
        name = record['model']
        if name not in MODELS:
            raise ValueError('Missing report model implementation: '+name)
        key = (name, record['period'], record['window'])
        if key in keys:
            raise ValueError('Duplicate model/period/window record')
        keys.add(key)
        daily = pd.read_csv(root/record['returns'], index_col=0, parse_dates=True)
        weights = pd.read_csv(root/record['weights'], index_col=0, parse_dates=True)
        if not daily.index.equals(weights.index) or not weights.columns.is_unique:
            raise ValueError('Misaligned saved return/weight stream: '+name)
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-8, rtol=0)
        calculated = summarize(daily['return'])
        for metric, expected in record['reported_metrics'].items():
            np.testing.assert_allclose(calculated[metric], expected, atol=2e-13, rtol=1e-10)
        period = record['period']
        summary_path = root/'results/report'/period/'metrics/summary.csv'
        summary = pd.read_csv(summary_path, index_col=0)
        diagnostics = pd.read_csv(root/record['diagnostics'], index_col=0, parse_dates=True)
        if not diagnostics.index.equals(daily.index):
            raise ValueError('Misaligned saved diagnostics: '+name)
        previous = weights.shift(1).fillna(1/len(weights.columns))
        target_turnover = .5*(weights-previous).abs().sum(axis=1)
        np.testing.assert_allclose(target_turnover, diagnostics.target_turnover, atol=2e-13)
        aggregate = {**calculated, **summarize_holdings(weights),
                     'average_target_turnover': float(target_turnover.mean()),
                     'average_trade_turnover': float(diagnostics.trade_turnover.mean())}
        if MODELS[name].market_beta:
            aggregate['average_beta'] = float(diagnostics.market_beta.mean())
        for metric, value in aggregate.items():
            np.testing.assert_allclose(value, summary.loc[name, metric], atol=2e-13, rtol=1e-10)
        np.testing.assert_allclose(daily.net_return, daily['return']-daily.trading_cost, atol=2e-13)
        np.testing.assert_allclose(daily.trading_cost, .001*diagnostics.trade_turnover, atol=2e-13)
        if data is not None:
            actual = data.loc[daily.index, weights.columns].to_numpy()
            np.testing.assert_allclose(daily['return'], np.sum(weights.to_numpy()*actual, axis=1),
                                       atol=2e-13, rtol=1e-10)
            drifted = weights.to_numpy()*(1+actual)/(1+daily['return'].to_numpy()[:, None])
            pretrade = np.vstack([np.full(len(weights.columns), 1/len(weights.columns)), drifted[:-1]])
            trade_turnover = .5*np.abs(weights.to_numpy()-pretrade).sum(axis=1)
            np.testing.assert_allclose(trade_turnover, diagnostics.trade_turnover, atol=2e-13)
    validation = {name for name, period, window in keys if period == 'validation' and window == 252}
    july = {name for name, period, _ in keys if period == 'july_test'}
    if (validation != set(MODELS) or july != set(JULY_MODELS)
            or len(keys) != len(MODELS)+len(JULY_MODELS)):
        raise ValueError('Saved result coverage differs from the approved model catalog')
    comparison = compare_report(root)
    return {'records': len(keys), 'artifacts': len(manifest['artifact_sha256']),
            'validation_models': len(validation), 'july_models': len(july),
            'report_cells': len(comparison),
            'matching_report_cells': int(comparison.matches_report.sum()),
            'report_discrepancies': int((~comparison.matches_report).sum()),
            'daily_returns_reconstructed_from_data': data is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Also verify gross returns against asset returns and saved holdings.')
    args = parser.parse_args()
    print(json.dumps(audit_results(source_csv=args.csv), indent=2))


if __name__ == '__main__':
    main()
