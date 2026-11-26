"""Check the published Word tables against the saved validation and final-test results."""
import argparse
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd

from portfolio.paths import PROJECT_ROOT

NAMES = {
    'EW': 'EW', 'MinVar': 'MinVar', 'Course LASSO': 'LASSO_CV', 'Course Ridge': 'Ridge_CV',
    'Exposure-penalized LASSO': 'LASSO_ExposurePenalty',
    'LASSO with a 10% asset cap': 'LASSO_WeightCap_10pct',
    'Historical top 20 in equal weights': 'HistoricalTop20_EW',
    'LASSO forecast with risk-constrained LP': 'LP_Return',
}
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def read_tables(document):
    with ZipFile(document) as archive:
        body = ElementTree.fromstring(archive.read('word/document.xml'))
    tables = []
    for tbl in body.findall('.//w:body/w:tbl', NS):
        rows = []
        for row in tbl.findall('w:tr', NS):
            rows.append([''.join(cell.itertext()).strip().replace('−', '-')
                         for cell in row.findall('w:tc', NS)])
        tables.append(rows)
    return tables


def verify(document, root):
    _, _, performance, attribution, forecasts, test = read_tables(document)
    val = root/'validation/metrics'
    totals = pd.read_csv(val/'summary.csv', index_col='model')
    gaps = pd.read_csv(val/'attribution.csv', index_col=0)
    preds = pd.read_csv(val/'forecasts.csv', index_col='method')
    final = pd.read_csv(root/'final_test/metrics/test_periods.csv').set_index(['period', 'model'])
    checks = []

    def check(row, expected):
        if row != expected:
            raise ValueError(f'Report row {row} differs from saved results {expected}')
        checks.extend(expected)

    if len(performance) != len(totals)+1 or len(test) != len(final)+1 or len(forecasts) != 3 or len(attribution) != 4:
        raise ValueError('Unexpected number of report rows')
    if {NAMES[row[0]] for row in performance[1:]} != set(totals.index):
        raise ValueError('Validation model roster differs from the report')
    for name, *values in performance[1:]:
        s = totals.loc[NAMES[name]]
        check(values, [f'{s["mean"]:.4%}', f'{s.volatility:.4%}', f'{s.sharpe:.5f}'])
    for (label, *values), group in zip(attribution[1:], ('Top 20: total gap to EW',
                                                         'Other 78: total gap to EW', 'Total gap to EW')):
        cells = gaps.loc[group, ['MinVar','LASSO_CV']]
        check(values, [f'{v:+.2f}' for v in cells])
    methods = {'Trailing 126-day mean': 'TrailingMean_126', 'Per-asset LASSO': 'LASSO_Forecast_126'}
    for label, *values in forecasts[1:]:
        s = preds.loc[methods[label]]
        check(values, [f'{s.pooled_r2:.5f}', f'{s.mean_hits:.2f} / 20', f'{s.top_sharpe:.5f}'])
    periods = {'August': '2026-08', 'September': '2026-09', 'Both months': 'combined'}
    period = None
    seen = set()
    for month, name, *values in test[1:]:
        if month:
            period = periods[month]
        seen.add((period, NAMES[name]))
        s = final.loc[(period, NAMES[name])]
        check(values, [f'{s["mean"]:.4%}', f'{s.sd:.4%}', f'{s.sharpe:.5f}'])
    if seen != set(final.index):
        raise ValueError('Incomplete final test roster or periods in the report')
    return len(checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path,
                        default=PROJECT_ROOT/'docs/DBA5106_Project1_Final_Report_Outline.docx')
    parser.add_argument('--results', type=Path, default=PROJECT_ROOT/'results/course_126')
    args = parser.parse_args()
    print(f'{verify(args.document, args.results)} entries in report tables 3–6 match.')
