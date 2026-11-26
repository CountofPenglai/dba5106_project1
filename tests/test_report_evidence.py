"""Regression checks on the frozen roster, period metrics and the actual report."""
import json
import unittest

import numpy as np
import pandas as pd

from portfolio.paths import PROJECT_ROOT
from tools.verify_report_results import verify


class ReportEvidenceTests(unittest.TestCase):
    def test_report_numeric_tables_match_curated_outputs(self):
        self.assertEqual(verify(PROJECT_ROOT/'docs/DBA5106_Project1_Final_Report_Outline.docx',
                                PROJECT_ROOT/'results/course_126'), 72)

    def test_frozen_holdout_and_monthly_statistics(self):
        root = PROJECT_ROOT/'results/course_126/final_test'
        protocol = json.loads((PROJECT_ROOT/'configs/final_test_126.json').read_text())
        self.assertEqual(protocol['window'], 126)
        self.assertEqual([m['name'] for m in protocol['models']],
                         ['EW', 'LASSO_CV', 'Ridge_CV', 'LASSO_ExposurePenalty'])
        metrics = pd.read_csv(root/'metrics/test_periods.csv').set_index(['model','period'])
        for model in protocol['models']:
            name = model['name']
            daily = pd.read_csv(root/'returns'/f'{name}.csv', index_col='date')['return']
            for period in ('2026-08','2026-09','combined'):
                r = daily if period == 'combined' else daily[daily.index.str.startswith(period)]
                self.assertEqual(len(r), 42 if period == 'combined' else 21)
                expected = [r.mean(), r.std(ddof=1), r.mean()/r.std(ddof=1)]
                np.testing.assert_allclose(metrics.loc[(name,period), ['mean','sd','sharpe']],
                                           expected, atol=1e-12, rtol=0)
