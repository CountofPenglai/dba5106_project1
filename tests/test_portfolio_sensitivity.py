import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/zhengce/projects/fundation_ba/project1')


class PortfolioSensitivityCliTests(unittest.TestCase):
    def test_four_entrypoints_use_causal_dates_and_categorized_artifacts(self):
        rng = np.random.default_rng(394)
        dates = pd.bdate_range('2025-12-01', periods=30)
        names = ['SMALL LoBM', 'ME1 BM2', 'ME2 BM1', 'ME2 BM2']
        raw = pd.DataFrame(rng.normal(.05, .8, (30, 4)), index=dates.strftime('%Y%m%d'), columns=names)
        raw.index.name = 'date'
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv = directory / 'data.csv'
            csv.write_text('Average Equal Weighted Returns -- Daily\n' + raw.to_csv() + '\n')
            starts = str(dates[20].date()), str(dates[22].date())
            expected = {
                'turnover': 'LASSO_Turnover_0p1',
                'smoothness': 'LASSO_Smooth_0p1',
                'shrinkage': 'LASSO_LedoitWolf',
                'concentration': 'LASSO_WeightCap_40pct',
            }
            for mode, added in expected.items():
                with self.subTest(mode=mode):
                    output = directory / mode
                    args = [sys.executable, str(ROOT / f'run_{mode}.py'), '--csv', str(csv),
                            '--window', '20', '--start', starts[0], '--end', starts[1],
                            '--output', str(output)]
                    if mode == 'turnover':
                        args += ['--turnover-strengths', '0', '.1']
                    elif mode == 'smoothness':
                        args += ['--smoothness-strengths', '0', '.1']
                    elif mode == 'concentration':
                        args += ['--weight-limit', '.4']
                    process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    summary = pd.read_csv(output / 'metrics' / 'summary.csv', index_col=0)
                    self.assertIn(added, summary.index)
                    self.assertEqual(set(summary.index) & {'EW', 'LASSO_Selected'}, {'EW', 'LASSO_Selected'})
                    saved = pd.read_csv(output / 'returns' / f'{added}_returns.csv', index_col=0)
                    self.assertEqual(len(saved), 3)
                    self.assertEqual(saved.index.max(), starts[1])
                    metadata = json.loads((output / 'config' / 'run_metadata.json').read_text())
                    self.assertEqual(metadata['window'], 20)
                    self.assertFalse(metadata['final_test']['evaluated'])
                    self.assertTrue((output / 'artifact_manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
