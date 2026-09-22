"""Exercise new recipes through real CLI execution and artifact readback."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from portfolio.evaluation.artifact_store import RunArtifacts, artifact_path
from portfolio.evaluation.provenance import code_hashes, source_paths, file_digest, verify_code_hashes
from portfolio.evaluation.reference_comparison import add_reference_comparison
import test_direct_lp_cli as reference_fixtures

ROOT = Path(__file__).resolve().parents[1]


class RefactoredOutputTests(unittest.TestCase):
    def test_baseline_module_categories_match_legacy_numeric_outputs(self):
        dates = pd.bdate_range('2025-12-01', periods=32)
        data = pd.DataFrame(np.random.default_rng(2026).normal(.03, .7, (32, 3)),
                            index=dates.strftime('%Y%m%d'), columns=list('ABC'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'input.csv'
            source.write_text('Average Equal Weighted Returns -- Daily\n' + data.to_csv() + '\n')
            options = ['--csv', str(source), '--window', '20', '--start', str(dates[25].date()),
                       '--end', str(dates[28].date()), '--growth']
            for command, dest in [(['run_example.py'], root/'old'),
                                  (['-m', 'experiments.regularized_baselines'], root/'new')]:
                run = subprocess.run([sys.executable, *command, *options, '--output', str(dest)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=60,
                                     env={**os.environ, 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            self.assertTrue((root/'old'/'summary.csv').is_file())
            self.assertTrue((root/'new'/'metrics'/'summary.csv').is_file())
            manifest = json.loads((root/'new'/'artifact_manifest.json').read_text())
            for name in manifest['artifacts']:
                if name.endswith('.csv'):
                    pd.testing.assert_frame_equal(pd.read_csv(artifact_path(root/'old', name)),
                                                  pd.read_csv(artifact_path(root/'new', name)))
            metadata = json.loads(artifact_path(root/'new','run_metadata.json').read_text())
            self.assertEqual(metadata['artifact_layout'],'categorized')
            self.assertIn('portfolio/models/constrained_regularized_portfolio.py',metadata['code_sha256'])
            before = {p.relative_to(root/'new'):p.read_bytes() for p in (root/'new').rglob('*') if p.is_file()}
            retry = subprocess.run([sys.executable,'-m','experiments.regularized_baselines',*options,
                                    '--output',str(root/'new')],cwd=ROOT,capture_output=True,timeout=15)
            self.assertNotEqual(retry.returncode,0)
            self.assertEqual(before,{p.relative_to(root/'new'):p.read_bytes() for p in (root/'new').rglob('*') if p.is_file()})

    def test_reference_comparison_reads_categorized_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, old, source, returns = reference_fixtures.LPReferenceTests().setup_reference(Path(tmp))
            store = RunArtifacts(Path(tmp)/'categorized',layout='categorized').prepare()
            for path in old.iterdir():
                store.path(path.name).write_bytes(path.read_bytes())
            store.finalize()
            add_reference_comparison(result,store.root,returns,source_csv=source)
            self.assertTrue(result.metadata['reference_comparison']['returns_reconstructed'])
            self.assertIn('Baseline_252d',result.frames['comparison'].index)

    def test_nested_code_fingerprints_are_portable_and_tampering_is_rejected(self):
        absolute = {str(path):file_digest(path) for path in source_paths()}
        verify_code_hashes(absolute)
        verify_code_hashes(code_hashes())
        # Different checkout paths do not change the source identity.
        portable = {str(Path('/some/other/checkout')/path.relative_to(ROOT)):file_digest(path)
                    for path in source_paths()}
        verify_code_hashes(portable)
        portable['/some/other/checkout/portfolio/models/lasso_return_forecast.py'] = 'changed'
        with self.assertRaisesRegex(ValueError,'source code'):
            verify_code_hashes(portable)

    def test_new_entrypoints_all_have_help(self):
        modules = ['experiments.'+name for name in (
            'regularized_baselines','lasso_training_windows','lasso_recency_weight_search',
            'lasso_reward_exposure_search','lasso_forecast_mixture','lasso_forecast_lp')]
        modules += ['tools.audit_missing_values']
        for module in modules:
            with self.subTest(module=module):
                run = subprocess.run([sys.executable,'-m',module,'--help'],cwd=ROOT,
                                     capture_output=True,text=True,timeout=15,
                                     env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
                self.assertEqual(run.returncode,0,run.stderr)
                self.assertIn('--layout',run.stdout)

if __name__=='__main__':unittest.main()
