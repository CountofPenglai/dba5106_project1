"""Storage behavior: paths, integrity and flat/categorized compatibility."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class ArtifactLayoutTests(unittest.TestCase):
    def store_api(self):
        self.assertIsNotNone(importlib.util.find_spec('portfolio.evaluation.artifact_store'),
                             'shared artifact storage is required')
        from portfolio.evaluation.artifact_store import RunArtifacts, artifact_path
        return RunArtifacts, artifact_path

    def test_categories_round_trip_and_preserve_exact_run_root(self):
        Store, resolve = self.store_api()
        categories = {'summary.csv':'metrics', 'LASSO_weights.csv':'weights',
                      'LASSO_returns.csv':'returns', 'LASSO_predictions.csv':'predictions',
                      'risk_diagnostics.csv':'diagnostics', 'run_metadata.json':'config',
                      'frozen_decisions.json':'config', 'screen_search.csv':'search',
                      'equity.png':'figures', 'training.log':'logs',
                      '252d_alpha_cv.csv':'search', 'gamma_validation.csv':'search',
                      'prediction_validation.csv':'metrics', 'decision_fits.csv':'config',
                      'market_beta_results.csv':'metrics'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'chosen-run'; store=Store(root,layout='categorized');store.prepare()
            for name,category in categories.items():
                target=store.path(name)
                self.assertEqual(target,root/category/name)
                target.write_text(name)
            store.finalize()
            for name in categories:
                self.assertEqual(resolve(root,name).read_text(),name)
            manifest=json.loads((root/'artifact_manifest.json').read_text())
            self.assertEqual(manifest['layout_version'],1)
            self.assertEqual(len(manifest['artifacts']),len(categories))

    def test_flat_output_and_historical_directory_remain_readable(self):
        Store, resolve = self.store_api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'old';store=Store(root);store.prepare()
            store.path('summary.csv').write_text('legacy')
            self.assertEqual(resolve(root,'summary.csv'),root/'summary.csv')
            store.finalize()
            self.assertEqual(resolve(root,'summary.csv').read_text(),'legacy')

    def test_reject_nonempty_and_unsafe_paths(self):
        Store, resolve = self.store_api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'keep.txt').write_text('preserve')
            with self.assertRaises(ValueError):Store(root).prepare()
            self.assertEqual((root/'keep.txt').read_text(),'preserve')
            for name in ['../escape.csv','/absolute.csv','weights/../../escape.csv']:
                with self.subTest(name=name),self.assertRaises(ValueError):Store(root).path(name)
            (root/'artifact_manifest.json').write_text(json.dumps({'layout_version':1,'artifacts':{'summary.csv':'../outside.csv'}}))
            with self.assertRaises(ValueError):resolve(root,'summary.csv')

    def test_default_root_is_repository_relative_and_unique(self):
        self.store_api()
        from portfolio.evaluation.artifact_store import default_output,PROJECT_ROOT
        one,two=default_output('lasso_forecast_lp'),default_output('lasso_forecast_lp')
        self.assertEqual(one.parent,PROJECT_ROOT/'results'/'lasso_forecast_lp')
        self.assertNotEqual(one,two)
        self.assertFalse(one.exists())

    def test_inventory_includes_nested_implementations_with_unique_paths(self):
        self.store_api()
        from portfolio.evaluation.provenance import source_paths,code_hashes
        paths=source_paths();hashes=code_hashes()
        self.assertEqual(len(paths),len(set(paths)))
        self.assertIn('portfolio/models/lasso_return_forecast.py',hashes)
        self.assertIn('portfolio/engine/forecast_lp_workflow.py',hashes)
        self.assertIn('experiments/lasso_forecast_lp.py',hashes)
        self.assertIn('portfolio/data/__init__.py',hashes)
        self.assertIn('portfolio/evaluation/__init__.py',hashes)

if __name__=='__main__':unittest.main()
