import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portfolio.two_stage import ExperimentResult, frame_digest, portfolio_metrics
from portfolio.two_stage_io import default_code_paths, file_digest
from portfolio.lp_reference import add_reference_comparison


class LPReferenceTests(unittest.TestCase):
    def setup_reference(self, root):
        dates = pd.bdate_range('2026-01-02', periods=4)
        returns = pd.DataFrame([[.01,-.002],[-.01,.02],[.005,.003],[-.002,.01]], index=dates, columns=['A','B'])
        w = pd.DataFrame(.5, index=dates, columns=returns.columns)
        summary = pd.DataFrame([dict(model='Baseline_252d', role='legacy_baseline', **portfolio_metrics(w,returns))]).set_index('model')
        source = root/'source.csv';source.write_text(returns.to_csv())
        reference = root/'reference';reference.mkdir()
        w.to_csv(reference/'Baseline_252d_weights.csv')
        ((w*returns).sum(axis=1)).to_frame('return').to_csv(reference/'Baseline_252d_returns.csv')
        summary.to_csv(reference/'comparison.csv')
        config = dict(validation_start='2026-01-02',validation_end=str(dates[-1].date()))
        config.update(history_start='2021-08-01', calibration_end='2024-07-31',
                      development_start='2024-08-01', development_end='2025-12-31',
                      alphas=[.001,.01], n_splits=3, test_size=21)
        shared = dict(forecast_configs={'252d': {'window':{'observations':252,'years':None,'min_observations':2},'half_life':None}},
                      alphas={'252d':.001},risk_window={'years':3,'observations':None,'min_observations':252})
        reference_config=dict(config)
        reference_config['decision_start']=reference_config.pop('development_start')
        reference_config['decision_end']=reference_config.pop('development_end')
        metadata = dict(phase='development',final_test_evaluated=False, assets=list(returns.columns),
                        config=reference_config, **shared, data_sha256=file_digest(source), development_data_digest=frame_digest(returns),
                        code_sha256={str(p.resolve()):file_digest(p) for p in default_code_paths()})
        (reference/'run_metadata.json').write_text(json.dumps(metadata))
        result = ExperimentResult({'comparison':summary.rename(index={'Baseline_252d':'LP_new'}).copy()},
                                  {'config':config,'assets':list(returns.columns),**shared},
                                  {'development_data_digest':frame_digest(returns)})
        return result, reference, source, returns

    def test_reference_reconstructs_returns_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            result, ref, source, returns = self.setup_reference(Path(directory))
            add_reference_comparison(result, ref, returns, source_csv=source)
            self.assertEqual(list(result.frames['comparison'].index), ['LP_new','Baseline_252d'])
            self.assertEqual(result.metadata['reference_comparison']['strategies'],1)
            path=ref/'Baseline_252d_returns.csv'
            bad=pd.read_csv(path,index_col=0);bad.iloc[0,0]+=.1;bad.to_csv(path)
            with self.assertRaisesRegex(ValueError,'return'):
                add_reference_comparison(result,ref,returns,source_csv=source)

    def test_reference_rejects_wrong_data_code_and_dates(self):
        for mistake in ['data','code','dates','calibration','forecast','alpha','risk']:
            with self.subTest(mistake=mistake), tempfile.TemporaryDirectory() as directory:
                result, ref, source, returns = self.setup_reference(Path(directory))
                path=ref/'run_metadata.json';meta=json.loads(path.read_text())
                if mistake=='data':meta['data_sha256']='bad'
                if mistake=='code':
                    name=next(x for x in meta['code_sha256'] if x.endswith('/forecasting.py'))
                    meta['code_sha256'][name]='bad'
                if mistake=='dates':meta['config']['validation_end']='2026-06-30'
                if mistake=='calibration':meta['config']['calibration_end']='2025-07-31'
                if mistake=='forecast':meta['forecast_configs']['252d']['window']['observations']=126
                if mistake=='alpha':meta['alphas']['252d']=.01
                if mistake=='risk':meta['risk_window']['years']=2
                path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    add_reference_comparison(result,ref,returns,source_csv=source)


class DirectLPCliTests(unittest.TestCase):
    def write_csv(self, path, end='2026-08-28'):
        dates=pd.bdate_range('2021-08-02',end)
        raw=pd.DataFrame(np.random.default_rng(19).normal(0,1,(len(dates),2)),
                         index=dates.strftime('%Y%m%d'),columns=['A','B'])
        path.write_text('Average Equal Weighted Returns -- Daily\n'+raw.to_csv()+'\n')

    def test_development_crops_future_and_exports_real_provenance(self):
        import run_direct_lp
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.csv';out=root/'out'
            self.write_csv(source)
            def core(returns, **kwargs):
                # Observe the data actually delivered after CSV processing;
                # substitute only the expensive portfolio experiment.
                self.assertFalse(kwargs['include_development'])
                return ExperimentResult({},dict(input_end=str(returns.index[-1].date()),
                                                assets=len(returns.columns)),{})
            with patch.object(run_direct_lp,'run_direct_lp',side_effect=core):
                run_direct_lp.main(['--csv',str(source),'--output',str(out)])
            meta=json.loads((out/'run_metadata.json').read_text())
            self.assertEqual(meta['input_end'],'2026-06-30')
            self.assertEqual(meta['data_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIn(str(Path(run_direct_lp.__file__).resolve()),meta['code_sha256'])
            with self.assertRaises(SystemExit):
                run_direct_lp.main(['--csv',str(source),'--output',str(out)])
            with patch.object(run_direct_lp,'run_direct_lp',return_value=ExperimentResult({}, {}, {})) as called:
                run_direct_lp.main(['--csv',str(source),'--output',str(root/'full'),'--include-development'])
                self.assertTrue(called.call_args.kwargs['include_development'])

    def test_missing_august_fails_before_artifact_export(self):
        import run_direct_lp
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.csv';out=root/'out';frozen=root/'frozen.json'
            self.write_csv(source,end='2026-07-31');frozen.write_text('{}')
            with self.assertRaises(SystemExit):
                run_direct_lp.main(['--csv',str(source),'--output',str(out),'--frozen',str(frozen)])
            self.assertFalse((out/'run_metadata.json').exists())


if __name__=='__main__':
    unittest.main()
