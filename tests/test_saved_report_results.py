import shutil
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SavedReportResultTests(unittest.TestCase):
    def test_report_comparison_respects_displayed_precision(self):
        from tools import verify_report_results as audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root/'results/report'
            (report/'validation/metrics').mkdir(parents=True)
            (report/'report_reference.json').write_text(json.dumps({'checks': [
                {'location': 'Table 1', 'model': 'EW', 'period': 'validation',
                 'metric': 'sharpe', 'expected': .0995, 'tolerance': .00005}
            ]}))
            summary = report/'validation/metrics/summary.csv'
            for actual, match in ((.09951, True), (.09961, False)):
                summary.write_text('model,sharpe\nEW,'+str(actual)+'\n')
                self.assertEqual(audit.compare_report(root).iloc[0]['matches_report'], match)

    def test_every_report_record_has_valid_daily_streams_and_recomputed_metrics(self):
        from tools.verify_report_results import audit_results
        result = audit_results(ROOT)
        self.assertEqual(result['records'], 26)
        self.assertEqual(result['validation_models'], 19)
        self.assertEqual(result['july_models'], 7)

    def test_changed_saved_artifact_is_detected(self):
        from tools.verify_report_results import audit_results
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT/'results', root/'results')
            path = root/'results/report/july_test/returns/LP_TradingPenalty.csv'
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                audit_results(root)
