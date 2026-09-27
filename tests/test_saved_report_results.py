import shutil
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SavedReportResultTests(unittest.TestCase):
    def test_every_report_record_has_valid_daily_streams_and_original_metrics(self):
        from tools.verify_report_results import audit_results
        result = audit_results(ROOT)
        self.assertEqual(result['records'], 32)
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
