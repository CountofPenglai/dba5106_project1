import tempfile
import unittest
from pathlib import Path

from audit_missing_values import audit_file


class MissingValueAuditTests(unittest.TestCase):
    def test_raw_markers_are_distinguished_and_excluded_columns_are_counted(self):
        content = '''Missing data are indicated by -99.99 or -999.

Average Value Weighted Returns -- Daily
, A , ME9 BM10 ,BIG HiBM,B
20260101,-99,-99.99,-999,0
20260102,-99.00,-99.990,1,-99.9

Average Equal Weighted Returns -- Daily
,A,ME9 BM10,BIG HiBM,B
20260101,1,-99.99,-999,0
20260102,2,-99.99,,3

Number of Firms in Portfolios
,A,ME9 BM10,BIG HiBM,B
20260101,1,2,3,4

Average Firm Size
,A,ME9 BM10,BIG HiBM,B
20260101,10,20,30,40
'''
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            source.write_text(content)
            report = audit_file(source)
            records = {(r['section'], r['asset']): r for r in report['columns']}
            value = 'Average Value Weighted Returns -- Daily'
            equal = 'Average Equal Weighted Returns -- Daily'
            self.assertEqual(len(records), 16)
            self.assertEqual(records[value, 'A']['count_minus_99'], 2)
            self.assertEqual(records[value, 'B']['count_minus_99'], 0)
            self.assertEqual(records[value, 'B']['missing_marker_count'], 0)
            self.assertEqual(records[equal, 'ME9 BM10']['count_minus_99_99'], 2)
            self.assertTrue(records[equal, 'ME9 BM10']['excluded_by_default'])
            self.assertEqual(records[equal, 'BIG HiBM']['count_minus_999'], 1)
            self.assertEqual(records[equal, 'BIG HiBM']['blank_count'], 1)
            self.assertEqual(records[equal, 'ME9 BM10']['first_marker_date'], '2026-01-01')
            self.assertEqual(records[equal, 'ME9 BM10']['last_marker_date'], '2026-01-02')
            self.assertEqual(report['sections'][equal]['rows_with_missing_after_default_exclusions'], 0)
            filtered = audit_file(source, start='2026-01-02', end='2026-01-02')
            row = next(r for r in filtered['columns'] if r['section'] == value and r['asset'] == 'A')
            self.assertEqual(row['observations'], 1)
            self.assertEqual(row['count_minus_99'], 1)
            self.assertEqual(source.read_text(), content)

    def test_missing_section_and_malformed_rows_fail(self):
        for content in ('No return table here\n',
                        'Average Equal Weighted Returns -- Daily\n',
                        'Average Equal Weighted Returns -- Daily\n,A,B\n20260101,-99.99\n'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'input.csv'
                source.write_text(content)
                with self.assertRaises(ValueError):
                    audit_file(source)


if __name__ == '__main__':
    unittest.main()
