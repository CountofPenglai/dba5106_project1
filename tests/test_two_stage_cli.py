import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


class TwoStageIoTests(unittest.TestCase):
    def test_export_round_trips_frames_metadata_frozen_and_provenance(self):
        from portfolio.two_stage_io import export_result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_bytes(b"source bytes\n")
            code = root / "code.py"
            code.write_text("VALUE = 1\n")
            output = root / "artifacts"
            result = mock.Mock(
                frames={"daily_weights": pd.DataFrame(
                    [[0.6, 0.4]], index=pd.DatetimeIndex(["2026-01-02"], name="date"),
                    columns=["A", "B"],
                )},
                metadata={"phase": "development", "nested": {"count": 1}},
                frozen={"selected_gamma": 0.1, "theta": [0.2, 0.3, 0.5]},
            )
            export_result(result, output, source_csv=source, code_paths=[code])

            stored = pd.read_csv(output / "daily_weights.csv", index_col=0, parse_dates=True)
            pd.testing.assert_frame_equal(stored, result.frames["daily_weights"], check_freq=False)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["phase"], "development")
            self.assertEqual(metadata["nested"], {"count": 1})
            self.assertEqual(metadata["source"]["sha256"], __import__("hashlib").sha256(b"source bytes\n").hexdigest())
            self.assertEqual(len(metadata["code_sha256"][str(code.resolve())]), 64)
            self.assertIn("python", metadata)
            self.assertIn("numpy", metadata["packages"])
            self.assertEqual(json.loads((output / "frozen_decisions.json").read_text()), result.frozen)

    def test_export_rejects_nonfinite_diagnostics_before_writing(self):
        from portfolio.two_stage_io import export_result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("x")
            output = root / "artifacts"
            result = mock.Mock(
                frames={"diagnostics": pd.DataFrame({"objective": [np.nan]})},
                metadata={"phase": "development"},
                frozen={},
            )
            with self.assertRaises(ValueError):
                export_result(result, output, source_csv=source, code_paths=[])
            self.assertFalse(output.exists())

    def test_export_rejects_unsafe_or_duplicate_artifact_names(self):
        from portfolio.two_stage_io import export_result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("x")
            for name in ("../escape", "run_metadata", "weights.csv"):
                result = mock.Mock(
                    frames={name: pd.DataFrame({"x": [1.0]})}, metadata={}, frozen={}
                )
                with self.subTest(name=name), self.assertRaises(ValueError):
                    export_result(result, root / "out", source_csv=source, code_paths=[])

    def test_export_allows_diagnostic_rows_to_share_a_date(self):
        from portfolio.two_stage_io import export_result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("x")
            date = pd.Timestamp("2026-01-02")
            result = mock.Mock(
                frames={"candidate_diagnostics": pd.DataFrame(
                    {"component": ["U", "V"], "objective": [0.1, 0.2]},
                    index=pd.DatetimeIndex([date, date], name="date"),
                )},
                metadata={},
                frozen={},
            )
            export_result(result, root / "out", source_csv=source, code_paths=[])
            stored = pd.read_csv(root / "out" / "candidate_diagnostics.csv", index_col=0)
            self.assertEqual(len(stored), 2)


class TwoStageCliTests(unittest.TestCase):
    def _write_csv(self, path, end="2026-08-28"):
        dates = pd.bdate_range("2021-08-02", end)
        values = np.arange(len(dates) * 2, dtype=float).reshape(len(dates), 2) / 10000
        raw = pd.DataFrame(values, index=dates.strftime("%Y%m%d"), columns=["A", "B"])
        path.write_text("Average Equal Weighted Returns -- Daily\n" + raw.to_csv() + "\n")

    def test_development_excludes_future_csv_rows_and_enables_baselines(self):
        import run_two_stage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "returns.csv", root / "out"
            self._write_csv(source)
            result = mock.Mock(frames={}, metadata={"phase": "development"}, frozen={})
            with mock.patch.object(run_two_stage, "run_experiment", return_value=result) as run:
                run_two_stage.main(["--csv", str(source), "--output", str(output)])
            returns = run.call_args.args[0]
            self.assertEqual(returns.index.max(), pd.Timestamp("2026-06-30"))
            self.assertTrue(run.call_args.kwargs["include_baselines"])
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["phase"], "development")
            self.assertFalse(metadata["final_test"]["evaluated"])

    def test_frozen_mode_loads_decisions_and_august_history(self):
        import run_two_stage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "returns.csv", root / "out"
            self._write_csv(source)
            frozen_path = root / "frozen.json"
            frozen = {"alpha": {"252d": 1e-4}, "theta": [0.2, 0.3, 0.5]}
            frozen_path.write_text(json.dumps(frozen))
            result = mock.Mock(frames={}, metadata={"phase": "final_test"}, frozen=frozen)
            with mock.patch.object(run_two_stage, "run_frozen", return_value=result) as run:
                run_two_stage.main([
                    "--csv", str(source), "--output", str(output), "--frozen", str(frozen_path)
                ])
            returns, supplied = run.call_args.args[:2]
            self.assertEqual(supplied, frozen)
            self.assertEqual(returns.index.max(), pd.Timestamp("2026-08-28"))
            self.assertEqual(run.call_args.kwargs["start"], "2026-08-01")
            self.assertEqual(run.call_args.kwargs["end"], "2026-08-30")

    def test_frozen_mode_rejects_missing_final_coverage_before_core(self):
        import run_two_stage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "returns.csv", root / "out"
            self._write_csv(source, end="2026-07-31")
            frozen = root / "frozen.json"
            frozen.write_text("{}")
            with mock.patch.object(run_two_stage, "run_frozen") as run:
                with self.assertRaises(SystemExit) as raised:
                    run_two_stage.main([
                        "--csv", str(source), "--output", str(output), "--frozen", str(frozen)
                    ])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
            self.assertFalse((output / "run_metadata.json").exists())

    def test_nonempty_output_is_rejected_before_core(self):
        import run_two_stage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "returns.csv", root / "out"
            self._write_csv(source)
            output.mkdir()
            (output / "keep.txt").write_text("keep")
            with mock.patch.object(run_two_stage, "run_experiment") as run:
                with self.assertRaises(SystemExit) as raised:
                    run_two_stage.main(["--csv", str(source), "--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
            self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_missing_default_csv_fails_without_creating_artifacts(self):
        import run_two_stage

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with mock.patch.object(run_two_stage, "DEFAULT_CSV", Path(directory) / "missing.csv"):
                with self.assertRaises(SystemExit) as raised:
                    run_two_stage.main(["--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((output / "run_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
