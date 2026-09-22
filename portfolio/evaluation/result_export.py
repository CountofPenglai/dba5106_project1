"""Validated, reproducible export shared by all forecast/decision experiments."""

import json
from pathlib import Path
import platform
from portfolio.evaluation.provenance import file_digest, package_versions as _package_versions, source_paths as default_code_paths
from portfolio.evaluation.artifact_store import RunArtifacts
import re

import numpy as np
import pandas as pd


_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validate_frames(frames):
    if not isinstance(frames, dict):
        raise ValueError("result.frames must be a dictionary")
    reserved = {"run_metadata", "frozen_decisions"}
    for name, frame in frames.items():
        if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name) or name in reserved:
            raise ValueError(f"unsafe or reserved artifact name: {name!r}")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"artifact {name!r} must be a DataFrame")
        if frame.columns.has_duplicates:
            raise ValueError(f"artifact {name!r} has duplicate columns")
        if frame.isna().to_numpy().any():
            raise ValueError(f"artifact {name!r} contains missing values")
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"artifact {name!r} contains non-finite values")
        for value in frame.select_dtypes(exclude=[np.number]).to_numpy().ravel():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError(f"artifact {name!r} contains non-finite values")


def _validated_json(value, label):
    try:
        return json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and JSON serializable") from exc


def export_result(result, output, *, source_csv, code_paths=None, layout="flat"):
    """Export every named frame plus metadata and frozen decisions."""
    frames = result.frames
    _validate_frames(frames)
    source_csv = Path(source_csv).resolve()
    if not source_csv.is_file():
        raise ValueError("source_csv must be an existing file")
    paths = default_code_paths() if code_paths is None else [Path(path).resolve() for path in code_paths]
    if any(not path.is_file() for path in paths):
        raise ValueError("every code path must be an existing file")
    data_hash = file_digest(source_csv)
    provenance = {
        "source": {"path": str(source_csv), "sha256": data_hash},
        "csv": str(source_csv),
        "data_sha256": data_hash,
        "python": platform.python_version(),
        "packages": _package_versions(),
        "code_sha256": {str(path): file_digest(path) for path in paths},
    }
    if not isinstance(result.metadata, dict) or not isinstance(result.frozen, dict):
        raise ValueError("result metadata and frozen decisions must be dictionaries")
    metadata = {**result.metadata, **provenance, "artifact_layout": layout, "artifact_layout_version": 1}
    metadata_json = _validated_json(metadata, "metadata")
    frozen_json = _validated_json(result.frozen, "frozen decisions")

    artifacts = RunArtifacts(output, layout=layout).prepare()
    for name, frame in frames.items():
        frame.to_csv(artifacts.path(f"{name}.csv"))
    artifacts.path("run_metadata.json").write_text(metadata_json)
    artifacts.path("frozen_decisions.json").write_text(frozen_json)
    artifacts.finalize()
