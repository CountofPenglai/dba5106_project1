"""Validated, reproducible artifact export for two-stage experiments."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd


_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_PACKAGES = ("numpy", "pandas", "scikit-learn", "cvxpy", "clarabel", "scs")


def file_digest(path):
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions():
    packages = {}
    for package in _PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def default_code_paths():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "run_two_stage.py", root / "requirements.txt"]
    paths.extend(sorted((root / "portfolio").glob("*.py")))
    return [path for path in paths if path.is_file()]


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


def export_result(result, output, *, source_csv, code_paths=None):
    """Export every named frame plus metadata and frozen decisions."""
    frames = result.frames
    _validate_frames(frames)
    source_csv = Path(source_csv).resolve()
    if not source_csv.is_file():
        raise ValueError("source_csv must be an existing file")
    paths = default_code_paths() if code_paths is None else [Path(path).resolve() for path in code_paths]
    if any(not path.is_file() for path in paths):
        raise ValueError("every code path must be an existing file")
    provenance = {
        "source": {"path": str(source_csv), "sha256": file_digest(source_csv)},
        "csv": str(source_csv),
        "data_sha256": file_digest(source_csv),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "code_sha256": {str(path): file_digest(path) for path in paths},
    }
    if not isinstance(result.metadata, dict) or not isinstance(result.frozen, dict):
        raise ValueError("result metadata and frozen decisions must be dictionaries")
    metadata = {**result.metadata, **provenance}
    metadata_json = _validated_json(metadata, "metadata")
    frozen_json = _validated_json(result.frozen, "frozen decisions")

    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be an empty or new directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv")
    (output / "run_metadata.json").write_text(metadata_json)
    (output / "frozen_decisions.json").write_text(frozen_json)
