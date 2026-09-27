"""Portable source/data fingerprints for reproducible experiment artifacts."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from portfolio.paths import PROJECT_ROOT

PACKAGES = ('numpy', 'pandas', 'scikit-learn', 'cvxpy', 'clarabel', 'scs', 'scipy')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame):
    return hashlib.sha256(frame.to_csv(float_format='%.17g', date_format='%Y-%m-%d').encode()).hexdigest()


def package_versions():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def source_paths():
    """Fingerprint active code and fixed configurations, excluding tests."""
    paths = [PROJECT_ROOT / 'requirements.txt', *PROJECT_ROOT.glob('*.py')]
    for directory in ('portfolio', 'tools'):
        paths.extend((PROJECT_ROOT / directory).rglob('*.py'))
    paths.extend((PROJECT_ROOT / 'configs').glob('*.json'))
    return sorted({path.resolve() for path in paths if path.is_file()})


def code_hashes():
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_digest(path) for path in source_paths()}


def run_provenance(source_csv):
    """Shared metadata for dataset identity, runtime versions and source code."""
    import platform
    source_csv = Path(source_csv).resolve()
    return dict(csv=str(source_csv), data_sha256=file_digest(source_csv),
                python=platform.python_version(), packages=package_versions(),
                code_sha256=code_hashes())
