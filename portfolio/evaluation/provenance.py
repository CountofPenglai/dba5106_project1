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
    """Include nested implementations and compatibility entrypoints, not tests."""
    paths = [PROJECT_ROOT / 'requirements.txt', *PROJECT_ROOT.glob('*.py')]
    for directory in ('portfolio', 'experiments', 'tools', 'examples'):
        paths.extend((PROJECT_ROOT / directory).rglob('*.py'))
    return sorted({path.resolve() for path in paths if path.is_file()})


def code_hashes():
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_digest(path) for path in source_paths()}


def verify_code_hashes(stored):
    """Support portable and old absolute keys without collapsing same basenames."""
    if not isinstance(stored, dict):
        raise ValueError('Source fingerprints must be a dictionary')
    # Absolute inventories carry their original checkout root via requirements.txt.
    roots = {Path(name).parent for name in stored
             if Path(name).is_absolute() and Path(name).name == 'requirements.txt'}
    normalized = {}
    for name, digest in stored.items():
        path = Path(name)
        if path.is_absolute():
            if len(roots) != 1 or not path.is_relative_to(next(iter(roots))):
                raise ValueError('Ambiguous reference source root')
            path = path.relative_to(next(iter(roots)))
        relative = path.as_posix()
        if relative in normalized:
            raise ValueError('Duplicate reference source path: ' + relative)
        normalized[relative] = digest
    for relative, expected in code_hashes().items():
        if normalized.get(relative) != expected:
            raise ValueError('Reference source code differs: ' + relative +
                             '; use its original repository revision to reproduce historical artifacts')


def run_provenance(source_csv):
    """Shared metadata for dataset identity, runtime versions and source code."""
    import platform
    source_csv = Path(source_csv).resolve()
    return dict(csv=str(source_csv), data_sha256=file_digest(source_csv),
                python=platform.python_version(), packages=package_versions(),
                code_sha256=code_hashes())
