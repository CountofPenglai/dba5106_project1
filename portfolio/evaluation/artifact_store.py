"""One output layout for experiments, with explicit flat-layout compatibility."""
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re

from portfolio.paths import DEFAULT_CSV, PROJECT_ROOT

CATEGORIES = ('metrics', 'returns', 'weights', 'predictions', 'diagnostics',
              'config', 'search', 'figures', 'logs')
MANIFEST = 'artifact_manifest.json'


def _relative_name(name):
    text = str(name)
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or '..' in path.parts or '\\' in text
            or text in ('.', MANIFEST)):
        raise ValueError(f'Unsafe artifact name: {text!r}')
    return path


def _inside(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path must stay inside the run directory')
    return path


def ensure_fresh_output(root):
    """Check before expensive work; checking alone never creates a directory."""
    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError('output must be an empty or new directory')


def default_output(direction):
    """A new run root independent of cwd, in UTC with microsecond resolution."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', direction):
        raise ValueError('Experiment direction must be a simple name')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    return PROJECT_ROOT / 'results' / direction / stamp


def _category(name):
    path = PurePosixPath(name)
    stem, suffix = path.stem, path.suffix.lower()
    if (len(path.parts) > 1 or 'search' in stem or 'calibration' in stem
            or 'cv_scores' in stem or stem.endswith('_alpha_cv') or stem == 'gamma_validation'):
        return 'search'
    if suffix in ('.png', '.svg', '.pdf', '.jpg'):
        return 'figures'
    if suffix in ('.log', '.txt'):
        return 'logs'
    if suffix == '.json' or stem == 'decision_fits':
        return 'config'
    for ending, category in (('_diagnostics', 'diagnostics'), ('_weights', 'weights'),
                             ('_returns', 'returns'), ('_components', 'returns'),
                             ('_predictions', 'predictions')):
        if stem.endswith(ending):
            return category
    if stem in ('summary', 'comparison', 'prediction_validation') or stem.endswith(('_summary', '_results')):
        return 'metrics'
    return 'diagnostics'


class RunArtifacts:
    """Resolve logical filenames; keep algorithms unaware of filesystem layout."""
    def __init__(self, root, layout='flat'):
        if layout not in ('flat', 'categorized'):
            raise ValueError('layout must be flat or categorized')
        self.root, self.layout = Path(root), layout

    def prepare(self):
        ensure_fresh_output(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def path(self, name):
        relative = _relative_name(name)
        if self.layout == 'categorized':
            relative = PurePosixPath(_category(str(relative))) / relative
        target = _inside(self.root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_backtest(self, name, result, diagnostics=None):
        result.returns.to_csv(self.path(f'{name}_returns.csv'))
        result.weights.to_csv(self.path(f'{name}_weights.csv'))
        if diagnostics is not None:
            diagnostics.to_csv(self.path(f'{name}_diagnostics.csv'))

    def finalize(self):
        """Index completed files; historical flat runs need no manifest to read."""
        artifacts = {}
        for path in sorted(self.root.rglob('*')):
            if not path.is_file() or path.name == MANIFEST:
                continue
            relative = path.relative_to(self.root)
            _inside(self.root, relative)
            logical = relative
            if self.layout == 'categorized' and relative.parts[0] in CATEGORIES:
                logical = Path(*relative.parts[1:])
            key = logical.as_posix()
            if key in artifacts:
                raise ValueError(f'Duplicate logical artifact: {key}')
            artifacts[key] = relative.as_posix()
        manifest = dict(layout_version=1, layout=self.layout, artifacts=artifacts)
        (self.root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        return manifest


def artifact_path(root, name):
    """Read either an indexed run or a historical flat run, without guessing."""
    root = Path(root)
    logical = _relative_name(name).as_posix()
    manifest = root / MANIFEST
    if not manifest.is_file():
        return _inside(root, logical)
    content = json.loads(manifest.read_text())
    if content.get('layout_version') != 1 or not isinstance(content.get('artifacts'), dict):
        raise ValueError('Unsupported artifact manifest')
    relative = content['artifacts'].get(logical)
    if relative is None:
        raise ValueError(f'Artifact missing from manifest: {logical}')
    return _inside(root, _relative_name(relative))


def configure_output(parser, args, direction, *, optional_flat=False):
    """Apply CLI defaults and check destinations before any model fitting."""
    if args.output is None and not (optional_flat and args.layout == 'flat'):
        args.output = default_output(direction)
    args.artifacts = None if args.output is None else RunArtifacts(args.output, args.layout)
    if args.output is not None:
        try:
            ensure_fresh_output(args.output)
        except ValueError as error:
            parser.error(str(error))
