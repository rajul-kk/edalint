"""Configuration for `edalint lint`: edalint.toml, or a [tool.edalint] table in pyproject.toml.

Search order (first found wins): --config FILE; edalint.toml in the current directory; [tool.edalint] in ./pyproject.toml; edalint.toml
in the dataset folder. Command-line flags override the file. Every key is optional.

    dataset = "data/irdrop"              # what to lint when the command line names no dataset (relative to the config file)
    split = "all"                        # split(s) to read; default 'train' for the built-in benchmarks, 'all' for a folder
    select = ["E001", "E002"]            # checks to run (default E001-E006)
    ignore = ["E005"]                    # checks not to run
    fail-on = "warning"                  # lowest severity that fails the run: error | warning | info | never
    baseline = "edalint-baseline.json"   # accepted findings (edalint lint --write-baseline FILE writes one)
    max-samples = 400                    # evenly spaced subsample per split for the per-sample checks

    [options]                            # check thresholds: any field of edalint.checks.Options, e.g.
    dup_corr = 0.999

    [dataset-format]                     # folder datasets only: see edalint.datasets.folder
    label_key = "label"
    inputs = ["current", "eff_dist"]
    hotspot = { kind = "frac_of_max", value = 0.9 }
    label_range = [0.0, 0.05]
    test_splits = ["test"]
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:                                                       # pragma: no cover - exercised on 3.9 / 3.10 only
    import tomli as tomllib

from .checks import Options

SEVERITY_RANK = {'none': -1, 'info': 0, 'warning': 1, 'error': 2}
FAIL_ON = ('error', 'warning', 'info', 'never')
TOP_KEYS = {'dataset', 'split', 'select', 'ignore', 'fail-on', 'baseline', 'max-samples', 'options', 'dataset-format'}
FOLDER_KEYS = {'name', 'label_key', 'inputs', 'primary_input', 'task', 'hotspot', 'label_range', 'label_units', 'test_splits',
               'samples_csv'}


class ConfigError(ValueError):
    """The configuration file is malformed or names an unknown key."""


def _read(path: Path, table: Optional[str]) -> Dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'{path}: {exc}') from exc
    if table:
        data = data.get('tool', {}).get(table, {})
    return data


def find_config(explicit: Optional[str] = None, dataset: Optional[str] = None, cwd: Optional[Path] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    """(config dict, file it came from or None). An explicit --config that does not exist is an error."""
    cwd = Path(cwd or Path.cwd())
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ConfigError(f'config file {p} does not exist')
        return validate(_read(p, 'edalint' if p.name == 'pyproject.toml' else None), p), p
    if (cwd / 'edalint.toml').is_file():
        p = cwd / 'edalint.toml'
        return validate(_read(p, None), p), p
    if (cwd / 'pyproject.toml').is_file():
        d = _read(cwd / 'pyproject.toml', 'edalint')
        if d:
            return validate(d, cwd / 'pyproject.toml'), cwd / 'pyproject.toml'
    if dataset and Path(dataset.removeprefix('folder:')).is_dir():
        p = Path(dataset.removeprefix('folder:')) / 'edalint.toml'
        if p.is_file():
            return validate(_read(p, None), p), p
    return {}, None


def validate(cfg: Dict[str, Any], where: Path) -> Dict[str, Any]:
    bad = sorted(set(cfg) - TOP_KEYS)
    if bad:
        raise ConfigError(f'{where}: unknown key(s) {bad}; allowed {sorted(TOP_KEYS)}')
    if cfg.get('fail-on', 'warning') not in FAIL_ON:
        raise ConfigError(f'{where}: fail-on must be one of {FAIL_ON}, not {cfg["fail-on"]!r}')
    fields = {f.name for f in dataclasses.fields(Options)} - {'split', 'max_samples'}
    bad = sorted(set(cfg.get('options', {})) - fields)
    if bad:
        raise ConfigError(f'{where}: unknown [options] key(s) {bad}; allowed {sorted(fields)}')
    bad = sorted(set(cfg.get('dataset-format', {})) - FOLDER_KEYS)
    if bad:
        raise ConfigError(f'{where}: unknown [dataset-format] key(s) {bad}; allowed {sorted(FOLDER_KEYS)}')
    for k in ('select', 'ignore'):
        if k in cfg and not (isinstance(cfg[k], list) and all(isinstance(c, str) for c in cfg[k])):
            raise ConfigError(f'{where}: {k} must be a list of check codes')
    return cfg


def options_from(cfg: Dict[str, Any], split: str, max_samples: Optional[int]) -> Options:
    kw = dict(cfg.get('options', {}))
    for k, v in list(kw.items()):
        if isinstance(v, list):
            kw[k] = tuple(v)
    return Options(split=split, max_samples=max_samples, **kw)


def fails(worst: str, fail_on: str) -> bool:
    return fail_on != 'never' and SEVERITY_RANK[worst] >= SEVERITY_RANK[fail_on]
