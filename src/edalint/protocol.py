"""Frozen-protocol guard and provenance helpers.

Nothing may read a TEST split (ICCAD'23 hidden cases, CircuitNet test designs) until that dataset's freeze file exists and its
`code_hash` equals the hash of the installed edalint source. Freeze state is per dataset/protocol: `protocol/FROZEN_<key>.json`
(key 'n14' for CircuitNet-N14) or, for the ICCAD'23 freeze made before this split, `protocol/FROZEN.json` with "dataset": "iccad23".
P4 creates the files; this module only checks them.

A freeze file must contain at least: {"code_hash": "<sha256 from edalint.protocol.code_hash()>", ...}.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .core import ProtocolError

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent.parent


def code_hash() -> str:
    """sha256 over the relative path and bytes of every .py file under the edalint package (sorted)."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.rglob('*.py')):
        h.update(p.relative_to(PKG_DIR).as_posix().encode())
        h.update(b'\0')
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
        h.update(b'\0')
    return h.hexdigest()


def protocol_dir(path: Optional[str] = None) -> Path:
    return Path(path or os.environ.get('EDALINT_PROTOCOL_DIR') or REPO_DIR / 'protocol')


def protocol_key(dataset: str) -> str:
    """Freeze key of a dataset name: all CircuitNet-N14 tasks/modes share the key 'n14'; every other name is its own key."""
    return 'n14' if dataset.startswith('circuitnet_n14') else dataset


def frozen_file(path: Optional[str] = None, dataset: Optional[str] = None) -> Path:
    """The freeze file that governs `dataset`: protocol/FROZEN_<key>.json, or the legacy protocol/FROZEN.json (see frozen_info)."""
    d = protocol_dir(path)
    if dataset is None:
        return d / 'FROZEN.json'
    new = d / f'FROZEN_{protocol_key(dataset)}.json'
    return new if new.is_file() else d / 'FROZEN.json'


def frozen_info(path: Optional[str] = None, dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Freeze record for `dataset` (None = the legacy protocol/FROZEN.json, whatever it covers).

    Per-dataset freezes: protocol/FROZEN_<key>.json. The legacy protocol/FROZEN.json (the ICCAD'23 freeze) counts for a dataset only if
    its "dataset" field equals the dataset's key, so one protocol's freeze never unlocks another dataset's test split."""
    if dataset is None:
        f = protocol_dir(path) / 'FROZEN.json'
        return json.loads(f.read_text(encoding='utf-8')) if f.is_file() else None
    f = frozen_file(path, dataset)
    if not f.is_file():
        return None
    info = json.loads(f.read_text(encoding='utf-8'))
    if f.name == 'FROZEN.json' and info.get('dataset') != protocol_key(dataset):
        return None
    return info


def require_frozen(path: Optional[str] = None, what: str = 'this operation', dataset: Optional[str] = None) -> Dict[str, Any]:
    """Return the freeze record, or raise ProtocolError (missing file, no code_hash, or hash mismatch).
    With `dataset`, the record must belong to that dataset's protocol (see frozen_info)."""
    f = frozen_file(path, dataset)
    info = frozen_info(path, dataset)
    if info is None:
        why = 'belongs to another dataset' if f.is_file() else 'does not exist'
        raise ProtocolError(f'refused: {what} touches a test split, but {f} {why} (no frozen protocol for {dataset or "this dataset"}).')
    want = info.get('code_hash')
    if not want:
        raise ProtocolError(f'refused: {f} has no "code_hash" field.')
    now = code_hash()
    if want != now:
        raise ProtocolError(f'refused: code hash differs from the frozen protocol (frozen {want[:12]}..., now {now[:12]}...). '
                            'Any edit of src/edalint after the freeze invalidates it.')
    return info


def git_head(cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=cwd or REPO_DIR).stdout.strip()
    except OSError:
        return ''


def git_dirty(cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True,
                              cwd=cwd or REPO_DIR).stdout.strip()
    except OSError:
        return ''


def cpu_info() -> Dict[str, Any]:
    """CPU model string (registry on Windows, /proc/cpuinfo on Linux), logical core count."""
    name = ''
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
            name = str(winreg.QueryValueEx(k, 'ProcessorNameString')[0]).strip()
    except Exception:                                                # noqa: BLE001 - not Windows
        try:
            for line in open('/proc/cpuinfo', encoding='utf-8'):
                if line.lower().startswith('model name'):
                    name = line.split(':', 1)[1].strip()
                    break
        except OSError:
            name = platform.processor()
    return {'cpu_model': name or platform.processor(), 'logical_cores': os.cpu_count()}


def versions() -> Dict[str, str]:
    import numpy
    import pandas
    import scipy
    out = {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
           'pandas': pandas.__version__, 'platform': platform.platform()}
    try:
        import sklearn
        out['sklearn'] = sklearn.__version__
    except ImportError:
        out['sklearn'] = 'absent'
    try:
        import pyamg
        out['pyamg'] = pyamg.__version__
    except ImportError:
        out['pyamg'] = 'absent'
    from . import __version__
    out['edalint'] = __version__
    out.update(cpu_info())
    return out


def provenance() -> Dict[str, Any]:
    return {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'git_head': git_head(), 'git_dirty_tracked': git_dirty(),
            'code_hash': code_hash(), 'versions': versions(), 'argv': list(sys.argv)}
