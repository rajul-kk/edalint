"""P9 freeze guard, `p9_code_hash`, and the ONLY openers of test data in P9 (protocol_p9.md section 7).

The N14 and ICCAD'23 dataset guards bind freeze records whose code_hash is the registered d3f6d2f6... / 750d252f... (CODE_HASH_LINEAGE.json);
they are not rewritten. P9 therefore has its own freeze record protocol/FROZEN_p9.json: dataset 'p9', code_hash = edalint.protocol.code_hash()
now, p9_code_hash = sha256 over src/edalint_p9/**/*.py and scripts/p9_run_*.py (same recipe as P7). Dev scripts (scripts/p9_dev_*.py) are
outside the hash and may not open test data; tests/test_p9_static.py checks that only this module calls the test openers' internals.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from edalint.core import ProtocolError
from edalint.protocol import require_frozen

from . import data as D

REPO_ROOT = Path(__file__).resolve().parents[2]


def p9_files(repo_root=None) -> List[Path]:
    root = Path(repo_root or REPO_ROOT)
    files = [p for p in (root / 'src' / 'edalint_p9').rglob('*.py') if '__pycache__' not in p.parts]
    files += list((root / 'scripts').glob('p9_run_*.py'))
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def p9_code_hash(repo_root=None) -> str:
    root = Path(repo_root or REPO_ROOT)
    h = hashlib.sha256()
    for p in p9_files(root):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b'\0')
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
        h.update(b'\0')
    return h.hexdigest()


def check_p9_freeze(protocol_dir=None, repo_root=None, what: str = 'P9 test data') -> Dict[str, Any]:
    info = require_frozen(protocol_dir, what=what, dataset='p9')
    if info.get('dataset') != 'p9':
        raise ProtocolError(f'refused: the freeze record says dataset {info.get("dataset")!r}, not \'p9\'.')
    want = info.get('p9_code_hash')
    if not want:
        raise ProtocolError('refused: FROZEN_p9.json has no "p9_code_hash" field.')
    now = p9_code_hash(repo_root)
    if want != now:
        raise ProtocolError(f'refused: P9 code hash differs from the frozen protocol (frozen {want[:12]}..., now {now[:12]}...). '
                            'Any edit of src/edalint_p9 or scripts/p9_run_*.py after the freeze invalidates it.')
    return info


def open_n14_test(task: str, *, protocol_dir=None, repo_root=None, root=None):
    """N14Cache that may read the test designs; freeze checked first. Returns (cache, freeze record)."""
    info = check_p9_freeze(protocol_dir, repo_root, what=f'opening N14 test rows for P9 ({task})')
    return D.N14Cache(task, root=root, allow_test=D._GUARD_TOKEN), info


def open_iccad_hidden(*, protocol_dir=None, repo_root=None, root=None):
    """(ICCAD23Dataset, loader for hidden cases, freeze record). The loader bypasses the ICCAD'23 dataset guard (whose freeze binds an older
    hash) only after the P9 freeze check."""
    info = check_p9_freeze(protocol_dir, repo_root, what='opening ICCAD\'23 hidden cases for P9')
    from edalint.datasets.iccad23 import ICCAD23Dataset
    ds = ICCAD23Dataset(root=root)
    return ds, ds._load, info
