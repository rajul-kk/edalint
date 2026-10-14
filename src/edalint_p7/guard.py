"""P7 freeze guard, `p7_code_hash`, the kernel-start check and `open_test_rows`, the ONLY path to test rows in P7.

Why this exists (protocol_p7.md sections 6 and 11 D7): the N14 dataset guard keys on `circuitnet_n14_*` -> FROZEN_n14.json, which is already
satisfied, so it does not stop a pre-P7-freeze test read. `open_test_rows` therefore calls `require_frozen(dataset='p7')` (FROZEN_p7.json with
`code_hash` = edalint.protocol.code_hash(), unchanged) AND recomputes `p7_code_hash` over src/edalint_p7/**/*.py plus scripts/p7_run_*.py.
Dev scripts (scripts/p7_dev_*.py) are outside the hash on purpose. Nothing else in src/edalint_p7 or scripts/p7_run_*.py may read test rows;
tests/test_p7_static.py checks that.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from edalint.core import ProtocolError
from edalint.datasets.circuitnet import N14Dataset
from edalint.protocol import code_hash, require_frozen

REPO_ROOT = Path(__file__).resolve().parents[2]


def p7_files(repo_root=None) -> List[Path]:
    """The hashed files: src/edalint_p7/**/*.py (no __pycache__) and scripts/p7_run_*.py, sorted by posix path relative to the repo root."""
    root = Path(repo_root or REPO_ROOT)
    files = [p for p in (root / 'src' / 'edalint_p7').rglob('*.py') if '__pycache__' not in p.parts]
    files += list((root / 'scripts').glob('p7_run_*.py'))
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def p7_code_hash(repo_root=None) -> str:
    """Same recipe as edalint.protocol.code_hash(): sha256 over relative posix path, NUL, bytes with CRLF -> LF, NUL, per file in sorted order."""
    root = Path(repo_root or REPO_ROOT)
    h = hashlib.sha256()
    for p in p7_files(root):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b'\0')
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
        h.update(b'\0')
    return h.hexdigest()


def check_p7_freeze(protocol_dir=None, repo_root=None, what: str = 'P7 test rows') -> Dict[str, Any]:
    """Raise ProtocolError unless protocol/FROZEN_p7.json exists, belongs to dataset 'p7', its `code_hash` equals the installed edalint hash
    (require_frozen) and its `p7_code_hash` equals the hash of the P7 code now. Returns the freeze record."""
    info = require_frozen(protocol_dir, what=what, dataset='p7')
    if info.get('dataset') != 'p7':
        raise ProtocolError(f'refused: the freeze record says dataset {info.get("dataset")!r}, not \'p7\'.')
    want = info.get('p7_code_hash')
    if not want:
        raise ProtocolError('refused: FROZEN_p7.json has no "p7_code_hash" field.')
    now = p7_code_hash(repo_root)
    if want != now:
        raise ProtocolError(f'refused: P7 code hash differs from the frozen protocol (frozen {want[:12]}..., now {now[:12]}...). '
                            'Any edit of src/edalint_p7 or scripts/p7_run_*.py after the freeze invalidates it.')
    return info


def kernel_start_check(bundle_root) -> Dict[str, Any]:
    """Run first thing in a Kaggle kernel: the bundle carries protocol/FROZEN_p7.json, src/edalint, src/edalint_p7 and scripts/p7_run_*.py.
    Recomputes both hashes and raises ProtocolError on any mismatch (the caller exits)."""
    root = Path(bundle_root)
    return check_p7_freeze(root / 'protocol', root, what='P7 kernel start')


@dataclass
class TestRows:
    """What `open_test_rows` returns: the staged dataset, the scored ids by design, and the freeze record that allowed it."""
    __test__ = False                 # not a pytest class
    ds: Any
    ids_by_design: Dict[str, List[str]]
    freeze: Dict[str, Any]


def open_test_rows(task: str, *, protocol_dir=None, repo_root=None, cache_dir=None, root=None, _factory=N14Dataset) -> TestRows:
    """THE only function in P7 that may open the CircuitNet-N14 test rows. Order: P7 freeze + both code hashes first, only then the dataset
    object is built and the test designs are staged (all official test rows, uncapped, no train staging)."""
    info = check_p7_freeze(protocol_dir, repo_root, what=f'opening the N14 test rows for P7 ({task})')
    ds = _factory(task=task, mode='authors', root=root, cache_dir=cache_dir, stage=False,
                  protocol_dir=str(protocol_dir) if protocol_dir else None)
    for des in sorted({d for (s, d) in ds._wanted if s == 'test'}):
        ds.stage_design(des, ('test',))
    by: Dict[str, List[str]] = {}
    for sid in ds.ids('test'):
        by.setdefault(ds.meta(sid)['design'], []).append(sid)
    return TestRows(ds, {d: by[d] for d in sorted(by)}, info)
