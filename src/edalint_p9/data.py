"""P9 data access: authors-space N14 samples from the registered staging cache, and lazy sample sequences.

The cache (`E:/edalint_data/n14_registered_cache/<task>/authors/<design>/<feature>/<name>.npy`, float32, resized, not yet min-max scaled)
was written by the registered N14 run (edalint code hash d3f6d2f6..., tarballs with sha256 in manifests/MANIFEST.json). It holds the
registered fit sample of the train designs (979 rows per task) and every official test row. Samples are assembled with
edalint.datasets.circuitnet.assemble (authors mode), the same function the registered run used. Every file read is sha256-hashed inline
and folded into a per-(task, design) digest, so the result files certify exactly which bytes were used.

Test designs are only reachable through edalint_p9.guard (open_n14_test); N14Cache refuses them unless constructed with the token the guard
passes after the freeze check.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from edalint.core import ProtocolError, Sample
from edalint.datasets.circuitnet import TASKS, assemble, packaged_split_csvs, read_split_csv, design_of

CACHE_ROOT = Path('E:/edalint_data/n14_registered_cache')
TRAIN_DESIGNS = ('RISCY', 'RISCY-FPU', 'Vortex-small', 'nvdla-small')
TEST_DESIGNS = ('zero-riscy', 'openc910-1', 'Vortex-large')
_GUARD_TOKEN = object()


def task_members(task: str) -> List[str]:
    spec = TASKS[task]
    return list(spec['inputs']) + list(spec['label'])


def official_rows() -> Dict[str, set]:
    tr, te = packaged_split_csvs()
    return {'train': set(read_split_csv(tr)), 'test': set(read_split_csv(te))}


class N14Cache:
    """Row listing and loading for one task. listing(design) = cached rows that are CSV-listed and have every member file."""

    def __init__(self, task: str, root: Optional[Path] = None, allow_test=None):
        self.task, self.root = task, Path(root or CACHE_ROOT) / task / 'authors'
        self.members = task_members(task)
        self._allow_test = allow_test is _GUARD_TOKEN
        self.digests: Dict[str, 'hashlib._Hash'] = {}
        self.n_files: Dict[str, int] = {}
        rows = official_rows()
        self.split_of = {}
        for sp, names in rows.items():
            for n in names:
                self.split_of[n] = sp

    def _check(self, design: str):
        if design in TEST_DESIGNS and not self._allow_test:
            raise ProtocolError(f'refused: {design} is an N14 test design; open it through edalint_p9.guard.open_n14_test after the P9 freeze')

    def listing(self, design: str) -> List[str]:
        self._check(design)
        sets = []
        for f in self.members:
            d = self.root / design / f
            sets.append({p.stem for p in d.glob('*.npy')} if d.is_dir() else set())
        names = set.intersection(*sets) if sets else set()
        want = 'test' if design in TEST_DESIGNS else 'train'
        return sorted(n for n in names if self.split_of.get(n) == want and design_of(n) == design)

    def _read(self, design: str, feature: str, name: str) -> np.ndarray:
        p = self.root / design / feature / f'{name}.npy'
        b = p.read_bytes()
        h = self.digests.setdefault(design, hashlib.sha256())
        h.update(f'{feature}/{name}\0'.encode())
        h.update(hashlib.sha256(b).digest())
        self.n_files[design] = self.n_files.get(design, 0) + 1
        return np.load(io.BytesIO(b))

    def load(self, design: str, name: str) -> Sample:
        self._check(design)
        ins, lab = assemble(self.task, 'authors', lambda f: self._read(design, f, name))
        sp = 'test' if design in TEST_DESIGNS else 'train'
        return Sample(name, sp, design, ins, lab, {'name': name, 'mode': 'authors'})

    def digest_report(self) -> Dict[str, dict]:
        """Order-dependent digest (rows are always read in a fixed order by the runners) and file count per design."""
        return {d: {'sha256_of_file_hashes': h.hexdigest(), 'n_file_reads': self.n_files.get(d, 0)} for d, h in sorted(self.digests.items())}


class LazySamples:
    """Indexable, re-iterable, lazily loaded sequence of samples: loader(key) -> Sample."""

    def __init__(self, keys: Sequence, loader: Callable):
        self.keys, self.loader = list(keys), loader

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.loader(self.keys[i])

    def __iter__(self):
        for k in self.keys:
            yield self.loader(k)


def seeded_split(names: Sequence[str], sizes: Sequence[int], seed: int) -> List[List[str]]:
    """Deterministic partition of sorted(names) into consecutive blocks of the given sizes after a seeded permutation; the last size may
    be -1 (= the rest)."""
    rng = np.random.default_rng(seed)
    order = [sorted(names)[i] for i in rng.permutation(len(names))]
    out, a = [], 0
    for k in sizes:
        b = len(order) if k == -1 else a + k
        out.append(sorted(order[a:b]))
        a = b
    return out
