"""P7 training data: channel stacking, the TRAIN-ONLY dataset opener, id checks, in-memory and sharded row sources, the exporter.

Everything here is train-side. Test rows are reachable only through `guard.open_test_rows` (tests/test_p7_static.py enforces it): the
opener below hands N14Dataset an EMPTY second split file, so the test split of the object it returns is empty by construction.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import edalint
from edalint.datasets.circuitnet import N14Dataset, design_of, read_split_csv

# depth slices 0-3 are the ones MAVI multiplies by its predicted maps
IR_CHANNELS = ['power_i', 'power_s', 'power_sca', 'power_all'] + [f'power_t[{k}]' for k in range(20)]
CONGESTION_CHANNELS = ['macro_region', 'RUDY', 'RUDY_pin']
CHANNELS = {'ir_drop': IR_CHANNELS, 'congestion': CONGESTION_CHANNELS}
TRAIN_DESIGNS = ('RISCY', 'RISCY-FPU', 'Vortex-small', 'nvdla-small')


def stack_inputs(task: str, ins: Dict[str, np.ndarray]) -> np.ndarray:
    """`assemble(...)[0]` -> one float32 array, the authors' layout: IR (24, H, W) = power_i, power_s, power_sca, power_all, then the 20
    power_t frames (the model adds a leading channel axis: (1, 24, H, W), irdrop_dataset.py); congestion (3, H, W) = macro_region, RUDY, RUDY_pin."""
    if task == 'ir_drop':
        x = np.concatenate([ins['power_i'][None], ins['power_s'][None], ins['power_sca'][None], ins['power_all'][None], ins['power_t']], 0)
    elif task == 'congestion':
        x = np.stack([ins[f] for f in CONGESTION_CHANNELS], 0)
    else:
        raise ValueError(f'P7 trains ir_drop and congestion only, got {task!r}')
    return x.astype(np.float32)


def model_input(task: str, x):
    """(B, C, H, W) tensor -> what the network takes: IR drop gets the extra axis (B, 1, 24, H, W), congestion is unchanged."""
    return x.unsqueeze(1) if task == 'ir_drop' else x


def model_target(task: str, y):
    """(B, H, W) -> MAVI predicts (B, H, W); GPDL predicts (B, 1, H, W)."""
    return y if task == 'ir_drop' else y.unsqueeze(1)


# -- ids ----------------------------------------------------------------------------------------------------------------------
def train_csv_path() -> Path:
    return Path(edalint.__file__).resolve().parent / 'data' / 'train_N14.csv'


def train_names() -> List[str]:
    """Every id of the official train split file (6,790 rows), before loader skips."""
    return read_split_csv(train_csv_path())


def check_disjoint(ids: Sequence[str], eval_ids_by_design: Dict[str, Sequence[str]]):
    """AssertionError unless `ids` has no duplicates and is disjoint, by id AND by design, from the scored (eval) ids."""
    ids = list(ids)
    assert len(set(ids)) == len(ids), 'duplicate train ids'
    ev = {i for v in eval_ids_by_design.values() for i in v}
    both = set(ids) & ev
    assert not both, f'{len(both)} train ids are also eval ids, e.g. {sorted(both)[:3]}'
    dboth = {design_of(i) for i in ids} & set(eval_ids_by_design)
    assert not dboth, f'train and eval share designs {sorted(dboth)}'


def check_train_ids(ids: Sequence[str], eval_ids_by_design: Dict[str, Sequence[str]], csv_names: Sequence[str],
                    skipped_ids: Sequence[str] = ()) -> Dict[str, Any]:
    """check_disjoint, plus: the ids are a subset of the official train file with exactly len(csv) - (loader-skipped csv rows) entries.
    Returns a small report."""
    ids = list(ids)
    check_disjoint(ids, eval_ids_by_design)
    csv = set(csv_names)
    assert set(ids) <= csv, 'train ids outside the official train file'
    skip = set(skipped_ids) & csv
    assert len(ids) == len(csv) - len(skip), f'{len(ids)} train ids but {len(csv)} listed - {len(skip)} skipped'
    return {'n_train': len(ids), 'n_listed': len(csv), 'n_skipped': len(skip), 'designs': sorted({design_of(i) for i in ids})}


def open_train_rows(task: str, *, designs: Optional[Sequence[str]] = None, cache_dir: Optional[str] = None, root: Optional[str] = None,
                    select=None, stage: bool = True, manifest_path: Optional[str] = None) -> N14Dataset:
    """N14Dataset (authors mode) over the official TRAIN rows only. Its second split file is empty, so `ids(<second split>)` is []."""
    empty = Path(tempfile.mkdtemp(prefix='p7_empty_split_')) / 'empty.csv'
    empty.write_text('')
    return N14Dataset(task=task, mode='authors', root=root, cache_dir=cache_dir, designs=designs, select=select, stage=stage,
                      split_csvs=(str(train_csv_path()), str(empty)), manifest_path=manifest_path)


# -- row sources --------------------------------------------------------------------------------------------------------------
class MemorySource:
    """Preloaded rows: x (C, H, W) float32, y (H, W) float32."""

    def __init__(self, ids: Sequence[str], xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]):
        self.ids, self._x, self._y = list(ids), list(xs), list(ys)
        self.designs = [design_of(i) for i in self.ids]
        self.nonfinite_ids = [i for i, x, y in zip(self.ids, self._x, self._y) if not (np.isfinite(x).all() and np.isfinite(y).all())]

    def __len__(self):
        return len(self.ids)

    def get(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._x[k], self._y[k]

    @classmethod
    def from_dataset(cls, task: str, ds, ids: Sequence[str]) -> 'MemorySource':
        keep, xs, ys = [], [], []
        for sid in ids:
            s = ds.get(sid)
            if s is None:
                continue
            keep.append(sid)
            xs.append(stack_inputs(task, s.inputs))
            ys.append(np.asarray(s.label, np.float32))
        return cls(keep, xs, ys)


class SubsetSource:
    """A view of a source restricted to `ids` (arm B: the registered 979 fit_ids)."""

    def __init__(self, source, ids: Sequence[str]):
        pos = {i: k for k, i in enumerate(source.ids)}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise ValueError(f'{len(missing)} requested ids are not in the source, e.g. {missing[:3]}')
        self._src, self._k = source, [pos[i] for i in ids]
        self.ids = list(ids)
        self.designs = [design_of(i) for i in self.ids]
        bad = set(source.nonfinite_ids)
        self.nonfinite_ids = [i for i in self.ids if i in bad]

    def __len__(self):
        return len(self.ids)

    def get(self, k: int):
        return self._src.get(self._k[k])


SHARD_ROWS = 128
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)          # fixed member timestamps: the same rows give the same shard bytes (reproducible sha256)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class _ShardWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.zf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
        self.ids: List[str] = []

    def add(self, sid: str, x: np.ndarray, y: np.ndarray):
        k = len(self.ids)
        for key, arr in ((f'x_{k}', x), (f'y_{k}', y)):
            zi = zipfile.ZipInfo(key + '.npy', FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            with self.zf.open(zi, 'w', force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.ascontiguousarray(arr), allow_pickle=False)
        self.ids.append(sid)

    def close(self):
        self.zf.close()


def export_rows(task: str, ds, ids: Sequence[str], out_dir, shard_rows: int = SHARD_ROWS, label: str = 'train') -> Dict[str, Any]:
    """Write `ids` of `ds` as compressed float32 .npz shards, one design per shard (<= shard_rows rows each), plus manifest.json with the
    sha256 of every shard. Rows are the exact `assemble(..., 'authors')` output (stack_inputs / label), no rounding. Member x_k / y_k of
    a shard is row k of its 'ids'; np.load(shard)['x_k'] decompresses that one row only. Returns the manifest (also written)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_design: Dict[str, List[str]] = {}
    for sid in ids:
        by_design.setdefault(design_of(sid), []).append(sid)
    shards, nonfinite, missing = [], [], []

    def close(writer, rel, des):
        writer.close()
        shards.append({'path': rel, 'design': des, 'ids': writer.ids, 'sha256': sha256_file(out / rel), 'bytes': (out / rel).stat().st_size})

    for des in sorted(by_design):
        part, writer, rel = 0, None, None
        for sid in sorted(by_design[des]):
            s = ds.get(sid)
            if s is None:
                missing.append(sid)
                continue
            x, y = stack_inputs(task, s.inputs), np.asarray(s.label, np.float32)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                nonfinite.append(sid)
            if writer is None or len(writer.ids) >= shard_rows:
                if writer is not None:
                    close(writer, rel, des)
                rel = f'{des}/shard_{part:03d}.npz'
                part += 1
                writer = _ShardWriter(out / rel)
            writer.add(sid, x, y)
        if writer is not None:
            close(writer, rel, des)
    man = {'task': task, 'label': label, 'format': 'npz (deflate), members x_<k> float32 (C,H,W) and y_<k> float32 (H,W), k = index into shard ids',
           'channels': CHANNELS[task], 'n_rows': sum(len(s['ids']) for s in shards), 'shards': shards,
           'nonfinite_ids': nonfinite, 'missing_ids': missing, 'loader_skipped': getattr(ds, 'skipped', [])}
    (out / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    return man


class ShardSource:
    """Rows from an exported directory (random access; verifies each shard's sha256 on first open)."""

    def __init__(self, out_dir, verify: bool = True, max_open: int = 8):
        self.dir = Path(out_dir)
        self.manifest = json.loads((self.dir / 'manifest.json').read_text(encoding='utf-8'))
        self.ids, self._loc, self.designs = [], [], []
        for si, sh in enumerate(self.manifest['shards']):
            for k, sid in enumerate(sh['ids']):
                self.ids.append(sid)
                self._loc.append((si, k))
                self.designs.append(sh['design'])
        self.nonfinite_ids = list(self.manifest['nonfinite_ids'])
        self._verify, self._ok, self._open, self._max_open = verify, set(), OrderedDict(), max_open

    def __len__(self):
        return len(self.ids)

    def _shard(self, si: int):
        if si in self._open:
            self._open.move_to_end(si)
            return self._open[si]
        sh = self.manifest['shards'][si]
        p = self.dir / sh['path']
        if self._verify and si not in self._ok:
            if sha256_file(p) != sh['sha256']:
                raise ValueError(f'{p}: sha256 differs from the manifest')
            self._ok.add(si)
        z = np.load(p)
        self._open[si] = z
        while len(self._open) > self._max_open:
            self._open.popitem(last=False)[1].close()
        return z

    def get(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        si, j = self._loc[k]
        z = self._shard(si)
        return z[f'x_{j}'], z[f'y_{j}']
