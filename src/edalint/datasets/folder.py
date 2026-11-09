"""Your own dataset: a folder of .npz files, one per sample. This is what `edalint lint <path>` reads.

Layout (split = sub-folder; .npz files directly in the root form one split named 'all-samples'):

    mydata/
      train/<sample>.npz         each file: the label map (key 'label', or [dataset] label_key) and any number of input maps,
      val/<sample>.npz           every other array in the file is an input: (H, W) or (K, H, W)
      test/<sample>.npz
      samples.csv                optional: columns id, design[, twin_of]; id = <split>/<file stem> or the bare file stem

Design ids drive E004 (leakage across splits). Without samples.csv every sample is its own design, so only identical sample ids and
explicit twin links can be flagged. Files are opened lazily with allow_pickle=False (an .npz cannot run code).

No split is guarded by default: the frozen-protocol guard is for registered benchmark studies. Set `test_splits` in the [dataset]
table of the config to guard splits behind a freeze record in `<root>/protocol/FROZEN_<name>.json`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import MapDataset, Sample

ROOT_SPLIT = 'all-samples'


def _scan(root: Path) -> Dict[str, List[Path]]:
    splits: Dict[str, List[Path]] = {}
    top = sorted(root.glob('*.npz'))
    if top:
        splits[ROOT_SPLIT] = top
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        files = sorted(d.glob('*.npz'))
        if files:
            splits[d.name] = files
    return splits


def _read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if rows and 'id' not in rows[0]:
        raise ValueError(f'{path}: needs an "id" column (have {list(rows[0])})')
    return {r['id'].strip(): {k: (v or '').strip() for k, v in r.items()} for r in rows}


class FolderDataset(MapDataset):
    def __init__(self, root, name: Optional[str] = None, label_key: str = 'label', inputs: Optional[Sequence[str]] = None,
                 primary_input: Optional[str] = None, task: str = 'map', hotspot: Optional[dict] = None,
                 label_range: Optional[Tuple[float, float]] = None, label_units: str = '', test_splits: Sequence[str] = (),
                 protocol_dir: Optional[str] = None, samples_csv: Optional[str] = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f'dataset folder {self.root} does not exist')
        files = _scan(self.root)
        if not files:
            raise ValueError(f'no .npz files in {self.root} or its sub-folders')
        table_path = Path(samples_csv) if samples_csv else self.root / 'samples.csv'
        table = _read_csv(table_path) if table_path.is_file() else {}
        self.label_key, self.input_keys = label_key, list(inputs) if inputs else None
        self._paths: Dict[str, Path] = {}
        splits: Dict[str, List[str]] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for split, paths in files.items():
            for p in paths:
                sid = p.stem if split == ROOT_SPLIT else f'{split}/{p.stem}'
                row = table.get(sid) or table.get(p.stem) or {}
                m: Dict[str, Any] = {'split': split, 'design': row.get('design') or sid, 'path': str(p)}
                if row.get('twin_of'):
                    m['twin_of'] = [t.strip() for t in row['twin_of'].split(';') if t.strip()]
                self._paths[sid] = p
                splits.setdefault(split, []).append(sid)
                meta[sid] = m
        if primary_input is None:
            primary_input = self.input_keys[0] if self.input_keys else self._first_input()
        super().__init__(name or self.root.name, task, splits, meta, primary_input=primary_input, hotspot=hotspot,
                         label_range=tuple(label_range) if label_range else None, label_units=label_units,
                         protocol_dir=protocol_dir or (str(self.root / 'protocol') if test_splits else None))
        self.test_splits = tuple(test_splits)

    def _first_input(self) -> str:
        p = next(iter(self._paths.values()))
        with np.load(p, allow_pickle=False) as z:
            files = z.files
        keys = [k for k in files if k != self.label_key]
        if self.label_key not in files:
            raise ValueError(f'{p}: no label array {self.label_key!r} (arrays: {files}); set label_key in [dataset-format]')
        if not keys:
            raise ValueError(f'{p}: only the label array; edalint lints input/label map pairs, so each .npz needs at least one input')
        return keys[0]

    def path(self, sid: str) -> Path:
        return self._paths[sid]

    def _load(self, sid: str) -> Sample:
        m = self.meta(sid)
        with np.load(self._paths[sid], allow_pickle=False) as z:
            if self.label_key not in z.files:
                raise KeyError(f'no label array {self.label_key!r} (arrays: {z.files})')
            keys = self.input_keys or [k for k in z.files if k != self.label_key]
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise KeyError(f'missing input array(s) {missing} (arrays: {z.files})')
            inputs = {k: np.asarray(z[k], dtype=np.float64) for k in keys}
            label = np.asarray(z[self.label_key], dtype=np.float64)
        return Sample(sid, m['split'], m['design'], inputs, label, dict(m))
