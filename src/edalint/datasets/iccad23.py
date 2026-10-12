"""ICCAD'23 Problem C loader (ASU-VDA-Lab/ML-for-IR-drop). Layout and conventions: docs/format_notes.md section 1.

Splits: `train` = 100 fake (BeGAN) + 10 visible real cases, `test` = the 10 hidden real cases (7, 8, 9, 10, 13, 14, 15, 16, 19, 20).
Convenience subsets (no extra data): `fake`, `visible` (the 10 visible real cases). `test` is guarded (needs protocol/FROZEN.json).

Native size per case; nothing is resized unless `resize=(H, W)` is given (optional, documented transform edalint.core.resize_sample).
Units: current A, IR drop V (the official MAE is reported in mV), eff_dist and pdn_density have no stated unit.

Twin pairs: each real case has a twin (same grid, same current-source count, different PDN). `design_id` is the shared key;
`meta['twin_of']` names the partner. The 10 hidden cases are 5 designs.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core import MapDataset, Sample, resize_sample

HIDDEN = (7, 8, 9, 10, 13, 14, 15, 16, 19, 20)
VISIBLE = (1, 2, 3, 4, 5, 6, 11, 12, 17, 18)
DEFAULT_ROOT = os.environ.get('EDALINT_ICCAD23', r'E:\edalint_data\iccad23\repo')
VDD = 1.1
INPUTS = ('current', 'eff_dist', 'pdn_density')


def read_map_csv(path) -> np.ndarray:
    """Comma-separated, no header, float64. (pandas C parser; np.loadtxt is ~3x slower.)"""
    return pd.read_csv(path, header=None, dtype=np.float64, engine='c').values


def csv_shape(path) -> Tuple[int, int]:
    """(rows, cols) without parsing the values: number of lines and commas in the first line + 1."""
    n_nl, last = 0, b''
    with open(path, 'rb') as fh:
        first = fh.readline()
        n_nl = 1 if first.endswith(b'\n') else 0
        last = first[-1:]
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            n_nl += chunk.count(b'\n')
            last = chunk[-1:]
    rows = n_nl + (0 if last == b'\n' else 1)
    return rows, first.count(b',') + 1


def count_current_sources(netlist_path) -> int:
    n = 0
    with open(netlist_path, 'rb') as fh:
        for line in fh:
            if line[:1] in (b'I', b'i'):
                n += 1
    return n


class _Case:
    __slots__ = ('id', 'name', 'group', 'paths', 'number')

    def __init__(self, id, name, group, paths, number):
        self.id, self.name, self.group, self.paths, self.number = id, name, group, paths, number


def discover(root) -> List[_Case]:
    root = Path(root)
    b = root / 'benchmarks'
    cases: List[_Case] = []
    fk = b / 'fake-circuit-data'
    if fk.is_dir():
        for p in sorted(fk.glob('current_map*_current.csv')):
            nm = p.name[:-len('_current.csv')]
            paths = {'current': p, 'eff_dist': fk / f'{nm}_eff_dist.csv', 'pdn_density': fk / f'{nm}_pdn_density.csv',
                     'ir_drop': fk / f'{nm}_ir_drop.csv', 'netlist': fk / f'{nm}.sp'}
            cases.append(_Case(f'fake/{nm}', nm, 'fake', paths, int(re.sub(r'\D', '', nm) or -1)))
    for group, sub in (('real', 'real-circuit-data'), ('hidden', 'hidden-real-circuit-data')):
        d = b / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob('testcase*'), key=lambda q: int(re.sub(r'\D', '', q.name) or 0)):
            if not p.is_dir():
                continue
            paths = {'current': p / 'current_map.csv', 'eff_dist': p / 'eff_dist_map.csv', 'pdn_density': p / 'pdn_density.csv',
                     'ir_drop': p / 'ir_drop_map.csv', 'netlist': p / 'netlist.sp'}
            num = int(re.sub(r'\D', '', p.name))
            cases.append(_Case(f'{group}/{p.name}', p.name, group, paths, num))
    return cases


def derive_twins(cases: List[_Case]) -> Dict[str, Dict[str, object]]:
    """Group the real + hidden cases by (grid shape, number of current sources). Uses file sizes / line counts of the CURRENT map
    and the netlist only; no map value and no label is read. Returns {case id: {'design_id', 'twin_of'}}."""
    keyed: Dict[tuple, List[_Case]] = {}
    for c in cases:
        if c.group == 'fake':
            continue
        key = (csv_shape(c.paths['current']), count_current_sources(c.paths['netlist']))
        keyed.setdefault(key, []).append(c)
    out: Dict[str, Dict[str, object]] = {}
    for key, cs in keyed.items():
        did = f'real_{key[0][0]}x{key[0][1]}_{key[1]}src'
        for c in cs:
            others = [o.id for o in cs if o is not c]
            out[c.id] = {'design_id': did, 'twin_of': others[0] if len(others) == 1 else (others or None), 'group_size': len(cs)}
    return out


class ICCAD23Dataset(MapDataset):
    def __init__(self, root: Optional[str] = None, resize: Optional[Tuple[int, int]] = None, protocol_dir: Optional[str] = None,
                 cache_items: int = 4):
        root = Path(root or DEFAULT_ROOT)
        cases = discover(root)
        if not cases:
            raise FileNotFoundError(f'no ICCAD23 cases found under {root}')
        twins = derive_twins(cases)
        self._cases = {c.id: c for c in cases}
        self.root = root
        self.resize = resize
        meta: Dict[str, dict] = {}
        splits: Dict[str, List[str]] = {'train': [], 'test': [], 'fake': [], 'visible': []}
        for c in cases:
            tw = twins.get(c.id, {})
            split = 'test' if c.group == 'hidden' else 'train'
            meta[c.id] = {'split': split, 'group': c.group, 'name': c.name, 'case_number': c.number,
                          'design_id': tw.get('design_id', c.id), 'twin_of': tw.get('twin_of'),
                          'twin_group_size': tw.get('group_size', 1), 'netlist': str(c.paths['netlist']), 'vdd': VDD}
            meta[c.id]['design'] = meta[c.id]['design_id']
            splits[split].append(c.id)
            if c.group == 'fake':
                splits['fake'].append(c.id)
            elif c.group == 'real':
                splits['visible'].append(c.id)
        super().__init__('iccad23', 'ir_drop', splits, meta, primary_input='current',
                         hotspot={'kind': 'frac_of_max', 'value': 0.9}, label_range=(0.0, VDD), label_units='V',
                         protocol_dir=protocol_dir, cache_items=cache_items)

    def _load(self, sid: str) -> Sample:
        c = self._cases[sid]
        m = self._meta[sid]
        inputs = {k: read_map_csv(c.paths[k]) for k in INPUTS}
        label = read_map_csv(c.paths['ir_drop'])
        s = Sample(sid, m['split'], m['design'], inputs, label, {k: v for k, v in m.items() if k not in ('split', 'design')})
        s.meta['group'] = c.group
        return resize_sample(s, self.resize) if self.resize else s

    def shape_of(self, sid: str) -> Tuple[int, int]:
        """Grid shape from the file layout (no values read); safe for test cases."""
        return csv_shape(self._cases[sid].paths['current'])

    def load_unguarded(self, sid: str) -> Sample:
        """Loader smoke test ONLY (shape / loading checks on test cases before the freeze). Bypasses the protocol guard:
        callers must not compute any statistic, metric or comparison on the returned label."""
        return self._load(sid)
