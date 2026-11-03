"""IC-ThermBench steady-state scopes (level2..level5) as a MapDataset: inputs = power / k / grid / scalar channels, label = temperature (K).

Files: <root>/<scope>_steady/input.mat (B, P, Z, Y, X) and output.mat (B, Z, Y, X), HDF5 (MATLAB v7.3). Z == 1 in every released scope, so the
maps are 2-D (64 x 64). Spatial input channels are stored transposed in-plane relative to the temperature (found 2026-10-05, Thermo repo
`scripts/icb_orientation_check.py`); `fix_orientation=True` swaps their in-plane axes so inputs and label share one convention. Default False
(as stored), because a linter must see the data as released. Scalar channels are returned as constant maps.

Split: the benchmark's own index split, no shuffling (72 / 8 / 20 % train / val / test; test is a guarded split). Design = case: level5 from
`manifest.json` (one row per sample, in file order; its `index` is the position within the case); levels 2-4 from the README layout (Case1-10 x 1500, contiguous), recorded as meta['case_source'] = 'readme'.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from edalint.core import MapDataset, Sample

SCOPES = {
    'level2': ['chiplet_power', 'grid_x', 'grid_y'],
    'level3': ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k'],
    'level4': ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k', 'ambient_K', 'h_w_m2k', 'r_convec_k_per_w'],
    'level5': ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k', 'ambient_K', 'h_w_m2k', 'r_convec_k_per_w'],
}
SCALAR_CHANNELS = {'ambient_K', 'h_w_m2k', 'r_convec_k_per_w'}
README_CASES = {'level2': (1, 1500), 'level3': (1, 1500), 'level4': (1, 1500)}          # (first case number, samples per case)
TRAIN_RATIO, VAL_FRACTION = 0.8, 0.9                                                    # the benchmark's index split


def split_indices(total: int) -> Dict[str, range]:
    trainval = max(2, min(int(total * TRAIN_RATIO), total - 1))
    n_train = max(1, min(int(trainval * VAL_FRACTION), trainval - 1))
    return {'train': range(0, n_train), 'val': range(n_train, trainval), 'test': range(trainval, total)}


class ICThermBenchDataset(MapDataset):
    test_splits = ('test',)

    def __init__(self, root, scope: str = 'level3', fix_orientation: bool = False, protocol_dir: Optional[str] = None):
        import h5py                                                                     # optional dependency, imported lazily
        if scope not in SCOPES:
            raise ValueError(f'unknown scope {scope!r}; use one of {sorted(SCOPES)}')
        self.root, self.scope, self.fix_orientation = Path(root), scope, fix_orientation
        self.channels = SCOPES[scope]
        folder = self.root / f'{scope}_steady'
        self._fx = h5py.File(folder / 'input.mat', 'r')['data']
        self._fy = h5py.File(folder / 'output.mat', 'r')['data']
        total, p, z, h, w = self._fx.shape
        if self._fy.shape != (total, z, h, w) or p != len(self.channels) or z != 1:
            raise ValueError(f'unexpected shapes {self._fx.shape} / {self._fy.shape} for {scope} (want P={len(self.channels)}, Z=1)')
        if fix_orientation and h != w:
            raise ValueError('fix_orientation needs a square in-plane grid')
        cases = self._cases(folder, total)
        idx = split_indices(total)
        self._index: Dict[str, int] = {}
        splits: Dict[str, List[str]] = {}
        meta: Dict[str, dict] = {}
        for sp, rng in idx.items():
            splits[sp] = []
            for i in rng:
                sid = f'{scope}_{i:05d}'
                self._index[sid] = i
                splits[sp].append(sid)
                meta[sid] = {'design': cases[i][0], 'case_source': cases[i][1], 'split': sp}
        super().__init__(f'icthermbench-{scope}', 'thermal', splits, meta, primary_input='chiplet_power',
                         hotspot={'kind': 'frac_of_max', 'value': 0.9}, label_range=None, label_units='K', protocol_dir=protocol_dir)

    def _cases(self, folder: Path, total: int):
        man = folder / 'manifest.json'
        if man.exists():
            rows = json.loads(man.read_text())                                           # 'index' is local to the case: use row order
            if len(rows) == total:
                return [(r['case'], 'manifest') for r in rows]
        first, per = README_CASES.get(self.scope, (1, max(1, total)))
        return [(f'Case{first + i // per}', 'readme') for i in range(total)]

    def _load(self, sid: str) -> Sample:
        i = self._index[sid]
        x = np.asarray(self._fx[i, :, 0], dtype=np.float32)                              # (P, Y, X)
        y = np.asarray(self._fy[i, 0], dtype=np.float32)                                 # (Y, X)
        ins = {}
        for c, name in enumerate(self.channels):
            a = x[c]
            if self.fix_orientation and name not in SCALAR_CHANNELS:
                a = np.ascontiguousarray(a.T)
            ins[name] = a
        m = self.meta(sid)
        return Sample(sid, m['split'], m['design'], ins, y, dict(m))
