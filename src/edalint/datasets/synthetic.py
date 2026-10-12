"""Synthetic ICCAD-like data with a known physical mechanism, plus fault injectors, for tests and development.

IR drop = (current * K)(x) * (1 + 0.5 * eff_dist/max) with K(r) = 1/(1 + r) (a smooth radial kernel), so a Green's-function
baseline can recover it and the label correlates with the current map in the STORED orientation. Faults are switched on by flags.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from ..core import ListDataset, Sample


def _kernel(r_max=12):
    yy, xx = np.mgrid[-r_max:r_max + 1, -r_max:r_max + 1]
    return 1.0 / (1.0 + np.hypot(xx, yy))


def make_case(rng: np.random.Generator, n: int, m: Optional[int] = None):
    m = m or n
    cur = np.zeros((n, m))
    for _ in range(rng.integers(4, 9)):
        cx, cy, sg = rng.uniform(0, n), rng.uniform(0, m), rng.uniform(2, 6)
        yy, xx = np.mgrid[0:n, 0:m]
        cur += rng.uniform(0.5, 2.0) * np.exp(-((xx - cy) ** 2 + (yy - cx) ** 2) / (2 * sg ** 2))
    cur *= 1e-7
    bumps = rng.uniform(0, [n, m], size=(4, 2))
    yy, xx = np.mgrid[0:n, 0:m]
    eff = np.min([np.hypot(yy - b[0], xx - b[1]) for b in bumps], axis=0)
    pdn = float(rng.integers(1, 4)) + gaussian_filter(rng.normal(size=(n, m)), 3) * 2.0
    ir = fftconvolve(cur, _kernel(), mode='same') * (1 + 0.5 * eff / eff.max()) * 1e4
    return cur, eff, pdn, ir


def make_synthetic(n_train_designs: int = 6, n_test_designs: int = 2, per_design: int = 2, size: int = 40, seed: int = 0,
                   transpose_train_labels: bool = False, n_duplicates: int = 0, leak_design: bool = False, cross_twin: bool = False,
                   n_zero_labels: int = 0, n_nan: int = 0, n_negative: int = 0, no_hotspot_label: bool = False,
                   name: str = 'synthetic', hotspot_rule: Optional[dict] = None, guard_test: bool = True) -> ListDataset:
    """Build a ListDataset with splits 'train' and 'test'. Each design has `per_design` samples; per_design == 2 makes each design
    a twin pair (meta['twin_of'] links the two samples)."""
    rng = np.random.default_rng(seed)
    samples: List[Sample] = []
    for split, nd in (('train', n_train_designs), ('test', n_test_designs)):
        for d in range(nd):
            did = f'{split[:2]}design{d}'
            n = size + 2 * (d % 3)
            ids = []
            for k in range(per_design):
                cur, eff, pdn, ir = make_case(rng, n)
                sid = f'{did}_{k}'
                ids.append(sid)
                samples.append(Sample(sid, split, did, {'current': cur, 'eff_dist': eff, 'pdn_density': pdn}, ir, {'group': split}))
            if per_design == 2:
                samples[-2].meta['twin_of'] = ids[1]
                samples[-1].meta['twin_of'] = ids[0]
    tr = [s for s in samples if s.split == 'train']
    if transpose_train_labels:
        for s in tr:
            s.label = np.ascontiguousarray(s.label.T)             # stored with swapped axes
    for j in range(n_duplicates):
        a = tr[j]
        samples.append(Sample(f'dup{j}', 'train', f'dupdesign{j}', {k: v.copy() for k, v in a.inputs.items()}, a.label.copy(), {}))
    if leak_design:                                               # a test-design sample (and its input) copied into train
        te = [s for s in samples if s.split == 'test'][0]
        samples.append(Sample('leak0', 'train', te.design, {k: v.copy() for k, v in te.inputs.items()}, te.label.copy(), {}))
    if cross_twin:
        a = [s for s in samples if s.split == 'train'][0]
        b = [s for s in samples if s.split == 'test'][0]
        a.meta['twin_of'], b.meta['twin_of'] = b.id, a.id
    for j in range(n_zero_labels):
        tr[j].label = np.zeros_like(tr[j].label)
    for j in range(n_nan):
        tr[j].inputs['current'] = tr[j].inputs['current'].copy()
        tr[j].inputs['current'][3, 3] = np.nan
    for j in range(n_negative):
        tr[-1 - j].label = tr[-1 - j].label.copy()
        tr[-1 - j].label[2, 2] = -1e-3
    if no_hotspot_label:
        tr[0].label = np.zeros_like(tr[0].label)
    return ListDataset(samples, name=name, test_splits=('test',) if guard_test else (), task='ir_drop', primary_input='current',
                       hotspot=hotspot_rule or {'kind': 'frac_of_max', 'value': 0.9}, label_range=(0.0, 1.1), label_units='V')
