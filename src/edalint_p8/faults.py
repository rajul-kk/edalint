"""Fault injectors for the P8 validation (protocol/protocol_faults.md). Every injector takes a list of Sample copies, an rng and a severity
level 0..4 (weakest to strongest) and returns modified samples; `build` wraps samples into an in-memory dataset with no guarded split.

A fault is detected when the TARGET check returns an error or warning whose message contains the fault's `needle` (see FAULTS). An
unrelated finding does not count. Injectors never touch the dataset they were drawn from: `materialise` deep-copies first.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from edalint.checks import _zdesc
from edalint.core import CheckResult, ListDataset, MapDataset, Sample, area_resize

LEVELS = 5


def materialise(ds: MapDataset, ids: Sequence[str]) -> List[Sample]:
    """Deep copies of the named samples (None-loading samples are dropped)."""
    out = []
    for sid in ids:
        s = ds.get(sid)
        if s is not None:
            out.append(Sample(s.id, s.split, s.design, {k: np.array(v, copy=True) for k, v in s.inputs.items()}, np.array(s.label, copy=True),
                              copy.deepcopy(s.meta)))
    return out


def build(samples: Sequence[Sample], like: MapDataset) -> ListDataset:
    """In-memory dataset with the labelling attributes of `like`; no split is guarded (the draw never contains a test sample)."""
    return ListDataset(list(samples), name=like.name, task=like.task, test_splits=(), primary_input=like.primary_input, hotspot=like.hotspot,
                       label_range=like.label_range, label_units=like.label_units)


def assign_splits(samples: List[Sample], rng: np.random.Generator, by_design: bool = True) -> List[Sample]:
    """Clean two-split layout ('train' / 'val'): whole designs go to one side (design-disjoint), so E003/E004 have a clean baseline.
    If fewer than 2 designs exist, the samples are split at random instead (E004 is then not meaningful and the caller must not use it)."""
    designs = sorted({s.design for s in samples})
    if by_design and len(designs) >= 2:
        perm = rng.permutation(len(designs))
        side = {designs[j]: ('val' if k < max(1, len(designs) // 3) else 'train') for k, j in enumerate(perm)}
        for s in samples:
            s.split = side[s.design]
            s.meta['split'] = s.split
    else:
        for s in samples:
            s.split = 'val' if rng.random() < 0.3 else 'train'
            s.meta['split'] = s.split
    return samples


def _count(frac: float, n: int) -> int:
    return int(min(n, max(1, round(frac * n))))


FRACS = (0.002, 0.005, 0.01, 0.02, 0.05)          # E002: fraction of the sampled cases affected
E005_FRACS = (0.10, 0.25, 0.50, 0.75, 1.00)
SNRS = (20.0, 10.0, 5.0, 2.0, 1.0)
PIXELS = (1, 4, 16, 64, None)                     # E001: None = 1% of the pixels
DUP_COUNTS = (1, 2, 5, 10, 20)
CORRS = (0.99, 0.995, 0.999, 0.9995, 0.99999)
DESIGNS_MOVED = (1, 2, 3, 5, 8)
TWINS_MOVED = (1, 2, 4, 8, 16)


def _first_channel(s: Sample) -> str:
    return next(iter(s.inputs))


# -- E001 ------------------------------------------------------------------------------------------------------------------------
def nonfinite(value: float):
    def inj(samples, rng, level):
        s = samples[int(rng.integers(len(samples)))]
        nm = _first_channel(s)
        a = np.array(s.inputs[nm], dtype=np.float64 if s.inputs[nm].dtype == np.float64 else np.float32, copy=True)
        k = PIXELS[level] or max(1, int(round(0.01 * a.size)))
        a.flat[rng.choice(a.size, size=min(k, a.size), replace=False)] = value
        s.inputs[nm] = a
        return samples
    return inj


def grid_mismatch(samples, rng, level):
    s = samples[int(rng.integers(len(samples)))]
    nm = _first_channel(s)
    h, w = s.inputs[nm].shape[-2:]
    s.inputs[nm] = area_resize(s.inputs[nm], (max(2, h // 2), max(2, w // 2)))
    return samples


# -- E002 ------------------------------------------------------------------------------------------------------------------------
def _pick(samples, rng, frac):
    return [samples[j] for j in rng.choice(len(samples), size=_count(frac, len(samples)), replace=False)]


def zero_label(samples, rng, level):
    for s in _pick(samples, rng, FRACS[level]):
        s.label = np.zeros_like(s.label)
    return samples


def negative_pixel(samples, rng, level):
    for s in _pick(samples, rng, FRACS[level]):
        L = np.array(s.label, copy=True)
        L.flat[int(rng.integers(L.size))] = -max(1e-3 * float(np.abs(L).max()), 1e-6) * 10
        s.label = L
    return samples


def zero_channel(samples, rng, level):
    for s in _pick(samples, rng, FRACS[level]):
        nm = _first_channel(s)
        s.inputs[nm] = np.zeros_like(s.inputs[nm])
    return samples


# -- E003 ------------------------------------------------------------------------------------------------------------------------
def _other(split: str) -> str:
    return 'val' if split == 'train' else 'train'


def exact_copy(samples, rng, level):
    new = []
    for j in rng.choice(len(samples), size=min(DUP_COUNTS[level], len(samples)), replace=False):
        s = samples[int(j)]
        new.append(Sample(f'{s.id}__dup', _other(s.split), s.design + '__dup', {k: np.array(v, copy=True) for k, v in s.inputs.items()},
                          np.array(s.label, copy=True), {}))
    return samples + new


def _noisy(a: np.ndarray, target: float, rng, g: int = 32) -> np.ndarray:
    """a plus white noise scaled (bisection) so the E003 descriptor correlation of a and the result equals `target`."""
    a = np.asarray(a, dtype=np.float64)
    if a.std() == 0 or target >= 1.0:
        return a.copy()
    n = rng.normal(size=a.shape)
    d0 = _zdesc(a, g)
    lo, hi = 0.0, 50.0 * a.std()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d1 = _zdesc(a + mid * n, g)
        c = float(d0 @ d1) / d0.size
        lo, hi = (mid, hi) if c > target else (lo, mid)
    return a + 0.5 * (lo + hi) * n


def near_copy(samples, rng, level):
    new = []
    for j in rng.choice(len(samples), size=min(5, len(samples)), replace=False):
        s = samples[int(j)]
        new.append(Sample(f'{s.id}__near', _other(s.split), s.design + '__near',
                          {k: _noisy(v, CORRS[level], rng) if v.ndim == 2 else v.copy() for k, v in s.inputs.items()},
                          _noisy(s.label, CORRS[level], rng), {}))
    return samples + new


# -- E004 ------------------------------------------------------------------------------------------------------------------------
def design_overlap(samples, rng, level):
    """k designs get ONE sample reassigned to the other side (the design then occurs in both splits)."""
    count = {}
    for s in samples:
        count[s.design] = count.get(s.design, 0) + 1
    designs = sorted(d for d, n in count.items() if n >= 2)             # a single-sample design cannot occur on both sides (amendment A4)
    k = min(DESIGNS_MOVED[level], len(designs))
    for d in rng.choice(designs, size=k, replace=False):
        members = [s for s in samples if s.design == d]
        s = members[int(rng.integers(len(members)))]
        s.split = _other(s.split)
        s.meta['split'] = s.split
    return samples


def twin_cross(samples, rng, level):
    """k sample pairs get a twin link across the two splits (pairs are chosen from different splits)."""
    tr = [s for s in samples if s.split == 'train']
    va = [s for s in samples if s.split == 'val']
    k = min(TWINS_MOVED[level], len(tr), len(va))
    for a, b in zip(rng.choice(len(tr), size=k, replace=False), rng.choice(len(va), size=k, replace=False)):
        tr[int(a)].meta['twin_of'], va[int(b)].meta['twin_of'] = va[int(b)].id, tr[int(a)].id
    return samples


# -- E005 ------------------------------------------------------------------------------------------------------------------------
def transpose_label(samples, rng, level, fracs=E005_FRACS):
    for s in _pick(samples, rng, fracs[level]):
        if s.label.shape[0] == s.label.shape[1]:
            s.label = np.ascontiguousarray(s.label.T)
    return samples


def flip_label(samples, rng, level):
    for s in _pick(samples, rng, E005_FRACS[level]):
        s.label = np.ascontiguousarray(s.label[::-1, :])
    return samples


def transpose_noisy(samples, rng, level):
    for s in samples:
        if s.label.shape[0] == s.label.shape[1]:
            s.label = np.ascontiguousarray(s.label.T)
        sd = float(np.std(s.label))
        s.label = s.label + rng.normal(scale=sd / SNRS[level], size=s.label.shape)
    return samples


@dataclass(frozen=True)
class Fault:
    name: str
    check: str
    inject: Callable
    needle: str                     # text of the intended finding
    needs: str = ''                 # '' | 'designs' (>= 2 designs) | 'twins' (base has twin_of) | 'square'
    severities: tuple = tuple(range(LEVELS))


FAULTS: Dict[str, Fault] = {f.name: f for f in (
    Fault('nan_pixels', 'E001', nonfinite(np.nan), 'non-finite'),
    Fault('inf_pixels', 'E001', nonfinite(np.inf), 'non-finite'),
    Fault('grid_mismatch', 'E001', grid_mismatch, 'different grid from the label', severities=(4,)),
    Fault('zero_label', 'E002', zero_label, 'all-zero label'),
    Fault('negative_pixel', 'E002', negative_pixel, 'negative label'),
    Fault('zero_channel', 'E002', zero_channel, 'is all-zero in'),
    Fault('exact_copy', 'E003', exact_copy, 'byte-identical'),
    Fault('near_copy', 'E003', near_copy, 'near-duplicate pair(s) ACROSS'),
    Fault('design_overlap', 'E004', design_overlap, 'occur in both', needs='designs'),
    Fault('twin_cross', 'E004', twin_cross, 'twin pair link', needs='twins'),
    Fault('transpose_label', 'E005', transpose_label, 'primary channel', needs='square'),
    Fault('flip_label', 'E005', flip_label, 'primary channel'),
    Fault('transpose_noisy', 'E005', transpose_noisy, 'primary channel', needs='square'),
)}


def detected(res: CheckResult, fault: Fault) -> bool:
    """The target check flagged an error or warning that contains the fault's needle."""
    return res.status == 'flagged' and any(f.severity in ('error', 'warning') and fault.needle in f.message for f in res.findings)


def applicable(fault: Fault, samples: Sequence[Sample]) -> Optional[str]:
    """None if the fault can be injected on this base, else the reason it cannot."""
    if fault.needs == 'designs':
        count = {}
        for s in samples:
            count[s.design] = count.get(s.design, 0) + 1
        if len(count) < 2 or max(count.values()) < 2:
            return 'fewer than 2 designs, or no design with 2 samples'
    if fault.needs == 'twins' and not any(s.meta.get('twin_of') for s in samples):
        return 'base has no twin metadata'
    if fault.needs == 'square' and not any(s.label.shape[0] == s.label.shape[1] for s in samples):
        return 'no square maps'
    return None
