"""Baseline interface, registry and shared helpers.

A baseline has `fit(train)` and `predict(sample) -> 2D array on the sample's label grid, in label units`. `train` is a re-iterable
stream of Samples (so multi-pass fits are possible without holding every map in RAM). Baselines see ONLY `sample.inputs` (plus
`sample.meta['netlist']` for the exact solver, which is the runtime reference and is not a skill claim) and the training labels.
Add a baseline with `@register('name')` on a class; the zoo is discovered through BASELINES.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..core import MapDataset, Sample, area_resize, box_mean

BASELINES: Dict[str, Callable[[], 'Baseline']] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        BASELINES[name] = cls
        return cls
    return deco


class SampleStream:
    """Re-iterable view of a dataset split (or an id list); loads lazily through the (guarded) dataset."""

    def __init__(self, ds: MapDataset, ids: Sequence[str]):
        self.ds, self.ids = ds, list(ids)

    def __len__(self):
        return len(self.ids)

    def __iter__(self) -> Iterator[Sample]:
        for i in self.ids:
            s = self.ds.get(i)
            if s is not None:
                yield s


def stream(ds: MapDataset, split_or_ids, max_n: Optional[int] = None) -> SampleStream:
    ids = ds.ids(split_or_ids, max_n) if isinstance(split_or_ids, str) else list(split_or_ids)
    return SampleStream(ds, ids)


class Baseline:
    name = 'base'
    tasks: Tuple[str, ...] = ()              # empty = any task
    needs_channels: Tuple[str, ...] = ()     # input groups that must exist
    needs_netlist = False
    needs_sklearn = False

    def applicable(self, ds: MapDataset, sample: Optional[Sample] = None) -> Tuple[bool, str]:
        if self.tasks and ds.task not in self.tasks:
            return False, f'only for task(s) {self.tasks}, dataset task is {ds.task!r}'
        if self.needs_sklearn:
            try:
                import sklearn  # noqa: F401
            except ImportError:
                return False, 'scikit-learn not installed'
        if sample is not None:
            miss = [c for c in self.needs_channels if c not in sample.inputs]
            if miss:
                return False, f'missing input channel(s) {miss}'
            if self.needs_netlist and not sample.meta.get('netlist'):
                return False, 'sample has no SPICE netlist'
        return True, 'ok'

    def fit(self, train: Iterable[Sample]) -> 'Baseline':
        return self

    def predict(self, s: Sample) -> np.ndarray:                  # pragma: no cover - abstract
        raise NotImplementedError


def resize_to(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D map to `shape`: area average when shrinking in both axes, bilinear otherwise."""
    if a.shape == tuple(shape):
        return a
    if shape[0] <= a.shape[0] and shape[1] <= a.shape[1]:
        return area_resize(a, shape)
    from scipy.ndimage import zoom
    z = zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)
    if z.shape != tuple(shape):                                # rounding of zoom
        z = np.pad(z, [(0, max(0, shape[0] - z.shape[0])), (0, max(0, shape[1] - z.shape[1]))], mode='edge')[:shape[0], :shape[1]]
    return z


def pixel_features(s: Sample, boxes: Sequence[int] = (3, 9)) -> np.ndarray:
    """(H*W, F): every input channel, plus its box means over each window in `boxes` (all channels, edge-replicated)."""
    x = s.stack()
    feats = [x] + [box_mean(x, b) for b in boxes]
    f = np.concatenate(feats, axis=0)
    return f.reshape(f.shape[0], -1).T


def sample_pixels(rng: np.random.Generator, n_pix: int, k: int) -> np.ndarray:
    return rng.choice(n_pix, size=min(k, n_pix), replace=False)
