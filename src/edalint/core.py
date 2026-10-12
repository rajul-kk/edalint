"""Core types: Sample, MapDataset, Finding / CheckResult, and the array helpers shared by loaders, checks and baselines.

Design follows fieldlint: a check is a function `check(ds, opt) -> CheckResult`; it never raises on bad data, it reports.
Loaders (edalint.datasets.*) are unguarded low-level readers. The dataset layer (MapDataset.get) is guarded: reading a
sample of a TEST split requires a frozen protocol (edalint.protocol), so no label of a test split is touched before the freeze.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SEVERITIES = ('error', 'warning', 'info')


# -- array helpers ------------------------------------------------------------------------------------------------------
def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) weights of exact fractional-area averaging (what cv2.INTER_AREA does when shrinking)."""
    edges_out = np.linspace(0.0, n_in, n_out + 1)
    lo, hi = edges_out[:-1, None], edges_out[1:, None]
    cell_lo = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, cell_lo + 1) - np.maximum(lo, cell_lo), 0.0, None)
    return overlap / (n_in / n_out)


def area_resize(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Area-average resize of the last two axes to `shape` = (H, W). Exactly mean-preserving; numpy only."""
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape[-2:]
    if (h, w) == tuple(shape):
        return a.copy()
    wy, wx = _area_matrix(h, shape[0]), _area_matrix(w, shape[1])
    return np.einsum('ij,...jk,lk->...il', wy, a, wx, optimize=True)


def box_mean(a: np.ndarray, size: int) -> np.ndarray:
    """Mean over a size x size window (edge-replicated), last two axes."""
    from scipy.ndimage import uniform_filter
    a = np.asarray(a, np.float64)
    return uniform_filter(a, size=(1,) * (a.ndim - 2) + (size, size), mode='nearest')


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


# -- findings and results (mirrors fieldlint.rules) -----------------------------------------------------------------------
@dataclass
class Finding:
    code: str
    severity: str                                           # error | warning | info
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    samples: List[str] = field(default_factory=list)        # ids of (some of) the affected samples


@dataclass
class CheckResult:
    code: str
    name: str
    status: str = 'ok'                                      # ok | flagged | skipped
    severity: str = 'none'                                  # worst severity among the findings
    findings: List[Finding] = field(default_factory=list)
    summary: str = ''
    metrics: Dict[str, Any] = field(default_factory=dict)
    skip_reason: str = ''

    def add(self, severity: str, message: str, samples: Sequence[str] = (), **metrics) -> Finding:
        f = Finding(self.code, severity, message, dict(metrics), list(samples)[:10])
        self.findings.append(f)
        self.finalise()
        return f

    def finalise(self) -> 'CheckResult':
        if self.status != 'skipped':
            real = [f for f in self.findings if f.severity in ('error', 'warning')]
            self.status = 'flagged' if real else 'ok'
            order = {'info': 0, 'warning': 1, 'error': 2}
            self.severity = max((f.severity for f in self.findings), key=order.get) if self.findings else 'none'
        return self

    def to_dict(self):
        return asdict(self)


def skipped(code: str, name: str, reason: str) -> CheckResult:
    return CheckResult(code, name, 'skipped', 'none', summary=f'skipped: {reason}', skip_reason=reason)


# -- samples and datasets -------------------------------------------------------------------------------------------------
@dataclass
class Sample:
    """One layout / case. `inputs` maps a channel-group name to an array: (H, W) or (K, H, W) (e.g. CircuitNet power_t).
    `label` is the 2D target map. `meta` carries design / twin information and any file paths."""
    id: str
    split: str
    design: str
    inputs: Dict[str, np.ndarray]
    label: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.label.shape[-2:])

    def channels(self) -> List[Tuple[str, np.ndarray]]:
        out = []
        for k, v in self.inputs.items():
            if v.ndim == 2:
                out.append((k, v))
            else:
                out.extend((f'{k}[{i}]', v[i]) for i in range(v.shape[0]))
        return out

    def stack(self) -> np.ndarray:
        """(C, H, W) float64; channels whose grid differs from the label's are area-resized to it."""
        return np.stack([a if a.shape == self.shape else area_resize(a, self.shape) for _, a in self.channels()]).astype(np.float64)

    def channel_names(self) -> List[str]:
        return [n for n, _ in self.channels()]

    def primary(self, name: str) -> np.ndarray:
        a = self.inputs[name]
        return a if a.shape == self.shape else area_resize(a, self.shape)


def resize_sample(s: Sample, shape: Tuple[int, int]) -> Sample:
    """OPTIONAL transform (never applied by default): area-resize every input and the label to `shape`.
    Changes the label (and the metric scale); a resize is a protocol decision, see docs/format_notes.md section 1.4."""
    return Sample(s.id, s.split, s.design, {k: area_resize(v, shape) for k, v in s.inputs.items()},
                  area_resize(s.label, shape), {**s.meta, 'resized_from': s.shape})


class ProtocolError(RuntimeError):
    """Raised when a test split is touched without a frozen protocol."""


class MapDataset:
    """Lazy collection of Samples with named splits. Subclasses implement `_load(id) -> Sample` and fill `_meta`.

    name, task          report labels
    splits              {split name: [ids]}; names in `test_splits` are guarded
    primary_input       channel used by orientation / superposition code (e.g. 'current')
    hotspot             rule used by E006: {'kind': 'frac_of_max'|'abs'|'top_frac', 'value': float}
    label_range         (lo, hi) plausible range of label values, for E002 out-of-range tests (None = no bound)
    skipped             [{'id' | 'design': ..., 'reason': ...}]: samples dropped by the loader, with the reason
    """
    test_splits: Tuple[str, ...] = ('test',)

    def __init__(self, name: str, task: str, splits: Dict[str, List[str]], meta: Dict[str, Dict[str, Any]],
                 primary_input: str = 'current', hotspot: Optional[dict] = None,
                 label_range: Optional[Tuple[float, float]] = None, label_units: str = '',
                 protocol_dir: Optional[str] = None, cache_items: int = 6):
        self.name, self.task = name, task
        self.splits = {k: list(v) for k, v in splits.items()}
        self._meta = meta
        self.primary_input = primary_input
        self.hotspot = hotspot or {'kind': 'frac_of_max', 'value': 0.9}
        self.label_range = label_range
        self.label_units = label_units
        self.protocol_dir = protocol_dir
        self.aux_splits: set = set()                 # splits that are views of other splits (fake, visible, per-design): not leakage units
        self.skipped: List[dict] = []
        self.errors: Dict[str, str] = {}
        self._cache: 'OrderedDict[str, Sample]' = OrderedDict()
        self._cache_items = cache_items

    # metadata (never touches map values)
    def __len__(self):
        return len(self.ids('all'))

    def meta(self, sid: str) -> Dict[str, Any]:
        return self._meta.get(sid, {})

    def _test_ids(self):
        return {i for s in self.test_splits for i in self.splits.get(s, [])}

    def is_test(self, sid: str) -> bool:
        return sid in self._test_ids()

    def is_test_split(self, split: Optional[str]) -> bool:
        """True if reading `split` would touch test samples ('all' and any split overlapping a test split do)."""
        if split is None or split == 'all':
            return bool(self._test_ids())
        return bool(set(self.splits.get(split, [])) & self._test_ids())

    def ids(self, split: Optional[str] = None, max_n: Optional[int] = None) -> List[str]:
        """Ids of a split (None or 'all' = every split, deduplicated); evenly spaced subsample if max_n."""
        if split is None or split == 'all':
            seen, ids = set(), []
            for v in self.splits.values():
                for i in v:
                    if i not in seen:
                        seen.add(i)
                        ids.append(i)
        else:
            if split not in self.splits:
                raise KeyError(f'unknown split {split!r}; available {sorted(self.splits)}')
            ids = list(self.splits[split])
        if max_n is not None and len(ids) > max_n:
            ids = [ids[j] for j in np.unique(np.linspace(0, len(ids) - 1, max_n).round().astype(int))]
        return ids

    def designs(self, split: str) -> List[str]:
        return sorted({self.meta(i).get('design', '') for i in self.splits.get(split, [])})

    # guarded access
    def _guard(self, sid: str):
        if self.is_test(sid):
            from .protocol import require_frozen
            require_frozen(self.protocol_dir, what=f'reading test sample {sid!r}', dataset=self.name)

    def _load(self, sid: str) -> Sample:                                       # pragma: no cover - abstract
        raise NotImplementedError

    def get(self, sid: str) -> Optional[Sample]:
        """Sample by id, or None if it fails to load (recorded in self.errors). Test samples need FROZEN.json."""
        self._guard(sid)
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        try:
            s = self._load(sid)
        except ProtocolError:
            raise
        except Exception as exc:                                               # noqa: BLE001 - reported by E001
            self.errors[sid] = f'{type(exc).__name__}: {exc}'
            return None
        self._cache[sid] = s
        while len(self._cache) > self._cache_items:
            self._cache.popitem(last=False)
        return s

    def iter(self, split: str, max_n: Optional[int] = None) -> Iterable[Sample]:
        for sid in self.ids(split, max_n):
            s = self.get(sid)
            if s is not None:
                yield s


class ListDataset(MapDataset):
    """In-memory dataset (synthetic data, tests, fixtures)."""

    def __init__(self, samples: Sequence[Sample], name='memory', task='ir_drop', test_splits=('test',), **kw):
        splits: Dict[str, List[str]] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for s in samples:
            splits.setdefault(s.split, []).append(s.id)
            meta[s.id] = {**s.meta, 'split': s.split, 'design': s.design}
        super().__init__(name, task, splits, meta, **kw)
        self.test_splits = tuple(test_splits)
        self._samples = {s.id: s for s in samples}

    def _load(self, sid):
        return self._samples[sid]
