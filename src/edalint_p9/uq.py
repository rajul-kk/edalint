"""P9 uncertainty layer on top of the baseline zoo (protocol_p9.md section 5; design D7-D11).

* BootstrapEnsemble wraps any zoo Baseline factory: member m is fitted on a bootstrap resample of the training maps (seed + m) with its
  pixel-subsample seed set to m. predict_members(sample) -> (M, H, W).
* Intervals: ensemble-Gaussian mu +/- z sigma (uncalibrated); conformal risk control (CRC) with loss = fraction of miscovered pixels of a
  map, for the absolute score |y - mu| (constant half-width lambda) and the normalised score |y - mu| / (sigma + eps) (half-width
  lambda (sigma + eps)).
* CRC (Angelopoulos, Bates, Fisch, Lei, Schuster 2022): lambda = inf{l : n/(n+1) R_n(l) + B/(n+1) <= alpha}, B = 1, R_n(l) = mean over
  calibration maps of the fraction of pixels with score > l. Guarantee: E[miscovered fraction of a new exchangeable map] <= alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

NOMINALS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)        # interval calibration error grid (coverage levels 1 - alpha)
PRIMARY_ALPHA = 0.1
EPS_FRAC = 0.05                                   # eps = EPS_FRAC x mean sigma_ens over the calibration pixels
HIGH_Q = 0.9                                      # high-value pixels: label >= the map's 90th percentile


class Subset:
    """Lazy re-iterable view train[idx[0]], train[idx[1]], ... (a bootstrap resample loads a repeated map again, never holds all)."""

    def __init__(self, seq, idx):
        self.seq, self.idx = seq, list(idx)

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        if hasattr(self.seq, 'prefetch'):
            yield from self.seq.prefetch(self.idx)
            return
        for i in self.idx:
            yield self.seq[i]


class BootstrapEnsemble:
    def __init__(self, factory: Callable[[], object], m: int = 5, seed: int = 20261105):
        self.factory, self.m, self.seed = factory, m, seed
        self.members: List[object] = []
        self.boot_: List[List[int]] = []

    def fit(self, train: Sequence) -> 'BootstrapEnsemble':
        """train: an indexable sequence of Samples (may be lazy, e.g. edalint_p9.data.LazySamples); members see lazy bootstrap views."""
        self.members, self.boot_ = [], []
        for k in range(self.m):
            rng = np.random.default_rng(self.seed + k)
            idx = sorted(rng.integers(0, len(train), size=len(train)).tolist())
            b = self.factory()
            if hasattr(b, 'seed'):
                b.seed = k                                 # pixel-subsample seed of _PixelModel members
            if hasattr(b, 'kw') and isinstance(b.kw, dict) and 'random_state' in b.kw:
                b.kw['random_state'] = k
            b.fit(Subset(train, idx))
            self.members.append(b)
            self.boot_.append(idx)
        return self

    def predict_members(self, s) -> np.ndarray:
        return np.stack([np.asarray(b.predict(s), dtype=np.float64) for b in self.members])


def mean_std(P: np.ndarray):
    return P.mean(0), P.std(0, ddof=1) if len(P) > 1 else np.zeros(P.shape[1:])


# -- conformal risk control ---------------------------------------------------------------------------------------------------
@dataclass
class CalMap:
    """Sorted nonconformity scores of one calibration map (float32 to bound memory)."""
    scores: np.ndarray

    def miss(self, lam: float) -> float:
        n = len(self.scores)
        return float(n - np.searchsorted(self.scores, lam, side='right')) / n      # fraction with score > lam


def calmap(score_map: np.ndarray) -> CalMap:
    return CalMap(np.sort(np.asarray(score_map, dtype=np.float32).ravel()))


def crc_lambda(cal: Sequence[CalMap], alpha: float) -> float:
    """Smallest lambda (among the pooled calibration scores, plus 0) with n/(n+1) R_n(lambda) + 1/(n+1) <= alpha; +inf if infeasible
    (alpha < 1/(n+1)). R_n is non-increasing in lambda, so bisection over the sorted candidate set is exact."""
    n = len(cal)
    if n == 0 or alpha * (n + 1) - 1 < 0:
        return float('inf')
    target = (alpha * (n + 1) - 1) / n                      # need R_n(lambda) <= target

    def risk(l):
        return float(np.mean([c.miss(l) for c in cal]))
    if risk(0.0) <= target:
        return 0.0
    cand = np.unique(np.concatenate([c.scores[np.linspace(0, len(c.scores) - 1, min(len(c.scores), 4096)).astype(int)] for c in cal]))
    lo, hi = 0, len(cand) - 1
    if risk(float(cand[hi])) > target:
        return float(cand[hi])                              # unreachable for valid scores: risk(max score) = 0
    while lo < hi:                                          # coarse search on a subsample of candidates ...
        mid = (lo + hi) // 2
        if risk(float(cand[mid])) <= target:
            hi = mid
        else:
            lo = mid + 1
    upper = float(cand[lo])
    lower = float(cand[lo - 1]) if lo > 0 else 0.0
    fine = np.unique(np.concatenate([c.scores[(c.scores > lower) & (c.scores <= upper)] for c in cal] + [np.array([upper], np.float32)]))
    lo, hi = 0, len(fine) - 1                               # ... then exact search among all scores in the bracket
    while lo < hi:
        mid = (lo + hi) // 2
        if risk(float(fine[mid])) <= target:
            hi = mid
        else:
            lo = mid + 1
    return float(fine[lo])


def eps_from(sigmas: Sequence[np.ndarray]) -> float:
    m = float(np.mean([float(np.mean(s)) for s in sigmas])) if len(sigmas) else 0.0
    return max(EPS_FRAC * m, 1e-12)


# -- intervals and per-map metrics --------------------------------------------------------------------------------------------
def half_width(method: str, sigma: np.ndarray, lam: float, eps: float, alpha: float) -> np.ndarray:
    if method == 'crc_abs':
        return np.full(sigma.shape, lam)
    if method == 'crc_norm':
        return lam * (sigma + eps)
    if method == 'ens_gauss':
        return norm.ppf(1 - alpha / 2) * sigma
    raise KeyError(method)


def score_map(method: str, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float) -> np.ndarray:
    r = np.abs(y - mu)
    return r if method == 'crc_abs' else r / (sigma + eps)


def map_metrics(y: np.ndarray, mu: np.ndarray, hw: np.ndarray) -> Dict[str, float]:
    cov = np.abs(y - mu) <= hw
    hi = y >= np.quantile(y, HIGH_Q)
    return {'coverage': float(cov.mean()), 'width': float(2 * np.mean(hw)),
            'coverage_high': float(cov[hi].mean()) if hi.any() else float('nan'),
            'mae': float(np.mean(np.abs(y - mu)))}


def summarise_maps(rows: Sequence[Dict[str, float]], nominal: float) -> Dict[str, float]:
    if not rows:
        return {'n': 0}
    cov = np.array([r['coverage'] for r in rows])
    out = {'n': len(rows), 'coverage': float(cov.mean()), 'coverage_gap': float(cov.mean() - nominal),
           'width': float(np.mean([r['width'] for r in rows])),
           'coverage_high': float(np.nanmean([r['coverage_high'] for r in rows])) if any(np.isfinite(r['coverage_high']) for r in rows) else float('nan'),
           'frac_maps_below_nominal_minus_0.1': float(np.mean(cov < nominal - 0.1)),
           'coverage_p10': float(np.quantile(cov, 0.1)), 'mae': float(np.mean([r['mae'] for r in rows]))}
    return out


def calibration_error(cov_by_nominal: Dict[float, float]) -> float:
    """Interval calibration error: mean over nominal levels of |empirical coverage - nominal| (each level calibrated separately)."""
    return float(np.mean([abs(c - n) for n, c in cov_by_nominal.items()]))


class Calibrator:
    """Holds calibration maps for every (method, alpha) and returns lambda / eps. Fed with (y, mu, sigma) of calibration maps."""
    methods = ('crc_abs', 'crc_norm')

    def __init__(self):
        self.ys: List[tuple] = []

    def add(self, y, mu, sigma):
        self.ys.append((np.asarray(y, np.float32), np.asarray(mu, np.float32), np.asarray(sigma, np.float32)))

    def fit(self, alphas: Sequence[float]) -> Dict[str, dict]:
        eps = eps_from([s for _, _, s in self.ys])
        out = {'eps': eps, 'n_maps': len(self.ys), 'lambda': {}}
        for m in self.methods:
            cal = [calmap(score_map(m, y, mu, s, eps)) for y, mu, s in self.ys]
            out['lambda'][m] = {f'{a:g}': crc_lambda(cal, a) for a in alphas}
        return out
