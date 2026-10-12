"""The baseline zoo. Order = the ladder in PLAN.md: zero -> mean map -> scalar -> ridge -> kNN -> eff-dist -> Green -> GBM -> exact solve."""
from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np

from ..core import Sample, area_resize, box_mean
from .base import Baseline, pixel_features, register, resize_to, sample_pixels

CANON = 128                       # canonical grid for stored mean / neighbour labels


@register('zero')
class Zero(Baseline):
    """Predict 0 everywhere. For ICCAD this has MAE = mean IR drop and no hotspot (F1 = 0 under the official rule)."""

    def predict(self, s):
        return np.zeros(s.shape)


@register('mean_map')
class MeanMap(Baseline):
    """Mean of the training labels on a canonical 128 x 128 grid, resized to each sample (area / bilinear)."""

    def fit(self, train):
        acc, n = np.zeros((CANON, CANON)), 0
        for s in train:
            acc += area_resize(s.label, (CANON, CANON))
            n += 1
        self.mean_ = acc / max(n, 1)
        return self

    def predict(self, s):
        return resize_to(self.mean_, s.shape)


@register('scalar')
class ScalarMean(Baseline):
    """Constant map equal to the mean of all training label pixels (pooled over samples)."""

    def fit(self, train):
        tot, cnt = 0.0, 0
        for s in train:
            tot += float(s.label.sum())
            cnt += s.label.size
        self.value_ = tot / max(cnt, 1)
        return self

    def predict(self, s):
        return np.full(s.shape, self.value_)


class _PixelModel(Baseline):
    max_rows = 1_500_000
    boxes = (3, 9)
    seed = 0

    def _collect(self, train):
        train = list(train) if not hasattr(train, '__len__') else train
        n = max(len(train), 1)
        per = max(200, self.max_rows // n)
        rng = np.random.default_rng(self.seed)
        X, y = [], []
        for s in train:
            f = pixel_features(s, self.boxes)
            idx = sample_pixels(rng, f.shape[0], per)
            X.append(f[idx].astype(np.float32))
            y.append(s.label.ravel()[idx])
        X, y = np.concatenate(X), np.concatenate(y)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        return (X - self.mu_) / self.sd_, y


@register('ridge')
class PixelRidge(_PixelModel):
    """Per-pixel ridge on the input channels plus 3x3 and 9x9 box means of each channel (features standardised on the train
    pixel subsample; closed-form normal equations; alpha is in units of rows)."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def fit(self, train):
        X, y = self._collect(train)
        X = np.c_[X.astype(np.float64), np.ones(len(X))]
        A = X.T @ X
        A[np.diag_indices(A.shape[0] - 1)] += self.alpha * len(X)
        self.w_ = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, s):
        f = (pixel_features(s, self.boxes) - self.mu_) / self.sd_
        return (f @ self.w_[:-1] + self.w_[-1]).reshape(s.shape)


@register('gbm')
class PixelGBM(_PixelModel):
    """sklearn HistGradientBoostingRegressor on the same per-pixel features as ridge; rows subsampled to fit in RAM."""
    needs_sklearn = True
    max_rows = 300_000

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.1, max_leaf_nodes: Optional[int] = None,
                 max_depth: Optional[int] = 6):
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                       random_state=0)

    def fit(self, train):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, y = self._collect(train)
        self.m_ = HistGradientBoostingRegressor(**self.kw).fit(X, y)
        return self

    def predict(self, s):
        f = ((pixel_features(s, self.boxes) - self.mu_) / self.sd_).astype(np.float32)
        out = np.empty(f.shape[0])
        for a in range(0, f.shape[0], 200_000):
            out[a:a + 200_000] = self.m_.predict(f[a:a + 200_000])
        return out.reshape(s.shape)


@register('knn')
class KNNMap(Baseline):
    """k nearest training cases in a feature space of the inputs downsampled to 16 x 16 (channels standardised over the training
    set); the prediction is the mean of their labels resized to the sample's grid. Neighbours may be of another size."""

    def __init__(self, k: int = 3, grid: int = 16):
        self.k, self.grid = k, grid

    def _feat(self, s):
        x = s.stack()
        return ((area_resize(x, (self.grid, self.grid)).reshape(x.shape[0], -1) - self.mu_[:, None]) / self.sd_[:, None]).ravel()

    def fit(self, train):
        raw, labs = [], []
        for s in train:
            x = s.stack()
            raw.append(area_resize(x, (self.grid, self.grid)).reshape(x.shape[0], -1))
            labs.append(area_resize(s.label, (CANON, CANON)).astype(np.float32))
        R = np.stack(raw)                                    # (n, C, g*g)
        self.mu_, self.sd_ = R.mean((0, 2)), R.std((0, 2))
        self.sd_[self.sd_ == 0] = 1.0
        self.F_ = ((R - self.mu_[None, :, None]) / self.sd_[None, :, None]).reshape(len(R), -1)
        self.L_ = np.stack(labs)
        return self

    def predict(self, s):
        d = ((self.F_ - self._feat(s)[None]) ** 2).sum(1)
        nn = np.argsort(d)[:self.k]
        return resize_to(self.L_[nn].mean(0).astype(np.float64), s.shape)


@register('effdist')
class EffDistRegression(Baseline):
    """1D regression of the label on t = eff_dist x (window-mean of the primary input), pooled over a train pixel subsample;
    polynomial of degree `degree` (1 = linear with intercept). ICCAD only (needs the eff_dist map)."""
    needs_channels = ('eff_dist',)
    tasks = ('ir_drop',)

    def __init__(self, window: int = 9, degree: int = 1, pixels_per_case: int = 5000, primary: str = 'current'):
        self.window, self.degree, self.ppc, self.primary = window, degree, pixels_per_case, primary

    def _t(self, s):
        return (s.primary('eff_dist') * box_mean(s.primary(self.primary), self.window)).ravel()

    def applicable(self, ds, sample=None):
        ok, why = super().applicable(ds, sample)
        if ok and sample is None and ds.name != 'iccad23':
            return False, 'needs an eff_dist input map (ICCAD only)'
        return ok, why

    def fit(self, train):
        rng = np.random.default_rng(0)
        T, Y = [], []
        for s in train:
            t = self._t(s)
            idx = sample_pixels(rng, t.size, self.ppc)
            T.append(t[idx])
            Y.append(s.label.ravel()[idx])
        T, Y = np.concatenate(T), np.concatenate(Y)
        self.scale_ = T.std() or 1.0
        self.coef_ = np.polyfit(T / self.scale_, Y, self.degree)
        return self

    def predict(self, s):
        return np.polyval(self.coef_, self._t(s) / self.scale_).reshape(s.shape)


@register('effdist_deg2')
class EffDistQuadratic(EffDistRegression):
    def __init__(self):
        super().__init__(degree=2)


_EDGES = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192], float)
# ring-bin edge sets for the Green baseline (P4 development grid; the frozen choice is the class default)
GREEN_EDGES = {
    'coarse': np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 192], float),
    'medium': _EDGES,
    'fine': np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192], float),
}


@register('green')
class GreenSuperposition(Baseline):
    """Green's-function superposition: prediction = (primary input * K)(x) + b with ONE radial kernel K(r), piecewise constant on
    ring bins [edges[i], edges[i+1]) pixels, calibrated on the train pixel subsample by non-negative least squares. Zero padding at
    the borders. Linear and shift-invariant by construction; knows nothing about the PDN beyond what K absorbs on average."""
    tasks = ('ir_drop',)

    def __init__(self, edges=_EDGES, pixels_per_case: int = 4000, primary: Optional[str] = None):
        self.edges, self.ppc, self.primary = np.asarray(edges, float), pixels_per_case, primary

    def _prim(self, s):
        for n in ([self.primary] if self.primary else []) + ['current', 'power_all']:
            if n in s.inputs:
                return s.primary(n)
        return s.primary(next(iter(s.inputs)))

    def _ring_kernels(self):
        R = int(np.ceil(self.edges[-1]))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        r = np.hypot(xx, yy)
        return [((r >= lo) & (r < hi)).astype(np.float64) for lo, hi in zip(self.edges[:-1], self.edges[1:])]

    def fit(self, train):
        from scipy.optimize import nnls
        from scipy.signal import fftconvolve
        kernels = self._ring_kernels()
        rng = np.random.default_rng(0)
        A, y = [], []
        for s in train:
            c = self._prim(s)
            idx = sample_pixels(rng, c.size, self.ppc)
            cols = [fftconvolve(c, k, mode='same').ravel()[idx] for k in kernels]
            A.append(np.stack(cols + [np.ones(len(idx)) * c.mean()], 1))
            y.append(s.label.ravel()[idx])
        A, y = np.concatenate(A), np.concatenate(y)
        self.scale_ = np.abs(A).max(0)
        self.scale_[self.scale_ == 0] = 1.0
        w, _ = nnls(A / self.scale_, y)
        self.w_ = w / self.scale_
        self.kernel_ = sum(wi * k for wi, k in zip(self.w_[:-1], kernels))
        self.bias_ = float(self.w_[-1])
        return self

    def predict(self, s):
        from scipy.signal import fftconvolve
        c = self._prim(s)
        return fftconvolve(c, self.kernel_, mode='same') + self.bias_ * c.mean()


@register('spice')
class SpiceExact(Baseline):
    """EXACT sparse solve of the SPICE PDN (ICCAD only): runtime reference, NOT a skill claim (the labels come from a solver of the
    same equations). No fit. Rasterisation follows the label generator (cubic inside the node hull, nearest outside)."""
    needs_netlist = True
    tasks = ('ir_drop',)

    def __init__(self, method: str = 'direct', interp: str = 'cubic'):
        self.method, self.interp = method, interp
        self.last_info = {}

    def applicable(self, ds, sample=None):
        ok, why = super().applicable(ds, sample)
        if ok and sample is None and ds.name != 'iccad23':
            return False, 'needs a SPICE netlist (ICCAD only)'
        return ok, why

    def predict(self, s):
        from ..spice import exact_ir_drop
        t0 = time.time()
        grid = s.meta.get('resized_from', s.shape)
        m, info = exact_ir_drop(s.meta['netlist'], tuple(grid), method=self.method, interp=self.interp)
        self.last_info = {**info, 'seconds_predict': time.time() - t0}
        return resize_to(m, s.shape) if tuple(grid) != s.shape else m
