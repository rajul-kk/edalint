"""Metrics: ICCAD'23 Problem C official definitions and the CircuitNet definitions. Every variant is named, nothing is implicit.

ICCAD'23 Problem C (contest description section 4, docs/format_notes.md 1.5, docs/published_numbers.md 1.1)
  mae_mV(pred, gold)          per-pixel mean |pred - gold|, in mV (maps in volts)
  f1_golden_max(...)          OFFICIAL RULE as read from the worked example: hotspot = value > frac * max(GOLDEN map), the same
                              absolute threshold is applied to the prediction. F1 = 2TP / (2TP + FP + FN).
  f1_top_fraction(...)        the "top 10 % of IR drop" reading of the same sentence (a pixel-fraction rule). AttUNet uses it.
  f1_pred_max(...)            UNOFFICIAL: each map thresholded at frac of its own max. Only for sensitivity analysis.

  Inconsistency in the contest PDF (version 4): the worked example prints the golden matrix [[4.63, 5.23], [5.93, 0.04]] mV with
  the threshold 0.9 * 5.93 = 5.337 mV, but its "actual hotspot matrix" is [[0, 1], [1, 0]], i.e. 5.23 mV counts as a hotspot although it
  is below 5.337 mV; with the stated rule the golden hotspot matrix is [[0, 0], [1, 0]] (one hotspot). The example's F1 of 0.67
  (TP=1, TN=2, FP=0, FN=1) follows from the printed matrix, not from the stated rule (which would give 1.0 for the same prediction).
  The official scorer is not in the repository; strict '>' versus '>=' is undetermined (`strict` argument, default '>').

CircuitNet (utils/metrics.py of github.com/circuitnet/CircuitNet, docs/published_numbers.md 2.1)
  nrms         skimage normalized_root_mse(gold, pred, 'min-max'): RMSE / (max(gold) - min(gold))
  ssim         own 11x11 Gaussian (sigma 1.5) SSIM, 'valid' region, constants (0.01 L)^2 and (0.03 L)^2
  nmae         mean |e| / (max - min of gold) (PDNNet's NMAE, per map; pixel or sample averaging is ambiguous in the paper)
  pearson / spearman / kendall   correlation over pixels
  roc_auc / pr_auc               rank-based AUC and average precision against gold >= threshold
  official_threshold_metrics     the repository's 200-threshold per-sample TPR/FPR/precision averaging (with its skip rules)
  official_running_mean          the test.py accumulation that skips metric values exactly equal to 1 but divides by the full count
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid as _trapz
from scipy import stats

# -- ICCAD'23 -------------------------------------------------------------------------------------------------------------
def mae_mV(pred: np.ndarray, gold: np.ndarray) -> float:
    """Per-pixel mean absolute error in millivolts (inputs in volts)."""
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(gold, np.float64))) * 1e3)


def confusion(pred_mask: np.ndarray, gold_mask: np.ndarray) -> Tuple[int, int, int, int]:
    p, g = np.asarray(pred_mask, bool), np.asarray(gold_mask, bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    return tp, fp, fn, tn


def f1_from_confusion(tp: int, fp: int, fn: int, both_empty: float = float('nan')) -> float:
    """F1 = 2TP / (2TP + FP + FN). With no positives in either map (0/0) returns `both_empty` (default NaN).
    A prediction with no hotspot and a golden map with some gives 0 (the official xlsx records F1 = 0 for those cases)."""
    den = 2 * tp + fp + fn
    return both_empty if den == 0 else 2.0 * tp / den


def hotspot_mask(x: np.ndarray, threshold: float, strict: bool = True) -> np.ndarray:
    return x > threshold if strict else x >= threshold


def f1_golden_max(pred: np.ndarray, gold: np.ndarray, frac: float = 0.9, strict: bool = True,
                  both_empty: float = float('nan')) -> float:
    """OFFICIAL (as read from the worked example): threshold = frac * max(gold), applied to both maps."""
    thr = frac * float(np.max(gold))
    tp, fp, fn, _ = confusion(hotspot_mask(pred, thr, strict), hotspot_mask(gold, thr, strict))
    return f1_from_confusion(tp, fp, fn, both_empty)


def f1_top_fraction(pred: np.ndarray, gold: np.ndarray, top: float = 0.10, pred_threshold: str = 'golden',
                    both_empty: float = float('nan')) -> float:
    """'Top 10 % of IR drop' variant: gold hotspot = pixels >= the (1 - top) quantile of gold. The prediction is cut at the golden
    quantile value (pred_threshold='golden') or at its own (1 - top) quantile ('own')."""
    tg = np.quantile(gold, 1.0 - top)
    tp_ = tg if pred_threshold == 'golden' else np.quantile(pred, 1.0 - top)
    tp, fp, fn, _ = confusion(pred >= tp_, gold >= tg)
    return f1_from_confusion(tp, fp, fn, both_empty)


def f1_pred_max(pred: np.ndarray, gold: np.ndarray, frac: float = 0.9, both_empty: float = float('nan')) -> float:
    """UNOFFICIAL: each map is cut at frac of its own maximum."""
    tp, fp, fn, _ = confusion(pred > frac * np.max(pred), gold > frac * np.max(gold))
    return f1_from_confusion(tp, fp, fn, both_empty)


def iccad_case_metrics(pred: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    return {'mae_mV': mae_mV(pred, gold), 'f1_golden_max': f1_golden_max(pred, gold),
            'f1_golden_max_ge': f1_golden_max(pred, gold, strict=False),
            'f1_top10pct': f1_top_fraction(pred, gold), 'f1_pred_max': f1_pred_max(pred, gold)}


# -- CircuitNet -------------------------------------------------------------------------------------------------------------
def nrms(pred: np.ndarray, gold: np.ndarray, official_inf: Optional[float] = None) -> float:
    """RMSE / (max(gold) - min(gold)) (skimage `normalized_root_mse(gold, pred, normalization='min-max')`). A constant gold gives
    inf (or 0/0 = nan); the official wrapper replaces inf by 0.05: pass official_inf=0.05 to reproduce that."""
    g = np.asarray(gold, np.float64)
    p = np.asarray(pred, np.float64)
    rng = float(g.max() - g.min())
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        v = rmse / rng if rng > 0 else (float('inf') if rmse > 0 else float('nan'))
    if official_inf is not None and math.isinf(v):
        return official_inf
    return v


def nmae(pred: np.ndarray, gold: np.ndarray) -> float:
    g = np.asarray(gold, np.float64)
    rng = float(g.max() - g.min())
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - g)) / rng) if rng > 0 else float('nan')


def _gauss_win(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def ssim(pred: np.ndarray, gold: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM with an 11x11 Gaussian window (sigma 1.5) over the fully-overlapping region, as the official `_ssim`
    (which scales maps to 0..255 and uses L = 255; data_range=1 on 0..1 maps is the same up to the uint8 rounding)."""
    from scipy.ndimage import convolve1d
    a, b = np.asarray(pred, np.float64), np.asarray(gold, np.float64)
    if min(a.shape) < 11:
        return float('nan')
    k = _gauss_win()

    def filt(x):
        return convolve1d(convolve1d(x, k, axis=0, mode='constant'), k, axis=1, mode='constant')[5:-5, 5:-5]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu1, mu2 = filt(a), filt(b)
    s11, s22, s12 = filt(a * a) - mu1 ** 2, filt(b * b) - mu2 ** 2, filt(a * b) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def to_u8(x: np.ndarray) -> np.ndarray:
    """The authors' `tensor2img` (utils/metrics.py 339-368), applied by `input_converter` to the arguments of nrms, ssim and psnr: clamp to
    [0, 1], scale by 255, round (half to even), cast to uint8. Returned as float64 so it can feed the metric code directly."""
    a = np.asarray(x, np.float64)
    return (np.clip(a, 0.0, 1.0) * 255.0).round()


def ssim_official(pred: np.ndarray, gold: np.ndarray) -> float:
    """The authors' SSIM as called by their test.py: both maps are clamped to [0, 1] and quantised to uint8 first (`input_converter`),
    then `_ssim` with L = 255 (utils/metrics.py 72-107). `ssim()` on the float maps is the secondary, un-quantised variant."""
    return ssim(to_u8(pred), to_u8(gold), data_range=255.0)


def nrms_official(pred: np.ndarray, gold: np.ndarray, official_inf: Optional[float] = None) -> float:
    """The authors' NRMS as called by their test.py: on the clamped, uint8-quantised maps (utils/metrics.py 110-124; inf replaced by 0.05 there)."""
    return nrms(to_u8(pred), to_u8(gold), official_inf=official_inf)


def pearson_r(pred, gold) -> float:
    p, g = np.ravel(pred).astype(np.float64), np.ravel(gold).astype(np.float64)
    if np.ptp(p) == 0 or np.ptp(g) == 0:
        return float('nan')
    return float(np.corrcoef(p, g)[0, 1])


def _subsample(p, g, max_n, seed=0):
    p, g = np.ravel(np.asarray(p, np.float64)), np.ravel(np.asarray(g, np.float64))
    if max_n and p.size > max_n:
        idx = np.random.default_rng(seed).choice(p.size, max_n, replace=False)
        return p[idx], g[idx]
    return p, g


def spearman_r(pred, gold, max_n: Optional[int] = None) -> float:
    p, g = _subsample(pred, gold, max_n)
    if np.ptp(p) == 0 or np.ptp(g) == 0:
        return float('nan')
    return float(stats.spearmanr(p, g)[0])


def kendall_tau(pred, gold, max_n: Optional[int] = 20000) -> float:
    """Kendall tau-b. O(n log n) in scipy but memory-heavy for 65k pixels x many maps: pixels are subsampled to max_n (seeded)."""
    p, g = _subsample(pred, gold, max_n)
    if np.ptp(p) == 0 or np.ptp(g) == 0:
        return float('nan')
    return float(stats.kendalltau(p, g)[0])


def roc_auc(pred, gold, threshold: float) -> float:
    """Rank-based ROC AUC of `pred` against gold >= threshold. NaN if one class is empty."""
    p = np.ravel(pred).astype(np.float64)
    y = np.ravel(gold) >= threshold
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = stats.rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(pred, gold, threshold: float) -> float:
    """Average precision (step-wise area under the precision-recall curve) against gold >= threshold."""
    p = np.ravel(pred).astype(np.float64)
    y = np.ravel(gold) >= threshold
    if y.sum() == 0:
        return float('nan')
    order = np.argsort(-p, kind='mergesort')
    ys = y[order]
    ps = p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    last = np.r_[ps[1:] != ps[:-1], True]                # one point per distinct score
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def circuitnet_case_metrics(pred, gold, task: str, threshold: Optional[float] = None) -> Dict[str, float]:
    out = {'nrms': nrms(pred, gold), 'nmae': nmae(pred, gold), 'ssim': ssim(pred, gold), 'pearson': pearson_r(pred, gold),
           'spearman': spearman_r(pred, gold, max_n=20000), 'kendall': kendall_tau(pred, gold)}
    if threshold is not None:
        out['roc_auc'] = roc_auc(pred, gold, threshold)
        out['pr_auc'] = pr_auc(pred, gold, threshold)
    return out


def official_threshold_metrics(preds: Sequence[np.ndarray], golds: Sequence[np.ndarray], label_threshold: float,
                               n_thresholds: int = 200) -> Dict[str, float]:
    """The repository's ROC / PRC procedure (utils/metrics.py calculated_score + calculate_all): thresholds linspace(0, 1, 200) on
    the prediction, label cut at `label_threshold`; for each threshold the per-sample TPR, FPR and precision are AVERAGED over the
    samples that are not skipped (a sample is skipped when fp == tn == 0, or tp == fn == 0, or tp == fp == 0). AUCs are trapezoid
    areas of the averaged curves. The skip rules make the curve depend on which samples survive at each threshold."""
    ths = np.linspace(0, 1, n_thresholds)
    tpr_l, fpr_l, pre_l = [], [], []
    for t in ths:
        tprs, fprs, pres = [], [], []
        for p, g in zip(preds, golds):
            gm, pm = np.ravel(g) >= label_threshold, np.ravel(p) >= t
            tp, fp = int((pm & gm).sum()), int((pm & ~gm).sum())
            fn, tn = int((~pm & gm).sum()), int((~pm & ~gm).sum())
            if (fp == 0 and tn == 0) or (tp == 0 and fn == 0) or (tp == 0 and fp == 0):
                continue
            tprs.append(tp / (tp + fn))
            fprs.append(fp / (fp + tn))
            pres.append(tp / (tp + fp))
        if tprs:
            tpr_l.append(np.mean(tprs))
            fpr_l.append(np.mean(fprs))
            pre_l.append(np.mean(pres))
    if len(tpr_l) < 2:
        return {'roc_auc': float('nan'), 'pr_auc': float('nan'), 'n_points': len(tpr_l)}
    fpr, tpr, pre = np.array(fpr_l), np.array(tpr_l), np.array(pre_l)
    o = np.argsort(fpr)
    roc = float(_trapz(tpr[o], fpr[o]))
    o2 = np.argsort(tpr)
    prc = float(_trapz(pre[o2], tpr[o2]))
    return {'roc_auc': roc, 'pr_auc': prc, 'n_points': len(tpr_l)}


def official_running_mean(values: Iterable[float], n_total: Optional[int] = None) -> float:
    """test.py: `if not metric(...) == 1: running_sum += metric`, then divides by the FULL dataset length. A sample whose metric is
    exactly 1 therefore contributes 0, not 1 (a perfect SSIM is counted as 0). Returns the value test.py would print."""
    vals = list(values)
    n = n_total if n_total is not None else len(vals)
    return float(sum(v for v in vals if not v == 1) / n) if n else float('nan')
