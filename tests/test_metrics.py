"""Metric unit tests against hand-computed examples."""
import math

import numpy as np
import pytest

from edalint import metrics as M
from edalint.core import area_resize


def test_mae_mV_contest_example():
    # contest description section 4a: |diff| entries 0.2, 0.6, 0, 1.0 mV -> MAE = 0.45 mV
    gold = np.zeros((2, 2))
    pred = np.array([[0.2e-3, 0.6e-3], [0.0, 1.0e-3]])
    assert M.mae_mV(pred, gold) == pytest.approx(0.45)


def test_f1_contest_worked_example_is_inconsistent():
    gold = np.array([[4.63, 5.23], [5.93, 0.04]]) * 1e-3
    thr = 0.9 * gold.max()
    assert thr == pytest.approx(5.337e-3)
    # stated rule: only 5.93 mV is a hotspot
    assert M.hotspot_mask(gold, thr).tolist() == [[False, False], [True, False]]
    # the PDF's printed "actual hotspot matrix" [[0,1],[1,0]] gives TP=1, TN=2, FP=0, FN=1 -> F1 0.67
    printed_gold = np.array([[0, 1], [1, 0]], bool)
    pred = np.array([[0, 0], [1, 0]], bool)
    tp, fp, fn, tn = M.confusion(pred, printed_gold)
    assert (tp, fp, fn, tn) == (1, 0, 1, 2)
    assert M.f1_from_confusion(tp, fp, fn) == pytest.approx(2 / 3, abs=1e-3)
    # with the stated rule the same prediction scores 1.0
    assert M.f1_golden_max(np.where(pred, 5.93e-3, 0.0), gold) == pytest.approx(1.0)


def test_f1_golden_max_by_hand():
    gold = np.array([[10.0, 9.5, 5.0], [9.1, 1.0, 0.0]])       # hotspots: > 9.0 -> 10, 9.5, 9.1
    pred = np.array([[9.5, 9.2, 9.6], [1.0, 0.0, 0.0]])        # predicted > 9.0: 9.5, 9.2, 9.6
    # TP = 2 (10, 9.5), FP = 1 (5.0 predicted 9.6), FN = 1 (9.1 predicted 1.0)
    assert M.f1_golden_max(pred, gold) == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_f1_strict_vs_non_strict_threshold():
    gold = np.array([[10.0, 9.0], [0.0, 0.0]])                 # 9.0 == 0.9 * 10 exactly
    assert M.f1_golden_max(gold, gold, strict=True) == 1.0
    pred = np.array([[10.0, 0.0], [0.0, 0.0]])
    assert M.f1_golden_max(pred, gold, strict=True) == 1.0     # 9.0 is not a hotspot under '>'
    assert M.f1_golden_max(pred, gold, strict=False) == pytest.approx(2 / 3)


def test_f1_empty_prediction_is_zero_and_both_empty_is_configurable():
    gold = np.array([[10.0, 0.0], [0.0, 0.0]])
    assert M.f1_golden_max(np.zeros_like(gold), gold) == 0.0
    z = np.zeros((2, 2))
    assert math.isnan(M.f1_golden_max(z, z))
    assert M.f1_golden_max(z, z, both_empty=1.0) == 1.0


def test_f1_top_fraction_and_pred_max_variants():
    gold = np.arange(100, dtype=float).reshape(10, 10)
    assert M.f1_top_fraction(gold, gold, top=0.1) == 1.0
    assert M.f1_top_fraction(gold * 0.5, gold, top=0.1, pred_threshold='golden') == 0.0   # half-scale never reaches the golden cut
    assert M.f1_top_fraction(gold * 0.5, gold, top=0.1, pred_threshold='own') == 1.0
    assert M.f1_pred_max(gold * 0.5, gold) == 1.0                                          # unofficial: own max
    assert M.f1_golden_max(gold * 0.5, gold) == 0.0


def test_nrms_nmae_by_hand():
    gold = np.array([[0.0, 1.0], [2.0, 4.0]])                  # range 4
    pred = np.array([[0.0, 1.0], [2.0, 2.0]])                  # one error of 2
    assert M.nrms(pred, gold) == pytest.approx(math.sqrt(4 / 4) / 4)
    assert M.nmae(pred, gold) == pytest.approx((2 / 4) / 4)
    const = np.ones((2, 2))
    assert math.isinf(M.nrms(const * 2, const))
    assert M.nrms(const * 2, const, official_inf=0.05) == 0.05


def test_nrms_matches_skimage():
    sk = pytest.importorskip('skimage.metrics')
    rng = np.random.default_rng(0)
    g, p = rng.random((30, 30)), rng.random((30, 30))
    assert M.nrms(p, g) == pytest.approx(sk.normalized_root_mse(g, p, normalization='min-max'))


def test_ssim_properties_and_value():
    rng = np.random.default_rng(1)
    a = rng.random((32, 32))
    assert M.ssim(a, a) == pytest.approx(1.0)
    assert M.ssim(a, 1 - a) < 0.3
    # constant maps: ssim = (2 mu1 mu2 + c1) / (mu1^2 + mu2^2 + c1)
    x, y = np.full((16, 16), 0.2), np.full((16, 16), 0.6)
    c1 = 0.01 ** 2
    assert M.ssim(x, y) == pytest.approx((2 * 0.2 * 0.6 + c1) / (0.04 + 0.36 + c1), rel=1e-9)


def test_ssim_close_to_skimage_gaussian():
    sk = pytest.importorskip('skimage.metrics')
    rng = np.random.default_rng(2)
    a = rng.random((40, 40))
    b = np.clip(a + 0.1 * rng.standard_normal((40, 40)), 0, 1)
    ref = sk.structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    assert M.ssim(b, a) == pytest.approx(ref, abs=0.03)       # skimage truncates the window at 3.5 sigma: 11 px, same window


def test_correlations_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.pearson_r(x, 2 * x + 1) == pytest.approx(1.0)
    assert M.spearman_r(x, x ** 3) == pytest.approx(1.0)
    assert M.kendall_tau(x, -x) == pytest.approx(-1.0)
    assert M.kendall_tau(np.array([1, 2, 3, 4.0]), np.array([1, 3, 2, 4.0])) == pytest.approx((5 - 1) / 6)
    assert math.isnan(M.pearson_r(np.ones(4), x))


def test_roc_and_pr_auc_by_hand():
    gold = np.array([0, 0, 1, 1])
    pred = np.array([0.1, 0.4, 0.35, 0.8])                      # classic example: ROC AUC 0.75, AP 0.8333
    assert M.roc_auc(pred, gold, 0.5) == pytest.approx(0.75)
    assert M.pr_auc(pred, gold, 0.5) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3), abs=1e-9)
    assert M.roc_auc(pred, np.zeros(4), 0.5) != M.roc_auc(pred, np.zeros(4), 0.5)   # one class: NaN


def test_roc_pr_match_sklearn():
    sk = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(3)
    g = rng.random(500)
    p = g + 0.3 * rng.standard_normal(500)
    y = g >= 0.7
    assert M.roc_auc(p, g, 0.7) == pytest.approx(sk.roc_auc_score(y, p))
    assert M.pr_auc(p, g, 0.7) == pytest.approx(sk.average_precision_score(y, p))


def test_official_running_mean_counts_a_perfect_metric_as_zero():
    assert M.official_running_mean([1.0, 0.5, 0.5]) == pytest.approx(1.0 / 3)
    assert np.mean([1.0, 0.5, 0.5]) == pytest.approx(2.0 / 3)


def test_official_threshold_metrics_skips_degenerate_samples():
    g1 = np.zeros((4, 4))
    g1[0, :2] = 1.0
    p1 = g1.copy()
    out = M.official_threshold_metrics([p1], [g1], 0.5, n_thresholds=11)
    assert out['n_points'] >= 2 and out['roc_auc'] == pytest.approx(1.0, abs=0.11)
    # a golden map without positives is skipped at every threshold
    z = np.zeros((4, 4))
    assert M.official_threshold_metrics([z], [z], 0.5, n_thresholds=11)['n_points'] == 0


def test_area_resize_is_mean_preserving_and_matches_cv2():
    rng = np.random.default_rng(4)
    a = rng.random((37, 53))
    r = area_resize(a, (16, 16))
    assert r.mean() == pytest.approx(a.mean())
    cv2 = pytest.importorskip('cv2')
    ref = cv2.resize(a, (16, 16), interpolation=cv2.INTER_AREA)
    assert np.allclose(r, ref, atol=1e-9)
