"""P9 UQ layer: CRC lambda (exactness, guarantee, infeasibility), ensembles over zoo baselines, metrics."""
import numpy as np
import pytest

from edalint.baselines.zoo import PixelRidge
from edalint.core import Sample
from edalint_p9 import uq as U


def _brute_lambda(score_maps, alpha):
    n = len(score_maps)
    target = (alpha * (n + 1) - 1) / n
    cand = np.unique(np.concatenate([np.ravel(s) for s in score_maps] + [np.array([0.0])]))
    for l in cand:
        if np.mean([np.mean(np.ravel(s) > l) for s in score_maps]) <= target + 1e-12:
            return float(l)
    return float('inf')


@pytest.mark.parametrize('alpha', [0.1, 0.2, 0.5])
def test_crc_lambda_equals_brute_force(alpha):
    rng = np.random.default_rng(0)
    maps = [rng.exponential(size=(9, 9)).astype(np.float32) * rng.uniform(0.5, 2) for _ in range(12)]
    got = U.crc_lambda([U.calmap(m) for m in maps], alpha)
    assert got == pytest.approx(_brute_lambda(maps, alpha), rel=1e-6)


def test_crc_infeasible_alpha_gives_infinite_lambda():
    maps = [np.ones((4, 4))] * 5                                      # 1/(n+1) = 0.167 > alpha
    assert U.crc_lambda([U.calmap(m) for m in maps], 0.1) == float('inf')


def test_crc_guarantee_holds_on_exchangeable_maps():
    """Mean miscovered fraction of a fresh map <= alpha (averaged over 300 calibration draws) for map-dependent noise scales."""
    rng = np.random.default_rng(1)
    alpha, miss = 0.1, []
    for _ in range(300):
        maps = [np.abs(rng.normal(size=(16, 16))) * rng.lognormal(0, 0.5) for _ in range(21)]
        lam = U.crc_lambda([U.calmap(m) for m in maps[:20]], alpha)
        miss.append(np.mean(maps[20] > lam))
    assert np.mean(miss) <= alpha + 0.01
    assert np.mean(miss) >= alpha - 0.05                              # and not absurdly conservative


def test_intervals_and_map_metrics():
    y = np.arange(100, dtype=float).reshape(10, 10)
    mu = y + 1.0
    m = U.map_metrics(y, mu, U.half_width('crc_abs', np.zeros_like(y), 1.0, 0.1, 0.1))
    assert m['coverage'] == 1.0 and m['width'] == 2.0 and m['mae'] == 1.0 and m['coverage_high'] == 1.0
    m = U.map_metrics(y, mu, U.half_width('crc_norm', np.full_like(y, 1.0), 0.5, 0.0, 0.1))
    assert m['coverage'] == 0.0
    assert U.half_width('ens_gauss', np.ones((2, 2)), 0, 0, 0.1)[0, 0] == pytest.approx(1.6448536, rel=1e-6)
    assert U.calibration_error({0.5: 0.4, 0.9: 0.95}) == pytest.approx(0.075)
    s = U.summarise_maps([{'coverage': 0.9, 'width': 1, 'coverage_high': 0.5, 'mae': 0.1},
                          {'coverage': 0.7, 'width': 3, 'coverage_high': np.nan, 'mae': 0.3}], 0.9)
    assert s['coverage'] == pytest.approx(0.8) and s['coverage_gap'] == pytest.approx(-0.1) and s['coverage_high'] == 0.5
    assert s['frac_maps_below_nominal_minus_0.1'] == 0.5


def _syn(n, rng, shift=0.0):
    out = []
    for i in range(n):
        p = rng.random((16, 16)) + shift
        out.append(Sample(f's{i}', 'train', 'd', {'p': p}, 3 * p + 0.1 * rng.normal(size=p.shape)))
    return out


def test_bootstrap_ensemble_of_a_zoo_baseline_and_calibrated_coverage():
    rng = np.random.default_rng(2)
    tr, cal, te = _syn(30, rng), _syn(40, rng), _syn(40, rng)
    ens = U.BootstrapEnsemble(lambda: PixelRidge(alpha=0.01), m=4, seed=7).fit(tr)
    assert len(ens.members) == 4 and len({tuple(b) for b in ens.boot_}) == 4
    P = ens.predict_members(te[0])
    assert P.shape == (4, 16, 16) and np.std(P, 0).mean() > 0
    c = U.Calibrator()
    for s in cal:
        mu, sd = U.mean_std(ens.predict_members(s))
        c.add(s.label, mu, sd)
    fit = c.fit([0.1])
    for meth in ('crc_abs', 'crc_norm'):
        lam = fit['lambda'][meth]['0.1']
        rows = []
        for s in te:
            mu, sd = U.mean_std(ens.predict_members(s))
            rows.append(U.map_metrics(s.label, mu, U.half_width(meth, sd, lam, fit['eps'], 0.1)))
        cov = U.summarise_maps(rows, 0.9)['coverage']
        assert 0.85 <= cov <= 0.97, (meth, cov)


def test_subset_is_lazy_and_repeats():
    calls = []

    class Seq:
        def __len__(self):
            return 3

        def __getitem__(self, i):
            calls.append(i)
            return i
    s = U.Subset(Seq(), [2, 2, 0])
    assert calls == [] and list(s) == [2, 2, 0] and calls == [2, 2, 0]
