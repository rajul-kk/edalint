import numpy as np
import pytest

from edalint.baselines import BASELINES, evaluate, run_zoo, stream, summarise
from edalint.core import ListDataset, Sample
from edalint.datasets import ICCAD23Dataset
from edalint.datasets.synthetic import make_synthetic
from fixtures import make_iccad_tree


@pytest.fixture(scope='module')
def ds():
    return make_synthetic(n_train_designs=8, per_design=2, size=40, guard_test=False)


def mae(b, ds, split='test'):
    return float(np.mean([r['mae_mV'] for r in evaluate(b, ds, split)]))


def fit(name, ds):
    b = BASELINES[name]()
    return b.fit(stream(ds, 'train'))


def test_registry_is_extensible_and_ordered():
    assert list(BASELINES)[:3] == ['zero', 'mean_map', 'scalar']
    from edalint.baselines import Baseline, register

    @register('const_test')
    class C(Baseline):
        def predict(self, s):
            return np.full(s.shape, 7.0)
    assert 'const_test' in BASELINES
    del BASELINES['const_test']


def test_zero_scalar_mean_map_by_hand():
    a = Sample('a', 'train', 'd1', {'x': np.zeros((4, 4))}, np.full((4, 4), 2.0))
    b = Sample('b', 'train', 'd2', {'x': np.zeros((4, 4))}, np.full((4, 4), 4.0))
    t = Sample('t', 'test', 'd3', {'x': np.zeros((8, 6))}, np.zeros((8, 6)))
    d = ListDataset([a, b, t])
    assert BASELINES['scalar']().fit(stream(d, 'train')).predict(t).mean() == pytest.approx(3.0)
    mm = BASELINES['mean_map']().fit(stream(d, 'train'))
    p = mm.predict(t)
    assert p.shape == (8, 6) and np.allclose(p, 3.0)
    assert BASELINES['zero']().predict(t).sum() == 0


def test_ladder_orders_on_synthetic_data(ds):
    zero = mae(BASELINES['zero'](), ds)
    scalar = mae(fit('scalar', ds), ds)
    ridge = mae(fit('ridge', ds), ds)
    green = mae(fit('green', ds), ds)
    assert zero > scalar > ridge          # linear per-pixel features beat a constant
    assert green < 0.5 * scalar           # the generator IS a radial kernel: one calibrated kernel recovers most of it
    assert green < ridge


def test_green_recovers_the_kernel_shape(ds):
    g = fit('green', ds)
    k = g.kernel_
    c = k.shape[0] // 2
    # true kernel 1/(1+r) (times 1e4 * average (1 + 0.5 eff/eff_max)): monotone decreasing with r
    prof = [k[c, c + r] for r in (0, 2, 5, 9)]
    assert prof[0] > prof[1] > prof[2] > prof[3] >= 0


def test_knn_effdist_gbm_run(ds):
    for name in ('knn', 'effdist', 'effdist_deg2'):
        rows = evaluate(fit(name, ds), ds, 'test')
        assert len(rows) == 4 and all(r['finite'] for r in rows)
    pytest.importorskip('sklearn')
    rows = evaluate(fit('gbm', ds), ds, 'test')
    assert all(r['finite'] for r in rows)
    assert mae(fit('gbm', ds), ds) < mae(BASELINES['zero'](), ds)


def test_effdist_needs_eff_dist_channel():
    s = Sample('a', 'train', 'd', {'power_all': np.ones((8, 8))}, np.ones((8, 8)))
    ok, why = BASELINES['effdist']().applicable(ListDataset([s]), s)
    assert not ok and 'eff_dist' in why


def test_run_zoo_reports_skips(ds):
    out = run_zoo(ds, 'train', 'train', ['zero', 'spice', 'nonsense'], log=lambda *_: None)
    assert 'summary' in out['zero'] and 'netlist' in out['spice']['skipped'] and out['nonsense']['skipped'] == 'unknown baseline'


def test_summarise_means():
    s = summarise([{'mae_mV': 1.0, 'seconds': 1.0}, {'mae_mV': 3.0, 'seconds': 3.0}])
    assert s['mean_mae_mV'] == 2.0 and s['mean_seconds'] == 2.0 and s['n'] == 2


def test_spice_baseline_on_fixture(tmp_path):
    from edalint.spice import exact_ir_drop
    make_iccad_tree(tmp_path)
    d = ICCAD23Dataset(root=str(tmp_path))
    sid = 'real/testcase1'
    s = d.get(sid)
    label, _ = exact_ir_drop(s.meta['netlist'], s.shape)
    d._cache.clear()
    d._cache[sid] = Sample(s.id, s.split, s.design, s.inputs, label, s.meta)       # label = exact map of this netlist
    b = BASELINES['spice']()
    ok, _ = b.applicable(d, d.get(sid))
    assert ok
    pred = b.predict(d.get(sid))
    assert np.abs(pred - label).max() < 1e-12 and 'seconds_solve' in b.last_info
    assert np.isfinite(pred).all() and pred.max() > 0
