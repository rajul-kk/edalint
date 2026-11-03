"""P8 fault injectors: injection is isolated from the source, the strongest level of each fault is flagged by its check on synthetic data,
the clean layout is not, and the IC-ThermBench loader reads a small HDF5 fixture."""
import json

import numpy as np
import pytest

from edalint_p8 import faults as F
from edalint.checks import CHECKS, Options
from edalint.datasets.synthetic import make_synthetic

OPT = Options(split='all', max_samples=None, dup_max_samples=800, orient_samples=40)


@pytest.fixture(scope='module')
def base():
    ds = make_synthetic(n_train_designs=8, n_test_designs=4, per_design=2, size=40, seed=3, guard_test=False)
    return ds, F.materialise(ds, ds.ids('all'))


def layout(base, seed=0):
    ds, samples = base
    return F.assign_splits([s for s in F.materialise(ds, [s.id for s in samples])], np.random.default_rng(seed))


def run(ds, code):
    return CHECKS[code](ds, OPT)


def test_materialise_is_isolated(base):
    ds, samples = base
    before = ds.get(samples[0].id).label.copy()
    copies = F.materialise(ds, [samples[0].id])
    copies[0].label[:] = 0
    assert np.array_equal(ds.get(samples[0].id).label, before)


def test_clean_layout_is_not_flagged(base):
    ds = F.build(layout(base), base[0])
    for code in ('E001', 'E002', 'E003', 'E004', 'E005'):
        r = run(ds, code)
        assert r.status in ('ok',), (code, [f.message for f in r.findings])


@pytest.mark.parametrize('name', sorted(F.FAULTS))
def test_strongest_level_detected(base, name):
    fault = F.FAULTS[name]
    samples = layout(base)
    if F.applicable(fault, samples):
        pytest.skip(F.applicable(fault, samples))
    level = fault.severities[-1]
    if name == 'zero_channel':
        pytest.skip('E002 reports a single all-zero channel as info only: expected limit (protocol section 5)')
    bad = fault.inject(samples, np.random.default_rng(1), level)
    r = run(F.build(bad, base[0]), fault.check)
    assert F.detected(r, fault), (name, [f.message for f in r.findings])


def test_zero_channel_is_a_documented_limit(base):
    fault = F.FAULTS['zero_channel']
    bad = fault.inject(layout(base), np.random.default_rng(1), 4)
    r = run(F.build(bad, base[0]), 'E002')
    assert not F.detected(r, fault)                                   # info only: the check does not raise it above info


def test_weakest_levels_are_ordered_by_strength(base):
    samples = layout(base)
    fault = F.FAULTS['near_copy']
    rng = np.random.default_rng(5)
    s = samples[0]
    from edalint.checks import _zdesc
    cs = []
    for lv in (0, 4):
        a = F._noisy(s.label, F.CORRS[lv], rng)
        cs.append(float(_zdesc(s.label, 32) @ _zdesc(a, 32)) / 1024)
    assert cs[0] < cs[1] and abs(cs[0] - F.CORRS[0]) < 5e-4 and abs(cs[1] - F.CORRS[4]) < 5e-4
    assert fault.check == 'E003'


def test_icthermbench_loader(tmp_path):
    h5py = pytest.importorskip('h5py')
    from edalint_p8.icthermbench import ICThermBenchDataset
    folder = tmp_path / 'level3_steady'
    folder.mkdir()
    n, hw = 40, 8
    rng = np.random.default_rng(0)
    x = rng.random((n, 4, 1, hw, hw), dtype=np.float32)
    y = rng.random((n, 1, hw, hw), dtype=np.float32) + 300
    with h5py.File(folder / 'input.mat', 'w') as f:
        f['data'] = x
    with h5py.File(folder / 'output.mat', 'w') as f:
        f['data'] = y
    ds = ICThermBenchDataset(tmp_path, 'level3')
    assert {k: len(v) for k, v in ds.splits.items()} == {'train': 28, 'val': 4, 'test': 8}
    s = ds.get(ds.splits['train'][0])
    assert s.shape == (hw, hw) and list(s.inputs) == ['chiplet_power', 'grid_x', 'grid_y', 'local_thermal_k']
    assert np.array_equal(s.inputs['chiplet_power'], x[0, 0, 0]) and np.array_equal(s.label, y[0, 0])
    fixed = ICThermBenchDataset(tmp_path, 'level3', fix_orientation=True).get(ds.splits['train'][0])
    assert np.array_equal(fixed.inputs['chiplet_power'], x[0, 0, 0].T) and np.array_equal(fixed.label, y[0, 0])
    with pytest.raises(Exception):
        ds.get(ds.splits['test'][0])                                  # guarded test split
    (tmp_path / 'level3_steady' / 'manifest.json').write_text(json.dumps([{'case': f'C{i // 20}', 'index': i % 20} for i in range(n)]))
    assert ICThermBenchDataset(tmp_path, 'level3').meta(ds.splits['train'][21])['design'] == 'C1'


@pytest.mark.parametrize('const', [0, 1, 3])
def test_e003_near_duplicate_is_not_diluted_by_constant_channels(const):
    """Amendment A3: constant channels used to cap the input correlation at (varying channels / channels), hiding near-duplicates."""
    from edalint_p8.e003_fixed import e003_duplicates
    from edalint.core import ListDataset, Sample
    rng = np.random.default_rng(0)

    def mk(i, split):
        ins = {'power': rng.random((64, 64)), 'k': rng.random((64, 64))}
        ins.update({f'c{j}': np.full((64, 64), 300.0) for j in range(const)})
        return Sample(f's{i}', split, f'd{i}', ins, ins['power'] * 2 + 1)
    a = mk(0, 'train')
    r2 = np.random.default_rng(1)
    b = Sample('s1', 'val', 'd1', {k: v + r2.normal(scale=1e-4 * v.std(), size=v.shape) if v.std() > 0 else v.copy() for k, v in a.inputs.items()},
               a.label + r2.normal(scale=1e-4, size=a.label.shape))
    res = e003_duplicates(ListDataset([a, b, mk(2, 'train')], test_splits=()), Options(split='all'))
    assert res.metrics['max_input_corr_offdiag'] > 0.999 and res.metrics['n_near_dup_pairs'] == 1
    assert any(f.severity == 'error' and 'ACROSS' in f.message for f in res.findings)
