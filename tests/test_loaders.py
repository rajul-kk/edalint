import json

import numpy as np
import pytest

from edalint.core import ProtocolError
from edalint.datasets import ICCAD23Dataset, N14Dataset, N28Dataset, open_dataset
from edalint.datasets import circuitnet as cn
from edalint.datasets.iccad23 import csv_shape, read_map_csv
from fixtures import make_iccad_tree, make_n14_tree


@pytest.fixture()
def iccad(tmp_path):
    make_iccad_tree(tmp_path)
    return ICCAD23Dataset(root=str(tmp_path), protocol_dir=str(tmp_path / 'protocol'))


def test_iccad_splits_and_twins(iccad):
    assert {k: len(v) for k, v in iccad.splits.items()} == {'train': 4, 'test': 2, 'fake': 2, 'visible': 2}
    assert set(iccad.splits['test']) == {'hidden/testcase7', 'hidden/testcase8'}
    m1, m2 = iccad.meta('real/testcase1'), iccad.meta('real/testcase2')
    assert m1['design_id'] == m2['design_id'] and m1['twin_of'] == 'real/testcase2'
    assert iccad.meta('hidden/testcase7')['design_id'] != m1['design_id']
    assert iccad.meta('hidden/testcase7')['twin_of'] == 'hidden/testcase8'
    assert iccad.meta('fake/current_map00')['twin_of'] is None


def test_iccad_native_size_and_units(iccad):
    s = iccad.get('real/testcase1')
    assert s.shape == (10, 10) and set(s.inputs) == {'current', 'eff_dist', 'pdn_density'}
    assert s.inputs['current'].dtype == np.float64
    assert csv_shape(iccad._cases['real/testcase1'].paths['current']) == (10, 10)


def test_iccad_optional_resize(tmp_path):
    make_iccad_tree(tmp_path)
    ds = ICCAD23Dataset(root=str(tmp_path), resize=(5, 5))
    s = ds.get('fake/current_map00')
    assert s.shape == (5, 5) and s.meta['resized_from'] == (10, 10)


def test_iccad_test_split_is_guarded_but_smoke_loadable(iccad):
    with pytest.raises(ProtocolError):
        iccad.get('hidden/testcase7')
    with pytest.raises(ProtocolError):
        list(iccad.iter('test'))
    s = iccad.load_unguarded('hidden/testcase7')          # shape / loading smoke test only
    assert s.shape == (12, 12)
    assert iccad.get('real/testcase1') is not None


def test_iccad_frozen_file_opens_the_guard(iccad, tmp_path):
    from edalint.protocol import code_hash
    (tmp_path / 'protocol').mkdir()
    (tmp_path / 'protocol' / 'FROZEN.json').write_text(json.dumps({'dataset': 'iccad23', 'code_hash': code_hash()}))
    assert iccad.get('hidden/testcase7') is not None
    (tmp_path / 'protocol' / 'FROZEN.json').write_text(json.dumps({'dataset': 'iccad23', 'code_hash': 'deadbeef'}))
    iccad._cache.clear()
    with pytest.raises(ProtocolError):
        iccad.get('hidden/testcase8')


def test_read_map_csv_roundtrip(tmp_path):
    a = np.random.default_rng(0).random((5, 7))
    np.savetxt(tmp_path / 'a.csv', a, delimiter=',', fmt='%.8e')
    assert np.allclose(read_map_csv(tmp_path / 'a.csv'), a, atol=1e-8)
    assert csv_shape(tmp_path / 'a.csv') == (5, 7)


# -- N14 ------------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def n14(tmp_path):
    root, csvs, man, names = make_n14_tree(tmp_path)
    return root, csvs, man, names


def _ds(n14, task, mode='authors', **kw):
    root, csvs, man, _ = n14
    return N14Dataset(task=task, mode=mode, root=str(root), split_csvs=csvs, manifest_path=man,
                      protocol_dir=str(root / 'protocol'), **kw)


def test_n14_splits_skip_incomplete_tarball_and_log(n14):
    ds = _ds(n14, 'drc')
    assert len(ds.splits['test']) == 4
    # DesC: .incomplete partner, no sha match -> whole tarball skipped
    assert [x for x in ds.skipped if x.get('design') == 'DesC' and 'tarball unusable' in x['reason']]
    assert all(ds.meta(i)['design'] != 'DesC' for i in ds.splits['train'])


def test_n14_reports_label_feature_mismatch(n14):
    ds = _ds(n14, 'drc')
    miss = [x for x in ds.skipped if x.get('id', '').startswith('DesA') and 'DRC_all' in x['reason']]
    assert len(miss) == 1 and len(ds.splits['train']) == 3
    ds2 = _ds(n14, 'congestion')                           # DRC_all is not needed for congestion: nothing skipped for DesA
    assert len(ds2.splits['train']) == 4
    assert ds2.unlisted['DesA'] == 0


def test_n14_authors_drc_shapes_and_transforms(n14):
    ds = _ds(n14, 'drc')
    s = ds.get(ds.ids('train')[0])
    assert len(s.inputs) == 9 and s.label.shape == (256, 256) and all(v.shape == (256, 256) for v in s.inputs.values())
    for nm in ('macro_region', 'RUDY_long', 'GR_vertical_overflow'):
        v = s.inputs[nm]
        assert v.min() == pytest.approx(0.0, abs=1e-6) and v.max() == pytest.approx(1.0, abs=1e-6)     # per-map min-max
    assert 0.0 <= s.label.min() and s.label.max() <= 1.0 and s.label.max() > 0.5                      # clip 200 then /200
    assert s.label.max() <= 1.0


def test_n14_authors_ir_24_channels_and_log_label(n14):
    ds = _ds(n14, 'ir_drop')
    s = ds.get(ds.ids('train')[0])
    assert len(s.channels()) == 24 and s.inputs['power_t'].shape == (20, 256, 256)
    # label in (0, 20) mV-ish, clipped to [1e-6, 50], log-scaled: (log10(x)+6)/(log10(50)+6)
    assert s.label.min() >= (np.log10(1e-6) + 6) / (np.log10(50) + 6) - 1e-9
    assert s.label.max() <= (np.log10(50) + 6) / (np.log10(50) + 6)
    assert s.label.max() > 0.8


def test_n14_authors_congestion_label_is_sum_then_minmax(n14):
    ds = _ds(n14, 'congestion')
    s = ds.get(ds.ids('train')[0])
    assert len(s.inputs) == 3 and s.label.min() == pytest.approx(0.0, abs=1e-6) and s.label.max() == pytest.approx(1.0, abs=1e-6)


def test_n14_raw_mode_keeps_native_grids(n14):
    ds = _ds(n14, 'drc', mode='raw')
    s = ds.get(ds.ids('train')[0])
    assert s.label.shape == (40, 44)
    assert s.inputs['eGR_horizontal_overflow'].shape == (72, 76)          # finer congestion_eGR grid
    assert s.stack().shape == (9, 40, 44)                                 # stack() resizes to the label grid
    assert s.label.max() > 100                                            # unclipped, unscaled
    ds_ir = _ds(n14, 'ir_drop', mode='raw')
    assert ds_ir.get(ds_ir.ids('train')[0]).inputs['power_t'].shape == (20, 40, 44)


def test_n14_test_design_not_read_without_freeze(n14):
    ds = _ds(n14, 'drc')
    assert ('DesB', 'test') not in ds._staged                              # test tarballs untouched at construction
    with pytest.raises(ProtocolError):
        ds.get(ds.ids('test')[0])
    shapes = ds.smoke_test_shapes(per_design=1)                            # shape-only smoke test is allowed
    assert shapes[0]['design'] == 'DesB' and shapes[0]['shapes']['DRC_all'] == (40, 44)


def test_n14_disk_cache_and_ram_cap(n14, tmp_path):
    ds = _ds(n14, 'congestion', cache_dir=str(tmp_path / 'cache'), max_per_design=2)
    assert len(ds.splits['train']) == 2
    s = ds.get(ds.ids('train')[0])
    assert s.label.shape == (256, 256)
    assert list((tmp_path / 'cache').rglob('*.npy'))
    # second construction re-uses the cache (presence json exists)
    ds2 = _ds(n14, 'congestion', cache_dir=str(tmp_path / 'cache'), max_per_design=2)
    assert np.allclose(ds2.get(ds2.ids('train')[0]).label, s.label)


def test_n14_std_faithful_to_authors_script():
    z = np.zeros((3, 3))
    assert cn._std(z) is z                                                # max == 0: returned unchanged
    c = np.full((3, 3), 2.0)
    assert not np.isfinite(cn._std(c)).all()                              # constant non-zero: NaN, as in the original script


def test_split_csv_parsing_and_packaged_files():
    tr, te = cn.packaged_split_csvs()
    a, b = cn.read_split_csv(tr), cn.read_split_csv(te)
    assert (len(a), len(b)) == (6790, 3544) and not set(a) & set(b)
    assert cn.design_of('RISCY-FPU_freq_50_mp_1_fpu_70_fpa_1.0_p_6_fi_ar') == 'RISCY-FPU'
    assert {cn.design_of(n) for n in b} == {'zero-riscy', 'openc910-1', 'Vortex-large'}


def test_stubs_and_factory():
    with pytest.raises(NotImplementedError):
        N28Dataset()
    with pytest.raises(ValueError):
        open_dataset('nonsense')
