"""P7 pre-freeze tests: train/eval disjointness, loader-skip-corrected train count, export round trip, scorer abort and parity with N14 `_score_rows`.
Only synthetic rows and (skippable) cached TRAIN designs Vortex-small / nvdla-small are read; no evaluation-design value is opened."""
import json
from pathlib import Path

import numpy as np
import pytest

from edalint.baselines import stream
from edalint.baselines.zoo import MeanMap, PixelRidge, ScalarMean
from edalint.core import ListDataset, ProtocolError, Sample
from edalint.datasets.circuitnet import assemble, design_of
from edalint.p4_n14 import LABEL_THRESHOLD, _score_rows
from edalint_p7.data import (ShardSource, SubsetSource, check_disjoint, check_train_ids, export_rows, open_train_rows, sha256_file,
                             stack_inputs, train_names)
from edalint_p7.score import check_scored_ids, load_eval_ids, score_predictions

REPO = Path(__file__).resolve().parents[1]
N14 = REPO / 'results' / 'n14'


# -- disjointness and counts --------------------------------------------------------------------------------------------------------------
def test_train_names_are_the_official_6790_rows():
    names = train_names()
    assert len(names) == 6790 and len(set(names)) == 6790
    assert {design_of(n) for n in names} == {'RISCY', 'RISCY-FPU', 'Vortex-small', 'nvdla-small'}


@pytest.mark.parametrize('task', ['ir_drop', 'congestion'])
def test_train_ids_disjoint_from_registered_eval_ids_by_id_and_by_design(task):
    if not (N14 / f'models_{task}.json').is_file():
        pytest.skip('registered N14 result file not present')
    ev = load_eval_ids(task)
    assert sum(len(v) for v in ev.values()) == 3544 and set(ev) == {'Vortex-large', 'openc910-1', 'zero-riscy'}
    names = train_names()
    rep = check_train_ids(names, ev, names, [])
    assert rep['n_train'] == 6790 and rep['n_skipped'] == 0


def test_disjointness_check_fails_on_overlap_or_shared_design():
    names = ['RISCY_freq_1', 'RISCY_freq_2', 'nvdla-small_freq_1']
    ev = {'zero-riscy': ['zero-riscy_freq_1']}
    check_train_ids(names, ev, names)
    with pytest.raises(AssertionError, match='also eval ids'):
        check_train_ids(names, {'x': ['RISCY_freq_2']}, names)
    with pytest.raises(AssertionError, match='share designs'):
        check_train_ids(names, {'RISCY': ['RISCY_freq_9']}, names)
    with pytest.raises(AssertionError, match='duplicate'):
        check_disjoint(names + names[:1], ev)


def test_train_count_equals_official_count_corrected_for_loader_skips():
    names = ['RISCY_freq_%d' % i for i in range(10)]
    skipped = names[:3]
    assert check_train_ids(names[3:], {}, names, skipped)['n_train'] == 7
    with pytest.raises(AssertionError, match='listed'):
        check_train_ids(names[3:], {}, names, [])                       # rows vanished without a loader-skip record
    with pytest.raises(AssertionError, match='listed'):
        check_train_ids(names, {}, names, skipped)                      # skipped rows still present
    with pytest.raises(AssertionError, match='outside'):
        check_train_ids(names[3:] + ['RISCY_freq_99'], {}, names, skipped)


def test_train_opener_has_no_evaluation_rows(tmp_path):
    ds = open_train_rows('congestion', stage=False, designs=['RISCY'])
    assert ds.ids('train') and ds.splits['test'] == []
    assert ds.is_test_split('all') is False


# -- export round trip -----------------------------------------------------------------------------------------------------------------------
def _members(task, seed, h=16):
    rng = np.random.default_rng(seed)
    feats = {'ir_drop': ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t', 'IR_drop'],
             'congestion': ['macro_region', 'RUDY', 'RUDY_pin', 'GR_horizontal_overflow', 'GR_vertical_overflow']}[task]
    return {f: (rng.random((20, h, h)) + 0.05 if f == 'power_t' else rng.random((h, h)) * (30 if f == 'IR_drop' else 1) + 0.05) for f in feats}


def _list_ds(task, designs=('RISCY', 'RISCY-FPU'), per=5):
    samples = []
    for d in designs:
        for k in range(per):
            ins, lab = assemble(task, 'authors', (lambda m: (lambda f: m[f]))(_members(task, len(samples))))
            samples.append(Sample(f'{d}_freq_{k}', 'train', d, ins, lab))
    return ListDataset(samples, name=f'circuitnet_n14_{task}_authors', task=task)


@pytest.mark.parametrize('task', ['congestion', 'ir_drop'])
def test_export_round_trip_is_bit_exact_with_sha256_manifest(task, tmp_path):
    ds = _list_ds(task)
    ids = ds.ids('train')
    man = export_rows(task, ds, ids, tmp_path / 'a', shard_rows=3)
    assert man['n_rows'] == 10 and man['nonfinite_ids'] == [] and man['missing_ids'] == []
    assert [len(s['ids']) for s in man['shards']] == [3, 2, 3, 2]
    assert all(len({design_of(i) for i in s['ids']}) == 1 and s['design'] == design_of(s['ids'][0]) for s in man['shards'])     # sharded by design
    src = ShardSource(tmp_path / 'a')
    assert sorted(src.ids) == sorted(ids) and len(src) == 10
    for k, sid in enumerate(src.ids):
        s = ds.get(sid)
        x, y = src.get(k)
        assert x.dtype == np.float32 and y.dtype == np.float32
        np.testing.assert_array_equal(x, stack_inputs(task, s.inputs))        # bit-exact against assemble
        np.testing.assert_array_equal(y, s.label.astype(np.float32))
    for sh in man['shards']:
        assert sha256_file(tmp_path / 'a' / sh['path']) == sh['sha256']
    assert json.loads((tmp_path / 'a' / 'manifest.json').read_text())['shards'] == man['shards']


def test_export_is_reproducible_and_detects_tampering(tmp_path):
    ds = _list_ds('congestion')
    m1 = export_rows('congestion', ds, ds.ids('train'), tmp_path / 'a', shard_rows=4)
    m2 = export_rows('congestion', ds, ds.ids('train')[::-1], tmp_path / 'b', shard_rows=4)
    assert [s['sha256'] for s in m1['shards']] == [s['sha256'] for s in m2['shards']]       # fixed zip timestamps, sorted rows
    p = tmp_path / 'a' / m1['shards'][0]['path']
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 1
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='sha256'):
        ShardSource(tmp_path / 'a').get(0)


def test_export_records_non_finite_rows_and_subset_source(tmp_path):
    ds = _list_ds('congestion')
    bad = ds.ids('train')[2]
    ds._samples[bad].inputs['RUDY'][0, 0] = np.nan
    man = export_rows('congestion', ds, ds.ids('train'), tmp_path / 'a')
    assert man['nonfinite_ids'] == [bad]
    src = ShardSource(tmp_path / 'a')
    assert src.nonfinite_ids == [bad]
    sub = SubsetSource(src, [i for i in src.ids if i != bad][:4])
    assert len(sub) == 4 and sub.nonfinite_ids == []
    with pytest.raises(ValueError):
        SubsetSource(src, ['nope'])


# -- scorer -------------------------------------------------------------------------------------------------------------------------------
def test_scorer_aborts_when_the_scored_id_set_differs():
    exp = {'A': ['A_1', 'A_2'], 'B': ['B_1']}
    check_scored_ids(['A_1', 'B_1', 'A_2'], exp)
    with pytest.raises(ProtocolError, match='1 missing'):
        check_scored_ids(['A_1', 'A_2'], exp)
    with pytest.raises(ProtocolError, match='unexpected'):
        check_scored_ids(['A_1', 'A_2', 'B_1', 'C_1'], exp)
    with pytest.raises(ProtocolError, match='duplicate'):
        check_scored_ids(['A_1', 'A_2', 'B_1', 'B_1'], exp)
    g = np.random.default_rng(0).random((16, 16))
    with pytest.raises(ProtocolError):
        score_predictions('congestion', [('A_1', 'A', g, g, 0.0)], {'A': ['A_1', 'A_2']})


def _assert_rows_equal(mine, ref):
    assert len(mine['rows']) == len(ref)
    for a, b in zip(mine['rows'], ref):
        assert a['id'] == b['id'] and a['design'] == b['design']
        for k in ('nrms', 'ssim', 'nrms_float', 'ssim_float', 'nmae', 'pearson', 'roc_auc_sample'):
            if k in b:
                assert a[k] == pytest.approx(b[k], rel=0, abs=1e-9, nan_ok=True), k
            else:
                assert k not in a


@pytest.mark.parametrize('task', ['congestion', 'ir_drop'])
def test_scorer_reproduces_score_rows_on_train_only_synthetic_rows(task):
    ds = _list_ds(task, designs=('RISCY', 'RISCY-FPU'), per=6)
    fit = ds.ids('train')[:8]
    base = {'mean_map': MeanMap().fit(stream(ds, fit)), 'scalar': ScalarMean().fit(stream(ds, fit))}
    by = {}
    for sid in ds.ids('train'):
        by.setdefault(ds.meta(sid)['design'], []).append(sid)
    rows, acc = _score_rows(task, ds, by, base)
    for name, b in base.items():
        items = ((sid, des, b.predict(ds.get(sid)), ds.get(sid).label, 0.0) for des, ids in by.items() for sid in ids)
        mine = score_predictions(task, items, by)
        _assert_rows_equal(mine, rows[name])
        assert mine['official_roc'] == {d: a.to_dict() for d, a in acc[name].items()} if task in LABEL_THRESHOLD else mine['official_roc'] == {}


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_scorer_applies_nan_to_num_like_the_baselines():
    g = np.random.default_rng(1).random((16, 16))
    pred = g.copy()
    pred[0, 0] = np.nan
    pred[1, 1] = np.inf
    r = score_predictions('congestion', [('A_1', 'A', pred, g, 0.0)], None)['rows'][0]
    ref = np.nan_to_num(pred)
    from edalint import metrics as M
    assert r['nrms'] == M.nrms_official(ref, g) and np.isfinite(r['nrms'])


CACHE = Path('E:/edalint_data/n14_registered_cache/congestion')


def test_scorer_reproduces_score_rows_on_cached_real_train_rows():
    """Real congestion rows of the train designs Vortex-small / nvdla-small (registered fit-sample cache), ridge + mean_map fitted on the same
    rows. Skipped when the cache is absent."""
    if not (CACHE / 'authors' / 'Vortex-small' / 'RUDY').is_dir():
        pytest.skip('registered cache not present')
    from edalint.p4_n14 import fit_selector
    ds = open_train_rows('congestion', designs=['Vortex-small', 'nvdla-small'], cache_dir=str(CACHE), select=fit_selector(400))
    ids = ds.ids('train')[::6]
    assert len(ids) > 20
    base = {'mean_map': MeanMap().fit(stream(ds, ids)), 'ridge': PixelRidge(alpha=10.0).fit(stream(ds, ids))}
    by = {}
    for sid in ids:
        by.setdefault(ds.meta(sid)['design'], []).append(sid)
    rows, _ = _score_rows('congestion', ds, by, base)
    for name, b in base.items():
        items = ((sid, des, b.predict(ds.get(sid)), ds.get(sid).label, 0.0) for des, v in by.items() for sid in v)
        _assert_rows_equal(score_predictions('congestion', items, by), rows[name])
