"""Registered CircuitNet-N14 runner: official ROC machinery against a brute-force transcription of the authors' code, the seeded fit
sample, the guards, and an end-to-end dry run on the synthetic N14 tarball tree (no real data)."""
import json

import numpy as np
import pytest

from edalint import p4_n14 as P
from edalint.core import ProtocolError
from edalint.protocol import code_hash
from fixtures import make_n14_tree


def _authors_roc(preds, golds, thr):
    """Line-by-line transcription of calculated_score + calculate_all + roc_prc (ROC part) for small inputs."""
    ths = np.linspace(0, 1, 200, endpoint=False)
    tl, fl = [], []
    for t in ths:
        n = 0
        tpr_s = fpr_s = 0.0
        for p, g in zip(preds, golds):
            y = (g.ravel() >= thr)
            q = (p.ravel() >= t)
            tp, fp, fn, tn = int((q & y).sum()), int((q & ~y).sum()), int((~q & y).sum()), int((~q & ~y).sum())
            if (fp == 0 and tn == 0) or (tp == 0 and fn == 0) or (tp == 0 and fp == 0):
                continue
            tpr_s += tp / (tp + fn)
            fpr_s += fp / (fp + tn)
            n += 1
        if n:
            tl.append(tpr_s / n)
            fl.append(fpr_s / n)
    f2, t2 = [], []
    for i, j in zip(fl, tl):
        if i not in f2:
            f2.append(i)
            t2.append(j)
    f2.reverse()
    t2.reverse()
    pairs = sorted(zip(f2, t2))
    f2 = [a for a, _ in pairs] + [1]
    t2 = [b for _, b in pairs] + [1]
    return sum((t2[i] + t2[i + 1]) * (f2[i + 1] - f2[i]) / 2 for i in range(len(t2) - 1))


def test_roc_acc_matches_authors_procedure():
    rng = np.random.default_rng(0)
    golds = [rng.random((12, 12)) for _ in range(5)]
    golds.append(np.zeros((12, 12)))                       # no positive pixel: skipped by the authors' rule
    preds = [g * 0.7 + rng.random((12, 12)) * 0.3 for g in golds]
    acc = P.RocAcc()
    for p, g in zip(preds, golds):
        acc.add(p, g, 0.8)
    assert acc.n_samples == 6 and acc.n_no_positive == 1
    assert abs(acc.area() - _authors_roc(preds, golds, 0.8)) < 1e-12
    again = P.RocAcc.from_dict(json.loads(json.dumps(acc.to_dict())))
    assert abs(again.area() - acc.area()) < 1e-15


def test_roc_constant_prediction_is_zero_officially_half_anchored_and_all_skipped_is_nan():
    g = np.random.default_rng(1).random((10, 10))
    acc = P.RocAcc()
    acc.add(np.full((10, 10), 0.4), g, 0.9)
    assert acc.area() == 0.0                                   # the authors' curve is not anchored at (0, 0): a constant predictor scores 0
    assert abs(acc.area(anchored=True) - 0.5) < 1e-12
    empty = P.RocAcc()
    empty.add(g, np.zeros((10, 10)), 0.9)
    assert np.isnan(empty.area())


def test_skill_definition():
    assert abs(P.skill('congestion', 0.1, 0.4) - 0.75) < 1e-12              # lower is better
    assert abs(P.skill('ir_drop', 0.75, 0.5) - 0.5) < 1e-12                 # higher is better, headroom 1 - ref
    assert np.isnan(P.skill('drc', float('nan'), 0.5))


def test_fit_sample_is_seeded_nested_and_sorted():
    names = [f'D_freq_{i:04d}' for i in range(1000)]
    a = P.fit_selector(400, 7)('train', 'D', names)
    assert a == P.fit_selector(400, 7)('train', 'D', names) and len(a) == 400 and a == sorted(a)
    assert a != P.fit_selector(400, 8)('train', 'D', names)
    assert set(P.seeded_order('D', names, 7)[:100]) <= set(a)                  # K_GAP sample is a prefix of the fit sample
    assert P.fit_selector(400, 7)('test', 'D', names) == names                 # test rows are never subsampled
    assert P.fit_selector(400, 7)('train', 'D', names[:50]) == names[:50]


def test_registered_stage_refuses_without_freeze(tmp_path, monkeypatch):
    monkeypatch.setenv('EDALINT_PROTOCOL_DIR', str(tmp_path))
    with pytest.raises(ProtocolError, match='frozen'):
        P._preflight(P.Run(out_dir=tmp_path / 'o'), 'faults')
    with pytest.raises(ProtocolError, match='official split'):
        P._preflight(P.Run(out_dir=tmp_path / 'o', csvs=('a', 'b')), 'faults')      # registered runs use the official files only


def test_dry_run_refuses_official_test_rows(tmp_path):
    tr, te = P.packaged_split_csvs()
    (tmp_path / 'p').mkdir()
    (tmp_path / 'p' / 'FROZEN_n14.json').write_text(json.dumps({'dataset': 'n14', 'code_hash': code_hash()}))
    run = P.Run(out_dir=tmp_path / 'o', csvs=(str(tr), str(te)), protocol_dir=str(tmp_path / 'p'), dry_run=True)
    with pytest.raises(ProtocolError, match='official test rows'):
        P._preflight(run, 'faults')
    with pytest.raises(ProtocolError, match='hold-out'):
        P.make_dry_run(tmp_path / 'd', ['no-such-design'])


def test_make_dry_run_holds_out_train_designs_only(tmp_path):
    run = P.make_dry_run(tmp_path / 'd', ['nvdla-small'])
    from edalint.datasets.circuitnet import read_split_csv
    held = set(read_split_csv(run.csvs[1]))
    assert held and not (held & P._official_test_names()) and all(n.startswith('nvdla-small') for n in held)
    assert not (set(read_split_csv(run.csvs[0])) & held)
    assert P._preflight(run, 'faults')['dry_run'] is True


@pytest.fixture()
def synth_run(tmp_path):
    root, csvs, man, names = make_n14_tree(tmp_path / 'data')
    # DesC is unusable (no sha recorded); drop it from the train CSV to keep the run small
    tr = [ln for ln in open(csvs[0]).read().split() if 'DesC' not in ln]
    (tmp_path / 'train.csv').write_text('\n'.join(tr) + '\n')
    (tmp_path / 'p').mkdir()
    (tmp_path / 'p' / 'FROZEN_n14.json').write_text(json.dumps({'dataset': 'n14', 'code_hash': code_hash(), 'dry_run': True}))
    return P.Run(out_dir=tmp_path / 'out', cache_root=tmp_path / 'cache', root=str(root), csvs=(str(tmp_path / 'train.csv'), csvs[1]),
                 protocol_dir=str(tmp_path / 'p'), dry_run=True, manifest_path=man)


def test_end_to_end_dry_run_on_synthetic_tree(synth_run):
    run = synth_run
    P.stage_faults(run)
    faults = json.loads((run.out_dir / 'faults.json').read_text())
    assert faults['label'] == 'DRY RUN'
    d = faults['tasks']['drc']['designs']
    assert d['DesA']['n_listed'] == 4 and d['DesA']['n_missing'] == 1 and d['DesB']['n_complete'] == 4      # one DRC_all file missing in DesA
    assert not faults['duplicates']['drc']['cross_split_exact']
    for t in ('congestion', 'drc'):
        P.stage_models(run, t)
        P.stage_gap(run, t)
    models = {t: json.loads((run.out_dir / f'models_{t}.json').read_text()) for t in ('congestion', 'drc')}
    assert set(models['drc']['results']) == set(P.BASELINES['drc']) and len(models['drc']['results']['ridge']['rows']) == 4
    assert models['drc']['fit_sample']['seed'] == P.SEED
    with pytest.raises(FileExistsError):
        P.stage_models(run, 'drc')                                             # write-once
    P.stage_verdicts(run)
    v = json.loads((run.out_dir / 'verdicts.json').read_text())
    assert set(v['H1']['tasks']) == {'congestion', 'drc'} and isinstance(v['H2_verdict'], bool)
    assert (run.out_dir / 'tables.md').is_file()


def test_h1_rule_and_not_evaluable():
    def mk(vals):
        rows = [{'id': f'{d}{i}', 'design': d, 'nrms': v, 'ssim': v, 'nmae': v, 'pearson': v, 'seconds': 0.0} for d, v in vals.items() for i in range(2)]
        return {'rows': rows, 'official_roc': {}}
    # congestion: lower is better; reference 0.020; mean_map 0.20 -> published gain 0.18, the 0.8 line is a gain of 0.144
    mm = {'a': 0.20, 'b': 0.20}
    ok = {'results': {'mean_map': mk(mm), 'ridge': mk({'a': 0.05, 'b': 0.05})}}
    bad = {'results': {'mean_map': mk(mm), 'ridge': mk({'a': 0.07, 'b': 0.07})}}
    assert P.h1_task('congestion', ok)['verdict'] is True
    assert P.h1_task('congestion', bad)['verdict'] is False
    # reference not better than mean_map: degenerate TRUE if any baseline beats it
    weak = {'results': {'mean_map': mk({'a': 0.01, 'b': 0.01}), 'ridge': mk({'a': 0.005, 'b': 0.005})}}
    r = P.h1_task('congestion', weak)
    assert r['degenerate'] is True and r['verdict'] is True
    # IR drop: official AUC undefined for every baseline -> not evaluable (mechanical rule)
    und = P.RocAcc()
    und.add(np.zeros((6, 6)), np.zeros((6, 6)), 0.9)
    blk = {'rows': [], 'official_roc': {'a': und.to_dict()}}
    assert P.h1_task('ir_drop', {'results': {'mean_map': blk, 'ridge': blk}})['verdict'] == 'not evaluable'


def test_ssim_primary_is_not_evaluable_when_mean_map_is_saturated():
    def mk(v):
        return {'rows': [{'id': f'{d}{i}', 'design': d, 'nrms': v, 'ssim': v, 'nmae': v, 'pearson': v, 'seconds': 0.0} for d in 'ab' for i in range(2)],
                'official_roc': {}}
    r = P.h1_task('drc', {'results': {'mean_map': mk(0.9995), 'ridge': mk(0.9999)}})
    assert r['verdict'] == 'not evaluable' and 'saturated' in r['reason']
    r2 = P.h1_task('drc', {'results': {'mean_map': mk(0.30), 'ridge': mk(0.95)}})          # not saturated: the usual rule (reference 0.7136)
    assert r2['verdict'] is True


def test_anchored_sensitivity_is_reported_and_flags_disagreement():
    # one design, predictions that reach high values only for 'good': the unanchored area of a weak predictor is far below its anchored area
    rng = np.random.default_rng(0)
    golds = [rng.random((16, 16)) for _ in range(4)]

    def block(scale):
        acc = P.RocAcc()
        for g in golds:
            acc.add(g * scale, g, 0.9)
        return {'rows': [], 'official_roc': {'d': acc.to_dict()}}
    models = {'results': {'mean_map': block(0.3), 'ridge': block(1.0)}}
    h = P.compute_verdicts({'tasks': {t: {'designs': {}} for t in P.TASK_LIST}, 'duplicates': {t: {'cross_split_exact': []} for t in P.TASK_LIST}},
                           {'ir_drop': models}, {})
    s = h['H1']['tasks']['ir_drop']['sensitivity_anchored']
    assert set(s) >= {'verdict', 'agrees_with_official'}
