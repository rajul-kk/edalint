"""P7 pre-freeze tests for the verdict / table aggregator (edalint_p7.verdicts) and the exporter's refusal. Synthetic, train-free rows only:
the ROC accumulators are built by hand, no dataset, split or test row is opened."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from edalint.core import ProtocolError
from edalint.runner import ResultExists
from edalint_p7 import guard, verdicts as V
from edalint_p7.verdicts import (compute_p7_verdicts, decide, gate_threshold, h_a_gate, h_b, h_c, nrms_view, render_tables_p7, run_verdicts,
                                 check_ids)
from test_p7_guard import make_repo, write_freeze

REPO = Path(__file__).resolve().parents[1]
DESIGNS = ('Vortex-large', 'openc910-1', 'zero-riscy')
F = 0.05                                    # one ROC point (F, T), area = (T + 1)(1 - F) / 2


def roc(area, n_samples=2, n_no_positive=0, undefined=False):
    """A RocAcc dict whose official area is `area` (one threshold with num 1); anchored area is the same value + F*T/2."""
    n = 200
    d = {k: [0.0] * n for k in ('tpr', 'fpr', 'pre', 'num')}
    if not undefined:
        t = 2 * area / (1 - F) - 1
        d['tpr'][5], d['fpr'][5], d['pre'][5], d['num'][5] = t, F, 0.5, 1.0
    d.update(n_samples=n_samples, n_no_positive=n_no_positive)
    return d


def rows_for(ids_by_design, nrms=0.3):
    return [{'id': i, 'design': d, 'seconds': 0.0, 'nrms': nrms, 'ssim': 0.5} for d, ids in ids_by_design.items() for i in ids]


IDS = {d: [f'{d}_a', f'{d}_b'] for d in DESIGNS}


def n14_models(areas=(0.60, 0.60, 0.60), ridge=(0.85, 0.85, 0.85)):
    res = {'mean_map': {'rows': rows_for(IDS), 'official_roc': {d: roc(a) for d, a in zip(DESIGNS, areas)}},
           'ridge': {'rows': rows_for(IDS), 'official_roc': {d: roc(a) for d, a in zip(DESIGNS, ridge)}},
           'gbm': {'rows': rows_for(IDS), 'official_roc': {d: roc(0.7) for d in DESIGNS}}}
    return {'eval_ids': IDS, 'results': res}


def payload(seed, area, **kw):
    nrms = kw.pop('nrms', None)
    rows = rows_for(IDS) if nrms is None else nrms
    return {'task': 'ir_drop', 'arm': 'A', 'seed': seed, 'rows': rows, 'eval_ids': IDS, 'p7_code_hash': 'x',
            'official_roc': {d: roc(area if not np.isscalar(area) else area, **kw) for d in DESIGNS}}


def train_rec(**kw):
    r = {'last_block_loss': 20.0, 'final_params_finite': True, 'skipped_steps': 4, 'iters_done': 30000, 'diverged_at': None}
    r.update(kw)
    return r


def cpv(seeds, n14, reg=None, tr='ok'):
    """compute_p7_verdicts with a clean train record for every seed unless `tr` is given."""
    if tr == 'ok':
        tr = {s: train_rec() for s in V.SEEDS}
    return compute_p7_verdicts(seeds, n14, reg, tr)


def seeds_with(areas):
    return {s: (None if a is None else payload(s, a)) for s, a in zip(V.SEEDS, areas)}


# -- verdict logic at boundary values ------------------------------------------------------------------------------------------
def test_constants_match_the_protocol_text():
    assert V.GATE_FACTOR == 0.5 and V.PUBLISHED == 0.8671 and V.H_B_FRACTION == 0.8 and V.BAND_AUC == 0.03
    assert V.SEEDS == (20261101, 20261102, 20261103)
    thr = gate_threshold(0.6301175015416564)                      # registered mean_map design-macro
    assert thr == pytest.approx(0.7486, abs=1e-4)                 # section 5 / 12.8: 0.7486


def test_gate_boundary_exact_binary_values():
    m, p = 0.5, 0.875                                             # threshold 0.6875 is exactly representable
    assert gate_threshold(m, p) == 0.6875
    assert h_a_gate(0.6875, m, p) is True                         # >= : equality passes
    assert h_a_gate(0.6875 - 1e-9, m, p) is False
    assert h_a_gate(float('nan'), m, p) is False


def test_gate_boundary_at_real_constants():
    m = 0.6301175015416564
    thr = gate_threshold(m)
    assert h_a_gate(thr + 1e-6, m) and not h_a_gate(thr - 1e-6, m)


def test_h_b_boundary():
    assert h_b(0.4, 0.5) is True                                  # 0.8 x 0.5 == 0.4: equality is TRUE
    assert h_b(0.4 - 1e-9, 0.5) is False
    assert h_b(0.5, 0.5) is True


def test_decide_cases():
    m = 0.6
    assert decide(0.70, m, 0.25)['verdict'] is None               # gate fails (0.7 < 0.7336): no verdict, H-B only descriptive
    assert decide(0.70, m, 0.25)['h_b_descriptive'] is True
    assert decide(0.80, m, 0.25)['verdict'] is True               # 0.25 >= 0.16
    assert decide(0.93, m, 0.25)['verdict'] is False              # 0.25 < 0.264
    assert decide(float('nan'), m, 0.25)['verdict'] is None


def test_h_c_band():
    assert h_c(0.8671 + 0.03)['within_band'] and h_c(0.8671 - 0.03 + 1e-9)['within_band']
    assert not h_c(0.8671 - 0.0301)['within_band']
    assert h_c(0.80)['D'] == pytest.approx(-0.0671)


# -- aggregation -----------------------------------------------------------------------------------------------------------------
def test_true_verdict_and_sensitivities():
    v = cpv(seeds_with([0.80, 0.82, 0.78]), n14_models())
    assert v['verdict'] is True and v['verdict_label'] == 'TRUE'
    assert v['inputs']['best_baseline'] == 'ridge' and v['inputs']['G_b'] == pytest.approx(0.25)
    assert v['primary']['r_t'] == pytest.approx(0.80) and v['H_A']['gate_pass'] is True
    s = v['sensitivities']
    assert s['best_seed']['seed'] == str(V.SEEDS[1]) and s['worst_seed']['seed'] == str(V.SEEDS[2]) and s['seed_sensitive'] is False
    assert s['sample_pooled']['verdict'] is True and s['anchored_roc']['verdict'] is True
    assert s['published_credit']['budget_sensitive'] is False
    assert s['per_design']['n_designs_beating_mean_map'] == 3 and v['registered_n14'] == {'available': False}


def test_false_verdict_when_the_model_beats_ridge_by_more_than_the_line():
    v = cpv(seeds_with([0.93, 0.93, 0.93]), n14_models())
    assert v['verdict'] is False and v['H_B']['ratio_G_b_over_G_R'] == pytest.approx(0.25 / 0.33)


def test_gate_failure_gives_no_verdict_and_h_b_descriptive_only():
    v = cpv(seeds_with([0.70, 0.70, 0.70]), n14_models())
    assert v['verdict'] is None and v['verdict_label'] == 'no verdict' and v['H_B']['descriptive_only'] is True
    assert v['H_B']['verdict'] is None and v['H_A']['gate_pass'] is False
    assert 'FAIL' in render_tables_p7(v) and 'gate failed' in render_tables_p7(v)


def test_seed_sensitive_and_budget_sensitive_flags():
    v = cpv(seeds_with([0.93, 0.80, 0.80]), n14_models())
    assert v['verdict'] is True                                   # mean 0.8433: 0.8 x 0.2433 = 0.195 <= 0.25
    assert v['sensitivities']['best_seed']['verdict'] is False and v['sensitivities']['seed_sensitive'] is True
    # published credit lifts G_R to G_ref = 0.2671 when the model is below it: ridge 0.80 (G_b 0.20) beats 0.8 x 0.16 but not 0.8 x 0.2671
    v2 = cpv(seeds_with([0.76, 0.76, 0.76]), n14_models(ridge=(0.80, 0.80, 0.80)))
    assert v2['verdict'] is True and v2['sensitivities']['published_credit']['verdict'] is False
    assert v2['sensitivities']['published_credit']['budget_sensitive'] is True
    assert v['sensitivities']['published_credit']['budget_sensitive'] is False


def test_fewer_than_two_finite_seeds_has_no_verdict():
    v = cpv(seeds_with([0.80, None, None]), n14_models())
    assert v['verdict'] is None and 'fewer than 2' in v['reason'] and v['seeds_defined'] == [str(V.SEEDS[0])]
    bad = seeds_with([0.80, 0.82, 0.80])
    bad[V.SEEDS[1]]['official_roc'] = {d: roc(0.8, undefined=True) for d in DESIGNS}      # a seed with an undefined metric
    bad[V.SEEDS[2]] = None
    v = cpv(bad, n14_models())
    assert v['verdict'] is None and v['seeds_defined'] == [str(V.SEEDS[0])]
    two = cpv(seeds_with([0.80, 0.82, None]), n14_models())
    assert two['verdict'] is True and two['primary']['r_t'] == pytest.approx(0.81)       # two defined seeds are enough
    assert 'No verdict' in render_tables_p7(v)


def test_undefined_design_is_dropped_from_the_design_macro():
    p = payload(V.SEEDS[0], 0.80)
    p['official_roc']['zero-riscy'] = roc(0.8, undefined=True)
    sv = V.seed_view(p)
    assert np.isnan(sv['design_values']['zero-riscy']) and sv['design_macro'] == pytest.approx(0.80)


def test_registered_verdict_shown_and_agreement():
    reg = {'H1': {'tasks': {'ir_drop': {'verdict': True, 'ratio': 3.0}}}}
    v = cpv(seeds_with([0.80] * 3), n14_models(), reg)
    assert v['registered_n14'] == {'available': True, 'verdict': True, 'ratio': 3.0, 'unchanged': True, 'p7_agrees': True}
    v = cpv(seeds_with([0.93] * 3), n14_models(), reg)
    assert v['registered_n14']['p7_agrees'] is False
    assert cpv(seeds_with([0.70] * 3), n14_models(), reg)['registered_n14']['p7_agrees'] is None


# -- finite-only NRMS and counts --------------------------------------------------------------------------------------------------
def test_nrms_finite_only_with_inf_and_nan_rows():
    rows = [{'id': 'a', 'design': 'D', 'nrms': 0.3}, {'id': 'b', 'design': 'D', 'nrms': float('inf')},
            {'id': 'c', 'design': 'D', 'nrms': float('nan')}, {'id': 'd', 'design': 'D', 'nrms': 0.5},
            {'id': 'e', 'design': 'E', 'nrms': 0.2}]
    n = nrms_view(rows)
    d = n['per_design']['D']
    assert d['mean_finite'] == pytest.approx(0.4) and d['n_undefined'] == 2 and d['undefined_ids'] == ['b', 'c'] and d['n_rows'] == 4
    assert d['mean_inf_to_0.05'] == pytest.approx((0.3 + 0.05 + 0.5) / 3)       # inf -> 0.05, nan still excluded
    assert n['design_macro_finite'] == pytest.approx((0.4 + 0.2) / 2) and n['pooled_finite'] == pytest.approx((0.3 + 0.5 + 0.2) / 3)
    assert n['n_undefined_total'] == 2
    allbad = nrms_view([{'id': 'a', 'design': 'D', 'nrms': float('inf')}])
    assert allbad['per_design']['D']['mean_finite'] is None and np.isnan(allbad['design_macro_finite'])


def test_undefined_nrms_and_no_positive_counts_reported_per_design_and_seed():
    rows = rows_for(IDS)
    rows[0]['nrms'] = float('inf')
    rows[1]['nrms'] = float('nan')
    p = payload(V.SEEDS[0], 0.8, nrms=rows)
    p['official_roc']['openc910-1'] = roc(0.8, n_samples=2, n_no_positive=1)
    seeds = {V.SEEDS[0]: p, V.SEEDS[1]: payload(V.SEEDS[1], 0.8), V.SEEDS[2]: payload(V.SEEDS[2], 0.8)}
    v = cpv(seeds, n14_models())
    k0, k1 = str(V.SEEDS[0]), str(V.SEEDS[1])
    assert v['undefined_nrms'][k0] == {'Vortex-large': 2, 'openc910-1': 0, 'zero-riscy': 0} and v['undefined_nrms'][k1]['Vortex-large'] == 0
    assert v['no_positive_rows'][k0] == {'Vortex-large': 0, 'openc910-1': 1, 'zero-riscy': 0} and v['no_positive_rows'][k1]['openc910-1'] == 0
    assert v['per_seed'][k0]['nrms']['per_design']['Vortex-large']['mean_finite'] is None            # both Vortex-large rows undefined
    assert '| n_no_positive |' in render_tables_p7(v)
    json.dumps(v)                                                                  # serialisable (no numpy scalars left)


def test_scored_ids_must_equal_registered_eval_ids():
    p = payload(V.SEEDS[0], 0.8)
    check_ids(p, n14_models(), 'x')
    p['eval_ids'] = {d: v[:1] for d, v in IDS.items()}
    with pytest.raises(ProtocolError, match='eval_ids differ'):
        check_ids(p, n14_models(), 'x')
    p = payload(V.SEEDS[0], 0.8)
    p['rows'] = p['rows'][:-1]
    with pytest.raises(ProtocolError, match='scored row ids'):
        check_ids(p, n14_models(), 'x')
    del p['eval_ids']
    with pytest.raises(ProtocolError, match='no eval_ids'):
        check_ids(p, n14_models(), 'x')


def test_header_mismatch_refused(tmp_path):
    p = payload(V.SEEDS[0], 0.8)
    p['seed'] = 1
    (tmp_path / f'models_ir_drop_A_s{V.SEEDS[0]}.json').write_text(json.dumps(p))
    with pytest.raises(ProtocolError, match='header says'):
        V.load_seed_files(tmp_path)


# -- refusal before the freeze, write_once ------------------------------------------------------------------------------------------
def _stage_files(tmp_path, areas=(0.80, 0.82, 0.78)):
    md, nd = tmp_path / 'p7', tmp_path / 'n14'
    md.mkdir()
    nd.mkdir()
    (nd / 'models_ir_drop.json').write_text(json.dumps(n14_models()))
    for s, a in zip(V.SEEDS, areas):
        (md / f'models_ir_drop_A_s{s}.json').write_text(json.dumps(payload(s, a)))
        (md / f'train_ir_drop_A_s{s}.json').write_text(json.dumps(train_rec()))
    return md, nd


def test_run_verdicts_refuses_before_the_freeze_and_writes_nothing(tmp_path):
    r = make_repo(tmp_path / 'repo')
    md, nd = _stage_files(tmp_path)
    with pytest.raises(ProtocolError):
        run_verdicts(md, nd, str(r / 'protocol'), r)
    write_freeze(r, p7_code_hash='1' * 64)
    with pytest.raises(ProtocolError, match='P7 code hash differs'):
        run_verdicts(md, nd, str(r / 'protocol'), r)
    assert not (md / 'verdicts_p7.json').exists() and not (md / 'tables_p7.md').exists()


def test_run_verdicts_is_write_once(tmp_path):
    r = make_repo(tmp_path / 'repo')
    write_freeze(r)
    md, nd = _stage_files(tmp_path)
    v = run_verdicts(md, nd, str(r / 'protocol'), r)
    assert v['verdict'] is True
    out = json.loads((md / 'verdicts_p7.json').read_text())
    assert out['verdict_label'] == 'TRUE' and out['label'].startswith('REGISTERED') and (md / 'tables_p7.md').read_text().startswith('# P7')
    before = (md / 'verdicts_p7.json').read_text()
    with pytest.raises(ResultExists):
        run_verdicts(md, nd, str(r / 'protocol'), r)
    assert (md / 'verdicts_p7.json').read_text() == before


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verdict_script_exits_2_before_the_freeze_and_after_a_first_run(tmp_path, capsys):
    mod = _load_script('p7_run_verdicts')
    r = make_repo(tmp_path / 'repo')
    md, nd = _stage_files(tmp_path)
    args = ['--models-dir', str(md), '--n14-dir', str(nd), '--bundle', str(r)]
    assert mod.main(args) == 2 and 'refused' in capsys.readouterr().out
    write_freeze(r)
    assert mod.main(args) == 0 and 'TRUE' in capsys.readouterr().out
    assert mod.main(args) == 2 and 'already exists' in capsys.readouterr().out


def test_export_script_refuses_the_evaluation_export_before_the_freeze(tmp_path, capsys):
    mod = _load_script('p7_run_export')
    out = tmp_path / 'export'
    assert mod.main(['--task', 'ir_drop', '--out', str(out), '--test', '--cache-dir', str(tmp_path / 'cache')]) == 2
    assert 'refused' in capsys.readouterr().out and not out.exists() and not (tmp_path / 'cache_eval').exists()


def test_aggregator_is_inside_the_p7_code_hash():
    names = {p.relative_to(guard.REPO_ROOT).as_posix() for p in guard.p7_files()}
    assert {'src/edalint_p7/verdicts.py', 'scripts/p7_run_verdicts.py'} <= names


# -- mechanical divergence rule (protocol_p7.md 12.10) --------------------------------------------------------------------------
def test_divergence_reason_rule():
    assert V.divergence_reason(train_rec()) is None
    assert V.divergence_reason(train_rec(skipped_steps=29999)) is None                     # skipped-step fraction is not a rule
    assert V.divergence_reason(None) == 'no train result'
    for bad in (float('nan'), float('inf'), float('-inf'), None, 'x'):
        assert 'last_block_loss' in V.divergence_reason(train_rec(last_block_loss=bad))
    assert 'final_params_finite' in V.divergence_reason(train_rec(final_params_finite=False))
    r = train_rec()
    del r['final_params_finite']
    assert 'final_params_finite' in V.divergence_reason(r)                                # a record without the field is not trusted


def test_diverged_seed_with_a_scored_file_is_excluded():
    seeds = seeds_with([0.80, 0.82, 0.20])                                                 # the diverged seed even has a (finite) scored file
    for bad in (train_rec(last_block_loss=float('nan')), train_rec(final_params_finite=False)):
        tr = {V.SEEDS[0]: train_rec(), V.SEEDS[1]: train_rec(), V.SEEDS[2]: bad}
        v = cpv(seeds, n14_models(), tr=tr)
        k = str(V.SEEDS[2])
        assert v['per_seed'][k]['defined'] is False and 'diverged' in v['per_seed'][k]['reason']
        assert v['seeds_defined'] == [str(V.SEEDS[0]), str(V.SEEDS[1])]
        assert v['primary']['r_t'] == pytest.approx(0.81) and v['verdict'] is True          # 0.20 does not enter the mean
        assert v['no_positive_rows'][k] is not None                                          # still reported


def test_missing_train_result_makes_the_seed_undefined():
    tr = {V.SEEDS[0]: train_rec(), V.SEEDS[1]: train_rec()}                                 # seed 3 has a scored file but no train result
    v = cpv(seeds_with([0.80, 0.82, 0.78]), n14_models(), tr=tr)
    assert v['per_seed'][str(V.SEEDS[2])]['reason'] == 'no train result' and v['seeds_defined'] == [str(V.SEEDS[0]), str(V.SEEDS[1])]
    v = cpv(seeds_with([0.80, 0.82, 0.78]), n14_models(), tr={})
    assert v['verdict'] is None and v['seeds_defined'] == []
    v = cpv(seeds_with([0.80, 0.82, None]), n14_models(), tr={V.SEEDS[0]: train_rec(), V.SEEDS[1]: train_rec(), V.SEEDS[2]: train_rec(last_block_loss=float('nan'))})
    assert 'no scored file' in v['per_seed'][str(V.SEEDS[2])]['reason'] and 'diverged' in v['per_seed'][str(V.SEEDS[2])]['reason']


def test_two_diverged_seeds_leave_no_verdict():
    tr = {V.SEEDS[0]: train_rec(), V.SEEDS[1]: train_rec(final_params_finite=False), V.SEEDS[2]: train_rec(last_block_loss=float('inf'))}
    v = cpv(seeds_with([0.80, 0.82, 0.78]), n14_models(), tr=tr)
    assert v['verdict'] is None and 'fewer than 2' in v['reason']


def test_run_verdicts_reads_train_files_and_excludes_a_diverged_seed(tmp_path):
    r = make_repo(tmp_path / 'repo')
    write_freeze(r)
    md, nd = _stage_files(tmp_path)
    (md / f'train_ir_drop_A_s{V.SEEDS[2]}.json').write_text(json.dumps(train_rec(last_block_loss=float('nan'))))
    v = run_verdicts(md, nd, str(r / 'protocol'), r)
    assert v['seeds_defined'] == [str(V.SEEDS[0]), str(V.SEEDS[1])]


def test_trainer_records_last_block_loss_and_final_params_finite():
    import torch
    from edalint_p7 import train as T
    class Src:
        designs = ['d'] * 32
        excluded_ids = []
        nonfinite_ids = []
        def __len__(self):
            return 32
        def get(self, k):
            g = np.random.default_rng(k)
            return g.random((3, 16, 16), dtype=np.float32), g.random((16, 16), dtype=np.float32)
    rec = T.train('congestion', Src(), 100, 1, 'fp32', 'cpu', log=lambda *a: None)
    assert rec['final_params_finite'] is True and np.isfinite(rec['last_block_loss']) and rec['last_block_loss'] == rec['loss_per_100_iters'][-1]
    with torch.no_grad():
        next(rec['_model'].parameters()).fill_(float('nan'))
    assert not all(torch.isfinite(v).all() for v in rec['_model'].state_dict().values() if v.is_floating_point())
