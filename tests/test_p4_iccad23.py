"""Mechanical verdicts of the registered ICCAD'23 run, on hand-built result dictionaries (no data read)."""
import numpy as np

from edalint import p4_iccad23 as P

IDS = [f'hidden/testcase{n}' for n in P.HIDDEN_ORDER]
DES = {i: f'd{k // 2}' for k, i in enumerate(IDS)}


def _rows(mae, f1=0.0):
    return [{'id': i, 'design': DES[i], 'mae_mV': m, 'f1_golden_max': f1, 'f1_golden_max_ge': f1, 'f1_top10pct': f1, 'seconds': 0.1}
            for i, m in zip(IDS, mae)]


def _h3(material=False):
    e3 = {'code': 'E003', 'findings': ([{'severity': 'error', 'message': '1 near-duplicate pair(s) ACROSS splits'}] if material else [])}
    return {'n_hidden_designs': 5, 'twin_structure': {}, 'identity_hidden_vs_train': [], 'identity_hidden_vs_hidden': [],
            'checks_train_vs_hidden': {'checks': [e3]}, 'checks_hidden': {'checks': [{'code': 'E006', 'metrics': {}, 'findings': []}]}}


def _models(best_skill_mae):
    base = 1.0
    res = {n: {'rows': _rows([base] * 10)} for n in P.BASELINE_ORDER}
    res['zero'] = {'rows': _rows([2.0] * 10)}
    res['ridge'] = {'rows': _rows([best_skill_mae] * 10)}
    return {'eval_ids': IDS, 'results': res}


def _solver(mae, secs):
    return {'rows': [{'id': i, 'design': DES[i], 'mae_mV': mae, 'f1_golden_max': 1.0, 'f1_golden_max_ge': 1.0, 'f1_top10pct': 1.0,
                      'seconds_end_to_end': secs, 'seconds_solve_only': 0.2} for i in IDS]}


def test_h1_false_when_baseline_far_below_published():
    v = P.compute_verdicts(_models(0.5), _solver(0.003, 7.0), _h3())
    assert v['mean_map_mae_mV'] == 1.0
    assert abs(v['ours']['ridge']['skill_mae'] - 0.5) < 1e-12
    assert v['H1']['best_non_solver_baseline'] == 'ridge' and v['H1']['verdict'] is False


def test_h1_true_at_threshold():
    # WACA skill = 1 - 0.0524/1.0 = 0.9476; threshold 0.8x = 0.75808 -> baseline MAE 0.24 gives skill 0.76
    assert P.compute_verdicts(_models(0.24), _solver(0.003, 7.0), _h3())['H1']['verdict'] is True
    assert P.compute_verdicts(_models(0.25), _solver(0.003, 7.0), _h3())['H1']['verdict'] is False


def test_h2_needs_both_conditions():
    assert P.compute_verdicts(_models(0.5), _solver(0.003, 14.77), _h3())['H2']['verdict'] is True
    assert P.compute_verdicts(_models(0.5), _solver(0.003, 14.78), _h3())['H2']['verdict'] is False
    assert P.compute_verdicts(_models(0.5), _solver(0.0524, 5.0), _h3())['H2']['verdict'] is False   # not strictly below


def test_h3_material_only_with_cross_split_duplicate():
    assert P.compute_verdicts(_models(0.5), _solver(0.003, 7.0), _h3(False))['H3']['verdict'] == 'BENIGN'
    assert P.compute_verdicts(_models(0.5), _solver(0.003, 7.0), _h3(True))['H3']['verdict'] == 'MATERIAL'
    h = _h3(False)
    h['identity_hidden_vs_train'] = [{'channel': 'current', 'non_constant': True, 'hidden': ['a'], 'train': ['b']}]
    assert P.compute_verdicts(_models(0.5), _solver(0.003, 7.0), h)['H3']['verdict'] == 'MATERIAL'
    h['identity_hidden_vs_train'] = [{'channel': 'pdn_density', 'non_constant': False, 'hidden': ['a'], 'train': ['b']}]
    assert P.compute_verdicts(_models(0.5), _solver(0.003, 7.0), h)['H3']['verdict'] == 'BENIGN'


def test_tables_render():
    v = P.compute_verdicts(_models(0.5), _solver(0.003, 7.0), _h3())
    assert 'Skill_MAE' in P.render_tables(v)
