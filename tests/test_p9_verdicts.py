"""P9 mechanical verdict rules on hand-made result fragments."""
from edalint_p9 import verdicts as V


def _ev(cov_id, cov_xd, gauss=0.7):
    blk = lambda c: {'0.1': {'coverage': c, 'width': 0.1, 'coverage_high': c - 0.1}, 'calibration_error': 0.02}   # noqa: E731
    return {'crc_norm|id': blk(cov_id), 'crc_norm|xd': blk(cov_xd), 'crc_abs|id': blk(cov_id), 'crc_abs|xd': blk(cov_xd),
            'ens_gauss|-': blk(gauss)}


def test_uq_rules():
    res = {'evaluation': {'T_id': _ev(0.905, 0.95, gauss=0.6), 'a': _ev(0.70, 0.88), 'b': _ev(0.80, 0.91), 'c': _ev(0.89, 0.93)}}
    v = V.uq(res, ['a', 'b', 'c'])
    assert v['U1_id_validity'] == {'crc_abs': True, 'crc_norm': True}
    assert v['U2_ood_undercoverage']['designs_below_line'] == 2 and v['U2_ood_undercoverage']['verdict'] is True
    assert v['U3_xd_repairs']['verdict'] is True
    assert v['U4_raw_ensemble_undercovers']['verdict'] is True
    assert abs(v['per_design']['a']['ood_degradation_id'] - (0.70 - 0.905)) < 1e-12
    v1 = V.uq({'evaluation': {'T_id': _ev(0.85, 0.9), 'h': _ev(0.86, 0.95)}}, ['h'])
    assert v1['U1_id_validity']['crc_norm'] is False and v1['U2_ood_undercoverage']['verdict'] is False


def test_leakage_rules():
    leak = {'L001': {'flagged_pairs_confirmation': [{'query': 'q1', 'confirmed': True}, {'query': 'q2', 'confirmed': False}]},
            'L002': {'per_design': {'zr': {'material': True, 'S_same': .9, 'S_other': .5, 'reference_q25_within_design': .8},
                                    'vl': {'material': True, 'S_same': .9, 'S_other': .5, 'reference_q25_within_design': .8},
                                    'oc': {'material': False, 'S_same': None, 'S_other': .5, 'reference_q25_within_design': .8}}}}
    assert V.lk1(leak, 50)['verdict'] == 'material' and V.lk1(leak, 1000)['verdict'] == 'present'
    l2 = V.lk2(leak, {'zr': 950, 'vl': 30, 'oc': 20})
    assert l2['zr']['severity'] == 'material' and l2['vl']['severity'] == 'benign' and l2['oc']['severity'] is None
    splits = {'x': {'files': {'test': {'rows': 100}}, 'pairs': [{'splits': ['train', 'test'], 'design_overlap': ['d'],
                                                                 'rows_in_overlap': {'train': 5, 'test': 3}}]},
              'y': {'files': {'test': {'rows': 100}}, 'pairs': [{'splits': ['train', 'val'], 'design_overlap': ['d'],
                                                                 'rows_in_overlap': {'train': 5, 'val': 3}}]}}
    v = V.lk3(splits)
    assert v['x']['verdict'] == 'material' and v['y']['verdict'] == 'none'
