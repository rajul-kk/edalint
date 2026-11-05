"""P9 mechanical verdicts (protocol_p9.md section 6) from the run-once result files. No judgement here: every rule is the protocol's."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

NOMINAL = 0.9
U1_TOL = 0.03
U2_LINE = 0.85
U3_MARGIN = 0.02
U4_LINE = 0.85
LK1_RATE = 0.01
LK2_SHARE = 0.10
LK3_RATE = 0.01
PRIMARY = 'crc_norm'


def _cov(ev: dict, key: str, a: str = '0.1') -> float:
    return float(ev.get(key, {}).get(a, {}).get('coverage', float('nan')))


def lk1(leak: dict, n_test_rows: int) -> dict:
    l1 = leak['L001']
    conf = [c for c in l1['flagged_pairs_confirmation'] if c['confirmed']]
    rows = {c['query'] for c in conf}
    rate = len(rows) / n_test_rows if n_test_rows else float('nan')
    return {'n_flagged': len(l1['flagged_pairs_confirmation']), 'n_confirmed_rows': len(rows), 'rate': rate,
            'verdict': 'material' if rate >= LK1_RATE else ('present' if rows else 'not found (lower bound)')}


def lk2(leak: dict, rows_per_design: Dict[str, int]) -> dict:
    tot = sum(rows_per_design.values())
    out = {}
    for d, r in leak['L002']['per_design'].items():
        share = rows_per_design.get(d, 0) / tot if tot else 0.0
        sev = None
        if r['material']:
            sev = 'material' if share >= LK2_SHARE else 'benign'
        out[d] = {'proximity_material': r['material'], 'share_of_test_rows': share, 'severity': sev, 'S_same': r['S_same'],
                  'S_other': r['S_other'], 'q25_reference': r['reference_q25_within_design']}
    return out


def lk3(splits: dict) -> dict:
    out = {}
    for name, res in splits.items():
        test_rows = res['files'].get('test', {}).get('rows', 0)
        bad = []
        for p in res['pairs']:
            if 'test' in p['splits'] and p['design_overlap']:
                other = [s for s in p['splits'] if s != 'test'][0]
                n = p['rows_in_overlap']['test']
                bad.append({'with': other, 'designs': p['design_overlap'], 'test_rows': n, 'rate': n / test_rows if test_rows else float('nan')})
        worst = max((b['rate'] for b in bad), default=0.0)
        out[name] = {'overlaps_with_test': bad, 'verdict': 'material' if worst >= LK3_RATE else ('present' if bad else 'none')}
    return out


def uq(res: dict, test_keys: List[str]) -> dict:
    ev = res['evaluation']
    tid = ev['T_id']
    u1 = {m: abs(_cov(tid, f'{m}|id') - NOMINAL) <= U1_TOL for m in ('crc_abs', 'crc_norm')}
    per = {d: {'id': _cov(ev[d], f'{PRIMARY}|id'), 'xd': _cov(ev[d], f'{PRIMARY}|xd'), 'abs_id': _cov(ev[d], 'crc_abs|id'),
               'abs_xd': _cov(ev[d], 'crc_abs|xd'), 'ens_gauss': _cov(ev[d], 'ens_gauss|-'),
               'width_id': ev[d].get(f'{PRIMARY}|id', {}).get('0.1', {}).get('width'),
               'width_xd': ev[d].get(f'{PRIMARY}|xd', {}).get('0.1', {}).get('width'),
               'coverage_high_id': ev[d].get(f'{PRIMARY}|id', {}).get('0.1', {}).get('coverage_high'),
               'calibration_error_id': ev[d].get(f'{PRIMARY}|id', {}).get('calibration_error'),
               'calibration_error_xd': ev[d].get(f'{PRIMARY}|xd', {}).get('calibration_error'),
               'ood_degradation_id': _cov(ev[d], f'{PRIMARY}|id') - _cov(tid, f'{PRIMARY}|id')} for d in test_keys}
    below = sum(1 for d in test_keys if per[d]['id'] < U2_LINE)
    need = 2 if len(test_keys) >= 3 else 1
    gap_id = float(np.mean([abs(per[d]['id'] - NOMINAL) for d in test_keys]))
    gap_xd = float(np.mean([abs(per[d]['xd'] - NOMINAL) for d in test_keys]))
    macro_xd = float(np.mean([per[d]['xd'] for d in test_keys]))
    return {'U1_id_validity': u1, 'U2_ood_undercoverage': {'designs_below_line': below, 'needed': need, 'verdict': below >= need},
            'U3_xd_repairs': {'mean_abs_gap_id': gap_id, 'mean_abs_gap_xd': gap_xd, 'macro_cov_xd': macro_xd,
                              'verdict': bool(gap_xd <= gap_id - U3_MARGIN and macro_xd >= U2_LINE)},
            'U4_raw_ensemble_undercovers': {'T_id_coverage': _cov(tid, 'ens_gauss|-'), 'verdict': _cov(tid, 'ens_gauss|-') < U4_LINE},
            'per_design': per, 'T_id': {'crc_norm_id': _cov(tid, 'crc_norm|id'), 'crc_abs_id': _cov(tid, 'crc_abs|id'),
                                        'width_id': tid.get('crc_norm|id', {}).get('0.1', {}).get('width')}}
