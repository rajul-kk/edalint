"""Registered runs for ICCAD'23 Problem C (protocol/protocol_iccad23.md). This module is part of the hashed code.

Stages (each writes ONE file under results/iccad23/ and refuses to overwrite; every stage first requires protocol/FROZEN.json with a
matching code hash and src/protocol/scripts/tests matching HEAD):

  python -m edalint.p4_iccad23 models     fit the 9 baselines on all train (100 fake + 10 visible), score on the 10 hidden cases
  python -m edalint.p4_iccad23 solver     exact SPICE solve (cubic rasterisation) on the 10 hidden cases, timed
  python -m edalint.p4_iccad23 h3         H3 data-structure checks (E001-E006 on the hidden split, E003/E004 across train and hidden,
                                          explicit exact-identity check of every input channel and label)
  python -m edalint.p4_iccad23 verdicts   per-case / per-design / mean tables, Skill_MAE, H1, H2, H3 (reads the three result files)

Crash rule (protocol section 9): a stage that crashed BEFORE its file was written may be re-launched once per crash and the
re-launch is logged in the protocol's deviation log; a stage is never re-run because of its content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .baselines import BASELINES, stream
from .baselines.evaluate import case_metrics
from .baselines.zoo import GREEN_EDGES, GreenSuperposition, KNNMap, PixelGBM, PixelRidge
from .checks import Options
from .core import ProtocolError
from .datasets import ICCAD23Dataset
from .protocol import REPO_DIR, require_frozen
from .runner import run_checks, write_once, write_text_once

OUT_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'iccad23'
HIDDEN_ORDER = [7, 8, 9, 10, 13, 14, 15, 16, 19, 20]

# Frozen baseline hyperparameters (chosen by leave-one-twin-pair-out CV on the train split in P4 development;
# results/dev/iccad23_hparam_cv_p4dev.json). Recorded here and in the protocol; the registered run builds the baselines from THIS table.
FROZEN_HPARAMS: Dict[str, Dict[str, Any]] = {
    'zero': {}, 'mean_map': {}, 'scalar': {},
    'ridge': {'alpha': 0.01},
    'knn': {'k': 3},
    'effdist': {'window': 9, 'degree': 1},
    'effdist_deg2': {},
    'green': {'edges': 'medium'},
    'gbm': {'max_iter': 300, 'max_depth': 6},
}
BASELINE_ORDER = ['zero', 'mean_map', 'scalar', 'ridge', 'knn', 'effdist', 'effdist_deg2', 'green', 'gbm']
SOLVER_INTERP = 'cubic'                    # frozen from results/dev/iccad23_spice_raster_p4dev.json
SOLVER_METHOD = 'direct'


def make_baseline(name: str):
    h = FROZEN_HPARAMS[name]
    if name == 'ridge':
        return PixelRidge(alpha=h['alpha'])
    if name == 'knn':
        return KNNMap(k=h['k'])
    if name == 'green':
        return GreenSuperposition(edges=GREEN_EDGES[h['edges']])
    if name == 'gbm':
        return PixelGBM(max_iter=h['max_iter'], max_leaf_nodes=None, max_depth=h['max_depth'])
    return BASELINES[name]()


# Published numbers on the hidden 10 (copied from docs/published_numbers.md sections 1.2-1.4 at freeze; see protocol section 8).
# Per-case lists are in the order HIDDEN_ORDER; None = failed case.
PUBLISHED: Dict[str, Dict[str, Any]] = {
    'contest_1st (cadc0009, xlsx)': {'mae': 0.1347, 'f1': 0.4549, 'runtime_s': 14.772, 'note': 'official; all 10 valid',
        'mae_cases': [0.0656, 0.0815, 0.0406, 0.0659, 0.2068, 0.4215, 0.0968, 0.1601, 0.0905, 0.1180],
        'f1_cases': [0.7828, 0.8164, 0.5889, 0.5320, 0, 0, 0.0877, 0.5286, 0.5014, 0.7111]},
    'contest_2nd (cadc1023, xlsx)': {'mae': 0.1498, 'f1': 0.4550, 'runtime_s': 3.042, 'note': 'official; all 10 valid',
        'mae_cases': [0.0776, 0.1132, 0.0727, 0.1142, 0.1255, 0.2321, 0.1920, 0.3440, 0.1202, 0.1069],
        'f1_cases': [0.5604, 0.8022, 0.5505, 0.1490, 0.6735, 0.0984, 0, 0.4808, 0.4942, 0.7413]},
    'contest_3rd (cadc1020, xlsx)': {'mae': 0.1069, 'f1': 0.5186, 'runtime_s': None,
        'note': 'FAILED cases 19 and 20; MAE and F1 are means over the 8 valid cases (7-16); xlsx F1 over 10 cases with 0 for the failures is 0.4143',
        'mae_cases': [0.0749, 0.1141, 0.0512, 0.1003, 0.0901, 0.2115, 0.0521, 0.1609, None, None],
        'f1_cases': [0.8657, 0.8742, 0.4720, 0.6468, 0, 0, 0.7363, 0.5478, None, None]},
    'CFIRSTNET (2502.12168)': {'mae': 0.0533, 'f1': 0.723, 'runtime_s': 0.409, 'note': 'inputs resized to 384x384; Xeon E3-1230 v6 + Quadro P4000',
        'mae_cases': [0.0177, 0.0257, 0.0278, 0.0459, 0.0774, 0.1895, 0.0353, 0.0763, 0.0187, 0.0191],
        'f1_cases': [0.923, 0.916, 0.526, 0.464, 0.680, 0.678, 0.733, 0.785, 0.752, 0.773]},
    'WACA-UNet (2507.19197)': {'mae': 0.0524, 'f1': 0.778, 'runtime_s': 1.390, 'note': '384x384 Lanczos resize; checkpoint by best validation F1; CPU model not stated',
        'mae_cases': [0.0147, 0.0215, 0.0241, 0.0402, 0.0861, 0.1518, 0.0549, 0.0734, 0.0366, 0.0205],
        'f1_cases': [0.953, 0.964, 0.721, 0.575, 0.561, 0.667, 0.749, 0.865, 0.817, 0.910]},
    'LMM-IR (2511.12581)': {'mae': 0.1356, 'f1': 0.583, 'runtime_s': 3.05, 'note': '512x512; H100',
        'mae_cases': [0.063, 0.084, 0.042, 0.071, 0.152, 0.324, 0.149, 0.333, 0.074, 0.064],
        'f1_cases': [0.72, 0.84, 0.47, 0.60, 0.52, 0.44, 0.54, 0.55, 0.61, 0.54]},
    'GL-Inception (2406.06541)': {'mae': 0.101, 'f1': 0.62, 'runtime_s': 3.37, 'note': 'includes feature extraction',
        'mae_cases': [0.081, 0.104, 0.057, 0.088, 0.121, 0.220, 0.066, 0.163, 0.047, 0.065],
        'f1_cases': [0.69, 0.63, 0.72, 0.34, 0.68, 0.43, 0.76, 0.40, 0.70, 0.80]},
    'AttUNet (2408.03292)': {'mae': 0.110, 'f1': 0.520, 'runtime_s': 5.37,
        'note': 'MAE as printed (per-case mean 0.1172); F1 rule DIFFERS (top 10% of pixels), not the contest rule',
        'mae_cases': [0.065, 0.081, 0.067, 0.090, 0.188, 0.078, 0.115, 0.338, 0.082, 0.068],
        'f1_cases': [0.469, 0.428, 0.169, 0.536, 0.895, 0.881, 0.700, 0.634, 0.330, 0.156]},
}
BEST_PUBLISHED = 'WACA-UNet (2507.19197)'
H1_FRACTION = 0.8
H2_WINNER_RUNTIME_S = 14.77


# -- guards ---------------------------------------------------------------------------------------------------------------------
def _preflight(stage: str):
    info = require_frozen(None, what=f'registered stage {stage}', dataset='iccad23')
    # only code, protocol, scripts and tests must match HEAD (other tracked files, e.g. download manifests written by unrelated jobs, may differ;
    # provenance records every tracked difference anyway)
    r = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no', '--', 'src', 'protocol', 'scripts', 'tests'],
                       capture_output=True, text=True, cwd=REPO_DIR)
    dirty = r.stdout.strip()
    if dirty:
        raise ProtocolError(f'refused: files differ from HEAD ({dirty.splitlines()[:5]}); commit before a registered run')
    return info


def _hidden_rank(sid: str) -> int:
    return HIDDEN_ORDER.index(int(sid.rsplit('testcase', 1)[1]))


def _test_ids(ds) -> List[str]:
    return sorted(ds.ids('test'), key=_hidden_rank)


# -- stages ---------------------------------------------------------------------------------------------------------------------
def stage_models(out_dir: Path):
    _preflight('models')
    ds = ICCAD23Dataset()
    train, test = ds.ids('train'), _test_ids(ds)
    assert len(train) == 110 and len(test) == 10, (len(train), len(test))
    res: Dict[str, Any] = {}
    for name in BASELINE_ORDER:
        b = make_baseline(name)
        t0 = time.perf_counter()
        b.fit(stream(ds, train))
        fit_s = time.perf_counter() - t0
        rows = _eval_ids(b, ds, test)
        res[name] = {'hparams': FROZEN_HPARAMS[name], 'fit_seconds': fit_s, 'rows': rows}
        print(name, 'fit', round(fit_s, 1), 's', [round(r['mae_mV'], 4) for r in rows], flush=True)
    write_once(out_dir / 'models.json', {'label': 'REGISTERED', 'fit_ids': train, 'eval_ids': test, 'native_resolution': True, 'results': res})


def _eval_ids(b, ds, ids):
    rows = []
    for sid in ids:
        s = ds.get(sid)
        t0 = time.perf_counter()
        pred = b.predict(s)
        dt = time.perf_counter() - t0
        row = {'id': sid, 'design': s.design, 'shape': list(s.shape), 'seconds': dt, 'finite': bool(np.isfinite(pred).all())}
        row.update(case_metrics(ds, np.nan_to_num(pred), s.label))
        rows.append(row)
    return rows


def stage_solver(out_dir: Path):
    _preflight('solver')
    from .spice import exact_ir_drop
    ds = ICCAD23Dataset()
    # untimed warm-up on a TRAIN case (lazy imports, allocator): the first scipy.interpolate call is not representative
    warm = ds.get('real/testcase11')
    exact_ir_drop(warm.meta['netlist'], warm.shape, method=SOLVER_METHOD, interp=SOLVER_INTERP)
    rows = []
    for sid in _test_ids(ds):
        s = ds.get(sid)
        t0 = time.perf_counter()
        m, info = exact_ir_drop(s.meta['netlist'], s.shape, method=SOLVER_METHOD, interp=SOLVER_INTERP)
        dt = time.perf_counter() - t0
        row = {'id': sid, 'design': s.design, 'shape': list(s.shape), 'seconds_end_to_end': dt, 'seconds_parse': info['seconds_parse'],
               'seconds_assemble_prepare': info['seconds_assemble'] + info['seconds_prepare'], 'seconds_solve_only': info['seconds_solve'],
               'seconds_rasterize': info['seconds_rasterize'], 'rel_residual': info['rel_residual'], 'n_nodes': info['n_nodes'],
               'n_floating_merged_nodes': info['n_floating_merged_nodes'], 'finite': bool(np.isfinite(m).all()),
               'max_abs_diff_to_label_mV': float(np.abs(m - s.label).max() * 1e3)}
        row.update(case_metrics(ds, np.nan_to_num(m), s.label))
        rows.append(row)
        print(sid, round(dt, 2), 's', round(row['mae_mV'], 5), flush=True)
    write_once(out_dir / 'solver.json', {'label': 'REGISTERED', 'method': SOLVER_METHOD, 'interp': SOLVER_INTERP,
                                         'timing_definition': 'end-to-end = perf_counter around exact_ir_drop(): netlist parse + sparse LU solve + '
                                                              'cubic rasterisation to the native grid; excludes reading the CSV maps. '
                                                              'solve-only = the sparse solve call only.',
                                         'rows': rows})


def _chan_hash(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes() + str(a.shape).encode()).hexdigest()


def stage_h3(out_dir: Path):
    _preflight('h3')
    ds = ICCAD23Dataset()
    test, train = _test_ids(ds), ds.ids('train')
    # 1. generic checks through the tool
    rep_test = run_checks(ds, ['E001', 'E002', 'E005', 'E006'], Options(split='test', max_samples=None))
    rep_all = run_checks(ds, ['E003', 'E004'], Options(split='all', max_samples=None, dup_max_samples=800))
    # 2. explicit exact-identity check (every input channel and label, hidden vs every train case, and hidden vs hidden)
    tr_hash: Dict[str, List[str]] = {}
    te_hash: Dict[str, List[str]] = {}
    for split_ids, tab in ((train, tr_hash), (test, te_hash)):
        for sid in split_ids:
            s = ds.get(sid)
            for nm, a in list(s.inputs.items()) + [('label', s.label)]:
                nonconst = bool(np.ptp(a) > 0)
                tab.setdefault(f'{nm}|{_chan_hash(a)}|{int(nonconst)}', []).append(sid)
    hidden_vs_train, hidden_vs_hidden = [], []
    for k, ids in te_hash.items():
        nm, h, nonconst = k.split('|')
        if k in tr_hash:
            hidden_vs_train.append({'channel': nm, 'non_constant': bool(int(nonconst)), 'hidden': ids, 'train': tr_hash[k]})
        if len(ids) > 1:
            hidden_vs_hidden.append({'channel': nm, 'non_constant': bool(int(nonconst)), 'hidden': ids})
    twins = {}
    for sid in test:
        m = ds.meta(sid)
        twins.setdefault(m['design_id'], []).append(sid)
    hot = []
    for sid in test:
        L = ds.get(sid).label
        thr = 0.9 * float(L.max())
        hot.append({'id': sid, 'shape': list(L.shape), 'label_max_mV': float(L.max() * 1e3), 'label_mean_mV': float(L.mean() * 1e3),
                    'n_hotspot_strict': int((L > thr).sum()), 'n_hotspot_ge': int((L >= thr).sum()), 'base_rate_strict': float((L > thr).mean())})
    write_once(out_dir / 'h3.json', {'label': 'REGISTERED', 'checks_hidden': rep_test.to_dict(), 'checks_train_vs_hidden': rep_all.to_dict(),
                                     'twin_structure': twins, 'n_hidden_designs': len(twins),
                                     'identity_hidden_vs_train': hidden_vs_train, 'identity_hidden_vs_hidden': hidden_vs_hidden,
                                     'hotspot_stats': hot})


# -- verdicts -------------------------------------------------------------------------------------------------------------------
def _load(out_dir: Path, name: str):
    return json.loads((out_dir / name).read_text(encoding='utf-8'))


def _f(x, nd=4):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def compute_verdicts(models: dict, solver: dict, h3: dict) -> dict:
    ids = models['eval_ids']
    design_of = {r['id']: r['design'] for r in models['results']['zero']['rows']}
    designs = sorted(set(design_of.values()))
    out: Dict[str, Any] = {'case_ids': ids, 'designs': designs}
    tab: Dict[str, Dict[str, Any]] = {}
    series = {n: models['results'][n]['rows'] for n in BASELINE_ORDER}
    series['spice'] = solver['rows']
    for n, rows in series.items():
        rows = sorted(rows, key=lambda r: ids.index(r['id']))
        t = {'mae_cases': [r['mae_mV'] for r in rows]}
        for k in ('f1_golden_max', 'f1_golden_max_ge', 'f1_top10pct'):
            t[k + '_cases'] = [r[k] for r in rows]
        per_design = {}
        for d in designs:
            sel = [r for r in rows if design_of[r['id']] == d]
            per_design[d] = {'cases': [r['id'] for r in sel], 'mae': float(np.mean([r['mae_mV'] for r in sel])),
                             'f1': float(np.mean([r['f1_golden_max'] for r in sel]))}
        t['per_design'] = per_design
        t['mean_mae_mV'] = float(np.mean(t['mae_cases']))
        t['median_mae_mV'] = float(np.median(t['mae_cases']))
        for k in ('f1_golden_max', 'f1_golden_max_ge', 'f1_top10pct'):
            t['mean_' + k] = float(np.mean(t[k + '_cases']))
        t['n_f1_zero_strict'] = int(sum(v == 0 for v in t['f1_golden_max_cases']))
        t['n_f1_zero_ge'] = int(sum(v == 0 for v in t['f1_golden_max_ge_cases']))
        t['n_f1_zero_top10pct'] = int(sum(v == 0 for v in t['f1_top10pct_cases']))
        dm = [per_design[d]['mae'] for d in designs]
        t['design_mae_range'] = [min(dm), max(dm)]
        tab[n] = t
    ref = tab['mean_map']['mean_mae_mV']
    for n, t in tab.items():
        t['skill_mae'] = 1.0 - t['mean_mae_mV'] / ref
        t['sign_count_beats_mean_map_designs'] = int(sum(t['per_design'][d]['mae'] < tab['mean_map']['per_design'][d]['mae'] for d in designs))
    pub = {}
    for n, p in PUBLISHED.items():
        refp = ref
        if n.startswith('contest_3rd'):          # 8 valid cases only: reference = our mean_map on the same 8 cases
            refp = float(np.mean(tab['mean_map']['mae_cases'][:8]))
        pub[n] = {'mae': p['mae'], 'f1': p['f1'], 'runtime_s': p['runtime_s'], 'skill_mae': 1.0 - p['mae'] / refp, 'reference_mean_map_mae': refp,
                  'note': p['note']}
    out.update({'ours': tab, 'published': pub, 'mean_map_mae_mV': ref})
    # H1
    non_solver = [n for n in BASELINE_ORDER]
    best_n = max(non_solver, key=lambda n: tab[n]['skill_mae'])
    best_skill = tab[best_n]['skill_mae']
    pub_skill = pub[BEST_PUBLISHED]['skill_mae']
    out['H1'] = {'best_non_solver_baseline': best_n, 'its_skill_mae': best_skill, 'best_published': BEST_PUBLISHED, 'its_skill_mae_published': pub_skill,
                 'threshold': H1_FRACTION * pub_skill, 'ratio': best_skill / pub_skill, 'verdict': bool(best_skill >= H1_FRACTION * pub_skill)}
    # H2
    sp = tab['spice']
    best_pub_mae = min(p['mae'] for p in PUBLISHED.values() if p['mae'] is not None)
    e2e = [r['seconds_end_to_end'] for r in solver['rows']]
    so = [r['seconds_solve_only'] for r in solver['rows']]
    out['H2'] = {'solver_mean_mae_mV': sp['mean_mae_mV'], 'best_published_mae_mV': best_pub_mae, 'mae_condition': bool(sp['mean_mae_mV'] < best_pub_mae),
                 'median_end_to_end_s': float(np.median(e2e)), 'mean_end_to_end_s': float(np.mean(e2e)), 'max_end_to_end_s': float(np.max(e2e)),
                 'median_solve_only_s': float(np.median(so)), 'mean_solve_only_s': float(np.mean(so)), 'winner_runtime_s': H2_WINNER_RUNTIME_S,
                 'runtime_condition': bool(float(np.median(e2e)) <= H2_WINNER_RUNTIME_S),
                 'verdict': bool(sp['mean_mae_mV'] < best_pub_mae and float(np.median(e2e)) <= H2_WINNER_RUNTIME_S),
                 'solver_mean_f1': sp['mean_f1_golden_max'], 'solver_max_case_mae_mV': float(np.max(sp['mae_cases']))}
    # H3
    ident_input = [x for x in h3['identity_hidden_vs_train'] if x['channel'] != 'label' and x['non_constant']]
    ident_label = [x for x in h3['identity_hidden_vs_train'] if x['channel'] == 'label' and x['non_constant']]
    e3 = next(c for c in h3['checks_train_vs_hidden']['checks'] if c['code'] == 'E003')
    cross_e3 = [f['message'] for f in e3['findings'] if f['severity'] == 'error' and 'ACROSS' in f['message']]
    material = bool(ident_input or ident_label or cross_e3)
    e6 = next(c for c in h3['checks_hidden']['checks'] if c['code'] == 'E006')
    out['H3'] = {'n_hidden_designs': h3['n_hidden_designs'], 'twin_structure': h3['twin_structure'],
                 'exact_input_channel_duplicates_hidden_vs_train': ident_input, 'exact_label_duplicates_hidden_vs_train': ident_label,
                 'E003_cross_split_errors': cross_e3, 'hidden_vs_hidden_shared_channels': h3['identity_hidden_vs_hidden'],
                 'E006_metrics': e6['metrics'], 'E006_findings': [f['message'] for f in e6['findings']], 'verdict': 'MATERIAL' if material else 'BENIGN'}
    return out


def render_tables(v: dict) -> str:
    ids, designs, ours = v['case_ids'], v['designs'], v['ours']
    names = [n for n in BASELINE_ORDER] + ['spice']
    L = ['## Per case (hidden set; MAE in mV, F1 official strict rule)', '', '| baseline | ' + ' | '.join(i.split('testcase')[1] for i in ids) + ' | mean |', '|---|' + '---|' * (len(ids) + 1)]
    for n in names:
        L.append(f'| {n} MAE | ' + ' | '.join(_f(x) for x in ours[n]['mae_cases']) + f' | {_f(ours[n]["mean_mae_mV"])} |')
    for n in names:
        L.append(f'| {n} F1 | ' + ' | '.join(_f(x, 3) for x in ours[n]['f1_golden_max_cases']) + f' | {_f(ours[n]["mean_f1_golden_max"], 3)} |')
    L += ['', '## Per design (mean of the twin pair) and overall', '', '| baseline | ' + ' | '.join(designs) + ' | overall MAE | range over designs | Skill_MAE | designs beating mean_map (of 5) |',
          '|---|' + '---|' * (len(designs) + 4)]
    for n in names:
        t = ours[n]
        L.append(f'| {n} | ' + ' | '.join(f'{_f(t["per_design"][d]["mae"])} / {_f(t["per_design"][d]["f1"], 3)}' for d in designs) +
                 f' | {_f(t["mean_mae_mV"])} | {_f(t["design_mae_range"][0])} to {_f(t["design_mae_range"][1])} | {_f(t["skill_mae"], 3)} | {t["sign_count_beats_mean_map_designs"]} |')
    L += ['', '(per-design cells: MAE mV / F1)', '', '## F1 variants (mean over 10 cases) and number of cases with F1 = 0', '',
          '| baseline | F1 strict (primary) | F1 >= | F1 top-10% | #F1=0 strict | #F1=0 >= | #F1=0 top-10% |', '|---|---|---|---|---|---|---|']
    for n in names:
        t = ours[n]
        L.append(f'| {n} | {_f(t["mean_f1_golden_max"], 3)} | {_f(t["mean_f1_golden_max_ge"], 3)} | {_f(t["mean_f1_top10pct"], 3)} | {t["n_f1_zero_strict"]} | {t["n_f1_zero_ge"]} | {t["n_f1_zero_top10pct"]} |')
    L += ['', '## Published comparison (hidden 10; Skill_MAE reference = our mean_map MAE %s mV)' % _f(v['mean_map_mae_mV']), '',
          '| model | MAE mV | F1 | runtime s | Skill_MAE | note |', '|---|---|---|---|---|---|']
    for n, p in v['published'].items():
        L.append(f'| {n} | {_f(p["mae"])} | {_f(p["f1"], 3)} | {_f(p["runtime_s"], 3)} | {_f(p["skill_mae"], 3)} | {p["note"]} |')
    for n in ('ridge', 'green', 'gbm', 'spice'):
        t = ours[n]
        L.append(f'| ours: {n} | {_f(t["mean_mae_mV"])} | {_f(t["mean_f1_golden_max"], 3)} | | {_f(t["skill_mae"], 3)} | |')
    return '\n'.join(L) + '\n'


def stage_verdicts(out_dir: Path):
    _preflight('verdicts')
    v = compute_verdicts(_load(out_dir, 'models.json'), _load(out_dir, 'solver.json'), _load(out_dir, 'h3.json'))
    write_once(out_dir / 'verdicts.json', {'label': 'REGISTERED', **v})
    write_text_once(out_dir / 'tables.md', render_tables(v))
    for h in ('H1', 'H2', 'H3'):
        print(h, v[h]['verdict'])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m edalint.p4_iccad23')
    ap.add_argument('stage', choices=['models', 'solver', 'h3', 'verdicts'])
    a = ap.parse_args(argv)
    try:
        {'models': stage_models, 'solver': stage_solver, 'h3': stage_h3, 'verdicts': stage_verdicts}[a.stage](OUT_DIR)
    except (ProtocolError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
