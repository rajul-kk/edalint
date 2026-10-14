"""P7 mechanical aggregator: H-A / H-B / H-C for IR drop (protocol_p7.md sections 5, 11 D5 and 12) from the scored row files only.

Inputs: `results/p7/train_ir_drop_A_s<seed>.json` (divergence rule) and `results/p7/models_ir_drop_A_s<seed>.json` (written by scripts/p7_run_score.py, the format of `edalint_p7.score.score_predictions` plus
`eval_ids`) and the REGISTERED N14 files `results/n14/models_ir_drop.json` (baselines, mean_map) and `results/n14/verdicts.json` (registered H1
verdict and ratio, shown next to the P7 verdict, never edited). Nothing here opens a split or a dataset; all arithmetic reuses the N14 helpers
(`edalint.p4_n14.design_values / pooled_value / macro`, `RocAcc`), so the metric definitions are the registered ones.

Outputs (run-once): results/p7/verdicts_p7.json and results/p7/tables_p7.md. The entry is `run_verdicts`; it refuses before the P7 freeze.
Rules applied, with the pre-freeze clarifications of protocol_p7.md section 12.9 where the text left a choice.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from edalint.core import ProtocolError
from edalint.p4_n14 import H1_FRACTION, PUBLISHED_REF, RocAcc, design_values, macro, pooled_value
from edalint.protocol import REPO_DIR
from edalint.runner import write_once, write_text_once

from .guard import check_p7_freeze

TASK = 'ir_drop'
ARM = 'A'
SEEDS = (20261101, 20261102, 20261103)
GATE_FACTOR = 0.5                                  # D5
PUBLISHED = PUBLISHED_REF[TASK]['value']           # 0.8671 (MAVIREC, CircuitNet 2.0 App. B.2)
H_B_FRACTION = H1_FRACTION                         # 0.8, the registered N14 line
BAND_AUC = 0.03                                    # H-C band
AUTHORS_INF = 0.05                                 # the authors' substitute for an infinite NRMS (secondary only)
MIN_FINITE_SEEDS = 2


# -- pure verdict logic (boundary-testable) -----------------------------------------------------------------------------------
def gate_threshold(m_t: float, published: float = PUBLISHED, factor: float = GATE_FACTOR) -> float:
    """R_t at which the gate G_R >= factor x G_ref just passes (G_ref = published - M_t)."""
    return m_t + factor * (published - m_t)


def h_a_gate(r: float, m_t: float, published: float = PUBLISHED, factor: float = GATE_FACTOR) -> bool:
    """H-A: G_R >= factor x G_ref, with G_R = r - M_t and G_ref = published - M_t."""
    if not np.isfinite(r):
        return False
    return bool((r - m_t) >= factor * (published - m_t))


def h_b(g_b: float, g_r: float) -> bool:
    """H-B: TRUE iff G_b >= 0.8 x G_R."""
    return bool(g_b >= H_B_FRACTION * g_r)


def decide(r: float, m_t: float, g_b: float, published: float = PUBLISHED) -> Dict[str, Any]:
    """Verdict for one value of R_t. verdict is True / False (gate passed) or None ('no verdict', gate failed or R undefined)."""
    if not np.isfinite(r):
        return {'r_t': None, 'gate_pass': False, 'g_r': None, 'h_b_descriptive': None, 'verdict': None}
    g_r = r - m_t
    gate = h_a_gate(r, m_t, published)
    hb = h_b(g_b, g_r)
    return {'r_t': float(r), 'g_r': float(g_r), 'gate_pass': gate, 'h_b_descriptive': hb, 'verdict': hb if gate else None}


def label(v: Optional[bool]) -> str:
    return 'TRUE' if v is True else 'FALSE' if v is False else 'no verdict'


def h_c(r: float, published: float = PUBLISHED, band: float = BAND_AUC) -> Dict[str, Any]:
    d = r - published
    return {'D': float(d), 'band': band, 'within_band': bool(abs(round(d, 12)) <= band)}


# -- reading the files ---------------------------------------------------------------------------------------------------------
def _load(p: Path):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def model_path(models_dir, seed: int) -> Path:
    return Path(models_dir) / f'models_{TASK}_{ARM}_s{seed}.json'


def load_seed_files(models_dir, seeds: Sequence[int] = SEEDS) -> Dict[int, Optional[dict]]:
    """seed -> payload, or None if the file is absent. Header checks: task, arm, seed."""
    out: Dict[int, Optional[dict]] = {}
    for s in seeds:
        p = model_path(models_dir, s)
        if not p.exists():
            out[s] = None
            continue
        d = _load(p)
        if d.get('task') != TASK or d.get('arm') != ARM or d.get('seed') != s:
            raise ProtocolError(f'{p.name}: header says task={d.get("task")!r} arm={d.get("arm")!r} seed={d.get("seed")!r}, expected {TASK}/{ARM}/{s}')
        out[s] = d
    return out


def train_path(models_dir, seed: int) -> Path:
    return Path(models_dir) / f'train_{TASK}_{ARM}_s{seed}.json'


def load_train_records(models_dir, seeds: Sequence[int] = SEEDS) -> Dict[int, Optional[dict]]:
    """seed -> train result (`train_ir_drop_A_s<seed>.json`, next to the scored file), or None if absent."""
    return {s: (_load(train_path(models_dir, s)) if train_path(models_dir, s).exists() else None) for s in seeds}


def divergence_reason(rec: Optional[dict]) -> Optional[str]:
    """Mechanical divergence rule (protocol_p7.md 12.10): a seed is diverged / undefined iff its train result is missing, its last-block mean
    loss is missing or non-finite, or `final_params_finite` is not True (a record without the field counts as not finite). The skipped-step
    fraction is reported but is not a rule. Returns the reason, or None if the seed is fine."""
    if rec is None:
        return 'no train result'
    lb = rec.get('last_block_loss')
    if not isinstance(lb, (int, float)) or isinstance(lb, bool) or not np.isfinite(lb):
        return f'diverged: last_block_loss is not finite ({lb!r})'
    if rec.get('final_params_finite') is not True:
        return f'diverged: final_params_finite is {rec.get("final_params_finite")!r}'
    return None


def _block(payload: dict) -> dict:
    return {'rows': payload['rows'], 'official_roc': payload['official_roc']}


def check_ids(payload: dict, n14_models: dict, name: str):
    """The scored ids must equal the registered eval_ids (design by design); a file without eval_ids is refused."""
    got = payload.get('eval_ids')
    if got is None:
        raise ProtocolError(f'{name}: no eval_ids in the scored file')
    exp = n14_models['eval_ids']
    if {d: sorted(v) for d, v in got.items()} != {d: sorted(v) for d, v in exp.items()}:
        raise ProtocolError(f'{name}: eval_ids differ from the registered N14 eval_ids')
    scored = sorted(r['id'] for r in payload['rows'])
    if scored != sorted(i for v in exp.values() for i in v):
        raise ProtocolError(f'{name}: scored row ids differ from the registered eval_ids')


# -- per-model views -----------------------------------------------------------------------------------------------------------
def nrms_view(rows: List[dict]) -> Dict[str, Any]:
    """NRMS (official, `nrms` field) aggregated over finite rows only; undefined = non-finite rows, counted and listed per design.
    Secondary: inf rows replaced by 0.05 (authors' wrapper), nan rows still excluded."""
    per: Dict[str, Any] = {}
    for d in sorted({r['design'] for r in rows}):
        rr = [r for r in rows if r['design'] == d]
        v = np.array([r['nrms'] for r in rr], float)
        fin = np.isfinite(v)
        sub = np.where(np.isposinf(v), AUTHORS_INF, v)
        sub = sub[np.isfinite(sub)]
        per[d] = {'n_rows': len(rr), 'n_undefined': int((~fin).sum()), 'undefined_ids': [r['id'] for r, f in zip(rr, fin) if not f],
                  'mean_finite': float(v[fin].mean()) if fin.any() else None,
                  'mean_inf_to_0.05': float(sub.mean()) if sub.size else None}
    macro_fin = macro({d: (x['mean_finite'] if x['mean_finite'] is not None else float('nan')) for d, x in per.items()})
    v = np.array([r['nrms'] for r in rows], float)
    v = v[np.isfinite(v)]
    return {'per_design': per, 'design_macro_finite': macro_fin, 'pooled_finite': float(v.mean()) if v.size else None,
            'n_undefined_total': int(sum(x['n_undefined'] for x in per.values()))}


def seed_view(payload: dict) -> Dict[str, Any]:
    b = _block(payload)
    dv = design_values(TASK, b)
    dva = design_values(TASK, b, anchored=True)
    roc = {d: RocAcc.from_dict(a) for d, a in payload['official_roc'].items()}
    return {'design_values': dv, 'design_macro': macro(dv), 'design_values_anchored': dva, 'design_macro_anchored': macro(dva),
            'sample_pooled': pooled_value(TASK, b), 'sample_pooled_anchored': pooled_value(TASK, b, anchored=True),
            'n_no_positive': {d: a.n_no_positive for d, a in roc.items()}, 'n_samples': {d: a.n_samples for d, a in roc.items()},
            'nrms': nrms_view(payload['rows']), 'p7_code_hash': payload.get('p7_code_hash')}


def baseline_views(n14_models: dict) -> Dict[str, Dict[str, Any]]:
    out = {}
    for n, r in n14_models['results'].items():
        dv, dva = design_values(TASK, r), design_values(TASK, r, anchored=True)
        out[n] = {'design_values': dv, 'design_macro': macro(dv), 'design_macro_anchored': macro(dva),
                  'sample_pooled': pooled_value(TASK, r), 'sample_pooled_anchored': pooled_value(TASK, r, anchored=True)}
    return out


def _mean(xs: Sequence[float]) -> float:
    a = np.array(xs, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float('nan')


# -- the aggregator ------------------------------------------------------------------------------------------------------------
def compute_p7_verdicts(seed_payloads: Dict[int, Optional[dict]], n14_models: dict, n14_verdicts: Optional[dict] = None,
                        train_records: Optional[Dict[int, Optional[dict]]] = None) -> Dict[str, Any]:
    """Mechanical H-A / H-B / H-C and the registered sensitivities for IR drop. `seed_payloads` maps each registered seed to its scored file
    (or None if absent) and `train_records` the matching train results (None / absent entry = missing = undefined). A seed is 'defined' when
    its train record passes `divergence_reason` AND its design-macro official ROC area is finite; with fewer than 2 defined seeds there is no verdict."""
    train_records = train_records if train_records is not None else {}
    base = baseline_views(n14_models)
    mm = base['mean_map']
    m_t = mm['design_macro']
    cand = {n: b['design_macro'] - m_t for n, b in base.items() if n != 'mean_map' and np.isfinite(b['design_macro'])}
    best = max(cand, key=cand.get)
    g_b = cand[best]
    seeds: Dict[str, Any] = {}
    for s, p in seed_payloads.items():
        if p is None:
            seeds[str(s)] = {'defined': False, 'reason': 'no scored file' + (f'; {divergence_reason(train_records.get(s))}' if divergence_reason(train_records.get(s)) else '')}
            continue
        check_ids(p, n14_models, f'seed {s}')
        v = seed_view(p)
        tr = train_records.get(s)
        why = divergence_reason(tr)
        v['defined'] = bool(np.isfinite(v['design_macro'])) and why is None
        v['train'] = {'last_block_loss': (tr or {}).get('last_block_loss'), 'final_params_finite': (tr or {}).get('final_params_finite'),
                      'skipped_steps': (tr or {}).get('skipped_steps'), 'iters_done': (tr or {}).get('iters_done')}
        if why:
            v['reason'] = why
        elif not np.isfinite(v['design_macro']):
            v['reason'] = 'design-macro ROC area not finite'
        seeds[str(s)] = v
    defined = {s: v for s, v in seeds.items() if v['defined']}
    thr = gate_threshold(m_t)
    out: Dict[str, Any] = {
        'task': TASK, 'arm': ARM, 'seeds_registered': [str(s) for s in seed_payloads], 'seeds_defined': sorted(defined),
        'inputs': {'M_t_mean_map_design_macro': m_t, 'published': PUBLISHED, 'G_ref': PUBLISHED - m_t, 'gate_factor': GATE_FACTOR,
                   'gate_threshold_R_t': thr, 'best_baseline': best, 'G_b': g_b, 'baseline_gains': cand,
                   'baseline_design_macro': {n: b['design_macro'] for n, b in base.items()},
                   'baseline_design_values': {n: b['design_values'] for n, b in base.items()}},
        'per_seed': seeds,
        'rule': 'H-A: R_t - M_t >= 0.5 x (0.8671 - M_t); H-B (only if H-A passes): TRUE iff G_b >= 0.8 x G_R; H-C descriptive |R_t - 0.8671| <= 0.03'}
    if len(defined) < MIN_FINITE_SEEDS:
        out.update(verdict=None, verdict_label='no verdict', reason=f'fewer than {MIN_FINITE_SEEDS} seeds with a finite metric ({len(defined)})')
        out['registered_n14'] = _registered(n14_verdicts, None)
        return out
    r = _mean([v['design_macro'] for v in defined.values()])
    prim = decide(r, m_t, g_b)
    out['primary'] = prim
    out['verdict'] = prim['verdict']
    out['verdict_label'] = label(prim['verdict'])
    out['H_A'] = {'gate_pass': prim['gate_pass'], 'R_t': r, 'threshold': thr, 'G_R': prim['g_r']}
    out['H_B'] = {'applies': prim['gate_pass'], 'G_b': g_b, 'G_R': prim['g_r'], 'ratio_G_b_over_G_R': (g_b / prim['g_r']) if prim['g_r'] else None,
                  'verdict': prim['verdict'], 'descriptive_only': not prim['gate_pass'], 'descriptive_value': prim['h_b_descriptive']}
    hc = h_c(r)
    hc['reading'] = ('consistent with a same-split model' if hc['within_band'] else 'split/preprocessing effect, signed D')
    hc['resolved_wording_allowed'] = bool(prim['gate_pass'] and hc['within_band'])
    hc['note'] = ('' if hc['resolved_wording_allowed'] else
                  ('gate passed, retrain below published' if prim['gate_pass'] and hc['D'] < 0 else
                   'gate passed, retrain above published band' if prim['gate_pass'] else 'gate failed: descriptive only'))
    out['H_C'] = hc
    # sensitivities (cannot change the verdict)
    dm = {s: v['design_macro'] for s, v in defined.items()}
    sens: Dict[str, Any] = {}
    for nm, s in (('best_seed', max(dm, key=dm.get)), ('worst_seed', min(dm, key=dm.get))):
        d = decide(dm[s], m_t, g_b)
        sens[nm] = {'seed': s, **d, 'verdict_label': label(d['verdict']), 'flips_verdict': d['verdict'] != prim['verdict']}
    sens['seed_sensitive'] = bool(sens['best_seed']['flips_verdict'] or sens['worst_seed']['flips_verdict'])

    def hb_alt(key_model: str, key_base: str):
        r_a = _mean([v[key_model] for v in defined.values()])
        m_a = base['mean_map'][key_base]
        g_a = max(b[key_base] - m_a for n, b in base.items() if n != 'mean_map' and np.isfinite(b[key_base]))
        hb = h_b(g_a, r_a - m_a)
        return {'R_t': r_a, 'M_t': m_a, 'G_b': g_a, 'G_R': r_a - m_a, 'h_b': hb, 'gate_taken_from_primary': prim['gate_pass'],
                'verdict': hb if prim['gate_pass'] else None, 'verdict_label': label(hb if prim['gate_pass'] else None),
                'agrees_with_primary': (hb if prim['gate_pass'] else None) == prim['verdict']}
    sens['anchored_roc'] = hb_alt('design_macro_anchored', 'design_macro_anchored')
    sens['sample_pooled'] = hb_alt('sample_pooled', 'sample_pooled')
    g_r_pub = max(prim['g_r'], PUBLISHED - m_t)
    hb_pub = h_b(g_b, g_r_pub)
    sens['published_credit'] = {'G_R_used': g_r_pub, 'h_b': hb_pub, 'verdict': hb_pub if prim['gate_pass'] else None,
                                'budget_sensitive': bool(prim['gate_pass'] and hb_pub != prim['verdict'])}
    designs = sorted(mm['design_values'])
    per_design = {d: _mean([v['design_values'][d] for v in defined.values()]) for d in designs}
    sens['per_design'] = {'model_seed_mean': per_design, 'mean_map': mm['design_values'],
                          'n_designs_beating_mean_map': int(sum(per_design[d] > mm['design_values'][d] for d in designs
                                                                if np.isfinite(per_design[d]) and np.isfinite(mm['design_values'][d]))),
                          'n_designs': len(designs),
                          'per_seed_n_designs_beating_mean_map': {s: int(sum(v['design_values'][d] > mm['design_values'][d] for d in designs
                                                                             if np.isfinite(v['design_values'][d]) and np.isfinite(mm['design_values'][d])))
                                                                  for s, v in defined.items()}}
    out['sensitivities'] = sens
    out['no_positive_rows'] = {s: v['n_no_positive'] for s, v in seeds.items() if 'n_no_positive' in v}
    out['undefined_nrms'] = {s: {d: x['n_undefined'] for d, x in v['nrms']['per_design'].items()} for s, v in seeds.items() if 'nrms' in v}
    out['registered_n14'] = _registered(n14_verdicts, prim['verdict'])
    return out


def _registered(n14_verdicts: Optional[dict], p7: Optional[bool]) -> Dict[str, Any]:
    if not n14_verdicts:
        return {'available': False}
    h = n14_verdicts['H1']['tasks'][TASK]
    return {'available': True, 'verdict': h.get('verdict'), 'ratio': h.get('ratio'), 'unchanged': True,
            'p7_agrees': (None if p7 is None else bool(p7 == h.get('verdict')))}


# -- tables --------------------------------------------------------------------------------------------------------------------
def _f(x, nd=4):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def render_tables_p7(v: Dict[str, Any]) -> str:
    L = ['# P7 (IR drop only): retrained MAVI on the N14 split', '',
         "Authors' architecture and recipe, reduced budget (30,000 iterations, fp16 training), our split. Reference is a same-split retrain, not the published model.", '']
    i = v['inputs']
    L += [f"Seeds defined: {', '.join(v['seeds_defined']) or 'none'} of {', '.join(v['seeds_registered'])}", '']
    if 'primary' not in v:
        L += [f"**No verdict:** {v['reason']}", '']
        return '\n'.join(L) + '\n'
    p = v['primary']
    L += ['## Verdict', '', '| item | value |', '|---|---|',
          f"| M_t (mean_map, design-macro ROC area) | {_f(i['M_t_mean_map_design_macro'])} |",
          f"| R_t (mean over seeds, design-macro) | {_f(p['r_t'])} |",
          f"| H-A gate threshold (0.5 x G_ref) | {_f(i['gate_threshold_R_t'])} |", f"| H-A gate | {'PASS' if p['gate_pass'] else 'FAIL'} |",
          f"| best baseline / G_b | {i['best_baseline']} / {_f(i['G_b'])} |", f"| G_R | {_f(p['g_r'])} |",
          f"| H-B | {v['verdict_label']}{'' if p['gate_pass'] else ' (gate failed; descriptive: ' + label(p['h_b_descriptive']) + ')'} |",
          f"| H-C D = R_t - 0.8671 | {_f(v['H_C']['D'])} ({v['H_C']['reading']}) |", '']
    r = v['registered_n14']
    if r.get('available'):
        L += [f"Registered N14 H1 (IR drop, unchanged): {r['verdict']}, ratio {_f(r['ratio'], 2)}; P7 agrees: {r['p7_agrees']}", '']
    s = v['sensitivities']
    L += ['## Sensitivities (cannot change the verdict)', '', '| sensitivity | result |', '|---|---|',
          f"| best seed ({s['best_seed']['seed']}) | {_f(s['best_seed']['r_t'])}: {s['best_seed']['verdict_label']} |",
          f"| worst seed ({s['worst_seed']['seed']}) | {_f(s['worst_seed']['r_t'])}: {s['worst_seed']['verdict_label']} |",
          f"| seed-sensitive | {s['seed_sensitive']} |",
          f"| anchored ROC area | R {_f(s['anchored_roc']['R_t'])}, G_b {_f(s['anchored_roc']['G_b'])}: {s['anchored_roc']['verdict_label']} (agrees {s['anchored_roc']['agrees_with_primary']}) |",
          f"| sample-pooled | R {_f(s['sample_pooled']['R_t'])}, G_b {_f(s['sample_pooled']['G_b'])}: {s['sample_pooled']['verdict_label']} (agrees {s['sample_pooled']['agrees_with_primary']}) |",
          f"| G_R = max(G_R, G_ref) | {label(s['published_credit']['verdict'])}; budget-sensitive: {s['published_credit']['budget_sensitive']} |",
          f"| designs beating mean_map | {s['per_design']['n_designs_beating_mean_map']} of {s['per_design']['n_designs']} |", '',
          '## Per design (official ROC area)', '', '| design | mean_map | model (seed mean) |', '|---|---|---|']
    for d, x in s['per_design']['model_seed_mean'].items():
        L.append(f"| {d} | {_f(s['per_design']['mean_map'][d])} | {_f(x)} |")
    L += ['', '## Rows with no positive pixel (ROC rule) and undefined NRMS, per design and seed', '',
          '| seed | design | n_samples | n_no_positive | n_undefined_nrms | NRMS finite mean | NRMS (inf=0.05) |', '|---|---|---|---|---|---|---|']
    for sd, x in v['per_seed'].items():
        if 'nrms' not in x:
            continue
        for d, n in x['nrms']['per_design'].items():
            L.append(f"| {sd} | {d} | {x['n_samples'].get(d)} | {x['n_no_positive'].get(d)} | {n['n_undefined']} | {_f(n['mean_finite'])} | {_f(n['mean_inf_to_0.05'])} |")
    return '\n'.join(L) + '\n'


# -- entry ---------------------------------------------------------------------------------------------------------------------
def run_verdicts(models_dir=None, n14_dir=None, protocol_dir=None, repo_root=None, out_dir=None, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Refuses (ProtocolError) before the P7 freeze or on a code-hash mismatch; then reads the scored files and writes verdicts_p7.json and
    tables_p7.md under `out_dir` (default = models_dir), run-once."""
    check_p7_freeze(protocol_dir, repo_root, what='P7 verdicts')
    md = Path(models_dir or REPO_DIR / 'results' / 'p7')
    nd = Path(n14_dir or REPO_DIR / 'results' / 'n14')
    od = Path(out_dir or md)
    n14_models = _load(nd / f'models_{TASK}.json')
    n14_ver = _load(nd / 'verdicts.json') if (nd / 'verdicts.json').exists() else None
    v = compute_p7_verdicts(load_seed_files(md, seeds), n14_models, n14_ver, load_train_records(md, seeds))
    write_once(od / 'verdicts_p7.json', {'label': 'REGISTERED (post-freeze)', **v}, dataset='p7')
    write_text_once(od / 'tables_p7.md', render_tables_p7(v))
    return v
