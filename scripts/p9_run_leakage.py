"""P9 leakage runner (protocol_p9.md sections 4 and 7): L001 NN near duplicates, L002 family proximity (N14), L003 published split files.

  python scripts/p9_run_leakage.py --target n14 --task congestion        registered: test rows through guard.open_n14_test (FROZEN_p9.json)
  python scripts/p9_run_leakage.py --target iccad23                       registered: hidden cases through guard.open_iccad_hidden
  python scripts/p9_run_leakage.py --target splits                        registered: L003 on the published split files (metadata only)
  python scripts/p9_run_leakage.py --target n14 --task drc --dev          development: train rows only (queries = Vortex-small, nvdla-small)
Run-once: refuses to overwrite results/p9/*.json (registered) or results/dev/*_p9dev.json (dev).
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                                  # noqa: E402
from edalint.runner import write_once                                   # noqa: E402
from edalint_p9 import data as D                                        # noqa: E402
from edalint_p9 import leakage as LK                                    # noqa: E402
from edalint_p9.guard import open_iccad_hidden, open_n14_test, p9_code_hash   # noqa: E402

EXTERNAL = Path('E:/edalint_data/p9_external')
CN_FILES = Path('E:/edalint_data/docs_cache/cn_gh/routability_ir_drop_prediction/files')


def _describe_rows(load, keys, log, every=500):
    X, Y = [], []
    t0 = time.time()
    for k, key in enumerate(keys):
        x, y = LK.describe(load(key))
        X.append(x)
        Y.append(y)
        if (k + 1) % every == 0:
            log(f'  {k + 1}/{len(keys)} rows described ({time.time() - t0:.0f} s)')
    return np.stack(X), np.stack(Y)


def _l001(q_keys, r_keys, QX, QY, RX, RY, load, q_design, log):
    SI = LK.similarity(QX, RX)
    SL = LK.label_similarity(QY, RY)
    qn = [f'{d}/{n}' for d, n in q_keys]
    rn = [f'{d}/{n}' for d, n in r_keys]
    recs = LK.nn_records(qn, rn, SI, SL)
    out = {'pooled': {'counts': LK.threshold_counts(recs)}, 'per_design': {}}
    for d in sorted(set(q_design)):
        sub = [r for r, dd in zip(recs, q_design) if dd == d]
        out['per_design'][d] = {'counts': LK.threshold_counts(sub),
                                'nn_input_sim_quantiles_0_10_50_90_99_100': np.quantile([r['input_sim'] for r in sub], [0, .1, .5, .9, .99, 1]).tolist(),
                                'nn_label_sim_quantiles_0_10_50_90_99_100': np.quantile([r['label_sim'] for r in sub], [0, .1, .5, .9, .99, 1]).tolist()}
    conf = []
    for r in recs:
        if r['input_sim'] >= LK.DUP_CORR and r['label_sim'] >= LK.DUP_CORR:
            a, b = r['id'].split('/', 1), r['nn'].split('/', 1)
            conf.append({'query': r['id'], 'nn': r['nn'], **LK.confirm_pair(load(tuple(a)), load(tuple(b)))})
    out['flagged_pairs_confirmation'] = conf
    out['n_confirmed'] = sum(1 for c in conf if c['confirmed'])
    out['rows'] = [{k: r[k] for k in ('id', 'nn', 'input_sim', 'label_sim', 'n_pair_ge_primary', 'n_input_ge_primary')} for r in recs]
    log(f"  L001: {out['pooled']['counts'][f'{LK.DUP_CORR:g}']['nn_pair_ge']} NN pairs >= {LK.DUP_CORR}, {out['n_confirmed']} confirmed")
    return out, SI


def run_n14(task: str, dev: bool, log):
    if dev:
        cache = D.N14Cache(task)
        ref_designs, q_designs = ('RISCY', 'RISCY-FPU'), ('Vortex-small', 'nvdla-small')
        freeze = None
    else:
        cache, freeze = open_n14_test(task)
        ref_designs, q_designs = D.TRAIN_DESIGNS, D.TEST_DESIGNS
    r_keys = [(d, n) for d in ref_designs for n in cache.listing(d)]
    q_keys = [(d, n) for d in q_designs for n in cache.listing(d)]
    log(f'{task}: reference {len(r_keys)} rows {ref_designs}, queries {len(q_keys)} rows {q_designs}')

    def load(key):
        return cache.load(key[0], key[1])
    t0 = time.time()
    RX, RY = _describe_rows(load, r_keys, log)
    QX, QY = _describe_rows(load, q_keys, log)
    t_desc = time.time() - t0
    q_design = [d for d, _ in q_keys]
    l001, SI = _l001(q_keys, r_keys, QX, QY, RX, RY, load, q_design, log)
    rdes = np.array([d for d, _ in r_keys])
    within = {}
    for d in ref_designs:
        m = np.flatnonzero(rdes == d)
        if len(m) >= 2:
            within[d] = LK.nearest(LK.similarity(RX[m], RX[m]), exclude_self=True)[1]
    qd = np.array(q_design)
    rows = {d: SI[qd == d] for d in q_designs if (qd == d).any()}
    l002 = LK.design_proximity(rows, list(ref_designs), sorted(rows), within, list(rdes))
    l002_ref = {d: {'median': float(np.median(v)), 'q25': float(np.quantile(v, .25)), 'n': int(len(v))} for d, v in within.items()}
    for d, r in l002.items():
        log(f"  L002 {d}: material={r['material']} ({r['reason']})")
    return {'task': task, 'reference_designs': list(ref_designs), 'query_designs': list(q_designs), 'n_reference': len(r_keys),
            'n_queries': len(q_keys), 'L001': l001, 'L002': {'per_design': l002, 'within_design_reference': l002_ref},
            'cache_digests': cache.digest_report(), 'seconds_describe': t_desc, 'freeze': freeze}


def run_iccad(dev: bool, log):
    from edalint.datasets.iccad23 import ICCAD23Dataset
    if dev:
        ds = ICCAD23Dataset()
        r_ids, q_ids = ds.ids('fake'), ds.ids('visible')
        loadq, freeze = ds.get, None
    else:
        ds, loadq, freeze = open_iccad_hidden()
        r_ids, q_ids = ds.ids('train'), ds.ids('test')
    log(f'iccad23: reference {len(r_ids)}, queries {len(q_ids)}')
    RX, RY = _describe_rows(ds.get, r_ids, log)
    QX, QY = _describe_rows(loadq, q_ids, log)
    r_keys = [('r', i) for i in r_ids]
    q_keys = [('q', i) for i in q_ids]

    def load(key):
        return ds.get(key[1]) if key[0] == 'r' else loadq(key[1])
    l001, _ = _l001(q_keys, r_keys, QX, QY, RX, RY, load, [ds.meta(i).get('design', i) for i in q_ids], log)
    return {'target': 'iccad23', 'n_reference': len(r_ids), 'n_queries': len(q_ids), 'L001': l001,
            'L002': 'not applicable: ICCAD\'23 has no design-family metadata (protocol section 4.2)', 'freeze': freeze}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_splits(log):
    sets = {
        'circuitnet_official_N14': {'train': CN_FILES / 'train_N14.csv', 'test': CN_FILES / 'test_N14.csv'},
        'circuitnet_official_N28': {'train': CN_FILES / 'train_N28.csv', 'test': CN_FILES / 'test_N28.csv'},
        'cldroute_N14': {s: EXTERNAL / 'cldroute' / f'n14_{s}.csv' for s in ('train', 'val', 'test')},
        'cldroute_N28': {s: EXTERNAL / 'cldroute' / f'n28_{s}.csv' for s in ('train', 'val', 'test')},
    }
    out = {}
    for name, files in sets.items():
        rows = {s: p.read_text(encoding='utf-8', errors='replace').splitlines() for s, p in files.items()}
        res = LK.split_file_overlap(rows)
        res['files'] = {s: {'path': str(p), 'sha256': _sha(p), 'rows': sum(1 for r in rows[s] if r.strip())} for s, p in files.items()}
        out[name] = res
        log(f"{name}: " + '; '.join(f"{'/'.join(p['splits'])} designs {p['design_overlap']} families {p['family_overlap']}" for p in res['pairs']))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=['n14', 'iccad23', 'splits'])
    ap.add_argument('--task', choices=['ir_drop', 'congestion', 'drc'])
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    log = lambda m: print(time.strftime('%H:%M:%S'), m, flush=True)       # noqa: E731
    name = {'n14': f'leakage_n14_{a.task}', 'iccad23': 'leakage_iccad23', 'splits': 'leakage_splits'}[a.target]
    out = Path(a.out) if a.out else (REPO / 'results' / 'dev' / f'{name}_p9dev.json' if a.dev else REPO / 'results' / 'p9' / f'{name}.json')
    if out.exists():
        print(f'refused: {out} exists (run-once)')
        return 3
    t0 = time.time()
    try:
        if a.target == 'n14':
            res = run_n14(a.task, a.dev, log)
        elif a.target == 'iccad23':
            res = run_iccad(a.dev, log)
        else:
            if a.dev:
                print('refused: the split-file check has no dev mode (metadata only, run once)')
                return 2
            res = run_splits(log)
    except ProtocolError as exc:
        print(exc)
        return 2
    res = {'label': 'DEVELOPMENT (train only)' if a.dev else 'REGISTERED (post-freeze, run once)', 'p9_code_hash': p9_code_hash(),
           'seconds_total': time.time() - t0, 'result': res}
    write_once(out, res, dataset='p9')
    log(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
