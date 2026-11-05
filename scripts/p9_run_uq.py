"""P9 UQ runner (protocol_p9.md sections 5 and 7): bootstrap ensembles of the zoo's ridge / GBM, CRC intervals calibrated in-distribution
(C_id) and cross-design (C_xd), ensemble-Gaussian intervals; coverage, width, calibration error, OOD degradation.

  python scripts/p9_run_uq.py --target n14 --task ir_drop --model ridge         registered (FROZEN_p9.json), test designs via the guard
  python scripts/p9_run_uq.py --target iccad23 --model gbm                      registered, hidden cases via the guard
  python scripts/p9_run_uq.py --target n14 --task congestion --model ridge --dev   development: no test design; evaluation sets = T_id and
                                                                                  the C_xd designs (each calibrated on the other one)
Run-once: refuses to overwrite.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                                  # noqa: E402
from edalint.runner import write_once                                   # noqa: E402
from edalint_p9 import data as D                                        # noqa: E402
from edalint_p9 import uq as U                                          # noqa: E402
from edalint_p9.guard import open_iccad_hidden, open_n14_test, p9_code_hash   # noqa: E402

SEED = 20261105
M = 5
ALPHAS = tuple(round(1 - n, 4) for n in U.NOMINALS)                     # 0.5, 0.4, 0.3, 0.2, 0.1, 0.05
N14_SUBSAMPLE = {}                                                      # per-design caps on evaluated test rows (protocol section 5.3)


def factory(target, task, model):
    if target == 'n14':
        from edalint.p4_n14 import make_baseline
        return lambda: make_baseline(model, task)
    from edalint.p4_iccad23 import make_baseline as mk
    return lambda: mk(model)


def predict(ens, s):
    return U.mean_std(ens.predict_members(s))


def calibrate(ens, samples, log, tag):
    c = U.Calibrator()
    for s in samples:
        mu, sd = predict(ens, s)
        c.add(s.label, mu, sd)
    fit = c.fit(ALPHAS)
    log(f'  calibrated {tag}: n={fit["n_maps"]}, lambda@0.1 abs {fit["lambda"]["crc_abs"]["0.1"]:.4g} norm {fit["lambda"]["crc_norm"]["0.1"]:.4g}')
    return fit


def evaluate(ens, samples, cals, log, tag):
    """Per-map metrics for every (regime, method, alpha) and ens_gauss; returns summaries per alpha and the primary-alpha rows."""
    acc = {}
    for s in samples:
        mu, sd = predict(ens, s)
        y = np.asarray(s.label, np.float64)
        for a in ALPHAS:
            acc.setdefault(('ens_gauss', '-', a), []).append(U.map_metrics(y, mu, U.half_width('ens_gauss', sd, 0, 0, a)))
            for reg, fit in cals.items():
                for meth in ('crc_abs', 'crc_norm'):
                    lam = fit['lambda'][meth][f'{a:g}']
                    acc.setdefault((meth, reg, a), []).append(U.map_metrics(y, mu, U.half_width(meth, sd, lam, fit['eps'], a)))
    out = {}
    for (meth, reg, a), rows in acc.items():
        key = f'{meth}|{reg}'
        out.setdefault(key, {})[f'{a:g}'] = U.summarise_maps(rows, 1 - a)
    for key, by in out.items():
        by['calibration_error'] = U.calibration_error({1 - float(a): v['coverage'] for a, v in by.items() if a != 'calibration_error'})
    p = out.get('crc_norm|id', {}).get('0.1', {})
    log(f'  {tag}: n={len(samples)}, crc_norm|id coverage@0.9 {p.get("coverage", float("nan")):.3f}')
    return out


def run_n14(task, model, dev, log):
    if dev:
        cache, freeze = D.N14Cache(task), None
    else:
        cache, freeze = open_n14_test(task)
    lst = {d: cache.listing(d) for d in D.TRAIN_DESIGNS}
    parts = {d: D.seeded_split(lst[d], [300, 50, -1], SEED) for d in ('RISCY', 'RISCY-FPU')}
    keys = {'F': [(d, n) for d in ('RISCY', 'RISCY-FPU') for n in parts[d][0]],
            'C_id': [(d, n) for d in ('RISCY', 'RISCY-FPU') for n in parts[d][1]],
            'T_id': [(d, n) for d in ('RISCY', 'RISCY-FPU') for n in parts[d][2]],
            'C_xd': [(d, n) for d in ('Vortex-small', 'nvdla-small') for n in lst[d]]}
    seq = lambda ks: D.LazySamples(ks, lambda k: cache.load(*k))       # noqa: E731
    log(f'{task}/{model}: ' + ', '.join(f'{k} {len(v)}' for k, v in keys.items()))
    t0 = time.time()
    ens = U.BootstrapEnsemble(factory('n14', task, model), m=M, seed=SEED).fit(seq(keys['F']))
    t_fit = time.time() - t0
    log(f'  ensemble fitted in {t_fit:.0f} s')
    cals = {'id': calibrate(ens, seq(keys['C_id']), log, 'C_id'), 'xd': calibrate(ens, seq(keys['C_xd']), log, 'C_xd')}
    ev = {'T_id': evaluate(ens, seq(keys['T_id']), cals, log, 'T_id')}
    if dev:
        # no test design in development: evaluate each C_xd design with xd calibrated on the other one
        for d, other in (('Vortex-small', 'nvdla-small'), ('nvdla-small', 'Vortex-small')):
            c2 = {'id': cals['id'], 'xd': calibrate(ens, seq([(other, n) for n in lst[other]]), log, f'C_xd={other}')}
            ev[d] = evaluate(ens, seq([(d, n) for n in lst[d]]), c2, log, d)
    else:
        for d in D.TEST_DESIGNS:
            names = cache.listing(d)
            cap = N14_SUBSAMPLE.get((task, model, d))
            if cap and cap < len(names):
                names = D.seeded_split(names, [cap], SEED)[0]
            t1 = time.time()
            ev[d] = evaluate(ens, seq([(d, n) for n in names]), cals, log, d)
            ev[d]['_seconds'] = time.time() - t1
            ev[d]['_n_listed'] = len(cache.listing(d))
    return {'task': task, 'model': model, 'partitions': {k: [f'{d}/{n}' for d, n in v] for k, v in keys.items()},
            'bootstrap_indices': ens.boot_, 'calibration': cals, 'evaluation': ev, 'seconds_fit': t_fit,
            'cache_digests': cache.digest_report(), 'freeze': freeze}


def run_iccad(model, dev, log):
    from edalint.datasets.iccad23 import ICCAD23Dataset
    if dev:
        ds, loadq, freeze = ICCAD23Dataset(), None, None
    else:
        ds, loadq, freeze = open_iccad_hidden()
    def get(i):
        s = ds.get(i)
        if s is None:
            raise RuntimeError(f'{i} failed to load: {ds.errors.get(i)}')
        return s
    fake = ds.ids('fake')
    F, Cid, Tid = D.seeded_split(fake, [70, 15, -1], SEED)
    vis = ds.ids('visible')
    keys = {'F': F, 'C_id': Cid, 'T_id': Tid, 'C_xd': vis}
    seq = lambda ks: D.LazySamples(ks, get)                          # noqa: E731
    log(f'iccad23/{model}: ' + ', '.join(f'{k} {len(v)}' for k, v in keys.items()))
    t0 = time.time()
    ens = U.BootstrapEnsemble(factory('iccad23', 'ir_drop', model), m=M, seed=SEED).fit(seq(F))
    t_fit = time.time() - t0
    cals = {'id': calibrate(ens, seq(Cid), log, 'C_id'), 'xd': calibrate(ens, seq(vis), log, 'C_xd')}
    ev = {'T_id': evaluate(ens, seq(Tid), cals, log, 'T_id')}
    if dev:
        a, b = D.seeded_split(vis, [5, -1], SEED)
        c2 = {'id': cals['id'], 'xd': calibrate(ens, seq(a), log, 'C_xd half')}
        ev['visible_half'] = evaluate(ens, seq(b), c2, log, 'visible half')
    else:
        ev['hidden'] = evaluate(ens, D.LazySamples(ds.ids('test'), loadq), cals, log, 'hidden')
    return {'target': 'iccad23', 'model': model, 'partitions': keys, 'bootstrap_indices': ens.boot_, 'calibration': cals,
            'evaluation': ev, 'seconds_fit': t_fit, 'freeze': freeze}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=['n14', 'iccad23'])
    ap.add_argument('--task', choices=['ir_drop', 'congestion'])
    ap.add_argument('--model', required=True, choices=['ridge', 'gbm'])
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    log = lambda m: print(time.strftime('%H:%M:%S'), m, flush=True)       # noqa: E731
    name = f'uq_n14_{a.task}_{a.model}' if a.target == 'n14' else f'uq_iccad23_{a.model}'
    out = Path(a.out) if a.out else (REPO / 'results' / 'dev' / f'{name}_p9dev.json' if a.dev else REPO / 'results' / 'p9' / f'{name}.json')
    if out.exists():
        print(f'refused: {out} exists (run-once)')
        return 3
    t0 = time.time()
    try:
        res = run_n14(a.task, a.model, a.dev, log) if a.target == 'n14' else run_iccad(a.model, a.dev, log)
    except ProtocolError as exc:
        print(exc)
        return 2
    res = {'label': 'DEVELOPMENT (train only)' if a.dev else 'REGISTERED (post-freeze, run once)', 'p9_code_hash': p9_code_hash(),
           'seed': SEED, 'M': M, 'alphas': ALPHAS, 'seconds_total': time.time() - t0, 'result': res}
    write_once(out, res, dataset='p9')
    log(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
