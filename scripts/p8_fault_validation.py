"""P8 fault-injection validation (protocol/protocol_faults.md). One base per call; result files are run-once.

  python scripts/p8_fault_validation.py --base synthetic --out results/dev --tag p8dev          # development, in-memory data only
  python scripts/p8_fault_validation.py --base icb3 --data-root <datasets> --out results/faults --tag p8
Bases: synthetic (dev only), iccad23, n14ir, icb2..icb5 (IC-ThermBench, orientation corrected; icbN-stored = as released, no injection).
Output: <out>/faults_<base>_<tag>.json and a table <out>/tables_<base>_<tag>.md.
"""
import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from edalint import faults as F                                    # noqa: E402
from edalint.checks import CHECKS, Options                         # noqa: E402
from edalint.runner import write_once, write_text_once              # noqa: E402

SEED0 = 20261104
CHECKED = ('E001', 'E002', 'E003', 'E004', 'E005')
OPT = Options(split='all', max_samples=None, dup_max_samples=800)


def open_base(name, root, cache):
    if name == 'synthetic':
        from edalint.datasets.synthetic import make_synthetic
        return make_synthetic(n_train_designs=14, n_test_designs=6, per_design=4, size=48, seed=11)
    if name == 'iccad23':
        from edalint.datasets import open_dataset
        return open_dataset('iccad23', **({'root': root} if root else {}))
    if name == 'n14ir':
        from edalint.datasets import open_dataset
        return open_dataset('n14-ir_drop', mode='authors', cache_dir=cache, **({'root': root} if root else {}))
    if name.startswith('icb'):
        from edalint.datasets.icthermbench import ICThermBenchDataset
        stored = name.endswith('-stored')
        return ICThermBenchDataset(root, f"level{name[3]}", fix_orientation=not stored)
    raise SystemExit(f'unknown base {name!r}')


def pool_samples(ds, size):
    names = [s for s in ds.splits if s not in ds.test_splits and s not in ds.aux_splits and ds.splits[s]]
    ids = [i for s in names for i in ds.splits[s]]
    ids = [ids[j] for j in np.unique(np.linspace(0, len(ids) - 1, min(size, len(ids))).round().astype(int))]
    samples, removed = F.materialise(ds, ids), Counter()
    keep = []
    for s in samples:
        L = s.label
        if not np.isfinite(L).all() or any(not np.isfinite(a).all() for _, a in s.channels()):
            removed['non-finite'] += 1
        elif np.all(L == L.flat[0]):
            removed['constant or all-zero label'] += 1
        elif L.min() < -max(1e-12, 1e-6 * float(np.abs(L).max())):         # E002's tolerance: rounding-level negatives are not negative labels
            removed['negative label'] += 1
        else:
            keep.append(s)
    for s in keep:                                              # grouping rule of amendment A2(c)
        s.design = str(s.meta.get('design_id', s.design))
    return keep, dict(removed), len(ids)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    global SEED0, CHECKED
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--data-root')
    ap.add_argument('--cache-dir')
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--pool', type=int, default=600)
    ap.add_argument('--sample', type=int, default=200)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--no-injection', action='store_true', help='false-positive run only (used for icbN-stored)')
    ap.add_argument('--seed0', type=int, default=SEED0, help='first draw seed (amendments A3/A4 reruns use 20261204)')
    ap.add_argument('--only', help='comma list of check codes: restrict the false-positive run and the faults to these checks')
    args = ap.parse_args()
    SEED0 = args.seed0
    if args.only:
        CHECKED = tuple(c for c in CHECKED if c in args.only.split(','))
    t0 = time.time()
    ds = open_base(args.base, args.data_root, args.cache_dir)
    pool, removed, n_pool0 = pool_samples(ds, args.pool)
    print(f'{args.base}: pool {len(pool)}/{n_pool0}, removed {removed}', flush=True)
    fp = {c: {'draws': 0, 'flagged': 0, 'messages': Counter()} for c in CHECKED}
    det = {}
    skipped = {}
    for d in range(args.draws):
        rng = np.random.default_rng(SEED0 + d)
        pick = sorted(rng.choice(len(pool), size=min(args.sample, len(pool)), replace=False))
        clean = F.assign_splits(F.materialise(F.build(pool, ds), [pool[j].id for j in pick]), rng)
        cds = F.build(clean, ds)
        for c in CHECKED:
            r = CHECKS[c](cds, OPT)
            fp[c]['draws'] += 1
            if r.status == 'flagged':
                fp[c]['flagged'] += 1
                for f in r.findings:
                    if f.severity in ('error', 'warning'):
                        fp[c]['messages'][f'{f.severity}: {f.message[:140]}'] += 1
        if args.no_injection:
            continue
        for name, fault in F.FAULTS.items():
            if fault.check not in CHECKED:
                continue
            why = F.applicable(fault, clean)
            if why:
                skipped[name] = why
                continue
            for lv in fault.severities:
                bad = fault.inject(F.materialise(cds, [s.id for s in clean]), np.random.default_rng(SEED0 + 1000 * (lv + 1) + d), lv)
                r = CHECKS[fault.check](F.build(bad, ds), OPT)
                e = det.setdefault(name, {}).setdefault(lv, {'n': 0, 'detected': 0})
                e['n'] += 1
                e['detected'] += int(F.detected(r, fault))
        print(f'  draw {d + 1}/{args.draws} done ({time.time() - t0:.0f} s)', flush=True)
    payload = {'base': args.base, 'protocol': 'protocol/protocol_faults.md', 'seed0': SEED0, 'draws': args.draws, 'sample': args.sample,
               'pool': len(pool), 'pool_before_removal': n_pool0, 'pool_removed_known_defects': removed,
               'false_positive': {c: {**v, 'messages': dict(v['messages'])} for c, v in fp.items()}, 'detection': det, 'not_applicable': skipped,
               'options': {k: getattr(OPT, k) for k in ('split', 'max_samples', 'dup_max_samples', 'dup_corr', 'orient_samples')},
               'seconds': time.time() - t0}
    out = Path(args.out)
    write_once(out / f'faults_{args.base}_{args.tag}.json', payload, with_provenance=True)
    write_text_once(out / f'tables_{args.base}_{args.tag}.md', table(payload))
    print(table(payload))


def table(p):
    L = [f"# P8 fault validation: base `{p['base']}` ({p['draws']} draws x {p['sample']} samples; pool {p['pool']} of {p['pool_before_removal']}, "
         f"removed known defects {p['pool_removed_known_defects']})", '', '## False positives (clean draws, error or warning)', '',
         '| check | flagged / draws | rate (Wilson 95%) | findings |', '|---|---|---|---|']
    for c, v in p['false_positive'].items():
        lo, hi = wilson(v['flagged'], v['draws'])
        L.append(f"| {c} | {v['flagged']}/{v['draws']} | {v['flagged'] / max(1, v['draws']):.2f} ({lo:.2f}-{hi:.2f}) | "
                 f"{'; '.join(f'{k} x{n}' for k, n in v['messages'].items()) or '-'} |")
    L += ['', '## Detection by severity level (0 weakest .. 4 strongest)', '', '| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |', '|---|---|---|---|---|---|---|---|---|']
    fpr = {c: v['flagged'] / max(1, v['draws']) for c, v in p['false_positive'].items()}
    for name, lv in p['detection'].items():
        f = F.FAULTS[name]
        cells, rates = [], {}
        for k in range(F.LEVELS):
            e = lv.get(str(k)) or lv.get(k)
            if e is None:
                cells.append('-')
                continue
            rates[k] = e['detected'] / e['n']
            lo, hi = wilson(e['detected'], e['n'])
            cells.append(f"{rates[k]:.2f} ({lo:.2f}-{hi:.2f})")
        mn = next((k for k in sorted(rates) if rates[k] >= 0.9), None)
        top = sorted(rates)[-2:]
        ok = all(rates[k] >= 0.9 for k in top) and fpr[f.check] <= 0.05
        L.append(f"| {f.check} | {name} | " + ' | '.join(cells) + f" | {mn if mn is not None else 'none'} | {'yes' if ok else 'NO'} |")
    if p['not_applicable']:
        L += ['', 'Not applicable on this base: ' + '; '.join(f'{k} ({v})' for k, v in p['not_applicable'].items())]
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
