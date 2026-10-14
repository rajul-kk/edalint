"""POST HOC (after freeze eba0ab4): E005 axis / orientation check on CircuitNet-N14, all 8 dihedral orientations.

NOT part of the registered N14 protocol. N14 ran E005 only in development; this run happens after the freeze and is reported as POST HOC.
Reads the registered staging cache (E:/edalint_data/n14_registered_cache) through a read-only store; writes ONLY under results/n14_posthoc/.

Two layers per task, per design, per split (train = the seeded K=400-per-design fit sample the registered models used; test = every official test row):
  1. `edalint.checks.e005_orientation`, UNCHANGED (Options(orient_samples=None): all samples), for every input channel: median |corr| per orientation.
  2. Per-sample paired statistics for the channels declared below (same smoothing / grid / |corr| as E005, via checks._dihedral): median per
     orientation, best alternative, margin, fraction of samples in which the best alternative beats the stored orientation, and a paired
     bootstrap 95 % interval of the median difference (noise).

Channels, declared before the run (physical tie to the label):
  ir_drop     power_all (total cell power map = the current drawn from the grid; IR drop is a smooth function of it). Info: power_i, power_s, power_sca.
  congestion  RUDY and RUDY_pin (wire-demand estimates; the label is GR horizontal + vertical overflow). Info: macro_region (blockages).
  drc         GR_horizontal_overflow, GR_vertical_overflow (global-routing overflow, the closest upstream of detailed-route DRC), RUDY_long
              (the registered primary input). Info: cell_density.
Verdict rule (E005's own defaults): SUSPECT if, for a main channel, the best alternative orientation beats the stored one by >= 0.10 in median
|corr|, reaches >= 0.10, and wins in >= 75 % of samples. OK otherwise. `note` lists weaker but statistically clear gaps (bootstrap interval
excludes 0, margin >= 0.02); they are not SUSPECT.

  python -I scripts/posthoc_n14_e005.py [--tasks ir_drop,congestion,drc]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint import p4_n14 as P
from edalint.checks import Options, _dihedral, e005_orientation
from edalint.core import area_resize, pearson
from edalint.protocol import REPO_DIR, code_hash, git_dirty, git_head
from edalint.runner import write_once

OUT_DIR = REPO_DIR / 'results' / 'n14_posthoc'
LABEL = 'POST HOC (after freeze eba0ab4)'
ORIENTS = ('stored', 'flip_ud', 'flip_lr', 'rot180', 'transpose', 'rot90', 'rot270', 'antitranspose')
MAIN = {'ir_drop': ['power_all'], 'congestion': ['RUDY', 'RUDY_pin'],
        'drc': ['GR_horizontal_overflow', 'GR_vertical_overflow', 'RUDY_long']}
INFO = {'ir_drop': ['power_i', 'power_s', 'power_sca'], 'congestion': ['macro_region'], 'drc': ['cell_density']}
MARGIN, FRAC, MIN_CORR, NOISE_MARGIN = 0.10, 0.75, 0.10, 0.02
BOOT, BOOT_SEED = 2000, 20261014


class ReadOnlyStore:
    """Wraps the registered cache so that nothing can be written into it."""

    def __init__(self, store):
        self._s = store

    def get(self, *a):
        return self._s.get(*a)

    def has(self, *a):
        return self._s.has(*a)

    def get_presence(self, key):
        return self._s.get_presence(key)

    def put(self, *a):
        raise RuntimeError('read-only cache')

    def set_presence(self, *a):
        raise RuntimeError('read-only cache')


def smooth64(a, opt):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(area_resize(a, (opt.orient_grid, opt.orient_grid)), opt.orient_sigma)


def per_sample_corr(samples, channels, opt):
    """{channel: array (n_samples, 8)} of |corr| between the smoothed label and each orientation of the smoothed channel (E005's recipe)."""
    rows = {c: [] for c in channels}
    for s in samples:
        if np.all(s.label == s.label.flat[0]):
            continue
        lab = smooth64(s.label, opt)
        for c in channels:
            a = s.inputs.get(c)
            if a is None or np.all(a == a.flat[0]):
                continue
            d = _dihedral(smooth64(a, opt))
            rows[c].append([abs(pearson(lab, d[o])) for o in ORIENTS])
    return {c: np.array(v, dtype=float) for c, v in rows.items()}


def summarise(m, seed=BOOT_SEED):
    """m: (n, 8) |corr| per sample and orientation. Stored column 0. Returns the stored / best-orientation statistics."""
    if m.ndim != 2:                      # channel constant in every sample of this group (e.g. macro_region of a macro-free design)
        return {'n': 0}
    m = m[np.isfinite(m).all(axis=1)]
    n = len(m)
    if n == 0:
        return {'n': 0}
    med = np.median(m, axis=0)
    alt = int(np.argmax(med[1:])) + 1
    diff = m[:, alt] - m[:, 0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOT, n))
    boot = np.median(diff[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    margin = float(med[alt] - med[0])
    frac = float(np.mean(diff > 0))
    suspect = bool(margin >= MARGIN and med[alt] >= MIN_CORR and frac >= FRAC)
    return {'n': n, 'median_abs_corr': {o: float(v) for o, v in zip(ORIENTS, med)}, 'stored': float(med[0]),
            'best_orientation': ORIENTS[int(np.argmax(med))], 'best_alternative': ORIENTS[alt], 'best_alternative_corr': float(med[alt]),
            'margin_best_alt_minus_stored': margin, 'frac_samples_best_alt_beats_stored': frac,
            'median_diff_ci95': [float(lo), float(hi)], 'suspect': suspect,
            'note_weak_gap': bool((not suspect) and margin >= NOISE_MARGIN and lo > 0)}


def run_task(task, opt):
    run = P.Run()
    ds = P.open_task(run, task, stage=False)
    ds.store = ReadOnlyStore(ds.store)
    ds.stage_all(include_test=True)           # frozen protocol + matching code hash required; reads the registered cache only
    out = {'main_channels': MAIN[task], 'info_channels': INFO[task], 'designs': {}, 'pooled': {}}
    groups = [(d, sp) for (sp, d) in sorted(ds._wanted)] 
    chans = MAIN[task] + INFO[task]
    for des, sp in groups:
        ids = [i for i in ds.splits[des]]
        samples = (s for s in (ds.get(i) for i in ids) if s is not None)
        stats = per_sample_corr(samples, chans, opt)
        entry = {'split': sp, 'n_samples': len(ids), 'channels': {c: summarise(stats[c]) for c in chans}}
        res = e005_orientation(ds, Options(split=des, orient_samples=None))      # the registered implementation, unchanged
        entry['e005_unchanged'] = {'n_checked': res.metrics.get('n_checked'), 'median_abs_corr': res.metrics.get('median_abs_corr'),
                                   'status': res.status, 'findings': [(f.severity, f.message) for f in res.findings]}
        out['designs'][des] = entry
        print(task, des, sp, len(ids), {c: (round(entry['channels'][c].get('stored', float('nan')), 3),
                                          entry['channels'][c].get('best_alternative'),
                                          round(entry['channels'][c].get('margin_best_alt_minus_stored', float('nan')), 3))
                                      for c in MAIN[task]}, flush=True)
    for sp in ('train', 'test'):
        res = e005_orientation(ds, Options(split=sp, orient_samples=None))
        out['pooled'][sp] = {'n_checked': res.metrics.get('n_checked'), 'median_abs_corr': res.metrics.get('median_abs_corr'),
                             'status': res.status, 'findings': [(f.severity, f.message) for f in res.findings]}
    out['verdict'] = verdict(out, task)
    return out


def verdict(out, task):
    sus, notes = [], []
    for des, e in out['designs'].items():
        for c in MAIN[task]:
            s = e['channels'][c]
            if not s.get('n'):
                continue
            tag = f"{des}/{e['split']}/{c}"
            if s['suspect']:
                sus.append({'where': tag, 'stored': s['stored'], 'best_alternative': s['best_alternative'], 'alt_corr': s['best_alternative_corr'],
                            'margin': s['margin_best_alt_minus_stored'], 'frac': s['frac_samples_best_alt_beats_stored']})
            elif s['note_weak_gap']:
                notes.append({'where': tag, 'best_alternative': s['best_alternative'], 'margin': s['margin_best_alt_minus_stored'],
                              'ci95': s['median_diff_ci95']})
    return {'verdict': 'SUSPECT' if sus else 'OK', 'suspect_cells': sus, 'weak_gap_notes': notes}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='ir_drop,congestion,drc')
    a = ap.parse_args(argv)
    opt = Options(orient_samples=None)
    for task in a.tasks.split(','):
        res = run_task(task, opt)
        body = {'label': LABEL, 'check': 'E005 orientation, 8 dihedral orientations', 'task': task,
                'posthoc_note': 'Run after the N14 freeze (eba0ab4); not part of the registered protocol or results/n14/.',
                'options': {'orient_grid': opt.orient_grid, 'orient_sigma': opt.orient_sigma, 'verdict_margin': MARGIN, 'verdict_frac': FRAC},
                'git_head_at_run': git_head(), 'git_dirty_tracked_at_run': git_dirty(), 'code_hash_at_run': code_hash(), **res}
        p = write_once(OUT_DIR / f'e005_orientation_{task}.json', body, dataset='circuitnet_n14')
        print('wrote', p, res['verdict']['verdict'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
