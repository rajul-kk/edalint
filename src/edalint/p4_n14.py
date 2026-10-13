"""Registered runs for CircuitNet-N14 (protocol/protocol_n14.md). This module is part of the hashed code.

Stages (each writes ONE file under results/n14/ and refuses to overwrite; every stage first requires protocol/FROZEN_n14.json with a
matching code hash and src/protocol/scripts/tests matching HEAD):

  python -m edalint.p4_n14 faults                 full scan of every CSV-listed row of every used design (H2, H3, R1, duplicates, accounting)
  python -m edalint.p4_n14 models --task T        fit on the seeded K=400-per-design train sample, score EVERY official test row (T in ir_drop,
                                                  congestion, drc)
  python -m edalint.p4_n14 gap --task T           C1a (train only) and C1b (random split of the pooled capped rows)
  python -m edalint.p4_n14 verdicts               H1, H2, H3, R1, C1 and tables, mechanically, from the files above

Dry run (not registered): `--dry-run DIR --hold-out nvdla-small,Vortex-small` swaps the test split for held-out TRAIN designs, writes
under DIR, uses a throw-away freeze record in DIR, and REFUSES to run if any official test row is in the substituted test CSV.
Crash rule as in the ICCAD'23 protocol: a stage that crashed before writing its file may be re-launched; content never triggers a re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as M
from .baselines import stream
from .baselines.zoo import GREEN_EDGES, GreenSuperposition, KNNMap, MeanMap, PixelGBM, PixelRidge, ScalarMean, Zero
from .core import ProtocolError
from .datasets.circuitnet import (BBOX_CHANNELS, FEATURE_DIRS, IRD, ROUT, TASKS, N14Dataset, assemble_label, design_of,
                                  packaged_split_csvs, read_split_csv, stage_member, stream_tarball, tarball_status, load_manifest)
from .protocol import REPO_DIR, code_hash, require_frozen
from .runner import write_once, write_text_once

OUT_DIR = REPO_DIR / 'results' / 'n14'
TASK_LIST = ('ir_drop', 'congestion', 'drc')
K_FIT = 400                      # train samples per design used for fitting (seeded random)
K_GAP = 100                      # nested first-100 of the same seeded order: C1a
K_C1B = 400                      # per design, pooled train + test, C1b
SEED = 20261013
N_THR = 200
LABEL_THRESHOLD = {'ir_drop': 0.9885, 'drc': 0.1}                       # authors' utils/configs.py, authors-space label
IR_CLIP, DRC_CLIP = 50.0, 200.0
RAW_THR_IR = 10 ** (0.9885 * (math.log10(50) + 6) - 6)                  # 40.7786: the hotspot threshold in raw IR-drop units
PRIMARY = {'congestion': ('nrms', False), 'drc': ('ssim', True), 'ir_drop': ('roc_auc', True)}      # (metric, higher_is_better)
PUBLISHED_REF = {                                                       # best published N14 value for the primary metric (protocol section 6)
    'congestion': {'value': 0.020, 'model': 'Circuit GNN', 'source': 'CircuitNet 2.0 App. B (ICLR 2024), N14 NRMSE', 'weak_reference': True},
    'drc': {'value': 0.7136, 'model': 'CLDRoute LDM', 'source': 'CLDRoute 2607.16674 Table 5, N14 SSIM (split 10368/169/250)', 'weak_reference': True},
    'ir_drop': {'value': 0.8671, 'model': 'MAVIREC', 'source': 'CircuitNet 2.0 App. B.2, N14 ROC-AUC (FPR<=5%, split unstated)', 'weak_reference': True},
}
SATURATION = 0.99                # an SSIM-primary pair is not evaluable if the mean_map SSIM is at least this
H1_FRACTION, H2_Y_POOLED, H2_Y_DESIGN, H3_RATE, R1_RATE, C1_GAP = 0.8, 0.01, 0.05, 0.05, 0.5, 0.10
BASELINES = {'ir_drop': ['zero', 'mean_map', 'scalar', 'ridge', 'knn', 'green', 'gbm'],
             'congestion': ['zero', 'mean_map', 'scalar', 'ridge', 'knn', 'gbm'],
             'drc': ['zero', 'mean_map', 'scalar', 'ridge', 'knn', 'gbm']}
# Frozen hyperparameters: chosen by results/dev/n14_hparam_cv_round2.json (round 1: n14_hparam_cv.json; leave-one-design-out on train, K=100 per design).
FROZEN_HPARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'ir_drop': {'ridge': {'alpha': 100.0}, 'knn': {'k': 1}, 'gbm': {'max_iter': 100, 'max_depth': 3}, 'green': {'edges': 'coarse'}},
    'congestion': {'ridge': {'alpha': 10.0}, 'knn': {'k': 5}, 'gbm': {'max_iter': 100, 'max_depth': 3}},
    'drc': {'ridge': {'alpha': 0.0001}, 'knn': {'k': 5}, 'gbm': {'max_iter': 100, 'max_depth': 3}},
}
SCAN_FEATURES = {IRD: ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t', 'IR_drop'],
                 ROUT: ['macro_region', 'RUDY', 'RUDY_pin', 'cell_density', 'RUDY_long', 'RUDY_short', 'RUDY_pin_long',
                        'eGR_horizontal_overflow', 'eGR_vertical_overflow', 'GR_horizontal_overflow', 'GR_vertical_overflow', 'DRC_all']}
TASK_TAR = {t: TASKS[t]['tar'] for t in TASK_LIST}


# -- run configuration and guards -----------------------------------------------------------------------------------------------
@dataclass
class Run:
    out_dir: Path = OUT_DIR
    cache_root: Path = Path('E:/edalint_data/n14_registered_cache')
    root: Optional[str] = None
    csvs: Optional[Tuple[str, str]] = None           # None = the packaged official files
    protocol_dir: Optional[str] = None
    dry_run: bool = False
    seed: int = SEED
    manifest_path: Optional[str] = None


def _official_test_names() -> set:
    return set(read_split_csv(packaged_split_csvs()[1]))


def _preflight(run: Run, stage: str):
    if run.dry_run:
        if run.csvs is None:
            raise ProtocolError('refused: a dry run needs substituted split CSVs')
        clash = set(read_split_csv(run.csvs[1])) & _official_test_names()
        if clash or Path(run.csvs[1]).resolve() == packaged_split_csvs()[1].resolve():
            raise ProtocolError(f'refused: the dry-run test split contains {len(clash)} official test rows')
        if Path(run.out_dir).resolve().is_relative_to((REPO_DIR / 'results').resolve()):
            raise ProtocolError('refused: dry-run output must not be written under results/')
        return require_frozen(run.protocol_dir, what=f'dry-run stage {stage}', dataset='n14')
    if run.csvs is not None or run.protocol_dir is not None:
        raise ProtocolError('refused: registered stages use the official split files and the repository protocol directory only')
    info = require_frozen(None, what=f'registered stage {stage}', dataset='n14')
    r = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no', '--', 'src', 'protocol', 'scripts', 'tests'],
                       capture_output=True, text=True, cwd=REPO_DIR)
    if r.stdout.strip():
        raise ProtocolError(f'refused: files differ from HEAD ({r.stdout.strip().splitlines()[:5]}); commit before a registered run')
    return info


def make_dry_run(dry_dir: Path, hold_out: Sequence[str], seed: int = SEED) -> Run:
    """Substitute the test split by whole held-out TRAIN designs. Writes the CSVs and a throw-away freeze record under dry_dir."""
    dry_dir = Path(dry_dir)
    (dry_dir / 'csv').mkdir(parents=True, exist_ok=True)
    (dry_dir / 'protocol').mkdir(parents=True, exist_ok=True)
    tr_csv, te_csv = packaged_split_csvs()
    rows = [ln for ln in Path(tr_csv).read_text().splitlines() if ln.strip()]
    held = [r for r in rows if design_of(r.split(',')[0].split('/')[-1].rsplit('.', 1)[0]) in hold_out]
    rest = [r for r in rows if r not in set(held)]
    if not held or not rest:
        raise ProtocolError('dry run: hold-out designs not found in the train CSV')
    (dry_dir / 'csv' / 'train.csv').write_text('\n'.join(rest) + '\n')
    (dry_dir / 'csv' / 'test.csv').write_text('\n'.join(held) + '\n')
    (dry_dir / 'protocol' / 'FROZEN_n14.json').write_text(json.dumps({'dataset': 'n14', 'code_hash': code_hash(), 'dry_run': True}))
    return Run(out_dir=dry_dir / 'out', cache_root=dry_dir / 'cache', csvs=(str(dry_dir / 'csv' / 'train.csv'), str(dry_dir / 'csv' / 'test.csv')),
               protocol_dir=str(dry_dir / 'protocol'), dry_run=True, seed=seed)


# -- seeded fit sample --------------------------------------------------------------------------------------------------------
def seeded_order(design: str, names: Sequence[str], seed: int = SEED) -> List[str]:
    """Names of a design (sorted CSV order) in a seeded random order; the first K are the fit sample (nested prefixes)."""
    rng = np.random.default_rng([seed, zlib.crc32(design.encode())])
    s = sorted(names)
    return [s[i] for i in rng.permutation(len(s))]


def fit_selector(k: int = K_FIT, seed: int = SEED):
    def sel(split, design, names):
        if split != 'train' or len(names) <= k:
            return names
        return sorted(seeded_order(design, names, seed)[:k])
    return sel


def open_task(run: Run, task: str, stage: bool = True) -> N14Dataset:
    return N14Dataset(task=task, mode='authors', root=run.root, cache_dir=str(Path(run.cache_root) / task), split_csvs=run.csvs,
                      protocol_dir=run.protocol_dir, select=fit_selector(K_FIT, run.seed), stage=stage,
                      manifest_path=run.manifest_path)


def fit_indices(ds: N14Dataset, seed: int) -> Dict[str, Any]:
    """Seed and, per train design, the indices (into the sorted CSV-listed names) of the fit sample."""
    out = {'seed': seed, 'k': K_FIT, 'designs': {}}
    for (split, des), _ in sorted(ds._wanted.items()):
        if split == 'train':
            listed = sorted(ds._listed[des])
            pos = {n: i for i, n in enumerate(listed)}
            out['designs'][des] = {'n_listed': len(listed), 'indices': sorted(pos[n] for n in ds._wanted[(split, des)])}
    return out


# -- official ROC machinery (authors' utils/metrics.py calculated_score / calculate_all / roc_prc) ------------------------------
THRESHOLDS = np.linspace(0, 1, N_THR, endpoint=False)


def threshold_counts(pred: np.ndarray, gold: np.ndarray, thr: float):
    """(tp, fp, fn, tn) arrays over the 200 thresholds; prediction binarised with >= t, label with >= thr (authors' lines 235-239)."""
    y = np.ravel(gold) >= thr
    p = np.nan_to_num(np.ravel(pred).astype(np.float64))
    pos, neg = np.sort(p[y]), np.sort(p[~y])
    tp = pos.size - np.searchsorted(pos, THRESHOLDS, side='left')
    fp = neg.size - np.searchsorted(neg, THRESHOLDS, side='left')
    return tp, fp, pos.size - tp, neg.size - fp


class RocAcc:
    """Sums of TPR / FPR / precision over the samples that the authors' calculate_all does not skip, per threshold."""

    def __init__(self):
        self.tpr, self.fpr, self.pre, self.num = (np.zeros(N_THR) for _ in range(4))
        self.n_samples = 0
        self.n_no_positive = 0

    def add(self, pred, gold, thr):
        tp, fp, fn, tn = (a.astype(np.float64) for a in threshold_counts(pred, gold, thr))
        self.n_samples += 1
        self.n_no_positive += int((tp[0] + fn[0]) == 0)
        skip = ((fp == 0) & (tn == 0)) | ((tp == 0) & (fn == 0)) | ((tp == 0) & (fp == 0))
        ok = ~skip
        with np.errstate(divide='ignore', invalid='ignore'):
            self.tpr[ok] += tp[ok] / (tp[ok] + fn[ok])
            self.fpr[ok] += fp[ok] / (fp[ok] + tn[ok])
            self.pre[ok] += tp[ok] / (tp[ok] + fp[ok])
        self.num[ok] += 1

    def merge(self, o: 'RocAcc'):
        for a in ('tpr', 'fpr', 'pre', 'num'):
            setattr(self, a, getattr(self, a) + getattr(o, a))
        self.n_samples += o.n_samples
        self.n_no_positive += o.n_no_positive
        return self

    def to_dict(self):
        return {'tpr': self.tpr.tolist(), 'fpr': self.fpr.tolist(), 'pre': self.pre.tolist(), 'num': self.num.tolist(),
                'n_samples': self.n_samples, 'n_no_positive': self.n_no_positive}

    @classmethod
    def from_dict(cls, d):
        r = cls()
        r.tpr, r.fpr, r.pre, r.num = (np.array(d[k], float) for k in ('tpr', 'fpr', 'pre', 'num'))
        r.n_samples, r.n_no_positive = d['n_samples'], d['n_no_positive']
        return r

    def area(self, anchored: bool = False) -> float:
        """ROC area exactly as the authors compute it (dedupe on FPR in threshold order, sort, append (1, 1), trapezoid); NaN if no point.
        The authors' curve has no (0, 0) anchor and drops thresholds at which nothing is predicted positive, so a constant predictor scores 0, not
        0.5. anchored=True adds the (0, 0) point (secondary variant, reported next to the official one)."""
        pts = [(self.fpr[i] / self.num[i], self.tpr[i] / self.num[i]) for i in range(N_THR) if self.num[i] > 0]
        if not pts:
            return float('nan')
        fl, tl = [], []
        for f, t in pts:
            if f not in fl:
                fl.append(f)
                tl.append(t)
        fl.reverse()
        tl.reverse()
        if anchored:
            fl.append(0.0)
            tl.append(0.0)
        pairs = sorted(zip(fl, tl))
        fl = [p[0] for p in pairs] + [1.0]
        tl = [p[1] for p in pairs] + [1.0]
        return float(sum((tl[i] + tl[i + 1]) * (fl[i + 1] - fl[i]) / 2 for i in range(len(tl) - 1)))


class PrimaryAcc:
    """Per-sample NRMS / SSIM and the official ROC accumulator for a group of samples (a design, a fold, ...)."""

    def __init__(self, task: str):
        self.task, self.nrms, self.ssim, self.roc = task, [], [], RocAcc()

    def add(self, pred, gold):
        self.nrms.append(M.nrms_official(pred, gold))         # the authors' pipeline: clamp to [0, 1], quantise to uint8 (metrics.to_u8)
        self.ssim.append(M.ssim_official(pred, gold))
        if self.task in LABEL_THRESHOLD:
            self.roc.add(pred, gold, LABEL_THRESHOLD[self.task])

    def value(self, anchored: bool = False) -> float:
        m = PRIMARY[self.task][0]
        if m == 'roc_auc':
            return self.roc.area(anchored=anchored)
        v = np.array(self.nrms if m == 'nrms' else self.ssim, float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan')


def skill(task: str, value: float, ref: float) -> float:
    """Gain over the reference as a fraction of the available headroom (lower-better: 1 - v/ref; higher-better: (v - ref)/(1 - ref))."""
    higher = PRIMARY[task][1]
    if not (np.isfinite(value) and np.isfinite(ref)):
        return float('nan')
    return (value - ref) / (1 - ref) if higher else 1 - value / ref


# -- baselines ------------------------------------------------------------------------------------------------------------------
def make_baseline(name: str, task: str, hp: Optional[dict] = None):
    h = (hp or FROZEN_HPARAMS)[task]
    if name == 'zero':
        return Zero()
    if name == 'mean_map':
        return MeanMap()
    if name == 'scalar':
        return ScalarMean()
    if name == 'ridge':
        return PixelRidge(alpha=h['ridge']['alpha'])
    if name == 'knn':
        return KNNMap(k=h['knn']['k'])
    if name == 'gbm':
        return PixelGBM(max_iter=h['gbm']['max_iter'], max_leaf_nodes=None, max_depth=h['gbm']['max_depth'])
    if name == 'green':
        return GreenSuperposition(edges=GREEN_EDGES[h['green']['edges']])
    raise KeyError(name)


def _by_design(ds, split_ids: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for i in split_ids:
        out.setdefault(ds.meta(i)['design'], []).append(i)
    return {d: out[d] for d in sorted(out)}


# -- stage: faults --------------------------------------------------------------------------------------------------------------
def _sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes() + str(a.shape).encode()).hexdigest()


def _scan_member(feat: str, arr: np.ndarray) -> dict:
    a = np.asarray(arr)
    rec = {'sha1': _sha(a), 'shape': list(a.shape), 'zero': bool(not a.any()), 'const': bool(np.ptp(a) == 0), 'finite': bool(np.isfinite(a).all())}
    if feat == 'IR_drop':
        lab = assemble_label('ir_drop', 'authors', lambda f: stage_member(f, a, 'authors'))
        raw = np.squeeze(a).astype(np.float64)
        rec.update(lab_zero=bool(np.all(lab == 0)), lab_const=bool(np.ptp(lab) == 0), n_pos=int((lab >= LABEL_THRESHOLD['ir_drop']).sum()),
                   n_pix=int(lab.size), raw_max=float(raw.max()), frac_gt_clip=float((raw > IR_CLIP).mean()),
                   n_raw_ge_thr=int((raw >= RAW_THR_IR).sum()), n_saturated=int((raw > IR_CLIP).sum()))
    elif feat == 'DRC_all':
        lab = assemble_label('drc', 'authors', lambda f: stage_member(f, a, 'authors'))
        rec.update(lab_zero=bool(np.all(lab == 0)), lab_const=bool(np.ptp(lab) == 0), n_pos=int((lab >= LABEL_THRESHOLD['drc']).sum()),
                   n_pix=int(lab.size), raw_max=float(a.max()), frac_gt_clip=float((a > DRC_CLIP).mean()))
    return rec


def scan_tarball(run: Run, subset: str, design: str, listed: set) -> Dict[str, Dict[str, dict]]:
    """Stream one tarball once; per listed sample name and feature a compact record (hash, zero/constant/finite flags, label statistics)."""
    feats = SCAN_FEATURES[subset]
    recs: Dict[str, Dict[str, dict]] = {}
    dirs = {FEATURE_DIRS[f]: f for f in feats}
    tar = Path(run.root or 'E:/edalint_data/circuitnet_n14') / 'CircuitNet-N14' / subset / f'{design}.tar.gz'

    def handle(feat, name, arr):
        recs.setdefault(name, {})[feat] = _scan_member(feat, arr)
    pres = stream_tarball(tar, design, dirs, listed, handle)
    return {'recs': recs, 'present': {f: sorted(v) for f, v in pres.items()}}


def stage_faults(run: Run):
    _preflight(run, 'faults')
    t0 = time.perf_counter()
    tr_csv, te_csv = run.csvs or packaged_split_csvs()
    rows = {'train': read_split_csv(tr_csv), 'test': read_split_csv(te_csv)}
    split_of, listed_by_design = {}, {}
    for sp, ns in rows.items():
        for n in ns:
            split_of[n] = sp
            listed_by_design.setdefault(design_of(n), set()).add(n)
    all_listed = set(rows['train']) | set(rows['test'])
    manifest = load_manifest(run.manifest_path)
    out: Dict[str, Any] = {'label': 'DRY RUN' if run.dry_run else 'REGISTERED', 'tasks': {t: {'designs': {}} for t in TASK_LIST}, 'accounting': {}, 'seconds': {}}
    hash_tab: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    for subset in (IRD, ROUT):
        tasks = [t for t in TASK_LIST if TASK_TAR[t] == subset]
        for design in sorted(listed_by_design):
            ok, why = tarball_status(run.root or 'E:/edalint_data/circuitnet_n14', subset, design, manifest)
            if not ok:
                out['accounting'].setdefault(subset, {})[design] = {'unusable': why}
                continue
            ts = time.perf_counter()
            listed = listed_by_design[design]
            sc = scan_tarball(run, subset, design, listed)
            recs, present = sc['recs'], sc['present']
            label_feats = {'ir_drop': ['IR_drop'], 'drc': ['DRC_all'], 'congestion': ['GR_horizontal_overflow', 'GR_vertical_overflow']}
            all_members = set().union(*[set(v) for v in present.values()])
            out['accounting'].setdefault(subset, {})[design] = {
                'n_listed': len(listed), 'n_members_union': len(all_members), 'n_unlisted_members': len(all_members - all_listed),
                'n_per_feature': {f: len(v) for f, v in present.items()}}
            for name, fr in recs.items():
                for f, r in fr.items():
                    if not (r['zero'] or r['const']):
                        hash_tab.setdefault((f, r['sha1']), []).append((split_of[name], design, name))
            for t in tasks:
                spec = TASKS[t]
                req = list(dict.fromkeys(spec['inputs'] + spec['label']))
                sp_names = {sp: sorted(n for n in listed if split_of[n] == sp) for sp in ('train', 'test')}
                for sp, names in sp_names.items():
                    if not names:
                        continue
                    complete = [n for n in names if n in recs and all(f in recs[n] for f in req)]
                    d = {'split': sp, 'n_listed': len(names), 'n_complete': len(complete), 'n_missing': len(names) - len(complete),
                         'missing_ids': [n for n in names if n not in complete][:50], 'flags': {}, 'ids': complete}
                    fl: Dict[str, List[str]] = {k: [] for k in ('zero_label', 'const_label', 'bbox_zero', 'no_positive', 'all_positive', 'nonfinite')}
                    clip_rows = []
                    for n in complete:
                        r = recs[n]
                        if any(not r[f]['finite'] for f in req):
                            fl['nonfinite'].append(n)
                        if t == 'congestion':
                            lz = all(r[f]['zero'] for f in label_feats[t])
                            lc = all(r[f]['const'] for f in label_feats[t])
                        else:
                            lz, lc = r[label_feats[t][0]]['lab_zero'], r[label_feats[t][0]]['lab_const']
                        if lz:
                            fl['zero_label'].append(n)
                        elif lc:
                            fl['const_label'].append(n)
                        bb = [f for f in spec['inputs'] if f in BBOX_CHANNELS]
                        if bb and all(r[f]['zero'] for f in bb):
                            fl['bbox_zero'].append(n)
                        if t in LABEL_THRESHOLD:
                            lr = r[label_feats[t][0]]
                            if lr['n_pos'] == 0:
                                fl['no_positive'].append(n)
                            if lr['n_pos'] == lr['n_pix']:
                                fl['all_positive'].append(n)
                            clip_rows.append({k: lr[k] for k in ('raw_max', 'frac_gt_clip') + (('n_raw_ge_thr', 'n_saturated') if t == 'ir_drop' else ())} | {'id': n})
                    d['flags'] = fl
                    d['clip_rows'] = clip_rows
                    out['tasks'][t]['designs'][design] = d
            out['seconds'][f'{subset}/{design}'] = time.perf_counter() - ts
            print('scanned', subset, design, round(out['seconds'][f'{subset}/{design}'], 1), 's', flush=True)
    dups: Dict[str, Any] = {}
    for t in TASK_LIST:
        spec = TASKS[t]
        feats = set(spec['inputs'] + spec['label'])
        lab = set(spec['label'])
        cross, within_test = [], []
        for (f, h), ents in hash_tab.items():
            if f not in feats:
                continue
            sp = {e[0] for e in ents}
            item = {'feature': f, 'is_label': f in lab, 'members': [list(e) for e in ents][:20], 'n': len(ents)}
            if sp == {'train', 'test'}:
                cross.append(item)
            elif sp == {'test'} and len(ents) > 1:
                within_test.append(item)
        dups[t] = {'cross_split_exact': cross, 'within_test_exact': within_test}
    out['duplicates'] = dups
    out['seconds']['total'] = time.perf_counter() - t0
    write_once(Path(run.out_dir) / 'faults.json', out, dataset='n14')


# -- stage: models ---------------------------------------------------------------------------------------------------------------
def _score_rows(task: str, ds, ids_by_design: Dict[str, List[str]], baselines: Dict[str, Any]):
    """One pass over the samples (outer) with every fitted baseline (inner). Returns rows and per-design accumulators."""
    thr = LABEL_THRESHOLD.get(task)
    rows = {n: [] for n in baselines}
    acc = {n: {} for n in baselines}
    for des, ids in ids_by_design.items():
        for n in baselines:
            acc[n][des] = RocAcc()
        for sid in ids:
            s = ds.get(sid)
            if s is None:
                continue
            for n, b in baselines.items():
                t0 = time.perf_counter()
                pred = np.nan_to_num(b.predict(s))
                dt = time.perf_counter() - t0
                row = {'id': sid, 'design': des, 'seconds': dt, 'nrms': M.nrms_official(pred, s.label), 'ssim': M.ssim_official(pred, s.label),
                       'nrms_float': M.nrms(pred, s.label), 'ssim_float': M.ssim(pred, s.label), 'nmae': M.nmae(pred, s.label),
                       'pearson': M.pearson_r(pred, s.label)}
                if thr is not None:
                    row['roc_auc_sample'] = M.roc_auc(pred, s.label, thr)
                    acc[n][des].add(pred, s.label, thr)
                rows[n].append(row)
    return rows, acc


def stage_models(run: Run, task: str):
    _preflight(run, f'models/{task}')
    t0 = time.perf_counter()
    ds = open_task(run, task)
    ds.stage_all(include_test=True)               # one design at a time; drops listed rows that lack files (logged in ds.skipped)
    train = ds.ids('train')
    test = _by_design(ds, ds.ids('test'))
    fitted, fit_s = {}, {}
    for n in BASELINES[task]:
        b = make_baseline(n, task)
        ts = time.perf_counter()
        b.fit(stream(ds, train))
        fit_s[n] = time.perf_counter() - ts
        fitted[n] = b
        print(task, n, 'fit', round(fit_s[n], 1), 's', flush=True)
    ts = time.perf_counter()
    rows, acc = _score_rows(task, ds, test, fitted)
    res = {n: {'hparams': FROZEN_HPARAMS[task].get(n, {}), 'fit_seconds': fit_s[n], 'rows': rows[n],
               'official_roc': {d: a.to_dict() for d, a in acc[n].items()}} for n in fitted}
    out = {'label': 'DRY RUN' if run.dry_run else 'REGISTERED', 'task': task, 'primary': PRIMARY[task],
           'fit_sample': fit_indices(ds, run.seed), 'fit_ids': train, 'eval_ids': {d: v for d, v in test.items()},
           'skipped_by_loader': ds.skipped, 'unlisted_in_csv': ds.unlisted, 'score_seconds': time.perf_counter() - ts,
           'results': res, 'seconds_total': time.perf_counter() - t0}
    write_once(Path(run.out_dir) / f'models_{task}.json', out, dataset='n14')


# -- stage: gap (C1) --------------------------------------------------------------------------------------------------------------
def _fold_skill(ds, task: str, fit_ids: Sequence[str], eval_ids: Sequence[str], alpha: float, anchored: bool = False) -> float:
    if not fit_ids or not eval_ids:
        return float('nan')
    r = PixelRidge(alpha=alpha).fit(stream(ds, fit_ids))
    sc = ScalarMean().fit(stream(ds, fit_ids))
    a_r, a_s = PrimaryAcc(task), PrimaryAcc(task)
    for sid in eval_ids:
        s = ds.get(sid)
        a_r.add(np.nan_to_num(r.predict(s)), s.label)
        a_s.add(sc.predict(s), s.label)
    return skill(task, a_r.value(anchored), a_s.value(anchored))


def _capped(ds, split: str, k: int, seed: int) -> List[str]:
    """Ids of `split` that belong to the first k of the seeded order of their design's CSV-listed names."""
    keep = []
    have = set(ds.ids(split))
    for (sp, des), _ in sorted(ds._wanted.items()):
        if sp == split:
            first = set(seeded_order(des, sorted(ds._listed[des]), seed)[:k])
            keep += [n for n in sorted(ds._wanted[(sp, des)]) if n in first and n in have]
    return keep


def stage_gap(run: Run, task: str):
    _preflight(run, f'gap/{task}')
    t0 = time.perf_counter()
    ds = open_task(run, task)
    ds.stage_all(include_test=True)
    alpha = FROZEN_HPARAMS[task]['ridge']['alpha']
    rng = np.random.default_rng(run.seed)
    tr = _capped(ds, 'train', K_GAP, run.seed)
    des_tr = np.array([ds.meta(i)['design'] for i in tr])
    perm = rng.permutation(len(tr))
    rand = [_fold_skill(ds, task, [tr[j] for j in range(len(tr)) if perm[j] % 5 != f], [tr[j] for j in range(len(tr)) if perm[j] % 5 == f], alpha)
            for f in range(5)]
    dsg = [_fold_skill(ds, task, [tr[j] for j in range(len(tr)) if des_tr[j] != d], [tr[j] for j in range(len(tr)) if des_tr[j] == d], alpha)
           for d in sorted(set(des_tr))]
    c1a = {'n_samples': len(tr), 'skill_random_folds': rand, 'skill_design_folds': dsg, 'skill_random': float(np.nanmean(rand)),
           'skill_design': float(np.nanmean(dsg)), 'gap': float(np.nanmean(rand) - np.nanmean(dsg)), 'design_fold_order': sorted(set(des_tr))}
    pool = _capped(ds, 'train', K_C1B, run.seed) + _capped(ds, 'test', K_C1B, run.seed)
    n_tr = int(round(len(pool) * 6790 / 10334))
    reps = []
    for rep in range(5):
        p = np.random.default_rng(rep).permutation(len(pool))
        reps.append(_fold_skill(ds, task, [pool[j] for j in p[:n_tr]], [pool[j] for j in p[n_tr:]], alpha))
    c1b = {'n_pool': len(pool), 'n_fit': n_tr, 'skill_random_reps': reps, 'skill_random': float(np.nanmean(reps))}
    if PRIMARY[task][0] == 'roc_auc':                     # declared sensitivity analysis: the same C1b with the (0, 0)-anchored ROC area
        reps_a = []
        for rep in range(5):
            p = np.random.default_rng(rep).permutation(len(pool))
            reps_a.append(_fold_skill(ds, task, [pool[j] for j in p[:n_tr]], [pool[j] for j in p[n_tr:]], alpha, anchored=True))
        c1b.update(skill_random_reps_anchored=reps_a, skill_random_anchored=float(np.nanmean(reps_a)))
    write_once(Path(run.out_dir) / f'gap_{task}.json', {'label': 'DRY RUN' if run.dry_run else 'REGISTERED', 'task': task, 'C1a': c1a, 'C1b': c1b,
                                                        'seconds': time.perf_counter() - t0}, dataset='n14')


# -- verdicts ---------------------------------------------------------------------------------------------------------------------
def _load(d: Path, name: str):
    return json.loads((Path(d) / name).read_text(encoding='utf-8'))


def design_values(task: str, bres: dict, anchored: bool = False) -> Dict[str, float]:
    """Primary metric per test design for one baseline's result block (official ROC area, or the mean of the finite per-sample values)."""
    m = PRIMARY[task][0]
    out = {}
    if m == 'roc_auc':
        for d, a in bres['official_roc'].items():
            out[d] = RocAcc.from_dict(a).area(anchored=anchored or bool(bres.get('anchored')))
    else:
        for d in sorted({r['design'] for r in bres['rows']}):
            v = np.array([r[m] for r in bres['rows'] if r['design'] == d], float)
            v = v[np.isfinite(v)]
            out[d] = float(v.mean()) if v.size else float('nan')
    return out


def pooled_value(task: str, bres: dict, anchored: bool = False) -> float:
    m = PRIMARY[task][0]
    if m == 'roc_auc':
        tot = RocAcc()
        for a in bres['official_roc'].values():
            tot.merge(RocAcc.from_dict(a))
        return tot.area(anchored=anchored or bool(bres.get('anchored')))
    v = np.array([r[m] for r in bres['rows']], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


def macro(vals: Dict[str, float]) -> float:
    v = np.array(list(vals.values()), float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')


def h1_task(task: str, models: dict) -> dict:
    higher = PRIMARY[task][1]
    res = models['results']
    dv = {n: design_values(task, r) for n, r in res.items()}
    mac = {n: macro(v) for n, v in dv.items()}
    pooled = {n: pooled_value(task, r) for n, r in res.items()}
    ref = PUBLISHED_REF[task]

    def gain(v, base):
        return (v - base) if higher else (base - v)
    out = {'task': task, 'metric': PRIMARY[task][0], 'higher_is_better': higher,
           'secondary_roc_anchored_design_macro': ({n: macro({d: RocAcc.from_dict(a).area(anchored=True) for d, a in r['official_roc'].items()}) for n, r in res.items()} if task in LABEL_THRESHOLD and PRIMARY[task][0] == 'roc_auc' else None), 'weak_reference': ref['weak_reference'], 'reference': ref,
           'design_values': dv, 'design_macro': mac, 'sample_pooled': pooled}
    base = mac['mean_map']
    undefined = [n for n in ('mean_map',) if not np.isfinite(mac[n])] + [d for d, v in dv['mean_map'].items() if not np.isfinite(v)]
    if task in LABEL_THRESHOLD and PRIMARY[task][0] != 'roc_auc':       # the authors' ROC-type metric as a secondary (rows without a positive pixel are skipped)
        out['secondary_roc_official_design_macro'] = {n: macro({d: RocAcc.from_dict(a).area() for d, a in r['official_roc'].items()}) for n, r in res.items()}
    if PRIMARY[task][0] == 'ssim' and np.isfinite(mac['mean_map']) and mac['mean_map'] >= SATURATION:
        out.update(verdict='not evaluable', reason=f'metric saturated: the mean_map SSIM is {mac["mean_map"]:.4f} >= {SATURATION}')
        return out
    if not np.isfinite(base) or all(not np.isfinite(v) for v in mac.values()):
        out.update(verdict='not evaluable', reason='the primary metric is undefined on the test split by the official procedure', undefined=undefined)
        return out
    gains = {n: gain(v, base) for n, v in mac.items() if np.isfinite(v)}
    best = max(gains, key=gains.get)
    g_pub = gain(ref['value'], base)
    out.update(gains=gains, best_baseline=best, best_gain=gains[best], published_gain=g_pub, undefined_designs=undefined,
               sign_count_designs_beating_mean_map={n: int(sum(gain(dv[n][d], dv['mean_map'][d]) > 0 for d in dv['mean_map'] if np.isfinite(dv[n][d]) and np.isfinite(dv['mean_map'][d]))) for n in dv})
    if g_pub <= 0:
        out.update(verdict=bool(gains[best] > 0), degenerate=True, note='published reference is not better than our mean_map')
    else:
        out.update(verdict=bool(gains[best] > 0 and gains[best] >= H1_FRACTION * g_pub), degenerate=False, ratio=gains[best] / g_pub)
    return out


def _anchored_view(models: dict) -> dict:
    """The same result block with the ROC area replaced by the anchored one (every other number untouched)."""
    import copy
    m = copy.deepcopy(models)
    for r in m['results'].values():
        r['anchored'] = True
    return m


def h2_h3_r1(faults: dict, models: Optional[Dict[str, dict]] = None) -> dict:
    out: Dict[str, Any] = {'H2': {}, 'H3': {}, 'R1': {}}
    h2_true, h3_true = False, False
    for t in TASK_LIST:
        des = {d: v for d, v in faults['tasks'][t]['designs'].items() if v['split'] == 'test'}
        n_tot = sum(v['n_complete'] for v in des.values())
        items = {}
        material = False
        for k in ('zero_label', 'const_label', 'bbox_zero', 'nonfinite'):
            per = {d: (len(v['flags'][k]), v['n_complete']) for d, v in des.items()}
            pooled = sum(a for a, _ in per.values()) / n_tot if n_tot else float('nan')
            dmax = max([a / b for a, b in per.values() if b], default=0.0)
            mat = bool(pooled >= H2_Y_POOLED or dmax >= H2_Y_DESIGN)
            items[k] = {'per_design': per, 'pooled_rate': pooled, 'max_design_rate': dmax, 'material': mat,
                        'applicable': not (k == 'bbox_zero' and t == 'ir_drop')}
            material |= mat and items[k]['applicable']
        dup = faults['duplicates'][t]['cross_split_exact']
        items['cross_split_exact_duplicates'] = {'n': len(dup), 'material': bool(dup), 'examples': dup[:5]}
        material |= bool(dup)
        out['H2'][t] = {'n_test_scanned': n_tot, 'items': items, 'material': material}
        h2_true |= material
        sh = {}
        if t in LABEL_THRESHOLD:
            for k in ('no_positive', 'all_positive'):
                sh[k] = sum(len(v['flags'][k]) for v in des.values()) / n_tot if n_tot else float('nan')
        sh['const_label_nrms_undefined'] = sum(len(v['flags']['const_label']) + len(v['flags']['zero_label']) for v in des.values()) / n_tot if n_tot else float('nan')
        if models and t in models:
            n1 = sum(1 for r in models[t]['results']['mean_map']['rows'] if r['ssim'] == 1.0)
            sh['ssim_exactly_1_mean_map'] = n1 / max(len(models[t]['results']['mean_map']['rows']), 1)
        hit = bool(any(v >= H3_RATE for v in sh.values() if np.isfinite(v)))
        out['H3'][t] = {'shares': sh, 'per_design_no_positive': {d: (len(v['flags']['no_positive']), v['n_complete']) for d, v in des.items()} if t in LABEL_THRESHOLD else None,
                        'hit': hit}
        h3_true |= hit
        if t == 'ir_drop':
            cr = [r for v in des.values() for r in v['clip_rows']]
            if cr:
                share = sum(r['raw_max'] > IR_CLIP for r in cr) / len(cr)
                n_ge = sum(r['n_raw_ge_thr'] for r in cr)
                out['R1'] = {'n_test': len(cr), 'share_raw_max_above_clip': share, 'mean_frac_pixels_clipped': float(np.mean([r['frac_gt_clip'] for r in cr])),
                             'share_of_raw_hotspot_pixels_saturated': (sum(r['n_saturated'] for r in cr) / n_ge) if n_ge else float('nan'),
                             'material': bool(share >= R1_RATE)}
        if t == 'drc':
            cr = [r for v in des.values() for r in v['clip_rows']]
            if cr:
                out['R1_drc_informational'] = {'n_test': len(cr), 'share_raw_max_above_200': sum(r['raw_max'] > DRC_CLIP for r in cr) / len(cr),
                                               'mean_frac_pixels_clipped': float(np.mean([r['frac_gt_clip'] for r in cr]))}
    out['H2_verdict'], out['H3_verdict'] = h2_true, h3_true
    return out


def c1_verdict(gaps: Dict[str, dict], models: Dict[str, dict]) -> dict:
    out, n = {}, 0
    for t, g in gaps.items():
        res = models[t]['results']
        off_r, off_s = macro(design_values(t, res['ridge'])), macro(design_values(t, res['scalar']))
        sk_off = skill(t, off_r, off_s)
        gap = g['C1b']['skill_random'] - sk_off
        out[t] = {'skill_official': sk_off, 'skill_random': g['C1b']['skill_random'], 'gap_C1b': gap, 'C1a_gap': g['C1a']['gap'],
                  'material': bool(np.isfinite(gap) and gap >= C1_GAP)}
        if 'skill_random_anchored' in g['C1b']:
            ar, as_ = (macro({d: RocAcc.from_dict(a).area(anchored=True) for d, a in res[n]['official_roc'].items()}) for n in ('ridge', 'scalar'))
            gap_a = g['C1b']['skill_random_anchored'] - skill(t, ar, as_)
            out[t]['sensitivity_anchored'] = {'gap_C1b': gap_a, 'material': bool(np.isfinite(gap_a) and gap_a >= C1_GAP),
                                              'agrees_with_official': bool(np.isfinite(gap_a) and gap_a >= C1_GAP) == out[t]['material']}
        n += out[t]['material']
    out['verdict_materially_optimistic'] = bool(n >= 2)
    return out


def compute_verdicts(faults: dict, models: Dict[str, dict], gaps: Dict[str, dict]) -> dict:
    h1 = {t: h1_task(t, models[t]) for t in models}
    for t, h in h1.items():                                   # declared sensitivity analysis: the same rule with the anchored ROC area
        if PRIMARY[t][0] == 'roc_auc':
            alt = h1_task(t, _anchored_view(models[t]))
            h['sensitivity_anchored'] = {'verdict': alt['verdict'], 'ratio': alt.get('ratio'), 'design_macro': alt.get('design_macro'),
                                         'agrees_with_official': alt['verdict'] == h['verdict']}
    n_true = sum(v['verdict'] is True for v in h1.values())
    out = {'H1': {'tasks': h1, 'n_true': n_true, 'overall': bool(n_true >= 2),
                  'rule': 'per-task: best baseline gain >= 0.8 x published gain (design-macro); overall TRUE if >= 2 of 3 tasks'},
           'E006_roc_anchor': ({n: {'official_design_macro': macro(design_values('ir_drop', models['ir_drop']['results'][n])),
                                    'anchored_design_macro': macro(design_values('ir_drop', models['ir_drop']['results'][n], anchored=True))}
                               for n in ('zero', 'scalar', 'mean_map') if n in models['ir_drop']['results']} if 'ir_drop' in models else None),
           **h2_h3_r1(faults, models), 'C1': c1_verdict(gaps, {t: models[t] for t in gaps}) if gaps else None}
    return out


def render_tables(v: dict) -> str:
    L = ['## H1 per task (design-macro primary metric; every reference is a WEAK reference: other split or preprocessing)', '',
         '| task | metric | mean_map | best baseline | its value | published (ref) | gain ratio | verdict |', '|---|---|---|---|---|---|---|---|']
    for t, h in v['H1']['tasks'].items():
        if h['verdict'] == 'not evaluable':
            L.append(f"| {t} | {h['metric']} | n/a | | | {h['reference']['value']} | | not evaluable |")
            continue
        L.append(f"| {t} | {h['metric']} | {h['design_macro']['mean_map']:.4f} | {h['best_baseline']} | {h['design_macro'][h['best_baseline']]:.4f} | "
                 f"{h['reference']['value']} ({h['reference']['model']}) | {h.get('ratio', float('nan')):.2f} | {h['verdict']} |")
    L += ['', f"H1 overall: {v['H1']['overall']} ({v['H1']['n_true']} of {len(v['H1']['tasks'])} tasks)", '', '## Primary metric per baseline (design-macro / sample-pooled)', '']
    for t, h in v['H1']['tasks'].items():
        L.append(f'### {t} ({h["metric"]})')
        L.append('| baseline | design-macro | pooled |')
        L.append('|---|---|---|')
        for n in h['design_macro']:
            L.append(f"| {n} | {h['design_macro'][n]:.4f} | {h['sample_pooled'][n]:.4f} |")
        L.append('')
    L += [f"H2 (material data fault on some task): {v['H2_verdict']}", f"H3 (metric undefined or restricted for >= 5 % of samples): {v['H3_verdict']}", '',
          f"R1 (IR raw label above the clip on >= 50 % of test samples): {v.get('R1', {}).get('material')}", '']
    if v.get('C1'):
        L.append(f"C1 (random split materially optimistic on >= 2 tasks): {v['C1']['verdict_materially_optimistic']}")
    return '\n'.join(L) + '\n'


def stage_verdicts(run: Run):
    _preflight(run, 'verdicts')
    d = Path(run.out_dir)
    faults = _load(d, 'faults.json')
    models = {t: _load(d, f'models_{t}.json') for t in TASK_LIST if (d / f'models_{t}.json').exists()}
    gaps = {t: _load(d, f'gap_{t}.json') for t in TASK_LIST if (d / f'gap_{t}.json').exists()}
    v = compute_verdicts(faults, models, gaps)
    write_once(d / 'verdicts.json', {'label': 'DRY RUN' if run.dry_run else 'REGISTERED', **v}, dataset='n14')
    write_text_once(d / 'tables.md', render_tables(v))
    print('H1', v['H1']['overall'], 'H2', v['H2_verdict'], 'H3', v['H3_verdict'])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m edalint.p4_n14')
    ap.add_argument('stage', choices=['faults', 'models', 'gap', 'verdicts'])
    ap.add_argument('--task', choices=TASK_LIST)
    ap.add_argument('--dry-run', metavar='DIR', help='unregistered dry run under DIR with the test split replaced by held-out train designs')
    ap.add_argument('--hold-out', default='nvdla-small,Vortex-small')
    a = ap.parse_args(argv)
    try:
        run = make_dry_run(Path(a.dry_run), a.hold_out.split(',')) if a.dry_run else Run()
        if a.stage in ('models', 'gap') and not a.task:
            ap.error('--task is required for models and gap')
        {'faults': lambda: stage_faults(run), 'models': lambda: stage_models(run, a.task), 'gap': lambda: stage_gap(run, a.task),
         'verdicts': lambda: stage_verdicts(run)}[a.stage]()
    except (ProtocolError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
