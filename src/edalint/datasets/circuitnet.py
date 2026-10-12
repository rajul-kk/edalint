"""CircuitNet-N14 loader (HF CircuitNet/CircuitNet, CircuitNet-N14/). Layout and the authors' preprocessing: docs/format_notes.md 2.

The data is streamed from the per-design tarballs with `tarfile` (never extracted wholesale). Each map is an .npz with key `data`.
Two modes:
  authors  the CircuitNet authors' `generate_training_set.py`, reproduced: congestion 3-channel input, DRC 9-channel input with
           the label clipped at 200 and divided by 200, IR drop 24-channel input with the label clipped to [1e-6, 50] and
           log-scaled; everything resized to 256 x 256 (cubic scipy zoom for congestion/DRC features, area resize for the rest)
           and min-max scaled per map per sample.
  raw      native resolution float32, no clipping / scaling. Labels: congestion = GR horizontal + vertical overflow,
           DRC = DRC_all, IR drop = IR_drop. Some inputs (congestion_eGR_*) are on a finer grid than the label.
Official split: train_N14.csv / test_N14.csv (design level). `test` is guarded (needs protocol/FROZEN.json); its tarballs are
only read after the guard passes (or by `smoke_test_shapes`).

Samples that lack their label or any feature of the task are SKIPPED and logged in `ds.skipped`; tarballs that are not
complete (an .incomplete partner, or no sha256 match recorded in manifests/MANIFEST.json) are skipped as a whole.
"""
from __future__ import annotations

import io
import json
import math
import os
import posixpath
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core import MapDataset, Sample, area_resize
from ..protocol import REPO_DIR, require_frozen

DEFAULT_ROOT = os.environ.get('EDALINT_N14', r'E:\edalint_data\circuitnet_n14')
ROUT, IRD = 'routability_features', 'IR_drop_features'
SIZE = 256

# feature name -> directory inside the design folder of the tarball
FEATURE_DIRS = {
    'macro_region': 'macro_region', 'cell_density': 'cell_density',
    'RUDY': 'RUDY/RUDY', 'RUDY_pin': 'RUDY/RUDY_pin', 'RUDY_long': 'RUDY/RUDY_long', 'RUDY_short': 'RUDY/RUDY_short',
    'RUDY_pin_long': 'RUDY/RUDY_pin_long',
    'eGR_horizontal_overflow': 'congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow',
    'eGR_vertical_overflow': 'congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow',
    'GR_horizontal_overflow': 'congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow',
    'GR_vertical_overflow': 'congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow',
    'DRC_all': 'DRC/DRC_all',
    'power_i': 'power_i', 'power_s': 'power_s', 'power_sca': 'power_sca', 'power_all': 'power_all', 'power_t': 'power_t',
    'IR_drop': 'IR_drop',
}
TASKS = {
    'congestion': {'tar': ROUT, 'inputs': ['macro_region', 'RUDY', 'RUDY_pin'],
                   'label': ['GR_horizontal_overflow', 'GR_vertical_overflow'], 'primary': 'RUDY',
                   'hotspot': {'kind': 'top_frac', 'value': 0.05}},
    'drc': {'tar': ROUT, 'inputs': ['macro_region', 'cell_density', 'RUDY_long', 'RUDY_short', 'RUDY_pin_long',
                                    'eGR_horizontal_overflow', 'eGR_vertical_overflow', 'GR_horizontal_overflow',
                                    'GR_vertical_overflow'],
            'label': ['DRC_all'], 'primary': 'RUDY_long', 'hotspot': {'kind': 'abs', 'value': 0.1}},
    'ir_drop': {'tar': IRD, 'inputs': ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t'], 'label': ['IR_drop'],
                'primary': 'power_all', 'hotspot': {'kind': 'abs', 'value': 0.9885}},
}
# how a stored member is resized in authors mode ('zoom3' = scipy.ndimage.zoom order 3, 'area' = cv2.INTER_AREA-like)
_RESIZE = {f: 'zoom3' for f in FEATURE_DIRS if f not in ('DRC_all', 'IR_drop') and not f.startswith('power')}
_RESIZE.update({f: 'area' for f in FEATURE_DIRS if f.startswith('power') or f in ('DRC_all', 'IR_drop')})
# bounding-box-derived inputs (GIF reports all-zero instance bounding boxes in N14): E002 looks at these
BBOX_CHANNELS = ('RUDY', 'RUDY_pin', 'RUDY_long', 'RUDY_short', 'RUDY_pin_long', 'cell_density')


# -- splits ---------------------------------------------------------------------------------------------------------------
def read_split_csv(path) -> List[str]:
    """Sample names (no extension) from a `feature/<name>.npy,label/<name>.npy` split file."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(',')[0]
        names.append(posixpath.basename(f).rsplit('.', 1)[0])
    return names


def design_of(name: str) -> str:
    return name.split('_freq_')[0]


def packaged_split_csvs() -> Tuple[Path, Path]:
    d = Path(__file__).resolve().parent.parent / 'data'
    return d / 'train_N14.csv', d / 'test_N14.csv'


# -- tarball completeness ----------------------------------------------------------------------------------------------------
def load_manifest(path: Optional[str] = None) -> dict:
    p = Path(path) if path else REPO_DIR / 'manifests' / 'MANIFEST.json'
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding='utf-8')).get('datasets', {}).get('circuitnet_n14', {}).get('files', {})


def tarball_status(root, subset: str, design: str, manifest: Optional[dict] = None, require_sha: bool = True) -> Tuple[bool, str]:
    """(usable, reason). Not usable if missing, if an .incomplete partner exists (next to the file, or in the HF download cache
    under the expected sha256), or, when require_sha, if MANIFEST.json does not record a sha256 match."""
    root = Path(root)
    p = root / 'CircuitNet-N14' / subset / f'{design}.tar.gz'
    key = f'CircuitNet-N14/{subset}/{design}.tar.gz'
    if not p.is_file():
        return False, f'tarball missing ({p})'
    for suffix in ('.incomplete', '.part'):
        if Path(str(p) + suffix).exists():
            return False, f'incomplete ({p.name}{suffix} present)'
    ent = (manifest or {}).get(key, {})
    sha = ent.get('sha256_expected_hf_lfs')
    cache = root / '.cache' / 'huggingface' / 'download' / 'CircuitNet-N14' / subset
    if sha and cache.is_dir() and any(f.name.endswith(f'.{sha}.incomplete') for f in cache.iterdir()):
        return False, 'incomplete (download still in progress: .incomplete in the HF cache)'
    if ent.get('size') and p.stat().st_size != ent['size']:
        return False, f'size {p.stat().st_size} != expected {ent["size"]}'
    if require_sha and not ent.get('sha256_matches_hf'):
        return False, 'no sha256 match recorded in MANIFEST.json'
    return True, 'ok'


# -- stores -------------------------------------------------------------------------------------------------------------------
class MemStore:
    def __init__(self):
        self.d: Dict[tuple, np.ndarray] = {}
        self.presence: Dict[tuple, Dict[str, set]] = {}

    def put(self, design, mode, feature, name, arr):
        self.d[(design, mode, feature, name)] = arr

    def get(self, design, mode, feature, name):
        return self.d.get((design, mode, feature, name))

    def has(self, design, mode, feature, name):
        return (design, mode, feature, name) in self.d

    def get_presence(self, key):
        return self.presence.get(key)

    def set_presence(self, key, pres):
        self.presence[key] = pres


class DirStore:
    """Per-map float32 .npy files under root/<mode>/<design>/<feature>/<name>.npy (constant memory during staging)."""

    def __init__(self, root):
        self.root = Path(root)

    def _p(self, design, mode, feature, name):
        return self.root / mode / design / feature / f'{name}.npy'

    def put(self, design, mode, feature, name, arr):
        p = self._p(design, mode, feature, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)

    def get(self, design, mode, feature, name):
        p = self._p(design, mode, feature, name)
        return np.load(p) if p.is_file() else None

    def has(self, design, mode, feature, name):
        return self._p(design, mode, feature, name).is_file()

    def get_presence(self, key):
        p = self.root / f'presence_{key[0]}_{key[1]}_{key[2]}.json'
        if not p.is_file():
            return None
        return {k: set(v) for k, v in json.loads(p.read_text()).items()}

    def set_presence(self, key, pres):
        p = self.root / f'presence_{key[0]}_{key[1]}_{key[2]}.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: sorted(v) for k, v in pres.items()}))


# -- preprocessing (authors) ----------------------------------------------------------------------------------------------
def _zoom3(x: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(x, (SIZE / x.shape[-2], SIZE / x.shape[-1]), order=3)


def _std(x: np.ndarray) -> np.ndarray:
    """Authors' per-map min-max: (x - min) / (max - min); returned unchanged when max == 0. A constant NON-zero map divides by
    zero and yields NaN / inf, exactly as in the original script (E001 reports it)."""
    mx = x.max()
    if mx == 0:
        return x
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x - x.min()) / (mx - x.min())


def stage_member(feature: str, arr: np.ndarray, mode: str) -> np.ndarray:
    """What is stored per member. authors: resized (and clipped for the two labels), not yet min-max scaled."""
    a = np.asarray(arr, dtype=np.float64)
    if mode == 'raw':
        return a.astype(np.float32)
    if feature == 'DRC_all':
        a = np.clip(a, 0, 200)
    elif feature == 'IR_drop':
        a = np.clip(np.squeeze(a), 1e-6, 50)
    if feature == 'power_t' or a.ndim == 3:
        out = np.stack([area_resize(s, (SIZE, SIZE)) for s in a])
    elif _RESIZE[feature] == 'zoom3':
        out = _zoom3(a)
    else:
        out = area_resize(np.squeeze(a), (SIZE, SIZE))
    return out.astype(np.float32)


def assemble(task: str, mode: str, get) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build (inputs, label) from staged members; `get(feature) -> array`. See module docstring for the exact transforms."""
    spec = TASKS[task]
    ins: Dict[str, np.ndarray] = {}
    for f in spec['inputs']:
        x = get(f).astype(np.float64)
        if mode == 'authors':
            x = np.stack([_std(s) for s in x]) if f == 'power_t' else _std(x)
        ins[f] = x
    if task == 'congestion':
        lab = get('GR_horizontal_overflow').astype(np.float64) + get('GR_vertical_overflow').astype(np.float64)
        if mode == 'authors':
            lab = _std(lab)
    elif task == 'drc':
        lab = get('DRC_all').astype(np.float64)
        if mode == 'authors':
            lab = lab / 200.0
    else:
        lab = get('IR_drop').astype(np.float64)
        if mode == 'authors':
            lab = (np.log10(lab) + 6) / (math.log10(50) + 6)
    return ins, np.squeeze(lab)


# -- streaming ----------------------------------------------------------------------------------------------------------------
def stream_tarball(tar_path, design: str, dir_to_feature: Dict[str, str], wanted: Optional[set], handle) -> Dict[str, set]:
    """One sequential pass over the tarball. For members `<design>/<dir>/<name>.npz` with dir in dir_to_feature: record presence
    of (feature, name) for every such member and, when `wanted` is None or name in wanted, call handle(feature, name, array)."""
    presence: Dict[str, set] = {f: set() for f in dir_to_feature.values()}
    with tarfile.open(tar_path, 'r|gz') as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith('.npz'):
                continue
            d, fn = posixpath.split(m.name)
            d = d[2:] if d.startswith('./') else d
            if d.startswith(design + '/'):
                d = d[len(design) + 1:]
            feat = dir_to_feature.get(d)
            if feat is None:
                continue
            name = fn[:-4]
            presence[feat].add(name)
            if wanted is None or name in wanted:
                with np.load(io.BytesIO(tf.extractfile(m).read())) as z:
                    handle(feat, name, z['data'])
    return presence


class N14Dataset(MapDataset):
    """CircuitNet-N14, one task per dataset object (`congestion`, `drc` or `ir_drop`).

    max_per_design   evenly spaced subsample of each (split, design) from the split CSV (None = all). RAM guard.
    cache_dir        if given, staged maps go to disk (float32 .npy); else they are kept in RAM (use max_per_design then).
    designs          restrict to these designs.
    split_csvs       (train.csv, test.csv); default the packaged official files.
    stage            stage non-test splits at construction (test splits only after the guard passes).
    """

    def __init__(self, task: str = 'ir_drop', mode: str = 'authors', root: Optional[str] = None, cache_dir: Optional[str] = None,
                 max_per_design: Optional[int] = None, designs: Optional[Sequence[str]] = None,
                 split_csvs: Optional[Tuple[str, str]] = None, manifest_path: Optional[str] = None, require_sha: bool = True,
                 protocol_dir: Optional[str] = None, stage: bool = True, cache_items: int = 8):
        if task not in TASKS:
            raise ValueError(f'task must be one of {sorted(TASKS)}')
        if mode not in ('authors', 'raw'):
            raise ValueError("mode must be 'authors' or 'raw'")
        self.task_spec = TASKS[task]
        self.mode, self.root = mode, Path(root or DEFAULT_ROOT)
        self.store = DirStore(cache_dir) if cache_dir else MemStore()
        self.manifest = load_manifest(manifest_path)
        self.require_sha = require_sha
        tr_csv, te_csv = split_csvs or packaged_split_csvs()
        rows = {'train': read_split_csv(tr_csv), 'test': read_split_csv(te_csv)}
        self.inventory: Dict[str, Dict[str, int]] = {}
        self.unlisted: Dict[str, int] = {}
        meta: Dict[str, dict] = {}
        splits: Dict[str, List[str]] = {'train': [], 'test': []}
        self._wanted: Dict[Tuple[str, str], List[str]] = {}
        skipped: List[dict] = []
        self._design_ok: Dict[str, Tuple[bool, str]] = {}
        for split, names in rows.items():
            bydes: Dict[str, List[str]] = {}
            for n in names:
                bydes.setdefault(design_of(n), []).append(n)
            for des, ns in sorted(bydes.items()):
                if designs and des not in designs:
                    continue
                ns = sorted(ns)
                if max_per_design and len(ns) > max_per_design:
                    ns = [ns[j] for j in np.unique(np.linspace(0, len(ns) - 1, max_per_design).round().astype(int))]
                ok, why = self._design_ok.setdefault(des, tarball_status(self.root, self.task_spec['tar'], des, self.manifest, require_sha))
                if not ok:
                    skipped.append({'design': des, 'split': split, 'n_samples': len(ns), 'reason': f'tarball unusable: {why}'})
                    continue
                self._wanted[(split, des)] = ns
                for n in ns:
                    meta[n] = {'split': split, 'design': des, 'name': n}
                    splits[split].append(n)
        spec = self.task_spec
        super().__init__(f'circuitnet_n14_{task}_{mode}', task, splits, meta, primary_input=spec['primary'],
                         hotspot=spec['hotspot'] if mode == 'authors' else {'kind': 'top_frac', 'value': 0.05},
                         label_range={'ir_drop': (0.0, 1.0) if mode == 'authors' else (0.0, 50.0),
                                      'drc': (0.0, 1.0) if mode == 'authors' else (0.0, None),
                                      'congestion': (0.0, 1.0) if mode == 'authors' else (0.0, None)}[task],
                         label_units='', protocol_dir=protocol_dir, cache_items=cache_items)
        self.skipped = skipped
        self._staged: set = set()
        if stage:
            self.stage_all(include_test=False)

    # -- staging
    def _dir_map(self) -> Dict[str, str]:
        feats = list(dict.fromkeys(self.task_spec['inputs'] + self.task_spec['label']))
        return {FEATURE_DIRS[f]: f for f in feats}

    def _required(self) -> List[str]:
        return list(dict.fromkeys(self.task_spec['inputs'] + self.task_spec['label']))

    def stage_design(self, design: str, splits: Iterable[str] = ('train', 'test')):
        """Stream one design's tarball, store the wanted maps, then drop samples that lack any required map (logged)."""
        for split in splits:
            self._stage_one(design, split)

    def _stage_one(self, design: str, split: str):
        if (design, split) in self._staged:
            return
        names = self._wanted.get((split, design), [])
        wanted = set(names)
        if not wanted:
            return
        key = (design, self.mode, self.task)
        req = self._required()
        pres = self.store.get_presence(key)
        need_stream = pres is None or any(not self.store.has(design, self.mode, f, n) for n in wanted for f in req
                                          if n in pres.get(f, ()))
        if need_stream:
            tar = self.root / 'CircuitNet-N14' / self.task_spec['tar'] / f'{design}.tar.gz'

            def handle(feat, name, arr):
                self.store.put(design, self.mode, feat, name, stage_member(feat, arr, self.mode))
            pres = stream_tarball(tar, design, self._dir_map(), wanted, handle)
            self.store.set_presence(key, pres)
        self.inventory[design] = {f: len(pres[f]) for f in req}
        listed = {n for (s, d), v in self._wanted.items() if d == design for n in v}
        union = set().union(*[pres[f] for f in req]) if req else set()
        self.unlisted[design] = len(union - listed)
        keep = []
        for n in names:
            missing = [f for f in req if n not in pres[f]]
            if missing:
                self.skipped.append({'id': n, 'design': design, 'split': split, 'reason': 'missing from tarball: ' + ', '.join(missing)})
            else:
                keep.append(n)
        gone = set(names) - set(keep)
        if gone:
            self.splits[split] = [i for i in self.splits[split] if i not in gone]
        self._staged.add((design, split))

    def stage_all(self, include_test: bool = False):
        splits = ('train', 'test') if include_test else ('train',)
        for des in sorted({d for (s, d) in self._wanted if s in splits}):
            self.stage_design(des, splits)

    def _load(self, sid: str) -> Sample:
        m = self._meta[sid]
        if (m['design'], m['split']) not in self._staged:
            self.stage_design(m['design'], (m['split'],))
        ins, lab = assemble(self.task, self.mode, lambda f: self.store.get(m['design'], self.mode, f, sid))
        return Sample(sid, m['split'], m['design'], ins, lab, {'name': sid, 'mode': self.mode})

    # -- reports
    def staging_report(self) -> dict:
        return {'designs_staged': sorted({k[0] for k in self._staged}), 'inventory': self.inventory,
                'unlisted_in_csv': self.unlisted, 'skipped': self.skipped}

    def smoke_test_shapes(self, per_design: int = 2) -> List[dict]:
        """Loader smoke test for TEST designs: stream the first `per_design` samples of each test design and report only
        array shapes and dtypes. Does not touch the guard because no value is returned or computed."""
        out = []
        for (split, des), names in sorted(self._wanted.items()):
            if split != 'test':
                continue
            tar = self.root / 'CircuitNet-N14' / self.task_spec['tar'] / f'{des}.tar.gz'
            shapes: Dict[str, tuple] = {}
            want = set(names[:per_design])

            def handle(feat, name, arr, shapes=shapes):
                shapes.setdefault(feat, (tuple(arr.shape), str(arr.dtype)))
            pres = stream_tarball(tar, des, self._dir_map(), want, handle)
            out.append({'design': des, 'shapes': {k: v[0] for k, v in shapes.items()}, 'n_members': {f: len(p) for f, p in pres.items()}})
        return out
