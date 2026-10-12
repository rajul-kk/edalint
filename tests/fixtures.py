"""Tiny on-disk fixtures: an ICCAD'23-shaped tree and a CircuitNet-N14-shaped tarball tree. All values are synthetic."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from edalint.datasets.circuitnet import FEATURE_DIRS


def _grid_netlist(n, seed, n_src=6):
    rng = np.random.default_rng(seed)
    lines, k = [], 0
    for x in range(n):
        for y in range(n):
            if x + 1 < n:
                lines.append(f'R{k} n1_m1_{x * 2000}_{y * 2000} n1_m1_{(x + 1) * 2000}_{y * 2000} 1.5'); k += 1
            if y + 1 < n:
                lines.append(f'R{k} n1_m1_{x * 2000}_{y * 2000} n1_m1_{x * 2000}_{(y + 1) * 2000} 1.5'); k += 1
    lines.append(f'V0 n1_m1_0_0 0 1.100000')
    lines.append(f'V1 n1_m1_{(n - 1) * 2000}_{(n - 1) * 2000} 0 1.100000')
    for j in range(n_src):
        lines.append(f'I{j} n1_m1_{rng.integers(1, n - 1) * 2000}_{rng.integers(1, n - 1) * 2000} 0 {rng.uniform(1e-4, 1e-3):.6e}')
    return '\n'.join(lines) + '\n.op\n.end'


def _csv(path, a):
    np.savetxt(path, a, delimiter=',', fmt='%.6e')


def make_iccad_tree(root: Path, n=10):
    b = root / 'benchmarks'
    rng = np.random.default_rng(0)

    def maps(d, prefix_map, n=n):
        cur = rng.random((n, n)) * 1e-6
        for key, arr in zip(('current', 'eff_dist', 'pdn_density', 'ir_drop'),
                            (cur, rng.random((n, n)) * 50, rng.integers(0, 4, (n, n)).astype(float), rng.random((n, n)) * 1e-3)):
            _csv(prefix_map(key), arr)
    fk = b / 'fake-circuit-data'
    fk.mkdir(parents=True)
    for i in range(2):
        nm = f'current_map{i:02d}'
        maps(fk, lambda k, nm=nm: fk / f'{nm}_{k}.csv')
        (fk / f'{nm}.sp').write_text(_grid_netlist(n, i))
    names = {'real-circuit-data': (1, 2), 'hidden-real-circuit-data': (7, 8)}
    for sub, (a, c) in names.items():
        for j, num in enumerate((a, c)):
            d = b / sub / f'testcase{num}'
            d.mkdir(parents=True)
            files = {'current': 'current_map.csv', 'eff_dist': 'eff_dist_map.csv', 'pdn_density': 'pdn_density.csv', 'ir_drop': 'ir_drop_map.csv'}
            nn = n if sub == 'real-circuit-data' else n + 2
            maps(d, lambda k, d=d, files=files: d / files[k], n=nn)
            # twins: same grid and same number of current sources, different PDN (resistor value)
            (d / 'netlist.sp').write_text(_grid_netlist(nn, a).replace(' 1.5', f' {1.5 + j}'))
    return root


def _npz(arr):
    bio = io.BytesIO()
    np.savez(bio, data=arr)
    return bio.getvalue()


def make_n14_tree(root: Path, n_per=4, h=40, w=44, missing=None):
    """Designs: DesA (train, routability + IR), DesB (test, routability), DesC (train, incomplete). Returns (root, csvs, manifest)."""
    rng = np.random.default_rng(1)
    names = {d: [f'{d}_freq_50_mp_1_fpu_50_fpa_1.0_p_{p}_fi_ap' for p in range(1, n_per + 1)] for d in ('DesA', 'DesB', 'DesC')}
    missing = missing or {('DesA', 'DRC_all'): names['DesA'][0]}
    rout_feats = [f for f in FEATURE_DIRS if not f.startswith('power') and f != 'IR_drop']
    ir_feats = ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t', 'IR_drop']
    (root / 'CircuitNet-N14' / 'routability_features').mkdir(parents=True)
    (root / 'CircuitNet-N14' / 'IR_drop_features').mkdir(parents=True)

    def build(subset, design, feats):
        p = root / 'CircuitNet-N14' / subset / f'{design}.tar.gz'
        with tarfile.open(p, 'w:gz') as tf:
            for nm in names[design]:
                for f in feats:
                    if missing.get((design, f)) == nm:
                        continue
                    hh, ww = (72, 76) if f.startswith('eGR') else (h, w)
                    if f == 'power_t':
                        arr = rng.random((20, hh, ww))
                    elif f == 'DRC_all':
                        arr = rng.random((hh, ww)) * 300
                    elif f == 'IR_drop':
                        arr = rng.random((hh, ww)) * 20
                    else:
                        arr = rng.random((hh, ww))
                    data = _npz(arr)
                    ti = tarfile.TarInfo(f'./{design}/{FEATURE_DIRS[f]}/{nm}.npz')
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
        return p
    build('routability_features', 'DesA', rout_feats)
    build('routability_features', 'DesB', rout_feats)
    build('routability_features', 'DesC', rout_feats)
    build('IR_drop_features', 'DesA', ir_feats)
    (root / 'CircuitNet-N14' / 'routability_features' / 'DesC.tar.gz.incomplete').write_text('')

    def csv(path, ds):
        path.write_text('\n'.join(f'feature/{n}.npy,label/{n}.npy' for d in ds for n in names[d]) + '\n')
    tr, te = root / 'train.csv', root / 'test.csv'
    csv(tr, ['DesA', 'DesC'])
    csv(te, ['DesB'])
    man = root / 'manifest.json'
    files = {f'CircuitNet-N14/{s}/{d}.tar.gz': {'sha256_matches_hf': True}
             for s, d in (('routability_features', 'DesA'), ('routability_features', 'DesB'), ('routability_features', 'DesC'),
                          ('IR_drop_features', 'DesA'))}
    del files['CircuitNet-N14/routability_features/DesC.tar.gz']['sha256_matches_hf']        # no sha match recorded -> unusable
    man.write_text(json.dumps({'datasets': {'circuitnet_n14': {'files': files}}}))
    return root, (str(tr), str(te)), str(man), names
