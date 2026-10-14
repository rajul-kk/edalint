"""P7 train-only T4 benchmark and held-out checks (DEVELOPMENT, label p7dev_kaggle). protocol_p7.md sections 7, 11 (D1, D2, D3, D8 step 3).

Shared body of the two Kaggle kernels (scripts/p7_dev_kaggle_gpdl.py, scripts/p7_dev_kaggle_mavi.py). Reads TRAIN shards only (two designs held out of
training, both official TRAIN designs). No test row, no FROZEN_p7.json, no freeze check (this is not a registered run); outputs are labelled
development and are not results.

For one model it measures:
  * seconds per training iteration on the GPU (pure step on one cached real batch, fp32 and fp16 autocast), seconds per batch of data loading,
    and end-to-end seconds per iteration from the check runs;
  * fp16 vs fp32 forward, max absolute difference on 20 train rows (D3);
  * the held-out check(s): train on the non-held-out designs, score the held-out designs with the P7 metrics against the train-only mean_map
    fitted on the same training rows.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HELD = ('Vortex-small', 'nvdla-small')


def find_root(marker):
    hits = sorted(glob.glob(f'/kaggle/input/**/{marker}', recursive=True))
    if not hits:
        raise SystemExit(f'cannot find {marker} under /kaggle/input')
    return hits[0][:-len(marker)].rstrip('/')


def setup(a):
    bundle = a.bundle or find_root('src/edalint_p7/__init__.py')
    sys.path.insert(0, str(Path(bundle) / 'src'))
    return bundle


def mean_map_and_ref(task, src, ids_idx):
    """Train-only mean_map (edalint MeanMap recipe: canonical 128x128 mean of area-resized labels) and the constant mean_map training loss."""
    from edalint.baselines.zoo import MeanMap
    from edalint.core import Sample
    labels = []

    class S:
        def __iter__(self):
            for k in ids_idx:
                y = src.get(k)[1]
                yield Sample(src.ids[k], 'train', src.designs[k], {}, y)
    mm = MeanMap().fit(S())
    return mm


def const_loss(task, mm, src, idx, weight=100.0):
    from edalint.baselines.base import resize_to
    tot, n = 0.0, 0
    for k in idx:
        y = src.get(k)[1]
        p = resize_to(mm.mean_, y.shape)
        d = np.abs(p - y) if task == 'ir_drop' else (p - y) ** 2
        tot += float(d.mean())
        n += 1
    return weight * tot / max(n, 1)


def heldout_eval(task, model, src, idx, mm, device, log):
    from edalint.baselines.base import resize_to
    from edalint.p4_n14 import PrimaryAcc
    from edalint_p7.train import predict
    am, ag = PrimaryAcc(task), PrimaryAcc(task)
    per = {}
    t0 = time.perf_counter()
    for j, k in enumerate(idx):
        x, y = src.get(k)
        pg = np.nan_to_num(predict(task, model, x, device))
        pm = np.nan_to_num(resize_to(mm.mean_, y.shape))
        am.add(pm, y)
        ag.add(pg, y)
        d = per.setdefault(src.designs[k], (PrimaryAcc(task), PrimaryAcc(task)))
        d[0].add(pm, y)
        d[1].add(pg, y)
        if (j + 1) % 200 == 0:
            log(f'held-out scored {j + 1}/{len(idx)} {time.perf_counter() - t0:.0f}s')
    res = {'n_rows': len(idx), 'seconds': time.perf_counter() - t0,
           'nrms_model': float(np.nanmean(ag.nrms)), 'nrms_mean_map': float(np.nanmean(am.nrms))}
    if task == 'ir_drop':
        res.update(roc_model=ag.roc.area(), roc_mean_map=am.roc.area(), roc_model_anchored=ag.roc.area(anchored=True),
                   roc_mean_map_anchored=am.roc.area(anchored=True))
    res['per_design'] = {}
    for d, (pm_, pg_) in per.items():
        e = {'n': len(pg_.nrms), 'nrms_model': float(np.nanmean(pg_.nrms)), 'nrms_mean_map': float(np.nanmean(pm_.nrms))}
        if task == 'ir_drop':
            e.update(roc_model=pg_.roc.area(), roc_mean_map=pm_.roc.area())
        res['per_design'][d] = e
    return res


def gpu_step_time(task, src, precision, device, n_warm=15, n_time=30):
    """Pure training step time (forward, loss, backward, AdamW step) on one cached real batch, CUDA-synchronised."""
    import torch
    from edalint_p7.recipe import RECIPE
    from edalint_p7.train import build_model, loss_fn, make_batch
    cfg = RECIPE[task]
    rng = np.random.default_rng(1)
    idx = np.arange(min(cfg['batch_size'], len(src)))
    t0 = time.perf_counter()
    x, y = make_batch(task, src, idx, rng, cfg['flip'], device)
    t_data = time.perf_counter() - t0
    torch.manual_seed(0)
    model = build_model(task).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    lf = loss_fn(task)
    fp16 = precision == 'fp16'
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)
    model.train()

    def step():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
            pred = model(x)
        loss = lf(pred.float(), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    for _ in range(n_warm):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_time
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    del model, opt
    torch.cuda.empty_cache()
    return {'precision': precision, 'sec_per_iteration': dt, 'sec_per_sample': dt / len(idx), 'batch_load_seconds_first_batch': t_data,
            'n_timed': n_time, 'peak_gpu_gib': mem}


def fp16_fp32_forward(task, model, src, idx, device):
    import torch
    from edalint_p7.data import model_input
    model.eval()
    worst, rel = 0.0, 0.0
    with torch.no_grad():
        for k in idx:
            x = model_input(task, torch.from_numpy(np.asarray(src.get(k)[0], np.float32))[None].to(device))
            a = model(x).float()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                b = model(x).float()
            worst = max(worst, float((a - b).abs().max()))
            rel = max(rel, float((a - b).abs().max() / (a.abs().max() + 1e-12)))
    return {'n_rows': len(idx), 'max_abs_diff': worst, 'max_abs_diff_over_max_abs_fp32': rel}


def e2e_per_iter(stamps):
    """stamps: perf_counter at each 100-iteration log line; seconds per iteration from the second to the last block."""
    if len(stamps) < 2:
        return None
    return (stamps[-1] - stamps[0]) / (100 * (len(stamps) - 1))


def run_check(task, src, tr_idx, ho_idx, iters, precision, seed, device, tag, log, max_heldout):
    from edalint_p7.data import SubsetSource
    from edalint_p7.train import train
    ids_tr = [src.ids[k] for k in tr_idx]
    sub = SubsetSource(src, ids_tr)
    stamps = []

    def lg(msg):
        if ' iter ' in msg:
            stamps.append(time.perf_counter())
        log(f'[{tag}] {msg}')
    t0 = time.perf_counter()
    rec = train(task, sub, iters, seed, precision, device, 'epoch', log=lg)
    model = rec.pop('_model')
    out = {'tag': tag, 'train': rec, 'e2e_sec_per_iteration': e2e_per_iter(stamps), 'train_seconds': time.perf_counter() - t0}
    curve = rec['loss_per_100_iters']
    out['loss_finite'] = bool(len(curve) > 0 and np.isfinite(curve).all() and rec['nonfinite_losses'] == 0 and rec['diverged_at'] is None)
    out['skipped_fraction'] = rec['skipped_steps'] / max(iters, 1)
    mm = mean_map_and_ref(task, src, tr_idx)
    out['const_mean_map_train_loss'] = const_loss(task, mm, src, tr_idx)
    out['last_block_train_loss'] = curve[-1] if curve else None
    out['first_block_train_loss'] = curve[0] if curve else None
    out['loss_drop_fraction'] = (1 - curve[-1] / curve[0]) if curve and curve[0] else None
    ho = list(ho_idx)
    if max_heldout and len(ho) > max_heldout:
        step = int(np.ceil(len(ho) / max_heldout))
        ho = ho[::step]
        out['heldout_subsample_stride'] = step
    out['heldout'] = heldout_eval(task, model, src, ho, mm, device, lg)
    if not precision == 'fp16' or True:
        out['fp16_vs_fp32_forward'] = fp16_fp32_forward(task, model, src, tr_idx[::max(1, len(tr_idx) // 20)][:20], device)
    return out


def main(model_name, argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--shards', default=None)
    ap.add_argument('--out', default='/kaggle/working')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--iters', type=int, nargs='*', default=None, help='override the check iteration counts (local smoke test only)')
    ap.add_argument('--max-heldout', type=int, default=0)
    ap.add_argument('--skip-timing', action='store_true')
    a = ap.parse_args(argv)
    bundle = setup(a)
    import torch
    from edalint.protocol import code_hash
    from edalint_p7.data import ShardSource, sha256_file
    from edalint_p7.guard import p7_code_hash
    task = 'ir_drop' if model_name == 'mavi' else 'congestion'
    logf = open(Path(a.out) / f'p7dev_kaggle_{model_name}.log', 'a', encoding='utf-8')

    def log(m):
        s = f'{time.strftime("%H:%M:%S")} {m}'
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()
    res = {'label': 'DEVELOPMENT, train-only (p7dev_kaggle); not a result', 'model': model_name, 'task': task,
           'code_hash': code_hash(), 'p7_code_hash': p7_code_hash(bundle), 'torch': torch.__version__}
    log(f'{model_name}: code_hash {res["code_hash"]} p7_code_hash {res["p7_code_hash"]}')
    if a.device == 'cuda':
        if not torch.cuda.is_available():
            log('GPU UNAVAILABLE: aborting')
            res['error'] = 'cuda not available'
            (Path(a.out) / f'p7dev_kaggle_{model_name}.json').write_text(json.dumps(res, indent=1))
            return 3
        res['gpu'] = torch.cuda.get_device_name(0)
        log(f'gpu {res["gpu"]}')
    shards = a.shards or find_root('manifest.json')
    src = ShardSource(shards, verify=False)
    # sha256 of every shard against the manifest (also warms the page cache)
    t0 = time.perf_counter()
    bad = [s['path'] for s in src.manifest['shards'] if sha256_file(Path(shards) / s['path']) != s['sha256']]
    res['manifest'] = {'n_rows': len(src), 'n_shards': len(src.manifest['shards']), 'n_excluded_nonfinite': src.manifest['n_excluded_nonfinite'],
                       'bad_sha256': bad, 'verify_seconds': time.perf_counter() - t0, 'designs': sorted(set(src.designs))}
    log(f'manifest verified: {len(src)} rows, bad sha {len(bad)}, {time.perf_counter() - t0:.0f}s')
    if bad:
        res['error'] = 'sha256 mismatch'
        (Path(a.out) / f'p7dev_kaggle_{model_name}.json').write_text(json.dumps(res, indent=1))
        return 4
    assert not (set(src.designs) - {'RISCY', 'RISCY-FPU', 'Vortex-small', 'nvdla-small'}), 'unexpected design in the train shards'
    tr_idx = [k for k, d in enumerate(src.designs) if d not in HELD]
    ho_idx = [k for k, d in enumerate(src.designs) if d in HELD]
    res['n_train_fold'], res['n_heldout_available'], res['held_out_designs'] = len(tr_idx), len(ho_idx), list(HELD)
    out_json = Path(a.out) / f'p7dev_kaggle_{model_name}.json'

    def save():
        out_json.write_text(json.dumps(res, indent=1), encoding='utf-8')
    if not a.skip_timing:
        res['gpu_step'] = []
        for prec in ('fp32', 'fp16'):
            if a.device != 'cuda':
                break
            r = gpu_step_time(task, src, prec, a.device)
            res['gpu_step'].append(r)
            log(f'gpu step {prec}: {r["sec_per_iteration"]:.4f} s/iter ({r["sec_per_sample"] * 1000:.1f} ms/sample), peak {r["peak_gpu_gib"]:.1f} GiB')
            save()
    plan = {'gpdl': [('d1_2000_fp32', 2000, 'fp32'), ('d1prime_10000_fp32', 10000, 'fp32')],
            'mavi': [('mavi_500_fp16', 500, 'fp16')]}[model_name]
    if a.iters:
        plan = [(f'smoke_{n}_{p}', n, p) for (_, _, p), n in zip(plan, a.iters)]
    res['checks'] = []
    for tag, n, prec in plan:
        log(f'check {tag}: train {len(tr_idx)} rows, held out {len(ho_idx)} rows, {n} iterations {prec}')
        res['checks'].append(run_check(task, src, tr_idx, ho_idx, n, prec, a.seed, a.device, tag, log, a.max_heldout))
        c = res['checks'][-1]
        if model_name == 'mavi':
            h = c['heldout']
            c['criteria'] = {'all_losses_finite': c['loss_finite'], 'skipped_fraction_le_5pct': c['skipped_fraction'] <= 0.05,
                             'last_block_below_const_mean_map_loss': bool(c['last_block_train_loss'] < c['const_mean_map_train_loss']),
                             'heldout_roc_above_train_only_mean_map': bool(h['roc_model'] > h['roc_mean_map'])}
        else:
            h = c['heldout']
            c['criteria'] = {'loss_finite': c['loss_finite'], 'loss_drop_ge_30pct': bool(c['loss_drop_fraction'] is not None and c['loss_drop_fraction'] >= 0.30),
                             'heldout_nrms_below_train_only_mean_map': bool(h['nrms_model'] < h['nrms_mean_map'])}
        c['pass'] = bool(all(c['criteria'].values()))
        log(f'check {tag}: {json.dumps(c["criteria"])} PASS={c["pass"]}')
        save()
    # projection for the full primary arm (D2 ladder), from the measured numbers
    proj = {}
    for c in res['checks']:
        if c['e2e_sec_per_iteration']:
            proj[c['tag']] = {'e2e_sec_per_iteration': c['e2e_sec_per_iteration']}
    if model_name == 'mavi' and res['checks']:
        spi = res['checks'][0]['e2e_sec_per_iteration']
        if spi:
            h30 = 30000 * spi / 3600
            proj['mavi_30k_hours_per_seed_fp16'] = h30
            proj['mavi_3_seeds_hours'] = 3 * h30
            proj['d2_ladder_choice'] = '30k x 3 seeds' if 3 * h30 <= 22 else '30k x 2 seeds'
    if model_name == 'gpdl' and res['checks']:
        spi = res['checks'][0]['e2e_sec_per_iteration']
        if spi:
            proj['gpdl_50k_hours_per_seed_fp32'] = 50000 * spi / 3600
            proj['gpdl_3_seeds_hours'] = 3 * 50000 * spi / 3600
    res['projection'] = proj
    save()
    log('done')
    return 0
