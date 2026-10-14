"""P7 training loop: the authors' train.py recipe (see recipe.py) on a row source. Train-side only; no test access.

Per iteration (train.py 133-163): lr from the cosine schedule (compressed to `iters`), forward, zero_grad, loss (100 x mean L1 for MAVI /
100 x mean MSE for GPDL), backward, AdamW step. Mean loss of every 100 iterations is the logged curve. No validation; the LAST state is
the one returned/saved. Optional fp16 autocast + GradScaler (protocol_p7.md section 11 D3): loss and BatchNorm in fp32, skipped steps counted.
"""
from __future__ import annotations

import hashlib
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from . import models as P
from .data import model_input, model_target
from .recipe import BETAS, BLOCK, LOSS_WEIGHT, MIN_LR, RECIPE, BlockSampler, cosine_lr, flip_pair

MAX_CONSECUTIVE_NONFINITE = 100


def build_model(task: str):
    return P.MAVI(1, 4) if task == 'ir_drop' else P.GPDL(3, 1)


def loss_fn(task: str):
    return P.l1_loss100 if RECIPE[task]['loss'] == 'l1' else P.mse_loss100


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware(device: str) -> Dict[str, Any]:
    from edalint.protocol import cpu_info
    h = {'device': device, 'torch': torch.__version__, 'threads': torch.get_num_threads(), 'platform': platform.platform(), **cpu_info()}
    if device.startswith('cuda') and torch.cuda.is_available():
        h['gpu'] = torch.cuda.get_device_name(0)
    return h


def sha256_bytes_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def make_batch(task: str, source, idx, rng: np.random.Generator, flip: bool, device: str):
    xs, ys = [], []
    for k in idx:
        x, y = source.get(int(k))
        if flip:
            x, y = flip_pair(x, y, rng)
        xs.append(np.asarray(x, np.float32))
        ys.append(np.asarray(y, np.float32))
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.from_numpy(np.stack(ys)).to(device)
    return model_input(task, x), model_target(task, y)


def train(task: str, source, iters: int, seed: int, precision: str = 'fp32', device: str = 'cpu', sampler_mode: str = 'block100',
          ckpt_path: Optional[str] = None, log=print, allow_nonfinite_rows: bool = False, model=None) -> Dict[str, Any]:
    """Train from scratch for `iters` iterations on `source` (len, get(k) -> (x, y), nonfinite_ids). Returns the train record
    (budget, loss per 100 iterations, hardware, wall time, checkpoint sha256, deviations); the trained model is in rec['_model'] (not JSON)."""
    if task not in RECIPE:
        raise ValueError(f'task must be one of {sorted(RECIPE)}')
    if precision not in ('fp32', 'fp16'):
        raise ValueError("precision must be 'fp32' or 'fp16'")
    if getattr(source, 'nonfinite_ids', None) and not allow_nonfinite_rows:
        raise ValueError(f'{len(source.nonfinite_ids)} train rows contain non-finite values (e.g. {source.nonfinite_ids[:3]}); '
                         'the protocol has no rule for them, refusing to train')
    cfg = RECIPE[task]
    seed_all(seed)
    if model is None:
        model = build_model(task)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=BETAS, weight_decay=cfg['weight_decay'])
    lf = loss_fn(task)
    fp16 = precision == 'fp16'
    dev_type = 'cuda' if device.startswith('cuda') else 'cpu'
    scaler = torch.amp.GradScaler(dev_type, enabled=fp16)
    sampler = BlockSampler(len(source), cfg['batch_size'], seed, sampler_mode)
    rng = np.random.default_rng([seed, 1])             # flip stream, separate from the sampler
    model.train()
    t0 = time.perf_counter()
    block, block_n, curve, skipped, nonfinite, consec = [], 0, [], 0, 0, 0
    diverged_at = None
    lrs = []
    for it in range(iters):
        lr = cosine_lr(it, iters, cfg['lr'], MIN_LR)
        lrs.append(lr)
        for g in opt.param_groups:
            g['lr'] = lr
        x, y = make_batch(task, source, sampler.next_batch(it), rng, cfg['flip'], device)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=fp16):
            pred = model(x)
        loss = lf(pred.float(), y)
        lv = float(loss.item())
        opt.zero_grad()
        if not np.isfinite(lv):
            nonfinite += 1
            consec += 1
            if not fp16 or consec >= MAX_CONSECUTIVE_NONFINITE:
                diverged_at = it
                log(f'non-finite loss at iteration {it}: stopping')
                break
            scaler.scale(loss).backward()          # the scaler skips the step on inf/nan gradients and lowers its scale
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
            continue
        consec = 0
        if fp16:
            before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
        else:
            loss.backward()
            opt.step()
        block.append(lv)
        if (it + 1) % BLOCK == 0:
            curve.append(float(np.mean(block)) if block else float('nan'))
            block = []
            log(f'{task} iter {it + 1}/{iters} loss {curve[-1]:.4f} lr {lr:.3e} {time.perf_counter() - t0:.0f}s')
    if block and diverged_at is None:
        curve.append(float(np.mean(block)))
    wall = time.perf_counter() - t0
    rec: Dict[str, Any] = {
        'task': task, 'model': cfg['model'], 'seed': seed, 'iters_requested': iters, 'iters_done': it + 1 if diverged_at is None else diverged_at,
        'batch_size': cfg['batch_size'], 'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'], 'optimizer': 'AdamW', 'betas': list(BETAS),
        'schedule': f'cosine {cfg["lr"]} -> {MIN_LR} compressed to {iters} iterations, set per iteration',
        'loss': f'{LOSS_WEIGHT:g} x mean {"L1" if cfg["loss"] == "l1" else "MSE"}',
        'flip_augmentation': cfg['flip'], 'sampler_mode': sampler_mode, 'precision': precision,
        'deviation': 'fp16' if fp16 else None, 'skipped_steps': skipped, 'nonfinite_losses': nonfinite, 'diverged_at': diverged_at,
        'n_train_rows': len(source), 'rows_per_design': {d: sum(1 for x in source.designs if x == d) for d in sorted(set(source.designs))},
        'lr_first': lrs[0] if lrs else None, 'lr_last': lrs[-1] if lrs else None, 'loss_per_100_iters': curve, 'wall_seconds': wall, 'hardware': hardware(device), 'checkpoint': 'last state, no validation',
    }
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict()}, ckpt_path)
        rec['checkpoint_path'] = str(ckpt_path)
        rec['checkpoint_sha256'] = sha256_bytes_file(ckpt_path)
    rec['_model'] = model
    return rec


@torch.no_grad()
def predict(task: str, model, x: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """eval mode (BatchNorm running statistics), batch 1, fp32: (C, H, W) float32 -> (H, W) float32 (test.py)."""
    model.eval()
    xb = model_input(task, torch.from_numpy(np.asarray(x, np.float32))[None].to(device))
    out = model(xb)
    return out.reshape(out.shape[-2:]).float().cpu().numpy()
