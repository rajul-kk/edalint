"""The CircuitNet authors' training recipe (protocol_p7.md sections 3.2, 3.3 and 11), written out as small testable pieces.

Source: github.com/circuitnet/CircuitNet commit 41ade1d7e913a418b4495b6b13630b26b0ff7279, routability_ir_drop_prediction/
  train.py (CosineRestartLr, the loop), utils/configs.py (per-task constants), utils/losses.py (loss_weight 100),
  datasets/augmentation.py (Flip), datasets/build_dataset.py (IterLoader).
The cosine horizon is the chosen iteration count (the schedule is compressed, not truncated).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

LOSS_WEIGHT = 100.0
MIN_LR = 1e-7
BETAS = (0.9, 0.999)
BLOCK = 100                       # train.py print_freq: one block of iterations between two loss prints
FLIP_RATIO = 0.5
AUTHORS_ITERS = 200000
SAMPLER_MODES = ('block100', 'epoch')

RECIPE: Dict[str, dict] = {
    'ir_drop': dict(model='MAVI', batch_size=2, lr=2e-4, weight_decay=1e-2, loss='l1', flip=False, in_channels=24),
    'congestion': dict(model='GPDL', batch_size=16, lr=2e-4, weight_decay=0.0, loss='mse', flip=True, in_channels=3),
}


def cosine_lr(it: int, total: int, base: float, min_lr: float = MIN_LR) -> float:
    """train.py CosineRestartLr with periods=[total], restart_weights=[1], min_lr=1e-7, evaluated at iteration index `it` (0-based,
    set before the step): alpha = min(it / total, 1); lr = min + 0.5 (base - min) (cos(pi alpha) + 1)."""
    alpha = min(it / total, 1)
    return min_lr + 0.5 * 1.0 * (base - min_lr) * (math.cos(math.pi * alpha) + 1)


def flip_pair(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """augmentation.Flip: with probability 0.5 flip feature AND label horizontally (the authors' arrays are H x W x C, mmcv.imflip_
    'horizontal' reverses the W axis; here the arrays are C x H x W / H x W, so the last axis). Returns (x, y) as new arrays."""
    if rng.random() < FLIP_RATIO:
        return np.ascontiguousarray(x[..., ::-1]), np.ascontiguousarray(y[..., ::-1])
    return x, y


class BlockSampler:
    """Shuffled batches without replacement, drop_last (DataLoader(shuffle=True, drop_last=True)).

    mode 'block100' (protocol_p7.md section 11 D2): a fresh permutation at the start of every 100-iteration block (iteration % 100 == 0);
        inside a block the permutation is consumed in order and, if it runs out (n < 100 x batch), a new one starts (IterLoader behaviour).
    mode 'epoch': one continuous stream of permutations, restarted only when exhausted. This is what the pinned `build_dataset.IterLoader`
        does: `IterLoader.__iter__` returns self, so the `break` at print_freq in train.py does NOT restart the loader.
    """

    def __init__(self, n: int, batch: int, seed: int, mode: str = 'block100', block: int = BLOCK):
        if mode not in SAMPLER_MODES:
            raise ValueError(f'mode must be one of {SAMPLER_MODES}')
        if n < batch:
            raise ValueError(f'{n} rows cannot fill one batch of {batch} (drop_last)')
        self.n, self.batch, self.mode, self.block = n, batch, mode, block
        self.rng = np.random.default_rng(seed)
        self._perm = None
        self._pos = 0

    def _new(self):
        self._perm = self.rng.permutation(self.n)
        self._pos = 0

    def next_batch(self, it: int) -> np.ndarray:
        if self.mode == 'block100' and it % self.block == 0:
            self._new()
        if self._perm is None or self._pos + self.batch > self.n:
            self._new()
        b = self._perm[self._pos:self._pos + self.batch]
        self._pos += self.batch
        return b
