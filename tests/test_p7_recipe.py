"""P7 pre-freeze tests: the training recipe against the authors' code (schedule, loss, sampler, flip), IR channel order, fp16 NaN check."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7_authors_ref as A                                           # noqa: E402

from edalint.datasets.circuitnet import assemble                     # noqa: E402
from edalint_p7 import models as P                                   # noqa: E402
from edalint_p7.data import (CONGESTION_CHANNELS, IR_CHANNELS, MemorySource, model_input, stack_inputs)   # noqa: E402
from edalint_p7.recipe import BLOCK, LOSS_WEIGHT, MIN_LR, RECIPE, BlockSampler, cosine_lr, flip_pair   # noqa: E402
from edalint_p7.train import loss_fn, make_batch, train              # noqa: E402


# -- recipe constants ----------------------------------------------------------------------------------------------------------
def test_recipe_constants_match_authors_configs():
    assert RECIPE['ir_drop'] == dict(model='MAVI', batch_size=2, lr=2e-4, weight_decay=1e-2, loss='l1', flip=False, in_channels=24)
    assert RECIPE['congestion'] == dict(model='GPDL', batch_size=16, lr=2e-4, weight_decay=0.0, loss='mse', flip=True, in_channels=3)
    assert LOSS_WEIGHT == 100.0 and MIN_LR == 1e-7 and BLOCK == 100


# -- schedule ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('total', [2000, 30000, 50000, 200000])
def test_cosine_schedule_equals_authors_at_0_1_half_and_last(total):
    ref = A.CosineRestartLr([2e-4], [total], [1], 1e-7)
    for it in (0, 1, total // 2, total - 1):
        assert cosine_lr(it, total, 2e-4) == pytest.approx(ref.get_lr(it, 2e-4), rel=0, abs=1e-18)
    assert cosine_lr(0, total, 2e-4) == pytest.approx(2e-4, abs=1e-15)         # starts at the base lr
    assert cosine_lr(total - 1, total, 2e-4) < 1e-6                              # ends near the floor (the floor itself is reached at it = total)


def test_cosine_schedule_is_compressed_not_truncated():
    # the lr at the last iteration of a short run equals the floor of the full shape, whatever the horizon
    assert cosine_lr(1999, 2000, 2e-4) < 1e-6 and cosine_lr(1999, 200000, 2e-4) > 1.9e-4


# -- loss ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('task,ref,shape', [('ir_drop', A.L1Loss, (2, 8, 8)), ('congestion', A.MSELoss, (4, 1, 8, 8))])
def test_loss_equals_100_times_mean_and_authors(task, ref, shape):
    g = torch.Generator().manual_seed(0)
    p, t = torch.rand(shape, generator=g), torch.rand(shape, generator=g)
    mine = loss_fn(task)(p, t)
    expect = 100.0 * (F_l1(p, t) if task == 'ir_drop' else F_mse(p, t))
    assert float(mine) == pytest.approx(float(expect), rel=1e-6)
    assert float(mine) == pytest.approx(float(ref()(p, t)), rel=1e-6)


def F_l1(p, t):
    return (p - t).abs().mean()


def F_mse(p, t):
    return ((p - t) ** 2).mean()


# -- sampler ---------------------------------------------------------------------------------------------------------------------
def _stream(sampler, iters):
    return [sampler.next_batch(it).tolist() for it in range(iters)]


def test_block100_gives_a_fresh_shuffle_every_100_iterations():
    n, b = 1000, 2
    s = BlockSampler(n, b, seed=1, mode='block100')
    batches = _stream(s, 300)
    blocks = [[i for bt in batches[k * 100:(k + 1) * 100] for i in bt] for k in range(3)]
    for blk in blocks:
        assert len(blk) == 200 and len(set(blk)) == 200               # no replacement inside a block
    assert blocks[0] != blocks[1] and blocks[1] != blocks[2]
    assert len(set(blocks[0]) & set(blocks[1])) > 0                    # blocks are independent draws: overlap expected (200 of 1000, twice)
    s2 = BlockSampler(n, b, seed=1, mode='block100')
    assert _stream(s2, 300) == batches                                 # deterministic per seed


def test_block100_wraps_when_the_block_needs_more_than_n_rows():
    s = BlockSampler(50, 16, seed=0, mode='block100')                  # 3 batches per permutation, 100 batches per block
    batches = _stream(s, 100)
    for k in range(0, 99, 3):                                          # every 3 consecutive batches inside one permutation are distinct
        flat = [i for bt in batches[k:k + 3] for i in bt]
        assert len(set(flat)) == len(flat) == 48
    assert all(len(bt) == 16 for bt in batches)                        # drop_last: always full batches


def test_epoch_mode_is_a_continuous_stream_like_the_authors_iterloader():
    """The pinned build_dataset.IterLoader.__iter__ returns self, so train.py's `break` every 100 iterations does not restart the loader:
    consecutive epochs, no reshuffle at block boundaries. Reproduce train.py's loop with the verbatim IterLoader and check that structure."""
    n, b = 1000, 2
    dl = A.IterLoader(DataLoader(list(range(n)), batch_size=b, shuffle=True, drop_last=True))
    seen = []
    for _ in range(3):                                                 # three train.py print blocks of 100 iterations
        k = 0
        for batch in dl:
            seen += batch.tolist()
            k += 1
            if k == 100:
                break
    assert len(seen) == 600 and len(set(seen)) == 600                  # three blocks together: no repeated row (still inside epoch 1 of 500 its? 300 its < 500)
    mine = BlockSampler(n, b, seed=1, mode='epoch')
    flat = [i for bt in _stream(mine, 300) for i in bt]
    assert len(set(flat)) == 600                                       # same structure: no restart at block borders
    mine_block = BlockSampler(n, b, seed=1, mode='block100')
    flat_b = [i for bt in _stream(mine_block, 300) for i in bt]
    assert len(set(flat_b)) < 600                                      # the section-11 reading restarts: repeats across blocks (600 draws from 1000)


def test_sampler_needs_a_full_batch():
    with pytest.raises(ValueError):
        BlockSampler(3, 16, 0)


# -- flip --------------------------------------------------------------------------------------------------------------------------
def test_flip_pair_is_mmcv_horizontal_flip_of_feature_and_label_together():
    rng = np.random.default_rng(0)
    x, y = rng.random((3, 6, 8)).astype(np.float32), rng.random((6, 8)).astype(np.float32)
    ref_x = np.flip(x.transpose(1, 2, 0), axis=1).transpose(2, 0, 1)       # mmcv.imflip_(H, W, C array, 'horizontal') = reverse W
    ref_y = np.flip(y, axis=1)
    flipped = unflipped = 0
    for k in range(200):
        fx, fy = flip_pair(x, y, np.random.default_rng(k))
        if np.array_equal(fx, x):
            assert np.array_equal(fy, y)
            unflipped += 1
        else:
            assert np.array_equal(fx, ref_x) and np.array_equal(fy, ref_y)   # both flipped, identically
            flipped += 1
    assert flipped > 60 and unflipped > 60                              # probability 0.5


def test_flip_is_applied_for_congestion_only():
    rng = np.random.default_rng(3)
    xs = [rng.random((3, 8, 8)).astype(np.float32) for _ in range(4)]
    ys = [rng.random((8, 8)).astype(np.float32) for _ in range(4)]
    src = MemorySource([f'RISCY_freq_{i}' for i in range(4)], xs, ys)
    changed = {'congestion': 0, 'ir_drop': 0}
    for task in changed:
        irx = [np.concatenate([x, x[:1].repeat(21, 0)]) for x in xs] if task == 'ir_drop' else xs
        s = MemorySource(src.ids, irx, ys)
        for k in range(40):
            xb, yb = make_batch(task, s, [0, 1], np.random.default_rng(k), RECIPE[task]['flip'], 'cpu')
            ref = torch.from_numpy(np.stack(irx[:2]))
            ref = ref.unsqueeze(1) if task == 'ir_drop' else ref
            changed[task] += int(not torch.equal(xb, ref))
    assert changed['ir_drop'] == 0 and changed['congestion'] > 0


# -- IR channel order -----------------------------------------------------------------------------------------------------------------
def _fake_members(task, h=16, seed=0):
    rng = np.random.default_rng(seed)
    feats = {'ir_drop': ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t', 'IR_drop'],
             'congestion': ['macro_region', 'RUDY', 'RUDY_pin', 'GR_horizontal_overflow', 'GR_vertical_overflow']}[task]
    return {f: (rng.random((20, h, h)) + 0.1 if f == 'power_t' else rng.random((h, h)) + 0.1) for f in feats}


def test_ir_channel_order_and_layout():
    mem = _fake_members('ir_drop')
    ins, lab = assemble('ir_drop', 'authors', lambda f: mem[f])
    x = stack_inputs('ir_drop', ins)
    assert x.shape == (24, 16, 16) and x.dtype == np.float32 and len(IR_CHANNELS) == 24
    for k, f in enumerate(['power_i', 'power_s', 'power_sca', 'power_all']):
        np.testing.assert_array_equal(x[k], ins[f].astype(np.float32))
    np.testing.assert_array_equal(x[4:], ins['power_t'].astype(np.float32))
    assert model_input('ir_drop', torch.from_numpy(x)[None]).shape == (1, 1, 24, 16, 16)       # irdrop_dataset.py: (1, 24, H, W) per sample


def test_mavi_uses_exactly_the_first_four_depth_slices():
    m = P.MAVI(1, 4).eval()
    x = torch.rand(1, 1, 24, 16, 16)
    only_tail, only_head = x.clone(), x.clone()
    only_tail[:, :, :4] = 0
    only_head[:, :, 4:] = 0
    with torch.no_grad():
        assert float(m(only_tail).abs().max()) == 0.0                   # power_i/s/sca/all zero -> prediction zero whatever the rest says
        assert float(m(only_head).abs().max()) > 0.0


def test_congestion_channel_order():
    mem = _fake_members('congestion')
    ins, _ = assemble('congestion', 'authors', lambda f: mem[f])
    x = stack_inputs('congestion', ins)
    assert x.shape == (3, 16, 16) and CONGESTION_CHANNELS == ['macro_region', 'RUDY', 'RUDY_pin']
    for k, f in enumerate(CONGESTION_CHANNELS):
        np.testing.assert_array_equal(x[k], ins[f].astype(np.float32))


# -- training loop: fp16 NaN check and bookkeeping ----------------------------------------------------------------------------------
def _toy_source(task, n=20, h=16):
    rng = np.random.default_rng(0)
    c = RECIPE[task]['in_channels']
    return MemorySource([f'RISCY_freq_{i}' for i in range(n)], [rng.random((c, h, h)).astype(np.float32) for _ in range(n)],
                        [rng.random((h, h)).astype(np.float32) for _ in range(n)])


@pytest.mark.parametrize('task', ['congestion', 'ir_drop'])
def test_fp16_autocast_train_steps_have_no_nan(task):
    rec = train(task, _toy_source(task), iters=3, seed=0, precision='fp16', device='cpu')
    m = rec.pop('_model')
    assert rec['nonfinite_losses'] == 0 and rec['diverged_at'] is None
    assert all(np.isfinite(rec['loss_per_100_iters']))
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert rec['deviation'] == 'fp16' and 'skipped_steps' in rec and rec['precision'] == 'fp16'


def test_fp32_train_record_and_schedule_endpoints():
    rec = train('congestion', _toy_source('congestion', 20), iters=4, seed=0)
    rec.pop('_model')
    assert rec['deviation'] is None and rec['batch_size'] == 16 and rec['flip_augmentation'] is True
    assert rec['lr_first'] == pytest.approx(2e-4) and rec['lr_last'] == pytest.approx(cosine_lr(3, 4, 2e-4))
    assert rec['iters_done'] == 4


def test_train_refuses_rows_with_non_finite_values():
    src = _toy_source('congestion', 20)
    src._x[3][0, 0, 0] = np.nan
    src.nonfinite_ids = [src.ids[3]]
    with pytest.raises(ValueError, match='non-finite'):
        train('congestion', src, iters=1, seed=0)


# -- port equivalence (skippable: needs the authors' files, which import mmcv) -------------------------------------------------------
def test_port_equals_authors_modules_if_available():
    import importlib.util
    import types
    src = Path('E:/edalint_data/cn_models_src')
    if not (src / 'models' / 'mavi.py').is_file():
        pytest.skip('authors source copy not present')
    stubs = {}
    m, c, u, pw = (types.ModuleType(n) for n in ('mmcv', 'mmcv.cnn', 'mmcv.utils', 'mmcv.utils.parrots_wrapper'))
    c.constant_init = lambda *a, **k: None
    c.kaiming_init = lambda *a, **k: None
    pw._BatchNorm = torch.nn.modules.batchnorm._BatchNorm
    m.cnn, m.utils, u.parrots_wrapper = c, u, pw
    stubs.update({'mmcv': m, 'mmcv.cnn': c, 'mmcv.utils': u, 'mmcv.utils.parrots_wrapper': pw})
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        def load(name):
            spec = importlib.util.spec_from_file_location('orig_' + name, src / 'models' / f'{name}.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        torch.manual_seed(0)
        pairs = (('mavi', 'MAVI', P.MAVI(1, 4), dict(in_channels=1, out_channels=4), torch.rand(1, 1, 24, 32, 32)),
                 ('gpdl', 'GPDL', P.GPDL(3, 1), dict(in_channels=3, out_channels=1), torch.rand(2, 3, 32, 32)))
        for mod, cls, port, kw, x in pairs:
            orig = getattr(load(mod), cls)(**kw)
            orig.init_weights(None)
            so, sp = orig.state_dict(), port.state_dict()
            assert list(so) == list(sp) and all(so[k].shape == sp[k].shape for k in so)
            port.load_state_dict(so)
            orig.eval()
            port.eval()
            with torch.no_grad():
                assert float((orig(x) - port(x)).abs().max()) == 0.0
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
