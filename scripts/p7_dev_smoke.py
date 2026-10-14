"""P7 DEVELOPMENT smoke test (train-only). Reads NO test design: only RISCY and RISCY-FPU maps from the registered-run staging cache
(authors-space float32 maps of the K=400 fit sample), assembled with the authors' scaling (edalint.datasets.circuitnet.assemble).

1. Port check: import the authors' mavi.py / gpdl.py (pinned commit, copied to E:/edalint_data/cn_models_src, mmcv stubbed) and compare with
   edalint_p7.models: state_dict keys and shapes, parameter counts, forward outputs with the original weights loaded into the port.
2. Timing: a few MAVI steps (batch 2) and GPDL steps (batch 16) on CPU, authors' optimizer (AdamW, betas .9/.999, wd as in utils/configs.py),
   loss and cosine schedule; plus MAVI eval-mode forward per sample. Also float16 .npz sizes of the compact export.

Output: results/dev/p7_smoke_p7dev.json (development, train-only; not a result).
Usage: python scripts/p7_dev_smoke.py [--threads 6]
"""
import argparse, importlib.util, json, math, os, sys, time, types, zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from edalint.datasets.circuitnet import assemble      # noqa: E402
from edalint_p7 import models as P                      # noqa: E402

SRC = Path('E:/edalint_data/cn_models_src')
CACHE = Path('E:/edalint_data/n14_registered_cache')
TRAIN_DESIGNS = ('RISCY', 'RISCY-FPU')                  # never a test design
IN = {'ir_drop': ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t'], 'congestion': ['macro_region', 'RUDY', 'RUDY_pin']}
LAB = {'ir_drop': ['IR_drop'], 'congestion': ['GR_horizontal_overflow', 'GR_vertical_overflow']}


def stub_mmcv():
    m = types.ModuleType('mmcv'); c = types.ModuleType('mmcv.cnn'); u = types.ModuleType('mmcv.utils'); pw = types.ModuleType('mmcv.utils.parrots_wrapper')
    c.constant_init = lambda *a, **k: None
    c.kaiming_init = lambda *a, **k: None
    pw._BatchNorm = torch.nn.modules.batchnorm._BatchNorm
    m.cnn, m.utils, u.parrots_wrapper = c, u, pw
    sys.modules.update({'mmcv': m, 'mmcv.cnn': c, 'mmcv.utils': u, 'mmcv.utils.parrots_wrapper': pw})


def load_orig(name):
    spec = importlib.util.spec_from_file_location('orig_' + name, SRC / 'models' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def port_check():
    stub_mmcv()
    out = {}
    torch.manual_seed(0)
    om = load_orig('mavi').MAVI(in_channels=1, out_channels=4)
    om.init_weights(None)
    pm = P.MAVI(1, 4)
    og = load_orig('gpdl').GPDL(in_channels=3, out_channels=1)
    og.init_weights(None)
    pg = P.GPDL(3, 1)
    for tag, o, p, x in (('MAVI', om, pm, torch.rand(1, 1, 24, 64, 64)), ('GPDL', og, pg, torch.rand(2, 3, 64, 64))):
        so, sp = o.state_dict(), p.state_dict()
        same_keys = list(so) == list(sp)
        same_shapes = all(so[k].shape == sp[k].shape for k in so) if same_keys else False
        p.load_state_dict(so)
        o.eval(); p.eval()
        with torch.no_grad():
            d = float((o(x) - p(x)).abs().max())
        out[tag] = {'same_state_dict_keys': same_keys, 'same_shapes': same_shapes, 'n_params_orig': sum(v.numel() for v in so.values() if v.dtype.is_floating_point),
                    'n_params_port': sum(v.numel() for v in sp.values() if v.dtype.is_floating_point), 'max_abs_forward_diff_64x64': d}
    return out


def names(task, design):
    d = CACHE / task / 'authors' / design
    have = None
    for f in IN[task] + LAB[task]:
        s = {p.stem for p in (d / f).glob('*.npy')}
        have = s if have is None else have & s
    return sorted(have)


def sample(task, design, name):
    def get(f):
        return np.load(CACHE / task / 'authors' / design / f / f'{name}.npy')
    ins, lab = assemble(task, 'authors', get)
    if task == 'ir_drop':
        x = np.concatenate([ins['power_i'][None], ins['power_s'][None], ins['power_sca'][None], ins['power_all'][None], ins['power_t']], 0)
    else:
        x = np.stack([ins[f] for f in IN[task]], 0)
    return x.astype(np.float32), lab.astype(np.float32)


def cos_lr(it, base, total, mn=1e-7):
    return mn + 0.5 * (base - mn) * (math.cos(math.pi * min(it / total, 1)) + 1)


def timing(task, model, loss_fn, lr, wd, batch, xs, ys, steps, total_iters=200000):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    model.train()
    times, losses = [], []
    n = len(xs)
    for s in range(steps):
        idx = [(s * batch + j) % n for j in range(batch)]
        x = torch.from_numpy(np.stack([xs[i] for i in idx]))
        y = torch.from_numpy(np.stack([ys[i] for i in idx]))
        if task == 'ir_drop':
            x = x.unsqueeze(1)
        else:
            y = y.unsqueeze(1) if y.ndim == 3 else y
        for g in opt.param_groups:
            g['lr'] = cos_lr(s, lr, total_iters)
        t0 = time.perf_counter()
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        times.append(time.perf_counter() - t0); losses.append(float(loss))
        print(task, 'step', s, round(times[-1], 2), 's loss', round(losses[-1], 4), flush=True)
    return times, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--mavi_steps', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {'label': 'DEVELOPMENT, train-only (RISCY, RISCY-FPU from the registered fit-sample cache); not a result', 'threads': a.threads,
           'torch': torch.__version__, 'orig_code_commit': '41ade1d7e913a418b4495b6b13630b26b0ff7279'}
    res['port_check'] = port_check()
    print(json.dumps(res['port_check'], indent=1), flush=True)

    # compact export size, from a few IR-drop samples
    pool = [(d, n) for d in TRAIN_DESIGNS for n in names('ir_drop', d)[:a.n // 2]]
    res['export_size'] = {}
    xs, ys = [], []
    for d, n in pool:
        x, y = sample('ir_drop', d, n)
        xs.append(x); ys.append(y)
    import io
    b32 = b16 = bz = 0
    for x, y in list(zip(xs, ys))[:10]:
        b32 += x.nbytes + y.nbytes
        b16 += x.astype(np.float16).nbytes + y.astype(np.float16).nbytes
        buf = io.BytesIO(); np.savez_compressed(buf, x=x.astype(np.float16), y=y.astype(np.float16)); bz += buf.getbuffer().nbytes
    res['export_size']['ir_drop_bytes_per_sample'] = {'float32': b32 / 10, 'float16': b16 / 10, 'float16_npz_compressed': bz / 10}
    res['export_size']['ir_drop_nonfinite_in_inputs'] = int(sum((~np.isfinite(x)).any() for x in xs))
    res['export_size']['float16_max_abs_err_inputs'] = float(max(np.abs(x.astype(np.float16).astype(np.float32) - x).max() for x in xs[:10]))
    res['n_samples'] = {'ir_drop': len(xs)}

    torch.manual_seed(0)
    m = P.MAVI(1, 4)
    res['mavi'] = {'n_params': sum(p.numel() for p in m.parameters())}
    t, l = timing('ir_drop', m, P.l1_loss100, 2e-4, 1e-2, 2, xs, ys, a.mavi_steps)
    res['mavi'].update({'batch': 2, 'train_step_seconds': t, 'loss': l, 'sec_per_sample_train': float(np.mean(t[1:] or t) / 2)})
    m.eval()
    ev = []
    with torch.no_grad():
        for i in range(2):
            t0 = time.perf_counter(); m(torch.from_numpy(xs[i])[None, None]); ev.append(time.perf_counter() - t0)
    res['mavi']['eval_forward_seconds_per_sample'] = ev
    del xs, ys, m

    pool = [(d, n) for d in TRAIN_DESIGNS for n in names('congestion', d)[:a.n // 2]]
    xs, ys = zip(*[sample('congestion', d, n) for d, n in pool])
    res['n_samples']['congestion'] = len(xs)
    torch.manual_seed(0)
    g = P.GPDL(3, 1)
    res['gpdl'] = {'n_params': sum(p.numel() for p in g.parameters())}
    steps = len(xs) // 16                                  # one epoch of the 50 samples at the authors' batch 16, drop_last
    t, l = timing('congestion', g, P.mse_loss100, 2e-4, 0.0, 16, list(xs), list(ys), steps)
    res['gpdl'].update({'batch': 16, 'steps': steps, 'train_step_seconds': t, 'loss': l, 'sec_per_sample_train': float(np.mean(t[1:] or t) / 16)})
    g.eval()
    with torch.no_grad():
        t0 = time.perf_counter(); g(torch.from_numpy(np.stack(xs[:16]))); res['gpdl']['eval_forward_seconds_per_16'] = time.perf_counter() - t0
    out = REPO / 'results' / 'dev' / 'p7_smoke_p7dev.json'
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
