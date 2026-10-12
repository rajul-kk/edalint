"""ICCAD'23 Problem C SPICE PDN netlists: parse, exact DC solve, and mapping of node voltages to the 1 um pixel grid.

Netlist format (contest description section 3b, docs/format_notes.md section 1.3):
    R<k> <nodeA> <nodeB> <ohm>      resistor (wire segment or via; some are exactly 0 ohm: those nodes are merged)
    I<k> <node> 0 <ampere>          current sink (a cell) at a lowest-layer node: current flows from <node> to ground
    V<k> <node> 0 <volt>            ideal supply (C4 bump) at a top-layer node
    .op / .end
Node name: n1_<layer>_<x>_<y>, x and y in database units, 2000 dbu = 1 um. Ground is node `0`.

IMPORTANT: a line must not be parsed with '.' as a comment character (that truncates '0.892706' to '0').

Label convention reproduced here (docs/format_notes.md section 1.7, src/generate_benchmark_maps.py): ir_drop = VDD - V at the
lowest metal layer (m1) nodes, interpolated onto the integer 1 um grid, axis 0 = x, cubic Clough-Tocher inside the convex
hull of the nodes and nearest-node value outside it (the stored real labels are non-zero there; checked on train cases). Whether the real-circuit labels were interpolated the same way is not documented; `compare_to_label` measures it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components

DBU_PER_UM = 2000.0


@dataclass
class Netlist:
    names: np.ndarray                 # node names; index 0 is always the ground node '0'
    layer: np.ndarray                 # layer string per node ('' for ground)
    x_um: np.ndarray
    y_um: np.ndarray
    r_a: np.ndarray
    r_b: np.ndarray
    r_ohm: np.ndarray
    i_node: np.ndarray
    i_amp: np.ndarray                 # positive = sink (current drawn from the node)
    v_node: np.ndarray
    v_volt: np.ndarray                # ideal source value at v_node (relative to ground)
    n_lines: int = 0
    parse_seconds: float = 0.0

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    def counts(self) -> Dict[str, int]:
        return {'nodes': self.n_nodes, 'R': len(self.r_ohm), 'R_zero': int((self.r_ohm == 0).sum()),
                'I': len(self.i_amp), 'V': len(self.v_volt)}


def _read_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None, names=['n', 'a', 'b', 'v'], engine='c', dtype=str,
                         on_bad_lines='error', skip_blank_lines=True)
    except Exception:                                           # lines with extra fields: slow tolerant path
        rows = []
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                t = line.split()
                if len(t) >= 4:
                    rows.append(t[:4])
        df = pd.DataFrame(rows, columns=['n', 'a', 'b', 'v'])
    first = df['n'].str[:1].str.upper()
    return df[first.isin(['R', 'I', 'V'])].assign(kind=first[first.isin(['R', 'I', 'V'])])


def parse_netlist(path: str) -> Netlist:
    t0 = time.time()
    df = _read_table(path)
    gnd = {'0', 'gnd', 'GND'}
    allnodes = pd.concat([df['a'], df['b']], ignore_index=True)
    allnodes = allnodes[~allnodes.isin(gnd)]
    uniq = pd.unique(allnodes)
    names = np.concatenate([['0'], uniq.astype(str)])
    index = pd.Series(np.arange(len(names)), index=names)
    a = np.where(df['a'].isin(gnd).values, 0, index.reindex(df['a'].values).fillna(0).astype(np.int64).values)
    b = np.where(df['b'].isin(gnd).values, 0, index.reindex(df['b'].values).fillna(0).astype(np.int64).values)
    val = df['v'].astype(np.float64).values
    kind = df['kind'].values
    r = kind == 'R'
    cur, vol = kind == 'I', kind == 'V'
    # current: I <n> 0 val -> sink at n. If the ground is the first node the sign flips.
    i_node = np.where(a[cur] == 0, b[cur], a[cur])
    i_amp = np.where(a[cur] == 0, -val[cur], val[cur])
    v_node = np.where(a[vol] == 0, b[vol], a[vol])
    v_volt = np.where(a[vol] == 0, -val[vol], val[vol])
    parts = pd.Series(names).str.split('_', expand=True)
    layer = np.array([''] * len(names), dtype=object)
    x = np.full(len(names), np.nan)
    y = np.full(len(names), np.nan)
    if parts.shape[1] >= 4:
        layer = parts[1].fillna('').values.astype(object)
        x = pd.to_numeric(parts[2], errors='coerce').values / DBU_PER_UM
        y = pd.to_numeric(parts[3], errors='coerce').values / DBU_PER_UM
    return Netlist(names, layer, x, y, a[r], b[r], val[r], i_node, i_amp, v_node, v_volt, n_lines=len(df),
                   parse_seconds=time.time() - t0)


@dataclass
class SolveResult:
    voltage: np.ndarray               # per node of the netlist; NaN for floating nodes (no path to a supply)
    info: Dict[str, float] = field(default_factory=dict)


def solve_dc(net: Netlist, method: str = 'direct', tol: float = 1e-12) -> SolveResult:
    """Nodal analysis G v = i with ideal supplies as Dirichlet nodes and zero-ohm resistors merged.

    method: 'direct' (sparse LU, scipy spsolve), 'cg' (Jacobi-preconditioned CG), 'cg_amg' (CG with a pyamg smoothed-aggregation
    preconditioner; needs pyamg). Returns node voltages; info holds timings and the residual of the reduced system.
    """
    t0 = time.time()
    n = net.n_nodes
    # merge zero-ohm resistors
    zr = net.r_ohm == 0
    if zr.any():
        zg = sp.coo_matrix((np.ones(zr.sum()), (net.r_a[zr], net.r_b[zr])), shape=(n, n))
        ncomp, rep = connected_components(zg, directed=False)
    else:
        ncomp, rep = n, np.arange(n)
    ra, rb = rep[net.r_a], rep[net.r_b]
    nz = ~zr & (ra != rb)
    g = 1.0 / net.r_ohm[nz]
    a, b = ra[nz], rb[nz]
    N = ncomp
    G = sp.coo_matrix((np.r_[g, g, -g, -g], (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(N, N)).tocsr()
    fixed = np.zeros(N, dtype=bool)
    vfix = np.zeros(N)
    gnd = rep[0]
    fixed[gnd] = True
    for node, val in zip(rep[net.v_node], net.v_volt):
        if fixed[node] and abs(vfix[node] - val) > 1e-12:
            raise ValueError('conflicting ideal sources on one (merged) node')
        fixed[node] = True
        vfix[node] = val
    rhs_i = np.zeros(N)
    np.add.at(rhs_i, rep[net.i_node], -net.i_amp)                 # sink: current leaves the node
    # floating components: no path to any fixed node
    ncc, cc = connected_components(abs(G), directed=False)
    has_fixed = np.zeros(ncc, dtype=bool)
    has_fixed[cc[fixed]] = True
    floating = ~has_fixed[cc]
    free = ~fixed & ~floating
    fi = np.flatnonzero(free)
    t_asm = time.time()
    Gff = G[fi][:, fi].tocsc()
    rhs = rhs_i[fi] - G[fi][:, np.flatnonzero(fixed)] @ vfix[fixed]
    t_solve0 = time.time()
    if method == 'direct':
        x = sla.spsolve(Gff, rhs)
    elif method in ('cg', 'cg_amg'):
        M = None
        if method == 'cg_amg':
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(Gff.tocsr())
            M = ml.aspreconditioner()
        else:
            d = Gff.diagonal()
            M = sla.LinearOperator(Gff.shape, matvec=lambda v: v / d)
        x, code = sla.cg(Gff, rhs, rtol=tol, atol=0.0, M=M, maxiter=20000) if _cg_has_rtol() else \
            sla.cg(Gff, rhs, tol=tol, atol=0.0, M=M, maxiter=20000)
        if code != 0:
            raise RuntimeError(f'CG did not converge (code {code})')
    else:
        raise ValueError(f'unknown method {method!r}')
    t_solve1 = time.time()
    v_rep = vfix.copy()
    v_rep[fi] = x
    v_rep[floating] = np.nan
    resid = float(np.linalg.norm(Gff @ x - rhs) / max(np.linalg.norm(rhs), 1e-300))
    v = v_rep[rep]
    info = {'method': method, 'n_nodes': n, 'n_free': int(len(fi)), 'n_floating_merged_nodes': int(floating.sum()),
            'seconds_assemble': t_asm - t0, 'seconds_prepare': t_solve0 - t_asm, 'seconds_solve': t_solve1 - t_solve0,
            'rel_residual': resid}
    return SolveResult(v, info)


def _cg_has_rtol() -> bool:
    import inspect
    return 'rtol' in inspect.signature(sla.cg).parameters


def rasterize_ir_drop(net: Netlist, volt: np.ndarray, shape: Tuple[int, int], vdd: float = 1.1, layer: str = 'm1',
                      method: str = 'cubic', offset: float = 0.0, origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """IR-drop map (VDD - V on `layer` nodes) on the integer um grid; axis 0 = x. Zero outside the convex hull."""
    from scipy.interpolate import griddata
    sel = (net.layer == layer) & np.isfinite(volt)
    pts = np.c_[net.x_um[sel] - origin[0], net.y_um[sel] - origin[1]]
    vals = vdd - volt[sel]
    gx, gy = np.meshgrid(np.arange(shape[0]) + offset, np.arange(shape[1]) + offset, indexing='ij')
    out = griddata(pts, vals, (gx, gy), method=method, fill_value=np.nan)
    hole = ~np.isfinite(out)
    if hole.any():                                       # outside the convex hull of the nodes: nearest node value
        out[hole] = griddata(pts, vals, (gx[hole], gy[hole]), method='nearest')
    return out


def lowest_layer(net: Netlist) -> str:
    cand = [l for l in set(net.layer.tolist()) if l]
    return min(cand, key=lambda s: (len(s), s)) if cand else 'm1'


def exact_ir_drop(netlist_path: str, shape: Tuple[int, int], vdd: Optional[float] = None, method: str = 'direct',
                  interp: str = 'cubic', origin: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, Dict[str, float]]:
    """Parse + solve + rasterize. Returns (ir_drop map in volts, timing / solver info). VDD defaults to the supply value."""
    t0 = time.time()
    net = parse_netlist(netlist_path)
    vd = float(np.max(net.v_volt)) if vdd is None else vdd
    res = solve_dc(net, method=method)
    t1 = time.time()
    m = rasterize_ir_drop(net, res.voltage, shape, vdd=vd, layer=lowest_layer(net), method=interp, origin=origin)
    t2 = time.time()
    info = {**res.info, 'seconds_parse': net.parse_seconds, 'seconds_rasterize': t2 - t1, 'seconds_total': t2 - t0,
            'vdd': vd, **{f'n_{k}': v for k, v in net.counts().items()}}
    return m, info


def compare_to_label(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    """Agreement of an exact-solve map with the stored label (volts and millivolts)."""
    e = np.abs(pred - label)
    ref = float(label.max() - label.min())
    return {'mae_mV': float(e.mean() * 1e3), 'max_abs_mV': float(e.max() * 1e3), 'rel_max_err': float(e.max() / max(ref, 1e-300)),
            'pearson': float(np.corrcoef(pred.ravel(), label.ravel())[0, 1]),
            'label_max_mV': float(label.max() * 1e3), 'pred_max_mV': float(pred.max() * 1e3)}
