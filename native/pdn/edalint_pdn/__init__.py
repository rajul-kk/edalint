"""Native (C++) parse and DC solve for ICCAD'23 power-delivery netlists, as a drop-in for edalint.spice.

parse_netlist and solve_dc return edalint.spice.Netlist and SolveResult, so either parser can feed either solver, and
exact_ir_drop rasterises with edalint.spice.rasterize_ir_drop (same interpolation, same label convention). Node order
differs from edalint.spice.parse_netlist (first appearance per line instead of all first nodes, then all second
nodes); compare results by node name.

Methods: 'direct' (sparse LDL^T, AMD ordering), 'cg_ic' (CG with incomplete Cholesky), 'cg' (CG with Jacobi, like
edalint.spice 'cg'). edalint's 'cg_amg' has no native counterpart.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from edalint import spice

from . import _pdn

__all__ = ["parse_netlist", "solve_dc", "exact_ir_drop", "METHODS"]

METHODS = ("direct", "cg_ic", "cg")


def parse_netlist(path: str) -> spice.Netlist:
    d = _pdn.parse_netlist(str(path))
    return spice.Netlist(
        names=np.array(d["names"], dtype=object), layer=np.array(d["layer"], dtype=object),
        x_um=d["x_um"], y_um=d["y_um"], r_a=d["r_a"], r_b=d["r_b"], r_ohm=d["r_ohm"],
        i_node=d["i_node"], i_amp=d["i_amp"], v_node=d["v_node"], v_volt=d["v_volt"],
        n_lines=int(d["n_lines"]), parse_seconds=float(d["parse_seconds"]))


def solve_dc(net: spice.Netlist, method: str = "direct", tol: float = 1e-12, max_iterations: int = 20000) -> spice.SolveResult:
    """Same reduction and conventions as edalint.spice.solve_dc; accepts a Netlist from either parser."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; native methods are {METHODS}")
    volt, info = _pdn.solve_dc(net.n_nodes, net.r_a, net.r_b, net.r_ohm, net.i_node, net.i_amp, net.v_node,
                               net.v_volt, method, tol, max_iterations)
    return spice.SolveResult(volt, dict(info))


def exact_ir_drop(netlist_path: str, shape: Tuple[int, int], vdd: Optional[float] = None, method: str = "direct",
                  interp: str = "cubic", origin: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, Dict[str, float]]:
    """Native parse + solve, then edalint's rasterisation. Same signature and info keys as edalint.spice.exact_ir_drop."""
    t0 = time.time()
    net = parse_netlist(netlist_path)
    vd = float(np.max(net.v_volt)) if vdd is None else vdd
    res = solve_dc(net, method=method)
    t1 = time.time()
    m = spice.rasterize_ir_drop(net, res.voltage, shape, vdd=vd, layer=spice.lowest_layer(net), method=interp, origin=origin)
    t2 = time.time()
    info = {**res.info, "seconds_parse": net.parse_seconds, "seconds_rasterize": t2 - t1, "seconds_total": t2 - t0,
            "vdd": vd, **{f"n_{k}": v for k, v in net.counts().items()}}
    return m, info
