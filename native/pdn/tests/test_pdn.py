"""The native solver against edalint.spice: analytic circuits, random networks, both parsers crossed with both
solvers, error cases, and (if present on disk) real ICCAD'23 netlists."""
import os
from pathlib import Path

import numpy as np
import pytest

import edalint_pdn as pdn
from edalint import spice


def write(tmp_path, text, name="n.sp"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def by_name(net, volt):
    return dict(zip(net.names.tolist(), volt))


def test_series_chain_analytic(tmp_path):
    p = write(tmp_path, "R0 n1_m9_0_0 n1_m1_0_0 2.000000 \nV0 n1_m9_0_0 0 1.100000 \nI0 n1_m1_0_0 0 1.000000e-01 \n.op\n.end")
    net = pdn.parse_netlist(p)
    assert net.counts() == {"nodes": 3, "R": 1, "R_zero": 0, "I": 1, "V": 1}
    for m in pdn.METHODS:
        res = pdn.solve_dc(net, m)
        assert by_name(net, res.voltage)["n1_m1_0_0"] == pytest.approx(0.9)
        assert res.info["rel_residual"] < 1e-12


def test_decimals_and_ground_written_first(tmp_path):
    # '.' is not a comment character, and "I 0 n v" is a source of v into n, i.e. a sink of -v
    p = write(tmp_path, "R0 n1_m9_0_0 n1_m1_0_0 0.892706\nV0 0 n1_m9_0_0 -1.1\nI0 0 n1_m1_0_0 -1.0\n.op\n.end\n")
    net = pdn.parse_netlist(p)
    assert net.r_ohm[0] == pytest.approx(0.892706) and net.v_volt[0] == pytest.approx(1.1) and net.i_amp[0] == pytest.approx(1.0)
    assert 1.1 - by_name(net, pdn.solve_dc(net).voltage)["n1_m1_0_0"] == pytest.approx(0.892706)


def test_zero_ohm_merge_and_floating_island(tmp_path):
    txt = ("R0 n1_m9_0_0 n1_m1_0_0 1.0\nR1 n1_m1_0_0 n1_m1_2000_0 0\nV0 n1_m9_0_0 0 1.1\nI0 n1_m1_2000_0 0 0.1\n"
           "R2 n1_m1_8000_0 n1_m1_8000_2000 5.0\n.op\n.end\n")
    p = write(tmp_path, txt)
    net = pdn.parse_netlist(p)
    res = pdn.solve_dc(net)
    v = by_name(net, res.voltage)
    assert v["n1_m1_2000_0"] == pytest.approx(1.0) and v["n1_m1_0_0"] == pytest.approx(1.0)
    assert np.isnan(v["n1_m1_8000_0"]) and np.isnan(v["n1_m1_8000_2000"])
    assert res.info["n_floating_merged_nodes"] == 2


def test_conflicting_supplies_are_rejected(tmp_path):
    p = write(tmp_path, "R0 a b 0\nV0 a 0 1.1\nV1 b 0 1.0\nR1 a c 1\n.op\n.end\n")
    with pytest.raises(ValueError, match="conflicting"):
        pdn.solve_dc(pdn.parse_netlist(p))
    with pytest.raises(ValueError, match="conflicting"):
        spice.solve_dc(spice.parse_netlist(p))


def test_non_numeric_value_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="non-numeric"):
        pdn.parse_netlist(write(tmp_path, "R0 a b x1\n"))


def test_layer_and_coordinates_match_the_python_parser(tmp_path):
    p = write(tmp_path, "R0 n1_m1_0_2000 n1_m4_4000_6000 1\nR1 n1_m4_4000_6000 odd 2\nV0 n1_m4_4000_6000 0 1\n.op\n")
    a, b = spice.parse_netlist(p), pdn.parse_netlist(p)
    for name in ("n1_m1_0_2000", "n1_m4_4000_6000", "odd", "0"):
        i, j = list(a.names).index(name), list(b.names).index(name)
        assert a.layer[i] == b.layer[j]
        np.testing.assert_equal([a.x_um[i], a.y_um[i]], [b.x_um[j], b.y_um[j]])


def random_grid(tmp_path, n=20, seed=0, zero_frac=0.05, islands=3):
    """Two-layer mesh with vias, random resistances, some zero-ohm segments, supplies on the top layer, sinks on m1,
    and a few disconnected islands."""
    rng = np.random.default_rng(seed)
    lines, k = [], 0

    def node(layer, x, y):
        return f"n1_{layer}_{x * 2000}_{y * 2000}"

    for layer in ("m1", "m4"):
        for x in range(n):
            for y in range(n):
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx < n and y + dy < n and (layer == "m1" or (x + y) % 2 == 0):
                        r = 0.0 if rng.random() < zero_frac else rng.uniform(0.05, 3.0)
                        lines.append(f"R{k} {node(layer, x, y)} {node(layer, x + dx, y + dy)} {r:.6f}"); k += 1
    for x in range(0, n, 3):
        for y in range(0, n, 3):
            lines.append(f"R{k} {node('m1', x, y)} {node('m4', x, y)} {rng.uniform(0.01, 0.5):.6f}"); k += 1
    for j, (x, y) in enumerate(((0, 0), (n - 2, n - 2), (0, n - 2))):
        lines.append(f"V{j} {node('m4', x, y)} 0 1.100000")
    for j in range(40):
        lines.append(f"I{j} {node('m1', rng.integers(n), rng.integers(n))} 0 {rng.uniform(1e-4, 5e-3):.6e}")
    for j in range(islands):
        lines.append(f"R{k} isl_{j}_a isl_{j}_b 1.0"); k += 1
        lines.append(f"I{100 + j} isl_{j}_a 0 1e-3")
    return write(tmp_path, "\n".join(lines) + "\n.op\n.end\n", f"grid{seed}.sp")


@pytest.mark.parametrize("seed", range(4))
def test_random_networks_match_edalint_for_every_parser_and_solver(tmp_path, seed):
    p = random_grid(tmp_path, seed=seed)
    py_net, nat_net = spice.parse_netlist(p), pdn.parse_netlist(p)
    ref = by_name(py_net, spice.solve_dc(py_net, "direct").voltage)
    for net in (py_net, nat_net):
        for m in pdn.METHODS:
            got = by_name(net, pdn.solve_dc(net, m, tol=1e-13).voltage)
            assert got.keys() == ref.keys()
            a = np.array([ref[k] for k in ref])
            b = np.array([got[k] for k in ref])
            assert np.array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(b[~np.isnan(b)], a[~np.isnan(a)], rtol=0, atol=1e-9)
    # the Python solver on the native parse agrees too (the parsers produce the same circuit)
    cross = by_name(nat_net, spice.solve_dc(nat_net, "direct").voltage)
    np.testing.assert_allclose([cross[k] for k in ref if not np.isnan(ref[k])],
                               [ref[k] for k in ref if not np.isnan(ref[k])], atol=1e-12)


def test_exact_ir_drop_matches_edalint(tmp_path):
    p = random_grid(tmp_path, n=16, seed=7, islands=0)
    a, ia = spice.exact_ir_drop(p, (16, 16), interp="linear")
    b, ib = pdn.exact_ir_drop(p, (16, 16), interp="linear")
    np.testing.assert_allclose(b, a, atol=1e-9)
    assert ia["vdd"] == ib["vdd"] and ib["n_nodes"] == ia["n_nodes"]


ICCAD = Path(os.environ.get("EDALINT_ICCAD23", "/mnt/e/edalint_data/iccad23/repo")) / "benchmarks" / "fake-circuit-data"


@pytest.mark.realdata
@pytest.mark.skipif(not ICCAD.is_dir(), reason="ICCAD'23 benchmark not on disk")
def test_real_iccad_netlist_matches_edalint():
    p = str(sorted(ICCAD.glob("current_map*.sp"))[0])
    py_net, nat_net = spice.parse_netlist(p), pdn.parse_netlist(p)
    assert nat_net.counts() == py_net.counts()
    ref = by_name(py_net, spice.solve_dc(py_net, "direct").voltage)
    got = by_name(nat_net, pdn.solve_dc(nat_net, "direct").voltage)
    keys = [k for k in ref if not np.isnan(ref[k])]
    assert max(abs(got[k] - ref[k]) for k in keys) < 1e-9
