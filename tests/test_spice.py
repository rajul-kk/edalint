"""Parser and exact solver against circuits with analytic answers."""
import numpy as np
import pytest

from edalint import spice


def write(tmp_path, text, name='n.sp'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_series_chain_analytic(tmp_path):
    # VDD=1.1 -> 2 ohm -> m1 node with a 0.1 A sink: V = 1.1 - 0.2 -> drop 0.2 V
    p = write(tmp_path, 'R0 n1_m9_0_0 n1_m1_0_0 2.000000 \nV0 n1_m9_0_0 0 1.100000 \nI0 n1_m1_0_0 0 1.000000e-01 \n.op\n.end')
    net = spice.parse_netlist(p)
    assert net.counts() == {'nodes': 3, 'R': 1, 'R_zero': 0, 'I': 1, 'V': 1}
    res = spice.solve_dc(net)
    i = list(net.names).index('n1_m1_0_0')
    assert res.voltage[i] == pytest.approx(0.9)
    assert res.info['rel_residual'] < 1e-12


def test_decimal_resistances_are_not_truncated(tmp_path):
    # '.' must not be a comment character: 0.892706 ohm, not 0
    p = write(tmp_path, 'R0 n1_m9_0_0 n1_m1_0_0 0.892706 \nV0 n1_m9_0_0 0 1.100000 \nI0 n1_m1_0_0 0 1.000000e-00 \n.op\n.end')
    net = spice.parse_netlist(p)
    assert net.r_ohm[0] == pytest.approx(0.892706)
    assert spice.exact_ir_drop(p, (1, 1), method='direct', interp='nearest')[0][0, 0] == pytest.approx(0.892706)


def test_parallel_branches_and_current_division(tmp_path):
    # two parallel 4 ohm branches (2 ohm), 0.5 A -> 1 V drop
    p = write(tmp_path, 'R0 n1_m9_0_0 n1_m1_0_0 4.0\nR1 n1_m9_0_0 n1_m1_0_0 4.0\nV0 n1_m9_0_0 0 1.1\nI0 n1_m1_0_0 0 0.5\n.op\n.end\n')
    net = spice.parse_netlist(p)
    res = spice.solve_dc(net)
    assert 1.1 - res.voltage[list(net.names).index('n1_m1_0_0')] == pytest.approx(1.0)


def test_zero_ohm_resistors_are_merged_and_floating_nodes_are_nan(tmp_path):
    txt = ('R0 n1_m9_0_0 n1_m1_0_0 1.0\nR1 n1_m1_0_0 n1_m1_2000_0 0\nV0 n1_m9_0_0 0 1.1\nI0 n1_m1_2000_0 0 0.1\n'
           'R2 n1_m1_8000_0 n1_m1_8000_2000 5.0\n.op\n.end\n')               # last resistor is an island
    p = write(tmp_path, txt)
    net = spice.parse_netlist(p)
    res = spice.solve_dc(net)
    names = list(net.names)
    assert res.voltage[names.index('n1_m1_2000_0')] == pytest.approx(1.0)
    assert res.voltage[names.index('n1_m1_0_0')] == pytest.approx(1.0)
    assert np.isnan(res.voltage[names.index('n1_m1_8000_0')])
    assert res.info['n_floating_merged_nodes'] == 2


def test_cg_matches_direct_on_a_grid(tmp_path):
    n = 12
    lines, k = [], 0
    for x in range(n):
        for y in range(n):
            if x + 1 < n:
                lines.append(f'R{k} n1_m1_{x * 2000}_{y * 2000} n1_m1_{(x + 1) * 2000}_{y * 2000} 1.5'); k += 1
            if y + 1 < n:
                lines.append(f'R{k} n1_m1_{x * 2000}_{y * 2000} n1_m1_{x * 2000}_{(y + 1) * 2000} 1.5'); k += 1
    lines.append('V0 n1_m1_0_0 0 1.1')
    lines.append('V1 n1_m1_22000_22000 0 1.1')
    for j in range(5):
        lines.append(f'I{j} n1_m1_{(3 + j) * 2000}_{(4 + j) * 2000} 0 1e-2')
    p = write(tmp_path, '\n'.join(lines) + '\n.op\n.end\n')
    net = spice.parse_netlist(p)
    a = spice.solve_dc(net, 'direct').voltage
    b = spice.solve_dc(net, 'cg', tol=1e-12).voltage
    assert np.allclose(a, b, atol=1e-9)
    # rasterise onto the 1 um grid: node spacing is 1 um, so 'nearest' reproduces node values exactly
    m = spice.rasterize_ir_drop(net, a, (n, n), vdd=1.1, method='nearest')
    assert m.shape == (n, n) and m.max() > 0 and m[0, 0] == pytest.approx(0.0)
    assert m[3, 4] == pytest.approx(1.1 - a[list(net.names).index('n1_m1_6000_8000')])


def test_axis0_is_x(tmp_path):
    p = write(tmp_path, 'R0 n1_m1_0_0 n1_m1_6000_0 1.0\nV0 n1_m1_0_0 0 1.1\nI0 n1_m1_6000_0 0 0.1\n.op\n.end\n')
    m, _ = spice.exact_ir_drop(p, (4, 2), interp='nearest')
    assert m[3, 0] == pytest.approx(0.1) and m[0, 0] == 0.0
