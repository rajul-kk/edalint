// Native DC solver for ICCAD'23 Problem C power-delivery netlists (edalint.spice semantics, in C++).
//
// Netlist lines: R<k> a b ohm | I<k> node 0 amp (sink) | V<k> node 0 volt (ideal supply); ground is 0 / gnd / GND.
// Solve: zero-ohm resistors merged (union-find), supplies and ground as Dirichlet nodes, nodes with no path to a
// fixed node marked floating (NaN), the reduced SPD system G_ff v_f = i_f - G_fx v_x solved with sparse LDL^T or
// preconditioned conjugate gradient.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pdn {

constexpr double kDbuPerUm = 2000.0;

struct Netlist {
    std::vector<std::string> names;   // index 0 is the ground node "0"
    std::vector<std::string> layer;   // per node: second '_' field of the name, "" if absent (see parse_netlist)
    std::vector<double> x_um, y_um;   // third and fourth fields / 2000, NaN if absent or not numeric
    std::vector<int64_t> r_a, r_b;
    std::vector<double> r_ohm;
    std::vector<int64_t> i_node;
    std::vector<double> i_amp;        // positive = sink (current drawn from the node)
    std::vector<int64_t> v_node;
    std::vector<double> v_volt;
    int64_t n_lines = 0;              // R, I and V lines read
    double parse_seconds = 0.0;
};

// Throws std::runtime_error on an unreadable file or a non-numeric value.
Netlist parse_netlist(const std::string& path);

enum class Method { Direct, CgIc, CgJacobi };

struct SolveInfo {
    int64_t n_nodes = 0, n_free = 0, n_floating_merged_nodes = 0, iterations = 0;
    double seconds_assemble = 0, seconds_prepare = 0, seconds_solve = 0, rel_residual = 0;
};

struct SolveResult {
    std::vector<double> voltage;      // per netlist node; NaN for floating nodes
    SolveInfo info;
};

// Arrays as in Netlist (node ids in [0, n_nodes), node 0 is ground). Throws std::invalid_argument on conflicting ideal
// sources on one merged node and std::runtime_error on a non-converging CG or a failed factorisation.
SolveResult solve_dc(int64_t n_nodes,
                     const int64_t* r_a, const int64_t* r_b, const double* r_ohm, int64_t n_r,
                     const int64_t* i_node, const double* i_amp, int64_t n_i,
                     const int64_t* v_node, const double* v_volt, int64_t n_v,
                     Method method, double tol, int64_t max_iterations);

inline SolveResult solve_dc(const Netlist& net, Method method = Method::Direct, double tol = 1e-12,
                            int64_t max_iterations = 20000) {
    return solve_dc(static_cast<int64_t>(net.names.size()), net.r_a.data(), net.r_b.data(), net.r_ohm.data(),
                    static_cast<int64_t>(net.r_ohm.size()), net.i_node.data(), net.i_amp.data(),
                    static_cast<int64_t>(net.i_amp.size()), net.v_node.data(), net.v_volt.data(),
                    static_cast<int64_t>(net.v_volt.size()), method, tol, max_iterations);
}

}  // namespace pdn
