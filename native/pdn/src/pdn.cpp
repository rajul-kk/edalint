// Implementation of pdn.hpp. Semantics follow edalint.spice (parse_netlist, solve_dc) so results can be compared
// node by node; see native/pdn/README.md for what differs (node order only).
#include "pdn.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace pdn {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

std::string read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string buf;
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size > 0) {
        buf.resize(static_cast<size_t>(size));
        size_t got = std::fread(buf.data(), 1, buf.size(), f);
        buf.resize(got);
    }
    std::fclose(f);
    return buf;
}

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f'; }

// Whole-token number parse; false if the token is not entirely a number (pandas to_numeric / astype semantics).
bool parse_double(std::string_view s, double& out) {
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    if (s.empty()) return false;
    auto res = std::from_chars(s.data(), s.data() + s.size(), out);
    return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

bool is_ground(std::string_view s) { return s == "0" || s == "gnd" || s == "GND"; }

// Find-with-path-halving union-find over [0, n).
struct UnionFind {
    std::vector<int64_t> parent, size;
    explicit UnionFind(int64_t n) : parent(n), size(n, 1) {
        for (int64_t i = 0; i < n; ++i) parent[i] = i;
    }
    int64_t find(int64_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(int64_t a, int64_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
};

}  // namespace

Netlist parse_netlist(const std::string& path) {
    const auto t0 = Clock::now();
    const std::string buf = read_file(path);
    Netlist net;
    std::unordered_map<std::string_view, int64_t> index;
    index.reserve(buf.size() / 48 + 16);
    net.names.emplace_back("0");

    auto node_id = [&](std::string_view tok) -> int64_t {
        if (is_ground(tok)) return 0;
        auto it = index.find(tok);
        if (it != index.end()) return it->second;
        const int64_t id = static_cast<int64_t>(net.names.size());
        index.emplace(tok, id);  // tok points into buf, which outlives the map
        net.names.emplace_back(tok);
        return id;
    };

    size_t pos = 0;
    const size_t n = buf.size();
    std::string_view tok[4];
    while (pos < n) {
        size_t end = buf.find('\n', pos);
        if (end == std::string::npos) end = n;
        // split the line into at most four whitespace-separated fields (extra fields are ignored, as in edalint)
        int k = 0;
        size_t i = pos;
        while (i < end && k < 4) {
            while (i < end && is_space(buf[i])) ++i;
            if (i >= end) break;
            size_t j = i;
            while (j < end && !is_space(buf[j])) ++j;
            tok[k++] = std::string_view(buf.data() + i, j - i);
            i = j;
        }
        pos = end + 1;
        if (k < 4) continue;
        const char kind = static_cast<char>(std::toupper(static_cast<unsigned char>(tok[0][0])));
        if (kind != 'R' && kind != 'I' && kind != 'V') continue;
        double val;
        if (!parse_double(tok[3], val))
            throw std::runtime_error("non-numeric value '" + std::string(tok[3]) + "' in " + path);
        const int64_t a = node_id(tok[1]);
        const int64_t b = node_id(tok[2]);
        ++net.n_lines;
        if (kind == 'R') {
            net.r_a.push_back(a);
            net.r_b.push_back(b);
            net.r_ohm.push_back(val);
        } else {
            // "I n 0 v" is a sink at n; if ground is written first the sign flips (same for V)
            const int64_t node = a == 0 ? b : a;
            const double v = a == 0 ? -val : val;
            if (kind == 'I') {
                net.i_node.push_back(node);
                net.i_amp.push_back(v);
            } else {
                net.v_node.push_back(node);
                net.v_volt.push_back(v);
            }
        }
    }

    // layer and coordinates from n1_<layer>_<x>_<y>; like pandas str.split(expand=True), fields are only read if
    // some name has at least four of them
    const size_t nn = net.names.size();
    net.layer.assign(nn, "");
    net.x_um.assign(nn, std::numeric_limits<double>::quiet_NaN());
    net.y_um.assign(nn, std::numeric_limits<double>::quiet_NaN());
    bool any4 = false;
    for (const auto& s : net.names) {
        int us = 0;
        for (char c : s) us += c == '_';
        if (us >= 3) {
            any4 = true;
            break;
        }
    }
    if (any4) {
        for (size_t id = 0; id < nn; ++id) {
            std::string_view s = net.names[id];
            std::string_view parts[4];
            int np = 0;
            size_t start = 0;
            while (np < 4) {
                size_t u = s.find('_', start);
                parts[np++] = s.substr(start, u == std::string_view::npos ? std::string_view::npos : u - start);
                if (u == std::string_view::npos) break;
                start = u + 1;
            }
            if (np > 1) net.layer[id] = std::string(parts[1]);
            double v;
            if (np > 2 && parse_double(parts[2], v)) net.x_um[id] = v / kDbuPerUm;
            if (np > 3 && parse_double(parts[3], v)) net.y_um[id] = v / kDbuPerUm;
        }
    }
    net.parse_seconds = seconds_since(t0);
    return net;
}

SolveResult solve_dc(int64_t n, const int64_t* r_a, const int64_t* r_b, const double* r_ohm, int64_t n_r,
                     const int64_t* i_node, const double* i_amp, int64_t n_i, const int64_t* v_node,
                     const double* v_volt, int64_t n_v, Method method, double tol, int64_t max_iterations) {
    const auto t0 = Clock::now();
    auto check_id = [n](int64_t id) {
        if (id < 0 || id >= n) throw std::runtime_error("node id out of range");
    };

    // 1. merge zero-ohm resistors
    UnionFind zero(n);
    for (int64_t k = 0; k < n_r; ++k) {
        check_id(r_a[k]);
        check_id(r_b[k]);
        if (r_ohm[k] == 0.0) zero.unite(r_a[k], r_b[k]);
    }
    std::vector<int64_t> rep(n), compact(n, -1);
    int64_t N = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t root = zero.find(i);
        if (compact[root] < 0) compact[root] = N++;
        rep[i] = compact[root];
    }

    // 2. conductances between distinct merged nodes
    struct Edge {
        int64_t a, b;
        double g;
    };
    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(n_r));
    for (int64_t k = 0; k < n_r; ++k) {
        if (r_ohm[k] == 0.0) continue;
        const int64_t a = rep[r_a[k]], b = rep[r_b[k]];
        if (a != b) edges.push_back({a, b, 1.0 / r_ohm[k]});
    }

    // 3. Dirichlet nodes: ground and ideal supplies
    std::vector<char> fixed(N, 0);
    std::vector<double> vfix(N, 0.0);
    fixed[rep[0]] = 1;
    for (int64_t k = 0; k < n_v; ++k) {
        check_id(v_node[k]);
        const int64_t node = rep[v_node[k]];
        if (fixed[node] && std::abs(vfix[node] - v_volt[k]) > 1e-12)
            throw std::invalid_argument("conflicting ideal sources on one (merged) node");
        fixed[node] = 1;
        vfix[node] = v_volt[k];
    }
    std::vector<double> rhs_i(N, 0.0);
    for (int64_t k = 0; k < n_i; ++k) {
        check_id(i_node[k]);
        rhs_i[rep[i_node[k]]] -= i_amp[k];  // sink: current leaves the node
    }

    // 4. floating components: no conductive path to any fixed node
    UnionFind cc(N);
    for (const auto& e : edges) cc.unite(e.a, e.b);
    std::vector<char> comp_fixed(N, 0);
    for (int64_t r = 0; r < N; ++r)
        if (fixed[r]) comp_fixed[cc.find(r)] = 1;
    std::vector<int64_t> fidx(N, -1);
    std::vector<char> floating(N, 0);
    int64_t n_free = 0, n_floating = 0;
    for (int64_t r = 0; r < N; ++r) {
        floating[r] = !comp_fixed[cc.find(r)];
        n_floating += floating[r];
        if (!fixed[r] && !floating[r]) fidx[r] = n_free++;
    }
    const auto t_asm = Clock::now();

    // 5. reduced system G_ff x = i_f - G_fx v_x (assembled directly; duplicate entries are summed)
    using SpMat = Eigen::SparseMatrix<double>;
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(edges.size() * 4);
    Eigen::VectorXd rhs(n_free);
    for (int64_t r = 0; r < N; ++r)
        if (fidx[r] >= 0) rhs[fidx[r]] = rhs_i[r];
    for (const auto& e : edges) {
        const int64_t fa = fidx[e.a], fb = fidx[e.b];
        if (fa >= 0) trip.emplace_back(static_cast<int>(fa), static_cast<int>(fa), e.g);
        if (fb >= 0) trip.emplace_back(static_cast<int>(fb), static_cast<int>(fb), e.g);
        if (fa >= 0 && fb >= 0) {
            trip.emplace_back(static_cast<int>(fa), static_cast<int>(fb), -e.g);
            trip.emplace_back(static_cast<int>(fb), static_cast<int>(fa), -e.g);
        } else if (fa >= 0 && fixed[e.b]) {
            rhs[fa] += e.g * vfix[e.b];
        } else if (fb >= 0 && fixed[e.a]) {
            rhs[fb] += e.g * vfix[e.a];
        }
    }
    SpMat G(static_cast<Eigen::Index>(n_free), static_cast<Eigen::Index>(n_free));
    G.setFromTriplets(trip.begin(), trip.end());
    std::vector<Eigen::Triplet<double>>().swap(trip);
    const auto t_solve0 = Clock::now();

    Eigen::VectorXd x(n_free);
    int64_t iterations = 0;
    if (n_free > 0) {
        if (method == Method::Direct) {
            Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt(G);
            if (ldlt.info() != Eigen::Success) throw std::runtime_error("LDL^T factorisation failed");
            x = ldlt.solve(rhs);
        } else if (method == Method::CgIc) {
            Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                                     Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<int>>> cg;
            cg.setTolerance(tol);
            cg.setMaxIterations(static_cast<Eigen::Index>(max_iterations));
            cg.compute(G);
            if (cg.info() != Eigen::Success) throw std::runtime_error("incomplete Cholesky failed");
            x = cg.solve(rhs);
            iterations = static_cast<int64_t>(cg.iterations());
            if (cg.info() != Eigen::Success)
                throw std::runtime_error("CG did not converge in " + std::to_string(iterations) + " iterations");
        } else {
            Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg;
            cg.setTolerance(tol);
            cg.setMaxIterations(static_cast<Eigen::Index>(max_iterations));
            cg.compute(G);
            x = cg.solve(rhs);
            iterations = static_cast<int64_t>(cg.iterations());
            if (cg.info() != Eigen::Success)
                throw std::runtime_error("CG did not converge in " + std::to_string(iterations) + " iterations");
        }
    }
    const auto t_solve1 = Clock::now();

    SolveResult res;
    const double rhs_norm = n_free > 0 ? rhs.norm() : 0.0;
    res.info.rel_residual = n_free > 0 ? (G * x - rhs).norm() / std::max(rhs_norm, 1e-300) : 0.0;
    std::vector<double> v_rep(vfix);
    for (int64_t r = 0; r < N; ++r) {
        if (fidx[r] >= 0) v_rep[r] = x[fidx[r]];
        if (floating[r]) v_rep[r] = std::numeric_limits<double>::quiet_NaN();
    }
    res.voltage.resize(n);
    for (int64_t i = 0; i < n; ++i) res.voltage[i] = v_rep[rep[i]];
    res.info.n_nodes = n;
    res.info.n_free = n_free;
    res.info.n_floating_merged_nodes = n_floating;
    res.info.iterations = iterations;
    res.info.seconds_assemble = std::chrono::duration<double>(t_asm - t0).count();
    res.info.seconds_prepare = std::chrono::duration<double>(t_solve0 - t_asm).count();
    res.info.seconds_solve = std::chrono::duration<double>(t_solve1 - t_solve0).count();
    return res;
}

}  // namespace pdn
