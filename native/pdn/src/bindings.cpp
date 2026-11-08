// pybind11 module edalint_pdn._pdn: parse_netlist(path) -> dict of arrays, solve_dc(n_nodes, arrays..., method, tol,
// max_iterations) -> (voltage, info). The Python package wraps these into edalint.spice.Netlist / SolveResult.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pdn.hpp"

namespace py = pybind11;

namespace {

template <class T>
py::array_t<T> to_array(std::vector<T>&& v) {
    auto* heap = new std::vector<T>(std::move(v));
    py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(heap->size())}, {sizeof(T)}, heap->data(), owner);
}

template <class T>
py::array_t<T, py::array::c_style | py::array::forcecast> as_array(const py::handle& h) {
    return py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(h);
}

pdn::Method parse_method(const std::string& m) {
    if (m == "direct") return pdn::Method::Direct;
    if (m == "cg_ic") return pdn::Method::CgIc;
    if (m == "cg") return pdn::Method::CgJacobi;
    throw std::invalid_argument("unknown method '" + m + "' (direct, cg_ic, cg)");
}

}  // namespace

PYBIND11_MODULE(_pdn, m) {
    m.doc() = "Native DC solver for ICCAD'23 power-delivery netlists (edalint.spice semantics)";

    m.def(
        "parse_netlist",
        [](const std::string& path) {
            pdn::Netlist net;
            {
                py::gil_scoped_release release;
                net = pdn::parse_netlist(path);
            }
            py::dict d;
            d["names"] = py::cast(net.names);
            d["layer"] = py::cast(net.layer);
            d["x_um"] = to_array(std::move(net.x_um));
            d["y_um"] = to_array(std::move(net.y_um));
            d["r_a"] = to_array(std::move(net.r_a));
            d["r_b"] = to_array(std::move(net.r_b));
            d["r_ohm"] = to_array(std::move(net.r_ohm));
            d["i_node"] = to_array(std::move(net.i_node));
            d["i_amp"] = to_array(std::move(net.i_amp));
            d["v_node"] = to_array(std::move(net.v_node));
            d["v_volt"] = to_array(std::move(net.v_volt));
            d["n_lines"] = net.n_lines;
            d["parse_seconds"] = net.parse_seconds;
            return d;
        },
        py::arg("path"));

    m.def(
        "solve_dc",
        [](int64_t n_nodes, py::handle r_a, py::handle r_b, py::handle r_ohm, py::handle i_node, py::handle i_amp,
           py::handle v_node, py::handle v_volt, const std::string& method, double tol, int64_t max_iterations) {
            auto ra = as_array<int64_t>(r_a), rb = as_array<int64_t>(r_b);
            auto ro = as_array<double>(r_ohm);
            auto in = as_array<int64_t>(i_node);
            auto ia = as_array<double>(i_amp);
            auto vn = as_array<int64_t>(v_node);
            auto vv = as_array<double>(v_volt);
            if (ra.size() != ro.size() || rb.size() != ro.size() || in.size() != ia.size() || vn.size() != vv.size())
                throw std::invalid_argument("array length mismatch");
            const pdn::Method meth = parse_method(method);
            pdn::SolveResult res;
            {
                py::gil_scoped_release release;
                res = pdn::solve_dc(n_nodes, ra.data(), rb.data(), ro.data(), ro.size(), in.data(), ia.data(),
                                    ia.size(), vn.data(), vv.data(), vv.size(), meth, tol, max_iterations);
            }
            py::dict info;
            info["method"] = method;
            info["n_nodes"] = res.info.n_nodes;
            info["n_free"] = res.info.n_free;
            info["n_floating_merged_nodes"] = res.info.n_floating_merged_nodes;
            info["iterations"] = res.info.iterations;
            info["seconds_assemble"] = res.info.seconds_assemble;
            info["seconds_prepare"] = res.info.seconds_prepare;
            info["seconds_solve"] = res.info.seconds_solve;
            info["rel_residual"] = res.info.rel_residual;
            return py::make_tuple(to_array(std::move(res.voltage)), info);
        },
        py::arg("n_nodes"), py::arg("r_a"), py::arg("r_b"), py::arg("r_ohm"), py::arg("i_node"), py::arg("i_amp"),
        py::arg("v_node"), py::arg("v_volt"), py::arg("method") = "direct", py::arg("tol") = 1e-12,
        py::arg("max_iterations") = 20000);
}
