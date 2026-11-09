"""Build the edalint_pdn extension. Needs a C++17 compiler and Eigen 3 headers (EIGEN3_INCLUDE_DIR, default
/usr/include/eigen3). Install from this directory: pip install .  (or pip install -e . for development)."""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

eigen = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
if not os.path.isfile(os.path.join(eigen, "Eigen", "SparseCore")):
    raise SystemExit(f"Eigen 3 headers not found in {eigen}; set EIGEN3_INCLUDE_DIR")

extra = ["-O3", "-DNDEBUG"] if os.name != "nt" else ["/O2", "/DNDEBUG"]

setup(
    ext_modules=[
        Pybind11Extension(
            "edalint_pdn._pdn",
            ["src/pdn.cpp", "src/bindings.cpp"],
            include_dirs=["src", eigen],
            cxx_std=17,
            extra_compile_args=extra,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
