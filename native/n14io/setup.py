"""Build edalint_n14io (C++17, zlib). On a system without a usable compiler, set EDALINT_N14IO_PURE=1 to install the
pure-Python reader only: pip install .  (or pip install -e . for development)."""
import os

from setuptools import setup

ext_modules, cmdclass = [], {}
if not os.environ.get("EDALINT_N14IO_PURE"):
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    extra = ["-O3", "-DNDEBUG"] if os.name != "nt" else ["/O2", "/DNDEBUG"]
    ext_modules = [Pybind11Extension("edalint_n14io._reader", ["src/reader.cpp"], cxx_std=17,
                                     extra_compile_args=extra, libraries=["z"])]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
