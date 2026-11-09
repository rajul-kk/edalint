"""Both readers against edalint's own stream_tarball, on synthetic tarballs that exercise the formats, and on a real
CircuitNet-N14 tarball when one is on disk."""
import gzip
import io
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

import edalint_n14io as n14
from edalint.datasets import circuitnet

IMPLS = ([n14.stream_tarball_python] + ([n14.stream_tarball_native] if n14.HAVE_NATIVE else [])
         + ([n14.stream_tarball_rust] if n14.HAVE_RUST else []))
DIRS = {"power_i": "power_i", "RUDY/RUDY": "RUDY", "congestion/a/deep/path/that/is/long/" + "x" * 90: "LONGDIR"}


def npz_bytes(arr, compressed):
    b = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(b, data=arr)
    return b.getvalue()


def build_tar(tmp_path, fmt=tarfile.GNU_FORMAT, prefix="des/", split_gzip=False, name="t.tar.gz"):
    rng = np.random.default_rng(0)
    entries = []
    for i in range(6):
        entries.append((f"{prefix}power_i/s{i}.npz", npz_bytes(rng.random((4, 5)).astype(np.float32), i % 2 == 0)))
    entries.append((f"{prefix}RUDY/RUDY/s0.npz", npz_bytes(np.asfortranarray(rng.random((3, 7))), True)))
    entries.append((f"{prefix}RUDY/RUDY/s1.npz", npz_bytes(rng.integers(0, 9, (2, 3, 4)).astype(np.int16), False)))
    entries.append((f"{prefix}RUDY/RUDY/notes.txt", b"not an npz"))
    entries.append((f"{prefix}other/s0.npz", npz_bytes(np.zeros(3), False)))
    entries.append((f"{prefix}{list(DIRS)[2]}/s2.npz", npz_bytes(np.arange(6.0).reshape(2, 3), True)))
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=fmt) as tf:
        info = tarfile.TarInfo(prefix + "power_i")
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
        for n, data in entries:
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    data = raw.getvalue()
    p = tmp_path / name
    if split_gzip:                                   # two concatenated gzip members
        cut = len(data) // 2 + 123
        p.write_bytes(gzip.compress(data[:cut]) + gzip.compress(data[cut:]))
    else:
        p.write_bytes(gzip.compress(data))
    return p


def run(impl, path, wanted, design="des"):
    calls = []
    pres = impl(path, design, DIRS, wanted, lambda f, n, a: calls.append((f, n, np.array(a), a.dtype, a.flags.f_contiguous)))
    return pres, calls


def same(a, b):
    pa, ca = a
    pb, cb = b
    assert pa == pb
    assert [(f, n) for f, n, *_ in ca] == [(f, n) for f, n, *_ in cb]
    for (_, _, x, dx, fx), (_, _, y, dy, fy) in zip(ca, cb):
        assert dx == dy and x.shape == y.shape and np.array_equal(x, y)


@pytest.mark.parametrize("impl", IMPLS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("fmt", [tarfile.GNU_FORMAT, tarfile.PAX_FORMAT, tarfile.USTAR_FORMAT], ids=["gnu", "pax", "ustar"])
@pytest.mark.parametrize("wanted", [None, {"s0", "s2", "s5"}, set()], ids=["all", "subset", "none"])
def test_matches_edalint(tmp_path, impl, fmt, wanted):
    p = build_tar(tmp_path, fmt)                     # ustar: the long directory goes through the prefix field
    ref = run(circuitnet.stream_tarball, p, wanted)
    got = run(impl, p, wanted)
    same(ref, got)
    assert ref[0]["power_i"] == {f"s{i}" for i in range(6)}
    if wanted is None:
        assert len(ref[1]) == 9                      # 6 power_i + 2 RUDY + 1 long directory


@pytest.mark.parametrize("impl", IMPLS, ids=lambda f: f.__name__)
def test_dot_prefix_no_design_prefix_and_split_gzip(tmp_path, impl):
    for prefix in ("./des/", "", "./"):
        p = build_tar(tmp_path, prefix=prefix, split_gzip=True, name=f"t{len(prefix)}.tar.gz")
        same(run(circuitnet.stream_tarball, p, None), run(impl, p, None))


@pytest.mark.parametrize("impl", IMPLS, ids=lambda f: f.__name__)
def test_fortran_order_and_dtypes_survive(tmp_path, impl):
    _, calls = run(impl, build_tar(tmp_path), {"s0", "s1"})
    by = {(f, n): (x, dt, fo) for f, n, x, dt, fo in calls}
    x, dt, fo = by[("RUDY", "s0")]
    assert dt == np.float64 and fo and x.shape == (3, 7)
    x, dt, _ = by[("RUDY", "s1")]
    assert dt == np.int16 and x.shape == (2, 3, 4)


@pytest.mark.parametrize("impl", IMPLS, ids=lambda f: f.__name__)
def test_truncated_archive_raises(tmp_path, impl):
    p = build_tar(tmp_path)
    data = gzip.decompress(p.read_bytes())
    cut = tmp_path / "cut.tar.gz"
    cut.write_bytes(gzip.compress(data[: len(data) // 2])[:-8])     # truncated gzip of a truncated tar
    with pytest.raises((EOFError, RuntimeError, OSError, zlib_error())):
        run(impl, cut, None)


@pytest.mark.parametrize("impl", IMPLS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("keep", ["members", "headers"])
def test_unfinished_gzip_on_a_block_boundary_raises(tmp_path, impl, keep):
    """A gzip stream that stops (no end marker, no trailer) where the inflated tar happens to end on a 512-byte block:
    an interrupted download. edalint's GzipFile raises; a reader that treats running out of input as end of archive
    would return a presence set and the staging code would record the tarball as scanned."""
    import zlib
    p = build_tar(tmp_path)
    data = gzip.decompress(p.read_bytes())
    end = len(data) - 1024 if keep == "members" else 512 * 3        # drop the end-of-archive blocks / most members
    while data[end - 512:end].count(0) == 512:
        end -= 512
    c = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    cut = tmp_path / "unfinished.tar.gz"
    cut.write_bytes(c.compress(data[:end]) + c.flush(zlib.Z_SYNC_FLUSH))  # everything inflates, but the stream never ends
    with pytest.raises((EOFError, RuntimeError, OSError, zlib_error())):
        run(circuitnet.stream_tarball, cut, None)                   # the reference behaviour
    with pytest.raises((EOFError, RuntimeError, OSError, zlib_error())):
        run(impl, cut, None)


def zlib_error():
    import zlib
    return zlib.error


def test_install_swaps_edalints_reader_and_uninstall_restores_it():
    original = circuitnet.stream_tarball
    try:
        assert n14.install() in ("native", "python")
        assert circuitnet.stream_tarball is n14.stream_tarball
    finally:
        n14.uninstall()
    assert circuitnet.stream_tarball is original


N14 = Path(os.environ.get("EDALINT_N14", "/mnt/e/edalint_data/circuitnet_n14")) / "CircuitNet-N14"


class _Enough(Exception):
    pass


def first_members(impl, path, design, dirs, n):
    """(feature, name) -> array for the first n wanted members (the loader stops by raising from handle)."""
    out = {}

    def handle(f, name, a):
        out[(f, name)] = np.array(a)
        if len(out) >= n:
            raise _Enough
    try:
        impl(path, design, dirs, None, handle)
    except _Enough:
        pass
    return out


@pytest.mark.realdata
@pytest.mark.skipif(not N14.is_dir(), reason="CircuitNet-N14 not on disk")
@pytest.mark.parametrize("impl", IMPLS, ids=lambda f: f.__name__)
def test_real_tarball_matches_edalint(impl):
    p = sorted((N14 / "IR_drop_features").glob("*.tar.gz"), key=lambda q: q.stat().st_size)[0]
    design = p.name[: -len(".tar.gz")]
    dirs = {circuitnet.FEATURE_DIRS[f]: f for f in circuitnet.TASKS["ir_drop"]["inputs"] + ["IR_drop"]}
    ref = first_members(circuitnet.stream_tarball, p, design, dirs, 300)
    got = first_members(impl, p, design, dirs, 300)
    assert list(ref) == list(got) and len(ref) == 300
    assert all(ref[k].dtype == got[k].dtype and np.array_equal(ref[k], got[k]) for k in ref)


@pytest.mark.skipif(not n14.HAVE_RUST, reason="edalint_n14io_rs not installed")
@pytest.mark.parametrize("workers", [1, 2, 7])
def test_rust_order_does_not_depend_on_worker_count(tmp_path, workers):
    p = build_tar(tmp_path)
    ref = run(circuitnet.stream_tarball, p, None)
    got = run(lambda *a: n14.stream_tarball_rust(*a, workers=workers), p, None)
    same(ref, got)


@pytest.mark.skipif(not n14.HAVE_RUST, reason="edalint_n14io_rs not installed")
def test_rust_reader_stops_when_python_stops_early(tmp_path):
    """The handle raising mid-archive must propagate, and the reader and decoder threads must wind down (Reader
    dropped -> stop flag and closed queues), so a second pass over the same file starts clean."""
    import threading
    p = build_tar(tmp_path)
    before = threading.active_count()

    class Stop(Exception):
        pass

    def handle(f, n, a):
        raise Stop

    for _ in range(20):
        with pytest.raises(Stop):
            n14.stream_tarball_rust(p, "des", DIRS, None, handle)
    same(run(circuitnet.stream_tarball, p, None), run(n14.stream_tarball_rust, p, None))
    assert threading.active_count() == before        # Rust threads are not Python threads; nothing leaked into Python
