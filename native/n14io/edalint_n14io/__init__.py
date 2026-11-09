"""Faster CircuitNet-N14 tarball streaming, as a drop-in for edalint.datasets.circuitnet.stream_tarball.

Three implementations with the semantics of edalint's stream_tarball (one sequential pass; presence of every matching
member; handle(feature, name, array) for the wanted ones, in archive order):
  python  pure Python: a hand-written tar header parser over one inflate source (rapidgzip, ISA-L or zlib) instead of
          the GzipFile -> BufferedReader -> tarfile stack, whose per-member overhead dominates edalint's loader.
  native  C++ (_reader): zlib inflate, tar, zip and npy parsing with the GIL released. Linux / WSL / macOS builds.
  rust    Rust (edalint_n14io_rs, native/n14io-rs): zlib-rs inflate on a reader thread, npz decoding on a pool.
stream_tarball() uses default_impl(), the fastest one installed.

install() points edalint.datasets.circuitnet.stream_tarball at this module's stream_tarball for the running process;
uninstall() restores it. Nothing under src/edalint changes, so edalint.protocol.code_hash() and the freezes are unaffected.
"""
from __future__ import annotations

import gzip
import io
import os
import posixpath
import zlib
from typing import Callable, Dict, Optional

import numpy as np

try:
    from . import _reader
except ImportError:  # extension not built (e.g. Windows): the pure-Python path is used
    _reader = None

try:  # the Rust reader (native/n14io-rs), a separate package
    from edalint_n14io_rs import Reader as _RustReader
except ImportError:
    _RustReader = None

# Inflate for the pure-Python reader, fastest available first (each can be turned off for testing or by choice):
#   rapidgzip  parallel gzip decompression over all cores (pip install rapidgzip); EDALINT_N14IO_NO_RAPIDGZIP=1 skips it
#   isal       Intel ISA-L, decompressobj API, about 1.9x zlib (pip install isal); EDALINT_N14IO_NO_ISAL=1 skips it
#   zlib       always there
# rapidgzip does not report an unfinished gzip stream (it returns the data, or nothing, without an error), so with it
# the reader only accepts an archive that reaches its tar end-of-archive block; see _RapidSource.
try:
    from isal import isal_zlib as _isal
except ImportError:
    _isal = None
try:
    import rapidgzip as _rapidgzip
except ImportError:
    _rapidgzip = None

__all__ = ["stream_tarball", "stream_tarball_native", "stream_tarball_python", "stream_tarball_rust", "install",
           "uninstall", "default_impl", "HAVE_NATIVE", "HAVE_RUST", "INFLATE", "INFLATES"]

HAVE_NATIVE = _reader is not None
HAVE_RUST = _RustReader is not None
INFLATES = ([] if _rapidgzip is None else ["rapidgzip"]) + ([] if _isal is None else ["isal"]) + ["zlib"]
INFLATE = next(i for i in INFLATES if not os.environ.get(f"EDALINT_N14IO_NO_{i.upper()}") or i == "zlib")
_CHUNK = 1 << 24


def stream_tarball_native(tar_path, design: str, dir_to_feature: Dict[str, str], wanted: Optional[set],
                          handle: Callable) -> Dict[str, set]:
    if _reader is None:
        raise ImportError("edalint_n14io._reader is not built")
    presence: Dict[str, set] = {f: set() for f in dir_to_feature.values()}
    r = _reader.Reader(str(tar_path), design, dict(dir_to_feature), None if wanted is None else list(wanted))
    while True:
        m = r.next()
        if m is None:
            return presence
        feat, name, arr = m
        presence[feat].add(name)
        if arr is not None:
            handle(feat, name, arr)


def stream_tarball_rust(tar_path, design: str, dir_to_feature: Dict[str, str], wanted: Optional[set],
                        handle: Callable, workers: int = 0) -> Dict[str, set]:
    """The Rust reader: zlib-rs inflate on a reader thread, the wanted npz members decoded by `workers` threads
    (0: two, enough for stored npz members; raise it for deflated ones), results in archive order."""
    if _RustReader is None:
        raise ImportError("edalint_n14io_rs is not installed (native/n14io-rs)")
    presence: Dict[str, set] = {f: set() for f in dir_to_feature.values()}
    r = _RustReader(str(tar_path), design, dict(dir_to_feature), None if wanted is None else list(wanted), workers)
    while True:
        m = r.next()
        if m is None:
            return presence
        feat, name, arr = m
        presence[feat].add(name)
        if arr is not None:
            descr, fortran, shape, data = arr
            a = np.frombuffer(data, dtype=np.dtype(descr)).reshape(shape, order="F" if fortran else "C")
            handle(feat, name, a)


class _Inflated:
    """Exact-size reads from a gzip file through one decompressobj (zlib or ISA-L; handles concatenated members).
    It knows whether the last gzip member reached its end marker, so a clean end from read() is a real end."""

    vouches_for_end = True

    def __init__(self, path, mod=zlib):
        self._mod = mod
        self.f = open(path, "rb")
        self.d = mod.decompressobj(16 + zlib.MAX_WBITS)
        self.buf = bytearray()
        self.pos = 0
        self.eof = False

    def close(self):
        self.f.close()

    def _fill(self, need):
        try:
            self._fill_raw(need)
        except self._mod.error as e:   # zlib.error / IsalError: report corrupt data the way edalint's GzipFile does
            raise gzip.BadGzipFile(f"corrupt gzip data: {e}") from e

    def _fill_raw(self, need):
        while len(self.buf) - self.pos < need and not self.eof:
            if self.pos > _CHUNK:                     # drop consumed bytes now and then
                del self.buf[:self.pos]
                self.pos = 0
            raw = self.f.read(_CHUNK)
            if not raw:
                self.buf += self.d.flush()
                self.eof = True
                break
            out = self.d.decompress(raw)
            while self.d.eof and self.d.unused_data:  # next gzip member
                rest = self.d.unused_data
                self.d = self._mod.decompressobj(16 + zlib.MAX_WBITS)
                out += self.d.decompress(rest)
            self.buf += out

    def read(self, n):
        self._fill(n)
        if len(self.buf) - self.pos < n:
            if len(self.buf) == self.pos:
                # Out of input exactly on a request boundary is only a clean end if the last gzip member finished
                # (d.eof); otherwise the file was cut off (e.g. an interrupted download), which edalint's GzipFile
                # reports. Returning None here would let the caller record a partial tarball as fully scanned.
                if not self.d.eof:
                    raise EOFError("gzip stream ended before its end-of-stream marker")
                return None
            raise EOFError("truncated gzip stream")
        b = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return b

    def skip(self, n):
        while n > 0:
            self._fill(min(n, _CHUNK))
            k = min(n, len(self.buf) - self.pos)
            if k == 0:
                raise EOFError("truncated gzip stream")
            self.pos += k
            n -= k


class _RapidSource:
    """Exact-size reads through rapidgzip (parallel inflate). rapidgzip returns an unfinished gzip stream's data, or
    nothing for a stream cut mid-way, without an error, so a clean end from read() proves nothing here
    (vouches_for_end = False): the caller must have seen the tar end-of-archive block."""

    vouches_for_end = False

    def __init__(self, path, threads=0):
        self.f = _rapidgzip.open(os.fspath(path), parallelization=threads)

    def close(self):
        self.f.close()

    def _read(self, n):
        try:
            return self.f.read(n)
        except (RuntimeError, ValueError) as e:  # rapidgzip's CRC, size and end-of-file errors, as edalint's types
            if "End of file" in str(e):
                raise EOFError(f"truncated gzip stream: {e}") from e
            raise gzip.BadGzipFile(f"corrupt gzip data: {e}") from e

    def read(self, n):
        parts, got = [], 0
        while got < n:
            b = self._read(n - got)
            if not b:
                break
            parts.append(b)
            got += len(b)
        if got == 0:
            return None
        if got < n:
            raise EOFError("truncated gzip stream")
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def skip(self, n):
        while n > 0:
            b = self._read(min(n, _CHUNK))
            if not b:
                raise EOFError("truncated gzip stream")
            n -= len(b)


def _open_source(path, inflate):
    if inflate == "rapidgzip":
        if _rapidgzip is None:
            raise ImportError("rapidgzip is not installed")
        return _RapidSource(path, int(os.environ.get("EDALINT_N14IO_THREADS", "0")))
    if inflate == "isal":
        if _isal is None:
            raise ImportError("isal is not installed")
        return _Inflated(path, _isal)
    if inflate == "zlib":
        return _Inflated(path, zlib)
    raise ValueError(f"unknown inflate {inflate!r}; one of {INFLATES}")


def _tar_number(b: bytes) -> int:
    if b[0] & 0x80:
        v = b[0] & 0x7F
        for c in b[1:]:
            v = (v << 8) | c
        return v
    s = b.replace(b"\0", b" ").strip()
    return int(s, 8) if s else 0


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


def _pax_path(rec: bytes) -> Optional[str]:
    path, i = None, 0
    while i < len(rec):
        sp = rec.find(b" ", i)
        if sp < 0:
            break
        n = int(rec[i:sp])
        if n <= 0:
            break
        kv = rec[sp + 1:i + n - 1]
        if kv.startswith(b"path="):
            path = kv[5:].decode("utf-8", "surrogateescape")
        i += n
    return path


def stream_tarball_python(tar_path, design: str, dir_to_feature: Dict[str, str], wanted: Optional[set],
                          handle: Callable, inflate: Optional[str] = None) -> Dict[str, set]:
    """inflate: "rapidgzip", "isal" or "zlib" (default INFLATE, the fastest installed)."""
    presence: Dict[str, set] = {f: set() for f in dir_to_feature.values()}
    src = _open_source(tar_path, inflate or INFLATE)
    try:
        long_name = None
        while True:
            hdr = src.read(512)
            if hdr is None:
                # Out of data at a header boundary with no end-of-archive block. edalint's tarfile accepts that, and so
                # do we when the source has verified the gzip end marker; rapidgzip cannot, so it is an error there.
                if not src.vouches_for_end:
                    raise EOFError("archive ended without its tar end-of-archive block (possibly truncated)")
                return presence
            if hdr.count(0) == 512:
                return presence
            size = _tar_number(hdr[124:136])
            padded = (size + 511) // 512 * 512
            kind = hdr[156:157]
            if kind in (b"L", b"x"):
                data = src.read(size) if size else b""
                if data is None:
                    raise EOFError("truncated tar extended header")
                src.skip(padded - size)
                long_name = (_cstr(data) if kind == b"L" else _pax_path(data)) or long_name
                continue
            if kind == b"g":
                src.skip(padded)
                continue
            name = long_name
            long_name = None
            if not name:
                name = _cstr(hdr[0:100])
                if hdr[257:262] == b"ustar":
                    prefix = _cstr(hdr[345:500])
                    if prefix:
                        name = prefix + "/" + name
            feat = None
            if kind in (b"0", b"\0", b"7") and name.endswith(".npz"):
                d, fn = posixpath.split(name)
                d = d[2:] if d.startswith("./") else d
                if d.startswith(design + "/"):
                    d = d[len(design) + 1:]
                feat = dir_to_feature.get(d)
            if feat is None:
                src.skip(padded)
                continue
            mname = fn[:-4]
            presence[feat].add(mname)
            if wanted is None or mname in wanted:
                data = src.read(size)
                if data is None:
                    raise EOFError(f"truncated tar member {name}")
                with np.load(io.BytesIO(data)) as z:
                    handle(feat, mname, z["data"])
                src.skip(padded - size)
            else:
                src.skip(padded)
    finally:
        src.close()


def default_impl() -> str:
    """The fastest reader available, by the benchmarks in README.md: the Python reader on rapidgzip (parallel inflate),
    else Rust, else C++, else the Python reader on ISA-L or zlib. Decided per call from module state, which is fixed
    at import (EDALINT_N14IO_NO_* are read once)."""
    if INFLATE == "rapidgzip":
        return "python"
    return "rust" if HAVE_RUST else "native" if HAVE_NATIVE else "python"


_IMPLS = {"python": stream_tarball_python, "rust": stream_tarball_rust, "native": stream_tarball_native}


def stream_tarball(tar_path, design, dir_to_feature, wanted, handle):
    """edalint.datasets.circuitnet.stream_tarball, faster: the reader default_impl() names."""
    return _IMPLS[default_impl()](tar_path, design, dir_to_feature, wanted, handle)


_original = None


def install():
    """Use stream_tarball for edalint's N14 loader in this process. Returns the implementation name."""
    global _original
    from edalint.datasets import circuitnet
    if _original is None:
        _original = circuitnet.stream_tarball
    circuitnet.stream_tarball = stream_tarball
    return default_impl()


def uninstall():
    global _original
    from edalint.datasets import circuitnet
    if _original is not None:
        circuitnet.stream_tarball = _original
        _original = None
