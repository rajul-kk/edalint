# edalint_n14io: faster CircuitNet-N14 tarball streaming

A drop-in for `edalint.datasets.circuitnet.stream_tarball`, the function that streams CircuitNet-N14's per-design
`.tar.gz` files during staging, in two implementations with identical results:

- **native** (`_reader`, C++17 + zlib, pybind11): inflate, tar parsing (ustar prefixes, GNU long names, pax paths),
  npz (zip, stored or deflated, zip64) and npy parsing (C or Fortran order) with the GIL released.
- **python**: one `zlib.decompressobj` over 16 MB chunks and a small tar header parser, replacing edalint's
  `GzipFile -> BufferedReader -> tarfile` stack. Works everywhere, including Windows, where this repo has no compiler for
  the extension.

```python
import edalint_n14io
edalint_n14io.install()      # edalint's N14 loader now streams through edalint_n14io (native if built, else python)
from edalint.datasets.circuitnet import N14Dataset
ds = N14Dataset("ir_drop", cache_dir="...")
```

`install()` swaps the function for the running process only; nothing under `src/edalint` changes, so `code_hash()`, the
lineage and the registered freezes are unaffected. `uninstall()` restores edalint's reader.

Build: `pip install --no-build-isolation -e native/n14io` (C++17 compiler, zlib headers); `EDALINT_N14IO_PURE=1` installs
the pure-Python reader only.

## Correctness

`tests/test_n14io.py` runs edalint's `stream_tarball` and both readers on synthetic tarballs (GNU, pax and ustar formats;
`./`, design and no prefix; concatenated gzip members; stored and compressed npz; float32, float64, int16; 2-D, 3-D and
Fortran-order arrays; non-npz members and unmapped directories; `wanted` = all, a subset, nothing) and requires the
same presence sets and the same `handle` calls with equal arrays, in the same order; truncated archives must raise; and,
when the data is on disk, the first 300 maps of the smallest real IR-drop tarball must be identical across all three.

## Speed (first 3000 maps of `IR_drop_features/RISCY.tar.gz`, 10 GB, on drive E:, best of 2)

| | edalint | python (zlib) | python (ISA-L) | native (zlib) |
|---|---|---|---|---|
| Windows, Python 3.10 | 52.6-64.1 s | 15.3 s (4.2x) | **9.6 s (5.5x)** | not built |
| WSL2, Python 3.14 | 46.2-54.2 s | 23.3 s (2.0x) | 19.3 s (2.8x) | 18.7-22.1 s (2.1-2.9x) |

With the Rust reader (`native/n14io-rs`, see its README), on a 800 MB prefix copied to WSL's own disk so the
Windows-drive cap does not hide the readers: edalint 16.2 s, python (ISA-L) 5.6 s, native 5.5 s, **rust 3.8 s (4.3x)**.

Speed-ups are against edalint's reader in the same run (two runs per platform; the ranges are run-to-run spread on a
loaded machine). `pip install isal` (or the `fast` extra) makes the pure-Python reader use Intel ISA-L for inflate
(344 MB/s against zlib's 180 MB/s on this tarball, Windows); `EDALINT_N14IO_NO_ISAL=1` turns it off. The C++ reader
still uses zlib: ISA-L's C headers are not installable here without root, and under WSL both readers are already
capped by reading the Windows drive.

Where the time went (600 MB compressed, Windows): raw read 0.6 s, zlib inflate 10.8 s, `np.load` 5.3 s, and 54 s in
the `GzipFile`/`BufferedReader`/`tarfile` layering. Both new readers remove that layer; what is left is inflate
(single-threaded zlib, or ISA-L in the pure-Python reader) and, under WSL, reading the Windows drive (about 90 MB/s compressed). That
is why the C++ reader is barely faster than the pure-Python one: the language was not the bottleneck. Further gains
need a faster inflate (zlib-ng is typically 1.5-2x) or a different storage format, not a rewrite.

## End-of-stream state in the readers

Both readers keep "input exhausted" separate from "gzip member finished": a file that runs out on a tar block boundary
without its gzip end marker (an interrupted download) raises, as edalint's `GzipFile` does, instead of being returned
as a complete archive. A finished `decompressobj` that is fed more input passes it to `unused_data`, which starts the
next member; zlib and ISA-L behave the same here (checked). `tests/test_n14io.py::test_unfinished_gzip_on_a_block_boundary_raises`
covers it.

## State this does not change (edalint's staging, for anyone parallelising it)

- `N14Dataset._stage_one` mutates `splits`, `skipped`, `inventory` and `_staged` without locks, so staging designs from
  several threads is not safe; this package only speeds up the per-tarball read.
- `DirStore.put` writes each staged `.npy` in place: a crash mid-write leaves a partial file that `has()` reports as
  present and the next run trusts. The presence JSON, written after a full pass, marks a tarball as scanned.
- Cached maps are keyed by (design, mode, feature, name) only, so a change to `stage_member` does not invalidate a
  `cache_dir`. These live in `src/edalint` (hashed) and are left as they are.
