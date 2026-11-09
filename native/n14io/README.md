# edalint_n14io: faster CircuitNet-N14 tarball streaming

A drop-in for `edalint.datasets.circuitnet.stream_tarball`, the function that streams CircuitNet-N14's per-design
`.tar.gz` files during staging. Three readers, all with edalint's results (same presence sets, same `handle` calls with
equal arrays, in the same order):

| Reader | What it is | Where it runs |
|---|---|---|
| **python** | A small tar header parser over one inflate source, replacing edalint's `GzipFile -> BufferedReader -> tarfile` stack. Inflate: **rapidgzip** (parallel, all cores), else Intel **ISA-L**, else **zlib** | Everywhere, including Windows |
| **native** (`_reader`) | C++17 + zlib, pybind11: inflate, tar, npz (stored, deflated, zip64) and npy parsing with the GIL released | Linux, WSL, macOS (needs a C++ compiler and zlib headers) |
| **rust** (`edalint_n14io_rs`) | Rust: zlib-rs inflate on a reader thread, npz decoding on a small pool; see `native/n14io-rs` | Linux, WSL, macOS; Windows through CI wheels |

`stream_tarball()` uses `default_impl()`: python on rapidgzip if installed, else rust, else native, else python on
ISA-L or zlib.

```python
import edalint_n14io
edalint_n14io.install()      # edalint's N14 loader now streams through the fastest installed reader
from edalint.datasets.circuitnet import N14Dataset
ds = N14Dataset("ir_drop", cache_dir="...")
```

`install()` swaps the function for the running process only; nothing under `src/edalint` changes, so `code_hash()`, the
lineage and the registered freezes are unaffected. `uninstall()` restores edalint's reader.

| Install | Command |
|---|---|
| Python reader + fast inflate | `pip install -e "native/n14io[fast]"` (rapidgzip, isal) with `EDALINT_N14IO_PURE=1` |
| + C++ reader | `pip install --no-build-isolation -e native/n14io` (C++17 compiler, zlib headers) |
| + Rust reader | `pip install ./native/n14io-rs` (Rust toolchain; maturin is fetched by pip) |

| Setting | Effect | Read |
|---|---|---|
| `EDALINT_N14IO_NO_RAPIDGZIP=1`, `EDALINT_N14IO_NO_ISAL=1` | Skip that inflate source when choosing `INFLATE` | Once, at import |
| `EDALINT_N14IO_THREADS=n` | rapidgzip threads (0 = all cores) | At every call |
| `stream_tarball_python(..., inflate="zlib")` | Force one source for a call | Per call |

## Speed (first 3000 maps of `IR_drop_features/RISCY.tar.gz`, 10 GB)

| Run | edalint | python + zlib | python + ISA-L | python + **rapidgzip** | native | rust |
|---|---|---|---|---|---|---|
| Windows, Python 3.10, drive E: | 30.5 s | 8.0 s (3.8x) | 5.7 s (5.4x) | **2.65 s (11.5x)** | not built | not built |
| WSL2, Python 3.14, 800 MB prefix on WSL's own disk | 14.1 s | 7.1 s (2.0x) | 4.8 s (2.9x) | **1.81 s (7.8x)** | 4.5 s (3.1x) | 3.4 s (4.2x) |
| WSL2, drive E: (earlier run) | 58.0 s | | 37.6 s (1.5x) | | 34.6 s (1.7x) | 31.0 s (1.9x) |

Best of 2 or 3, against edalint in the same run; the machine was loaded and absolute times move between runs (edalint
took 52.6 s on Windows in an earlier run), so compare within a row. Under WSL, drive E: (a Windows drive) is read at
about 90 MB/s compressed, which caps every reader; use WSL's own disk for the cache and source copies where possible.

Where the time goes: edalint spends most of it in the `GzipFile`/`BufferedReader`/`tarfile` layering (54 s of 71 s in
the first profile); all three readers remove that. What is left is the single gzip stream. CircuitNet's `.npz` members
are *stored* (not deflated), so there is no inner inflate to spread over threads. Only rapidgzip, which inflates one
gzip stream on many cores (1120 MB/s on 12 threads against zlib's 196 MB/s), changes this.

## State machines in the readers, and what each one guarantees

| State | Where | Guarantee | Test |
|---|---|---|---|
| End of data vs end of gzip stream | python on zlib/ISA-L (`_Inflated.read` checks `d.eof`), native (`finished_`), rust (`MultiGzDecoder`) | A file that runs out without the gzip end marker (interrupted download) raises `EOFError`, as edalint's `GzipFile` does, even when it stops exactly on a tar block | `test_unfinished_gzip_on_a_block_boundary_raises` |
| End of data with rapidgzip | python on rapidgzip (`_RapidSource.vouches_for_end = False`) | rapidgzip returns an unfinished stream's data, or nothing for a stream cut mid-way, without an error. So with rapidgzip, an archive is only complete if its tar end-of-archive block was read; otherwise `EOFError` | `test_rapidgzip_source_requires_the_end_of_archive_block` |
| Tar with no end-of-archive block, gzip finished | all | python on zlib/ISA-L and native accept it as edalint does; python on rapidgzip and rust refuse it (stricter on purpose). Real CircuitNet tarballs have the block (checked: whole `Vortex-small.tar.gz`, same 576 maps as ISA-L) | `test_verifying_sources_accept_a_tar_without_end_blocks_like_edalint` |
| Concatenated gzip members | `_Inflated` swaps in a new `decompressobj` on `unused_data`; zlib and ISA-L behave the same after a member ends | Multi-member files read fully | `test_dot_prefix_no_design_prefix_and_split_gzip` |
| Corrupt data | `_Inflated._fill`, `_RapidSource._read` | `gzip.BadGzipFile` (or `EOFError`), edalint's types, whichever inflate is used; native and rust raise `RuntimeError` | `test_corrupt_archive_raises_edalints_error_types` |
| GNU/pax long name pending | `long_name` in python, native and rust | Applies to the next entry only; a global pax record (`g`) does not consume it; a cut-off extended header raises | `test_matches_edalint[pax/gnu]` |
| Reader lifecycle | rust `State` (`Running`/`Done`) and stop flag | After the end or a failure nothing more is returned; a dropped Reader stops its threads (50 early stops, no leftover OS threads) | `test_rust_reader_stops_when_python_stops_early` |
| Source lifetime | python `try/finally: src.close()` | The file and rapidgzip's threads are released even when `handle` raises | the early-stop benchmark |
| Chosen reader and inflate | `INFLATE`, `HAVE_*` (import time), `default_impl()` (per call) | `NO_*` variables only matter before import | `test_default_reader_order` |
| `install()` / `uninstall()` | module global `_original` | Calling `install()` twice keeps edalint's original; `uninstall()` puts it back even if something else patched the function in between | `test_install_swaps_edalints_reader_and_uninstall_restores_it` |

Shared by every reader, as in edalint: reading stops at the tar end-of-archive block, so bytes after it (including the
gzip trailer) are not checked; a tarball cut after that block is accepted, and its content is complete. Members handled
before a later error have already been passed to `handle`; staging writes no presence JSON for a pass that raised.

## State this does not change (edalint's staging, for anyone parallelising it)

- `N14Dataset._stage_one` mutates `splits`, `skipped`, `inventory` and `_staged` without locks, so staging designs from
  several threads is not safe; this package only speeds up the per-tarball read.
- `DirStore.put` writes each staged `.npy` in place: a crash mid-write leaves a partial file that `has()` reports as
  present and the next run trusts. The presence JSON, written after a full pass, marks a tarball as scanned.
- Cached maps are keyed by (design, mode, feature, name) only, so a change to `stage_member` does not invalidate a
  `cache_dir`. These live in `src/edalint` (hashed) and are left as they are.
