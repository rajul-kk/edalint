# edalint_n14io_rs: the CircuitNet-N14 reader in Rust

A third implementation of `edalint.datasets.circuitnet.stream_tarball` for `edalint_n14io`, next to the C++ reader
and the pure-Python one, with the same results (same presence sets, same `handle` calls with equal arrays, same order).

| Piece | What it does |
|---|---|
| Reader thread | Inflates the gzip stream with `flate2` on the pure-Rust `zlib-rs` backend (no C library, no root), parses tar headers (ustar prefix, GNU long names, pax paths) |
| Decoder threads (default 2) | Pull the `data.npy` entry out of each wanted `.npz` (stored or deflated, zip64) and parse its npy header |
| Python side | `edalint_n14io.stream_tarball_rust(...)`; `Reader.next()` releases the GIL while it waits |

```bash
pip install maturin
pip install ./native/n14io-rs              # or, for development: cd native/n14io-rs && maturin develop --release
python -c "import edalint_n14io as n; print(n.HAVE_RUST)"
```

## Toolchain

| File | Purpose |
|---|---|
| `rust-toolchain.toml` | Pins Rust 1.99.1 with rustfmt and clippy; rustup installs it on first use, locally and in CI |
| `Cargo.toml` / `Cargo.lock` | `pyo3` 0.30 (stable ABI, one wheel per platform for Python 3.9+), `flate2` 1.1 with `zlib-rs`; release profile with fat LTO, one codegen unit |
| `pyproject.toml` | maturin build backend; module `edalint_n14io_rs._reader` |
| `rustfmt.toml` | 120-column lines, as in the rest of the repo |
| `.github/workflows/n14io.yml` | `cargo fmt --check`, `cargo clippy -D warnings`, then the parity tests on Linux, macOS and Windows |

## Speed (first 3000 maps of `IR_drop_features/RISCY.tar.gz`, WSL2, Python 3.14)

| Reader | Prefix copied to WSL's own disk | Drive E: (Windows drive) |
|---|---|---|
| edalint | 16.2 s | 58.0 s |
| python (ISA-L) | 5.6 s (2.9x) | 37.6 s (1.5x) |
| native C++ (zlib) | 5.5 s (3.0x) | 34.6 s (1.7x) |
| **rust (zlib-rs)** | **3.8 s (4.3x)** | **31.0 s (1.9x)** |

Best of 3 (local disk) and of 2 (drive E:), on a loaded machine; the E: run was slower for every reader than earlier
runs, so compare within a column. The gain over C++ is the faster inflate only: CircuitNet's `.npz` members are
*stored*, not deflated, so the decoder threads have little to do and their count barely matters (measured 1-8).
What is left is the single outer gzip stream, which no reader here can split across cores.

## State, made explicit

The C++ and Python readers track the end of the stream and the reader's lifecycle with flags. Here they are types:

| Type | States | Guarantees |
|---|---|---|
| `Fail` | `Truncated`, `Corrupt`, `Io` | A cut-off file is an `EOFError`, never a short result; `MultiGzDecoder` errors on a member with no end marker or trailer |
| `Header` | `EndOfArchive`, `LongName`, `GlobalPax`, `Entry` | Running out of data without the end-of-archive block is `Truncated`, not the end |
| `Msg` | `Member`, `End`, `Fail` | The reader thread ends with exactly one `End` or `Fail`; a closed queue without one is an error |
| `Slot` | `Present`, `Array` | Members come back in archive order even when decoders finish out of order |
| `State` | `Running`, `Done` | After `End` or a failure, `next()` returns `None` |

Dropping the `Reader` (Python stopped early, e.g. the handle raised) sets a stop flag and closes the queues, so the
reader thread quits at its next header and the decoder threads exit; tested by stopping 50 runs early with no
leftover OS threads.
