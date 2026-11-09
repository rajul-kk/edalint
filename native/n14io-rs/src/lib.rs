//! CircuitNet-N14 tarball streaming for edalint_n14io, in Rust.
//!
//! One reader thread inflates the gzip stream (pure-Rust zlib-rs) and parses tar headers; the `.npz` members that are
//! wanted go to a pool of decoder threads (zip + inner deflate + npy header), so the per-map inflate runs in parallel
//! with the outer one. Results come back to Python in archive order through a queue of per-member result slots.
//!
//! The states that the C++ and Python readers keep in flags are types here:
//!   `Fail` separates a truncated stream (EOFError) from corrupt data and I/O errors.
//!   `Msg` is what the reader thread can say: a member, the clean end of the archive, or a failure; after `End` or
//!     `Fail` it says nothing else, and the Python side turns a closed queue without either into an error.
//!   `Slot` is a member in archive order: present only, or an array still being decoded by a worker.
//!   `Header` is what a tar header block is: end of archive, a long-name record, a pax global record, or an entry.
//!   `State` is the Python-facing side: still reading, or done (after End or a failure every next() returns None).
//! A dropped Reader sets `stop`, so the reader thread quits at its next header instead of inflating ahead.

mod npz;

use flate2::read::MultiGzDecoder;
use pyo3::exceptions::{PyEOFError, PyOSError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyTuple};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug)]
pub enum Fail {
    Truncated(String),
    Corrupt(String),
    Io(String),
}

impl From<io::Error> for Fail {
    fn from(e: io::Error) -> Self {
        match e.kind() {
            io::ErrorKind::UnexpectedEof => Fail::Truncated(format!("truncated gzip stream: {e}")),
            io::ErrorKind::InvalidData | io::ErrorKind::InvalidInput => {
                Fail::Corrupt(format!("corrupt gzip data: {e}"))
            }
            _ => Fail::Io(e.to_string()),
        }
    }
}

impl From<Fail> for PyErr {
    fn from(f: Fail) -> PyErr {
        match f {
            Fail::Truncated(m) => PyEOFError::new_err(m),
            Fail::Corrupt(m) => PyRuntimeError::new_err(m),
            Fail::Io(m) => PyOSError::new_err(m),
        }
    }
}

pub enum Slot {
    Present { feat: String, name: String },
    Array { feat: String, name: String, result: Receiver<Result<npz::Npy, String>> },
}

pub enum Msg {
    Member(Slot),
    End,
    Fail(Fail),
}

enum Header {
    EndOfArchive,
    LongName { pax: bool, size: u64 },
    GlobalPax { size: u64 },
    Entry { name: String, size: u64, regular: bool },
}

fn tar_number(b: &[u8]) -> u64 {
    if b[0] & 0x80 != 0 {
        return b[1..].iter().fold((b[0] & 0x7f) as u64, |v, &c| (v << 8) | c as u64);
    }
    b.iter()
        .skip_while(|&&c| c == b' ' || c == 0)
        .take_while(|&&c| (b'0'..=b'7').contains(&c))
        .fold(0, |v, &c| v * 8 + (c - b'0') as u64)
}

fn cstr(b: &[u8]) -> String {
    String::from_utf8_lossy(&b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())]).into_owned()
}

fn pax_path(rec: &[u8]) -> Option<String> {
    let (mut path, mut i) = (None, 0);
    while i < rec.len() {
        let Some(off) = rec[i..].iter().position(|&c| c == b' ') else { break };
        let sp = i + off;
        let Some(n) = std::str::from_utf8(&rec[i..sp]).ok().and_then(|s| s.parse::<usize>().ok()) else { break };
        if n == 0 || i + n > rec.len() || sp + 1 > i + n - 1 {
            break;
        }
        if let Some(p) = rec[sp + 1..i + n - 1].strip_prefix(b"path=") {
            path = Some(String::from_utf8_lossy(p).into_owned());
        }
        i += n;
    }
    path
}

/// Reads exactly `buf.len()` bytes. Ok(false) only when the stream ends cleanly before the first byte: MultiGzDecoder
/// reports an unfinished gzip member (no end marker or trailer) as an error, so a clean end here is a real end.
fn read_block(r: &mut impl Read, buf: &mut [u8]) -> Result<bool, Fail> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) if got == 0 => return Ok(false),
            Ok(0) => return Err(Fail::Truncated("truncated tar header".into())),
            Ok(k) => got += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}

fn read_exact(r: &mut impl Read, n: u64, what: &str) -> Result<Vec<u8>, Fail> {
    let mut v = Vec::with_capacity(n as usize);
    r.by_ref().take(n).read_to_end(&mut v)?;
    if (v.len() as u64) < n {
        return Err(Fail::Truncated(format!("truncated {what}")));
    }
    Ok(v)
}

fn skip(r: &mut impl Read, n: u64, what: &str) -> Result<(), Fail> {
    if io::copy(&mut r.by_ref().take(n), &mut io::sink())? < n {
        return Err(Fail::Truncated(format!("truncated {what}")));
    }
    Ok(())
}

fn parse_header(h: &[u8; 512]) -> Header {
    if h.iter().all(|&c| c == 0) {
        return Header::EndOfArchive;
    }
    let size = tar_number(&h[124..136]);
    match h[156] {
        b'L' => Header::LongName { pax: false, size },
        b'x' => Header::LongName { pax: true, size },
        b'g' => Header::GlobalPax { size },
        kind => {
            let mut name = cstr(&h[0..100]);
            if &h[257..262] == b"ustar" {
                let prefix = cstr(&h[345..500]);
                if !prefix.is_empty() {
                    name = format!("{prefix}/{name}");
                }
            }
            Header::Entry { name, size, regular: matches!(kind, b'0' | 0 | b'7') }
        }
    }
}

struct Job {
    npz: Vec<u8>,
    reply: SyncSender<Result<npz::Npy, String>>,
}

fn decode(job: Job) {
    let r = npz::zip_entry(&job.npz, "data.npy").and_then(npz::parse_npy);
    let _ = job.reply.send(r); // the slot is gone if Python stopped early
}

struct Walk {
    design: String,
    dirs: HashMap<String, String>,
    wanted: Option<HashSet<String>>,
}

impl Walk {
    /// (feature, map name) for a member `[./][<design>/]<dir>/<name>.npz` whose dir is mapped, else None.
    fn feature(&self, name: &str) -> Option<(String, String)> {
        let base = name.strip_suffix(".npz")?;
        let (d, f) = base.rsplit_once('/').unwrap_or(("", base));
        let d = d.strip_prefix("./").unwrap_or(d);
        let d = d.strip_prefix(&format!("{}/", self.design)).unwrap_or(d);
        Some((self.dirs.get(d)?.clone(), f.to_string()))
    }
}

fn pad(size: u64) -> u64 {
    size.div_ceil(512) * 512
}

/// The reader thread: one pass over the tarball, sending a Slot per matching member, then End or Fail.
fn walk(path: String, w: Walk, out: SyncSender<Msg>, jobs: SyncSender<Job>, stop: Arc<AtomicBool>) {
    let r = (|| -> Result<(), Fail> {
        let f = File::open(&path).map_err(|e| Fail::Io(format!("cannot open {path}: {e}")))?;
        let mut gz = MultiGzDecoder::new(BufReader::with_capacity(1 << 22, f));
        let mut long_name: Option<String> = None;
        let mut hdr = [0u8; 512];
        loop {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            if !read_block(&mut gz, &mut hdr)? {
                return Err(Fail::Truncated("tar ended without its end-of-archive block".into()));
            }
            match parse_header(&hdr) {
                Header::EndOfArchive => {
                    let _ = out.send(Msg::End);
                    return Ok(());
                }
                Header::GlobalPax { size } => skip(&mut gz, pad(size), "pax global header")?,
                Header::LongName { pax, size } => {
                    let data = read_exact(&mut gz, size, "tar extended header")?;
                    skip(&mut gz, pad(size) - size, "tar extended header")?;
                    let n = if pax { pax_path(&data) } else { Some(cstr(&data)) };
                    long_name = n.or(long_name);
                }
                Header::Entry { name, size, regular } => {
                    let name = long_name.take().filter(|n| !n.is_empty()).unwrap_or(name);
                    let hit = if regular { w.feature(&name) } else { None };
                    let Some((feat, mname)) = hit else {
                        skip(&mut gz, pad(size), "tar member")?;
                        continue;
                    };
                    let slot = if w.wanted.as_ref().is_none_or(|s| s.contains(&mname)) {
                        let what = format!("tar member {name}");
                        let npz = read_exact(&mut gz, size, &what)?;
                        skip(&mut gz, pad(size) - size, &what)?;
                        let (reply, result) = sync_channel(1);
                        if jobs.send(Job { npz, reply }).is_err() {
                            return Ok(()); // workers gone: the Reader was dropped
                        }
                        Slot::Array { feat, name: mname, result }
                    } else {
                        skip(&mut gz, pad(size), "tar member")?;
                        Slot::Present { feat, name: mname }
                    };
                    if out.send(Msg::Member(slot)).is_err() {
                        return Ok(()); // the Reader was dropped
                    }
                }
            }
        }
    })();
    if let Err(f) = r {
        let _ = out.send(Msg::Fail(f));
    }
}

enum State {
    Running(Receiver<Msg>),
    Done,
}

type Item = (String, String, Option<npz::Npy>);

#[pyclass(module = "edalint_n14io_rs._reader")]
struct Reader {
    state: Mutex<State>,
    stop: Arc<AtomicBool>,
}

#[pymethods]
impl Reader {
    #[new]
    #[pyo3(signature = (path, design, dir_to_feature, wanted=None, workers=0))]
    fn new(
        path: String,
        design: String,
        dir_to_feature: HashMap<String, String>,
        wanted: Option<Vec<String>>,
        workers: usize,
    ) -> Self {
        // CircuitNet npz members are stored (not deflated), so a decoder only copies and parses a header: two keep up
        // with the outer inflate. More help only for archives of deflated npz files.
        let n = if workers > 0 { workers } else { 2 };
        let (out_tx, out_rx) = sync_channel::<Msg>(256);
        let (job_tx, job_rx) = sync_channel::<Job>(4 * n);
        let job_rx = Arc::new(Mutex::new(job_rx));
        for _ in 0..n {
            let rx = Arc::clone(&job_rx);
            thread::spawn(move || {
                loop {
                    let job = match rx.lock() {
                        Ok(g) => g.recv(),
                        Err(_) => return,
                    };
                    match job {
                        Ok(j) => decode(j),
                        Err(_) => return, // the reader thread finished and dropped the sender
                    }
                }
            });
        }
        let stop = Arc::new(AtomicBool::new(false));
        let w = Walk { design, dirs: dir_to_feature, wanted: wanted.map(|v| v.into_iter().collect()) };
        let s = Arc::clone(&stop);
        thread::spawn(move || walk(path, w, out_tx, job_tx, s));
        Reader { state: Mutex::new(State::Running(out_rx)), stop }
    }

    /// The next matching member as (feature, name, None) or (feature, name, (descr, fortran, shape, bytearray)),
    /// or None at the end of the archive.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let got: Result<Option<Item>, Fail> = py.detach(|| {
            let mut st = self.state.lock().map_err(|_| Fail::Io("reader state poisoned".into()))?;
            let State::Running(rx) = &*st else { return Ok(None) };
            let r = match rx.recv() {
                Ok(Msg::Member(Slot::Present { feat, name })) => return Ok(Some((feat, name, None))),
                Ok(Msg::Member(Slot::Array { feat, name, result })) => match result.recv() {
                    Ok(Ok(a)) => return Ok(Some((feat, name, Some(a)))),
                    Ok(Err(m)) => Err(Fail::Corrupt(format!("{feat}/{name}: {m}"))),
                    Err(_) => Err(Fail::Io("npz decoder thread failed".into())),
                },
                Ok(Msg::End) => Ok(None),
                Ok(Msg::Fail(f)) => Err(f),
                Err(_) => Err(Fail::Io("reader thread stopped without reporting the end of the archive".into())),
            };
            *st = State::Done; // after End or any failure the reader says nothing more
            r
        });
        let Some((feat, name, arr)) = got? else { return Ok(None) };
        let arr = match arr {
            None => py.None().into_bound(py),
            Some(a) => {
                let data = PyByteArray::new(py, &a.buf[a.offset..]);
                (a.descr, a.fortran, PyTuple::new(py, a.shape)?, data).into_pyobject(py)?.into_any()
            }
        };
        let t = PyTuple::new(py, [feat.into_pyobject(py)?.into_any(), name.into_pyobject(py)?.into_any(), arr])?;
        Ok(Some(t))
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[pymodule]
fn _reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reader>()
}
