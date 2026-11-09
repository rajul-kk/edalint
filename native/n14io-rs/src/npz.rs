//! The `data.npy` entry of an in-memory `.npz` (zip, stored or deflated, zip64) and its npy header.

use flate2::read::DeflateDecoder;
use std::io::Read;

fn le16(z: &[u8], p: usize) -> Result<u64, String> {
    z.get(p..p + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as u64).ok_or_else(|| "npz: truncated".into())
}
fn le32(z: &[u8], p: usize) -> Result<u64, String> {
    z.get(p..p + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64).ok_or_else(|| "npz: truncated".into())
}
fn le64(z: &[u8], p: usize) -> Result<u64, String> {
    Ok(le32(z, p)? | (le32(z, p + 4)? << 32))
}

/// The bytes of entry `want` of the zip archive `z`.
pub fn zip_entry(z: &[u8], want: &str) -> Result<Vec<u8>, String> {
    if z.len() < 22 {
        return Err("npz too small".into());
    }
    let lo = z.len().saturating_sub(65557);
    let eocd = (lo..=z.len() - 22)
        .rev()
        .find(|&i| z[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or("npz: no end of central directory")?;
    let mut cd_off = le32(z, eocd + 16)?;
    let mut n_entries = le16(z, eocd + 10)?;
    if (cd_off == 0xffff_ffff || n_entries == 0xffff) && eocd >= 20 && le32(z, eocd - 20)? == 0x0706_4b50 {
        let z64 = le64(z, eocd - 20 + 8)? as usize;
        if le32(z, z64)? != 0x0606_4b50 {
            return Err("npz: bad zip64 record".into());
        }
        n_entries = le64(z, z64 + 32)?;
        cd_off = le64(z, z64 + 48)?;
    }
    let mut p = cd_off as usize;
    for _ in 0..n_entries {
        if le32(z, p)? != 0x0201_4b50 {
            return Err("npz: bad central directory".into());
        }
        let method = le16(z, p + 10)?;
        let (mut csize, mut usize_, mut loff) = (le32(z, p + 20)?, le32(z, p + 24)?, le32(z, p + 42)?);
        let (nlen, xlen, clen) = (le16(z, p + 28)? as usize, le16(z, p + 30)? as usize, le16(z, p + 32)? as usize);
        let name = z.get(p + 46..p + 46 + nlen).ok_or("npz: truncated")?;
        // zip64 extra field: the 0xffffffff fields follow in the order usize, csize, offset
        let (mut x, xend) = (p + 46 + nlen, p + 46 + nlen + xlen);
        while x + 4 <= xend {
            let (id, sz) = (le16(z, x)?, le16(z, x + 2)? as usize);
            if id == 1 {
                let mut q = x + 4;
                for f in [&mut usize_, &mut csize, &mut loff] {
                    if *f == 0xffff_ffff {
                        *f = le64(z, q)?;
                        q += 8;
                    }
                }
            }
            x += 4 + sz;
        }
        if name == want.as_bytes() {
            let loff = loff as usize;
            if le32(z, loff)? != 0x0403_4b50 {
                return Err("npz: bad local header".into());
            }
            let data = loff + 30 + le16(z, loff + 26)? as usize + le16(z, loff + 28)? as usize;
            let raw = z.get(data..data + csize as usize).ok_or("npz: entry runs past the archive")?;
            return match method {
                0 if csize == usize_ => Ok(raw.to_vec()),
                0 => Err("npz: stored entry with csize != usize".into()),
                8 => {
                    let mut out = Vec::with_capacity(usize_ as usize);
                    DeflateDecoder::new(raw)
                        .read_to_end(&mut out)
                        .map_err(|e| format!("npz: bad deflate entry: {e}"))?;
                    if out.len() as u64 != usize_ {
                        return Err("npz: bad deflate entry (size)".into());
                    }
                    Ok(out)
                }
                m => Err(format!("npz: unsupported compression method {m}")),
            };
        }
        p += 46 + nlen + xlen + clen;
    }
    Err(format!("npz has no entry {want}"))
}

/// A parsed npy array: dtype string, memory order, shape, and the raw data at `buf[offset..]`.
pub struct Npy {
    pub descr: String,
    pub fortran: bool,
    pub shape: Vec<usize>,
    pub buf: Vec<u8>,
    pub offset: usize,
}

fn dict_value<'a>(h: &'a str, key: &str) -> Result<&'a str, String> {
    let k = h.find(&format!("'{key}'")).ok_or_else(|| format!("npy header without {key}"))?;
    let rest = h[k + key.len() + 2..].trim_start().strip_prefix(':').ok_or("npy: bad header")?.trim_start();
    let (open, close) = match rest.as_bytes().first() {
        Some(b'\'') => ('\'', '\''),
        Some(b'(') => ('(', ')'),
        _ => {
            let e = rest.find([',', '}']).unwrap_or(rest.len());
            return Ok(rest[..e].trim());
        }
    };
    let body = &rest[open.len_utf8()..];
    Ok(&body[..body.find(close).ok_or("npy: bad header")?])
}

pub fn parse_npy(buf: Vec<u8>) -> Result<Npy, String> {
    if buf.len() < 10 || buf[..6] != [0x93, b'N', b'U', b'M', b'P', b'Y'] {
        return Err("not an npy array".into());
    }
    let (hlen, hstart) = if buf[6] == 1 { (le16(&buf, 8)? as usize, 10) } else { (le32(&buf, 8)? as usize, 12) };
    let h = std::str::from_utf8(buf.get(hstart..hstart + hlen).ok_or("npy header runs past the data")?)
        .map_err(|_| "npy: header is not text")?;
    let descr = dict_value(h, "descr")?.to_string();
    let fortran = dict_value(h, "fortran_order")? == "True";
    let shape = dict_value(h, "shape")?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| format!("npy: bad shape entry {s}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Npy { descr, fortran, shape, buf, offset: hstart + hlen })
}
