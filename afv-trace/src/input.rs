//! Prediction readers: Zarr v2 and NPY, as `Predictions` in the reference
//!
//! Mirrors `storage.Predictions`: a `(2, Z, Y, X)` array of uint8 or floats in [0, 1], returned in V, H
//! channel order, with floats quantized by `rint(x * 255)` in float32. Both array orders and both byte
//! orders are read. Zarr chunks are decoded by the c-blosc release numcodecs 0.15 bundles, or by
//! numcodecs' Zstd, Zlib, GZip and LZ4 formats

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, ensure, eyre};
use flate2::read::{MultiGzDecoder, ZlibDecoder};
use half::f16;
use memmap2::Mmap;
use serde_json::Value;

const DTYPE_ERROR: &str = "Predictions must be uint8 [0,255] or floating point [0,1]";
const VALUE_ERROR: &str = "Predictions contain NaN, infinity or values outside [0,1]";

/// Element kind of a prediction array
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    U8,
    F16,
    F32,
    F64,
}

/// Element type of a prediction array: kind and byte order
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dtype {
    kind: Kind,
    big_endian: bool,
}

impl Dtype {
    fn size(self) -> usize {
        match self.kind {
            Kind::U8 => 1,
            Kind::F16 => 2,
            Kind::F32 => 4,
            Kind::F64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        self.kind != Kind::U8
    }

    /// Parse a numpy type string such as `|u1`, `<f4` or `>f8`
    fn parse(text: &str) -> Result<Self> {
        let (big_endian, code) = match text.as_bytes().first() {
            Some(b'>') => (true, &text[1..]),
            Some(b'<' | b'|' | b'=') => (false, &text[1..]),
            _ => (false, text),
        };
        let kind = match code {
            "u1" => Kind::U8,
            "f2" => Kind::F16,
            "f4" => Kind::F32,
            "f8" => Kind::F64,
            // numpy's long double: a float the reference accepts but no input pipeline writes
            c if c.starts_with('f') => bail!("extended-precision float predictions ({text}) are not supported"),
            _ => bail!("{DTYPE_ERROR}"),
        };
        Ok(Self { kind, big_endian })
    }

    /// One element as float64; floats only
    fn value(self, b: &[u8]) -> f64 {
        let big = self.big_endian;
        match self.kind {
            Kind::F16 => {
                let bits = [b[0], b[1]];
                f16::from_bits(if big { u16::from_be_bytes(bits) } else { u16::from_le_bytes(bits) }).to_f64()
            }
            Kind::F32 => {
                let bits = [b[0], b[1], b[2], b[3]];
                (if big { f32::from_be_bytes(bits) } else { f32::from_le_bytes(bits) }) as f64
            }
            Kind::F64 => {
                let bits = b[..8].try_into().expect("8 bytes");
                if big { f64::from_be_bytes(bits) } else { f64::from_le_bytes(bits) }
            }
            Kind::U8 => b[0] as f64,
        }
    }

    /// `v` as one element's bytes in this type and byte order
    fn encode(self, v: f64) -> Vec<u8> {
        let big = self.big_endian;
        match self.kind {
            Kind::U8 => vec![v as u8],
            Kind::F16 => {
                let bits = f16::from_f64(v).to_bits();
                (if big { bits.to_be_bytes() } else { bits.to_le_bytes() }).to_vec()
            }
            Kind::F32 => (if big { (v as f32).to_be_bytes() } else { (v as f32).to_le_bytes() }).to_vec(),
            Kind::F64 => (if big { v.to_be_bytes() } else { v.to_le_bytes() }).to_vec(),
        }
    }
}

/// A Zarr v2 compressor this reader decodes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compressor {
    Raw,
    Blosc,
    Zstd,
    Zlib,
    Gzip,
    Lz4,
}

enum Store {
    Zarr { dir: PathBuf, chunks: [usize; 4], sep: char, fill: Vec<u8>, compressor: Compressor, fortran: bool },
    Npy { map: Mmap, offset: usize, fortran: bool },
}

/// A local two-channel prediction array
pub struct Predictions {
    /// Resolved input path, as recorded in `run.json`
    pub path: PathBuf,
    /// Spatial shape, z, y, x
    pub shape: [usize; 3],
    pub dtype: Dtype,
    reverse: bool,
    store: Store,
}

impl Predictions {
    /// Open `path` (a `.npy` file or a Zarr array or group, with `key` naming the array in a group)
    pub fn open(path: &Path, key: Option<&str>, order: &str) -> Result<Self> {
        let path = fs::canonicalize(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
        let (store, full_shape, dtype) = if path.extension().is_some_and(|e| e == "npy") {
            if key.is_some_and(|k| !k.is_empty()) {
                bail!("--array is only valid for a Zarr group");
            }
            open_npy(&path)?
        } else {
            open_zarr(&path, key)?
        };
        if full_shape.len() != 4 {
            bail!("Predictions must be an array with shape (2, Z, Y, X)");
        }
        if full_shape[0] != 2 || full_shape[1..].iter().min().copied().unwrap_or(0) < 2 {
            bail!("Expected two channels and at least 2 voxels along each spatial axis");
        }
        let shape = [full_shape[1], full_shape[2], full_shape[3]];
        Ok(Self { path, shape, dtype, reverse: order == "HV", store })
    }

    /// Read `(2, hi - lo)` as uint8 in V, H order (`Predictions.read(lo, hi)`)
    pub fn read_u8(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Vec<u8>> {
        let raw = self.read_raw(lo, hi)?;
        if self.dtype.kind == Kind::U8 {
            return Ok(raw);
        }
        let floats = self.to_f32(&raw)?;
        // np.rint(block * 255).astype("u1") in float32; values are already checked to lie in [0, 1]
        Ok(floats.iter().map(|&v| (v * 255.0f32).round_ties_even() as u8).collect())
    }

    /// Read `(2, hi - lo)` as float32 without quantization (`read(..., quantized=False)`)
    pub fn read_f32(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Vec<f32>> {
        let raw = self.read_raw(lo, hi)?;
        if self.dtype.kind == Kind::U8 {
            return Ok(raw.iter().map(|&v| v as f32).collect());
        }
        self.to_f32(&raw)
    }

    /// The reference's finite and [0, 1] check in the input type, then `astype("f4")`
    fn to_f32(&self, raw: &[u8]) -> Result<Vec<f32>> {
        let dtype = self.dtype;
        raw.chunks_exact(dtype.size())
            .map(|b| {
                let v = dtype.value(b);
                (v.is_finite() && (0.0..=1.0).contains(&v)).then_some(v as f32)
            })
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| eyre!("{VALUE_ERROR}"))
    }

    /// Raw element bytes of `(2, hi - lo)`, C order, channels already in V, H order
    fn read_raw(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Vec<u8>> {
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
        let es = self.dtype.size();
        let plane = dims.iter().product::<usize>() * es;
        let mut out = vec![0u8; 2 * plane];
        match &self.store {
            Store::Npy { map, offset, fortran } => {
                let s = self.shape;
                // element strides of (c, z, y, x)
                let strides = match fortran {
                    false => [s[0] * s[1] * s[2], s[1] * s[2], s[2], 1],
                    true => [1, 2, 2 * s[0], 2 * s[0] * s[1]],
                };
                let data = &map[*offset..];
                for c in 0..2 {
                    for z in 0..dims[0] {
                        for y in 0..dims[1] {
                            let row = c * strides[0] + (lo[0] + z) * strides[1] + (lo[1] + y) * strides[2];
                            let dst = c * plane + ((z * dims[1] + y) * dims[2]) * es;
                            copy_strided(data, row + lo[2] * strides[3], strides[3], dims[2], es, &mut out[dst..]);
                        }
                    }
                }
            }
            Store::Zarr { chunks, .. } => {
                let first = [0, 1, 2].map(|a| lo[a] / chunks[a + 1]);
                let last = [0, 1, 2].map(|a| (hi[a] - 1) / chunks[a + 1]);
                for kc in 0..2usize.div_ceil(chunks[0]) {
                    for kz in first[0]..=last[0] {
                        for ky in first[1]..=last[1] {
                            for kx in first[2]..=last[2] {
                                self.copy_chunk([kc, kz, ky, kx], lo, dims, &mut out)?;
                            }
                        }
                    }
                }
            }
        }
        if self.reverse {
            let (a, b) = out.split_at_mut(plane);
            a.swap_with_slice(b);
        }
        Ok(out)
    }

    /// Copy the part of chunk `key` (channel chunk, z, y, x) that falls in the window at `lo`
    fn copy_chunk(&self, key: [usize; 4], lo: [usize; 3], dims: [usize; 3], out: &mut [u8]) -> Result<()> {
        let Store::Zarr { chunks, .. } = &self.store else { unreachable!("zarr only") };
        let es = self.dtype.size();
        let data = self.chunk_bytes(key)?;
        let cs = [chunks[1], chunks[2], chunks[3]];
        let origin = [0, 1, 2].map(|a| key[a + 1] * cs[a]);
        let a0 = [0, 1, 2].map(|a| lo[a].max(origin[a]));
        let a1 = [0, 1, 2].map(|a| (lo[a] + dims[a]).min(origin[a] + cs[a]).min(self.shape[a]));
        let plane = dims.iter().product::<usize>() * es;
        let cplane = cs.iter().product::<usize>() * es;
        let n = (a1[2] - a0[2]) * es;
        let c0 = key[0] * chunks[0];
        for local in 0..chunks[0].min(2 - c0) {
            let c = c0 + local;
            for z in a0[0]..a1[0] {
                for y in a0[1]..a1[1] {
                    let src =
                        local * cplane + (((z - origin[0]) * cs[1] + y - origin[1]) * cs[2] + a0[2] - origin[2]) * es;
                    let dst = c * plane + (((z - lo[0]) * dims[1] + y - lo[1]) * dims[2] + a0[2] - lo[2]) * es;
                    out[dst..dst + n].copy_from_slice(&data[src..src + n]);
                }
            }
        }
        Ok(())
    }

    /// Decoded bytes of one chunk in C order, or the fill value when the chunk file is absent
    fn chunk_bytes(&self, key: [usize; 4]) -> Result<Vec<u8>> {
        let Store::Zarr { dir, chunks, sep, fill, compressor, fortran } = &self.store else {
            unreachable!("zarr only")
        };
        let n = chunks.iter().product::<usize>() * self.dtype.size();
        let name = format!("{}{sep}{}{sep}{}{sep}{}", key[0], key[1], key[2], key[3]);
        let bytes = match fs::read(dir.join(&name)) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(fill.iter().copied().cycle().take(n).collect());
            }
            Err(e) => return Err(eyre!("{}: {e}", dir.join(&name).display())),
        };
        let out = decode(*compressor, bytes, n).map_err(|e| eyre!("chunk {name}: {e}"))?;
        ensure!(out.len() == n, "chunk {name} decoded to {} bytes, expected {n}", out.len());
        Ok(if *fortran { fortran_to_c(&out, *chunks, self.dtype.size()) } else { out })
    }

    /// Zarr chunk edge along each spatial axis, or `None` for NPY input
    pub fn chunk_shape(&self) -> Option<[usize; 3]> {
        match &self.store {
            Store::Zarr { chunks, .. } => Some([chunks[1], chunks[2], chunks[3]]),
            Store::Npy { .. } => None,
        }
    }

    /// One whole spatial chunk as uint8 `(2, cz, cy, cx)` in V, H order, clipped to the array shape;
    /// returns the chunk's origin and extent too. NPY input is served in blocks of the same size
    pub fn read_chunk_u8(&self, key: [usize; 3], edge: [usize; 3]) -> Result<([usize; 3], [usize; 3], Vec<u8>)> {
        let lo = [0, 1, 2].map(|a| key[a] * edge[a]);
        let hi = [0, 1, 2].map(|a| (lo[a] + edge[a]).min(self.shape[a]));
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
        Ok((lo, dims, self.read_u8(lo, hi)?))
    }
}

/// Copy `count` elements of `es` bytes, `stride` elements apart from element `first`, contiguously into `out`
fn copy_strided(data: &[u8], first: usize, stride: usize, count: usize, es: usize, out: &mut [u8]) {
    if stride == 1 {
        out[..count * es].copy_from_slice(&data[first * es..(first + count) * es]);
        return;
    }
    for i in 0..count {
        let at = (first + i * stride) * es;
        out[i * es..(i + 1) * es].copy_from_slice(&data[at..at + es]);
    }
}

/// A 4-D chunk of `dims` in Fortran order, reordered to C order
fn fortran_to_c(data: &[u8], dims: [usize; 4], es: usize) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    let mut at = 0;
    for i0 in 0..dims[0] {
        for i1 in 0..dims[1] {
            for i2 in 0..dims[2] {
                for i3 in 0..dims[3] {
                    let src = (i0 + dims[0] * (i1 + dims[1] * (i2 + dims[2] * i3))) * es;
                    out[at..at + es].copy_from_slice(&data[src..src + es]);
                    at += es;
                }
            }
        }
    }
    out
}

/// One chunk file through its compressor; `n` is the decoded size
fn decode(compressor: Compressor, bytes: Vec<u8>, n: usize) -> Result<Vec<u8>> {
    Ok(match compressor {
        Compressor::Raw => bytes,
        Compressor::Blosc => blosc_decompress(&bytes)?,
        Compressor::Zstd => zstd::stream::decode_all(&bytes[..])?,
        Compressor::Zlib => {
            let mut out = Vec::with_capacity(n);
            ZlibDecoder::new(&bytes[..]).read_to_end(&mut out)?;
            out
        }
        Compressor::Gzip => {
            let mut out = Vec::with_capacity(n);
            MultiGzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
            out
        }
        Compressor::Lz4 => {
            // numcodecs' LZ4: the decoded size as a little-endian int32, then one LZ4 block
            ensure!(bytes.len() >= 4, "LZ4 chunk too short");
            let size = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            ensure!(size >= 0, "bad LZ4 chunk size {size}");
            lz4_flex::block::decompress(&bytes[4..], size as usize).map_err(|e| eyre!("LZ4: {e}"))?
        }
    })
}

/// Decompress one Blosc 1 frame with c-blosc
fn blosc_decompress(src: &[u8]) -> Result<Vec<u8>> {
    let mut nbytes = 0usize;
    // SAFETY: c-blosc reads at most `src.len()` bytes of the header to validate it
    let valid = unsafe { blosc_src::blosc_cbuffer_validate(src.as_ptr().cast(), src.len(), &mut nbytes) };
    ensure!(valid == 0, "not a valid Blosc frame");
    let mut out = vec![0u8; nbytes];
    if nbytes == 0 {
        return Ok(out);
    }
    // SAFETY: the frame was validated against `src.len()`, and c-blosc writes at most `nbytes` bytes;
    // the context call keeps no global state, so rayon threads may decode at once
    let n = unsafe { blosc_src::blosc_decompress_ctx(src.as_ptr().cast(), out.as_mut_ptr().cast(), nbytes, 1) };
    ensure!(n >= 0 && n as usize == nbytes, "Blosc frame failed to decode ({n})");
    Ok(out)
}

fn open_npy(path: &Path) -> Result<(Store, Vec<usize>, Dtype)> {
    let file = fs::File::open(path)?;
    // SAFETY: the input is read-only for the whole run; the reference memory-maps it the same way
    let map = unsafe { Mmap::map(&file)? };
    ensure!(map.len() >= 10 && &map[..6] == b"\x93NUMPY", "{}: not an NPY file", path.display());
    let (hlen, start) = match map[6] {
        1 => (u16::from_le_bytes([map[8], map[9]]) as usize, 10),
        _ => (u32::from_le_bytes([map[8], map[9], map[10], map[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&map[start..start + hlen])?;
    let field = |name: &str| -> Result<&str> {
        let at = header.find(&format!("'{name}'")).ok_or_else(|| eyre!("NPY header lacks {name}"))?;
        let rest = header[at + name.len() + 2..].trim_start().trim_start_matches(':').trim_start();
        Ok(rest)
    };
    let descr = field("descr")?;
    let descr = descr.trim_start_matches('\'').split('\'').next().unwrap_or_default();
    let dtype = Dtype::parse(descr)?;
    let fortran = field("fortran_order")?.starts_with("True");
    let shape_text = field("shape")?;
    let shape_text = &shape_text[1..shape_text.find(')').ok_or_else(|| eyre!("bad NPY shape"))?];
    let shape: Vec<usize> =
        shape_text.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::parse).collect::<Result<_, _>>()?;
    let offset = start + hlen;
    ensure!(
        map.len() >= offset + shape.iter().product::<usize>() * dtype.size(),
        "{}: NPY file is truncated",
        path.display()
    );
    Ok((Store::Npy { map, offset, fortran }, shape, dtype))
}

fn open_zarr(path: &Path, key: Option<&str>) -> Result<(Store, Vec<usize>, Dtype)> {
    let dir = match key {
        Some(k) if !k.is_empty() => {
            let d = path.join(k);
            if !d.join(".zarray").is_file() {
                bail!("Zarr array not found: {k}");
            }
            d
        }
        _ => path.to_path_buf(),
    };
    let text = fs::read_to_string(dir.join(".zarray"))
        .map_err(|_| eyre!("Predictions must be an array with shape (2, Z, Y, X)"))?;
    let meta: Value = serde_json::from_str(&text)?;
    ensure!(meta["zarr_format"] == 2, "only Zarr v2 is supported");
    let fortran = match meta["order"].as_str() {
        Some("C") => false,
        Some("F") => true,
        other => bail!("unsupported Zarr order {other:?}"),
    };
    // numcodecs filters (delta, scale-offset, quantize and so on) are not used by any prediction writer
    ensure!(meta["filters"].as_array().is_none_or(Vec::is_empty), "Zarr filters are not supported");
    let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
    let chunks: Vec<usize> = serde_json::from_value(meta["chunks"].clone())?;
    ensure!(shape.len() == 4 && chunks.len() == 4, "Predictions must be an array with shape (2, Z, Y, X)");
    ensure!(chunks.iter().all(|&c| c > 0), "Zarr chunks must be positive");
    let dtype = Dtype::parse(meta["dtype"].as_str().ok_or_else(|| eyre!("{DTYPE_ERROR}"))?)?;
    let sep = match meta.get("dimension_separator").and_then(Value::as_str) {
        Some("/") => '/',
        _ => '.',
    };
    let compressor = match meta["compressor"]["id"].as_str() {
        None => Compressor::Raw,
        Some("blosc") => Compressor::Blosc,
        Some("zstd") => Compressor::Zstd,
        Some("zlib") => Compressor::Zlib,
        Some("gzip") => Compressor::Gzip,
        Some("lz4") => Compressor::Lz4,
        Some(other) => bail!("unsupported Zarr compressor {other}"),
    };
    let fill = dtype.encode(fill_value(&meta["fill_value"])?);
    let chunks = [chunks[0], chunks[1], chunks[2], chunks[3]];
    Ok((Store::Zarr { dir, chunks, sep, fill, compressor, fortran }, shape, dtype))
}

/// A Zarr v2 `fill_value`: a number, null (zero here; zarr leaves such chunks uninitialized), or one of
/// the strings the spec uses for non-finite floats
fn fill_value(v: &Value) -> Result<f64> {
    Ok(match v {
        Value::Null => 0.0,
        Value::Number(n) => n.as_f64().ok_or_else(|| eyre!("bad Zarr fill value {n}"))?,
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => bail!("unsupported Zarr fill value {other:?}"),
        },
        other => bail!("unsupported Zarr fill value {other}"),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use super::Predictions;

    /// Every case in `tests/fixtures/readers` (the expected values were written by the reference's own
    /// reader): the same bytes as `Predictions.read`, quantized and not, over the whole
    /// array and a window that cuts chunks, or the same error
    #[test]
    fn readers_match_the_reference_on_every_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/readers");
        let manifest: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).expect("manifest"))
                .expect("manifest json");
        assert!(manifest.len() >= 30, "fixtures missing");
        let expected =
            |case: &Value, field: &str| std::fs::read(root.join("expected").join(case[field].as_str().unwrap()));
        let corner = |v: &Value| -> [usize; 3] { serde_json::from_value(v.clone()).unwrap() };
        for case in &manifest {
            let name = case["name"].as_str().unwrap();
            let read = || -> color_eyre::Result<Vec<(String, Vec<u8>)>> {
                let p = Predictions::open(
                    &root.join(case["path"].as_str().unwrap()),
                    case["key"].as_str(),
                    case["order"].as_str().unwrap(),
                )?;
                let (lo, hi) = (corner(&case["window"][0]), corner(&case["window"][1]));
                let le = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                Ok(vec![
                    ("full_u8".into(), p.read_u8([0; 3], p.shape)?),
                    ("full_f32".into(), le(p.read_f32([0; 3], p.shape)?)),
                    ("window_u8".into(), p.read_u8(lo, hi)?),
                    ("window_f32".into(), le(p.read_f32(lo, hi)?)),
                ])
            };
            match (read(), case["error"].as_str()) {
                (Ok(reads), None) => {
                    for (field, got) in reads {
                        assert!(got == expected(case, &field).unwrap(), "{name}: {field} differs from the reference");
                    }
                }
                (Err(e), Some(message)) => assert_eq!(e.to_string(), message, "{name}"),
                (Ok(_), Some(message)) => panic!("{name}: read succeeded, the reference raised {message:?}"),
                (Err(e), None) => panic!("{name}: {e}"),
            }
        }
    }
}
