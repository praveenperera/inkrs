//! Zarr v2 I/O for whole-volume 3D inference: reads uint8 boxes of a `[Z, Y, X]` array chunk by chunk,
//! and writes the uint8 prediction array chunk by chunk
//!
//! The reader takes the codecs [`crate::zarr`] takes (raw, zstd, blosc-lz4), either dimension separator,
//! and missing chunks as the fill value. The writer makes a group with array `0`, raw chunks, `/`
//! separators and fill 0, and skips all-zero chunks as villa's `skip_empty` arrays do. villa also writes
//! five downsampled levels; this writer does not

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use rayon::prelude::*;
use serde::Deserialize;

use crate::ensure_supported;
use crate::error::Unsupported;
use crate::zarr::{Codec, chunk_file, decode};

#[derive(Debug, Deserialize)]
struct Meta {
    zarr_format: u32,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: String,
    fill_value: Option<serde_json::Value>,
    order: String,
    filters: Option<serde_json::Value>,
    dimension_separator: Option<String>,
    compressor: Option<serde_json::Value>,
}

/// An open uint8 `[Z, Y, X]` zarr v2 array
pub struct Array {
    pub dir: PathBuf,
    pub shape: [usize; 3],
    pub chunks: [usize; 3],
    fill: u8,
    sep: String,
    codec: Codec,
}

impl Array {
    /// Opens the array at `path`, or array `0` of the group at `path`
    pub fn open(path: &Path) -> Result<Self> {
        let dir = if path.join(".zarray").exists() {
            path.to_path_buf()
        } else if path.join("0/.zarray").exists() {
            path.join("0")
        } else {
            bail!("{}: neither a zarr v2 array nor a group with array 0", path.display());
        };
        let meta: Meta = serde_json::from_slice(
            &std::fs::read(dir.join(".zarray")).wrap_err_with(|| format!("read {}/.zarray", dir.display()))?,
        )
        .wrap_err_with(|| format!("parse {}/.zarray", dir.display()))?;
        ensure!(meta.zarr_format == 2, "zarr format {}", meta.zarr_format);
        ensure_supported!(meta.shape.len() == 3 && meta.chunks.len() == 3, "{}: need a 3D array", dir.display());
        ensure_supported!(meta.dtype == "|u1", "dtype {}, need |u1", meta.dtype);
        ensure_supported!(meta.order == "C", "order {}, need C", meta.order);
        ensure_supported!(
            meta.filters.as_ref().is_none_or(|f| f.is_null() || f.as_array().is_some_and(|a| a.is_empty())),
            "filters {:?}",
            meta.filters
        );
        let fill = match &meta.fill_value {
            None | Some(serde_json::Value::Null) => 0u8,
            Some(v) => u8::try_from(v.as_u64().ok_or_else(|| eyre!("fill value {v}"))?)?,
        };
        let sep = meta.dimension_separator.clone().unwrap_or_else(|| ".".into());
        ensure!(sep == "." || sep == "/", "dimension separator {sep:?}");
        let codec = match &meta.compressor {
            None | Some(serde_json::Value::Null) => Codec::Raw,
            Some(c) => match c.get("id").and_then(|v| v.as_str()) {
                Some("zstd") => Codec::Zstd,
                Some("blosc") if c.get("cname").and_then(|v| v.as_str()) == Some("lz4") => Codec::BloscLz4,
                _ => return Err(Unsupported(format!("compressor {c} is not supported")).into()),
            },
        };
        let s = &meta.shape;
        let c = &meta.chunks;
        ensure!(c.iter().all(|&v| v > 0), "zero chunk size");
        Ok(Self { dir, shape: [s[0], s[1], s[2]], chunks: [c[0], c[1], c[2]], fill, sep, codec })
    }

    fn chunk_path(&self, k: [usize; 3]) -> PathBuf {
        chunk_file(&self.dir, &format!("{}{s}{}{s}{}", k[0], k[1], k[2], s = self.sep))
    }

    /// Reads the box `[lo, hi)` (inside the array) into `out`, C order, decoding chunks in parallel
    pub fn read_box(&self, lo: [usize; 3], hi: [usize; 3], out: &mut [u8]) -> Result<()> {
        let dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        ensure!((0..3).all(|a| lo[a] < hi[a] && hi[a] <= self.shape[a]), "box {lo:?}..{hi:?} outside {:?}", self.shape);
        ensure!(out.len() == dims.iter().product::<usize>(), "box buffer size");
        let c = self.chunks;
        let mut keys = Vec::new();
        for kz in lo[0] / c[0]..hi[0].div_ceil(c[0]) {
            for ky in lo[1] / c[1]..hi[1].div_ceil(c[1]) {
                for kx in lo[2] / c[2]..hi[2].div_ceil(c[2]) {
                    keys.push([kz, ky, kx]);
                }
            }
        }
        let decoded: Vec<Option<Vec<u8>>> = keys
            .par_iter()
            .map(|&k| {
                let path = self.chunk_path(k);
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let raw = decode(self.codec, &bytes).wrap_err_with(|| format!("decode {}", path.display()))?;
                        ensure!(
                            raw.len() == c[0] * c[1] * c[2],
                            "{}: {} bytes, expected a full chunk",
                            path.display(),
                            raw.len()
                        );
                        Ok(Some(raw))
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                    Err(e) => Err(e).wrap_err_with(|| format!("read {}", path.display())),
                }
            })
            .collect::<Result<_>>()?;
        for (k, data) in keys.iter().zip(&decoded) {
            let c0 = [k[0] * c[0], k[1] * c[1], k[2] * c[2]];
            let a = [lo[0].max(c0[0]), lo[1].max(c0[1]), lo[2].max(c0[2])];
            let b = [hi[0].min(c0[0] + c[0]), hi[1].min(c0[1] + c[1]), hi[2].min(c0[2] + c[2])];
            for z in a[0]..b[0] {
                for y in a[1]..b[1] {
                    let o = ((z - lo[0]) * dims[1] + y - lo[1]) * dims[2] + a[2] - lo[2];
                    let n = b[2] - a[2];
                    match data {
                        Some(d) => {
                            let i = ((z - c0[0]) * c[1] + y - c0[1]) * c[2] + a[2] - c0[2];
                            out[o..o + n].copy_from_slice(&d[i..i + n]);
                        }
                        None => out[o..o + n].fill(self.fill),
                    }
                }
            }
        }
        Ok(())
    }
}

/// The uint8 prediction array `0` of a new group, chunk `chunk^3`, raw chunks
pub struct Writer {
    dir: PathBuf,
    pub shape: [usize; 3],
    pub chunk: usize,
}

impl Writer {
    /// Creates `path` (refusing an existing one unless `overwrite`) with array `0` of `shape`
    pub fn create(path: &Path, shape: [usize; 3], chunk: usize, overwrite: bool) -> Result<Self> {
        if path.exists() {
            ensure!(overwrite, "{} exists; pass --overwrite", path.display());
            std::fs::remove_dir_all(path).wrap_err_with(|| format!("remove {}", path.display()))?;
        }
        let dir = path.join("0");
        std::fs::create_dir_all(&dir)?;
        std::fs::write(path.join(".zgroup"), "{\"zarr_format\": 2}")?;
        let chunks: Vec<usize> = shape.iter().map(|&s| chunk.min(s)).collect();
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "|u1", "fill_value": 0,
            "order": "C", "filters": null, "compressor": null, "dimension_separator": "/",
        });
        std::fs::write(dir.join(".zarray"), serde_json::to_string_pretty(&meta)?)?;
        std::fs::write(dir.join(".zattrs"), "{\"_ARRAY_DIMENSIONS\": [\"z\", \"y\", \"x\"]}")?;
        Ok(Self { dir, shape, chunk })
    }

    /// Writes the chunks covered by `slab`, `[d, h, w]` C order at global corner `lo`, which must lie on
    /// the chunk grid with full chunks except at the array's end. All-zero chunks are not written
    pub fn write_slab(&self, lo: [usize; 3], dims: [usize; 3], slab: &[u8]) -> Result<usize> {
        let c = self.chunk;
        ensure!(slab.len() == dims.iter().product::<usize>(), "slab size");
        ensure!(lo.iter().all(|&v| v % c == 0), "slab corner {lo:?} off the chunk grid");
        let mut keys = Vec::new();
        for z in (0..dims[0]).step_by(c) {
            for y in (0..dims[1]).step_by(c) {
                for x in (0..dims[2]).step_by(c) {
                    keys.push([z, y, x]);
                }
            }
        }
        let written: Vec<bool> = keys
            .par_iter()
            .map(|&k| {
                let g = [lo[0] + k[0], lo[1] + k[1], lo[2] + k[2]];
                let ext: Vec<usize> = (0..3).map(|a| c.min(self.shape[a] - g[a])).collect();
                ensure!((0..3).all(|a| k[a] + ext[a] <= dims[a]), "slab ends inside chunk {g:?}");
                // a chunk at the array's end is stored full size, padded with the fill value
                let cc: Vec<usize> = (0..3).map(|a| c.min(self.shape[a])).collect();
                let mut buf = vec![0u8; cc[0] * cc[1] * cc[2]];
                let mut any = false;
                for z in 0..ext[0] {
                    for y in 0..ext[1] {
                        let s = ((k[0] + z) * dims[1] + k[1] + y) * dims[2] + k[2];
                        let row = &slab[s..s + ext[2]];
                        any |= row.iter().any(|&v| v != 0);
                        buf[(z * cc[1] + y) * cc[2]..][..ext[2]].copy_from_slice(row);
                    }
                }
                if !any {
                    return Ok(false);
                }
                let path = chunk_file(&self.dir, &format!("{}/{}/{}", g[0] / c, g[1] / c, g[2] / c));
                std::fs::create_dir_all(path.parent().expect("chunk paths have parents"))?;
                std::fs::write(&path, &buf).wrap_err_with(|| format!("write {}", path.display()))?;
                Ok(true)
            })
            .collect::<Result<_>>()?;
        Ok(written.iter().filter(|&&w| w).count())
    }
}
