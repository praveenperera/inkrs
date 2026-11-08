//! A zarr v2 reader for the layer stacks the reader runs on: one `uint8` array `[Z, H, W]`,
//! C order, no filters, zstd or blosc-lz4 chunks, either dimension separator, and missing
//! chunks read as the fill value
//!
//! A path to a group opens its array `0`, the full-resolution level villa reads, and
//! [`LowRes::scan`] reads the pyramid level villa's first empty-tile skip uses

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use rayon::prelude::*;
use serde::Deserialize;

use crate::ensure_supported;
use crate::error::Unsupported;

#[derive(Debug, Deserialize)]
struct ArrayMeta {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Codec {
    Raw,
    Zstd,
    BloscLz4,
}

/// A whole `[Z, H, W]` uint8 volume in memory, C order
pub struct Volume {
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<u8>,
    /// The array directory actually read
    pub array: PathBuf,
}

impl Volume {
    /// Reads every chunk of the array at `path` (or of `path/0` for a group), in parallel
    pub fn read(path: &Path) -> Result<Self> {
        let array = if path.join(".zarray").exists() {
            path.to_path_buf()
        } else if path.join(".zgroup").exists() && path.join("0").join(".zarray").exists() {
            path.join("0")
        } else {
            bail!("{}: neither a zarr v2 array nor a group with array 0", path.display());
        };
        let (shape, data) = read_u8(&array)?;
        ensure_supported!(shape.len() == 3, "{}: need a 3D array, got shape {shape:?}", array.display());
        // villa `_volume_axes` reads the array as [H, W, Z] unless depth is the smallest axis
        // (`argmin(shape) == 0`); this engine only reads [Z, H, W]
        ensure_supported!(
            shape[0] <= shape[1] && shape[0] <= shape[2],
            "{}: shape {shape:?} has depth larger than height or width, which villa reads as [H, W, Z]; not supported",
            array.display()
        );
        Ok(Self { depth: shape[0], height: shape[1], width: shape[2], data, array })
    }

    /// The volume as a borrowed view
    pub fn view(&self) -> VolumeRef<'_> {
        VolumeRef { depth: self.depth, height: self.height, width: self.width, data: &self.data }
    }
}

/// A volume must be smaller than this many bytes: the kernels index it with u32
pub const MAX_VOLUME_BYTES: usize = u32::MAX as usize;

/// A borrowed `[Z, H, W]` uint8 volume, C order: a [`Volume`] read from zarr, or a caller's array
#[derive(Clone, Copy, Debug)]
pub struct VolumeRef<'a> {
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub data: &'a [u8],
}

impl<'a> VolumeRef<'a> {
    /// Wraps `data`, which must hold exactly `depth * height * width` bytes
    pub fn new(data: &'a [u8], depth: usize, height: usize, width: usize) -> Result<Self> {
        let n = depth
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| eyre!("volume size overflows"))?;
        ensure!(data.len() == n, "volume of {} bytes is not {depth}x{height}x{width}", data.len());
        Ok(Self { depth, height, width, data })
    }
}

/// villa's low-resolution occupancy scan (`build_lowres_block_mask` with no user mask): a 2D
/// nonzero mask from one pyramid level of a group, and the level-0 pixels per mask pixel
pub struct LowRes {
    /// The pyramid level read
    pub level: String,
    pub height: usize,
    pub width: usize,
    /// `[height, width]`, row-major: whether any voxel along depth is nonzero
    pub mask: Vec<bool>,
    pub scale_y: usize,
    pub scale_x: usize,
}

/// villa `DEFAULT_OCCUPANCY_SCAN_LEVEL`
pub const OCCUPANCY_LEVEL: &str = "3";

impl LowRes {
    /// villa's scan for the input at `path`, whose level 0 is `height x width`: none for a bare
    /// array, otherwise the group's level [`OCCUPANCY_LEVEL`] or the nearest numeric level
    /// (`choose_pyramid_array`), collapsed over depth (`compute_nonempty_mask_from_lowres_array`)
    pub fn scan(path: &Path, height: usize, width: usize) -> Result<Option<Self>> {
        if path.join(".zarray").exists() {
            return Ok(None);
        }
        ensure!(path.join(".zgroup").exists(), "{}: not a zarr v2 group", path.display());
        let mut keys = Vec::new();
        for entry in std::fs::read_dir(path).wrap_err_with(|| format!("list {}", path.display()))? {
            let entry = entry?;
            if entry.path().join(".zarray").exists() {
                keys.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        let level =
            choose_level(&keys, OCCUPANCY_LEVEL).ok_or_else(|| eyre!("{}: no arrays in the group", path.display()))?;
        let (shape, data) = read_u8(&path.join(&level))?;
        Self::from_level(level, &shape, &data, height, width).map(Some)
    }

    /// villa's scan from one pyramid level already in memory: `shape` is rank 2 (a mask) or
    /// rank 3 (collapsed over its depth axis as villa does), `data` C order, and level 0 is
    /// `height x width`. `level` only labels the report
    pub fn from_level(level: String, shape: &[usize], data: &[u8], height: usize, width: usize) -> Result<Self> {
        ensure!(
            shape.iter().product::<usize>() == data.len(),
            "occupancy level of {} bytes is not {shape:?}",
            data.len()
        );
        // a zero axis would reach chunks_exact(0), which panics and poisons a caller's engine lock
        ensure!(!shape.contains(&0), "occupancy level of shape {shape:?} is empty");
        let (mh, mw, mask) = match shape[..] {
            [h, w] => (h, w, data.iter().map(|&v| v != 0).collect()),
            [a, b, c] => {
                // numpy argmin: the first smallest axis; depth is axis 0 if that is it, else axis 2
                if a <= b && a <= c {
                    let mut m = vec![false; b * c];
                    data.chunks_exact(b * c).for_each(|plane| m.iter_mut().zip(plane).for_each(|(o, &v)| *o |= v != 0));
                    (b, c, m)
                } else {
                    (a, b, data.chunks_exact(c).map(|row| row.iter().any(|&v| v != 0)).collect())
                }
            }
            _ => bail!("occupancy array must be rank 2 or 3, got {shape:?}"),
        };
        // python's round() is half to even
        let scale = |full: usize, low: usize| ((full as f64 / low.max(1) as f64).round_ties_even() as usize).max(1);
        Ok(Self { level, height: mh, width: mw, mask, scale_y: scale(height, mh), scale_x: scale(width, mw) })
    }

    /// villa `iter_blocks`'s mask test for the tile at `(y0, x0)` with `valid_h x valid_w`
    /// pixels: any nonzero mask pixel in its footprint, numpy slices clamped to the mask
    pub fn any(&self, y0: usize, x0: usize, valid_h: usize, valid_w: usize) -> bool {
        let (ly0, lx0) = (y0 / self.scale_y, x0 / self.scale_x);
        let ly1 = (ly0 + 1).max((y0 + valid_h).div_ceil(self.scale_y)).min(self.height);
        let lx1 = (lx0 + 1).max((x0 + valid_w).div_ceil(self.scale_x)).min(self.width);
        (ly0..ly1).any(|y| self.mask[y * self.width + lx0.min(lx1)..y * self.width + lx1].iter().any(|&v| v))
    }
}

/// villa `choose_pyramid_array` over a group's array keys: `preferred` when present, else the
/// numeric key nearest to it (the lower on a tie), else the first key in sorted order
fn choose_level(keys: &[String], preferred: &str) -> Option<String> {
    let mut keys = keys.to_vec();
    keys.sort();
    if keys.iter().any(|k| k == preferred) {
        return Some(preferred.to_owned());
    }
    let digits = |k: &str| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit());
    let numeric: Vec<(i128, &String)> =
        keys.iter().filter(|k| digits(k)).filter_map(|k| Some((k.parse().ok()?, k))).collect();
    if digits(preferred) && !numeric.is_empty() {
        let p: i128 = preferred.parse().ok()?;
        return numeric.into_iter().min_by_key(|&(v, _)| ((v - p).abs(), v)).map(|(_, k)| k.clone());
    }
    keys.first().cloned()
}

/// Reads a whole rank 2 or 3 `uint8` zarr v2 array, chunks decoded in parallel; returns its
/// shape and C-order data
fn read_u8(array: &Path) -> Result<(Vec<usize>, Vec<u8>)> {
    let meta: ArrayMeta = serde_json::from_slice(
        &std::fs::read(array.join(".zarray")).wrap_err_with(|| format!("read {}/.zarray", array.display()))?,
    )
    .wrap_err_with(|| format!("parse {}/.zarray", array.display()))?;
    ensure!(meta.zarr_format == 2, "zarr format {}", meta.zarr_format);
    ensure!(
        meta.chunks.len() == meta.shape.len(),
        "{}: shape {:?} and chunks {:?} differ in rank",
        array.display(),
        meta.shape,
        meta.chunks
    );
    ensure_supported!(
        matches!(meta.shape.len(), 2 | 3),
        "{}: need a 2D or 3D array, got shape {:?}",
        array.display(),
        meta.shape
    );
    // before any chunk is read, so an input over the limit is refused without filling host memory
    let bytes = meta.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).unwrap_or(usize::MAX);
    ensure_supported!(
        bytes < MAX_VOLUME_BYTES,
        "{}: shape {:?} is {bytes} bytes, over 4 GiB; the kernels index a volume with u32",
        array.display(),
        meta.shape
    );
    ensure_supported!(meta.dtype == "|u1", "{}: dtype {}, need |u1", array.display(), meta.dtype);
    ensure_supported!(meta.order == "C", "order {}, need C", meta.order);
    ensure_supported!(
        meta.filters.as_ref().is_none_or(|f| f.is_null() || f.as_array().is_some_and(|a| a.is_empty())),
        "filters {:?} are not supported",
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

    // a 2D array reads as one plane of depth 1, with 2-part chunk keys
    let rank2 = meta.shape.len() == 2;
    let pad = |v: &[usize]| if rank2 { [1, v[0], v[1]] } else { [v[0], v[1], v[2]] };
    let [d, h, w] = pad(&meta.shape);
    let [cd, ch, cw] = pad(&meta.chunks);
    let grid = [d.div_ceil(cd), h.div_ceil(ch), w.div_ceil(cw)];
    let keys: Vec<[usize; 3]> =
        (0..grid[0]).flat_map(|a| (0..grid[1]).flat_map(move |b| (0..grid[2]).map(move |c| [a, b, c]))).collect();

    // decode in parallel, then copy each chunk into place
    let chunks: Vec<Option<Vec<u8>>> = keys
        .par_iter()
        .map(|k| {
            let name =
                if rank2 { format!("{}{sep}{}", k[1], k[2]) } else { format!("{}{sep}{}{sep}{}", k[0], k[1], k[2]) };
            let file = chunk_file(array, &name);
            let bytes = match std::fs::read(&file) {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(eyre!(e).wrap_err(format!("read chunk {name}"))),
            };
            let raw = decode(codec, &bytes).wrap_err_with(|| format!("decode chunk {name}"))?;
            ensure!(raw.len() == cd * ch * cw, "chunk {name}: {} bytes, expected {}", raw.len(), cd * ch * cw);
            Ok(Some(raw))
        })
        .collect::<Result<_>>()?;

    let mut data = vec![fill; d * h * w];
    for (k, chunk) in keys.iter().zip(chunks) {
        let Some(chunk) = chunk else { continue };
        let (z0, y0, x0) = (k[0] * cd, k[1] * ch, k[2] * cw);
        let (nz, ny, nx) = (cd.min(d - z0), ch.min(h - y0), cw.min(w - x0));
        for z in 0..nz {
            for y in 0..ny {
                let src = (z * ch + y) * cw;
                let dst = ((z0 + z) * h + y0 + y) * w + x0;
                data[dst..dst + nx].copy_from_slice(&chunk[src..src + nx]);
            }
        }
    }
    Ok((meta.shape, data))
}

pub(crate) fn decode(codec: Codec, bytes: &[u8]) -> Result<Vec<u8>> {
    match codec {
        Codec::Raw => Ok(bytes.to_vec()),
        Codec::Zstd => Ok(zstd::stream::decode_all(bytes)?),
        Codec::BloscLz4 => blosc_lz4(bytes),
    }
}

/// The file of chunk `key` under `array`, joining a `/`-separated key component by component: a Windows path with
/// the `\\?\` prefix takes no `/` as a separator
pub fn chunk_file(array: &Path, key: &str) -> PathBuf {
    key.split('/').fold(array.to_path_buf(), |path, part| path.join(part))
}

/// Decodes one blosc1 frame of lz4 blocks
///
/// Header: version, lz4 version, flags, typesize, then u32 `nbytes`, `blocksize`, `cbytes`.
/// Flags bit 0 is byte shuffle, bit 1 a memcpy'd (stored) frame, bit 2 bit shuffle, bit 4
/// "don't split", bits 5-7 the codec (1 = lz4). Then one u32 start offset per block; each
/// block is one stream, or `typesize` split streams, each an i32 compressed size followed by
/// lz4 data, stored raw when that size equals the stream's length
pub fn blosc_lz4(bytes: &[u8]) -> Result<Vec<u8>> {
    ensure!(bytes.len() >= 16, "blosc frame shorter than its header");
    let flags = bytes[2];
    let typesize = bytes[3] as usize;
    let u32_at = |o: usize| -> Result<usize> {
        let b = bytes.get(o..o + 4).ok_or_else(|| eyre!("blosc frame truncated at {o}"))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let (nbytes, blocksize, cbytes) = (u32_at(4)?, u32_at(8)?, u32_at(12)?);
    ensure!(cbytes == bytes.len(), "blosc cbytes {cbytes} but the frame has {} bytes", bytes.len());
    if flags & 0x2 != 0 {
        ensure!(bytes.len() == 16 + nbytes, "stored blosc frame of the wrong size");
        return Ok(bytes[16..].to_vec());
    }
    ensure!(flags & 0x4 == 0, "bit-shuffled blosc frames are not supported");
    ensure!((flags >> 5) == 1, "blosc codec {} is not lz4", flags >> 5);
    let shuffle = flags & 0x1 != 0 && typesize > 1;
    ensure!(!shuffle, "byte-shuffled blosc frames with typesize {typesize} are not supported (uint8 needs none)");
    ensure!(blocksize > 0, "blosc blocksize 0");

    let nblocks = nbytes.div_ceil(blocksize);
    let mut out = vec![0u8; nbytes];
    for i in 0..nblocks {
        let start = u32_at(16 + 4 * i)?;
        let block_len = blocksize.min(nbytes - i * blocksize);
        // blosc splits a block into typesize streams unless told not to; at typesize 1 it is
        // one stream either way
        let split = flags & 0x10 == 0 && typesize > 1;
        ensure!(!split, "split blosc blocks need typesize 1");
        let dst = &mut out[i * blocksize..i * blocksize + block_len];
        let csize = u32_at(start)?;
        let src = bytes.get(start + 4..start + 4 + csize).ok_or_else(|| eyre!("blosc block {i} truncated"))?;
        if csize == block_len {
            dst.copy_from_slice(src);
        } else {
            let n = lz4_flex::block::decompress_into(src, dst).map_err(|e| eyre!("lz4 block {i}: {e}"))?;
            ensure!(n == block_len, "lz4 block {i}: {n} bytes, expected {block_len}");
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::LowRes;

    #[test]
    fn an_empty_occupancy_level_is_an_error_not_a_panic() {
        for shape in [[1, 0, 0].as_slice(), &[0, 4, 4], &[4, 0, 4], &[0, 4], &[4, 0]] {
            assert!(LowRes::from_level("3".into(), shape, &[], 512, 512).is_err(), "{shape:?}");
        }
        assert!(LowRes::from_level("3".into(), &[1, 4, 4], &[1; 16], 512, 512).is_ok());
    }
}

#[cfg(test)]
mod reader_tests {
    use std::path::PathBuf;

    use super::Volume;
    use crate::error::is_unsupported;

    fn array(name: &str, shape: &str, chunks: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("inkrs-zarr-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let meta = format!(
            r#"{{"zarr_format": 2, "shape": {shape}, "chunks": {chunks}, "dtype": "|u1", "order": "C",
                "compressor": null, "fill_value": 0, "filters": null}}"#
        );
        std::fs::write(dir.join(".zarray"), meta).unwrap();
        dir
    }

    #[test]
    fn unsupported_ranks_and_sizes_fail_before_any_chunk_is_read() {
        for (name, shape, chunks) in [
            ("rank2", "[256, 256]", "[128, 128]"),
            ("rank4", "[2, 17, 256, 256]", "[1, 17, 128, 128]"),
            // 17 x 65536 x 65536 is 68 GiB: refused from the metadata, never allocated
            ("huge", "[17, 65536, 65536]", "[17, 128, 128]"),
        ] {
            let dir = array(name, shape, chunks);
            let err = Volume::read(&dir).err().expect("reads");
            assert!(is_unsupported(&err), "{name}: {err:#}");
            std::fs::remove_dir_all(dir).unwrap();
        }
        let dir = array("badchunks", "[17, 256, 256]", "[17, 128]");
        let err = Volume::read(&dir).err().expect("reads");
        assert!(!is_unsupported(&err), "mismatched chunk rank is a corrupt array: {err:#}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
