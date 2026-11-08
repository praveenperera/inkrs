//! Hecate 9.6 um, the 2D ink map of `hecate.py` at huggingface.co/scrollprize/hecate 9cb86e5 (MIT,
//! "Copyright (c) 2026 Vesuvius Challenge")
//!
//! A 3D ResNet-152 (Bottleneck [3, 8, 36, 3], stride-1 stem, no max pool) on 16 x 64 x 64 tiles of
//! the central 16 planes, a 3D U-Net decoder with GroupNorm, and a depth-attention head that
//! collapses the 16 depths to one probability per column. [`model`] runs the network on cuDNN with
//! the glue kernels of `inkrs-kernels/src/hecate.rs`; [`engine`] tiles a volume, blends the tiles as
//! `hecate.py`'s `predict` does and writes its PNG and JSON

#[cfg(not(target_os = "macos"))]
pub mod engine;
#[cfg(not(target_os = "macos"))]
pub mod model;
#[cfg(not(target_os = "macos"))]
pub mod p0;

#[cfg(target_os = "macos")]
pub use crate::metal::hecate::{engine, model};

use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, bail, ensure};
#[cfg(not(target_os = "macos"))]
use cudakit::ComputeCapability;

#[cfg(not(target_os = "macos"))]
use crate::ensure_supported;
use crate::zarr::Volume;

/// Tile edge
pub const P: usize = 64;
/// Planes per tile
pub const D: usize = 16;
/// Pixels per tile plane
pub const PIX: usize = P * P;
/// The voxel size the released checkpoint was trained at, um
pub const SAMPLING_UM: f64 = 9.6;

/// Refuses an index space of `n` elements that a kernel's u32 offsets or slice lengths can't hold
pub fn u32_extent(what: &str, n: usize) -> Result<()> {
    ensure!(n <= u32::MAX as usize, "{what}: {n} elements overflow the kernels' u32 indexing");
    Ok(())
}

/// Numerics of a run, `hecate.py --precision`
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Precision {
    /// float32 activations; `tf32` lets cuDNN use TF32 tensor cores, PyTorch's default for convs
    Fp32 { tf32: bool },
    /// PyTorch's bf16 autocast
    Bf16,
}

impl Precision {
    /// Whether activations are stored in bf16
    pub fn bf16(self) -> bool {
        self == Self::Bf16
    }

    /// `hecate.py`'s name for it
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp32 { .. } => "fp32",
            Self::Bf16 => "bf16",
        }
    }

    /// Fails with [`crate::error::Unsupported`] on a GPU that can't run this precision: bf16 needs
    /// the BF16 tensor cores of compute capability 8.0 and up, so a Turing GPU (T4, RTX 20) runs fp32.
    /// CUDA only: every Apple GPU the Metal backend runs on has bf16
    #[cfg(not(target_os = "macos"))]
    pub fn check_device(self, cc: ComputeCapability) -> Result<()> {
        ensure_supported!(
            !self.bf16() || cc >= ComputeCapability::new(8, 0),
            "Hecate bf16 needs a GPU of compute capability 8.0 or above, for its BF16 tensor cores; this GPU is {cc}, \
             so run it at fp32"
        );
        Ok(())
    }
}

/// `torch.hann_window(64, periodic=False)` as float32 bits, read from torch 2.13 on the CPU,
/// because the closed form in other libraries differs in the last bit at some entries
const HANN64_BITS: [u32; 64] = [
    0x0, 0x3b22d500, 0x3c226d40, 0x3cb5f940, 0x3d20d110, 0x3d7966f8, 0x3db1ee74, 0x3def920c, 0x3e1a80b4, 0x3e40c5f2,
    0x3e6a3760, 0x3e8b35c8, 0x3ea27902, 0x3ebaaa32, 0x3ed38bca, 0x3eecde7d, 0x3f0330ee, 0x3f0fea7f, 0x3f1c7b91,
    0x3f28c42c, 0x3f34a50f, 0x3f400000, 0x3f4ab81e, 0x3f54b220, 0x3f5dd4a4, 0x3f660870, 0x3f6d38a2, 0x3f7352f2,
    0x3f7847d9, 0x3f7c0aba, 0x3f7e9202, 0x3f7fd744, 0x3f7fd744, 0x3f7e9202, 0x3f7c0aba, 0x3f7847d9, 0x3f7352f2,
    0x3f6d38a2, 0x3f660870, 0x3f5dd4a4, 0x3f54b21e, 0x3f4ab81e, 0x3f3fffff, 0x3f34a50f, 0x3f28c42a, 0x3f1c7b92,
    0x3f0fea7d, 0x3f0330ee, 0x3eecde82, 0x3ed38bcc, 0x3ebaaa37, 0x3ea27902, 0x3e8b35ca, 0x3e6a3760, 0x3e40c5f6,
    0x3e1a80b2, 0x3def9214, 0x3db1ee74, 0x3d796700, 0x3d20d110, 0x3cb5f950, 0x3c226d40, 0x3b22d500, 0x0,
];

/// `hecate.py`'s blend window `[64, 64]`: the outer product of the 1D Hann window, divided by its
/// maximum (floored at float32 eps) and clamped below at 0.001, all in float32
pub fn window() -> Vec<f32> {
    let h: Vec<f32> = HANN64_BITS.iter().map(|&b| f32::from_bits(b)).collect();
    let mut w: Vec<f32> = (0..PIX).map(|i| h[i / P] * h[i % P]).collect();
    let max = w.iter().copied().fold(f32::MIN, f32::max).max(f32::EPSILON);
    w.iter_mut().for_each(|v| *v = (*v / max).max(0.001));
    w
}

/// `hecate.py`'s `axis_starts`: every `stride` from 0, then the last full tile
pub fn axis_starts(length: usize, stride: usize) -> Vec<usize> {
    let last = length.saturating_sub(P);
    let mut v: Vec<usize> = (0..last).step_by(stride).collect();
    v.push(last);
    v
}

/// The first input plane and the plane step of the central 16, `hecate.py`'s `zs` (reversed
/// with `reverse`, which flips the whole Z axis before selecting)
pub fn central_planes(depth: usize, reverse: bool) -> Result<(i32, i32)> {
    ensure!(depth >= D, "the volume has {depth} planes; Hecate needs at least {D}");
    let z0 = depth / 2 - D / 2;
    Ok(if reverse { ((depth - z0 - 1) as i32, -1) } else { (z0 as i32, 1) })
}

/// A uint8 `[Z, Y, X]` volume from a zarr v2 array or group (array `0`), or from a `.npy`; refuses one
/// the gather kernel's u32 offsets can't index
pub fn read_volume(path: &Path) -> Result<Volume> {
    let npy = path.extension().is_some_and(|e| e == "npy");
    // every command that runs the gather kernel reads its volume here, so none can skip this bound;
    // it is checked on the declared shape, before any data is read or allocated
    let dims = if npy { npy_header(&read_head(path)?, path)?.0 } else { zarr_shape(path)? };
    ensure!(dims.len() == 3, "{}: shape {dims:?}, need [Z, Y, X]", path.display());
    check_volume(dims[0], dims[1], dims[2])?;
    let vol = if npy { read_npy(path)? } else { Volume::read(path)? };
    ensure!([vol.depth, vol.height, vol.width] == dims[..], "{}: shape changed while reading", path.display());
    Ok(vol)
}

/// Refuses a `[Z, Y, X]` volume past the gather kernel's u32 indexing
pub fn check_volume(depth: usize, height: usize, width: usize) -> Result<()> {
    u32_extent("volume", depth.saturating_mul(height).saturating_mul(width))
}

/// A C-order uint8 3D `.npy`, as `np.load` reads what `hecate.py` accepts
fn read_npy(path: &Path) -> Result<Volume> {
    let bytes = std::fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?;
    let (dims, data_at) = npy_header(&bytes, path)?;
    ensure!(dims.len() == 3, "{}: shape {dims:?}, need [Z, Y, X]", path.display());
    let n = dims.iter().product::<usize>();
    let data = &bytes[data_at..];
    ensure!(data.len() == n, "{}: {} data bytes for shape {dims:?}", path.display(), data.len());
    Ok(Volume { depth: dims[0], height: dims[1], width: dims[2], data: data.to_vec(), array: path.to_path_buf() })
}

/// The first bytes of a `.npy`, enough for any header: version 1 headers are at most 65,535 bytes,
/// and longer version 2 and 3 headers are refused
fn read_head(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;

    let mut head = Vec::new();
    std::fs::File::open(path)
        .wrap_err_with(|| format!("read {}", path.display()))?
        .take(12 + 65_536)
        .read_to_end(&mut head)?;
    Ok(head)
}

/// The shape and data offset of a C-order uint8 `.npy` from its header bytes
fn npy_header(bytes: &[u8], path: &Path) -> Result<(Vec<usize>, usize)> {
    ensure!(bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY", "{}: not a .npy file", path.display());
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "{}: truncated header", path.display());
            (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        }
        v => bail!("{}: .npy version {v} is not supported", path.display()),
    };
    ensure!(bytes.len() >= start + header_len, "{}: truncated header", path.display());
    let header = std::str::from_utf8(&bytes[start..start + header_len])?;
    let field = |key: &str| -> Option<&str> {
        let at = header.find(&format!("'{key}':"))? + key.len() + 3;
        Some(header[at..].trim_start())
    };
    let descr = field("descr").unwrap_or_default();
    ensure!(descr.starts_with("'|u1'") || descr.starts_with("'u1'"), "{}: dtype {descr}, need uint8", path.display());
    ensure!(field("fortran_order").is_some_and(|v| v.starts_with("False")), "{}: Fortran order", path.display());
    let shape = field("shape").unwrap_or_default();
    let dims: Vec<usize> = shape
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()
        .wrap_err_with(|| format!("{}: shape {shape}", path.display()))?;
    Ok((dims, start + header_len))
}

/// The declared shape of a zarr v2 array, or of a group's array `0`, from its `.zarray`
fn zarr_shape(path: &Path) -> Result<Vec<usize>> {
    let meta = [path.join(".zarray"), path.join("0/.zarray")].into_iter().find(|p| p.exists());
    let Some(meta) = meta else { bail!("{}: neither a zarr v2 array nor a group with array 0", path.display()) };
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta)?).wrap_err_with(|| format!("parse {}", meta.display()))?;
    v["shape"]
        .as_array()
        .and_then(|a| a.iter().map(|d| d.as_u64().map(|d| d as usize)).collect())
        .ok_or_else(|| color_eyre::eyre::eyre!("{}: no shape", meta.display()))
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    /// bf16 is refused below compute capability 8.0 as unsupported, so villa and Python callers fall back
    /// instead of failing inside cuDNN's planning; fp32 runs everywhere
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn bf16_needs_sm80() {
        let cc = ComputeCapability::new;
        let err = Precision::Bf16.check_device(cc(7, 5)).unwrap_err();
        assert!(crate::error::is_unsupported(&err), "{err}");
        assert!(Precision::Bf16.check_device(cc(8, 0)).is_ok());
        assert!(Precision::Bf16.check_device(cc(12, 0)).is_ok());
        for tf32 in [false, true] {
            assert!(Precision::Fp32 { tf32 }.check_device(cc(7, 5)).is_ok());
        }
    }

    /// A volume past the gather kernel's u32 offsets is refused by `read_volume`, the reader every
    /// command uses, from its declared shape: these files hold a header and no data, so a reader that
    /// skipped the bound would fail on the data instead, with a different message
    #[test]
    fn read_volume_refuses_past_u32_from_the_header() {
        let dir = std::env::temp_dir().join(format!("inkrs-hecate-extent-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("big.zarr")).unwrap();
        let npy = |shape: &str| {
            let mut h = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': {shape}, }}");
            while (10 + h.len() + 1) % 64 != 0 {
                h.push(' ');
            }
            h.push('\n');
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend_from_slice(&(h.len() as u16).to_le_bytes());
            b.extend_from_slice(h.as_bytes());
            b
        };
        std::fs::write(dir.join("big.npy"), npy("(128, 8192, 8192)")).unwrap();
        std::fs::write(dir.join("small.npy"), npy("(16, 64, 64)")).unwrap();
        std::fs::write(
            dir.join("big.zarr/.zarray"),
            r#"{"zarr_format": 2, "shape": [128, 8192, 8192], "chunks": [16, 512, 512], "dtype": "|u1",
                "order": "C", "compressor": null, "fill_value": 0, "filters": null}"#,
        )
        .unwrap();

        let err = |p: &str| format!("{:#}", read_volume(&dir.join(p)).err().expect(p));
        assert!(err("big.npy").contains("overflow the kernels' u32 indexing"), "{}", err("big.npy"));
        assert!(err("big.zarr").contains("overflow the kernels' u32 indexing"), "{}", err("big.zarr"));
        // the same header at a size in bounds gets past the check and fails on the missing data
        assert!(err("small.npy").contains("data bytes"), "{}", err("small.npy"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// torch 2.13: `sha256(win.numpy().tobytes())` for `predict`'s `win`
    #[test]
    fn window_matches_torch() {
        let bytes: Vec<u8> = window().iter().flat_map(|v| v.to_le_bytes()).collect();
        let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "4d3d6dd90de1d806cf5dc42345afad281951adaaad036b73dba79d354c99483e");
    }

    #[test]
    fn starts_and_planes_follow_hecate() {
        assert_eq!(axis_starts(2682, 32).len(), 83);
        assert_eq!(*axis_starts(2682, 32).last().unwrap(), 2618);
        // a length the stride divides: the last start is not repeated
        assert_eq!(axis_starts(128, 32), [0, 32, 64]);
        assert_eq!(axis_starts(40, 32), [0]);
        // 28 planes: z0 = 6, forward 6..22, reverse planes 21 down to 6
        assert_eq!(central_planes(28, false).unwrap(), (6, 1));
        assert_eq!(central_planes(28, true).unwrap(), (21, -1));
        // 31 planes: z0 = 7, forward 7..23, reverse depth - z0 - 16 = 8 .. 24, read from 23 down
        assert_eq!(central_planes(31, true).unwrap(), (23, -1));
    }
}
