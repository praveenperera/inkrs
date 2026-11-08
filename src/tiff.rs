//! villa's output TIFF: one uint8 plane, BigTIFF, 128x128 tiles, LZW, with the tags, types and
//! values `tifffile.imwrite(..., compression="lzw", tile=(128, 128), bigtiff=True, metadata=None,
//! software="vesuvius.ink_detection.inference.infer")` writes (tifffile 2026.9.20): LONG image
//! and tile dimensions, unit resolution, LONG8 tile offsets and LONG tile byte counts
//!
//! Edge tiles are padded with zeros to the full tile, as tifffile pads them. Tiles are
//! encoded in parallel

use std::io::Write;
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use rayon::prelude::*;

/// villa's `software` tag
pub const SOFTWARE: &str = "vesuvius.ink_detection.inference.infer";
/// villa's tile edge for a 128 patch (`patch_size` when it is a multiple of 16)
pub const TILE: usize = 128;

const ASCII: u16 = 2;
const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;
const LONG8: u16 = 16;

/// Writes `image` (`height x width`, row-major) to `path`
pub fn write(path: &Path, image: &[u8], height: usize, width: usize) -> Result<()> {
    ensure!(image.len() == height * width, "image of {} bytes is not {height}x{width}", image.len());
    let (width32, height32) =
        (u32::try_from(width).wrap_err("width over u32")?, u32::try_from(height).wrap_err("height over u32")?);
    let (across, down) = (width.div_ceil(TILE), height.div_ceil(TILE));
    let tiles: Vec<Vec<u8>> = (0..across * down)
        .into_par_iter()
        .map(|t| {
            let (ty, tx) = (t / across, t % across);
            let mut tile = vec![0u8; TILE * TILE];
            let (y0, x0) = (ty * TILE, tx * TILE);
            let w = TILE.min(width - x0);
            for y in 0..TILE.min(height - y0) {
                let src = (y0 + y) * width + x0;
                tile[y * TILE..y * TILE + w].copy_from_slice(&image[src..src + w]);
            }
            weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                .encode(&tile)
                .map_err(|e| eyre!("LZW tile {t}: {e:?}"))
        })
        .collect::<Result<_>>()?;

    let mut out: Vec<u8> = Vec::with_capacity(tiles.iter().map(Vec::len).sum::<usize>() + 4096);
    out.extend_from_slice(b"II");
    out.extend_from_slice(&43u16.to_le_bytes());
    out.extend_from_slice(&8u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // first IFD offset, patched below

    let mut offsets = Vec::with_capacity(tiles.len());
    for t in &tiles {
        offsets.push(out.len() as u64);
        out.extend_from_slice(t);
    }
    // tifffile writes byte counts as LONG; a 128x128 LZW tile is far below 4 GiB
    let counts: Vec<u32> = tiles.iter().map(|t| u32::try_from(t.len())).collect::<Result<_, _>>()?;

    // out-of-line values: the software string and the two tile tables
    let align = |out: &mut Vec<u8>| {
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
    };
    align(&mut out);
    let software_at = out.len() as u64;
    out.extend_from_slice(SOFTWARE.as_bytes());
    out.push(0);
    align(&mut out);
    let offsets_at = out.len() as u64;
    offsets.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    let counts_at = out.len() as u64;
    counts.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    align(&mut out);

    let ifd_at = out.len() as u64;
    out[8..16].copy_from_slice(&ifd_at.to_le_bytes());
    let n_tiles = tiles.len() as u64;
    // a rational 1/1: numerator then denominator, inline in BigTIFF's 8-byte value field
    let unit_rational = 1u64 | (1u64 << 32);
    // (tag, type, count, value or offset), ascending by tag as TIFF requires
    let entries: Vec<(u16, u16, u64, u64)> = vec![
        (256, LONG, 1, u64::from(width32)),
        (257, LONG, 1, u64::from(height32)),
        (258, SHORT, 1, 8),
        (259, SHORT, 1, 5),
        (262, SHORT, 1, 1),
        (277, SHORT, 1, 1),
        (282, RATIONAL, 1, unit_rational),
        (283, RATIONAL, 1, unit_rational),
        (296, SHORT, 1, 1),
        (305, ASCII, SOFTWARE.len() as u64 + 1, software_at),
        (322, LONG, 1, TILE as u64),
        (323, LONG, 1, TILE as u64),
        // tifffile's `_bytecount_format`: a lone tile's count is LONG8, otherwise LONG (its size
        // guess, 10x the raw tile, is past u16); one 8-byte value or two 4-byte ones fit inline
        (324, LONG8, n_tiles, if n_tiles == 1 { offsets[0] } else { offsets_at }),
        match n_tiles {
            1 => (325, LONG8, 1, u64::from(counts[0])),
            2 => (325, LONG, 2, u64::from(counts[0]) | (u64::from(counts[1]) << 32)),
            _ => (325, LONG, n_tiles, counts_at),
        },
    ];
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (tag, ty, count, value) in entries {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&0u64.to_le_bytes());

    let tmp = path.with_extension("tif.partial");
    let mut f = std::fs::File::create(&tmp).wrap_err_with(|| format!("create {}", tmp.display()))?;
    f.write_all(&out)?;
    f.sync_data().ok();
    std::fs::rename(&tmp, path).wrap_err_with(|| format!("rename to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Command;

    /// Writes `image` with [`super::write`], then has tifffile (villa's writer) read it back and
    /// write the same array with villa's arguments; the pixels and every tag's type, count and
    /// value (except the tile tables' offsets) must match
    fn matches_tifffile(height: usize, width: usize) {
        let Some(python) = std::env::var("INKRS_PYTHON").ok().filter(|p| !p.is_empty()) else {
            eprintln!("skipped: INKRS_PYTHON is not set");
            return;
        };
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-tiff");
        std::fs::create_dir_all(&dir).unwrap();
        let ours = dir.join(format!("ours-{height}x{width}.tif"));
        let raw = dir.join(format!("image-{height}x{width}.u8"));
        // varied but compressible content, so LZW codes of every width appear
        let image: Vec<u8> =
            (0..height * width).map(|i| ((i / width) * 7 + (i % width) / 3) as u8 ^ (i % 251) as u8).collect();
        super::write(&ours, &image, height, width).unwrap();
        std::fs::write(&raw, &image).unwrap();
        let script = r#"
import sys, numpy as np, tifffile
ours, raw, h, w = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
image = np.fromfile(raw, np.uint8).reshape(h, w)
ref = ours + ".tifffile.tif"
tifffile.imwrite(ref, image, compression="lzw", tile=(128, 128), bigtiff=True, metadata=None,
                 software="vesuvius.ink_detection.inference.infer")
def tags(path):
    with tifffile.TiffFile(path) as t:
        assert t.is_bigtiff and len(t.pages) == 1, path
        return [(g.code, int(g.dtype), g.count, None if g.code in (324, 325) else g.value) for g in t.pages[0].tags.values()], t.pages[0].asarray()
a, pa = tags(ours)
b, pb = tags(ref)
assert a == b, f"tags differ: ours {a} tifffile {b}"
assert pa.dtype == np.uint8 and pa.shape == (h, w) and np.array_equal(pa, image), "pixels differ"
print("ok", h, w, len(a), "tags")
"#;
        let out = Command::new(&python)
            .args([
                "-I",
                "-c",
                script,
                ours.to_str().unwrap(),
                raw.to_str().unwrap(),
                &height.to_string(),
                &width.to_string(),
            ])
            .output()
            .unwrap();
        eprintln!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success(), "tifffile check failed for {height}x{width}");
    }

    #[test]
    fn tiff_matches_tifffile_small() {
        matches_tifffile(300, 200);
    }

    /// One and two tiles: the tile tables inline in the IFD
    #[test]
    fn tiff_matches_tifffile_inline_tables() {
        matches_tifffile(100, 90);
        matches_tifffile(100, 200);
    }

    /// Both dimensions above u16, the case SHORT tags could not hold
    #[test]
    fn tiff_matches_tifffile_above_u16() {
        matches_tifffile(3, 70_001);
        matches_tifffile(66_000, 2);
    }
}
