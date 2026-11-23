//! The binary prediction array as villa's `vesuvius.predict` lays it out: a zarr v2 array (not a group) of
//! shape `(1, Z, Y, X)`, chunks `(1, c, c, c)`, uint8, fill 0, `.` separators, all-zero chunks not stored
//! (`ZarrWriter(skip_empty=True)`). Its `.zattrs` marks it as this command's output, which is what lets
//! `--overwrite` replace it
//!
//! villa compresses with blosc (zstd level 1, byte shuffle, a no-op at one byte per value); this writer
//! uses numcodecs' plain `zstd` codec at level 1, which every zarr v2 reader decodes to the same bytes

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, ensure};
use rayon::prelude::*;

use crate::unet3d::zio::{array_meta_is, fresh_output_dir, holds_only, is_marked, mark};

/// A new prediction array
pub struct Writer {
    dir: PathBuf,
    shape: [usize; 3],
    chunk: usize,
}

impl Writer {
    /// The command named in each output's mark
    const COMMAND: &str = "nnunet-predict";

    /// Creates the array at `path` for a volume of `shape`; an existing path is replaced only with `overwrite` and
    /// only when it is an earlier prediction of this writer ([`fresh_output_dir`])
    pub fn create(path: &Path, shape: [usize; 3], chunk: usize, overwrite: bool) -> Result<Self> {
        ensure!(chunk > 0, "chunk 0");
        fresh_output_dir(path, overwrite, Self::is_prediction)?;
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": [1, shape[0], shape[1], shape[2]], "chunks": [1, chunk, chunk, chunk],
            "dtype": "|u1", "fill_value": 0, "order": "C", "filters": null,
            "compressor": {"id": "zstd", "level": 1}, "dimension_separator": ".",
        });
        std::fs::write(path.join(".zarray"), serde_json::to_string_pretty(&meta)?)?;
        mark(path, Self::COMMAND)?;
        Ok(Self { dir: path.to_path_buf(), shape, chunk })
    }

    /// Whether `path` holds only what this writer makes: its `.zarray`, its mark and chunk files keyed `0.z.y.x`
    fn is_prediction(path: &Path) -> bool {
        let chunk_key = |name: &str| {
            let mut parts = name.split('.');
            parts.next() == Some("0") && parts.clone().count() == 3 && parts.all(|p| p.parse::<u64>().is_ok())
        };
        let compressor = serde_json::json!({"id": "zstd", "level": 1});
        holds_only(path, |name, dir| !dir && (name == ".zarray" || name == ".zattrs" || chunk_key(name)))
            && is_marked(path, Self::COMMAND)
            && array_meta_is(path, &compressor, ".", 4)
    }

    /// Writes the chunks of one chunk row: `data` is `[d, h, w]` C order at global corner `lo`, on the chunk
    /// grid, `d` the row's depth; `keep[cy * nx + cx]` says whether villa writes chunk `(cy, cx)` of the row.
    /// A kept chunk that is all zero is still not stored. Returns the chunks stored
    pub fn write_row(&self, lo: [usize; 3], dims: [usize; 3], data: &[u8], keep: &[bool]) -> Result<usize> {
        let c = self.chunk;
        ensure!(data.len() == dims.iter().product::<usize>(), "row size");
        ensure!(lo.iter().all(|&v| v % c == 0), "row corner {lo:?} off the chunk grid");
        ensure!(dims[0] == c.min(self.shape[0] - lo[0]), "row depth {} for chunk {c}", dims[0]);
        let (ny, nx) = (dims[1].div_ceil(c), dims[2].div_ceil(c));
        ensure!(keep.len() == ny * nx, "{} keep flags for {ny} x {nx} chunks", keep.len());
        let stored: Vec<bool> = (0..ny * nx)
            .into_par_iter()
            .filter(|&i| keep[i])
            .map(|i| {
                let (cy, cx) = (i / nx, i % nx);
                let ext = [dims[0], c.min(dims[1] - cy * c), c.min(dims[2] - cx * c)];
                // a chunk at the array's end is stored full size, padded with the fill value
                let mut buf = vec![0u8; c * c * c];
                let mut any = false;
                for z in 0..ext[0] {
                    for y in 0..ext[1] {
                        let s = (z * dims[1] + cy * c + y) * dims[2] + cx * c;
                        let row = &data[s..s + ext[2]];
                        any |= row.iter().any(|&v| v != 0);
                        buf[(z * c + y) * c..][..ext[2]].copy_from_slice(row);
                    }
                }
                if !any {
                    return Ok(false);
                }
                let key = format!("0.{}.{}.{}", lo[0] / c, lo[1] / c + cy, lo[2] / c + cx);
                let bytes = zstd::bulk::compress(&buf, 1).wrap_err("zstd")?;
                let path = self.dir.join(key);
                std::fs::write(&path, bytes).wrap_err_with(|| format!("write {}", path.display()))?;
                Ok(true)
            })
            .collect::<Result<_>>()?;
        Ok(stored.iter().filter(|&&w| w).count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row's kept, non-zero chunks land under villa's keys, full size, readable back; skipped and
    /// all-zero chunks are absent
    #[test]
    fn row_chunks_round_trip() {
        let dir = std::env::temp_dir().join(format!("inkrs-nnunet-out-{}", std::process::id()));
        let w = Writer::create(&dir, [5, 6, 7], 4, true).unwrap();
        // the last row of a 5-deep volume is 1 deep; 6 x 7 is 2 x 2 chunks, the far ones short
        let dims = [1, 6, 7];
        let mut data = vec![0u8; 42];
        data[0] = 9; // chunk (0, 0)
        data[6] = 7; // chunk (0, 1), x = 6
        data[5 * 7 + 6] = 3; // chunk (1, 1), y = 5, x = 6
        let n = w.write_row([4, 0, 0], dims, &data, &[true, true, false, true]).unwrap();
        assert_eq!(n, 3);
        let read = |k: &str| zstd::bulk::decompress(&std::fs::read(dir.join(k)).unwrap(), 64).unwrap();
        let c01 = read("0.1.0.1");
        assert_eq!((c01.len(), c01[2], c01.iter().map(|&v| v as u32).sum::<u32>()), (64, 7, 7));
        assert_eq!(read("0.1.1.1")[4 + 2], 3);
        assert!(!dir.join("0.1.1.0").exists());
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(".zarray")).unwrap()).unwrap();
        assert_eq!(meta["shape"], serde_json::json!([1, 5, 6, 7]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `--overwrite` replaces an earlier prediction of this writer only: villa's own output (blosc) and a
    /// directory holding anything besides chunks are refused
    #[test]
    fn overwrite_replaces_only_an_earlier_prediction() {
        let root = std::env::temp_dir().join(format!("inkrs-nnunet-overwrite-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let pred = root.join("pred.zarr");
        let w = Writer::create(&pred, [4, 4, 4], 4, false).unwrap();
        w.write_row([0, 0, 0], [4, 4, 4], &[9; 64], &[true]).unwrap();
        Writer::create(&pred, [4, 4, 4], 4, true).expect("an earlier prediction is replaced");

        let villa = root.join("villa.zarr");
        std::fs::create_dir_all(&villa).unwrap();
        let meta = serde_json::json!({"zarr_format": 2, "shape": [1, 4, 4, 4], "dtype": "|u1",
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1}, "dimension_separator": "."});
        std::fs::write(villa.join(".zarray"), meta.to_string()).unwrap();
        // another tool's array laid out as this writer's, with no mark
        let lookalike = root.join("lookalike.zarr");
        Writer::create(&lookalike, [4, 4, 4], 4, false)
            .unwrap()
            .write_row([0, 0, 0], [4, 4, 4], &[9; 64], &[true])
            .unwrap();
        std::fs::remove_file(lookalike.join(".zattrs")).unwrap();
        let w = Writer::create(&pred, [4, 4, 4], 4, true).unwrap();
        w.write_row([0, 0, 0], [4, 4, 4], &[9; 64], &[true]).unwrap();
        std::fs::write(pred.join("m7.json"), b"{}").unwrap();
        for path in [&villa, &lookalike, &pred] {
            let e = Writer::create(path, [4, 4, 4], 4, true).err().unwrap_or_else(|| panic!("{}", path.display()));
            assert!(format!("{e:#}").contains("not a prediction this command wrote"), "{}: {e:#}", path.display());
            // unsupported, so villa runs PyTorch, which replaces any output
            assert!(crate::error::is_unsupported(&e), "{}: {e:#}", path.display());
        }
        assert!(villa.join(".zarray").exists() && lookalike.join("0.0.0.0").exists() && pred.join("m7.json").exists());
        // a model folder is an error, not unsupported, so villa never falls back onto replacing it
        let model = root.join("surface_m7_nnunet");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("m7.json"), b"{}").unwrap();
        let e = Writer::create(&model, [4, 4, 4], 4, true).err().expect("a model folder");
        assert!(!crate::error::is_unsupported(&e) && model.join("m7.json").exists(), "{e:#}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
