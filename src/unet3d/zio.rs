//! Zarr v2 I/O for whole-volume 3D inference: reads uint8 boxes of a `[Z, Y, X]` array chunk by chunk,
//! and writes the uint8 prediction array chunk by chunk
//!
//! The reader takes the codecs [`crate::zarr`] takes (raw, zstd, blosc-lz4), either dimension separator,
//! and missing chunks as the fill value. The writer makes a group with array `0`, raw chunks, `/`
//! separators and fill 0, and skips all-zero chunks as villa's `skip_empty` arrays do. villa also writes
//! five downsampled levels; this writer does not. The group's `.zattrs` marks it as an inkrs output, which is
//! what lets `--overwrite` replace it

use std::path::{Component, Path, PathBuf};

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
    /// The path [`Array::open`] was given: the array, or the group holding it as `0`
    pub opened: PathBuf,
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
        let shape = [s[0], s[1], s[2]];
        Ok(Self { opened: path.to_path_buf(), dir, shape, chunks: [c[0], c[1], c[2]], fill, sep, codec })
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

impl Array {
    /// Fails when `output` holds this array or the group it was opened through, or sits inside the array, so an
    /// input that is itself an earlier prediction is never replaced and the output's chunks never land among the
    /// input's. The group counts apart from the array because its `0` may be a symlink to an array elsewhere
    pub fn ensure_apart(&self, output: &Path) -> Result<()> {
        let resolve = |p: &Path| p.canonicalize().wrap_err_with(|| format!("resolve {}", p.display()));
        let (array, opened) = (resolve(&self.dir)?, resolve(&self.opened)?);
        let out = resolve_through_ancestor(&output_path(output)?)?;
        ensure!(
            !array.starts_with(&out) && !opened.starts_with(&out) && !out.starts_with(&array),
            "output {} would replace or write into the input {}",
            output.display(),
            self.opened.display()
        );
        Ok(())
    }
}

/// `path` without a trailing separator, which would make the filesystem follow a symlink it names. An empty
/// path is refused, as it would write into the working directory, and so is a `..` component: it can land on an
/// existing directory, and on Linux creating the path makes the directory before it inside whatever holds it
fn output_path(path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "the output path is empty");
    ensure!(
        !path.components().any(|c| c == Component::ParentDir),
        "output {} has a `..` component; give the path without it",
        path.display()
    );
    Ok(path.components().collect())
}

/// `path` resolved through its nearest existing ancestor, with the missing components after it as given
fn resolve_through_ancestor(path: &Path) -> Result<PathBuf> {
    let mut missing = Vec::new();
    let mut base = path;
    loop {
        let probe = if base.as_os_str().is_empty() { Path::new(".") } else { base };
        if let Ok(found) = probe.canonicalize() {
            return Ok(missing.iter().rev().fold(found, |p, name| p.join(name)));
        }

        let (Some(name), Some(parent)) = (base.file_name(), base.parent()) else {
            bail!("resolve {}", path.display());
        };
        missing.push(name);
        base = parent;
    }
}

/// Makes `path` the empty directory a new prediction is written into. An existing path is replaced only with
/// `overwrite` and only when `is_prediction` recognises it as an earlier prediction of the same writer, so
/// `--overwrite` deletes nothing else; a symlink is never followed or removed. A refusal is an error, never
/// unsupported: unsupported lets a caller such as villa fall back on a writer that replaces whatever the path
/// holds, which would delete exactly what this refused to
pub fn fresh_output_dir(path: &Path, overwrite: bool, is_prediction: impl Fn(&Path) -> bool) -> Result<()> {
    let path = &output_path(path)?;
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            ensure!(
                !meta.file_type().is_symlink(),
                "{} is a symlink, which --overwrite never follows; give the prediction's own path",
                path.display()
            );
            ensure!(overwrite, "{} exists; pass --overwrite", path.display());
            ensure!(
                meta.is_dir() && is_prediction(path),
                "{} is not a prediction this command wrote, so --overwrite will not delete it (outputs of inkrs \
                 before 0.4.0 carry no mark); remove it yourself or give another output",
                path.display()
            );

            std::fs::remove_dir_all(path).wrap_err_with(|| format!("remove {}", path.display()))?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).wrap_err_with(|| format!("inspect {}", path.display())),
    }

    std::fs::create_dir_all(path).wrap_err_with(|| format!("create {}", path.display()))
}

/// Whether `dir` holds only regular files and directories named by `keep`, none of them a symlink
pub fn holds_only(dir: &Path, keep: impl Fn(&str, bool) -> bool) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else { return false };
    entries.into_iter().all(|e| {
        let Ok(e) = e else { return false };
        let Ok(kind) = e.file_type() else { return false };
        !kind.is_symlink() && e.file_name().to_str().is_some_and(|name| keep(name, kind.is_dir()))
    })
}

/// Whether `dir/.zarray` is uint8 zarr v2 metadata with these compressor and separator and `rank` axes
pub fn array_meta_is(dir: &Path, compressor: &serde_json::Value, separator: &str, rank: usize) -> bool {
    let Ok(bytes) = std::fs::read(dir.join(".zarray")) else { return false };
    let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    meta["zarr_format"] == 2
        && meta["dtype"] == "|u1"
        && meta["compressor"] == *compressor
        && meta["dimension_separator"] == separator
        && meta["shape"].as_array().is_some_and(|s| s.len() == rank)
}

/// Writes `dir/.zattrs` naming the inkrs command whose output `dir` is, read back by [`is_marked`]. Metadata
/// alone cannot tell an inkrs prediction from another tool's array of the same layout
pub fn mark(dir: &Path, command: &str) -> Result<()> {
    let attrs = serde_json::json!({"inkrs": {"output": command}});
    let path = dir.join(".zattrs");
    std::fs::write(&path, serde_json::to_string_pretty(&attrs)?).wrap_err_with(|| format!("write {}", path.display()))
}

/// Whether `dir/.zattrs` holds the mark [`mark`] writes for `command`
pub fn is_marked(dir: &Path, command: &str) -> bool {
    let Ok(bytes) = std::fs::read(dir.join(".zattrs")) else { return false };
    serde_json::from_slice::<serde_json::Value>(&bytes).is_ok_and(|attrs| attrs["inkrs"]["output"] == command)
}

/// The uint8 prediction array `0` of a new group, chunk `chunk^3`, raw chunks
pub struct Writer {
    dir: PathBuf,
    pub shape: [usize; 3],
    pub chunk: usize,
}

impl Writer {
    /// The mark of a prediction
    const COMMAND: &str = "ink3d";
    /// The mark of the coverage group inside one, distinct so another prediction saved at that path is not taken
    /// for it
    const COVERAGE: &str = "ink3d-coverage";

    /// Creates `path` with array `0` of `shape`; an existing path is replaced only with `overwrite` and only when it
    /// is an earlier prediction of this writer ([`fresh_output_dir`])
    pub fn create(path: &Path, shape: [usize; 3], chunk: usize, overwrite: bool) -> Result<Self> {
        Self::create_marked(path, shape, chunk, overwrite, Self::COMMAND)
    }

    /// Creates the coverage group of the new prediction at `prediction`
    pub fn create_coverage(prediction: &Path, shape: [usize; 3], chunk: usize) -> Result<Self> {
        Self::create_marked(&prediction.join("coverage"), shape, chunk, false, Self::COVERAGE)
    }

    fn create_marked(path: &Path, shape: [usize; 3], chunk: usize, overwrite: bool, command: &str) -> Result<Self> {
        fresh_output_dir(path, overwrite, |p| Self::is_output(p, Self::COMMAND))?;
        let dir = path.join("0");
        std::fs::create_dir_all(&dir)?;
        std::fs::write(path.join(".zgroup"), "{\"zarr_format\": 2}")?;
        mark(path, command)?;
        let chunks: Vec<usize> = shape.iter().map(|&s| chunk.min(s)).collect();
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "|u1", "fill_value": 0,
            "order": "C", "filters": null, "compressor": null, "dimension_separator": "/",
        });
        std::fs::write(dir.join(".zarray"), serde_json::to_string_pretty(&meta)?)?;
        std::fs::write(dir.join(".zattrs"), "{\"_ARRAY_DIMENSIONS\": [\"z\", \"y\", \"x\"]}")?;
        Ok(Self { dir, shape, chunk })
    }

    /// Whether `path` holds only what this writer makes for an output marked `command`: `.zgroup`, the mark,
    /// array `0` with its metadata and chunks, and for a prediction the coverage group beside it
    fn is_output(path: &Path, command: &str) -> bool {
        let array = path.join("0");
        let names = holds_only(path, |name, dir| match name {
            ".zgroup" | ".zattrs" => !dir,
            "0" => dir,
            "coverage" => dir && command == Self::COMMAND && Self::is_output(&path.join("coverage"), Self::COVERAGE),
            _ => false,
        });
        let chunks = holds_only(&array, |name, dir| match name {
            ".zarray" | ".zattrs" => !dir,
            _ => dir && Self::is_chunk_tree(&array.join(name), name, 2),
        });
        names && chunks && is_marked(path, command) && array_meta_is(&array, &serde_json::Value::Null, "/", 3)
    }

    /// Whether `dir`, named `name`, is one `/`-separated chunk index with `levels` more indices below it, the
    /// last of them chunk files
    fn is_chunk_tree(dir: &Path, name: &str, levels: usize) -> bool {
        let index = |name: &str| name.parse::<u64>().is_ok();
        index(name)
            && holds_only(dir, |child, is_dir| match levels {
                1 => !is_dir && index(child),
                _ => is_dir && Self::is_chunk_tree(&dir.join(child), child, levels - 1),
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An output that is the input group, its array, or a directory above them is refused before anything is
    /// deleted; a sibling or a path under a missing directory is not
    #[test]
    fn output_must_stay_apart_from_the_input() {
        let root = std::env::temp_dir().join(format!("inkrs-zio-apart-{}", std::process::id()));
        let group = root.join("scan.zarr");
        Writer::create(&group, [2, 3, 4], 2, true).unwrap();
        let vol = Array::open(&group).unwrap();

        // an output under missing directories inside the input still lands among its chunks
        let inside = [group.join("0/pred.zarr"), group.join("0/9/9/9"), group.join("0/")];
        for bad in [group.clone(), group.join("0"), root.clone()].into_iter().chain(inside) {
            let e = vol.ensure_apart(&bad).expect_err(&bad.display().to_string());
            assert!(format!("{e:#}").contains("write into the input"), "{}: {e:#}", bad.display());
        }
        for ok in [root.join("pred.zarr"), group.join("pred.zarr"), root.join("missing/pred.zarr")] {
            vol.ensure_apart(&ok).unwrap_or_else(|e| panic!("{}: {e:#}", ok.display()));
        }
        // `work/../scan.zarr` is the input
        for bad in [group.join("0/new/.."), root.join("work/../scan.zarr")] {
            let e = vol.ensure_apart(&bad).expect_err(&bad.display().to_string());
            assert!(!crate::error::is_unsupported(&e) && format!("{e:#}").contains("`..`"), "{e:#}");
        }
        assert!(group.join("0/.zarray").exists(), "the input survives");

        // a group whose array 0 links to an array elsewhere is still the input, as is the linked array
        #[cfg(unix)]
        {
            let linked = root.join("linked.zarr");
            std::fs::create_dir_all(&linked).unwrap();
            std::fs::write(linked.join(".zgroup"), "{\"zarr_format\": 2}").unwrap();
            std::os::unix::fs::symlink(group.join("0"), linked.join("0")).unwrap();
            let vol = Array::open(&linked).unwrap();
            for bad in [linked.clone(), group.join("0")] {
                let e = vol.ensure_apart(&bad).expect_err(&bad.display().to_string());
                assert!(format!("{e:#}").contains("write into the input"), "{}: {e:#}", bad.display());
            }
            vol.ensure_apart(&root.join("pred.zarr")).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    /// `--overwrite` replaces an earlier prediction and nothing else: not a directory holding other files, an
    /// input group, a symlink however it is spelled, another prediction saved as a coverage group, or a path
    /// through `..`. Each refusal is an error, so villa never falls back onto a writer that would replace it
    #[test]
    fn overwrite_replaces_only_an_earlier_prediction() {
        let root = std::env::temp_dir().join(format!("inkrs-zio-overwrite-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let pred = root.join("pred.zarr");
        let w = Writer::create(&pred, [4, 4, 4], 2, false).unwrap();
        w.write_slab([0, 0, 0], [4, 4, 4], &[7; 64]).unwrap();
        Writer::create_coverage(&pred, [4, 4, 4], 2).unwrap();
        let e = Writer::create(&pred, [4, 4, 4], 2, false).err().expect("exists");
        assert!(format!("{e:#}").contains("pass --overwrite"), "{e:#}");
        Writer::create(&pred, [4, 4, 4], 2, true).expect("an earlier prediction is replaced");

        let refused = |path: &Path, why: &str| {
            let e = Writer::create(path, [4, 4, 4], 2, true).err().unwrap_or_else(|| panic!("{}", path.display()));
            assert!(format!("{e:#}").contains(why), "{}: {e:#}", path.display());
            assert!(!crate::error::is_unsupported(&e), "{}: {e:#}", path.display());
        };
        // a directory holding anything else, such as a model folder
        let run = root.join("run");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("m7.json"), b"{}").unwrap();
        refused(&run, "not a prediction this command wrote");
        assert!(run.join("m7.json").exists());
        refused(Path::new(""), "empty");
        // an input group: a level besides 0
        let scan = root.join("scan.zarr");
        Writer::create(&scan, [4, 4, 4], 2, false).unwrap();
        std::fs::create_dir_all(scan.join("1")).unwrap();
        refused(&scan, "not a prediction this command wrote");
        // a path through `..` is refused before anything is made
        refused(&scan.join("0/new/.."), "`..`");
        assert!(!scan.join("0/new").exists());
        // an earlier prediction whose array holds anything besides its chunks, such as a checkpoint
        let stash = root.join("stash.zarr");
        Writer::create(&stash, [4, 4, 4], 2, false).unwrap().write_slab([0, 0, 0], [4, 4, 4], &[7; 64]).unwrap();
        std::fs::write(stash.join("0/model.safetensors"), b"weights").unwrap();
        refused(&stash, "not a prediction this command wrote");
        std::fs::remove_file(stash.join("0/model.safetensors")).unwrap();
        std::fs::write(stash.join("0/0/0/m7.json"), b"{}").unwrap();
        refused(&stash, "not a prediction this command wrote");
        assert!(stash.join("0/0/0/m7.json").exists());
        std::fs::remove_file(stash.join("0/0/0/m7.json")).unwrap();
        // another tool's group laid out as this writer's, with no mark
        std::fs::remove_file(stash.join(".zattrs")).unwrap();
        refused(&stash, "not a prediction this command wrote");
        // a separate prediction saved where a prediction without coverage would keep it
        let result = root.join("result.zarr");
        Writer::create(&result, [4, 4, 4], 2, false).unwrap();
        Writer::create(&result.join("coverage"), [4, 4, 4], 2, false).unwrap();
        refused(&result, "not a prediction this command wrote");
        assert!(result.join("coverage/0/.zarray").exists());
        #[cfg(unix)]
        {
            let link = root.join("link.zarr");
            std::os::unix::fs::symlink(&pred, &link).unwrap();
            refused(&link, "symlink");
            // a trailing separator would otherwise make the filesystem follow the link
            refused(&root.join("link.zarr/"), "symlink");
            assert!(pred.join("0/.zarray").exists(), "the link's target survives");
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
