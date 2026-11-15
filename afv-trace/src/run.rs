//! The command line and the run (`cli.run`): validation, the work directory, the stages and the export
//!
//! Flags, defaults, choices, validation messages and exit codes are the reference's, so the executable
//! can replace `avf-spline-generator` in a venv. The work directory differs: fields live in memory, and
//! the checkpoint is one finished curve file per family instead of a part file and an fsync per 16
//! seeds. `run.json` carries a `native` key, so neither tool resumes the other's work directory

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::Instant;

use clap::error::ErrorKind;
use clap::{Arg, ArgAction, Command, CommandFactory, Parser, ValueEnum};
use color_eyre::eyre::{Result, bail, eyre};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::afv::{ExportOptions, MinLength, Summary, export_afv};
use crate::axes::{Lapack, direction_field, find_lapack};
use crate::fields::{FamilyFields, catalogue, centrality, presence};
use crate::finish::{FinishCtx, NativeCache, NativeSampler};
use crate::input::Predictions;
use crate::kernels::{Coverage, Fields, RawCurve, Scratch, trace_seed};
use crate::log;
use crate::vol::{EDGE, Shape, Vol};

/// The reference release whose interface and output this reproduces
pub const REFERENCE_VERSION: &str = "0.2.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ChannelOrder {
    #[value(name = "VH")]
    Vh,
    #[value(name = "HV")]
    Hv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Family {
    #[value(name = "both")]
    Both,
    #[value(name = "V")]
    V,
    #[value(name = "H")]
    H,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Device {
    #[value(name = "cpu")]
    Cpu,
    #[value(name = "cuda")]
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Storage {
    #[value(name = "dense")]
    Dense,
    #[value(name = "compressed")]
    Compressed,
}

/// Trace H/V probability volumes into AFV splines.
#[derive(Parser, Debug)]
#[command(
    name = "avf-spline-generator",
    disable_version_flag = true,
    infer_long_args = true,
    args_override_self = true,
    allow_negative_numbers = true
)]
pub struct Args {
    /// Native (2,Z,Y,X) NPY or Zarr
    #[arg(long)]
    pub predictions: PathBuf,
    /// Array key inside a native Zarr group
    #[arg(long)]
    pub array: Option<String>,
    #[arg(long, value_enum, default_value = "VH")]
    pub channel_order: ChannelOrder,
    /// Optional prepooled 2x array
    #[arg(long)]
    pub tracing_predictions: Option<PathBuf>,
    /// Array key inside the prepooled Zarr group
    #[arg(long)]
    pub tracing_array: Option<String>,
    /// Isotropic native voxel size
    #[arg(long)]
    pub voxel_size_um: f64,
    /// Destination .afv
    #[arg(long)]
    pub output: PathBuf,
    /// Persistent scratch directory
    #[arg(long)]
    pub work_dir: PathBuf,
    /// Resume the same immutable inputs
    #[arg(long)]
    pub resume: bool,
    #[arg(long, value_enum, default_value = "both")]
    pub family: Family,
    /// AFV export cutoff (inclusive)
    #[arg(long)]
    pub min_length_mm: Option<f64>,
    // Set, not clap's default Append: a repeated three-value option replaces the earlier one, as in argparse
    #[arg(long, num_args = 3, action = ArgAction::Set, value_names = ["X", "Y", "Z"], default_values_t = [0i64, 0, 0])]
    pub origin_xyz: Vec<i64>,
    /// Full native domain; required for a nonzero crop origin
    #[arg(long, num_args = 3, action = ArgAction::Set, value_names = ["Z", "Y", "X"])]
    pub volume_shape_zyx: Option<Vec<i64>>,
    /// Optional AFV scan identifier
    #[arg(long)]
    pub coordinate_space: Option<String>,
    /// Centrality backend; tracing always runs on CPU
    #[arg(long, value_enum, default_value = "cpu")]
    pub device: Device,
    /// Visible CUDA device index
    #[arg(long, default_value_t = 0)]
    pub cuda_device: i64,
    #[arg(long)]
    pub threads: Option<i64>,
    #[arg(long, default_value_t = 128)]
    pub native_cache_mib: i64,
    /// Intermediate fields: memory-mapped NPY or sparse lossless Zarr
    #[arg(long, value_enum, default_value = "dense")]
    pub storage: Storage,
    /// Compressed-mode tracing cache; minimum 16 MiB (default: 1024)
    #[arg(long, default_value_t = 1024)]
    pub field_cache_mib: i64,
    /// Print the version and exit
    #[arg(long)]
    version: bool,
}

/// The outcome of argparse's left-to-right pass, replayed before clap parses
#[derive(Debug)]
pub enum Prescan {
    /// A help action came first
    Help,
    /// A version action came first
    Version,
    /// An option error came before any action
    Error(clap::Error),
    /// No action: the arguments for clap, single values written as `--option=value` so that clap reads a
    /// value such as `-.5` as argparse does, not as a flag
    Args(Vec<OsString>),
}

/// Replay argparse's left-to-right pass over `args` (without the program name)
///
/// Long options match by unique prefix, each option takes its value count, and its values are converted
/// as they are read, so an ambiguous prefix or a missing or bad value before a help or version action is
/// an error, as in argparse. Unknown options and stray values only fail at argparse's end, so they are
/// passed on for clap to reject
pub fn prescan(args: &[OsString]) -> Prescan {
    let mut command = Args::command();
    command.build();
    let longs: Vec<&str> = command.get_arguments().filter_map(Arg::get_long).collect();
    let error = |kind, message: String| Prescan::Error(Args::command().error(kind, message));
    let mut out: Vec<OsString> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let original = &args[i];
        let token = original.to_string_lossy();
        i += 1;
        if token == "--" {
            out.extend(args[i - 1..].iter().cloned());
            break;
        }
        if let Some(cluster) = token.strip_prefix("-h") {
            // a short-flag cluster: -h, -hh, -hx (the rest after a non-h is left over); an `=` attaches an
            // explicit value to -h, which argparse rejects
            match cluster.trim_start_matches('h').starts_with('=') {
                true => return error(ErrorKind::TooManyValues, "argument -h/--help: ignored explicit argument".into()),
                false => return Prescan::Help,
            }
        }
        let Some(body) = token.strip_prefix("--").filter(|b| !b.is_empty()) else {
            out.push(original.clone());
            continue;
        };
        let (name, explicit) = match body.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (body, None),
        };
        let candidates: Vec<&str> = match longs.iter().find(|&&l| l == name) {
            Some(&l) => vec![l],
            None => longs.iter().copied().filter(|l| l.starts_with(name)).collect(),
        };
        // a token that names no option is left for clap to reject, as argparse leaves it to its end; one with a
        // space would be a stray value to argparse, which ends the same way
        let long = match candidates[..] {
            [] => {
                out.push(original.clone());
                continue;
            }
            [long] => long,
            _ => {
                let options = candidates.iter().map(|c| format!("--{c}")).collect::<Vec<_>>().join(", ");
                return error(ErrorKind::UnknownArgument, format!("ambiguous option: --{name} could match {options}"));
            }
        };
        let arg = command.get_arguments().find(|a| a.get_long() == Some(long)).expect("a listed option");
        if !arg.get_action().takes_values() {
            if let Some(value) = explicit {
                return error(
                    ErrorKind::TooManyValues,
                    format!("argument --{long}: ignored explicit argument '{value}'"),
                );
            }
            match long {
                "help" => return Prescan::Help,
                "version" => return Prescan::Version,
                _ => out.push(format!("--{long}").into()),
            }
            continue;
        }
        let count = arg.get_num_args().map_or(1, |r| r.min_values());
        let values: Vec<OsString> = match explicit {
            Some(_) if count == 1 => {
                // keep the value's own bytes: only the option name is text
                let bytes = original.as_encoded_bytes();
                let at = bytes.iter().position(|&b| b == b'=').expect("an explicit value") + 1;
                // SAFETY: splitting just after an ASCII '=' keeps a valid encoded-bytes boundary
                vec![unsafe { OsString::from_encoded_bytes_unchecked(bytes[at..].to_vec()) }]
            }
            Some(_) => {
                return error(ErrorKind::WrongNumberOfValues, format!("argument --{long}: expected {count} arguments"));
            }
            None => {
                let values: Vec<OsString> =
                    args[i..].iter().take_while(|v| is_value(&v.to_string_lossy())).take(count).cloned().collect();
                if values.len() < count {
                    let message = format!("argument --{long}: expected {count} argument(s)");
                    return error(ErrorKind::WrongNumberOfValues, message);
                }
                i += count;
                values
            }
        };
        // Python's int() and float() accept surrounding whitespace and single underscores between digits
        let values: Vec<OsString> = match NUMERIC_OPTIONS.contains(&long) {
            true => values.into_iter().map(|v| py_number(&v.to_string_lossy()).map_or(v, OsString::from)).collect(),
            false => values,
        };
        let option = |value: &OsString| {
            let mut joined = OsString::from(format!("--{long}="));
            joined.push(value);
            joined
        };
        // the option's own value parser decides, through a command that holds only this option
        let probe = Command::new("avf-spline-generator").arg(arg.clone().required(false));
        let probe_args: Vec<OsString> = match &values[..] {
            [value] => vec![option(value)],
            _ => std::iter::once(OsString::from(format!("--{long}"))).chain(values.iter().cloned()).collect(),
        };
        let argv = std::iter::once(OsString::from("avf-spline-generator")).chain(probe_args.iter().cloned());
        if let Err(e) = probe.allow_negative_numbers(true).try_get_matches_from(argv) {
            return Prescan::Error(e);
        }
        out.extend(probe_args);
    }
    Prescan::Args(out)
}

/// The options argparse converts with `int` or `float`
const NUMERIC_OPTIONS: [&str; 8] = [
    "voxel-size-um",
    "min-length-mm",
    "origin-xyz",
    "volume-shape-zyx",
    "cuda-device",
    "threads",
    "native-cache-mib",
    "field-cache-mib",
];

/// A numeric argument in the form Rust parses, when Python's `int` or `float` would read it differently:
/// whitespace stripped and underscores between digits removed. `None` leaves the text as it is, for the
/// option's parser to accept or reject
fn py_number(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    let digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
    let grouped = bytes.iter().enumerate().all(|(i, &b)| b != b'_' || (i > 0 && digit(i - 1) && digit(i + 1)));
    if !grouped {
        return None;
    }
    (trimmed.len() != text.len() || trimmed.contains('_')).then(|| trimmed.replace('_', ""))
}

/// Whether argparse reads `token` as a value rather than an option: no leading dash, a lone dash, a
/// negative number (the parser has no option that looks like one) or text with a space
fn is_value(token: &str) -> bool {
    let Some(rest) = token.strip_prefix('-') else { return true };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let negative = digits(rest)
        || rest.split_once('.').is_some_and(|(whole, frac)| whole.bytes().all(|b| b.is_ascii_digit()) && digits(frac));
    rest.is_empty() || negative || token.contains(' ')
}

/// Numba's thread ceiling: `NUMBA_NUM_THREADS`, else the CPUs this process may run on
fn numba_max_threads() -> i64 {
    if let Some(n) = std::env::var("NUMBA_NUM_THREADS").ok().and_then(|v| v.trim().parse().ok()) {
        return n;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get() as i64)
}

/// `Path.resolve()` without `strict`: symlinks of the existing prefix resolved, the rest appended
fn resolve(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    let mut rest = Vec::new();
    let mut base = path.as_path();
    loop {
        if let Ok(real) = fs::canonicalize(base) {
            let mut out = real;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return Ok(normalize(&out));
        }
        let Some(name) = base.file_name() else { return Ok(normalize(&path)) };
        rest.push(name.to_os_string());
        base = base.parent().unwrap_or(Path::new("/"));
    }
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// `storage.file_signature`: sha256 over each file's relative path, size and mtime in Python's
/// `sorted(Path)` order, which compares path components rather than whole strings
fn file_signature(path: &Path) -> Result<String> {
    let root = fs::canonicalize(path)?;
    let mut digest = Sha256::new();
    let mut entries: Vec<(Vec<String>, PathBuf)> = Vec::new();
    if root.is_file() {
        let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        entries.push((vec![name], root.clone()));
    } else {
        walk(&root, &mut Vec::new(), &mut entries)?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
    }
    for (parts, file) in entries {
        let meta = fs::metadata(&file)?;
        let mtime_ns = meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128;
        digest.update(format!("{}\0{}\0{}\n", parts.join("/"), meta.len(), mtime_ns).as_bytes());
    }
    Ok(digest.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `rglob("*")` keeping files: symlinks to files count, symlinked directories are not entered
fn walk(dir: &Path, prefix: &mut Vec<String>, out: &mut Vec<(Vec<String>, PathBuf)>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type()?;
        prefix.push(name);
        if kind.is_dir() {
            walk(&entry.path(), prefix, out)?;
        } else if entry.path().is_file() {
            out.push((prefix.clone(), entry.path()));
        }
        prefix.pop();
    }
    Ok(())
}

/// `save_json`: indented JSON, newline, fsync, rename
fn save_json(path: &Path, value: &Value) -> Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    let mut f = File::create(&temporary)?;
    f.write_all(crate::pyfmt::py_json_indent(value).as_bytes())?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

/// An exclusive, non-blocking lock on `work/.lock` held for the run (`FileLock(timeout=0)`)
struct WorkLock(#[allow(dead_code)] File);

impl WorkLock {
    fn acquire(path: &Path) -> Result<Self> {
        let f = OpenOptions::new().read(true).write(true).create(true).truncate(true).mode(0o644).open(path)?;
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            bail!("The file lock '{}' could not be acquired.", path.display());
        }
        Ok(Self(f))
    }
}

/// Run the whole pipeline: the reference's `run(args)`
pub fn run(args: Args) -> Result<Summary> {
    let min_mm = args.min_length_mm.map_or(MinLength::Default, MinLength::Given);
    if !args.voxel_size_um.is_finite()
        || args.voxel_size_um <= 0.0
        || !min_mm.value().is_finite()
        || min_mm.value() < 0.0
    {
        bail!("Voxel size must be positive and minimum length nonnegative");
    }
    if args.origin_xyz.iter().min().copied().unwrap_or(0) < 0 || args.native_cache_mib < 1 || args.cuda_device < 0 {
        bail!("Invalid origin, cache size or CUDA device");
    }
    let max_threads = numba_max_threads();
    let threads = args.threads.unwrap_or(8.min(max_threads));
    if threads < 1 || threads > max_threads {
        bail!("The number of threads must be between 1 and {max_threads}");
    }
    if args.field_cache_mib < 16 {
        bail!("--field-cache-mib must be at least 16");
    }
    if args.output.exists() {
        bail!("Output already exists: {}", args.output.display());
    }
    let order = match args.channel_order {
        ChannelOrder::Vh => "VH",
        ChannelOrder::Hv => "HV",
    };
    let native = Predictions::open(&args.predictions, args.array.as_deref(), order)?;
    if args.tracing_array.is_some() && args.tracing_predictions.is_none() {
        bail!("--tracing-array requires --tracing-predictions");
    }
    let supplied = match &args.tracing_predictions {
        Some(p) => {
            let s = Predictions::open(p, args.tracing_array.as_deref(), order)?;
            if s.shape != native.shape.map(|n| n / 2) {
                bail!("Prepooled tracing shape must equal native shape // 2");
            }
            Some(s)
        }
        None => {
            if !native.dtype.is_float() {
                log!(
                    "Pooling quantized uint8 inputs. Supply prepooled predictions to preserve \
                     pooling-before-quantization values."
                );
            }
            None
        }
    };
    if native.shape.iter().min().copied().unwrap_or(0) < 8 {
        bail!("Native shape must be at least (8,8,8)");
    }
    if native.shape.iter().any(|n| n % 2 == 1) {
        log!("The final unpaired voxel plane on odd-sized axes is excluded from tracing");
    }
    if args.origin_xyz.iter().any(|&v| v != 0) && args.volume_shape_zyx.is_none() {
        bail!("A nonzero origin requires --volume-shape-zyx");
    }
    let domain: Vec<i64> = args.volume_shape_zyx.clone().unwrap_or_else(|| native.shape.map(|n| n as i64).to_vec());
    let origin_zyx = [args.origin_xyz[2], args.origin_xyz[1], args.origin_xyz[0]];
    if (0..3).any(|a| native.shape[a] as i64 + origin_zyx[a] > domain[a]) {
        bail!("Prediction crop lies outside the declared native volume");
    }
    let mut paths = vec![native.path.clone()];
    if let Some(s) = &supplied {
        paths.push(s.path.clone());
    }
    let work = resolve(&args.work_dir)?;
    let output = resolve(&args.output)?;
    for path in &paths {
        let inside = |a: &Path, b: &Path| a != b && a.starts_with(b);
        if *path == output || *path == work || inside(&work, path) || inside(path, &work) {
            bail!("Inputs must be separate from the work directory and output");
        }
        if path.is_dir() && inside(&output, path) {
            bail!("Output cannot be inside an input Zarr store");
        }
    }
    if args.device == Device::Cuda {
        // centrality is the only CUDA stage in the reference; this port computes it on the CPU only
        bail!("inkrs-afv-trace computes centrality on the CPU; use --device cpu");
    }
    fs::create_dir_all(&work)?;
    let _lock = WorkLock::acquire(&work.join(".lock"))?;
    let family_name = match args.family {
        Family::Both => "both",
        Family::V => "V",
        Family::H => "H",
    };
    let signatures = paths.iter().map(|p| file_signature(p)).collect::<Result<Vec<_>>>()?;
    let config = json!({
        "version": REFERENCE_VERSION,
        "predictions": native.path.to_string_lossy(),
        "array": args.array,
        "tracing_predictions": supplied.as_ref().map(|s| s.path.to_string_lossy().into_owned()),
        "tracing_array": args.tracing_array,
        "channel_order": order,
        "shape": native.shape,
        "device": "cpu",
        "storage": match args.storage { Storage::Dense => "dense", Storage::Compressed => "compressed" },
        "family": family_name,
        "voxel_size_um": args.voxel_size_um,
        "origin_xyz": args.origin_xyz,
        "volume_shape_zyx": domain,
        "coordinate_space": args.coordinate_space,
        "input_signatures": signatures,
        "native": {"tool": "inkrs-afv-trace", "version": env!("CARGO_PKG_VERSION"), "checkpoint": 1},
    });
    let manifest = work.join("run.json");
    if manifest.exists() {
        if !args.resume {
            bail!("Work directory already contains a run; use --resume");
        }
        let saved: Value = serde_json::from_str(&fs::read_to_string(&manifest)?)?;
        if saved.get("native").is_none() {
            bail!("Work directory holds a reference avf-spline-generator run; use a new work directory");
        }
        if saved != config {
            bail!("Inputs or tracing settings changed; use a new work directory");
        }
    } else {
        if args.resume {
            bail!("No run to resume in this work directory");
        }
        for entry in fs::read_dir(&work)? {
            if entry?.file_name() != ".lock" {
                bail!("A new run requires an empty work directory");
            }
        }
        save_json(&manifest, &config)?;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build()?;
    let families: Vec<usize> = match args.family {
        Family::Both => vec![1, 2],
        Family::V => vec![1],
        Family::H => vec![2],
    };
    let pending: Vec<usize> = families.iter().copied().filter(|&f| !family_complete(&work, f)).collect();
    if !pending.is_empty() {
        let cache_bytes = (args.native_cache_mib + args.field_cache_mib) as usize * 1024 * 1024;
        pool.install(|| trace_families(&native, supplied.as_ref(), &work, &pending, threads as usize, cache_bytes))?;
    }
    let started = Instant::now();
    let opt = ExportOptions {
        voxel_um: args.voxel_size_um,
        origin_xyz: [args.origin_xyz[0], args.origin_xyz[1], args.origin_xyz[2]],
        volume_shape: [domain[0], domain[1], domain[2]],
        minimum_mm: min_mm,
        coordinate_space: args.coordinate_space.as_deref(),
    };
    let curves = families.iter().flat_map(|&f| CurveReader::open(&work, f).unwrap_or_else(CurveReader::failed));
    let summary = export_afv(curves, &args.output, &opt)?;
    log!("Export: {:.1} s", started.elapsed().as_secs_f64());
    log!("Wrote {} fibres ({} points) to {}", summary.fiber_count, summary.point_count, output.display());
    Ok(summary)
}

fn family_dir(work: &Path, family: usize) -> PathBuf {
    work.join(format!("family-{family}"))
}

fn family_complete(work: &Path, family: usize) -> bool {
    let progress = family_dir(work, family).join("progress.json");
    let Ok(text) = fs::read_to_string(progress) else { return false };
    serde_json::from_str::<Value>(&text).is_ok_and(|v| v["complete"] == json!(true))
}

/// The pooled tracing predictions, both channels (`prepare.prepare_tracing`)
fn load_tracing(native: &Predictions, supplied: Option<&Predictions>) -> Result<[Vol<u8>; 2]> {
    let shape = native.shape.map(|n| n / 2);
    // tiles are the supplied array's chunks so each is decoded once; the values do not depend on tiling
    let tile = supplied.and_then(Predictions::chunk_shape).unwrap_or([64, 64, 64]);
    let tgrid = [0, 1, 2].map(|a| shape[a].div_ceil(tile[a]));
    let mut out = [Vol::new(shape, 0u8), Vol::new(shape, 0u8)];
    let brick_rows = out[0].grid()[0];
    let slab = (tile[0] / EDGE).max(1);
    let mut bz = 0;
    while bz < brick_rows {
        let rows = bz..(bz + slab).min(brick_rows);
        let (z0, z1) = (rows.start * EDGE, (rows.end * EDGE).min(shape[0]));
        let tz = (z0 / tile[0])..z1.div_ceil(tile[0]);
        let jobs: Vec<Shape> =
            tz.flat_map(|z| (0..tgrid[1]).flat_map(move |y| (0..tgrid[2]).map(move |x| [z, y, x]))).collect();
        let tiles: Vec<Option<(Shape, Shape, Vec<u8>)>> = jobs
            .par_iter()
            .map(|&t| {
                let lo = [0, 1, 2].map(|a| t[a] * tile[a]);
                let hi = [0, 1, 2].map(|a| (lo[a] + tile[a]).min(shape[a]));
                let data = match supplied {
                    Some(s) => s.read_u8(lo, hi)?,
                    None => pool_block(native, lo, hi)?,
                };
                let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
                Ok(data.iter().any(|&v| v != 0).then_some((lo, dims, data)))
            })
            .collect::<Result<_>>()?;
        let index = |t: Shape| ((t[0] - z0 / tile[0]) * tgrid[1] + t[1]) * tgrid[2] + t[2];
        for (ch, vol) in out.iter_mut().enumerate() {
            vol.fill_brick_rows(rows.clone(), |lo, dims, dst| {
                let mut any = false;
                let t0 = [0, 1, 2].map(|a| lo[a] / tile[a]);
                let t1 = [0, 1, 2].map(|a| (lo[a] + dims[a] - 1) / tile[a]);
                for tz in t0[0]..=t1[0] {
                    for ty in t0[1]..=t1[1] {
                        for tx in t0[2]..=t1[2] {
                            let Some((tlo, tdims, data)) = &tiles[index([tz, ty, tx])] else { continue };
                            let plane = tdims.iter().product::<usize>();
                            let a0 = [0, 1, 2].map(|a| lo[a].max(tlo[a]));
                            let a1 = [0, 1, 2].map(|a| (lo[a] + dims[a]).min(tlo[a] + tdims[a]));
                            for z in a0[0]..a1[0] {
                                for y in a0[1]..a1[1] {
                                    let src = ch * plane + ((z - tlo[0]) * tdims[1] + y - tlo[1]) * tdims[2];
                                    let dst_at = ((z - lo[0]) * dims[1] + y - lo[1]) * dims[2];
                                    for x in a0[2]..a1[2] {
                                        let v = data[src + x - tlo[2]];
                                        dst[dst_at + x - lo[2]] = v;
                                        any |= v != 0;
                                    }
                                }
                            }
                        }
                    }
                }
                any
            });
        }
        bz = rows.end;
    }
    Ok(out)
}

/// 2×2×2 average pooling of the native predictions in float32, eight additions in corner order, then
/// `rint` and clip to uint8
fn pool_block(native: &Predictions, lo: Shape, hi: Shape) -> Result<Vec<u8>> {
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
    let nd = dims.map(|n| 2 * n);
    let block = native.read_f32(lo.map(|v| 2 * v), hi.map(|v| 2 * v))?;
    let nplane = nd.iter().product::<usize>();
    let plane = dims.iter().product::<usize>();
    let scale = native.dtype.is_float();
    let mut out = vec![0u8; 2 * plane];
    for ch in 0..2 {
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                for x in 0..dims[2] {
                    let mut acc = 0.0f32;
                    for c in 0..8 {
                        let (dz, dy, dx) = (c >> 2 & 1, c >> 1 & 1, c & 1);
                        acc += block[ch * nplane + ((2 * z + dz) * nd[1] + 2 * y + dy) * nd[2] + 2 * x + dx];
                    }
                    let mut v = acc / 8.0;
                    if scale {
                        v *= 255.0;
                    }
                    out[ch * plane + (z * dims[1] + y) * dims[2] + x] = v.round_ties_even().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// Raw curves handed from the tracing thread to the finishing threads at a time
const FINISH_BATCH: usize = 2048;

fn trace_families(
    native: &Predictions,
    supplied: Option<&Predictions>,
    work: &Path,
    families: &[usize],
    threads: usize,
    cache_bytes: usize,
) -> Result<()> {
    let t = Instant::now();
    log!("Preparing tracing predictions");
    let tracing = load_tracing(native, supplied)?;
    let resident = tracing.iter().map(Vol::resident_bytes).sum::<usize>();
    log!("Tracing predictions: {:.1} s, {:.2} GiB resident", t.elapsed().as_secs_f64(), gib(resident));
    let t = Instant::now();
    let lapack = Lapack::init(&find_lapack()?)?;
    let (nx, ny) = direction_field(&tracing, lapack)?;
    release_freed();
    debug_dump(work, "nx", &nx)?;
    debug_dump(work, "ny", &ny)?;
    log!("Direction field: {:.1} s", t.elapsed().as_secs_f64());
    debug_mem("direction field", &[("tracing", resident), ("nx", nx.resident_bytes()), ("ny", ny.resident_bytes())]);
    let cache = NativeCache::new(native, cache_bytes);
    for &family in families {
        trace_family(&tracing, &nx, &ny, &cache, work, family, threads)?;
        release_freed();
    }
    Ok(())
}

/// With `INKRS_AFV_DEBUG_FIELDS` set, write a field densely as raw little-endian `NAME.raw` for comparison
/// with the reference's work arrays; only sensible for small crops
fn debug_dump<T: Copy + PartialEq + Send + Sync + ToLe>(dir: &Path, name: &str, vol: &Vol<T>) -> Result<()> {
    if std::env::var_os("INKRS_AFV_DEBUG_FIELDS").is_none() {
        return Ok(());
    }
    let dense = vol.read_block([0, 0, 0], vol.shape());
    let bytes: Vec<u8> = dense.iter().flat_map(|v| v.to_le()).collect();
    fs::write(dir.join(format!("{name}.raw")), bytes)?;
    Ok(())
}

/// Little-endian bytes of a field element, for debug dumps
trait ToLe {
    fn to_le(self) -> Vec<u8>;
}

impl ToLe for u8 {
    fn to_le(self) -> Vec<u8> {
        vec![self]
    }
}

impl ToLe for u16 {
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

/// Hand the heap pages freed since the last call back to the system
///
/// Each stage frees what the next does not read (the direction field's and centrality's tile results, the seed
/// field, a finished family's presence, centrality and coverage), but glibc keeps freed memory in the arena of
/// the thread that freed it and only trims the top of each heap. The next stage allocates from other threads'
/// arenas (coverage grows on the tracing thread, the next family's fields on the pool's), so without this the
/// freed bytes stay resident beside the new ones: on a 0826 band the H family peaked 1.7 GiB above V with the
/// same fields
fn release_freed() {
    // SAFETY: malloc_trim only returns free pages to the kernel; it touches no live allocation
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// With `INKRS_AFV_DEBUG_MEM` set, log named resident sizes and the process RSS
fn debug_mem(stage: &str, parts: &[(&str, usize)]) {
    if std::env::var_os("INKRS_AFV_DEBUG_MEM").is_none() {
        return;
    }
    let rss =
        fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok());
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let parts: Vec<String> = parts.iter().map(|(k, v)| format!("{k} {:.2}", gib(*v))).collect();
    log!("mem {stage}: {} GiB; rss {:.2} GiB", parts.join(", "), gib(rss.unwrap_or(0) * page));
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn trace_family(
    tracing: &[Vol<u8>; 2],
    nx: &Vol<u8>,
    ny: &Vol<u8>,
    cache: &NativeCache,
    work: &Path,
    family: usize,
    threads: usize,
) -> Result<()> {
    let label = if family == 1 { "V" } else { "H" };
    let folder = family_dir(work, family);
    fs::create_dir_all(&folder)?;
    let t = Instant::now();
    let FamilyFields { pres, cent, seeds } = centrality(presence(tracing, family), nx, ny);
    log!("{label} centrality: {:.1} s", t.elapsed().as_secs_f64());
    debug_mem(
        label,
        &[("pres", pres.resident_bytes()), ("cent", cent.resident_bytes()), ("seeds", seeds.resident_bytes())],
    );
    let t = Instant::now();
    let cat = catalogue(&seeds, &cent);
    debug_dump(&folder, "presence", &pres)?;
    debug_dump(&folder, "centrality", &cent)?;
    debug_dump(&folder, "seeds", &seeds)?;
    if std::env::var_os("INKRS_AFV_DEBUG_FIELDS").is_some() {
        let bytes: Vec<u8> = cat.coords.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(folder.join("coordinates.i4"), bytes)?;
    }
    drop(seeds);
    release_freed();
    log!("{label} seeds: {} in {:.1} s", cat.coords.len(), t.elapsed().as_secs_f64());
    let shape = pres.shape();
    let fields = Fields { pres: &pres, cent: &cent, nx, ny };
    let ctx = FinishCtx { tracing, pres: &pres, family };
    let finish_pool = rayon::ThreadPoolBuilder::new().num_threads(threads.saturating_sub(1).max(1)).build()?;
    let path = folder.join("curves.bin");
    let mut writer = CurveWriter::create(&path)?;
    // INKRS_AFV_DEBUG_RAW=1 also keeps the raw curves (family-N/raw.bin, same format) to compare with the
    // reference's part files stage by stage
    let mut raw_writer = match std::env::var_os("INKRS_AFV_DEBUG_RAW") {
        Some(_) => Some(CurveWriter::create(&folder.join("raw.bin"))?),
        None => None,
    };
    let t = Instant::now();
    let total = cat.coords.len();
    let mut pieces = 0;
    let produce = |tx: &SyncSender<Vec<RawCurve>>| -> Result<usize> {
        let mut cov = Coverage::new(shape);
        let mut scratch = Scratch::default();
        let mut batch = Vec::new();
        let mut count = 0;
        let mut last_log = Instant::now();
        for r in 0..cat.starts.len() - 1 {
            let (start, end) = (cat.starts[r], cat.starts[r + 1]);
            let primary_end = start + cat.primary[r];
            for (i, &seed) in cat.coords.iter().enumerate().take(end).skip(start) {
                trace_seed(&fields, &mut cov, &mut scratch, seed, i < primary_end, &mut batch)?;
                if batch.len() >= FINISH_BATCH {
                    count += batch.len();
                    // a closed channel means finishing failed; its error is the one reported
                    if tx.send(std::mem::take(&mut batch)).is_err() {
                        return Ok(count);
                    }
                }
                if last_log.elapsed().as_secs_f64() >= 10.0 {
                    log!("{label} seeds: {}/{total}; {} raw curves", i + 1, count + batch.len());
                    last_log = Instant::now();
                }
            }
        }
        count += batch.len();
        if !batch.is_empty() {
            let _ = tx.send(batch);
        }
        let cov_bytes = [&cov.cnt, &cov.nx, &cov.ny].map(Vol::resident_bytes);
        debug_mem(label, &[("coverage", cov_bytes.iter().sum()), ("cache", cache.bytes()), ("catalogue", 12 * total)]);
        Ok(count)
    };
    let consume = |batch: Vec<RawCurve>| -> Result<()> {
        if let Some(w) = raw_writer.as_mut() {
            for curve in &batch {
                w.push(&curve.pts)?;
            }
        }
        for curve in finish_batch(&finish_pool, &ctx, cache, &batch)?.iter().flatten() {
            writer.push(curve)?;
            pieces += 1;
        }
        Ok(())
    };
    let raw = pipeline(2, produce, consume)?;
    writer.finish()?;
    if let Some(w) = raw_writer {
        w.finish()?;
    }
    log!(
        "{label} traced: {raw} raw curves, {pieces} finished pieces in {:.1} s ({} native chunks decoded)",
        t.elapsed().as_secs_f64(),
        cache.decoded()
    );
    save_json(&folder.join("progress.json"), &json!({"complete": true, "raw_curves": raw, "curves": pieces}))?;
    Ok(())
}

/// Run `produce` on its own thread, handing what it sends to `consume` through a channel of `capacity`
/// items; returns the producer's result, or the first error of either side
///
/// The receiver is dropped as soon as `consume` fails, so a producer blocked on a full channel gets a send
/// error and returns instead of holding the scope open. A producer error closes the channel from its side
fn pipeline<T: Send, R: Send>(
    capacity: usize,
    produce: impl FnOnce(&SyncSender<T>) -> Result<R> + Send,
    mut consume: impl FnMut(T) -> Result<()>,
) -> Result<R> {
    let (tx, rx) = sync_channel(capacity);
    std::thread::scope(|scope| {
        let producer = scope.spawn(move || produce(&tx));
        let consumed = rx.into_iter().try_for_each(&mut consume);
        let produced = producer.join().map_err(|_| eyre!("tracing thread panicked"))?;
        consumed?;
        produced
    })
}

/// Finish one batch on the finishing pool, results in input order; work is handed out in native-chunk
/// order of each curve's first point so threads share decoded chunks
fn finish_batch(
    pool: &rayon::ThreadPool,
    ctx: &FinishCtx,
    cache: &NativeCache,
    batch: &[RawCurve],
) -> Result<Vec<Vec<Vec<[f32; 3]>>>> {
    let mut order: Vec<usize> = (0..batch.len()).collect();
    order.sort_by_key(|&i| {
        let p = batch[i].pts[0];
        let v = |c: f32| (c.max(0.0) * 2.0 + 0.5) as usize;
        cache.key(v(p[2]), v(p[1]), v(p[0]))
    });
    let done: Vec<(usize, Vec<Vec<[f32; 3]>>)> = pool.install(|| {
        order
            .par_iter()
            .map_init(|| NativeSampler::new(cache), |s, &i| ctx.finish(&batch[i].pts, s).map(|p| (i, p)))
            .collect::<Result<_>>()
    })?;
    let mut out = vec![Vec::new(); batch.len()];
    for (i, p) in done {
        out[i] = p;
    }
    Ok(out)
}

const MAGIC: &[u8; 8] = b"AFVTCRV1";

/// The per-family checkpoint: finished curves in export order, tracing-space float32 x, y, z
struct CurveWriter {
    path: PathBuf,
    temporary: PathBuf,
    out: BufWriter<File>,
}

impl CurveWriter {
    fn create(path: &Path) -> Result<Self> {
        let temporary = path.with_extension("bin.tmp");
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&temporary)?);
        out.write_all(MAGIC)?;
        Ok(Self { path: path.to_path_buf(), temporary, out })
    }

    fn push(&mut self, curve: &[[f32; 3]]) -> Result<()> {
        self.out.write_all(&(curve.len() as u32).to_le_bytes())?;
        for p in curve {
            for v in p {
                self.out.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        let f = self.out.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        fs::rename(&self.temporary, &self.path)?;
        Ok(())
    }
}

/// Reads one family's finished curves back for the export
enum CurveReader {
    Open { family: usize, input: BufReader<File> },
    Failed(Option<color_eyre::Report>),
}

impl CurveReader {
    fn open(work: &Path, family: usize) -> Result<Self> {
        if !family_complete(work, family) {
            bail!("Cannot export an unfinished tracing run");
        }
        let mut input = BufReader::with_capacity(1 << 20, File::open(family_dir(work, family).join("curves.bin"))?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("Invalid curve file for family {family}");
        }
        Ok(Self::Open { family, input })
    }

    fn failed(e: color_eyre::Report) -> Self {
        Self::Failed(Some(e))
    }
}

impl Iterator for CurveReader {
    type Item = Result<(usize, Vec<[f32; 3]>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let (family, input) = match self {
            Self::Failed(e) => return e.take().map(Err),
            Self::Open { family, input } => (*family, input),
        };
        let mut len = [0u8; 4];
        match input.read_exact(&mut len) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e.into())),
        }
        let n = u32::from_le_bytes(len) as usize;
        let mut bytes = vec![0u8; n * 12];
        if let Err(e) = input.read_exact(&mut bytes) {
            return Some(Err(e.into()));
        }
        let pts = bytes.as_chunks::<4>().0.as_chunks::<3>().0.iter().map(|p| p.map(f32::from_le_bytes)).collect();
        Some(Ok((family, pts)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;
    use std::time::Duration;

    use super::*;

    #[test]
    fn pipeline_returns_a_consumer_error_while_the_producer_still_sends() {
        let (done_tx, done_rx) = channel();
        std::thread::spawn(move || {
            let produce = |tx: &SyncSender<usize>| -> Result<usize> {
                let mut sent = 0;
                for i in 0..1000 {
                    if tx.send(i).is_err() {
                        break;
                    }
                    sent += 1;
                }
                Ok(sent)
            };
            let consume = |_: usize| -> Result<()> { bail!("disk full") };
            let _ = done_tx.send(pipeline(2, produce, consume).map_err(|e| e.to_string()));
        });
        let result = done_rx.recv_timeout(Duration::from_secs(10)).expect("pipeline deadlocked");
        assert_eq!(result, Err("disk full".to_string()));
    }

    /// The prescan outcome for one argument line: "help", "version", "error" or "args"
    fn action(line: &str) -> &'static str {
        let args: Vec<OsString> = line.split_whitespace().map(OsString::from).collect();
        match prescan(&args) {
            Prescan::Help => "help",
            Prescan::Version => "version",
            Prescan::Error(_) => "error",
            Prescan::Args(_) => "args",
        }
    }

    /// Parse one argument line as main does: prescan, then clap
    fn parse(line: &str) -> Args {
        let args: Vec<OsString> = line.split_whitespace().map(OsString::from).collect();
        let Prescan::Args(args) = prescan(&args) else { panic!("{line}: no clap parse") };
        Args::try_parse_from(std::iter::once(OsString::from("x")).chain(args)).expect("parses")
    }

    #[test]
    fn early_action_follows_argparse_order_and_abbreviations() {
        assert_eq!(action("--vers"), "version");
        assert_eq!(action("--ve --help"), "version");
        assert_eq!(action("--help --version"), "help");
        assert_eq!(action("--threads 4 -h"), "help");
        assert_eq!(action("--origin-xyz -1 0 0 --version"), "version");
        assert_eq!(action("--voxel-size-um -.5 --version"), "version");
        // unknown options and stray values fail only at the end of argparse's pass
        assert_eq!(action("--bogus 3 --version"), "version");
        assert_eq!(action("-hh --threads x"), "help");
        assert_eq!(action("-hx --family X"), "help");
        assert_eq!(action("--threads 1_0 --help"), "help");
        assert_eq!(action("--voxel-size-um 1_0.5 --version"), "version");
        assert_eq!(action("-- --version"), "args");
        assert_eq!(action("--threads 4"), "args");
    }

    #[test]
    fn early_action_reports_option_errors_before_the_action() {
        for line in [
            "--threads x --help",
            "--threads --help",
            "--origin-xyz 1 --help",
            "--version=1 --help",
            "--v --version",
            "--family X --version",
            "--resume=1 --version",
            "--origin-xyz=1 --version",
            // argparse's negative-number pattern has no exponent form, so `-1e-3` reads as an option
            "--min-length-mm -1e-3 --version",
            "--family=X_Y --version",
            "--threads=x_y --help",
            "-h=bad --version",
            "-hh=x",
            "--threads _1 --help",
            "--threads 1__0 --help",
        ] {
            assert_eq!(action(line), "error", "{line}");
        }
    }

    #[test]
    fn values_with_spaces_are_checked_before_the_action() {
        for argv in [["--family=X Y", "--version"], ["--threads=x y", "--help"], ["--version=1 2", "--help"]] {
            let args: Vec<OsString> = argv.iter().map(OsString::from).collect();
            assert!(matches!(prescan(&args), Prescan::Error(_)), "{argv:?}");
        }
    }

    #[test]
    fn repeated_options_keep_the_last_value() {
        let args = parse(
            "--predictions p --voxel-size-um 1 --output o --work-dir w --threads 2 --threads 3 \
             --origin-xyz 1 2 3 --origin-xyz 4 5 6 --resume --resume --volume-shape-zyx 9 9 9 \
             --volume-shape-zyx 1 2 3",
        );
        assert_eq!(args.threads, Some(3));
        assert_eq!(args.origin_xyz, [4, 5, 6]);
        assert_eq!(args.volume_shape_zyx, Some(vec![1, 2, 3]));
        assert!(args.resume);
    }

    #[test]
    fn values_that_look_like_flags_parse_as_argparse_reads_them() {
        let args = parse("--pred p --voxel-size-um -.5 --output o --work-dir w --min-length-mm -2 --origin-xyz -1 0 0");
        assert_eq!(args.voxel_size_um, -0.5);
        assert_eq!(args.min_length_mm, Some(-2.0));
        assert_eq!(args.origin_xyz, [-1, 0, 0]);
        let mut args: Vec<OsString> = "--pred p --voxel-size-um 1_0.5 --output o --work-dir w --threads"
            .split_whitespace()
            .map(OsString::from)
            .collect();
        args.push(" 1_2 ".into());
        args.push("--family=V".into());
        let Prescan::Args(args) = prescan(&args) else { panic!("no clap parse") };
        let args = Args::try_parse_from(std::iter::once(OsString::from("x")).chain(args)).expect("parses");
        assert_eq!((args.voxel_size_um, args.threads, args.family), (10.5, Some(12), Family::V));
        assert_eq!(args.predictions, PathBuf::from("p"));
    }

    #[test]
    fn pipeline_reports_a_producer_error_after_draining() {
        let mut seen = Vec::new();
        let produce = |tx: &SyncSender<usize>| -> Result<()> {
            tx.send(1)?;
            tx.send(2)?;
            bail!("trace failed")
        };
        let result = pipeline(2, produce, |v| {
            seen.push(v);
            Ok(())
        });
        assert_eq!(result.map_err(|e| e.to_string()), Err("trace failed".to_string()));
        assert_eq!(seen, [1, 2]);
    }
}
