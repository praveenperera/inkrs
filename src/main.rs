//! inkrs: GPU-resident ink inference for villa's flat ink readers (`hybrid_3d2d-seed42`, `reader-v2`)
//! and for Hecate 9.6 um (`inkrs hecate`) and the v8-in family (`inkrs v8in`)

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};

#[cfg(not(target_os = "macos"))]
use inkrs::conv::ConvBackend;
use inkrs::dnn::Pick;
#[cfg(not(target_os = "macos"))]
use inkrs::dnn::algo_cache;
use inkrs::engine::{self, MapSpec, Options};
use inkrs::fast::{self, ReaderPrecision};
use inkrs::hecate;
#[cfg(not(target_os = "macos"))]
use inkrs::model::NormArith;
use inkrs::sched::{DEPTH, Edge, select_layers};
use inkrs::tta::Tta;
use inkrs::v8in;
#[cfg(not(target_os = "macos"))]
use inkrs::{checks, probe};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// `check-norm --arith`: whose normaliser the fixture holds
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum NormChoice {
    Villa,
    Fastink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Direction {
    Forward,
    Reverse,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BlendMode {
    Auto,
    Constant,
    Gaussian,
    Hann,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum HecatePrecision {
    Fp32,
    Bf16,
}

impl HecatePrecision {
    fn with_tf32(self, no_tf32: bool) -> hecate::Precision {
        match self {
            Self::Fp32 => hecate::Precision::Fp32 { tf32: !no_tf32 },
            Self::Bf16 => hecate::Precision::Bf16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum V8Convention {
    Release,
    #[value(name = "306b")]
    Ahn306b,
}

impl V8Convention {
    fn convention(self) -> v8in::Convention {
        match self {
            Self::Release => v8in::Convention::Release,
            Self::Ahn306b => v8in::Convention::Ahn306b,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Algo {
    /// Time every cuDNN algorithm per layer at plan time
    Find,
    /// cuDNN's heuristic ranking
    Heuristic,
}

/// inkrs-only engine options
#[derive(Args, Clone, Debug)]
struct EngineArgs {
    /// How cuDNN algorithms are chosen
    #[arg(long, value_enum, default_value_t = Algo::Find)]
    algo: Algo,
    /// Largest cuDNN workspace an algorithm may use, in MiB
    #[arg(long, default_value_t = 1024)]
    max_workspace_mib: usize,
    /// Which library runs the 2D convs: `inkrs` (own kernels, summed in a fixed order) or `cudnn`;
    /// defaults to `INKRS_CONV`, else by the GPU: `inkrs` on compute capability 12.x, `cudnn` on
    /// others (the sm75 tier always runs `cudnn`), so default maps from a 12.x GPU and another GPU
    /// can differ by cuDNN's rounding, within the parity contract
    #[cfg(not(target_os = "macos"))]
    #[arg(long)]
    conv: Option<ConvBackend>,
    /// Download all logits at the end and fail on any non-finite value
    #[arg(long)]
    check_finite: bool,
    /// Also write each map's coverage mask from the blend kernel (uint8 TIFF, 1 where the
    /// weight sum is above villa's 1e-6 cut): `sweep` under OUT/coverage (OUT/pass<k>/coverage),
    /// `infer` next to each map as `<map>.coverage.tif`. Gates compare it with villa's schedule
    #[arg(long)]
    coverage: bool,
    /// TIFF writer threads
    #[arg(long, default_value_t = 4)]
    writers: usize,
    /// Test-time augmentation variants: 1 (villa's single read) or 8, the PHerc. 0306B recipe's dihedral
    /// TTA. 8 reads each map twice, as stored and turned a quarter, runs every tile as four mirrored
    /// copies and averages them, and writes float32 `.npy` maps instead of uint8 TIFFs (`infer`'s output
    /// must end in `.npy`; `sweep` names its maps `.npy`). It runs no low-resolution occupancy scan, as
    /// the recipe, and takes neither --coverage nor --check-finite
    #[arg(long, default_value_t = 1)]
    tta: u32,
    /// Write the run's timings and cuDNN choices to this JSON file
    #[arg(long)]
    report: Option<PathBuf>,
}

impl Algo {
    fn pick(self) -> Pick {
        match self {
            Algo::Find => Pick::Find,
            Algo::Heuristic => Pick::Heuristic,
        }
    }
}

impl EngineArgs {
    fn options(
        &self,
        stride: usize,
        batch: usize,
        precision: ReaderPrecision,
        fast: bool,
        started: Instant,
    ) -> Result<Options> {
        Ok(Options {
            started,
            stride,
            batch,
            precision,
            fast,
            pick: self.algo.pick(),
            max_workspace: self.max_workspace_mib << 20,
            #[cfg(not(target_os = "macos"))]
            conv: self.conv.map_or_else(ConvBackend::from_env, |c| Ok(Some(c)))?,
            check_finite: self.check_finite,
            coverage: self.coverage,
            writers: self.writers,
            tta: Tta::from_count(self.tta)?,
        })
    }
}

#[derive(Subcommand)]
enum Command {
    /// Integration preflight: open the GPU, check one kernel, cuDNN conv3d/conv2d and the
    /// GEMM against CPU references, and print cuDNN's algorithm choices
    Probe {
        /// Batch size for the timed model shapes
        #[arg(long, default_value_t = 64)]
        batch: i32,
    },
    /// The GPU normaliser on every tile of a fixture from scripts/norm_fixture.py, against
    /// villa's normalised patches rounded to FP16
    CheckNorm {
        fixture: PathBuf,
        input_zarr: PathBuf,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        /// The fixture's tile edge: 128, or 384 for Reader V3
        #[arg(long, default_value_t = 128)]
        tile: usize,
        /// The fixture's normaliser: villa's numpy one, or the 306B recipe's torch one (`fastink`),
        /// which the 8-fold TTA reads replay
        #[arg(long, value_enum, default_value_t = NormChoice::Villa)]
        arith: NormChoice,
    },
    /// P0: one forward on the tiles of a JSON list, dumping every layer as `.npy`
    P0 {
        input_zarr: PathBuf,
        /// The `.safetensors` export
        weights: PathBuf,
        #[arg(long)]
        tiles: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = Algo::Find)]
        algo: Algo,
    },
    /// villa `infer` for one map (or two with --direction both): the subset of its flags
    /// this reader uses; any other flag is rejected
    ///
    /// The export's sidecar picks the engine. A v8-in family export (v8-in, A-hn seeds 0-2) writes
    /// a float32 `.npy` probability map stitched by the export's map convention (`--convention`
    /// overrides it); `--layer-start` is its first layer of 24 (default the central 24), and
    /// --overlap, --blend-mode and --layer-end do not apply; --coverage writes `<map>.coverage.npy`
    /// (uint8, 1 where an admitted tile covers the pixel)
    Infer {
        input_zarr: PathBuf,
        /// villa `.pth` (with an export in the weights directory) or the `.safetensors` export
        checkpoint: PathBuf,
        output_tiff: PathBuf,
        /// Tile overlap; the stride is villa's `round(edge * (1 - overlap))` for the reader's tile edge
        /// (128, or 384 for Reader V3). Default 0.5 (stride 64, or 192). Expert flag: any other value
        /// leaves villa parity; on ink labels, reader-v2 lost up to
        /// 0.007 AUC at stride 72, 0.025 at 80, 0.015 at 96 and 0.047 at 128 (FAST.md)
        #[arg(long, conflicts_with = "stride")]
        overlap: Option<f64>,
        /// Tile stride in px, 1 to the tile edge, instead of --overlap. Expert flag: any stride but half
        /// the edge leaves villa parity; on ink labels, reader-v2 lost up to
        /// 0.007 AUC at stride 72, 0.025 at 80, 0.015 at 96 and 0.047 at 128 (FAST.md)
        #[arg(long)]
        stride: Option<usize>,
        /// The reader's numerics: fp16, the default (villa's FP16 autocast), already within 2/255 of
        /// villa's float32 maps on MPS; or fp32, Metal only, for a closer match (within 1/255 on an
        /// M4 Pro) at 1.2-1.3x the run time
        #[arg(long, value_enum, default_value_t = ReaderPrecision::Fp16)]
        precision: ReaderPrecision,
        /// The reader's fast preset; refused for every reader, since no stride above 64 kept label
        /// AUC within 0.005 (FAST.md)
        #[arg(long, conflicts_with_all = ["overlap", "stride"])]
        fast: bool,
        #[arg(long, value_enum, default_value_t = BlendMode::Auto)]
        blend_mode: BlendMode,
        /// Tiles per batch; default 1 (villa's), 16 for a v8-in export
        #[arg(long)]
        batch_size: Option<usize>,
        /// v8-in only: how tiles are admitted and stitched, `release` (ink8um's predict_stack) or
        /// `306b` (the 306B A-hn runner); default the export's own
        #[arg(long, value_enum)]
        convention: Option<V8Convention>,
        /// v8-in only: run the map this many times in one process, the model and volume resident,
        /// writing the last run's map and every run's timings in --report
        #[arg(long, default_value_t = 1)]
        passes: usize,
        #[arg(long, value_enum, default_value_t = Direction::Forward)]
        direction: Direction,
        #[arg(long, allow_negative_numbers = true)]
        layer_start: Option<i64>,
        #[arg(long, allow_negative_numbers = true)]
        layer_end: Option<i64>,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// A depth sweep with the volume and model resident: one map per depth and direction,
    /// written as `OUT/ink/<name>__<reader>__L<Z>s<S>[_reverse].tif`, where `<reader>` is the
    /// export's reader name
    Sweep {
        input_zarr: PathBuf,
        #[arg(long)]
        ckpt: PathBuf,
        /// Depths `A..B`, inclusive; layers start at `Z/2 + d - 8`
        #[arg(long, allow_hyphen_values = true)]
        depths: String,
        #[arg(long, value_enum, default_value_t = Direction::Both)]
        direction: Direction,
        /// Tile overlap; default 0.5 (stride 64 for 128 px readers, 192 for Reader V3). Expert flag: any
        /// other value leaves villa parity; on ink labels, reader-v2 lost up to
        /// 0.007 AUC at stride 72, 0.025 at 80, 0.015 at 96 and 0.047 at 128 (FAST.md)
        #[arg(long, conflicts_with = "stride")]
        overlap: Option<f64>,
        /// Tile stride in px, 1 to the tile edge. Expert flag: any stride but half the edge leaves villa
        /// parity; on ink labels, reader-v2 lost up to
        /// 0.007 AUC at stride 72, 0.025 at 80, 0.015 at 96 and 0.047 at 128 (FAST.md)
        #[arg(long)]
        stride: Option<usize>,
        /// The reader's numerics: fp16, the default (villa's FP16 autocast), already within 2/255 of
        /// villa's float32 maps on MPS; or fp32, Metal only, for a closer match (within 1/255 on an
        /// M4 Pro) at 1.2-1.3x the run time
        #[arg(long, value_enum, default_value_t = ReaderPrecision::Fp16)]
        precision: ReaderPrecision,
        /// The reader's fast preset; refused for every reader, since no stride above 64 kept label
        /// AUC within 0.005 (FAST.md)
        #[arg(long, conflicts_with_all = ["overlap", "stride"])]
        fast: bool,
        /// Tiles per batch; default 64 for 128-pixel readers and 8 for Reader V3's 384-pixel
        /// tiles, about the same memory
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        out: PathBuf,
        /// Volume name in the output file names; defaults to the zarr's stem
        #[arg(long)]
        name: Option<String>,
        /// Run the whole sweep this many times in one process, the model resident; pass 1
        /// writes OUT/ink, pass k >= 2 writes OUT/pass<k>/ink. Each later pass reads, schedules
        /// and uploads the volume again, so it times a further segment at steady state
        #[arg(long, default_value_t = 1)]
        passes: usize,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Both blend kernels on the GPU against a CPU reference, at strides from 1 to the tile edge, on
    /// sides the strides do not divide and with skipped tiles; the gate for any-stride blending
    CheckBlend,
    /// Print every model's `--fast` preset as JSON; a reader that is not listed has none (FAST.md)
    FastPresets,
    /// Whole-volume 3D ink with a villa 3D U-Net checkpoint (`ink_3d_dino_guided` v3 and others of its
    /// family): villa's native patch plan over a zarr region, written as a uint8 zarr
    Ink3d {
        /// The `.safetensors` export from scripts/export_unet3d.py, its `.json` sidecar beside it
        #[arg(long)]
        checkpoint: PathBuf,
        /// uint8 `[Z, Y, X]` zarr v2 array, or a group whose array `0` is read
        #[arg(long)]
        input: PathBuf,
        /// Output zarr group (array `0`, uint8 probabilities times 255)
        #[arg(long)]
        output: PathBuf,
        /// Region `z0,y0,x0,z1,y1,x1` on the chunk grid; default the whole volume
        #[arg(long, value_delimiter = ',')]
        bbox: Option<Vec<usize>>,
        /// villa's --overlap
        #[arg(long, default_value_t = 0.5)]
        overlap: f64,
        /// Output chunk edge
        #[arg(long, default_value_t = 128)]
        chunk: usize,
        /// Cap on the GPU accumulators, GB
        #[arg(long, default_value_t = 8.0)]
        max_accum_gb: f64,
        #[arg(long, value_enum, default_value_t = Algo::Find)]
        algo: Algo,
        #[arg(long)]
        overwrite: bool,
        /// After the run, time this many forwards of the last patch
        #[arg(long, default_value_t = 0)]
        forward_bench: usize,
        /// Also write villa's coverage mask (blend weight > 1e-6) as group `coverage` in the output
        #[arg(long)]
        coverage: bool,
        /// CUDA device ordinal
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// Write the run's timings and cuDNN choices to this JSON file
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// P0-3D: one 3D U-Net patch, each layer's centre cube dumped for scripts/p0_ink3d.py
    P0Ink3d {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        input: PathBuf,
        /// Patch corner `z,y,x`
        #[arg(long, value_delimiter = ',')]
        corner: Vec<usize>,
        /// Centre cube edge kept per tap
        #[arg(long, default_value_t = 64)]
        crop: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = Algo::Find)]
        algo: Algo,
    },
    /// `hecate.py --output` for the 9.6 um release: the 2D probability PNG of the central 16
    /// planes and its JSON, with hecate.py's flags. Writes OUT.png and OUT.json, and with --both
    /// also OUT_reverse.png and OUT_reverse.json from the same resident volume and model
    Hecate {
        /// hecate.py's `.pth` (with an export in the weights directory) or the `.safetensors` export
        #[arg(long)]
        checkpoint: PathBuf,
        /// uint8 `[Z, Y, X]` `.npy`, or a local zarr v2 array or group
        #[arg(long)]
        input: PathBuf,
        /// Array within a zarr group; only `0` is read
        #[arg(long, default_value = "0")]
        array: String,
        /// 2D probability PNG; must not exist
        #[arg(long)]
        output: PathBuf,
        /// The render's actual voxel size, um; must be within 2% of 9.6
        #[arg(long)]
        spacing_um: f64,
        /// Reverse the Z axis before selecting the central planes
        #[arg(long, conflicts_with = "both")]
        reverse: bool,
        /// Write the forward and the reverse map in one run
        #[arg(long)]
        both: bool,
        /// Tiles per forward; default 8, or 32 with --fast
        #[arg(long)]
        batch_size: Option<usize>,
        /// XY stride, 1 to 64; default half the tile, 32, hecate.py's. Expert flag: on ink labels, bf16
        /// lost up to 0.017 AUC at stride 40, 0.023 at 48 and 0.041 at 64 (FAST.md)
        #[arg(long, conflicts_with = "overlap")]
        stride: Option<usize>,
        /// XY tile overlap instead of --stride: stride `round(64 * (1 - overlap))`; 0.5 is the default 32.
        /// Expert flag, with --stride's accuracy cost
        #[arg(long)]
        overlap: Option<f64>,
        /// Numerics; default fp32 (with TF32 convs), hecate.py's default. bf16 at stride 32 lost at most
        /// 0.0006 AUC on ink labels, and is --fast's precision (FAST.md)
        #[arg(long, value_enum)]
        precision: Option<HecatePrecision>,
        /// fp32 without TF32 convs; PyTorch's default, which hecate.py keeps, allows TF32. Not scored
        /// on ink labels
        #[arg(long)]
        no_tf32: bool,
        /// Hecate's fast preset, bf16 at stride 32 and batch 32, which gives up parity with hecate.py's
        /// fp32 default for speed at a measured accuracy cost (FAST.md)
        #[arg(long, conflicts_with_all = ["stride", "overlap", "precision", "no_tf32"])]
        fast: bool,
        #[arg(long, value_enum, default_value_t = Algo::Find)]
        algo: Algo,
        /// Largest cuDNN workspace an algorithm may use, in MiB
        #[arg(long, default_value_t = 1024)]
        max_workspace_mib: usize,
        /// Write the run's timings and cuDNN choices to this JSON file
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// P0-H: one Hecate forward on the tiles of a JSON list of `{y0, x0, reverse}`, dumping every
    /// layer as `.npy`
    P0Hecate {
        input: PathBuf,
        /// The `.safetensors` export (or the `.pth` with an export in the weights directory)
        checkpoint: PathBuf,
        #[arg(long)]
        tiles: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = HecatePrecision::Fp32)]
        precision: HecatePrecision,
        #[arg(long)]
        no_tf32: bool,
        #[arg(long, value_enum, default_value_t = Algo::Find)]
        algo: Algo,
    },
    /// P0 for v8-in: the network on the tiles of a JSON list of `[y0, x0]`, writing each tile's
    /// probability (`prob.npy`, float32 `[n, 64, 64]`), the decoder-side taps (FP16 `.npy`), the run's
    /// settings (`meta.json`), and with --stages the encoder stages (`stats.npy`, `stem.npy`,
    /// `layer1.npy`..`layer4.npy` at a few depths). `INKRS_V8_UNFUSED=1` runs every conv unfused
    #[command(name = "p0-v8in")]
    P0V8in {
        input: PathBuf,
        checkpoint: PathBuf,
        #[arg(long)]
        tiles: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Default the export's own (`map_convention` in its sidecar)
        #[arg(long, value_enum)]
        convention: Option<V8Convention>,
        #[arg(long)]
        layer0: Option<usize>,
        #[arg(long)]
        reverse: bool,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, value_enum, default_value_t = Algo::Find)]
        algo: Algo,
        /// Also dump the encoder stages
        #[arg(long)]
        stages: bool,
    },
}

fn main() -> Result<()> {
    // Windows: before any library loads, so that a load by bare name after the DLL checks never searches the current
    // directory or PATH (src/dll.rs)
    #[cfg(windows)]
    inkrs::dll::restrict_search()?;

    // the driver JIT cache is shared by every CUDA program and evicts its oldest entries at 1 GiB
    // (a busy box sits at that cap), so other programs can evict the fused convs' compiled kernels
    // and bring their first-use compile back; ours goes in the inkrs cache. Set here, before any
    // CUDA call, since the driver reads it once at initialisation, and never by the Python module,
    // which may share a process with torch, whose CUDA is already initialised
    #[cfg(not(target_os = "macos"))]
    if let Some(path) = algo_cache::cuda_cache_path(&|name| std::env::var_os(name)) {
        // the driver's handling of a missing directory is not documented; a failure here leaves it
        // to the driver
        let _ = std::fs::create_dir_all(&path);
        // SAFETY: the start of main, before any other thread exists
        unsafe { std::env::set_var("CUDA_CACHE_PATH", path) };
    }
    let started = Instant::now();
    color_eyre::install()?;
    match Cli::parse().command {
        // the CUDA-only tools, which the Metal backend does not run
        #[cfg(target_os = "macos")]
        Command::Probe { .. }
        | Command::CheckBlend
        | Command::CheckNorm { .. }
        | Command::P0 { .. }
        | Command::P0Hecate { .. }
        | Command::Ink3d { .. }
        | Command::P0Ink3d { .. } => {
            Err(inkrs::error::Unsupported("this command is not available on the Metal backend yet".into()).into())
        }
        #[cfg(not(target_os = "macos"))]
        Command::Probe { batch } => probe::run(batch),
        #[cfg(not(target_os = "macos"))]
        Command::CheckBlend => checks::check_blend(),
        Command::FastPresets => {
            println!("{}", serde_json::to_string_pretty(&fast::presets())?);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        Command::CheckNorm { fixture, input_zarr, batch, tile, arith } => {
            let edge = Edge::from_crop(&[DEPTH, tile, tile]).ok_or_else(|| eyre!("--tile must be 128 or 384"))?;
            let arith = match arith {
                NormChoice::Villa => NormArith::Villa,
                NormChoice::Fastink => NormArith::Fastink,
            };
            checks::check_norm(&fixture, &input_zarr, batch, edge, arith)
        }
        #[cfg(not(target_os = "macos"))]
        Command::P0 { input_zarr, weights, tiles, out, algo } => {
            checks::p0(&input_zarr, &weights, &tiles, &out, algo.pick())
        }
        #[cfg(not(target_os = "macos"))]
        Command::Ink3d {
            checkpoint,
            input,
            output,
            bbox,
            overlap,
            chunk,
            max_accum_gb,
            algo,
            overwrite,
            forward_bench,
            coverage,
            device,
            report,
        } => {
            let bbox =
                bbox.map(|b| <[usize; 6]>::try_from(b).map_err(|_| eyre!("--bbox takes six numbers"))).transpose()?;
            let opts = inkrs::unet3d::engine::Options {
                overlap,
                bbox,
                chunk,
                pick: algo.pick(),
                max_accum_gb,
                overwrite,
                forward_bench,
                coverage,
                device,
            };
            let r = inkrs::unet3d::engine::run(&input, &checkpoint, &output, &opts)?;
            eprintln!(
                "ink3d: {} patches in {:.1} s ({:.4} s/patch; read {:.1} s, write {:.1} s), load {:.1} s, forward {:?} ms",
                r.patches, r.run_s, r.s_per_patch, r.read_s, r.write_s, r.load_s, r.forward_ms
            );
            if let Some(path) = report {
                std::fs::write(&path, serde_json::to_string_pretty(&r)?)?;
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        Command::P0Ink3d { checkpoint, input, corner, crop, out, algo } => {
            let corner = <[usize; 3]>::try_from(corner).map_err(|_| eyre!("--corner takes three numbers"))?;
            inkrs::unet3d::p0::run(&input, &checkpoint, corner, crop, &out, algo.pick())
        }
        #[cfg(not(target_os = "macos"))]
        Command::P0Hecate { input, checkpoint, tiles, out, precision, no_tf32, algo } => {
            ensure!(!(no_tf32 && precision == HecatePrecision::Bf16), "--no-tf32 applies to fp32 only");
            hecate::p0::run(&input, &checkpoint, &tiles, &out, precision.with_tf32(no_tf32), algo.pick())
        }
        Command::P0V8in { input, checkpoint, tiles, out, convention, layer0, reverse, batch_size, algo, stages } => {
            let yx: Vec<[usize; 2]> =
                serde_json::from_slice(&std::fs::read(&tiles).wrap_err_with(|| format!("read {}", tiles.display()))?)?;
            ensure!(!yx.is_empty(), "no tiles in {}", tiles.display());
            std::fs::create_dir_all(&out)?;
            let opts = v8in::engine::Options {
                convention: convention.map(V8Convention::convention),
                layer0,
                reverse,
                batch: batch_size,
                pick: algo.pick(),
                max_workspace: 1 << 30,
            };
            let eng = v8in::engine::Engine::new(&checkpoint, batch_size, algo.pick(), 1 << 30)?;
            let vol = hecate::read_volume(&input)?;
            let p0 = eng.p0(vol.view(), &yx, &opts, stages)?;
            v8in::engine::write_npy_f32(&out.join("prob.npy"), &p0.probs, &[yx.len(), v8in::P, v8in::P])?;
            for (name, shape, bits) in &p0.taps {
                v8in::engine::write_npy(&out.join(format!("{name}.npy")), bytemuck::cast_slice(bits), "<f2", shape)?;
            }
            if stages {
                v8in::engine::write_npy_f32(&out.join("stats.npy"), &p0.stats, &[yx.len(), 2])?;
            }
            std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&p0.meta)? + "\n")?;
            println!("{}", out.display());
            Ok(())
        }
        Command::Hecate {
            checkpoint,
            input,
            array,
            output,
            spacing_um,
            reverse,
            both,
            batch_size,
            stride,
            overlap,
            precision,
            no_tf32,
            fast,
            algo,
            max_workspace_mib,
            report,
        } => {
            ensure!(array == "0", "--array {array:?}: only array 0 of a zarr group is read");
            ensure!(!(no_tf32 && precision == Some(HecatePrecision::Bf16)), "--no-tf32 applies to fp32 only");
            let precision =
                precision.map(|p| p.with_tf32(no_tf32)).or(no_tf32.then_some(hecate::Precision::Fp32 { tf32: false }));
            let settings = fast::hecate_settings(fast, stride, overlap, precision, batch_size)?;
            let mut maps = vec![hecate::engine::MapSpec { reverse, png: output.clone() }];
            if both {
                maps.push(hecate::engine::MapSpec { reverse: true, png: suffixed(&output, "_reverse")? });
            }
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let opts = hecate::engine::Options {
                stride: settings.stride,
                fast,
                batch: settings.batch,
                precision: settings.precision,
                pick: algo.pick(),
                max_workspace: max_workspace_mib << 20,
                spacing_um,
                checkpoint: checkpoint.clone(),
                weights_dir: engine::weights_dir(),
                device: 0,
                started,
            };
            let rep = hecate::engine::run(&input, &checkpoint, &maps, &opts)?;
            let json = serde_json::to_string_pretty(&rep)?;
            if let Some(p) = &report {
                std::fs::write(p, &json).wrap_err_with(|| format!("write {}", p.display()))?;
            }
            eprintln!(
                "hecate {}: {} maps, {} tiles of {} positions in {} batches of {}: total {:.3}s = weights {:.2}s + gpu {:.2}s + plans {:.2}s ({} of {} cached) + read {:.2}s + schedule {:.2}s + upload {:.2}s + run {:.3}s (gpu {:.3}s, {:.0} tiles/s) + write {:.2}s",
                rep.precision,
                maps.len(),
                rep.tiles,
                rep.positions,
                rep.batches,
                rep.batch,
                rep.total_s,
                rep.load_weights_s,
                rep.open_gpu_s,
                rep.plan_s,
                rep.algo_cache_hits,
                rep.algos.len(),
                rep.read_volume_s,
                rep.schedule_s,
                rep.upload_s,
                rep.run_s,
                rep.gpu_s,
                rep.tiles_per_s,
                rep.write_s
            );
            for m in &maps {
                println!("{}", m.png.display());
            }
            Ok(())
        }
        Command::Infer {
            input_zarr,
            checkpoint,
            output_tiff,
            overlap,
            stride,
            precision,
            fast,
            blend_mode,
            batch_size,
            convention,
            passes,
            direction,
            layer_start,
            layer_end,
            engine,
        } => {
            if v8in::is_v8in_export(&checkpoint) {
                // the readers' tiling and engine flags would be ignored here, so they are refused
                ensure!(
                    overlap.is_none() && stride.is_none() && !fast && blend_mode == BlendMode::Auto && engine.tta == 1,
                    "--overlap, --stride, --fast, --blend-mode and --tta do not apply to v8-in; its map convention fixes the tiling and the window"
                );
                ensure!(
                    precision == ReaderPrecision::Fp16,
                    "--precision applies to the villa readers only; v8-in runs its own FP16 numerics"
                );
                #[cfg(not(target_os = "macos"))]
                let conv_set = engine.conv.is_some();
                #[cfg(target_os = "macos")]
                let conv_set = false;
                ensure!(
                    !conv_set && !engine.check_finite,
                    "--conv and --check-finite do not apply to v8-in; its convs always run on cuDNN"
                );
                let layer0 = layer_start
                    .map(usize::try_from)
                    .transpose()
                    .map_err(|_| eyre!("--layer-start must be >= 0 for v8-in"))?;
                ensure!(
                    layer_end.is_none() || layer_end.zip(layer_start).is_some_and(|(e, s)| e == s + v8in::D as i64),
                    "v8-in reads 24 layers: give --layer-start only, or --layer-end = --layer-start + 24"
                );
                return infer_v8in(&V8Run {
                    input: &input_zarr,
                    checkpoint: &checkpoint,
                    output: &output_tiff,
                    convention: convention.map(V8Convention::convention),
                    layer0,
                    direction,
                    batch: batch_size.unwrap_or(16),
                    passes,
                    engine: &engine,
                    started,
                });
            }
            ensure!(convention.is_none() && passes == 1, "--convention and --passes apply to v8-in exports only");
            ensure!(
                engine.tta == 1 || output_tiff.extension().is_some_and(|e| e == "npy"),
                "--tta {} writes a float32 map: the output must end in .npy",
                engine.tta
            );
            let batch_size = batch_size.unwrap_or(1);
            // refused before a checkpoint is hashed or the input read, so the backend is what the error names
            engine::check_precision(precision)?;
            // the export is resolved once (a `.pth` is hashed), and its sidecar fixes the tile edge
            let export = engine::resolve_weights(&checkpoint, &engine::weights_dir())?;
            let patch = engine::export_edge(&export)?.px();
            let stride = reader_stride(&export, patch, overlap, stride, fast)?;
            // villa's auto picks constant at stride >= patch; only the Hann blend is built
            let blend = match blend_mode {
                BlendMode::Auto if stride >= patch => BlendMode::Constant,
                BlendMode::Auto => BlendMode::Hann,
                other => other,
            };
            ensure!(blend == BlendMode::Hann, "--blend-mode {blend:?} is not supported; only hann is");
            let depth = zarr_depth(&input_zarr)?;
            let mut maps = Vec::new();
            for (reverse, wanted) in [(false, direction != Direction::Reverse), (true, direction != Direction::Forward)]
            {
                if !wanted {
                    continue;
                }
                let (first, step) = select_layers(depth, layer_start, layer_end, reverse)?;
                // villa suffixes only the reverse map of a two-direction run
                let path = if reverse && direction == Direction::Both {
                    suffixed(&output_tiff, "_reverse")?
                } else {
                    output_tiff.clone()
                };
                let coverage = if engine.coverage { vec![suffixed(&path, ".coverage")?] } else { Vec::new() };
                maps.push(MapSpec { first, step, paths: vec![path], coverage });
            }
            if let Some(parent) = output_tiff.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let rep = engine::run(
                &input_zarr,
                &export,
                &maps,
                &engine.options(stride, batch_size, precision, fast, started)?,
            )?;
            finish(&rep, engine.report.as_deref())
        }
        Command::Sweep {
            input_zarr,
            ckpt,
            depths,
            direction,
            overlap,
            stride,
            precision,
            fast,
            batch_size,
            out,
            name,
            passes,
            engine,
        } => {
            ensure!(passes >= 1, "--passes must be at least 1");
            // refused before a checkpoint is hashed or the input read, so the backend is what the error names
            engine::check_precision(precision)?;
            // the export is resolved once (a `.pth` is hashed), and its sidecar fixes the tile edge
            let export = engine::resolve_weights(&ckpt, &engine::weights_dir())?;
            let edge = engine::export_edge(&export)?;
            let stride = reader_stride(&export, edge.px(), overlap, stride, fast)?;
            let batch_size = batch_size.unwrap_or(match edge {
                Edge::P128 => 64,
                Edge::P384 => 8,
            });
            let (a, b) = parse_depths(&depths)?;
            let depth = zarr_depth(&input_zarr)?;
            let name = match name {
                Some(n) => n,
                None => input_zarr
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| eyre!("no name in {}", input_zarr.display()))?
                    .to_owned(),
            };
            let inks: Vec<PathBuf> = (1..=passes)
                .map(|k| if k == 1 { out.join("ink") } else { out.join(format!("pass{k}")).join("ink") })
                .collect();
            // coverage masks sit beside each pass's ink/ directory
            let covs: Vec<PathBuf> = if engine.coverage {
                inks.iter().map(|ink| ink.with_file_name("coverage")).collect()
            } else {
                Vec::new()
            };
            for dir in inks.iter().chain(&covs) {
                std::fs::create_dir_all(dir).wrap_err_with(|| format!("create {}", dir.display()))?;
            }
            let reader = engine::export_reader(&export)?;
            let mut maps = Vec::new();
            for d in a..=b {
                let s = depth as i64 / 2 + d - 8;
                ensure!(
                    s >= 0 && s as usize + DEPTH <= depth,
                    "depth {d}: layers {s}..{} outside the volume's {depth}",
                    s + DEPTH as i64
                );
                for (reverse, wanted) in
                    [(false, direction != Direction::Reverse), (true, direction != Direction::Forward)]
                {
                    if !wanted {
                        continue;
                    }
                    let (first, step) = select_layers(depth, Some(s), Some(s + DEPTH as i64), reverse)?;
                    let suffix = if reverse { "_reverse" } else { "" };
                    let ext = if engine.tta == 1 { "tif" } else { "npy" };
                    let file = format!("{name}__{reader}__L{depth}s{s}{suffix}.{ext}");
                    maps.push(MapSpec {
                        first,
                        step,
                        paths: inks.iter().map(|ink| ink.join(&file)).collect(),
                        coverage: covs.iter().map(|c| c.join(&file)).collect(),
                    });
                }
            }
            let rep = engine::run(
                &input_zarr,
                &export,
                &maps,
                &engine.options(stride, batch_size, precision, fast, started)?,
            )?;
            let report = engine.report.clone().unwrap_or_else(|| out.join("inkrs.json"));
            finish(&rep, Some(&report))
        }
    }
}

fn finish(rep: &engine::Report, path: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(rep)?;
    if let Some(p) = path {
        std::fs::write(p, &json).wrap_err_with(|| format!("write {}", p.display()))?;
    }
    eprintln!("2D convs on {} ({})", rep.conv, rep.conv_reason);
    eprintln!(
        "cold start {:.3}s (weights {:.2}s, gpu {:.2}s, plans {:.2}s with {} of {} cached, then pass 1)",
        rep.cold_s,
        rep.load_weights_s,
        rep.open_gpu_s,
        rep.plan_s,
        rep.algo_cache_hits,
        rep.algos.len()
    );
    for (i, p) in rep.passes.iter().enumerate() {
        eprintln!(
            "pass {}: {} maps, {} tiles in {} batches of {}: pass {:.3}s = read {:.2}s + schedule {:.2}s + upload {:.2}s + run {:.3}s (gpu {:.3}s, {:.0} tiles/s)",
            i + 1,
            p.maps,
            p.tiles,
            p.batches,
            rep.batch,
            p.total_s,
            p.read_volume_s,
            p.schedule_s,
            p.upload_s,
            p.run_s,
            p.gpu_s,
            p.tiles_per_s
        );
    }
    Ok(())
}

/// [`fast::reader_stride`] for the resolved export `export`, whose tiles are `patch` px, naming the
/// reader from its sidecar only for `--fast`, where an unreadable sidecar only changes the refusal's
/// message
fn reader_stride(
    export: &Path,
    patch: usize,
    overlap: Option<f64>,
    stride: Option<usize>,
    fast: bool,
) -> Result<usize> {
    let reader = if fast { engine::export_reader(export).ok() } else { None };
    fast::reader_stride(reader.as_deref(), patch, fast, stride, overlap)
}

fn parse_depths(s: &str) -> Result<(i64, i64)> {
    let (a, b) = s.split_once("..").ok_or_else(|| eyre!("--depths {s:?} is not A..B"))?;
    let (a, b): (i64, i64) = (a.trim().parse()?, b.trim().parse()?);
    ensure!(a <= b, "--depths {s:?} is empty");
    Ok((a, b))
}

/// An `infer` call on a v8-in family export
struct V8Run<'a> {
    input: &'a Path,
    checkpoint: &'a Path,
    output: &'a Path,
    convention: Option<v8in::Convention>,
    layer0: Option<usize>,
    direction: Direction,
    batch: usize,
    passes: usize,
    engine: &'a EngineArgs,
    started: Instant,
}

/// `infer` for a v8-in export: one float32 `.npy` map per direction (the reverse one suffixed `_reverse`
/// when both run), from one resident model and volume
fn infer_v8in(r: &V8Run<'_>) -> Result<()> {
    ensure!(r.passes >= 1, "--passes must be at least 1");
    ensure!(
        r.output.extension().is_some_and(|e| e == "npy"),
        "v8-in writes a float32 .npy map; give an output path ending in .npy"
    );
    let pick = r.engine.algo.pick();
    let max_ws = r.engine.max_workspace_mib << 20;
    let eng = v8in::engine::Engine::new(r.checkpoint, r.batch, pick, max_ws)?;
    let t = Instant::now();
    let vol = hecate::read_volume(r.input)?;
    let read_s = t.elapsed().as_secs_f64();
    if let Some(parent) = r.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut maps = Vec::new();
    for reverse in [false, true] {
        if (reverse && r.direction == Direction::Forward) || (!reverse && r.direction == Direction::Reverse) {
            continue;
        }
        let path = if reverse && r.direction == Direction::Both {
            suffixed(r.output, "_reverse")?
        } else {
            r.output.to_path_buf()
        };
        let opts = v8in::engine::Options {
            convention: r.convention,
            layer0: r.layer0,
            reverse,
            batch: r.batch,
            pick,
            max_workspace: max_ws,
        };
        let mut runs = Vec::new();
        let mut last = None;
        for _ in 0..r.passes {
            let mut out = eng.run(vol.view(), &opts)?;
            out.report.read_volume_s = read_s;
            eprintln!(
                "v8in {} ({}{}): {} tiles in {} batches of {}: run {:.3}s, gpu {:.3}s, {:.2} ms/tile; plans {:.2}s",
                out.report.reader,
                out.report.convention,
                if reverse { ", reverse" } else { "" },
                out.report.tiles,
                out.report.batches,
                out.report.batch,
                out.report.run_s,
                out.report.gpu_s,
                out.report.ms_per_tile,
                out.report.plan_s
            );
            runs.push(serde_json::json!({"run_s": out.report.run_s, "gpu_s": out.report.gpu_s, "schedule_s": out.report.schedule_s, "upload_s": out.report.upload_s}));
            last = Some(out);
        }
        let out = last.ok_or_else(|| eyre!("no run"))?;
        v8in::engine::write_npy_f32(&path, &out.map, &[out.height, out.width])?;
        if r.engine.coverage {
            let cov = suffixed(&path, ".coverage")?;
            v8in::engine::write_npy(&cov, &out.coverage(), "|u1", &[out.height, out.width])?;
        }
        let mut json = serde_json::to_value(&out.report)?;
        json["runs"] = serde_json::Value::from(runs);
        json["output"] = serde_json::Value::from(path.display().to_string());
        maps.push(json);
        println!("{}", path.display());
    }
    if let Some(p) = &r.engine.report {
        let json = serde_json::json!({
            "checkpoint": r.checkpoint.display().to_string(),
            "input": r.input.display().to_string(),
            "wall_s": r.started.elapsed().as_secs_f64(),
            "maps": maps,
        });
        std::fs::write(p, serde_json::to_string_pretty(&json)? + "\n")
            .wrap_err_with(|| format!("write {}", p.display()))?;
    }
    Ok(())
}

fn suffixed(path: &Path, suffix: &str) -> Result<PathBuf> {
    let stem = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| eyre!("no file name in {}", path.display()))?;
    let ext = path.extension().and_then(|s| s.to_str()).map(|e| format!(".{e}")).unwrap_or_default();
    Ok(path.with_file_name(format!("{stem}{suffix}{ext}")))
}

/// The layer count from the array metadata, before the volume is read
fn zarr_depth(path: &Path) -> Result<usize> {
    let meta = [path.join(".zarray"), path.join("0/.zarray")]
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| eyre!("{}: no zarr array", path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta)?)?;
    let Some(d) = v.get("shape").and_then(|s| s.get(0)).and_then(|d| d.as_u64()) else {
        bail!("{}: no shape", meta.display());
    };
    Ok(d as usize)
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::{Cli, Command};

    // clap's `num_args = N` with a value delimiter rejected the comma form the jobs pass
    #[test]
    fn comma_lists_parse() {
        let cli = Cli::try_parse_from([
            "inkrs",
            "p0-ink3d",
            "--checkpoint",
            "c",
            "--input",
            "i",
            "--out",
            "o",
            "--corner",
            "256,384,384",
        ])
        .expect("p0-ink3d parses");
        assert!(matches!(cli.command, Command::P0Ink3d { ref corner, .. } if corner == &[256, 384, 384]));
        let cli = Cli::try_parse_from([
            "inkrs",
            "ink3d",
            "--checkpoint",
            "c",
            "--input",
            "i",
            "--output",
            "o",
            "--bbox",
            "0,0,0,256,256,256",
        ])
        .expect("ink3d parses");
        assert!(matches!(cli.command, Command::Ink3d { bbox: Some(ref b), .. } if b == &[0, 0, 0, 256, 256, 256]));
    }

    // a variant's doc comment is its help text, and one written above the wrong variant gave ink3d Hecate's
    #[test]
    fn each_command_has_its_own_help() {
        let cli = Cli::command();
        let about = |name: &str| {
            let sub = cli.find_subcommand(name).unwrap_or_else(|| panic!("no {name} subcommand"));
            sub.get_long_about().or(sub.get_about()).map(|a| a.to_string()).unwrap_or_default()
        };

        let ink3d = about("ink3d");
        assert!(ink3d.starts_with("Whole-volume 3D ink"), "ink3d help: {ink3d}");
        assert!(!ink3d.contains("hecate.py"), "ink3d help: {ink3d}");
        let hecate = about("hecate");
        assert!(hecate.starts_with("`hecate.py --output`"), "hecate help: {hecate}");
    }
}
