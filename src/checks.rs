//! GPU gates: the normaliser on every fixture tile (`check-norm`) and the per-layer activation
//! dump the P0 gate compares with PyTorch (`p0`)

use std::io::Write;
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, ensure};
use half::f16;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::conv::ConvBackend;
use crate::dnn::{Dnn, Pick};
use crate::fast::ReaderPrecision;
use crate::gpu::Gpu;
use crate::model::{Ctx, Model, NormArith, TilePrep, logits_buf};
use crate::sched::{DEPTH, DESC, Edge};
use crate::weights::Checkpoint;
use crate::zarr::Volume;

/// One tile: corner and layers
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct TileSpec {
    pub y0: i32,
    pub x0: i32,
    pub first: i32,
    pub step: i32,
}

fn descriptors(tiles: &[TileSpec], vol: &Volume, batch: usize, patch: usize) -> Result<Vec<i32>> {
    let mut desc = Vec::with_capacity(batch * DESC);
    for (i, t) in tiles.iter().enumerate() {
        let last = t.first + 16 * t.step;
        ensure!(
            t.y0 >= 0 && t.x0 >= 0 && t.y0 as usize + patch <= vol.height && t.x0 as usize + patch <= vol.width,
            "tile ({}, {}) outside the volume",
            t.y0,
            t.x0
        );
        ensure!(t.first.min(last) >= 0 && (t.first.max(last) as usize) < vol.depth, "tile layers outside the volume");
        desc.extend_from_slice(&[t.y0, t.x0, t.first, t.step, i as i32, 0, 0, 0]);
    }
    // pad a short batch with copies of the first tile that write no logits
    while desc.len() < batch * DESC {
        desc.extend_from_within(..4);
        desc.extend_from_slice(&[-1, 0, 0, 0]);
    }
    Ok(desc)
}

/// Runs the GPU normaliser on every tile of `fixture` (scripts/norm_fixture.py), tiles of `edge`,
/// and compares each FP16 patch with the reference's normalised float32 patch rounded to FP16, by
/// SHA-256. `arith` must be the fixture's: villa's numpy normaliser, or with `--arith fastink` the
/// 306B recipe's torch one, whose std fallback inkrs does not replay, so its fallback tiles are
/// counted but not gated
pub fn check_norm(fixture: &Path, zarr: &Path, batch: usize, edge: Edge, arith: NormArith) -> Result<()> {
    #[derive(Deserialize)]
    struct Rec {
        #[serde(flatten)]
        tile: TileSpec,
        fallback: bool,
        sha256_f16: String,
        /// the fixture's normaliser; fixtures from before the fastink mode are villa's
        #[serde(default = "villa")]
        arith: String,
    }
    fn villa() -> String {
        "villa".into()
    }
    let recs: Vec<Rec> = std::fs::read_to_string(fixture)
        .wrap_err_with(|| format!("read {}", fixture.display()))?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    ensure!(!recs.is_empty(), "empty fixture");
    let want = match arith {
        NormArith::Villa => "villa",
        NormArith::Fastink => "fastink",
    };
    ensure!(recs.iter().all(|r| r.arith == want), "{} is not a fixture of the {want} normaliser", fixture.display());
    let vol = Volume::read(zarr)?;
    let gpu = Gpu::open()?;
    let vbuf = gpu.upload(&vol.data)?;
    let prep = TilePrep::new(&gpu, batch, edge)?;
    let per_tile = DEPTH * edge.pixels();
    let (mut ok, mut bad, mut bad_fallback, mut fallback) = (0, 0, 0, 0);
    for chunk in recs.chunks(batch) {
        let tiles: Vec<TileSpec> = chunk.iter().map(|r| r.tile).collect();
        let desc = gpu.upload(&descriptors(&tiles, &vol, batch, edge.px())?)?;
        prep.run(&gpu, (&vbuf, vol.height, vol.width), desc.arg::<i32>(), arith)?;
        let x: Vec<u16> = gpu.download_range(&prep.x0, 0, chunk.len() * per_tile)?;
        for (i, r) in chunk.iter().enumerate() {
            let bytes: Vec<u8> = x[i * per_tile..(i + 1) * per_tile].iter().flat_map(|v| v.to_le_bytes()).collect();
            let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            fallback += r.fallback as usize;
            if hex == r.sha256_f16 {
                ok += 1;
            } else {
                bad += 1;
                bad_fallback += r.fallback as usize;
                if bad <= 5 {
                    eprintln!(
                        "mismatch at ({}, {}) layers {}:{} fallback={}",
                        r.tile.y0, r.tile.x0, r.tile.first, r.tile.step, r.fallback
                    );
                }
            }
        }
    }
    println!(
        "check-norm {}: {} tiles ({fallback} MAD fallback), {ok} identical to {want} in FP16, {bad} differ ({bad_fallback} of them fallback), ptx tier {}",
        fixture.display(),
        recs.len(),
        gpu.inner.ptx_tier()
    );
    match arith {
        NormArith::Villa => ensure!(bad == 0, "{bad} normalised tiles differ from villa"),
        NormArith::Fastink => {
            ensure!(
                bad == bad_fallback,
                "{} normalised tiles outside the std fallback differ from fastink",
                bad - bad_fallback
            )
        }
    }
    Ok(())
}

/// Writes `data` as a float32 `.npy` of `shape`
pub fn write_npy(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == data.len(), "npy shape {shape:?} for {} values", data.len());
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let dims = if dims.len() == 1 { format!("{},", dims[0]) } else { dims.join(", ") };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut f =
        std::io::BufWriter::new(std::fs::File::create(path).wrap_err_with(|| format!("create {}", path.display()))?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// One forward on the tiles in `tiles` (a JSON list of `{y0, x0, first, step}`), dumping every
/// tapped activation and the logits as channels-last float32 `.npy` files in `out`, with a
/// `manifest.json`; fails on any non-finite value
pub fn p0(zarr: &Path, weights: &Path, tiles: &Path, out: &Path, pick: Pick) -> Result<()> {
    let tiles: Vec<TileSpec> =
        serde_json::from_slice(&std::fs::read(tiles).wrap_err_with(|| format!("read {}", tiles.display()))?)?;
    ensure!(!tiles.is_empty(), "no tiles");
    let batch = tiles.len();
    let vol = Volume::read(zarr)?;
    let ck = Checkpoint::load(weights, ReaderPrecision::Fp16)?;
    let gpu = Gpu::open()?;
    let dnn = Dnn::new(&gpu)?;
    let ctx = Ctx { gpu: &gpu, dnn: &dnn, pick, max_workspace: 1 << 30 };
    let edge = ck.edge;
    let (conv, why) =
        ConvBackend::resolve(ConvBackend::from_env()?, gpu.inner.compute_capability(), gpu.inner.ptx_tier())?;
    println!("p0: 2D convs on {} ({why})", conv.name());
    let model = Model::new(&ctx, &ck, batch, edge, conv)?;
    let schema = ck.finish()?;
    let vbuf = gpu.upload(&vol.data)?;
    let desc = gpu.upload(&descriptors(&tiles, &vol, batch, edge.px())?)?;
    let logits = logits_buf(&gpu, batch, edge)?;
    model.forward(&ctx, (&vbuf, vol.height, vol.width), desc.arg::<i32>(), &logits)?;
    gpu.sync()?;

    std::fs::create_dir_all(out)?;
    let mut manifest = Vec::new();
    let mut taps: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for t in model.taps(&gpu)? {
        let v: Vec<f16> = gpu.download(t.buf)?;
        let n: usize = t.shape.iter().product();
        ensure!(v.len() == n, "tap {}: {} values for shape {:?}", t.name, v.len(), t.shape);
        taps.push((t.name, t.shape, v.iter().map(|x| x.to_f32()).collect()));
    }
    let l: Vec<f16> = gpu.download(&logits)?;
    taps.push((
        "logits".into(),
        vec![batch, edge.px(), edge.px(), 1],
        l[..batch * edge.pixels()].iter().map(|x| x.to_f32()).collect(),
    ));
    for (name, shape, data) in &taps {
        let bad = data.iter().filter(|v| !v.is_finite()).count();
        ensure!(bad == 0, "tap {name}: {bad} non-finite values");
        write_npy(&out.join(format!("{name}.npy")), shape, data)?;
        manifest.push(serde_json::json!({"name": name, "shape": shape, "layout": "channels_last"}));
    }
    let algos: Vec<_> = model
        .convs()
        .into_iter()
        .map(|(n, c)| serde_json::json!({"layer": n, "shape": c.shape().to_string(), "algo": c.algo()}))
        .collect();
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"tiles": batch, "taps": manifest, "algos": algos, "schema": schema, "cudnn": Dnn::version(),
                "ptx_tier": gpu.inner.ptx_tier().to_string()}),
        )?,
    )?;
    println!("p0: {} taps for {batch} tiles in {}, ptx tier {}", taps.len(), out.display(), gpu.inner.ptx_tier());
    Ok(())
}

/// One `check-blend` case's outcome: pixels compared, pixels off by one step, the largest
/// difference, pixels that had to match exactly and did not, and coverage mismatches
#[derive(Default)]
struct BlendDiff {
    pixels: usize,
    off_by_one: usize,
    max: u8,
    inexact: usize,
    coverage: usize,
}

impl BlendDiff {
    /// `exact`: the reference value is far enough from a quantisation boundary (or the arithmetic is
    /// the kernel's own, op for op) that only a wrong blend or rounding rule can move it
    fn add(&mut self, got: u8, want: u8, exact: bool) {
        let d = got.abs_diff(want);
        self.pixels += 1;
        self.off_by_one += usize::from(d == 1);
        self.max = self.max.max(d);
        self.inexact += usize::from(exact && d != 0);
    }
}

/// A deterministic xorshift stream, so a failing case can be rerun as is
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Which voxels of a `check-blend` volume are zero, and so which tiles the schedule skips
#[derive(Clone, Copy, Debug)]
enum Holes {
    /// One block of half each side, so tiles straddle its edge and a few lie inside it
    Block,
    /// Everything but a `margin` px frame, so whole tile rows inside are skipped and the pixels only
    /// they covered are left uncovered
    Frame { margin: usize },
    /// All zero: no tile is kept and nothing is covered
    Empty,
}

/// A uint8 volume of random nonzero voxels with `holes` zero
fn test_volume(rng: &mut Rng, depth: usize, height: usize, width: usize, holes: Holes) -> Volume {
    let zero = |y: usize, x: usize| match holes {
        Holes::Block => {
            (height / 3..height / 3 + height / 2).contains(&y) && (width / 4..width / 4 + width / 2).contains(&x)
        }
        Holes::Frame { margin } => (margin..height - margin).contains(&y) && (margin..width - margin).contains(&x),
        Holes::Empty => true,
    };
    let data = (0..depth * height * width)
        .map(|i| if zero(i / width % height, i % width) { 0 } else { 1 + (rng.next() % 255) as u8 })
        .collect();
    Volume { depth, height, width, data, array: "check-blend".into() }
}

/// A reference `255 p` closer than this to a quantisation boundary may land one step away on the
/// GPU, whose `expf` can differ from the CPU's `exp` in the last bit; farther than this, the reader
/// must match exactly, so truncation swapped for rounding (or the reverse) fails
const NEAR_BOUNDARY: f32 = 1e-3;

/// The reader's per-pixel field: random logits, or logit 0 everywhere, whose sigmoid is exactly 0.5
/// on both sides, so every covered pixel's renormalised `255 p` is exactly 127.5 and must truncate to
/// 127
#[derive(Clone, Copy, PartialEq)]
enum ReaderField {
    Random,
    Half,
}

/// `check-blend`: the reader's `blend_finish` (128 px tiles) and `blend_finish_384` (Reader V3's 384
/// px tiles) and Hecate's `hec_blend` on the GPU against a CPU reference, at strides from 1 to the
/// tile edge, on sides the strides do not divide, with skipped tiles, uncovered pixels and an
/// all-zero volume. The schedules are the engines' own; the reference walks every tile position per
/// pixel instead of the kernels' spans.
///
/// - Reader: coverage must match exactly, the value exactly wherever the reference is more than
///   [`NEAR_BOUNDARY`] from a step (else within one), and the logit-0 field must give exactly 127
///   on every covered pixel (renormalisation and truncation). A synthetic window around the 1e-6
///   cut checks coverage at the boundary. The cases together must hold covered and uncovered pixels.
/// - Hecate: its blend is the reference's arithmetic op for op (no expf, rounded multiplies and adds
///   in tile order, an IEEE divide), so every pixel must match exactly; one case places
///   probabilities whose `255 p` is exactly a half step, rounding both up and down to even
pub fn check_blend() -> Result<()> {
    use crate::gpu::Arg;
    use crate::hecate::{self, P, PIX};
    use crate::sched::{Occupancy, Schedule, hann_weights};

    let gpu = Gpu::open()?;
    // the gate job checks this line against the tier it asked for
    println!("check-blend: PTX tier {}", gpu.inner.ptx_tier());
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut failures = Vec::new();

    // the reader at both tile edges (128 px, and Reader V3's 384 px with its own `blend_finish_384`):
    // logits in FP16, villa's trunc(255 * clip(p, 0, 1)) and 1e-6 cut
    let (mut covered_px, mut uncovered_px) = (0usize, 0usize);
    for edge in [Edge::P128, Edge::P384] {
        let patch = edge.px();
        let hann = hann_weights(edge);
        // around the cut: zero, below, exactly at (not covered: the test is `den > 1e-6`), the next f32
        // above, and well above; the half field makes every covered value 127 and every uncovered one 0
        let at = 1e-6f32;
        let cut_steps = [0.0, 5e-7, at, f32::from_bits(at.to_bits() + 1), 2e-6, 0.5];
        let cut: Vec<f32> = (0..edge.pixels()).map(|i| cut_steps[i % cut_steps.len()]).collect();
        let reader_cases: Vec<(usize, usize, &[usize], Holes, bool)> = match edge {
            Edge::P128 => vec![
                (128, 128, &[1, 64, 128], Holes::Block, false),
                (129, 131, &[1, 2, 127], Holes::Block, false),
                (300, 457, &[23, 37, 64, 96, 127, 128], Holes::Block, false),
                (640, 200, &[48, 80, 100, 128], Holes::Block, false),
                (384, 400, &[64, 96, 128], Holes::Frame { margin: 64 }, false),
                (200, 300, &[64, 128], Holes::Empty, false),
                (128, 128, &[128], Holes::Block, true),
            ],
            // the same shapes of case at three times the edge, fewer strides since the reference
            // walks every tile per pixel
            Edge::P384 => vec![
                (384, 384, &[1, 192, 384], Holes::Block, false),
                (385, 391, &[1, 2, 383], Holes::Block, false),
                (700, 901, &[69, 192, 288, 383, 384], Holes::Block, false),
                (1152, 900, &[192, 384], Holes::Frame { margin: 192 }, false),
                (400, 500, &[192, 384], Holes::Empty, false),
                (384, 384, &[384], Holes::Block, true),
            ],
        };
        for (h, w, strides, holes, synthetic) in reader_cases {
            let weight = if synthetic { &cut } else { &hann };
            let wbuf = gpu.upload(weight)?;
            let vol = test_volume(&mut rng, DEPTH, h, w, holes);
            for &stride in strides {
                for field in [ReaderField::Random, ReaderField::Half] {
                    if synthetic && field == ReaderField::Random {
                        continue;
                    }
                    let occ = Occupancy::new(vol.view(), stride, None, patch);
                    let sched = Schedule::new(&occ, h, w, &[(0, 1)], 4, 1)?;
                    let grid = &sched.maps[0].grid;
                    let logits: Vec<u16> = (0..sched.slots.max(1) * edge.pixels())
                        .map(|_| match field {
                            ReaderField::Half => f16::ZERO,
                            ReaderField::Random => f16::from_f32(16.0 * rng.unit() - 8.0),
                        })
                        .map(f16::to_bits)
                        .collect();
                    let (lbuf, gbuf) = (gpu.upload(&logits)?, gpu.upload(grid)?);
                    let (ys, xs) = (gpu.upload(&sched.ys)?, gpu.upload(&sched.xs)?);
                    let (ysp, xsp) = (gpu.upload(&sched.y_span)?, gpu.upload(&sched.x_span)?);
                    let (out, cov) = (gpu.alloc(h * w)?, gpu.alloc(h * w)?);
                    gpu.launch_1d(
                        edge.kernel("blend_finish"),
                        h * w,
                        &[
                            out.arg::<u8>(),
                            cov.arg::<u8>(),
                            lbuf.arg::<u16>(),
                            gbuf.arg::<i32>(),
                            ys.arg::<u32>(),
                            xs.arg::<u32>(),
                            ysp.arg::<[u32; 2]>(),
                            xsp.arg::<[u32; 2]>(),
                            wbuf.arg::<f32>(),
                            Arg::U32(w as u32),
                        ],
                    )?;
                    let (got, got_cov): (Vec<u8>, Vec<u8>) = (gpu.download(&out)?, gpu.download(&cov)?);
                    ensure!(got.len() == h * w && got_cov.len() == h * w, "blend output of the wrong size");
                    let mut diff = BlendDiff::default();
                    let nx = occ.xs.len();
                    for y in 0..h {
                        for x in 0..w {
                            let (mut num, mut den) = (0f32, 0f32);
                            for (iy, &ty) in occ.ys.iter().enumerate() {
                                for (ix, &tx) in occ.xs.iter().enumerate() {
                                    let slot = grid[iy * nx + ix];
                                    if slot < 0 || !(ty..ty + patch).contains(&y) || !(tx..tx + patch).contains(&x) {
                                        continue;
                                    }
                                    let lp = (y - ty) * patch + (x - tx);
                                    let l = f16::from_bits(logits[slot as usize * edge.pixels() + lp]).to_f32();
                                    num += (1.0 / (1.0 + (-l).exp())) * weight[lp];
                                    den += weight[lp];
                                }
                            }
                            let covered = den > 1e-6;
                            let p = if covered { num / den } else { num }.clamp(0.0, 1.0);
                            let v = p * 255.0;
                            let i = y * w + x;
                            let far = (v - v.round()).abs() > NEAR_BOUNDARY;
                            let half = field == ReaderField::Half;
                            // the half field's 127.5 sits on no truncation boundary and is exact on both sides
                            diff.add(got[i], v as u8, far || half);
                            diff.coverage += usize::from(got_cov[i] != u8::from(covered));
                            if half && covered && got[i] != 127 {
                                diff.inexact += 1;
                            }
                            covered_px += usize::from(covered);
                            uncovered_px += usize::from(!covered);
                        }
                    }
                    let label =
                        format!("{holes:?}{} {}", if synthetic { " cut-window" } else { "" }, field_name(field));
                    report_case(&mut failures, &format!("reader {patch}"), h, w, stride, &label, sched.slots, &diff);
                }
            }
        }
    }
    println!("reader: {covered_px} covered and {uncovered_px} uncovered reference pixels over all cases");
    if covered_px == 0 || uncovered_px == 0 {
        failures.push(format!(
            "reader cases never compared both covered ({covered_px}) and uncovered ({uncovered_px}) pixels"
        ));
    }

    // Hecate: 64 px tiles, f32 probabilities, an empty tile's window still in the denominator, rint
    let weight = hecate::window();
    let wbuf = gpu.upload(&weight)?;
    let hecate_cases: [(usize, usize, &[usize], Holes); 6] = [
        (64, 64, &[1, 32, 64], Holes::Block),
        (65, 67, &[1, 2, 63], Holes::Block),
        (150, 211, &[17, 32, 40, 48, 63, 64], Holes::Block),
        (300, 96, &[24, 64], Holes::Block),
        (256, 200, &[32, 64], Holes::Frame { margin: 32 }),
        (100, 90, &[32], Holes::Empty),
    ];
    let mut ties = (0usize, 0usize);
    for (h, w, strides, holes) in hecate_cases {
        let vol = test_volume(&mut rng, 20, h, w, holes);
        for &stride in strides {
            for constant in [false, true] {
                let maps = [hecate::engine::MapSpec { reverse: false, png: "check-blend.png".into() }];
                let sched = hecate::engine::Schedule::new(&vol, &maps, stride, 4)?;
                let grid = &sched.grids[0];
                let c = 0.6f32;
                let mut prob: Vec<f32> =
                    (0..sched.slots.max(1) * PIX).map(|_| if constant { c } else { rng.unit() }).collect();
                // one tile per pixel: plant half-step ties in the random field (see `plant_ties`)
                let single = (h, w, stride) == (64, 64, 64) && !constant;
                if single {
                    ensure!(sched.slots == 1 && grid == &[0], "the 64 x 64 stride 64 case must be one kept tile");
                    ties = plant_ties(&mut prob[..PIX], &weight)?;
                }
                let (pbuf, gbuf) = (gpu.upload(&prob)?, gpu.upload(grid)?);
                let (ys, xs) = (gpu.upload(&sched.ys)?, gpu.upload(&sched.xs)?);
                let ysp = gpu.upload(&hecate::engine::spans(&sched.ys, h))?;
                let xsp = gpu.upload(&hecate::engine::spans(&sched.xs, w))?;
                let out = gpu.alloc(h * w)?;
                gpu.launch_1d(
                    "hec_blend",
                    h * w,
                    &[
                        out.arg::<u8>(),
                        pbuf.arg::<f32>(),
                        gbuf.arg::<i32>(),
                        ys.arg::<u32>(),
                        xs.arg::<u32>(),
                        ysp.arg::<[u32; 2]>(),
                        xsp.arg::<[u32; 2]>(),
                        wbuf.arg::<f32>(),
                        Arg::U32(w as u32),
                    ],
                )?;
                let got: Vec<u8> = gpu.download(&out)?;
                ensure!(got.len() == h * w, "blend output of the wrong size");
                let mut diff = BlendDiff::default();
                let nx = sched.xs.len();
                for y in 0..h as u32 {
                    for x in 0..w as u32 {
                        let (mut num, mut den, mut all_kept) = (0f32, 0f32, true);
                        for (iy, &ty) in sched.ys.iter().enumerate() {
                            for (ix, &tx) in sched.xs.iter().enumerate() {
                                if !(ty..ty + P as u32).contains(&y) || !(tx..tx + P as u32).contains(&x) {
                                    continue;
                                }
                                let lp = ((y - ty) * P as u32 + (x - tx)) as usize;
                                let slot = grid[iy * nx + ix];
                                if slot >= 0 {
                                    num += prob[slot as usize * PIX + lp] * weight[lp];
                                } else {
                                    all_kept = false;
                                }
                                den += weight[lp];
                            }
                        }
                        let p = (num / den).clamp(0.0, 1.0);
                        let g = got[(y as usize) * w + x as usize];
                        diff.add(g, (p * 255.0).round_ties_even() as u8, true);
                        if constant && all_kept && g != (c * 255.0).round_ties_even() as u8 {
                            diff.inexact += 1;
                        }
                    }
                }
                let label = format!(
                    "{holes:?} {}{}",
                    if constant { "constant" } else { "random" },
                    if single { " +ties" } else { "" }
                );
                report_case(&mut failures, "hecate", h, w, stride, &label, sched.slots, &diff);
            }
        }
    }
    println!("hecate: planted {} half steps that round up to even and {} that round down", ties.0, ties.1);
    if ties.0 == 0 || ties.1 == 0 {
        failures.push(format!("hecate: planted {} up and {} down half-step ties; need both", ties.0, ties.1));
    }
    ensure!(failures.is_empty(), "check-blend failed: {}", failures.join("; "));
    println!("check-blend: PASS");
    Ok(())
}

fn field_name(field: ReaderField) -> &'static str {
    match field {
        ReaderField::Random => "random",
        ReaderField::Half => "logit-0",
    }
}

/// Puts into a single kept tile's probabilities values whose blended `255 * (p * w) / w` is exactly
/// `k + 0.5` in f32, the reference's and the kernel's arithmetic alike, so `rint` must round half to
/// even: up for odd k, down for even k (truncation, or rounding half away from zero, fails one of
/// the two). Returns how many round up and how many down
fn plant_ties(prob: &mut [f32], weight: &[f32]) -> Result<(usize, usize)> {
    let (mut up, mut down) = (0, 0);
    let targets = [0.5f32, 1.5, 126.5, 127.5, 128.5, 200.5, 253.5, 254.5];
    let mut lp = 7;
    let mut used = Vec::new();
    for &t in &targets {
        // try a few window positions and a few hundred f32s around t / 255 at each
        'pos: for _ in 0..64 {
            lp = (lp * 31 + 17) % prob.len();
            if used.contains(&lp) {
                continue;
            }
            let w = weight[lp];
            let base = (t / 255.0).to_bits();
            for k in 0..512u32 {
                let p = f32::from_bits(if k % 2 == 0 { base + k / 2 } else { base - k / 2 - 1 });
                if (p * w / w) * 255.0 == t {
                    prob[lp] = p;
                    used.push(lp);
                    if t.round_ties_even() > t {
                        up += 1
                    } else {
                        down += 1
                    }
                    break 'pos;
                }
            }
        }
    }
    ensure!(up + down > 0, "no half-step tie could be planted");
    Ok((up, down))
}

#[allow(clippy::too_many_arguments)]
fn report_case(
    failures: &mut Vec<String>,
    model: &str,
    h: usize,
    w: usize,
    stride: usize,
    label: &str,
    tiles: usize,
    d: &BlendDiff,
) {
    let ok = d.pixels == h * w && d.max <= 1 && d.inexact == 0 && d.coverage == 0;
    println!(
        "{model} {h}x{w} stride {stride} {label}: {tiles} tiles, {} px, {} off by 1, max {}, {} should be exact, \
         coverage diffs {} {}",
        d.pixels,
        d.off_by_one,
        d.max,
        d.inexact,
        d.coverage,
        if ok { "ok" } else { "FAIL" }
    );
    if !ok {
        failures.push(format!("{model} {h}x{w} stride {stride} {label}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plant_ties_lands_on_half_steps_both_ways() {
        let weight = crate::hecate::window();
        let mut prob = vec![0.25f32; crate::hecate::PIX];
        let (up, down) = plant_ties(&mut prob, &weight).unwrap();
        assert!(up > 0 && down > 0, "{up} up, {down} down");
        let halves: Vec<f32> =
            prob.iter().zip(&weight).map(|(&p, &w)| (p * w / w) * 255.0).filter(|v| v.fract() == 0.5).collect();
        assert_eq!(halves.len(), up + down, "{halves:?}");
    }
}
