//! Whole-volume 3D inference over a zarr region with villa's native plan
//! (`infer_full3d_tifxyz.build_native_plan` with every chunk of the region a target)
//!
//! Patches start on villa's global stride grid (`_global_patch_starts_for_interval`) and run in villa's
//! order, sorted by (z, y, x). For each z the input slab under that row of patches is read once and
//! uploaded once. Per patch: a GPU histogram of its in-volume voxels, villa's `percentile_minmax` table
//! from it on the host (float64, as villa's uint8 table), the bf16 input, the forward, and the head,
//! sigmoid and Gaussian blend into float32 accumulators on the GPU. The accumulators cover the region's
//! y and x and a ring of z chunk layers; a layer is finished to uint8 and written once no later patch
//! reaches it

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure};

use super::Contract;
use super::model::{Ctx, Model, u32_extent};
use super::zio::{Array, Writer};
use crate::dnn::{Blas, Dnn, Pick, algo_cache};
use crate::gpu::{Arg, Buf, Gpu};
use crate::weights::Checkpoint;

/// Run options
#[derive(Clone, Debug)]
pub struct Options {
    /// villa's `--overlap`: stride `max(1, round(p * (1 - overlap)))`
    pub overlap: f64,
    /// Region `[z0, y0, x0, z1, y1, x1)` on the output chunk grid; the whole volume when `None`
    pub bbox: Option<[usize; 6]>,
    /// Output chunk edge, villa's write chunk
    pub chunk: usize,
    pub pick: Pick,
    /// Cap on the GPU accumulators
    pub max_accum_gb: f64,
    pub overwrite: bool,
    /// After the run, time this many forwards of the last patch
    pub forward_bench: usize,
    /// Also write villa's coverage (blend weight > 1e-6) as a 0/1 uint8 group `coverage` inside the
    /// output, for `scripts/mapgate3d.py`
    pub coverage: bool,
    /// CUDA device ordinal
    pub device: usize,
}

/// What a run did and how long it took
#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub patches: usize,
    pub chunks_written: usize,
    pub load_s: f64,
    pub run_s: f64,
    pub s_per_patch: f64,
    pub read_s: f64,
    pub write_s: f64,
    pub forward_ms: Option<f64>,
    pub plans: Vec<String>,
    /// The algorithm cache's state, and why it is off when it is
    pub algo_cache: algo_cache::Status,
    pub stem_padded: bool,
    pub ptx_tier: String,
}

/// villa's `_global_patch_starts_for_interval`
fn starts_for_interval(start: usize, stop: usize, len: usize, p: usize, stride: usize) -> Vec<usize> {
    let max_start = len.saturating_sub(p);
    let first = ((start as i64 - p as i64 + 1).div_euclid(stride as i64) * stride as i64).max(0) as usize;
    let mut out = Vec::new();
    let mut s = first;
    while s <= max_start && s < stop {
        if s + p > start {
            out.push(s);
        }
        s += stride;
    }
    if stop > max_start && max_start + p > start {
        out.push(max_start);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Patch starts on one axis covering every chunk of `[lo, hi)`
fn axis_starts(lo: usize, hi: usize, chunk: usize, len: usize, p: usize, stride: usize) -> Vec<usize> {
    let mut all = Vec::new();
    let mut c = lo;
    while c < hi {
        all.extend(starts_for_interval(c, (c + chunk).min(len), len, p, stride));
        c += chunk;
    }
    all.sort_unstable();
    all.dedup();
    all
}

/// villa's uint8 `percentile_minmax` table from a 256-bin histogram (`uint8_normalization_table`):
/// NumPy's linear percentile from order statistics, a float32 clamp, then `(x - low) / (high - low)`
/// in float64 rounded to float32; zeros when the span is under 1e-6 or nothing is counted
pub fn percentile_table(hist: &[u32; 256], lower: f64, upper: f64) -> [f32; 256] {
    let total: u64 = hist.iter().map(|&v| v as u64).sum();
    let mut table = [0f32; 256];
    if total == 0 {
        return table;
    }
    let mut cumulative = [0u64; 256];
    let mut run = 0;
    for (c, &h) in hist.iter().enumerate() {
        run += h as u64;
        cumulative[c] = run;
    }
    // np.searchsorted(cumulative, rank, side="right")
    let order = |rank: u64| -> f64 { cumulative.iter().take_while(|&&v| v <= rank).count() as f64 };
    let percentile = |percent: f64| -> f64 {
        let q = percent / 100.0;
        // numpy's linear virtual index n*q + (alpha + q*(1 - alpha - beta)) - 1 with alpha = beta = 1;
        // 1 + (-q) and 1 - q round identically
        let position = total as f64 * q + (1.0 - q) - 1.0;
        if position >= (total - 1) as f64 {
            return order(total - 1);
        }
        if position < 0.0 {
            return order(0);
        }
        let below = position.floor();
        let fraction = position - below;
        let (lo, hi) = (order(below as u64), order(below as u64 + 1));
        let span = hi - lo;
        if fraction >= 0.5 { hi - span * (1.0 - fraction) } else { lo + span * fraction }
    };
    let (low, high) = (percentile(lower), percentile(upper));
    let scale = high - low;
    if !(scale.is_finite() && scale >= 1e-6) {
        return table;
    }
    let (l32, h32) = (low as f32, high as f32);
    for (c, t) in table.iter_mut().enumerate() {
        let v = c as f32;
        let clamped = if v < l32 {
            l32
        } else if v > h32 {
            h32
        } else {
            v
        };
        let r = ((clamped as f64 - low) / scale) as f32;
        *t = if r.is_finite() { r } else { 0.0 };
    }
    table
}

/// The GPU state of one run
struct Run<'a> {
    gpu: &'a Gpu,
    region: [usize; 6],
    ring_layers: usize,
    chunk: usize,
    acc: Buf,
    wacc: Buf,
    out8: Buf,
    imp: Buf,
    hist: Buf,
    table: Buf,
    slab: Buf,
}

impl Run<'_> {
    fn layer_elems(&self) -> usize {
        self.chunk * (self.region[4] - self.region[1]) * (self.region[5] - self.region[2])
    }

    /// Finishes chunk layer `k` (global z `[k * chunk, ...)`) to uint8, writes it and clears its slots
    fn flush(&self, k: usize, writer: &Writer, coverage: Option<&Writer>) -> Result<usize> {
        let (rz0, rz1) = (self.region[0], self.region[3]);
        let z0 = k * self.chunk;
        let depth = self.chunk.min(rz1 - z0);
        let (rh, rw) = (self.region[4] - self.region[1], self.region[5] - self.region[2]);
        let n = depth * rh * rw;
        let slot = ((z0 - rz0) % (self.ring_layers * self.chunk)) * rh * rw;
        self.gpu.launch_1d(
            "u3_finish",
            n,
            &[self.out8.view::<u8>(0, n), self.acc.view::<f32>(slot, n), self.wacc.view::<f32>(slot, n)],
        )?;
        let host: Vec<u8> = self.gpu.download_range(&self.out8, 0, n)?;
        if let Some(cw) = coverage {
            self.gpu.launch_1d("u3_covered", n, &[self.out8.view::<u8>(0, n), self.wacc.view::<f32>(slot, n)])?;
            let mask: Vec<u8> = self.gpu.download_range(&self.out8, 0, n)?;
            cw.write_slab([z0, self.region[1], self.region[2]], [depth, rh, rw], &mask)?;
        }
        let le = self.layer_elems();
        // zero the layer's slots for the layer that reuses them
        zero_range(self.gpu, &self.acc, slot * 4, le * 4)?;
        zero_range(self.gpu, &self.wacc, slot * 4, le * 4)?;
        writer.write_slab([z0, self.region[1], self.region[2]], [depth, rh, rw], &host)
    }
}

/// Where a patch sits in an uploaded slab: the slab's y and x dims, the patch corner in the slab, and
/// the patch's in-volume extent per axis from its corner
pub(super) struct Geometry {
    pub slab_hw: [usize; 2],
    pub origin: [usize; 3],
    pub valid: [usize; 3],
}

/// Writes one patch's bf16 model input into [`Model::input`]: the GPU histogram of its in-volume
/// voxels, villa's table from it on the host, then the table lookup; returns the table
#[allow(clippy::too_many_arguments)]
pub(super) fn prep_input(
    gpu: &Gpu,
    model: &Model,
    slab: &Buf,
    slab_dims: [usize; 3],
    g: &Geometry,
    hist: &Buf,
    table: &Buf,
    contract: &Contract,
) -> Result<[f32; 256]> {
    let p = model.patch;
    let geo = [
        Arg::U32(g.slab_hw[0] as u32),
        Arg::U32(g.slab_hw[1] as u32),
        Arg::U32(g.origin[0] as u32),
        Arg::U32(g.origin[1] as u32),
        Arg::U32(g.origin[2] as u32),
        Arg::U32(p as u32),
        Arg::U32(0),
        Arg::U32(g.valid[0] as u32),
        Arg::U32(0),
        Arg::U32(g.valid[1] as u32),
        Arg::U32(0),
        Arg::U32(g.valid[2] as u32),
    ];
    let n = slab_dims.iter().product();
    ensure!(
        (0..3).all(|a| g.origin[a] + g.valid[a] <= slab_dims[a]) && g.slab_hw == [slab_dims[1], slab_dims[2]],
        "patch geometry {:?} + {:?} outside the slab {slab_dims:?}",
        g.origin,
        g.valid
    );
    gpu.fill(hist, 0)?;
    let mut args = vec![hist.arg::<u32>(), slab.view::<u8>(0, n)];
    args.extend_from_slice(&geo);
    gpu.launch("u3_hist", (p as u32, 1, 1), (256, 1, 1), 0, &args)?;
    let h: Vec<u32> = gpu.download(hist)?;
    let h: [u32; 256] = h.try_into().map_err(|_| color_eyre::eyre::eyre!("histogram size"))?;
    let t = percentile_table(&h, contract.normalization.lower, contract.normalization.upper);
    gpu.upload_into(table, 0, &t)?;
    let mut args = vec![model.input.arg::<u16>(), slab.view::<u8>(0, n), table.arg::<f32>()];
    args.extend_from_slice(&geo);
    args.push(Arg::U32(model.in_c as u32));
    gpu.launch_1d("u3_prep", p.pow(3), &args)?;
    Ok(t)
}

/// Zeroes `bytes` of `buf` from byte `offset`, queued on the stream
fn zero_range(gpu: &Gpu, buf: &Buf, offset: usize, bytes: usize) -> Result<()> {
    ensure!(offset + bytes <= buf.bytes(), "zero range past the end of the buffer");
    // SAFETY: the range lies inside the live allocation, checked above
    unsafe { cudarc::driver::result::memset_d8_async(buf.ptr() + offset as u64, 0, bytes, gpu.cu_stream()) }?;
    Ok(())
}

/// Runs the model over `input` (a zarr array or group) into a new zarr group at `out`
pub fn run(input: &Path, ckpt: &Path, out: &Path, opts: &Options) -> Result<Report> {
    let t_load = Instant::now();
    let (ck, sidecar) = Checkpoint::read(ckpt)?;
    let contract = Contract::from_sidecar(&sidecar)?;
    let vol = Array::open(input)?;
    let p = contract.patch();
    let c = opts.chunk;
    ensure!(c > 0, "chunk size 0");
    ensure!((0.0..1.0).contains(&opts.overlap), "overlap {} not in [0, 1)", opts.overlap);
    let stride = ((p as f64 * (1.0 - opts.overlap)).round_ties_even() as usize).max(1);
    let [zd, yd, xd] = vol.shape;
    let region = opts.bbox.unwrap_or([0, 0, 0, zd, yd, xd]);
    for a in 0..3 {
        ensure!(region[a] < region[a + 3] && region[a + 3] <= vol.shape[a], "bbox {region:?} outside {:?}", vol.shape);
        ensure!(region[a].is_multiple_of(c), "bbox start {} not on the {c}-voxel chunk grid", region[a]);
        ensure!(
            region[a + 3].is_multiple_of(c) || region[a + 3] == vol.shape[a],
            "bbox end {} not on the chunk grid",
            region[a + 3]
        );
    }

    let per_axis: Vec<Vec<usize>> =
        (0..3).map(|a| axis_starts(region[a], region[a + 3], c, vol.shape[a], p, stride)).collect();
    let mut patches = Vec::new();
    for &z in &per_axis[0] {
        for &y in &per_axis[1] {
            for &x in &per_axis[2] {
                patches.push([z, y, x]);
            }
        }
    }
    ensure!(!patches.is_empty(), "no patches");

    let gpu = Gpu::open_device(opts.device)?;
    let dnn = Dnn::new(&gpu)?;
    let blas = Blas::new(&gpu)?;
    let ctx = Ctx { gpu: &gpu, dnn: &dnn, blas: &blas, pick: opts.pick, max_workspace: 1 << 30 };
    let mut model = Model::new(&ctx, &ck, &contract)?;
    ck.finish()?;

    // ring accumulators over the region's y and x
    let (rh, rw) = (region[4] - region[1], region[5] - region[2]);
    let ring_layers = p.div_ceil(c) + 1;
    let ring = ring_layers * c * rh * rw;
    u32_extent("accumulator ring", ring)?;
    let gb = (ring * 8) as f64 / 1e9;
    ensure!(
        gb <= opts.max_accum_gb,
        "accumulators need {gb:.1} GB over --max-accum-gb {}; use a smaller --bbox",
        opts.max_accum_gb
    );
    let (ymin, ymax) = (per_axis[1][0], (per_axis[1].last().unwrap() + p).min(yd));
    let (xmin, xmax) = (per_axis[2][0], (per_axis[2].last().unwrap() + p).min(xd));
    let slab_max = p.min(zd) * (ymax - ymin) * (xmax - xmin);
    u32_extent("input slab", slab_max)?;
    let run = Run {
        gpu: &gpu,
        region,
        ring_layers,
        chunk: c,
        acc: gpu.alloc_fill(ring * 4, 0)?,
        wacc: gpu.alloc_fill(ring * 4, 0)?,
        out8: gpu.alloc(c * rh * rw)?,
        imp: gpu.upload(&contract.importance_map())?,
        hist: gpu.alloc(256 * 4)?,
        table: gpu.alloc(256 * 4)?,
        slab: gpu.alloc(slab_max)?,
    };
    let writer = Writer::create(out, vol.shape, c, opts.overwrite)?;
    let cov_writer =
        if opts.coverage { Some(Writer::create(&out.join("coverage"), vol.shape, c, false)?) } else { None };
    let (head_c, head_b) = {
        let h = model.head();
        (h.1, h.3)
    };
    gpu.sync()?;
    let load_s = t_load.elapsed().as_secs_f64();

    let t_run = Instant::now();
    let (mut read_s, mut write_s) = (0f64, 0f64);
    let mut written = 0;
    let mut next_flush = region[0] / c;
    let mut slab_z = usize::MAX;
    let mut slab_dims = [0usize; 3];
    let mut host_slab: Vec<u8> = Vec::new();
    for (i, &[z0, y0, x0]) in patches.iter().enumerate() {
        // finish every layer no later patch reaches
        while next_flush * c < region[3] && (next_flush + 1) * c <= z0 {
            let t = Instant::now();
            written += run.flush(next_flush, &writer, cov_writer.as_ref())?;
            write_s += t.elapsed().as_secs_f64();
            next_flush += 1;
        }
        if z0 != slab_z {
            let t = Instant::now();
            let zhi = (z0 + p).min(zd);
            slab_dims = [zhi - z0, ymax - ymin, xmax - xmin];
            host_slab.resize(slab_dims.iter().product(), 0);
            vol.read_box([z0, ymin, xmin], [zhi, ymax, xmax], &mut host_slab)?;
            gpu.upload_into(&run.slab, 0, &host_slab)?;
            slab_z = z0;
            read_s += t.elapsed().as_secs_f64();
        }
        let geo = Geometry {
            slab_hw: [slab_dims[1], slab_dims[2]],
            origin: [0, y0 - ymin, x0 - xmin],
            valid: [p.min(zd - z0), p.min(yd - y0), p.min(xd - x0)],
        };
        prep_input(&gpu, &model, &run.slab, slab_dims, &geo, &run.hist, &run.table, &contract)?;
        model.forward(&ctx)?;
        let (x, _, hw, _) = model.head();
        gpu.launch_1d(
            "u3_head_blend",
            p.pow(3),
            &[
                run.acc.arg::<f32>(),
                run.wacc.arg::<f32>(),
                x.view::<u32>(0, p.pow(3) * head_c / 2),
                hw.arg::<f32>(),
                Arg::F32(head_b),
                run.imp.arg::<f32>(),
                Arg::U32(head_c as u32),
                Arg::U32(p as u32),
                Arg::U32(z0 as u32),
                Arg::U32(y0 as u32),
                Arg::U32(x0 as u32),
                Arg::U32(region[0] as u32),
                Arg::U32(region[1] as u32),
                Arg::U32(region[2] as u32),
                Arg::U32((region[3] - region[0]) as u32),
                Arg::U32(rh as u32),
                Arg::U32(rw as u32),
                Arg::U32((ring_layers * c) as u32),
            ],
        )?;
        if i + 1 == patches.len() || (i + 1) % 64 == 0 {
            eprintln!("ink3d: {}/{} patches, {:.1} s", i + 1, patches.len(), t_run.elapsed().as_secs_f64());
        }
    }
    while next_flush * c < region[3] {
        let t = Instant::now();
        written += run.flush(next_flush, &writer, cov_writer.as_ref())?;
        write_s += t.elapsed().as_secs_f64();
        next_flush += 1;
    }
    gpu.sync()?;
    let run_s = t_run.elapsed().as_secs_f64();

    let forward_ms = if opts.forward_bench > 0 {
        model.forward(&ctx)?;
        gpu.sync()?;
        let t = Instant::now();
        for _ in 0..opts.forward_bench {
            model.forward(&ctx)?;
        }
        gpu.sync()?;
        Some(t.elapsed().as_secs_f64() * 1e3 / opts.forward_bench as f64)
    } else {
        None
    };
    Ok(Report {
        patches: patches.len(),
        chunks_written: written,
        load_s,
        run_s,
        s_per_patch: run_s / patches.len() as f64,
        read_s,
        write_s,
        forward_ms,
        plans: model.plans(),
        algo_cache: algo_cache::status(),
        stem_padded: model.in_c != 1,
        ptx_tier: gpu.inner.ptx_tier().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn villa_starts_cover_the_crop() {
        // the 245-patch Paris 4 crop: 768 x 1024 x 1024, patch 256, stride 128, chunk 128
        let z = axis_starts(0, 768, 128, 768, 256, 128);
        let y = axis_starts(0, 1024, 128, 1024, 256, 128);
        assert_eq!(z, vec![0, 128, 256, 384, 512]);
        assert_eq!(y, vec![0, 128, 256, 384, 512, 640, 768]);
        assert_eq!(z.len() * y.len() * y.len(), 245);
        // an unaligned end adds villa's last start at len - p
        assert_eq!(axis_starts(0, 300, 128, 300, 256, 128), vec![0, 44]);
    }

    #[test]
    fn percentile_table_matches_numpy_cases() {
        // one value everywhere: span 0, all zeros
        let mut h = [0u32; 256];
        h[7] = 100;
        assert!(percentile_table(&h, 1.0, 99.0).iter().all(|&v| v == 0.0));
        // codes 0..=99 once each; the expected values are villa's normalize_image(percentile_minmax, 1, 99)
        // output for np.arange(100): the f32 clamp of the f64 low percentile leaves code 0 just above zero
        let mut h = [0u32; 256];
        h[..100].fill(1);
        let t = percentile_table(&h, 1.0, 99.0);
        assert_eq!((t[0], t[50], t[99], t[200]), (9.829667e-11, 0.5051536, 1.0, 1.0));
    }
}
