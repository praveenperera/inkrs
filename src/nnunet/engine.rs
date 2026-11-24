//! `vesuvius.predict --blend_in_memory` for a binary nnU-Net (m7) on one GPU
//!
//! Patches come from villa's grid ([`super::grid::Plan`]) and run in its order, z, then y, then x. For each
//! z the uint8 slab under that row of patches is read once and uploaded once. A patch whose in-volume voxels
//! all normalise to one fp16 value is skipped, as villa skips patches its dataset marks empty: their zero
//! logits add neither logits nor weight. Otherwise per flip (8 with mirroring) the fp16 input, the forward
//! and the head go into a float32 sum, and the flips' fp16 mean times villa's Gaussian into float32
//! accumulators, in villa's float32 order; a patch whose mean logits are all 0 is skipped there too, as
//! villa's blender skips it, and a non-finite logit fails the run before any output row it reaches is
//! written. The accumulators cover the output chunks the patches touch on y
//! and x and a ring of chunk rows on z; a row is finalised and written once no later patch reaches it, chunk
//! by chunk as villa writes them (a chunk without weight, or whose logits are all `allclose` to its first,
//! is not written)

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure};

use super::grid::{Bbox, Plan};
pub use super::host::{FLIPS, Options, logit_cutoff};
use super::host::{NONFINITE, NONZERO, constant_patch};
use super::out::Writer;
use super::{Contract, Precision, Predict};
use crate::dnn::{Blas, Dnn, algo_cache};
use crate::ensure_supported;
use crate::gpu::{Arg, Buf, Gpu};
use crate::unet3d::model::{Ctx, HostNet, Model, u32_extent};
use crate::unet3d::zio::Array;
use crate::weights::Checkpoint;

/// What a run did and how long it took
#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub model: String,
    pub patches: usize,
    /// Every patch corner of villa's grid, in its order
    pub positions: Vec<[usize; 3]>,
    /// Patches whose input normalised to one value, skipped as villa skips them
    pub patches_skipped: usize,
    pub skipped: Vec<[usize; 3]>,
    /// Patches the model ran on whose mean logits were all 0, left out of the blend as villa's blender
    /// leaves them out
    pub patches_empty_logits: usize,
    pub empty_logits: Vec<[usize; 3]>,
    pub flips: usize,
    pub threshold: Option<f64>,
    pub canvas: [usize; 6],
    pub chunk: usize,
    pub chunks_written: usize,
    pub load_s: f64,
    pub run_s: f64,
    pub s_per_patch: f64,
    pub read_s: f64,
    pub finalize_s: f64,
    pub plans: Vec<String>,
    pub algo_cache: algo_cache::Status,
    /// The loaded cuDNN's cudnnGetVersion, the value the algorithm cache keys on: inside a process where torch
    /// loaded its pip cuDNN first, that one, not the system copy
    pub cudnn: usize,
    pub stem_padded: bool,
    pub ptx_tier: String,
    pub flops_per_patch: u64,
}

/// The accumulators: chunk rows from `k0` on z, `ring` rows live at a time, over `[y0, y0 + rh) x [x0, x0 +
/// rw)`; class `k` of slot `o` at `acc[k * plane + o]`
struct Accum<'a> {
    gpu: &'a Gpu,
    shape: [usize; 3],
    chunk: usize,
    k0: usize,
    ring: usize,
    y0: usize,
    x0: usize,
    rh: usize,
    rw: usize,
    acc: Buf,
    wacc: Buf,
    out8: Buf,
    flags: Buf,
    cutoff: Option<f32>,
}

impl Accum<'_> {
    fn plane(&self) -> usize {
        self.ring * self.chunk * self.rh * self.rw
    }

    fn row_elems(&self) -> usize {
        self.chunk * self.rh * self.rw
    }

    fn chunks_yx(&self) -> (usize, usize) {
        (self.rh.div_ceil(self.chunk), self.rw.div_ceil(self.chunk))
    }

    /// Finalises chunk row `k`, writes the chunks villa writes, and zeroes the row's slots; returns the
    /// chunks written
    fn flush(&self, k: usize, writer: &Writer) -> Result<usize> {
        let gpu = self.gpu;
        let c = self.chunk;
        let z0 = k * c;
        let rows = c.min(self.shape[0] - z0);
        let az = ((k - self.k0) % self.ring) * c;
        let n = rows * self.rh * self.rw;
        let (ny, nx) = self.chunks_yx();
        gpu.fill(&self.flags, 0)?;
        gpu.launch_1d(
            "nn_finalize",
            n,
            &[
                self.out8.view::<u8>(0, n),
                self.flags.view::<u32>(0, 2 * ny * nx),
                self.acc.arg::<f32>(),
                self.wacc.arg::<f32>(),
                Arg::U32(az as u32),
                Arg::U32(rows as u32),
                Arg::U32(self.rh as u32),
                Arg::U32(self.rw as u32),
                Arg::U32(self.ring as u32 * c as u32),
                Arg::U32(c as u32),
                Arg::U32(self.cutoff.is_some() as u32),
                Arg::F32(self.cutoff.unwrap_or(0.0)),
            ],
        )?;
        let host: Vec<u8> = gpu.download_range(&self.out8, 0, n)?;
        let flags: Vec<u32> = gpu.download_range(&self.flags, 0, 2 * ny * nx)?;
        let slot = az * self.rh * self.rw;
        let row = self.row_elems();
        zero_range(gpu, &self.wacc, slot * 4, row * 4)?;
        for cls in 0..2 {
            zero_range(gpu, &self.acc, (cls * self.plane() + slot) * 4, row * 4)?;
        }
        // villa's _write_blended_chunk: a chunk with no weight above 0 is skipped, and apply_finalization
        // reports a chunk whose logits are all allclose to its first as empty
        let keep: Vec<bool> = (0..ny * nx).map(|i| flags[2 * i] != 0 && flags[2 * i + 1] != 0).collect();
        writer.write_row([z0, self.y0, self.x0], [rows, self.rh, self.rw], &host, &keep)
    }
}

/// Zeroes `bytes` of `buf` from byte `offset`, queued on the stream
fn zero_range(gpu: &Gpu, buf: &Buf, offset: usize, bytes: usize) -> Result<()> {
    ensure!(offset + bytes <= buf.bytes(), "zero range past the end of the buffer");
    // SAFETY: the range lies inside the live allocation, checked above
    unsafe { cudarc::driver::result::memset_d8_async(buf.ptr() + offset as u64, 0, bytes, gpu.cu_stream()) }?;
    Ok(())
}

/// The status words of the first `n` patches; fails when one of them had a non-finite logit
fn patch_status(gpu: &Gpu, stat: &Buf, n: usize, positions: &[[usize; 3]]) -> Result<Vec<u32>> {
    let words: Vec<u32> = gpu.download_range(stat, 0, n)?;
    let bad: Vec<_> = (0..n).filter(|&i| words[i] & NONFINITE != 0).map(|i| positions[i]).collect();
    ensure!(bad.is_empty(), "non-finite logits in {} patches, the first at {:?}", bad.len(), bad[0]);
    Ok(words)
}

/// Runs the export at `model` (`.safetensors` with its `.json` sidecar) over `input` (a uint8 zarr array,
/// or a group whose array `0` is read) into a new zarr array at `output`, as villa's `vesuvius.predict
/// --blend_in_memory --mode binary` writes it
pub fn run(model: &Path, input: &Path, output: &Path, opts: &Options) -> Result<Report> {
    let t_load = Instant::now();
    let (ck, sidecar) = Checkpoint::read(model)?;
    let contract = Contract::from_sidecar(&sidecar)?;
    let recipe: Predict = contract.predict()?;
    let p = contract.patch_size[0];
    let gauss = ck.take(&recipe.gaussian_key, &[p, p, p])?;
    ensure!(gauss.iter().all(|&v| (0.0..=1.0).contains(&v)), "blend weights outside [0, 1]");
    let host = HostNet::take(&ck, &contract.program, Precision::Fp16)?;
    ck.finish()?;

    let vol = Array::open(input)?;
    vol.ensure_apart(output)?;
    let shape = vol.shape;
    let overlap = opts.overlap.unwrap_or(recipe.overlap);
    let bbox = opts.bbox.as_deref().map(|b| Bbox::parse(b, shape)).transpose()?;
    let plan = Plan::new(shape, [p; 3], overlap, bbox)?;
    let flips = match opts.tta {
        Some(false) => 1,
        _ => recipe.flips,
    };
    let cutoff = opts.threshold.map(logit_cutoff).transpose()?;
    let c = opts.chunk.unwrap_or(p);
    ensure!(c > 0, "chunk size 0");

    // accumulators: the output chunks the canvas touches on y and x, a ring of chunk rows on z
    let canvas = plan.canvas();
    let (k0, k1) = (canvas.lo[0] / c, canvas.hi[0].div_ceil(c));
    let y0 = canvas.lo[1] / c * c;
    let x0 = canvas.lo[2] / c * c;
    let rh = (canvas.hi[1].div_ceil(c) * c).min(shape[1]) - y0;
    let rw = (canvas.hi[2].div_ceil(c) * c).min(shape[2]) - x0;
    let ring = p.div_ceil(c) + 1;
    let plane = ring * c * rh * rw;
    let gb = (3 * plane * 4) as f64 / 1e9;
    ensure_supported!(
        gb.total_cmp(&opts.max_accum_gb).is_le(),
        "the accumulators need {gb:.1} GB, over the {} GB cap; use a smaller --bbox",
        opts.max_accum_gb
    );
    u32_extent("accumulator", 2 * plane)?;
    let (ny, nx) = (rh.div_ceil(c), rw.div_ceil(c));

    let gpu = Gpu::open_device(opts.device)?;
    let dnn = Dnn::new(&gpu)?;
    let blas = Blas::new(&gpu)?;
    let ctx = Ctx { gpu: &gpu, dnn: &dnn, blas: &blas, pick: opts.pick, max_workspace: 1 << 30 };
    let mut net = Model::new(&ctx, &host, &contract.program, p)?;
    let head = net.heads();
    ensure!(head.classes == 2, "{} head classes", head.classes);
    let (head_cin, classes) = (head.cin, head.classes);

    let table16: [u16; 256] = std::array::from_fn(|i| Precision::Fp16.bits(recipe.ct.table()[i]));
    let ymax = (plan.starts[1].last().copied().unwrap_or(0) + p).min(shape[1]);
    let xmax = (plan.starts[2].last().copied().unwrap_or(0) + p).min(shape[2]);
    let (sy0, sx0) = (plan.starts[1][0], plan.starts[2][0]);
    let slab_max = p.min(shape[0]) * (ymax - sy0) * (xmax - sx0);
    u32_extent("input slab", slab_max)?;
    let acc = Accum {
        gpu: &gpu,
        shape,
        chunk: c,
        k0,
        ring,
        y0,
        x0,
        rh,
        rw,
        acc: gpu.alloc_fill(2 * plane * 4, 0)?,
        wacc: gpu.alloc_fill(plane * 4, 0)?,
        out8: gpu.alloc(c * rh * rw)?,
        flags: gpu.alloc(2 * ny * nx * 4)?,
        cutoff,
    };
    let gauss_buf = gpu.upload(&gauss)?;
    let table = gpu.upload(&table16)?;
    let slab = gpu.alloc(slab_max.max(1))?;
    let sum = gpu.alloc(classes * p.pow(3) * 4)?;
    let positions: Vec<[usize; 3]> = plan.patches().collect();
    let n_patches = positions.len();
    u32_extent("patch count", n_patches)?;
    let stat = gpu.alloc_fill(n_patches.max(1) * 4, 0)?;
    let writer = Writer::create(output, shape, c, opts.overwrite)?;
    gpu.sync()?;
    let load_s = t_load.elapsed().as_secs_f64();

    let t_run = Instant::now();
    let (mut read_s, mut finalize_s) = (0f64, 0f64);
    let mut written = 0;
    let mut next_flush = k0;
    let mut slab_z = usize::MAX;
    let mut dims = [0usize; 3];
    let mut host_slab: Vec<u8> = Vec::new();
    let mut skipped = Vec::new();
    let mut ran = vec![false; n_patches];
    let inv_n = 1.0 / flips as f32;
    for (i, &[z0, py, px]) in positions.iter().enumerate() {
        // finish every chunk row no later patch reaches, once no patch so far had a non-finite logit
        while next_flush < k1 && (next_flush + 1) * c <= z0 {
            let t = Instant::now();
            patch_status(&gpu, &stat, i, &positions)?;
            written += acc.flush(next_flush, &writer)?;
            finalize_s += t.elapsed().as_secs_f64();
            next_flush += 1;
        }
        if z0 != slab_z {
            let t = Instant::now();
            let zhi = (z0 + p).min(shape[0]);
            dims = [zhi - z0, ymax - sy0, xmax - sx0];
            host_slab.resize(dims.iter().product(), 0);
            vol.read_box([z0, sy0, sx0], [zhi, ymax, xmax], &mut host_slab)?;
            gpu.upload_into(&slab, 0, &host_slab)?;
            slab_z = z0;
            read_s += t.elapsed().as_secs_f64();
        }
        let origin = [0, py - sy0, px - sx0];
        let valid = [p.min(shape[0] - z0), p.min(shape[1] - py), p.min(shape[2] - px)];
        if recipe.skip_constant_patches && constant_patch(&host_slab, dims, origin, valid, &table16) {
            skipped.push([z0, py, px]);
            continue;
        }
        ran[i] = true;
        for (fi, &flip) in FLIPS[..flips].iter().enumerate() {
            let mut args = vec![net.input.arg::<u16>(), slab.view::<u8>(0, dims.iter().product()), table.arg::<u16>()];
            args.extend(
                [dims[1], dims[2], origin[0], origin[1], origin[2], p, 0, valid[0], 0, valid[1], 0, valid[2]]
                    .map(|v| Arg::U32(v as u32)),
            );
            args.extend([Arg::U32(net.in_c as u32), Arg::U32(flip)]);
            gpu.launch_1d("nn_prep", p.pow(3), &args)?;
            net.forward(&ctx)?;
            let h = net.heads();
            gpu.launch_1d(
                "nn_head_tta",
                p.pow(3),
                &[
                    sum.arg::<f32>(),
                    h.x.view::<u32>(0, p.pow(3) * head_cin / 2),
                    h.w.arg::<f32>(),
                    h.b.arg::<f32>(),
                    Arg::U32(head_cin as u32),
                    Arg::U32(classes as u32),
                    Arg::U32(p as u32),
                    Arg::U32(flip),
                    Arg::U32((fi == 0) as u32),
                ],
            )?;
        }
        gpu.launch_1d(
            "nn_check",
            p.pow(3),
            &[
                stat.view::<u32>(0, n_patches),
                sum.arg::<f32>(),
                Arg::F32(inv_n),
                Arg::U32(classes as u32),
                Arg::U32(p as u32),
                Arg::U32(i as u32),
            ],
        )?;
        gpu.launch_1d(
            "nn_fold",
            p.pow(3),
            &[
                acc.acc.arg::<f32>(),
                acc.wacc.arg::<f32>(),
                sum.arg::<f32>(),
                gauss_buf.arg::<f32>(),
                stat.view::<u32>(0, n_patches),
                Arg::U32(i as u32),
                Arg::U32(classes as u32),
                Arg::F32(inv_n),
                Arg::U32(p as u32),
                Arg::U32(z0 as u32),
                Arg::U32(py as u32),
                Arg::U32(px as u32),
                Arg::U32((k0 * c) as u32),
                Arg::U32(y0 as u32),
                Arg::U32(x0 as u32),
                Arg::U32((k1 * c).min(shape[0]) as u32 - (k0 * c) as u32),
                Arg::U32(rh as u32),
                Arg::U32(rw as u32),
                Arg::U32((ring * c) as u32),
            ],
        )?;
        if i + 1 == n_patches || (i + 1) % 32 == 0 {
            eprintln!("nnunet: {}/{n_patches} patches, {:.1} s", i + 1, t_run.elapsed().as_secs_f64());
        }
    }
    let words = patch_status(&gpu, &stat, n_patches, &positions)?;
    while next_flush < k1 {
        let t = Instant::now();
        written += acc.flush(next_flush, &writer)?;
        finalize_s += t.elapsed().as_secs_f64();
        next_flush += 1;
    }
    gpu.sync()?;
    let run_s = t_run.elapsed().as_secs_f64();
    let empty_logits: Vec<_> =
        (0..n_patches).filter(|&i| ran[i] && words[i] & NONZERO == 0).map(|i| positions[i]).collect();
    let lo = canvas.lo;
    let hi = canvas.hi;
    Ok(Report {
        model: contract.name.clone(),
        patches: n_patches,
        positions,
        patches_skipped: skipped.len(),
        skipped,
        patches_empty_logits: empty_logits.len(),
        empty_logits,
        flips,
        threshold: opts.threshold,
        canvas: [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]],
        chunk: c,
        chunks_written: written,
        load_s,
        run_s,
        s_per_patch: run_s / n_patches as f64,
        read_s,
        finalize_s,
        plans: net.plans(),
        algo_cache: algo_cache::status(),
        cudnn: Dnn::version(),
        stem_padded: net.in_c != 1,
        ptx_tier: gpu.inner.ptx_tier().to_string(),
        flops_per_patch: contract.flops_per_patch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nn_check` then `nn_fold` on one 32^3 two-class patch: all-zero logits fold nothing (villa's blender
    /// skips the patch), one non-zero logit folds the whole patch's weight, a sum that rounds to 0 in the
    /// fp16 mean counts as zero, and a NaN or an infinity sets the non-finite bit
    #[test]
    #[ignore = "needs an NVIDIA driver"]
    fn check_kernel_skips_zero_patches_and_flags_nonfinite() {
        let gpu = Gpu::open_device(0).unwrap();
        let p = 32usize;
        let vox = p.pow(3);
        let gauss = gpu.upload(&vec![0.5f32; vox]).unwrap();
        let run = |sum: &[f32]| -> (u32, f32) {
            let sum = gpu.upload(sum).unwrap();
            let stat = gpu.alloc_fill(4, 0).unwrap();
            let acc = gpu.alloc_fill(2 * vox * 4, 0).unwrap();
            let wacc = gpu.alloc_fill(vox * 4, 0).unwrap();
            let check = [
                stat.view::<u32>(0, 1),
                sum.arg::<f32>(),
                Arg::F32(0.125),
                Arg::U32(2),
                Arg::U32(p as u32),
                Arg::U32(0),
            ];
            gpu.launch_1d("nn_check", vox, &check).unwrap();
            let mut fold = vec![acc.arg::<f32>(), wacc.arg::<f32>(), sum.arg::<f32>(), gauss.arg::<f32>()];
            fold.extend([stat.view::<u32>(0, 1), Arg::U32(0), Arg::U32(2), Arg::F32(0.125)]);
            fold.extend([p, 0, 0, 0, 0, 0, 0, p, p, p, p].map(|v| Arg::U32(v as u32)));
            gpu.launch_1d("nn_fold", vox, &fold).unwrap();
            let stat: Vec<u32> = gpu.download(&stat).unwrap();
            let wacc: Vec<f32> = gpu.download(&wacc).unwrap();
            (stat[0], wacc.iter().sum())
        };

        let zeros = vec![0f32; 2 * vox];
        assert_eq!(run(&zeros), (0, 0.0));
        let mut one = zeros.clone();
        one[vox + 7] = 8.0;
        assert_eq!(run(&one), (NONZERO, 0.5 * vox as f32));
        let mut tiny = zeros.clone();
        tiny[3] = 1e-9;
        assert_eq!(run(&tiny), (0, 0.0));
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = zeros.clone();
            bad[2 * vox - 1] = v;
            assert_eq!(run(&bad).0 & NONFINITE, NONFINITE, "{v}");
        }
    }
}
