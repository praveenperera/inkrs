//! `vesuvius.predict --blend_in_memory` for a binary nnU-Net ResEnc-L (m7) on Metal: `nnunet.metal`'s matmul2d
//! convs and the glue that repeats the CUDA engine's rounding ([`crate::nnunet`] on Linux)
//!
//! Patches run in villa's grid order (z, then y, then x), each flip's fp16 forward and head summed in float32,
//! the flips' fp16 mean times villa's Gaussian blended into a float32 accumulator in patch order, as the CUDA engine
//! does. Three choices keep a 512^3 zone under 3 GB of unified memory (the CUDA engine's accumulators alone take
//! 4.7 GB there):
//! - the accumulator is a ring of `patch` z-slices over the canvas only, and a slice is finalised as soon as the patch
//!   rows pass it, into the output chunk row it belongs to;
//! - it holds the logit difference l1 - l0, all a binary softmax reads, instead of both class sums (`nn_fold` and
//!   `nn_finalize` say where that differs from the CUDA engine);
//! - the weight sum is not stored: the finaliser adds the Gaussians of the blended patches that cover a voxel in
//!   patch order, the same float32 adds the CUDA engine's accumulator makes, so it is that sum bit for bit
//!
//! The forward keeps fp16 channels-last activations in one buffer per encoder level (zero-padded, the decoder's skip)
//! plus two shared ones: R, every conv's unpadded output, and H, the padded intermediate of a block or the decoder's
//! upsampled half, whose border is cleared whenever it changes level

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure, eyre};
use half::f16;

use crate::ensure_supported;
use crate::error::Unsupported;
use crate::metal::gpu::{Arg, Buf, Fence, Gpu};
use crate::nnunet::grid::{Bbox, Plan};
pub use crate::nnunet::host::{FLIPS, Options, logit_cutoff};
use crate::nnunet::host::{NONFINITE, NONZERO, constant_patch};
use crate::nnunet::out::Writer;
use crate::nnunet::{Contract, Precision, Predict};
use crate::unet3d::Program;
use crate::unet3d::host::{HostConv, HostNet, u32_extent};
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
    /// Patches the model ran on whose mean logits were all 0, left out of the blend as villa's blender leaves them
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
    pub device: String,
    /// Bytes of the run's largest GPU buffers: activations, accumulators, the output row
    pub device_bytes: usize,
    /// The process's physical footprint after each phase and its lifetime peak, GB
    pub footprint_gb: Vec<(String, f64)>,
    pub peak_footprint_gb: f64,
    pub flops_per_patch: u64,
}

/// This process's physical footprint now and its lifetime peak, bytes (`proc_pid_rusage` RUSAGE_INFO_V4: what
/// `/usr/bin/time -l` reports as the peak memory footprint), so a run reports its own memory phase by phase
fn footprint() -> (u64, u64) {
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }
    const RUSAGE_INFO_V4: i32 = 4;
    // 16 bytes of uuid, then u64 counters: phys_footprint is the 8th, lifetime_max_phys_footprint the 29th
    let mut buf = [0u64; 2 + 40];
    // SAFETY: the buffer is larger than rusage_info_v4 (16 + 35 x 8 bytes) and lives across the call
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, RUSAGE_INFO_V4, buf.as_mut_ptr()) };
    if rc != 0 {
        return (0, 0);
    }
    (buf[2 + 7], buf[2 + 28])
}

/// The fp16 `[cout, 27, c]` weight of one source of a conv whose host weight is `[cout, 27, cin]`: channels `lo..hi`
fn source_bits(h: &HostConv, lo: usize, hi: usize) -> Vec<f16> {
    let taps = h.w.len() / (h.cout * h.cin);
    let mut out = Vec::with_capacity(h.cout * taps * (hi - lo));
    for row in h.w.chunks_exact(h.cin) {
        out.extend(row[lo..hi].iter().map(|&v| f16::from_f32(v)));
    }
    out
}

/// A conv's bias and norm on the device
struct Norm {
    bias: Option<Buf>,
    gamma: Buf,
    beta: Buf,
    eps: f32,
    leaky: Option<f32>,
}

impl Norm {
    fn new(gpu: &Gpu, h: &HostConv, leaky: Option<f32>) -> Result<Self> {
        let (g, b, eps) = h.norm.as_ref().ok_or_else(|| eyre!("conv without a norm"))?;
        Ok(Self {
            bias: h.bias.as_ref().map(|b| gpu.upload(b)).transpose()?,
            gamma: gpu.upload(g)?,
            beta: gpu.upload(b)?,
            eps: *eps,
            leaky,
        })
    }
}

/// A 3x3x3 conv on the device: its kernel instance, one weight per source, its norm
struct Conv3 {
    kernel: &'static str,
    bm: usize,
    bn: usize,
    stride: usize,
    /// Output edge
    e: usize,
    c: [usize; 2],
    cout: usize,
    w: [Buf; 2],
    norm: Norm,
}

impl Conv3 {
    /// The conv at output edge `e`, its input split into sources of `split` and `cin - split` channels (0 for one)
    fn new(gpu: &Gpu, h: &HostConv, stride: usize, e: usize, split: usize, leaky: Option<f32>) -> Result<Self> {
        let c = if split == 0 { [h.cin, 0] } else { [split, h.cin - split] };
        let (kernel, bm, bn) = match stride {
            1 => conv3_tile(e, c[1] > 0, h.cout)?,
            _ => conv3s2_tile(e, h.cout)?,
        };
        ensure_supported!(h.w.len() == h.cout * 27 * h.cin, "conv weight of {} values", h.w.len());
        let w0 = gpu.upload(&source_bits(h, 0, c[0]))?;
        let w1 = if c[1] > 0 { gpu.upload(&source_bits(h, c[0], h.cin))? } else { gpu.alloc(4)? };
        Ok(Self { kernel, bm, bn, stride, e, c, cout: h.cout, w: [w0, w1], norm: Norm::new(gpu, h, leaky)? })
    }

    fn tiles(&self) -> usize {
        let e = self.e;
        match self.stride {
            1 => e * (e * (e + 2)).div_ceil(self.bm),
            _ => e * e * e.div_ceil(self.bm),
        }
    }

    /// Encodes the conv of padded `x` (and `x1` for a second source) into R with its IN partials in `part`
    fn run(&self, gpu: &Gpu, x: &Buf, x1: Option<&Buf>, r: &Buf, part: &Buf, dummy: &Buf) -> Result<()> {
        let bias = self.norm.bias.as_ref().unwrap_or(dummy);
        let has_bias = self.norm.bias.is_some() as u32;
        let groups = [self.cout / self.bn, self.tiles(), 1];
        if self.stride == 1 {
            let x1 = x1.unwrap_or(x);
            let args = [
                x.arg(),
                self.w[0].arg(),
                x1.arg(),
                self.w[1].arg(),
                r.arg(),
                bias.arg(),
                part.arg(),
                Arg::U32(self.e as u32),
                Arg::U32(self.c[0] as u32),
                Arg::U32(self.c[1] as u32),
                Arg::U32(self.cout as u32),
                Arg::U32(has_bias),
                Arg::U32(1),
            ];
            return gpu.dispatch(self.kernel, groups, [128, 1, 1], &args);
        }
        let args = [
            x.arg(),
            self.w[0].arg(),
            r.arg(),
            bias.arg(),
            part.arg(),
            Arg::U32(self.e as u32),
            Arg::U32(self.c[0] as u32),
            Arg::U32(self.cout as u32),
            Arg::U32(has_bias),
        ];
        gpu.dispatch(self.kernel, groups, [128, 1, 1], &args)
    }
}

/// The stride-1 conv instance for output edge `e`: the fastest measured on the M4 Pro (`conv_bench`), BM 128 at
/// 192^3 to halve the partials there for 1-2% of its time
fn conv3_tile(e: usize, two: bool, cout: usize) -> Result<(&'static str, usize, usize)> {
    let pick = match (e, two) {
        (192.., _) => ("nn_conv3_128x32", 128, 32),
        (96.., _) => ("nn_conv3_64x64", 64, 64),
        (48.., false) => ("nn_conv3_32x128", 32, 128),
        (48.., true) => ("nn_conv3_64x32", 64, 32),
        (24.., false) => ("nn_conv3_32x128", 32, 128),
        (24.., true) => ("nn_conv3_64x64", 64, 64),
        (12.., _) => ("nn_conv3_64x32", 64, 32),
        _ => ("nn_conv3_32x64", 32, 64),
    };
    if cout.is_multiple_of(pick.2) {
        return Ok(pick);
    }
    ensure_supported!(
        cout.is_multiple_of(32),
        "a conv of {cout} output channels (the Metal convs take multiples of 32)"
    );
    Ok(("nn_conv3_64x32", 64, 32))
}

/// The stride-2 conv instance for output edge `e`, as [`conv3_tile`]
fn conv3s2_tile(e: usize, cout: usize) -> Result<(&'static str, usize, usize)> {
    let pick = match e {
        96.. => ("nn_conv3s2_32x64", 32, 64),
        48.. => ("nn_conv3s2_16x128", 16, 128),
        24.. => ("nn_conv3s2_32x64", 32, 64),
        _ => ("nn_conv3s2_16x64", 16, 64),
    };
    if cout.is_multiple_of(pick.2) {
        return Ok(pick);
    }
    ensure_supported!(
        cout.is_multiple_of(64),
        "a stride-2 conv of {cout} output channels (the Metal convs take multiples of 64)"
    );
    Ok(("nn_conv3s2_16x64", 16, 64))
}

/// A BasicBlockD's skip on the device
enum Skip {
    /// The block input itself (stride 1, same channels)
    Identity,
    /// AvgPool 2 of the input, then optionally the bias-free 1x1 conv and its norm
    Pool { proj: Option<(Buf, Norm)>, cin: usize },
}

struct Block {
    /// Level of the block's output
    level: usize,
    conv1: Conv3,
    conv2: Conv3,
    skip: Skip,
    leaky: f32,
}

struct Dec {
    level: usize,
    /// The transposed conv's GEMM weight `[8 cout, cin]` and bias
    up_w: Buf,
    up_b: Buf,
    up_cin: usize,
    up_cout: usize,
    conv: Conv3,
}

/// The m7 network on the device and its activations for one patch edge
struct Net {
    p: usize,
    /// Edge and channels per level
    levels: Vec<(usize, usize)>,
    stem_w: Buf,
    stem_norm: Norm,
    blocks: Vec<Block>,
    decs: Vec<Dec>,
    head_w: Buf,
    head_b: Buf,
    head_cin: usize,
    classes: usize,
    /// The padded 1-channel input
    input: Buf,
    /// Each level's padded output (the decoder's skip)
    x: Vec<Buf>,
    r: Buf,
    h: Buf,
    /// Where in H the pooled skip input and the projection's output go: past the level's intermediate
    h_pool: usize,
    h_proj: usize,
    part: Buf,
    st: Buf,
    rst: Buf,
    dummy: Buf,
    bytes: usize,
}

/// Bytes of a padded `[e + 2]^3 x c` fp16 buffer
fn padded_bytes(e: usize, c: usize) -> usize {
    (e + 2).pow(3) * c * 2
}

impl Net {
    /// Uploads `host`'s weights and frees it before allocating the activations, so the float32 host copy and the
    /// activations are never resident together
    fn new(gpu: &Gpu, host: HostNet, prog: &Program, p: usize) -> Result<Self> {
        ensure_supported!(prog.stem.len() == 1 && prog.stem[0].cin == 1, "a stem other than one 1-channel conv");
        let stem = &host.stem[0];
        ensure_supported!(stem.cout == 32, "a stem of {} channels (the Metal stem writes 32)", stem.cout);
        let stem_w: Vec<f32> = stem.w.iter().map(|&v| Precision::Fp16.round(v)).collect();
        let mut levels = vec![(p, stem.cout)];
        let mut blocks = Vec::new();
        let mut hb = 0usize;
        let mut rmax = p.pow(3) * stem.cout * 2;
        let mut parts = p.pow(3).div_ceil(128) * stem.cout;
        let mut level = 0usize;
        for (bs, hbk) in prog.stages.iter().flatten().zip(&host.blocks) {
            let s = bs.conv1.stride;
            ensure_supported!(s[0] == s[1] && s[1] == s[2] && s[0] <= 2, "conv stride {s:?}");
            ensure_supported!(bs.conv1.kernel == [3; 3] && bs.conv2.kernel == [3; 3], "a block kernel other than 3^3");
            let stride = s[0];
            if stride == 2 {
                level += 1;
                ensure_supported!(
                    levels[level - 1].0 % 2 == 0,
                    "an odd edge {} under a stride-2 conv",
                    levels[level - 1].0
                );
                levels.push((levels[level - 1].0 / 2, bs.conv1.cout));
            }
            let (e, cin) = (levels[level].0, bs.conv1.cin);
            ensure_supported!(bs.conv1.cout == levels[level].1 && bs.conv2.cout == levels[level].1, "block channels");
            let conv1 = Conv3::new(gpu, &hbk.conv1, stride, e, 0, bs.conv1.leaky)?;
            let conv2 = Conv3::new(gpu, &hbk.conv2, 1, e, 0, None)?;
            parts = parts.max(conv1.tiles() * conv1.cout).max(conv2.tiles() * conv2.cout);
            rmax = rmax.max(e.pow(3) * conv1.cout * 2);
            hb = hb.max(padded_bytes(e, conv1.cout));
            let skip = match (bs.skip.pool.axes(), &hbk.proj) {
                ([1, 1, 1], None) => {
                    ensure_supported!(stride == 1 && cin == conv2.cout, "an identity skip across a change of shape");
                    Skip::Identity
                }
                ([2, 2, 2], proj) => {
                    ensure_supported!(stride == 2, "a pooled skip on a stride-1 block");
                    let proj = match proj {
                        Some(pj) => {
                            ensure_supported!(pj.bias.is_none() && pj.cin == cin, "a skip projection with a bias");
                            let w: Vec<f16> = pj.w.iter().map(|&v| f16::from_f32(v)).collect();
                            parts = parts.max(e.pow(3).div_ceil(gemm_tile(pj.cout).1) * pj.cout);
                            Some((gpu.upload(&w)?, Norm::new(gpu, pj, None)?))
                        }
                        None => {
                            ensure_supported!(cin == conv2.cout, "a pooled skip across a change of channels");
                            None
                        }
                    };
                    Skip::Pool { proj, cin }
                }
                (axes, _) => return Err(Unsupported(format!("skip pool {axes:?}")).into()),
            };
            blocks.push(Block { level, conv1, conv2, skip, leaky: bs.leaky });
        }
        ensure_supported!(
            levels.len() == prog.decoder.len() + 1,
            "{} levels for {} decoder stages",
            levels.len(),
            prog.decoder.len()
        );

        // the pooled input and the projection's output share H past the block's intermediate (H is free then)
        let mut h_pool = 0usize;
        let mut h_proj = 0usize;
        for b in &blocks {
            if let Skip::Pool { cin, proj } = &b.skip {
                let (e, c) = levels[b.level];
                let pool_at = padded_bytes(e, c).next_multiple_of(256);
                let proj_at = (pool_at + e.pow(3) * cin * 2).next_multiple_of(256);
                h_pool = h_pool.max(pool_at);
                h_proj = h_proj.max(proj_at);
                hb = hb.max(proj_at + if proj.is_some() { e.pow(3) * c * 2 } else { 0 });
            }
        }

        let mut decs = Vec::new();
        for (d, (ds, hd)) in prog.decoder.iter().zip(&host.decs).enumerate() {
            let level = levels.len() - 2 - d;
            let (e, c) = levels[level];
            let (ci, co) = (hd.tcin, ds.transp.cout);
            ensure_supported!(
                ds.transp.kernel == [2; 3] && ds.transp.stride == [2; 3],
                "a transposed conv other than k2 s2"
            );
            ensure_supported!(
                ds.convs.len() == 1 && ds.convs[0].cin == co + c,
                "a decoder stage other than one conv on the concat"
            );
            ensure_supported!((8 * co) % 64 == 0, "a transposed conv of {co} channels");
            let mut wt = vec![f16::ZERO; 8 * co * ci];
            for i in 0..ci {
                for col in 0..8 * co {
                    wt[col * ci + i] = f16::from_f32(hd.g[i * 8 * co + col]);
                }
            }
            let conv = Conv3::new(gpu, &hd.convs[0], 1, e, co, ds.convs[0].leaky)?;
            parts = parts.max(conv.tiles() * conv.cout);
            rmax = rmax.max(e.pow(3) * conv.cout * 2);
            hb = hb.max(padded_bytes(e, co));
            decs.push(Dec { level, up_w: gpu.upload(&wt)?, up_b: gpu.upload(&hd.tb)?, up_cin: ci, up_cout: co, conv });
        }
        let classes = host.head_w.len() / host.head_cin;
        ensure_supported!(
            host.head_cin.is_multiple_of(2) && host.head_cin == levels[0].1,
            "a head of {} channels",
            host.head_cin
        );

        let (stem_w, stem_norm) = (gpu.upload(&stem_w)?, Norm::new(gpu, stem, prog.stem[0].leaky)?);
        let (head_w, head_b, head_cin) = (gpu.upload(&host.head_w)?, gpu.upload(&host.head_b)?, host.head_cin);
        drop(host);

        let x: Vec<Buf> = levels.iter().map(|&(e, c)| gpu.alloc(padded_bytes(e, c))).collect::<Result<_>>()?;
        let net = Self {
            p,
            stem_w,
            stem_norm,
            blocks,
            decs,
            head_w,
            head_b,
            head_cin,
            classes,
            input: gpu.alloc(padded_bytes(p, 1))?,
            bytes: 0,
            x,
            r: gpu.alloc(rmax)?,
            h: gpu.alloc(hb)?,
            h_pool,
            h_proj,
            part: gpu.alloc(parts * 16)?,
            st: gpu.alloc(512 * 8)?,
            rst: gpu.alloc(512 * 8)?,
            dummy: gpu.alloc(16)?,
            levels,
        };
        Ok(Self { bytes: net.buffer_bytes(), ..net })
    }

    /// Bytes of activation buffers
    fn buffer_bytes(&self) -> usize {
        let x: usize = self.x.iter().map(Buf::bytes).sum();
        x + self.r.bytes() + self.h.bytes() + self.part.bytes() + self.input.bytes()
    }

    /// Reduces `tiles` partials of `cols` columns into `st` ([mean, invstd] per column)
    fn stats(&self, gpu: &Gpu, st: &Buf, tiles: usize, cols: usize, eps: f32) -> Result<()> {
        let args = [st.arg(), self.part.arg(), Arg::U32(tiles as u32), Arg::U32(cols as u32), Arg::F32(eps)];
        gpu.dispatch("nn_in_stats", [cols, 1, 1], [256, 1, 1], &args)
    }

    /// Encodes nn_apply: `out` (padded per `out_pad`) = leaky?(affine(R) + residual)
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        gpu: &Gpu,
        n: &Norm,
        e: usize,
        c: usize,
        res: Res,
        out: Arg,
        out_pad: u32,
        act: Option<f32>,
    ) -> Result<()> {
        let d = &self.dummy;
        let (mode, r, pad, rn) = match res {
            Res::None => (0u32, d.arg(), 0u32, None),
            Res::Plain(r, pad) => (1, r, pad, None),
            Res::Normed(r, norm) => (2, r, 0, Some(norm)),
        };
        let (rg, rb) = rn.map_or((d.arg(), d.arg()), |nm| (nm.gamma.arg(), nm.beta.arg()));
        let args = [
            out,
            self.r.arg(),
            self.st.arg(),
            n.gamma.arg(),
            n.beta.arg(),
            r,
            self.rst.arg(),
            rg,
            rb,
            Arg::U32(e as u32),
            Arg::U32(c as u32),
            Arg::U32(mode),
            Arg::U32(pad),
            Arg::U32(act.is_some() as u32),
            Arg::F32(act.unwrap_or(0.0)),
            Arg::U32(out_pad),
        ];
        ensure!(c.is_multiple_of(8), "nn_apply takes channels in groups of 8, not {c}");
        gpu.dispatch_1d("nn_apply", e.pow(3) * c / 8, &args)
    }

    /// Clears the border of H laid out as padded `[e + 2]^3 x c`, which other levels' layouts wrote into
    fn clear_h_border(&self, gpu: &Gpu, e: usize, c: usize) -> Result<()> {
        let ep = e + 2;
        let n = (2 * ep * ep + 2 * e * ep + 2 * e * e) * c / 2;
        gpu.dispatch_1d("nn_border", n, &[self.h.arg(), Arg::U32(e as u32), Arg::U32(c as u32)])
    }

    /// One forward of the padded input into R: the last decoder stage's output, unpadded `[p^3, head_cin]`
    fn forward(&self, gpu: &Gpu) -> Result<()> {
        let p = self.p;
        let (r, part) = (&self.r, &self.part);
        // stem into level 0's x
        let tiles = p.pow(3).div_ceil(128);
        let bias = self.stem_norm.bias.as_ref().unwrap_or(&self.dummy);
        let args = [r.arg(), self.input.arg(), self.stem_w.arg(), bias.arg(), part.arg(), Arg::U32(p as u32)];
        gpu.dispatch("nn_stem", [tiles, 1, 1], [128, 1, 1], &args)?;
        self.stats(gpu, &self.st, tiles, 32, self.stem_norm.eps)?;
        self.apply(gpu, &self.stem_norm, p, 32, Res::None, self.x[0].arg(), 1, self.stem_norm.leaky)?;

        let last = self.blocks.len() - 1;
        let mut h_level = usize::MAX;
        for (bi, b) in self.blocks.iter().enumerate() {
            let (e, c) = self.levels[b.level];
            let input = if b.conv1.stride == 2 { &self.x[b.level - 1] } else { &self.x[b.level] };
            if h_level != b.level {
                self.clear_h_border(gpu, e, c)?;
                h_level = b.level;
            }
            b.conv1.run(gpu, input, None, r, part, &self.dummy)?;
            self.stats(gpu, &self.st, b.conv1.tiles(), c, b.conv1.norm.eps)?;
            self.apply(gpu, &b.conv1.norm, e, c, Res::None, self.h.arg(), 1, b.conv1.norm.leaky)?;
            b.conv2.run(gpu, &self.h, None, r, part, &self.dummy)?;
            self.stats(gpu, &self.st, b.conv2.tiles(), c, b.conv2.norm.eps)?;
            let res = match &b.skip {
                Skip::Identity => Res::Plain(input.arg(), 1),
                Skip::Pool { proj, cin } => {
                    let pa = self.h.at::<u8>(self.h_pool);
                    let args = [pa, input.arg(), Arg::U32(e as u32), Arg::U32(*cin as u32)];
                    gpu.dispatch_1d("nn_pool2", e.pow(3) * cin / 2, &args)?;
                    match proj {
                        None => Res::Plain(self.h.at::<u8>(self.h_pool), 0),
                        Some((w, norm)) => {
                            let (name, bm, bn) = gemm_tile(c);
                            let m = e.pow(3);
                            let args = [
                                self.h.at::<u8>(self.h_pool),
                                w.arg(),
                                self.h.at::<u8>(self.h_proj),
                                part.arg(),
                                Arg::U32(m as u32),
                                Arg::U32(c as u32),
                                Arg::U32(*cin as u32),
                            ];
                            gpu.dispatch(name, [c / bn, m.div_ceil(bm), 1], [128, 1, 1], &args)?;
                            self.stats(gpu, &self.rst, m.div_ceil(bm), c, norm.eps)?;
                            Res::Normed(self.h.at::<u8>(self.h_proj), norm)
                        }
                    }
                }
            };
            // the bottleneck's output goes unpadded into R, in place, for the first transposed conv
            let (out, pad) = if bi == last { (r.arg(), 0) } else { (self.x[b.level].arg(), 1) };
            self.apply(gpu, &b.conv2.norm, e, c, res, out, pad, Some(b.leaky))?;
        }

        for d in &self.decs {
            let (e, c) = self.levels[d.level];
            let co = d.up_cout;
            self.clear_h_border(gpu, e, co)?;
            let m = (e / 2).pow(3);
            ensure!(co.is_multiple_of(8), "nn_up stores channels in groups of 8, not {co}");
            let args = [
                r.arg(),
                d.up_w.arg(),
                self.h.arg(),
                d.up_b.arg(),
                Arg::U32((e / 2) as u32),
                Arg::U32(d.up_cin as u32),
                Arg::U32(co as u32),
            ];
            gpu.dispatch("nn_up_64x64", [8 * co / 64, m.div_ceil(64), 1], [128, 1, 1], &args)?;
            d.conv.run(gpu, &self.h, Some(&self.x[d.level]), r, part, &self.dummy)?;
            let cout = d.conv.cout;
            self.stats(gpu, &self.st, d.conv.tiles(), cout, d.conv.norm.eps)?;
            ensure!(cout == c, "decoder conv of {cout} channels at a level of {c}");
            self.apply(gpu, &d.conv.norm, e, cout, Res::None, r.arg(), 0, d.conv.norm.leaky)?;
        }
        Ok(())
    }
}

/// The 1x1 projection GEMM instance for `n` output channels
fn gemm_tile(n: usize) -> (&'static str, usize, usize) {
    if n.is_multiple_of(64) { ("nn_gemm_in_64x64", 64, 64) } else { ("nn_gemm_in_64x32", 64, 32) }
}

/// An apply's residual
enum Res<'a> {
    None,
    /// Added as stored, padded (1) or not (0)
    Plain(Arg<'a>, u32),
    /// A bias-free raw conv output, unpadded, normalised on load with `rst` and the norm's affine
    Normed(Arg<'a>, &'a Norm),
}

/// The logit-difference accumulator over the canvas, a ring of `ring` z-slices, and the chunk row it finalises into
struct Accum {
    canvas: Bbox,
    ring: usize,
    chunk: usize,
    /// The output chunk rows' y-x extent, on the chunk grid (their corner is in `fin`)
    rh: usize,
    rw: usize,
    acc: Buf,
    out8: Buf,
    flags: Buf,
    first: Buf,
    fin: Buf,
    grid: Buf,
    starts: [Buf; 3],
}

impl Accum {
    fn elems(&self) -> usize {
        (self.canvas.hi[1] - self.canvas.lo[1]) * (self.canvas.hi[2] - self.canvas.lo[2])
    }

    fn chunks_yx(&self) -> (usize, usize) {
        (self.rh.div_ceil(self.chunk), self.rw.div_ceil(self.chunk))
    }

    /// Encodes the finalisation of global slices `[z0, z1)`, all in the chunk row starting at `rz0`, and clears
    /// their accumulator slots
    fn finalize(&self, gpu: &Gpu, stat: &Buf, gauss: &Buf, z0: usize, z1: usize, rz0: usize) -> Result<()> {
        let fixed = [
            self.acc.arg(),
            self.starts[0].arg(),
            self.starts[1].arg(),
            self.starts[2].arg(),
            stat.arg(),
            gauss.arg(),
            self.fin.arg(),
            self.grid.arg(),
        ];
        if z0 == rz0 {
            let (ny, nx) = self.chunks_yx();
            let args: Vec<Arg> = [self.first.arg()].into_iter().chain(fixed).chain([Arg::U32(z0 as u32)]).collect();
            gpu.dispatch_1d("nn_first", ny * nx, &args)?;
        }
        let tail = [Arg::U32(z0 as u32), Arg::U32((z1 - z0) as u32), Arg::U32(rz0 as u32)];
        let head = [self.out8.arg(), self.flags.arg(), self.first.arg()];
        let args: Vec<Arg> = head.into_iter().chain(fixed).chain(tail).collect();
        gpu.dispatch_1d("nn_finalize", (z1 - z0) * self.rh * self.rw, &args)?;
        // the canvas slots among them
        let (lo, hi) = (z0.max(self.canvas.lo[0]), z1.min(self.canvas.hi[0]));
        if lo < hi {
            let args = [
                self.acc.arg(),
                Arg::U32(1),
                Arg::U32(self.ring as u32),
                Arg::U32(self.elems() as u32),
                Arg::U32(((lo - self.canvas.lo[0]) % self.ring) as u32),
                Arg::U32((hi - lo) as u32),
            ];
            gpu.dispatch_1d("nn_zero_slices", (hi - lo) * self.elems(), &args)?;
        }
        Ok(())
    }
}

/// The status words of the first `n` patches; fails when one of them had a non-finite logit
fn patch_status(gpu: &Gpu, stat: &Buf, n: usize, positions: &[[usize; 3]]) -> Result<Vec<u32>> {
    let mut words = vec![0u32; n];
    gpu.download(stat, 0, &mut words)?;
    let bad: Vec<_> = (0..n).filter(|&i| words[i] & NONFINITE != 0).map(|i| positions[i]).collect();
    ensure!(bad.is_empty(), "non-finite logits in {} patches, the first at {:?}", bad.len(), bad[0]);
    Ok(words)
}

/// Runs the export at `model` (`.safetensors` with its `.json` sidecar) over `input` (a uint8 zarr array, or a
/// group whose array `0` is read) into a new zarr array at `output`, as villa's `vesuvius.predict --blend_in_memory
/// --mode binary` writes it
pub fn run(model: &Path, input: &Path, output: &Path, opts: &Options) -> Result<Report> {
    let t_load = Instant::now();
    let gpu = Gpu::open(opts.device)?;
    gpu.ready()?;
    let mut phases = vec![("metal".to_string(), footprint().0)];
    let (ck, sidecar) = Checkpoint::read(model)?;
    phases.push(("checkpoint".into(), footprint().0));
    let contract = Contract::from_sidecar(&sidecar)?;
    let recipe: Predict = contract.predict()?;
    let p = contract.patch_size[0];
    let gauss = ck.take(&recipe.gaussian_key, &[p, p, p])?;
    ensure!(gauss.iter().all(|&v| (0.0..=1.0).contains(&v)), "blend weights outside [0, 1]");
    let host = HostNet::take(&ck, &contract.program, Precision::Fp16)?;
    ck.finish()?;
    drop(ck);
    phases.push(("host weights".into(), footprint().0));

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

    // the output chunk rows the canvas touches; the accumulators cover the canvas only
    let canvas = plan.canvas();
    let (k0, k1) = (canvas.lo[0] / c, canvas.hi[0].div_ceil(c));
    let y0 = canvas.lo[1] / c * c;
    let x0 = canvas.lo[2] / c * c;
    let rh = (canvas.hi[1].div_ceil(c) * c).min(shape[1]) - y0;
    let rw = (canvas.hi[2].div_ceil(c) * c).min(shape[2]) - x0;
    let (ch, cw) = (canvas.hi[1] - canvas.lo[1], canvas.hi[2] - canvas.lo[2]);
    let ring = p.min(canvas.hi[0] - canvas.lo[0]);
    // one float32 plane: the logit difference (nn_fold)
    let gb = (ring * ch * cw * 4) as f64 / 1e9;
    ensure_supported!(
        gb.total_cmp(&opts.max_accum_gb).is_le(),
        "the accumulators need {gb:.1} GB, over the {} GB cap; use a smaller --bbox",
        opts.max_accum_gb
    );
    u32_extent("accumulator", ring * ch * cw)?;
    u32_extent("output row", c * rh * rw)?;
    let (ny, nx) = (rh.div_ceil(c), rw.div_ceil(c));

    let net = Net::new(&gpu, host, &contract.program, p)?;
    ensure!(net.classes == 2, "{} head classes", net.classes);
    phases.push(("network".into(), footprint().0));

    let table16: [u16; 256] = std::array::from_fn(|i| Precision::Fp16.bits(recipe.ct.table()[i]));
    let positions: Vec<[usize; 3]> = plan.patches().collect();
    let n_patches = positions.len();
    u32_extent("patch count", n_patches)?;
    let u32s = |v: &[usize]| v.iter().map(|&x| x as u32).collect::<Vec<u32>>();
    let fin: [u32; 14] = [
        canvas.lo[0] as u32,
        canvas.lo[1] as u32,
        canvas.lo[2] as u32,
        (canvas.hi[0] - canvas.lo[0]) as u32,
        ch as u32,
        cw as u32,
        ring as u32,
        y0 as u32,
        x0 as u32,
        rh as u32,
        rw as u32,
        c as u32,
        cutoff.is_some() as u32,
        cutoff.unwrap_or(0.0).to_bits(),
    ];
    let grid = [plan.starts[0].len(), plan.starts[1].len(), plan.starts[2].len(), p];
    let acc = Accum {
        canvas,
        ring,
        chunk: c,
        rh,
        rw,
        acc: gpu.alloc(ring * ch * cw * 4)?,
        out8: gpu.alloc(c * rh * rw)?,
        flags: gpu.alloc(2 * ny * nx * 4)?,
        first: gpu.alloc(ny * nx * 4)?,
        fin: gpu.upload(&fin)?,
        grid: gpu.upload(&u32s(&grid))?,
        starts: [
            gpu.upload(&u32s(&plan.starts[0]))?,
            gpu.upload(&u32s(&plan.starts[1]))?,
            gpu.upload(&u32s(&plan.starts[2]))?,
        ],
    };
    let gauss_buf = gpu.upload(&gauss)?;
    let table = gpu.upload(&table16.map(f16::from_bits))?;
    let slabs = [gpu.alloc(p.pow(3))?, gpu.alloc(p.pow(3))?];
    let mut slab_fence: [Option<Fence>; 2] = [None, None];
    let sum = gpu.alloc(net.classes * p.pow(3) * 4)?;
    let stat = gpu.alloc(n_patches.max(1) * 4)?;
    let writer = Writer::create(output, shape, c, opts.overwrite)?;
    let device_bytes = net.bytes
        + [&acc.acc, &acc.out8, &gauss_buf, &sum, &slabs[0], &slabs[1]].iter().map(|b| b.bytes()).sum::<usize>();
    gpu.ready()?;
    phases.push(("accumulators".into(), footprint().0));
    let load_s = t_load.elapsed().as_secs_f64();

    let t_run = Instant::now();
    let (mut read_s, mut finalize_s) = (0f64, 0f64);
    let mut written = 0;
    let (oz0, oz1) = (k0 * c, (k1 * c).min(shape[0]));
    let mut fin_z = oz0;
    let mut skipped = Vec::new();
    let mut ran = vec![false; n_patches];
    let inv_n = 1.0 / flips as f32;
    // finalise every slice below `zt`, writing each chunk row once its last slice is done
    let finalize_until = |zt: usize, done: usize, fin_z: &mut usize, written: &mut usize| -> Result<f64> {
        let t = Instant::now();
        while *fin_z < zt {
            let rz0 = *fin_z / c * c;
            let row_end = (rz0 + c).min(shape[0]);
            let seg = zt.min(row_end).min(*fin_z + 32);
            acc.finalize(&gpu, &stat, &gauss_buf, *fin_z, seg, rz0)?;
            *fin_z = seg;
            if seg < row_end {
                continue;
            }
            // a row is written only once no patch so far had a non-finite logit
            patch_status(&gpu, &stat, done, &positions)?;
            let rows = row_end - rz0;
            let mut flags = vec![0u32; 2 * ny * nx];
            gpu.download(&acc.flags, 0, &mut flags)?;
            // villa's _write_blended_chunk: a chunk with no weight above 0 is skipped, and apply_finalization
            // reports a chunk whose logits are all allclose to its first as empty
            let keep: Vec<bool> = (0..ny * nx).map(|i| flags[2 * i] != 0 && flags[2 * i + 1] != 0).collect();
            // SAFETY: the download above synchronised the GPU, and nothing is queued until the row is written
            let data = unsafe { &acc.out8.host::<u8>()[..rows * rh * rw] };
            *written += writer.write_row([rz0, y0, x0], [rows, rh, rw], data, &keep)?;
            // SAFETY: as above
            unsafe { acc.flags.host_mut::<u32>().fill(0) };
        }
        Ok(t.elapsed().as_secs_f64())
    };
    for (i, &[z0, py, px]) in positions.iter().enumerate() {
        finalize_s += finalize_until(z0, i, &mut fin_z, &mut written)?;
        let t = Instant::now();
        let slot = i % 2;
        if let Some(f) = slab_fence[slot].take() {
            f.wait()?;
        }
        let valid = [p.min(shape[0] - z0), p.min(shape[1] - py), p.min(shape[2] - px)];
        let n = valid.iter().product::<usize>();
        // SAFETY: the fence above covers every kernel that read this slab
        let host_slab = unsafe { &mut slabs[slot].host_mut::<u8>()[..n] };
        vol.read_box([z0, py, px], [z0 + valid[0], py + valid[1], px + valid[2]], host_slab)?;
        read_s += t.elapsed().as_secs_f64();
        if recipe.skip_constant_patches && constant_patch(host_slab, valid, [0; 3], valid, &table16) {
            skipped.push([z0, py, px]);
            continue;
        }
        ran[i] = true;
        for (fi, &flip) in FLIPS[..flips].iter().enumerate() {
            let mut args = vec![net.input.arg(), slabs[slot].arg(), table.arg()];
            args.extend([valid[1], valid[2], 0, 0, 0, p, valid[0], valid[1], valid[2], 1].map(|v| Arg::U32(v as u32)));
            args.push(Arg::U32(flip));
            gpu.dispatch_1d("nn_prep", p.pow(3), &args)?;
            net.forward(&gpu)?;
            let args = [
                sum.arg(),
                net.r.arg(),
                net.head_w.arg(),
                net.head_b.arg(),
                Arg::U32(net.head_cin as u32),
                Arg::U32(net.classes as u32),
                Arg::U32(p as u32),
                Arg::U32(flip),
                Arg::U32((fi == 0) as u32),
            ];
            gpu.dispatch_1d("nn_head", p.pow(3), &args)?;
        }
        let args = [
            stat.arg(),
            sum.arg(),
            Arg::F32(inv_n),
            Arg::U32(net.classes as u32),
            Arg::U32(p as u32),
            Arg::U32(i as u32),
        ];
        gpu.dispatch_1d("nn_check", p.pow(3), &args)?;
        let mut args = vec![acc.acc.arg(), sum.arg(), gauss_buf.arg(), stat.arg(), Arg::U32(i as u32), Arg::F32(inv_n)];
        let geo = [p, z0, py, px, canvas.lo[0], canvas.lo[1], canvas.lo[2], canvas.hi[0] - canvas.lo[0], ch, cw, ring];
        args.extend(geo.map(|v| Arg::U32(v as u32)));
        gpu.dispatch_1d("nn_fold", p.pow(3), &args)?;
        slab_fence[slot] = gpu.flush()?;
        if i + 1 == n_patches || (i + 1) % 32 == 0 {
            eprintln!("nnunet: {}/{n_patches} patches, {:.1} s", i + 1, t_run.elapsed().as_secs_f64());
        }
    }
    let words = patch_status(&gpu, &stat, n_patches, &positions)?;
    finalize_s += finalize_until(oz1, n_patches, &mut fin_z, &mut written)?;
    gpu.synchronize()?;
    let run_s = t_run.elapsed().as_secs_f64();
    // per-kernel GPU time, with INKRS_METAL_PROFILE set
    gpu.print_profile()?;
    let empty_logits: Vec<_> =
        (0..n_patches).filter(|&i| ran[i] && words[i] & NONZERO == 0).map(|i| positions[i]).collect();
    let (lo, hi) = (canvas.lo, canvas.hi);
    let (now, peak) = footprint();
    phases.push(("end".into(), now));
    let gb = |b: u64| b as f64 / 1e9;
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
        device: gpu.name(),
        device_bytes,
        footprint_gb: phases.into_iter().map(|(k, v)| (k, gb(v))).collect(),
        peak_footprint_gb: gb(peak),
        flops_per_patch: contract.flops_per_patch,
    })
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::metal::gpu::{Arg, Buf, Gpu};

    /// A deterministic value in [-1, 1) for index `i` of stream `k`
    fn wave(i: usize, k: usize) -> f32 {
        let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        let h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// `n` fp16 values of stream `k` scaled by `s`
    fn halves(n: usize, k: usize, s: f32) -> Vec<f16> {
        (0..n).map(|i| f16::from_f32(wave(i, k) * s)).collect()
    }

    /// A padded [e+2]^3 x c buffer holding `x` [e]^3 x c in its interior, zero elsewhere
    fn padded(x: &[f16], e: usize, c: usize) -> Vec<f16> {
        let ep = e + 2;
        let mut out = vec![f16::ZERO; ep * ep * ep * c];
        for z in 0..e {
            for y in 0..e {
                for xx in 0..e {
                    let s = ((z * e + y) * e + xx) * c;
                    let d = (((z + 1) * ep + y + 1) * ep + xx + 1) * c;
                    out[d..d + c].copy_from_slice(&x[s..s + c]);
                }
            }
        }
        out
    }

    /// f64 3x3x3 pad-1 conv with `stride` of `x` [ei]^3 x c (weights [o, 27, c]) at output voxel `[z, y, x]`, channel k
    fn conv_ref(x: &[f16], w: &[f16], ei: usize, c: usize, stride: usize, [z, y, xx]: [usize; 3], k: usize) -> f64 {
        let mut s = 0f64;
        for t in 0..27 {
            let (dz, dy, dx) = (t / 9, (t / 3) % 3, t % 3);
            let (iz, iy, ix) =
                ((stride * z + dz) as isize - 1, (stride * y + dy) as isize - 1, (stride * xx + dx) as isize - 1);
            if [iz, iy, ix].iter().any(|&v| v < 0 || v >= ei as isize) {
                continue;
            }
            let base = ((iz as usize * ei + iy as usize) * ei + ix as usize) * c;
            for ci in 0..c {
                s += x[base + ci].to_f64() * w[(k * 27 + t) * c + ci].to_f64();
            }
        }
        s
    }

    /// One fp16 ulp at magnitude `v`
    fn ulp16(v: f64) -> f64 {
        let v = v.abs().max(6.1e-5);
        2f64.powi(v.log2().floor() as i32 - 10)
    }

    /// Checks a conv's stored output [eo]^3 x o against `want(v, k)` = (f64 conv, bias) and the f64 IN statistics
    /// of every column against `stats`. The store is r16(r16(conv) + bias): FP32 accumulation lands within a hair of
    /// the f64 conv, so the first rounding may differ by one ulp of the conv at a near-tie and the second adds half
    /// an ulp of the result
    fn check(
        got: &[f16],
        stats: &[[f32; 2]],
        eo: usize,
        o: usize,
        want: impl Fn(usize, usize) -> (f64, f64),
        what: &str,
    ) {
        let vox = eo.pow(3);
        assert_eq!(got.len(), vox * o, "{what}: output length");
        let (mut worst, mut flips) = (0f64, 0usize);
        for v in 0..vox {
            for k in 0..o {
                let (a, b) = want(v, k);
                let r = f16::from_f64(a).to_f64() + b;
                let g = got[v * o + k].to_f64();
                let d = (g - r).abs();
                let tol = ulp16(a) + 0.5 * ulp16(r) + 1e-6;
                worst = worst.max(d / tol);
                flips += (d > 0.5 * ulp16(r) + 1e-6) as usize;
                assert!(g.is_finite() && d <= tol, "{what}: voxel {v} channel {k}: {g} against {r} (conv {a})");
            }
        }
        // a near-tie is rare; many one-ulp flips would mean the accumulation is not FP32
        assert!(flips * 100 <= vox * o, "{what}: {flips} of {} values a whole ulp off", vox * o);
        for k in 0..o {
            let col = || (0..vox).map(|v| got[v * o + k].to_f64());
            let mean = col().sum::<f64>() / vox as f64;
            let var = col().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vox as f64;
            let invstd = 1.0 / (var + 1e-5).sqrt();
            let [gm, gi] = stats[k];
            assert!(
                (gm as f64 - mean).abs() <= 1e-5 * (mean.abs() + var.sqrt())
                    && ((gi as f64 - invstd) / invstd).abs() <= 1e-4,
                "{what}: channel {k} stats {gm} {gi} against {mean} {invstd}"
            );
        }
        eprintln!("{what}: worst error {worst:.3} of the tolerance, {flips} near-tie flips of {}", vox * o);
    }

    fn stats(gpu: &Gpu, part: &Buf, tiles: usize, cols: usize) -> Vec<[f32; 2]> {
        let st = gpu.alloc_fill(cols * 8, 0xFF).unwrap();
        gpu.dispatch(
            "nn_in_stats",
            [cols, 1, 1],
            [256, 1, 1],
            &[st.arg(), part.arg(), tiles.into(), cols.into(), Arg::F32(1e-5)],
        )
        .unwrap();
        let mut out = vec![[0f32; 2]; cols];
        gpu.download(&st, 0, &mut out).unwrap();
        out
    }

    /// A two-source stride-1 conv with bias and a stride-2 conv, against f64, every tile shape the host may pick
    #[test]
    fn convs_match_f64() {
        let gpu = Gpu::open(0).unwrap();
        let (e, c0, c1) = (7usize, 16usize, 32usize);
        for (name, bm, bn, o) in [
            ("nn_conv3_64x32", 64, 32, 32),
            ("nn_conv3_128x32", 128, 32, 32),
            ("nn_conv3_64x64", 64, 64, 64),
            ("nn_conv3_32x64", 32, 64, 64),
            ("nn_conv3_32x128", 32, 128, 128),
        ] {
            let x0 = halves(e.pow(3) * c0, 1, 1.0);
            let x1 = halves(e.pow(3) * c1, 2, 1.0);
            let w0 = halves(o * 27 * c0, 3, 0.2);
            let w1 = halves(o * 27 * c1, 4, 0.2);
            let bias: Vec<f32> = (0..o).map(|k| f16::from_f32(wave(k, 5)).to_f32()).collect();
            let bx0 = gpu.upload(&padded(&x0, e, c0)).unwrap();
            let bx1 = gpu.upload(&padded(&x1, e, c1)).unwrap();
            let (bw0, bw1, bb) = (gpu.upload(&w0).unwrap(), gpu.upload(&w1).unwrap(), gpu.upload(&bias).unwrap());
            let y = gpu.alloc_fill(e.pow(3) * o * 2, 0xFF).unwrap();
            let tiles = e * (e * (e + 2)).div_ceil(bm);
            let part = gpu.alloc_fill(tiles * o * 16, 0xFF).unwrap();
            let args = [bx0.arg(), bw0.arg(), bx1.arg(), bw1.arg(), y.arg(), bb.arg(), part.arg()];
            let tail: [Arg; 6] = [e.into(), c0.into(), c1.into(), o.into(), 1usize.into(), 1usize.into()];
            let all: Vec<Arg> = args.into_iter().chain(tail).collect();
            gpu.dispatch(name, [o / bn, tiles, 1], [128, 1, 1], &all).unwrap();
            let mut got = vec![f16::ZERO; e.pow(3) * o];
            gpu.download(&y, 0, &mut got).unwrap();
            let st = stats(&gpu, &part, tiles, o);
            let want = |v: usize, k: usize| {
                let (z, yy, xx) = (v / (e * e), (v / e) % e, v % e);
                let a = conv_ref(&x0, &w0, e, c0, 1, [z, yy, xx], k) + conv_ref(&x1, &w1, e, c1, 1, [z, yy, xx], k);
                (a, bias[k] as f64)
            };
            check(&got, &st, e, o, want, name);
        }

        let (eo, c) = (5usize, 32usize);
        let ei = 2 * eo;
        for (name, bm, bn, o) in [
            ("nn_conv3s2_32x64", 32, 64, 64),
            ("nn_conv3s2_64x64", 64, 64, 64),
            ("nn_conv3s2_16x64", 16, 64, 64),
            ("nn_conv3s2_32x128", 32, 128, 128),
            ("nn_conv3s2_16x128", 16, 128, 128),
        ] {
            let x = halves(ei.pow(3) * c, 6, 1.0);
            let w = halves(o * 27 * c, 7, 0.2);
            let bias: Vec<f32> = (0..o).map(|k| f16::from_f32(wave(k, 8)).to_f32()).collect();
            let (bx, bw, bb) =
                (gpu.upload(&padded(&x, ei, c)).unwrap(), gpu.upload(&w).unwrap(), gpu.upload(&bias).unwrap());
            let y = gpu.alloc_fill(eo.pow(3) * o * 2, 0xFF).unwrap();
            let tiles = eo * eo * eo.div_ceil(bm);
            let part = gpu.alloc_fill(tiles * o * 16, 0xFF).unwrap();
            let args =
                [bx.arg(), bw.arg(), y.arg(), bb.arg(), part.arg(), eo.into(), c.into(), o.into(), 1usize.into()];
            gpu.dispatch(name, [o / bn, tiles, 1], [128, 1, 1], &args).unwrap();
            let mut got = vec![f16::ZERO; eo.pow(3) * o];
            gpu.download(&y, 0, &mut got).unwrap();
            let st = stats(&gpu, &part, tiles, o);
            let want = |v: usize, k: usize| {
                let (z, yy, xx) = (v / (eo * eo), (v / eo) % eo, v % eo);
                (conv_ref(&x, &w, ei, c, 2, [z, yy, xx], k), bias[k] as f64)
            };
            check(&got, &st, eo, o, want, name);
        }
    }

    /// GPU seconds per dispatch of `name` over `reps` back-to-back runs, after one warm-up
    fn time(gpu: &Gpu, name: &'static str, groups: [usize; 3], args: &[Arg], reps: usize) -> f64 {
        gpu.dispatch(name, groups, [128, 1, 1], args).unwrap();
        gpu.synchronize().unwrap();
        for _ in 0..reps {
            gpu.dispatch(name, groups, [128, 1, 1], args).unwrap();
        }
        let fence = gpu.flush().unwrap().expect("a fence");
        fence.wait().unwrap();
        let (a, b) = fence.gpu_span();
        (b - a) / reps as f64
    }

    /// The structure gate: TFLOPS of every tile shape on each of m7's conv shapes (`INKRS_NN_BENCH=1`, through the
    /// GPU queue)
    /// The negative control for `scripts/metal_validate.sh`: a store past the end of a buffer, which shader validation
    /// must report, so a validation run of this module is known to be able to fail
    #[test]
    #[ignore = "writes out of bounds on purpose; run only under metal_validate.sh as its negative control"]
    fn validation_negative_control() {
        let gpu = Gpu::open(0).unwrap();
        let acc = gpu.alloc(64 * 4).unwrap();
        // 1024 elements of one slot into a 64-float buffer
        let args = [Arg::Buf(&acc, 0), Arg::U32(1), Arg::U32(1), Arg::U32(1024), Arg::U32(0), Arg::U32(1)];
        gpu.dispatch_1d("nn_zero_slices", 1024, &args).unwrap();
        gpu.synchronize().unwrap();
    }

    #[test]
    #[ignore = "a GPU benchmark; run through the gpu0 queue"]
    fn conv_bench() {
        let gpu = Gpu::open(0).unwrap();
        gpu.ready().unwrap();
        let reps = 5;
        // (edge, c0, c1, cout): encoder convs, then the decoder's two-source convs
        let s1: [(usize, usize, usize, usize); 10] = [
            (192, 32, 0, 32),
            (96, 64, 0, 64),
            (48, 128, 0, 128),
            (24, 256, 0, 256),
            (12, 320, 0, 320),
            (6, 320, 0, 320),
            (192, 32, 32, 32),
            (96, 64, 64, 64),
            (48, 128, 128, 128),
            (24, 256, 256, 256),
        ];
        let shapes: [(&'static str, usize, usize); 5] = [
            ("nn_conv3_64x32", 64, 32),
            ("nn_conv3_128x32", 128, 32),
            ("nn_conv3_64x64", 64, 64),
            ("nn_conv3_32x64", 32, 64),
            ("nn_conv3_32x128", 32, 128),
        ];
        for (e, c0, c1, o) in s1 {
            let ep = e + 2;
            let bx0 = gpu.upload(&halves(ep.pow(3) * c0, 1, 1.0)).unwrap();
            let bx1 = gpu.upload(&halves(ep.pow(3) * c1.max(1), 2, 1.0)).unwrap();
            let bw0 = gpu.upload(&halves(o * 27 * c0, 3, 0.05)).unwrap();
            let bw1 = gpu.upload(&halves(o * 27 * c1.max(1), 4, 0.05)).unwrap();
            let bb = gpu.upload(&vec![0f32; o]).unwrap();
            let y = gpu.alloc(e.pow(3) * o * 2).unwrap();
            let flop = 2.0 * e.pow(3) as f64 * o as f64 * 27.0 * (c0 + c1) as f64;
            for (name, bm, bn) in shapes {
                if o % bn != 0 {
                    continue;
                }
                let tiles = e * (e * ep).div_ceil(bm);
                let part = gpu.alloc(tiles * o * 16).unwrap();
                let args = [bx0.arg(), bw0.arg(), bx1.arg(), bw1.arg(), y.arg(), bb.arg(), part.arg()];
                let tail: [Arg; 6] = [e.into(), c0.into(), c1.into(), o.into(), 1usize.into(), 1usize.into()];
                let all: Vec<Arg> = args.into_iter().chain(tail).collect();
                let t = time(&gpu, name, [o / bn, tiles, 1], &all, reps);
                eprintln!("BENCH s1 e={e} c={c0}+{c1} o={o} {name}: {:.3} ms {:.2} TFLOPS", t * 1e3, flop / t / 1e12);
            }
        }
        let s2: [(usize, usize, usize); 5] =
            [(96, 32, 64), (48, 64, 128), (24, 128, 256), (12, 256, 320), (6, 320, 320)];
        let shapes2: [(&'static str, usize, usize); 5] = [
            ("nn_conv3s2_32x64", 32, 64),
            ("nn_conv3s2_64x64", 64, 64),
            ("nn_conv3s2_16x64", 16, 64),
            ("nn_conv3s2_32x128", 32, 128),
            ("nn_conv3s2_16x128", 16, 128),
        ];
        for (eo, c, o) in s2 {
            let ep = 2 * eo + 2;
            let bx = gpu.upload(&halves(ep.pow(3) * c, 1, 1.0)).unwrap();
            let bw = gpu.upload(&halves(o * 27 * c, 3, 0.05)).unwrap();
            let bb = gpu.upload(&vec![0f32; o]).unwrap();
            let y = gpu.alloc(eo.pow(3) * o * 2).unwrap();
            let flop = 2.0 * eo.pow(3) as f64 * o as f64 * 27.0 * c as f64;
            for (name, bm, bn) in shapes2 {
                if o % bn != 0 {
                    continue;
                }
                let tiles = eo * eo * eo.div_ceil(bm);
                let part = gpu.alloc(tiles * o * 16).unwrap();
                let args =
                    [bx.arg(), bw.arg(), y.arg(), bb.arg(), part.arg(), eo.into(), c.into(), o.into(), 1usize.into()];
                let t = time(&gpu, name, [o / bn, tiles, 1], &args, reps);
                eprintln!("BENCH s2 e={eo} c={c} o={o} {name}: {:.3} ms {:.2} TFLOPS", t * 1e3, flop / t / 1e12);
            }
        }
    }
}
