//! inkrs's own 2D convolutions for the villa readers (the `conv_*` kernels of
//! `inkrs-kernels/src/conv2d.rs`): which kernel runs each shape, the weights in the kernels' B-fragment
//! order, and the launch of a conv with its InstanceNorm statistics
//!
//! `--conv inkrs` runs every 2D conv of the reader here, at any tile size whose maps are multiples
//! of 4 (both reader tiles are), and leaves any other shape on cuDNN; `--conv cudnn` (or
//! `INKRS_CONV=cudnn`) runs them all through cuDNN plans. The own convs sum in a fixed order, so a
//! map is the same bytes in every process and on both PTX tiers, which cuDNN's timed algorithm
//! search does not promise
//!
//! The default depends on the GPU: the own convs on compute capability 12.x, cuDNN everywhere
//! else. [`Cfg::choose`] is a table baked from RTX 5090 probes, with no timing or cuDNN fallback at
//! run time, and its speed and parity on sm80/86/89 hardware were never measured. So a default run
//! on a 12.x GPU can differ from the same run on another GPU by cuDNN's rounding, within the parity
//! contract. An explicit `--conv` or `INKRS_CONV` wins on any tier that has the own kernels

use std::str::FromStr;

use color_eyre::eyre::{Result, bail, ensure, eyre};
use cudakit::{ComputeCapability, PtxTier};

use crate::dnn::ConvShape;
use crate::ensure_supported;
use crate::gpu::{Arg, Buf, Gpu};

/// Which library runs the reader's 2D convs; no default, since it depends on the device
/// ([`ConvBackend::resolve`])
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvBackend {
    /// inkrs's own kernels, with the InstanceNorm statistics in their epilogue
    Inkrs,
    /// cuDNN plans (`--algo` picks their algorithms)
    Cudnn,
}

impl ConvBackend {
    /// Overrides the default when the CLI does not name a backend
    pub const ENV: &'static str = "INKRS_CONV";

    /// The backend `INKRS_CONV` names; None when it is unset, so [`Self::resolve`] picks the GPU's
    /// default
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var(Self::ENV) {
            Ok(v) => v.parse().map(Some),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => bail!("{}: {e}", Self::ENV),
        }
    }

    /// The backend that runs on a device of compute capability `cc` with PTX tier `tier` loaded, and
    /// why: `requested` (`--conv` or `INKRS_CONV`) when that tier has it, else [`Self::default_on`]
    /// for the device. The sm75 tier has no own conv kernels (`inkrs-kernels/src/lib.rs` leaves them
    /// out), so it always runs cuDNN, and refuses an explicit `inkrs` as unsupported rather than
    /// switching
    pub fn resolve(requested: Option<Self>, cc: ComputeCapability, tier: PtxTier) -> Result<(Self, ConvReason)> {
        let own = Self::tier_has_own(tier);
        ensure_supported!(
            own || requested != Some(Self::Inkrs),
            "--conv inkrs needs the sm80 or sm120 PTX tier (compute capability 8.0 or above); this GPU runs the \
             {tier} tier, so its 2D convs run on cuDNN (--conv cudnn)"
        );
        Ok(match requested {
            _ if !own => (Self::Cudnn, ConvReason::TierWithout(tier)),
            Some(b) => (b, ConvReason::Requested),
            None => (Self::default_on(cc), ConvReason::Default(cc)),
        })
    }

    /// The default where nothing is requested, on a tier that has the own kernels. Keyed on the
    /// device rather than the tier, so a forced sm80 tier on a 12.x GPU keeps the own convs: their
    /// kernel table was tuned and gated on the RTX 5090 (CC 12.0) only
    fn default_on(cc: ComputeCapability) -> Self {
        if cc.major == 12 { Self::Inkrs } else { Self::Cudnn }
    }

    /// Whether `tier`'s embedded PTX carries the own `conv_*` kernels
    pub fn tier_has_own(tier: PtxTier) -> bool {
        tier != PtxTier::Sm75
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Inkrs => "inkrs",
            Self::Cudnn => "cudnn",
        }
    }
}

/// Why [`ConvBackend::resolve`] picked a backend, for the run report and the log
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvReason {
    /// `--conv` or `INKRS_CONV` named it
    Requested,
    /// Nothing was requested: the default for a device of this compute capability
    Default(ComputeCapability),
    /// The loaded PTX tier has no own conv kernels
    TierWithout(PtxTier),
}

impl std::fmt::Display for ConvReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Requested => write!(f, "requested by --conv or {}", ConvBackend::ENV),
            Self::Default(cc) => write!(f, "default for compute capability {cc}"),
            Self::TierWithout(tier) => write!(f, "the {tier} PTX tier has no own convs"),
        }
    }
}

impl FromStr for ConvBackend {
    type Err = color_eyre::Report;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "inkrs" => Ok(Self::Inkrs),
            "cudnn" => Ok(Self::Cudnn),
            _ => Err(eyre!("conv backend {s:?}: expected inkrs or cudnn")),
        }
    }
}

/// One `conv_*` kernel's fixed geometry, as its `conv_kernel!` line in `conv2d.rs` declares it
#[derive(Debug)]
pub struct Cfg {
    pub name: &'static str,
    stride: i32,
    ksize: i32,
    /// Output patch rows and columns per sample, and samples per block
    th: i32,
    tw: i32,
    samp: i32,
    /// Input channels per stage and output channels per block
    ck: i32,
    bn: i32,
    threads: u32,
}

// every kernel the probe checks and times; `Cfg::choose` picks from these, and the rest stay as
// candidates so a new tile size or batch can be re-tuned with the probe alone
const CFGS: &[Cfg] = &[
    Cfg { name: "conv_s1_p16_n32", stride: 1, ksize: 3, th: 16, tw: 16, samp: 1, ck: 32, bn: 32, threads: 256 },
    Cfg { name: "conv_s1_p16_n64", stride: 1, ksize: 3, th: 16, tw: 16, samp: 1, ck: 32, bn: 64, threads: 256 },
    Cfg { name: "conv_s1_p8_n64", stride: 1, ksize: 3, th: 8, tw: 8, samp: 1, ck: 32, bn: 64, threads: 128 },
    Cfg { name: "conv_s1_p4_n32", stride: 1, ksize: 3, th: 4, tw: 4, samp: 4, ck: 32, bn: 32, threads: 128 },
    Cfg { name: "conv_s1_p4x1_n32", stride: 1, ksize: 3, th: 4, tw: 4, samp: 1, ck: 32, bn: 32, threads: 32 },
    Cfg { name: "conv_s2_p8x16_n64", stride: 2, ksize: 3, th: 8, tw: 16, samp: 1, ck: 16, bn: 64, threads: 256 },
    Cfg { name: "conv_s2_p8_n64", stride: 2, ksize: 3, th: 8, tw: 8, samp: 1, ck: 32, bn: 64, threads: 128 },
    Cfg { name: "conv_s2_p4_n32", stride: 2, ksize: 3, th: 4, tw: 4, samp: 4, ck: 32, bn: 32, threads: 128 },
    Cfg { name: "conv_s2_p4x1_n32", stride: 2, ksize: 3, th: 4, tw: 4, samp: 1, ck: 32, bn: 32, threads: 32 },
    Cfg { name: "conv_k1_p16_n64", stride: 1, ksize: 1, th: 16, tw: 16, samp: 1, ck: 32, bn: 64, threads: 256 },
    Cfg { name: "conv_k1_p8_n64", stride: 1, ksize: 1, th: 8, tw: 8, samp: 1, ck: 32, bn: 64, threads: 128 },
    Cfg { name: "conv_s1_p16_n32_m2", stride: 1, ksize: 3, th: 16, tw: 16, samp: 1, ck: 32, bn: 32, threads: 256 },
    Cfg { name: "conv_s1_p8_n64_m4", stride: 1, ksize: 3, th: 8, tw: 8, samp: 1, ck: 32, bn: 64, threads: 128 },
    Cfg { name: "conv_s1_p8_n32", stride: 1, ksize: 3, th: 8, tw: 8, samp: 1, ck: 32, bn: 32, threads: 128 },
    Cfg { name: "conv_s1_p4_n16", stride: 1, ksize: 3, th: 4, tw: 4, samp: 4, ck: 32, bn: 16, threads: 128 },
];

impl Cfg {
    /// Samples, output rows and output columns of one patch
    pub fn patch(&self) -> [i32; 3] {
        [self.samp, self.th, self.tw]
    }

    /// Output channels a block computes, so `cout / bn()` blocks along y
    pub fn bn(&self) -> i32 {
        self.bn
    }

    /// Every kernel that runs `shape` exactly
    pub fn candidates(s: &ConvShape) -> impl Iterator<Item = &'static Cfg> + '_ {
        CFGS.iter().filter(|c| c.fits(s))
    }

    /// Whether this kernel runs `shape` exactly: its geometry, and channel and size multiples
    fn fits(&self, s: &ConvShape) -> bool {
        let out = s.out_spatial();
        s.spatial.len() == 2
            && s.kernel == [self.ksize; 2]
            && s.stride == [self.stride; 2]
            && s.pad == [(self.ksize - 1) / 2; 2]
            && out[0] % self.th == 0
            && out[1] % self.tw == 0
            && s.n % self.samp == 0
            && s.c % self.ck == 0
            && s.k % self.bn == 0
    }

    /// The kernel for `shape`: the first that fits from a preference list keyed on the output size,
    /// fastest on the RTX 5090 probe at 128 x 128 (batch 64) and 384 x 384 (batch 8) tiles; none where
    /// the output is not a multiple of 4
    pub fn choose(s: &ConvShape) -> Option<&'static Cfg> {
        let o = *s.out_spatial().first()?;
        // large maps want 16 x 16 patches at two blocks per SM; mid-size maps want 8 x 8 patches with
        // narrower channel blocks so more blocks fill the 170 SMs; the 8 x 8 map itself is one patch
        // per sample, where the wide n64 block halves the input reloads
        let prefs: &[&str] = match (s.kernel.first()?, s.stride.first()?) {
            (3, 1) if o % 16 == 0 && o >= 64 => &["conv_s1_p16_n32_m2", "conv_s1_p8_n32"],
            (3, 1) if o % 8 == 0 && o > 8 => &["conv_s1_p8_n32", "conv_s1_p8_n64"],
            (3, 1) if o % 8 == 0 => &["conv_s1_p8_n64", "conv_s1_p8_n32"],
            (3, 1) if o % 4 == 0 => &["conv_s1_p4_n16", "conv_s1_p4x1_n32"],
            (3, 2) if o % 8 == 0 => &["conv_s2_p8_n64", "conv_s2_p4_n32", "conv_s2_p4x1_n32"],
            (3, 2) if o % 4 == 0 => &["conv_s2_p4_n32", "conv_s2_p4x1_n32"],
            (1, 1) if o % 16 == 0 && o >= 48 => &["conv_k1_p16_n64", "conv_k1_p8_n64"],
            (1, 1) if o % 8 == 0 => &["conv_k1_p8_n64"],
            _ => return None,
        };
        prefs.iter().filter_map(|name| CFGS.iter().find(|c| c.name == *name)).find(|c| c.fits(s))
    }
}

/// A PyTorch conv weight `[cout, cin, taps]` in float32 as the `conv_*` kernels' B fragments, `u32`
/// FP16 pairs: for each (16 input channels `q`, tap `t`, 16 output channels `j`, lane `l` with `g = l /
/// 4`, `th = l % 4`), four pairs `[n-tile 0: k-half 0, 1; n-tile 1: k-half 0, 1]`, each the weights
/// from input channels `16 q + 8 half + 2 th + {0, 1}` (low half first) to output channel `16 j + 8
/// ntile + g`, at chunk `((q * taps + t) * (cout / 16) + j) * 32 + l`: the `mma.m16n8k16` B layout
pub fn conv_frags(w: &[f32], cout: usize, cin: usize, taps: usize) -> Vec<u32> {
    assert_eq!(w.len(), cout * cin * taps, "conv_frags: weight length");
    assert!(cout.is_multiple_of(16) && cin.is_multiple_of(16), "conv_frags: channels must be multiples of 16");
    let at = |k: usize, c: usize, t: usize| half::f16::from_f32(w[(k * cin + c) * taps + t]).to_bits() as u32;
    let mut out = Vec::with_capacity(cout * cin * taps / 2);
    for q in 0..cin / 16 {
        for t in 0..taps {
            for j in 0..cout / 16 {
                for lane in 0..32 {
                    let (g, th) = (lane / 4, lane % 4);
                    for nt in 0..2 {
                        for half in 0..2 {
                            let (k, c) = (16 * j + 8 * nt + g, 16 * q + 8 * half + 2 * th);
                            out.push(at(k, c, t) | (at(k, c + 1, t) << 16));
                        }
                    }
                }
            }
        }
    }
    out
}

/// A 2D conv on inkrs's kernels: its kernel, shape and B fragments
pub struct OwnConv {
    pub cfg: &'static Cfg,
    pub shape: ConvShape,
    frags: Buf,
}

impl OwnConv {
    /// `w` is the PyTorch weight `[k, c, kh, kw]` in float32; fails if no kernel runs `shape`
    pub fn new(gpu: &Gpu, shape: ConvShape, w: &[f32]) -> Result<Self> {
        let Some(cfg) = Cfg::choose(&shape) else { bail!("no inkrs conv kernel runs {shape}") };
        Self::with_cfg(gpu, shape, w, cfg)
    }

    /// As [`OwnConv::new`] on kernel `cfg`, which must run `shape`
    pub fn with_cfg(gpu: &Gpu, shape: ConvShape, w: &[f32], cfg: &'static Cfg) -> Result<Self> {
        ensure!(cfg.fits(&shape), "{} does not run {shape}", cfg.name);
        let taps = (shape.kernel[0] * shape.kernel[1]) as usize;
        let frags = gpu.upload(&conv_frags(w, shape.k as usize, shape.c as usize, taps))?;
        Ok(Self { cfg, shape, frags })
    }

    /// Patches per sample: one InstanceNorm partial each
    pub fn segs(&self) -> usize {
        let out = self.shape.out_spatial();
        ((out[0] / self.cfg.th) * (out[1] / self.cfg.tw)) as usize
    }

    /// Floats-pairs of partial scratch [`OwnConv::run`] needs
    pub fn partial_len(&self) -> usize {
        self.shape.n as usize * self.segs() * self.shape.k as usize
    }

    /// Queues the conv of `x` into `y` (both FP16 channels-last) and its per-(sample, channel)
    /// InstanceNorm `[mean, invstd]` into `stats`, through `partial`
    pub fn run(&self, gpu: &Gpu, x: &Buf, y: &Buf, partial: &Buf, stats: &Buf, eps: f32) -> Result<()> {
        self.run_grid(gpu, x, y, partial, stats, eps, u32::MAX)
    }

    /// As [`OwnConv::run`] with at most `blocks` blocks along x, each striding over the patches
    #[allow(clippy::too_many_arguments)]
    pub fn run_grid(
        &self,
        gpu: &Gpu,
        x: &Buf,
        y: &Buf,
        partial: &Buf,
        stats: &Buf,
        eps: f32,
        blocks: u32,
    ) -> Result<()> {
        let s = &self.shape;
        let (n, k) = (s.n as usize, s.k as usize);
        let out = s.out_spatial();
        let (xe, _, ye) = s.elements();
        ensure!(x.bytes() >= xe * 2 && y.bytes() >= ye * 2, "{s}: buffers too small");
        ensure!(partial.bytes() >= self.partial_len() * 8 && stats.bytes() >= n * k * 8, "{s}: IN scratch too small");
        let c = self.cfg;
        let patches = (s.n / c.samp) * (out[0] / c.th) * (out[1] / c.tw);
        gpu.launch(
            c.name,
            ((patches as u32).min(blocks.max(1)), (s.k / c.bn) as u32, 1),
            (c.threads, 1, 1),
            0,
            &[
                y.view::<u32>(0, ye / 2),
                partial.view::<[f32; 2]>(0, self.partial_len()),
                x.view::<[u32; 4]>(0, xe / 8),
                self.frags.arg::<[u32; 4]>(),
                Arg::U32(s.n as u32),
                Arg::U32(s.spatial[0] as u32),
                Arg::U32(s.spatial[1] as u32),
                Arg::U32(s.c as u32),
                Arg::U32(s.k as u32),
            ],
        )?;
        gpu.launch(
            "conv_in_merge",
            ((n * k).div_ceil(128) as u32, 1, 1),
            (128, 1, 1),
            0,
            &[
                stats.view::<[f32; 2]>(0, n * k),
                partial.view::<[f32; 2]>(0, self.partial_len()),
                Arg::U32(self.segs() as u32),
                Arg::U32(k as u32),
                Arg::U32((c.th * c.tw) as u32),
                Arg::F32(eps),
            ],
        )
    }
}

/// Every 2D conv of the villa hybrid_3d2d reader at batch `n` and `tile` x `tile` slices (128 for
/// reader-v2, 384 for reader-v3), deduplicated, with how many layers run each
pub fn reader_convs(n: i32, tile: i32) -> Vec<(ConvShape, usize)> {
    let mut v: Vec<(ConvShape, usize)> = Vec::new();
    let mut add = |s: ConvShape| match v.iter_mut().find(|(t, _)| *t == s) {
        Some((_, count)) => *count += 1,
        None => v.push((s, 1)),
    };
    add(ConvShape::same(n, 32, 32, &[tile, tile], 3, 1));
    let (mut cin, mut h) = (32, tile);
    for (s, (&c, &blocks)) in [32, 64, 128, 256, 320, 320].iter().zip(&[1, 3, 4, 6, 6, 6]).enumerate() {
        for j in 0..blocks {
            let strided = j == 0 && s > 0;
            let hout = if strided { h / 2 } else { h };
            add(ConvShape::same(n, cin, c, &[h, h], 3, if strided { 2 } else { 1 }));
            add(ConvShape::same(n, c, c, &[hout, hout], 3, 1));
            if cin != c {
                add(ConvShape::same(n, cin, c, &[hout, hout], 1, 1));
            }
            (cin, h) = (c, hout);
        }
    }

    for (c, div) in [(320, 16), (256, 8), (128, 4), (64, 2), (32, 1)] {
        add(ConvShape::same(n, 2 * c, c, &[tile / div, tile / div], 3, 1));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel the table names is an entry of every embedded PTX tier, so a stale PTX build fails
    /// here rather than at the first launch of that shape
    #[test]
    fn every_kernel_is_in_every_tier() {
        for (tier, ptx) in crate::gpu::PTX {
            for c in CFGS.iter().map(|c| c.name).chain(["conv_in_merge"]) {
                let entry = format!(".visible .entry {c}(");
                // the sm75 tier leaves them all out, so resolve must never pick them there
                if ConvBackend::tier_has_own(*tier) {
                    assert!(ptx.contains(&entry), "{tier}: no kernel {c}");
                } else {
                    assert!(!ptx.contains(&entry), "{tier} has kernel {c}, which it does not run");
                }
            }
        }
    }

    /// sm75 runs every 2D conv on cuDNN whatever is asked, and refuses an explicit `inkrs`; on the
    /// tiers with the kernels a request wins, and the default is the own convs on CC 12.x only,
    /// whichever tier is loaded
    #[test]
    fn the_backend_follows_the_device() {
        use ConvBackend::{Cudnn, Inkrs};
        let cc = ComputeCapability::new;
        let resolve = ConvBackend::resolve;
        let (turing, ampere, ada, blackwell) = (cc(7, 5), cc(8, 6), cc(8, 9), cc(12, 0));
        assert_eq!(resolve(None, turing, PtxTier::Sm75).unwrap(), (Cudnn, ConvReason::TierWithout(PtxTier::Sm75)));
        assert_eq!(resolve(Some(Cudnn), turing, PtxTier::Sm75).unwrap().0, Cudnn);
        let err = resolve(Some(Inkrs), turing, PtxTier::Sm75).unwrap_err();
        assert!(crate::error::is_unsupported(&err), "{err}");
        // forced sm75 on a newer GPU is the same tier, so the same choice
        assert_eq!(resolve(None, blackwell, PtxTier::Sm75).unwrap().0, Cudnn);
        // the default follows the device: cuDNN on GPUs the own kernels were never timed on
        for (dev, tier, want) in [
            (ampere, PtxTier::Sm80, Cudnn),
            (ada, PtxTier::Sm80, Cudnn),
            (cc(8, 0), PtxTier::Sm80, Cudnn),
            (blackwell, PtxTier::Sm80, Inkrs),
            (blackwell, PtxTier::Sm120, Inkrs),
            (cc(12, 1), PtxTier::Sm120, Inkrs),
        ] {
            assert_eq!(resolve(None, dev, tier).unwrap(), (want, ConvReason::Default(dev)), "{dev} {tier}");
            for b in [Inkrs, Cudnn] {
                assert_eq!(resolve(Some(b), dev, tier).unwrap(), (b, ConvReason::Requested), "{dev} {tier} {b:?}");
            }
        }
    }

    /// Every 2D conv of both reader tiles at any batch has a kernel, so `--conv inkrs` runs the
    /// whole UNet on its own kernels
    #[test]
    fn every_reader_conv_has_a_kernel() {
        for tile in [128, 384] {
            for n in [1, 3, 8, 13, 64] {
                for (s, _) in reader_convs(n, tile) {
                    assert!(Cfg::choose(&s).is_some(), "no kernel for {s}");
                }
            }
        }
    }
}
