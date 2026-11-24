//! A 3D U-Net's weights read from an export and laid out on the CPU, shared by the CUDA forward ([`super::model`])
//! and the Metal one (`crate::metal::nnunet`)

use color_eyre::eyre::Result;

use super::{ConvSpec, Precision, Program};
use crate::ensure_supported;
use crate::weights::Checkpoint;

/// One conv's weights read from the checkpoint and laid out on the CPU: the weight channels-last
/// `[K, taps, C]` float32 (rounded at upload, after any channel padding), the bias rounded to the
/// precision, the norm's float32 affine
pub struct HostConv {
    pub cin: usize,
    pub cout: usize,
    pub(crate) w: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) norm: Option<(Vec<f32>, Vec<f32>, f32)>,
}

impl HostConv {
    pub(crate) fn take(ck: &Checkpoint, spec: &ConvSpec, precision: Precision) -> Result<Self> {
        let (k, c) = (spec.cout, spec.cin);
        let taps: usize = spec.kernel.iter().product();
        let shape: Vec<usize> = [k, c].into_iter().chain(spec.kernel.iter().copied()).collect();
        let w = ck.take(&spec.w, &shape)?;
        let mut ktc = vec![0f32; k * taps * c];
        for ko in 0..k {
            for ci in 0..c {
                for t in 0..taps {
                    ktc[(ko * taps + t) * c + ci] = w[(ko * c + ci) * taps + t];
                }
            }
        }
        let bias = match &spec.b {
            Some(key) => Some(ck.take(key, &[k])?.iter().map(|&v| precision.round(v)).collect()),
            None => None,
        };
        let norm = match &spec.norm {
            Some(n) => Some((ck.take(&n.w, &[k])?, ck.take(&n.b, &[k])?, n.eps)),
            None => None,
        };
        Ok(Self { cin: c, cout: k, w: ktc, bias, norm })
    }

    /// The weight as `[K, taps, c_pad]` storage bits, the channels past `cin` zero (the CUDA forward's upload;
    /// the Metal one splits sources itself)
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn bits(&self, precision: Precision, c_pad: usize) -> Vec<u16> {
        let taps = self.w.len() / (self.cout * self.cin);
        let mut out = vec![0u16; self.cout * taps * c_pad];
        for (row, src) in self.w.chunks_exact(self.cin).enumerate() {
            for (ci, &v) in src.iter().enumerate() {
                out[row * c_pad + ci] = precision.bits(v);
            }
        }
        out
    }
}

/// One BasicBlockD's weights on the CPU
pub struct HostBlock {
    pub(crate) conv1: HostConv,
    pub(crate) conv2: HostConv,
    pub(crate) proj: Option<HostConv>,
}

/// One decoder stage's weights on the CPU: the transposed conv as a GEMM weight `[cin, taps * cout]`
/// (column `tap * cout + co`), its bias rounded, and the convs
pub struct HostDec {
    pub(crate) g: Vec<f32>,
    pub(crate) tb: Vec<f32>,
    pub(crate) tcin: usize,
    pub(crate) convs: Vec<HostConv>,
}

/// Every network tensor read from the checkpoint and laid out on the CPU, each taken once by name and
/// shape, so [`Checkpoint::finish`] after this proves the program reads the whole export
pub struct HostNet {
    #[cfg(not(target_os = "macos"))]
    pub(crate) precision: Precision,
    pub(crate) stem: Vec<HostConv>,
    pub(crate) blocks: Vec<HostBlock>,
    pub(crate) decs: Vec<HostDec>,
    /// The head's weight `[classes, cin]` and bias, rounded
    pub(crate) head_w: Vec<f32>,
    pub(crate) head_b: Vec<f32>,
    pub(crate) head_cin: usize,
}

impl HostNet {
    /// Takes the program's tensors from `ck`
    pub fn take(ck: &Checkpoint, prog: &Program, precision: Precision) -> Result<Self> {
        let stem = prog.stem.iter().map(|s| HostConv::take(ck, s, precision)).collect::<Result<_>>()?;
        let mut blocks = Vec::new();
        for b in prog.stages.iter().flatten() {
            blocks.push(HostBlock {
                conv1: HostConv::take(ck, &b.conv1, precision)?,
                conv2: HostConv::take(ck, &b.conv2, precision)?,
                proj: b.skip.proj.as_ref().map(|pj| HostConv::take(ck, pj, precision)).transpose()?,
            });
        }
        let mut decs = Vec::new();
        for ds in &prog.decoder {
            let t = &ds.transp;
            let (ci, co) = (t.cin, t.cout);
            let taps: usize = t.kernel.iter().product();
            let shape: Vec<usize> = [ci, co].into_iter().chain(t.kernel.iter().copied()).collect();
            let w = ck.take(&t.w, &shape)?;
            let mut g = vec![0f32; ci * taps * co];
            for i in 0..ci {
                for o in 0..co {
                    for tap in 0..taps {
                        g[i * taps * co + tap * co + o] = w[(i * co + o) * taps + tap];
                    }
                }
            }
            let key = t.b.as_ref().expect("checked by the contract");
            let tb = ck.take(key, &[co])?.iter().map(|&v| precision.round(v)).collect();
            let convs = ds.convs.iter().map(|c| HostConv::take(ck, c, precision)).collect::<Result<_>>()?;
            decs.push(HostDec { g, tb, tcin: ci, convs });
        }
        let h = &prog.head;
        let head_w = ck.take(&h.w, &[h.cout, h.cin, 1, 1, 1])?.iter().map(|&v| precision.round(v)).collect();
        let key = h.b.as_ref().expect("checked by the contract");
        let head_b = ck.take(key, &[h.cout])?.iter().map(|&v| precision.round(v)).collect();
        Ok(Self {
            #[cfg(not(target_os = "macos"))]
            precision,
            stem,
            blocks,
            decs,
            head_w,
            head_b,
            head_cin: h.cin,
        })
    }
}

/// Fails as unsupported unless `n` elements index with u32
pub fn u32_extent(what: &str, n: usize) -> Result<()> {
    // a limit of the kernels' indexing, not a bad input, so callers such as villa can fall back on PyTorch
    ensure_supported!(u32::try_from(n).is_ok(), "{what}: {n} elements overflow the kernels' u32 indexing");
    Ok(())
}
