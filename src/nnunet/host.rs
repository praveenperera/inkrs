//! The host side of an `inkrs nnunet predict` run that both backends share: the options, villa's flip order, the
//! constant-patch rule and the threshold's logit cutoff

use color_eyre::eyre::{Result, ensure};

use super::Precision;
use crate::dnn::Pick;

/// The order villa's mirroring TTA stacks the flips in (`tta.py`: none, x, y, z, xy, xz, yz, xyz), as
/// flip bits x = 1, y = 2, z = 4; the mean sums them in this order
pub const FLIPS: [u32; 8] = [0, 1, 2, 4, 3, 5, 6, 7];

/// `nn_check`'s status bits (inkrs-kernels `nnunet.rs`): a mean logit other than 0, a non-finite one
pub const NONZERO: u32 = 1;
pub const NONFINITE: u32 = 2;

/// Run options; `None` takes the recipe's value
#[derive(Clone, Debug)]
pub struct Options {
    /// villa's `--bbox` `z0:z1,y0:y1,x0:x1`
    pub bbox: Option<String>,
    /// villa's `--overlap`
    pub overlap: Option<f64>,
    /// `false` for villa's `--disable_tta`
    pub tta: Option<bool>,
    /// The cutoff villa's `--threshold --threshold_value T` resolves to: write 255 where foreground
    /// probability exceeds `T`, 0 elsewhere
    pub threshold: Option<f64>,
    /// Output chunk edge; villa's default is the patch edge
    pub chunk: Option<usize>,
    pub pick: Pick,
    /// Cap on the GPU accumulators
    pub max_accum_gb: f64,
    pub overwrite: bool,
    /// GPU ordinal: a CUDA device, or 0 for the one Metal device
    pub device: usize,
}

/// Whether every voxel of the box `[lo, lo + n)` of `slab` (`[d, h, w]`) normalises to the same fp16
/// value: villa's dataset marks such a patch empty (`min == max` of its normalised tensor) and its zero
/// logits are never blended
pub fn constant_patch(slab: &[u8], dims: [usize; 3], lo: [usize; 3], n: [usize; 3], table16: &[u16; 256]) -> bool {
    let first = table16[slab[(lo[0] * dims[1] + lo[1]) * dims[2] + lo[2]] as usize];
    let value = Precision::Fp16.value(first);
    for z in lo[0]..lo[0] + n[0] {
        for y in lo[1]..lo[1] + n[1] {
            let row = &slab[(z * dims[1] + y) * dims[2] + lo[2]..][..n[2]];
            // equal bits, or the two fp16 zeros, compare equal as villa's min and max do
            if row.iter().any(|&c| table16[c as usize] != first && Precision::Fp16.value(table16[c as usize]) != value)
            {
                return false;
            }
        }
    }
    true
}

/// The threshold's logit cutoff as villa compares it: `float(np.log(T / (1 - T)))`, a Python float that
/// numpy compares against float32 logit differences as float32
pub fn logit_cutoff(t: f64) -> Result<f32> {
    ensure!(t > 0.0 && t < 1.0, "threshold must be in (0, 1), got {t}");
    Ok((t / (1.0 - t)).ln() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// m7 at T = 0.2: villa compares float32 logit differences against ln(0.25) as float32
    #[test]
    fn cutoff_is_villas() {
        assert_eq!(logit_cutoff(0.2).unwrap(), -1.386_294_4_f32);
        assert_eq!(logit_cutoff(0.5).unwrap(), 0.0);
        assert!(logit_cutoff(1.0).is_err());
    }

    /// A patch is skipped only when every in-volume code normalises to one fp16 value: codes clamped to
    /// the same table entry count as one value, any other code does not
    #[test]
    fn constant_patch_rule() {
        let mut table = [0u16; 256];
        for (i, t) in table.iter_mut().enumerate() {
            *t = Precision::Fp16.bits(i.min(212) as f32 / 10.0);
        }
        let dims = [4, 4, 4];
        let mut slab = vec![0u8; 64];
        assert!(constant_patch(&slab, dims, [0; 3], [4; 3], &table));
        slab[63] = 1;
        assert!(!constant_patch(&slab, dims, [0; 3], [4; 3], &table));
        assert!(constant_patch(&slab, dims, [0; 3], [3, 4, 4], &table));
        slab.fill(220);
        slab[5] = 250;
        assert!(constant_patch(&slab, dims, [0; 3], [4; 3], &table));
    }

    /// The flips in villa's stacking order, each once
    #[test]
    fn flips_are_villas_order() {
        assert_eq!(FLIPS, [0, 1, 2, 4, 3, 5, 6, 7]);
        let mut sorted = FLIPS;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
