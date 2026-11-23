//! villa's sliding-window patch plan for `vesuvius.predict`
//!
//! The grid is always the whole volume's (`compute_steps_for_sliding_window` per axis, nnU-Net's rule); a
//! bounding box keeps the patches of that grid that touch it rather than laying a grid over the box, so
//! a region's output matches the same region of a whole-volume run (`VCDataset`, vc_dataset.py:189-212)

use color_eyre::eyre::{Result, bail, ensure};

/// A half-open box `[lo, hi)` per axis, z, y, x
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bbox {
    pub lo: [usize; 3],
    pub hi: [usize; 3],
}

impl Bbox {
    /// Parses villa's `--bbox` spelling `z0:z1,y0:y1,x0:x1`; an empty bound is the volume's edge
    pub fn parse(s: &str, shape: [usize; 3]) -> Result<Self> {
        let axes: Vec<&str> = s.split(',').collect();
        ensure!(axes.len() == 3, "bbox {s:?}: need z0:z1,y0:y1,x0:x1");
        let mut b = Self { lo: [0; 3], hi: shape };
        for (a, part) in axes.iter().enumerate() {
            let Some((lo, hi)) = part.split_once(':') else { bail!("bbox axis {a} {part:?}: need lo:hi") };
            if !lo.trim().is_empty() {
                b.lo[a] = lo.trim().parse()?;
            }
            if !hi.trim().is_empty() {
                b.hi[a] = hi.trim().parse()?;
            }
        }
        Ok(b)
    }

    /// villa's `_clamp_bbox_to_image`: clipped to the volume and grown to at least a patch, shifted back
    /// inside the volume when growing would overhang
    pub fn clamp(self, shape: [usize; 3], patch: [usize; 3]) -> Result<Self> {
        let mut out = self;
        for a in 0..3 {
            let (lo, hi) = (self.lo[a].min(shape[a]), self.hi[a].min(shape[a]));
            ensure!(
                hi > lo,
                "bbox axis {a} [{}, {}) lies outside the volume (size {})",
                self.lo[a],
                self.hi[a],
                shape[a]
            );
            let (mut lo, mut hi) = (lo, hi);
            if hi - lo < patch[a] {
                hi = (lo + patch[a]).min(shape[a]);
                lo = hi.saturating_sub(patch[a]);
            }
            out.lo[a] = lo;
            out.hi[a] = hi;
        }
        Ok(out)
    }
}

/// `compute_steps_for_sliding_window(extent, patch, step_factor)`: `n` evenly spread starts from 0 to
/// `extent - patch`, `n` the fewest whose spacing is at most `int(patch * step_factor)`, each start
/// `round(spacing * i)` (half to even, as `np.round`)
pub fn steps(extent: usize, patch: usize, step_factor: f64) -> Result<Vec<usize>> {
    ensure!(extent >= patch, "image size {extent} must be at least the patch {patch}");
    ensure!((0.0..=1.0).contains(&step_factor), "step factor {step_factor}");
    let target = if step_factor == 0.0 { patch } else { (patch as f64 * step_factor) as usize }.max(1);
    let max = extent - patch;
    let n = (max as f64 / target as f64).ceil() as usize + 1;
    let actual = if n > 1 { max as f64 / (n - 1) as f64 } else { 99_999_999_999.0 };
    Ok((0..n).map(|i| ((actual * i as f64).round_ties_even() as usize).min(max)).collect())
}

/// The patches of one run: starts per axis, run in z, then y, then x order
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub shape: [usize; 3],
    pub patch: [usize; 3],
    pub starts: [Vec<usize>; 3],
}

impl Plan {
    /// villa's plan for a volume of `shape`, optionally restricted to the patches touching `bbox`;
    /// `overlap` is `--overlap` (the step factor is `1 - overlap`)
    pub fn new(shape: [usize; 3], patch: [usize; 3], overlap: f64, bbox: Option<Bbox>) -> Result<Self> {
        ensure!((0.0..1.0).contains(&overlap), "overlap {overlap} must be in [0, 1)");
        let step = 1.0 - overlap;
        let roi = bbox.map(|b| b.clamp(shape, patch)).transpose()?;
        let mut starts: [Vec<usize>; 3] = Default::default();
        for a in 0..3 {
            let all = steps(shape[a], patch[a], step)?;
            starts[a] = match roi {
                Some(r) => all.into_iter().filter(|&s| s < r.hi[a] && s + patch[a] > r.lo[a]).collect(),
                None => all,
            };
            ensure!(!starts[a].is_empty(), "no patch on axis {a}");
        }
        Ok(Self { shape, patch, starts })
    }

    /// Patch count
    pub fn len(&self) -> usize {
        self.starts.iter().map(Vec::len).product()
    }

    /// Whether the plan has no patches (never, once built)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The union of the patches, clipped to the volume: villa's in-memory canvas
    pub fn canvas(&self) -> Bbox {
        let lo = std::array::from_fn(|a| self.starts[a][0]);
        let hi =
            std::array::from_fn(|a| (self.starts[a].last().copied().unwrap_or(0) + self.patch[a]).min(self.shape[a]));
        Bbox { lo, hi }
    }

    /// Every patch start in run order
    pub fn patches(&self) -> impl Iterator<Item = [usize; 3]> + '_ {
        let [zs, ys, xs] = &self.starts;
        zs.iter().flat_map(move |&z| ys.iter().flat_map(move |&y| xs.iter().map(move |&x| [z, y, x])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Starts as villa's `compute_steps_for_sliding_window` gives them (golden values from villa @ 7e6b82164
    /// with numpy), including the single-patch case
    #[test]
    fn villa_steps() {
        assert_eq!(steps(192, 192, 0.5).unwrap(), vec![0]);
        assert_eq!(steps(500, 192, 0.5).unwrap(), vec![0, 77, 154, 231, 308]);
        assert_eq!(steps(16920, 192, 0.5).unwrap().len(), 176);
        assert_eq!(steps(8169, 192, 0.5).unwrap()[..4], [0, 95, 190, 285]);
        assert_eq!(steps(8169, 192, 0.5).unwrap().last(), Some(&7977));
        assert_eq!(steps(1000, 128, 0.75).unwrap(), vec![0, 87, 174, 262, 349, 436, 523, 610, 698, 785, 872]);
        assert_eq!(steps(197, 192, 0.5).unwrap(), vec![0, 5]);
        assert_eq!(steps(389, 192, 0.5).unwrap(), vec![0, 66, 131, 197]);
    }

    /// The 0826 zone's plan: the global grid's patches touching `12032:12544,5248:5760,4736:5248`
    #[test]
    fn zone_plan() {
        let shape = [16920, 8169, 8169];
        let bbox = Bbox::parse("12032:12544,5248:5760,4736:5248", shape).unwrap();
        let plan = Plan::new(shape, [192; 3], 0.5, Some(bbox)).unwrap();
        assert_eq!(plan.starts, ZONE_STARTS.map(|v| v.to_vec()));
        assert_eq!(plan.len(), ZONE_STARTS.iter().map(|v| v.len()).product::<usize>());
        let c = plan.canvas();
        assert!((0..3).all(|a| c.lo[a] <= bbox.lo[a] && c.hi[a] >= bbox.hi[a]));
        assert_eq!(plan.patches().next(), Some([ZONE_STARTS[0][0], ZONE_STARTS[1][0], ZONE_STARTS[2][0]]));
    }

    /// A box thinner than a patch grows to a patch, shifted inside the volume at the far edge
    #[test]
    fn bbox_clamp() {
        let b = Bbox { lo: [10, 0, 990], hi: [20, 1000, 1000] };
        let c = b.clamp([1000; 3], [192; 3]).unwrap();
        assert_eq!(c, Bbox { lo: [10, 0, 808], hi: [202, 1000, 1000] });
        assert!(Bbox { lo: [2000, 0, 0], hi: [3000, 5, 5] }.clamp([1000; 3], [192; 3]).is_err());
    }

    const ZONE_STARTS: [&[usize]; 3] = [
        &[11853, 11949, 12044, 12140, 12235, 12331, 12427, 12522],
        &[5128, 5223, 5318, 5413, 5508, 5603, 5698],
        &[4558, 4653, 4748, 4843, 4938, 5033, 5128, 5223],
    ];
}
