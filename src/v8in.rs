//! v8-in, Youssef Nader's 8 um ink model (huggingface.co/YoussefMoNader/ink-8um-v8in at d89166b4, code and
//! weights MIT), and its fine-tunes A-hn seeds 0-2 (huggingface.co/Prodigle/pherc0306b-ink-ahn, weights CC
//! BY-NC 4.0): the 2D float32 ink map of a 24-layer window
//!
//! A ResNet3D-50 on 24 x 64 x 64 tiles that the model upsamples 4x to 96 x 256 x 256 after an
//! InstanceNorm, max over depth at its four levels, and a 2D U-Net decoder. [`model`] runs the network on
//! cuDNN with the kernels of `inkrs-kernels/src/v8in.rs`; [`engine`] tiles a volume, admits tiles by a
//! [`Convention`] and stitches them as the convention's reference runner does. The stem folds the upsample into
//! the conv ([`StemTables`])

// the network runs on cuDNN, or on macOS on the Metal backend's own v8-in engine; the export
// contract and the CPU-side conventions are shared
#[cfg(not(target_os = "macos"))]
pub mod engine;
pub mod host;
#[cfg(not(target_os = "macos"))]
pub mod model;

#[cfg(target_os = "macos")]
pub use crate::metal::v8in::{engine, model};
pub use host::is_v8in_export;

use color_eyre::eyre::{Result, ensure};
use serde::Serialize;

/// Tile edge, pixels per tile slice and depth slices per tile
pub const P: usize = 64;
pub const PIX: usize = P * P;
pub const D: usize = 24;
/// The tile grid's stride, every runner's
pub const STRIDE: usize = 21;
/// Stem output, before the max pool: depth and edge
pub const SD: usize = 96;
pub const SH: usize = 128;
/// Edge of the max pool's output
pub const PH: usize = 64;
/// Stem channels
pub const SC: usize = 64;
/// The trilinear upsample's factor on every axis
const UP: usize = 4;
/// Stem conv size and padding
const K7: usize = 7;
const PAD: i64 = 3;
/// Composed stem kernel width per axis, and taps
const J: usize = 4;
pub const TAPS: usize = J * J * J;
/// A stem block's pooled patch edge and source window edge (`inkrs-kernels` `PP` and `WIN`)
const PP: usize = 8;
const WIN: i32 = 12;

/// How tiles are admitted and stitched, after the runner each convention reproduces
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Convention {
    /// `ink8um.inference.predict_stack` as released: coverage = max over the window > 0, closed with a
    /// 7 x 7 ellipse; only tiles entirely inside it; a ±1 sigma window; divided by the tile count;
    /// `sigmoid` of the FP16 logit; clipped to 0..1 with 0 where no tile
    Release,
    /// `aclick/bench/infer.py --tta 1` of the 306B claim (the A-hn member maps): a tile is admitted when
    /// at least 25 % of it has CT on stored layer `layer0 + 12`; a ±3 sigma window; divided by the
    /// summed weight; `sigmoid` of the float32 logit; NaN where no tile
    Ahn306b,
}

impl Convention {
    /// The convention a sidecar's `map_convention.name` or a `--convention` value names
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "release" => Ok(Self::Release),
            "306b" => Ok(Self::Ahn306b),
            other => color_eyre::eyre::bail!("map convention {other:?}: need release or 306b"),
        }
    }

    /// The convention as `scripts/export_v8in.py` writes it into the sidecar's `map_convention`; an
    /// export whose record differs in any field is refused, so the metadata cannot drift from the code
    pub fn spec(self) -> serde_json::Value {
        let common = serde_json::json!({
            "layers": "central 24, stored order",
            "reverse": false,
            "lattice": "stride 21 from the origin, tiles entirely inside",
        });
        let own = match self {
            Self::Release => serde_json::json!({
                "name": "release", "runner": "ink8um.inference.predict_stack",
                "admit": "every pixel inside the 7x7-ellipse-closed coverage of the 24 layers", "admit_layer": null,
                "window_nsig": 1, "divide": "tile count", "empty": "0", "clip": [0, 1], "sigmoid": "fp16 logit",
            }),
            Self::Ahn306b => serde_json::json!({
                "name": "306b", "runner": "aclick/bench/infer.py --tta 1",
                "admit": "at least a quarter of the tile nonzero on one layer", "admit_layer": 12,
                "window_nsig": 3, "divide": "weight sum", "empty": "NaN", "clip": null, "sigmoid": "fp32 logit",
            }),
        };
        let mut v = common;
        for (k, x) in own.as_object().into_iter().flatten() {
            v[k] = x.clone();
        }
        v
    }

    /// The Gaussian window's half-width in standard deviations
    pub fn nsig(self) -> u32 {
        match self {
            Self::Release => 1,
            Self::Ahn306b => 3,
        }
    }

    /// `v8_blend`'s mode: 0 divides by the tile count, 1 by the weight sum
    pub fn blend_mode(self) -> u32 {
        match self {
            Self::Release => 0,
            Self::Ahn306b => 1,
        }
    }

    /// Whether the probability is `sigmoid` of the FP16 logit, rounded to FP16
    pub fn sigmoid16(self) -> bool {
        self == Self::Release
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Ahn306b => "306b",
        }
    }
}

/// `ink8um.inference.gkern(64, nsig) / max` as float32, from a fixture written by torch-free numpy
/// (`scripts/v8_windows.py`), since the reference builds it from Python's `math.erf`
pub fn window(nsig: u32) -> Result<Vec<f32>> {
    let bytes: &[u8] = match nsig {
        1 => include_bytes!("../fixtures/v8-gauss1.f32"),
        3 => include_bytes!("../fixtures/v8-gauss3.f32"),
        _ => color_eyre::eyre::bail!("no window for ±{nsig} sigma"),
    };
    ensure!(bytes.len() == PIX * 4, "window fixture of {} bytes", bytes.len());
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

/// PyTorch's `upsample_trilinear3d` source index for output `u` of an axis of `len` source samples at
/// `align_corners=False` and a given size (scale `len / (4 len)` = 0.25): `(i0, i1, lambda0, lambda1)`
fn interp(u: usize, len: usize) -> (usize, usize, f64, f64) {
    // float32 in PyTorch, exact here: 0.25 * (u + 0.5) - 0.5 has at most 3 fractional bits
    let src = (0.25 * (u as f64 + 0.5) - 0.5).max(0.0);
    let i0 = src as usize;
    let i1 = i0 + usize::from(i0 < len - 1);
    let l1 = src - i0 as f64;
    (i0, i1, 1.0 - l1, l1)
}

/// One axis of the composed stem: per stem output, its class and its window's first source index; per
/// class, the `[7, 4]` weights that map the conv's 7 taps on the upsampled axis to the 4 source samples
struct Axis {
    cls: Vec<u32>,
    base: Vec<i32>,
    mats: Vec<[[f64; J]; K7]>,
}

impl Axis {
    // the tap index enters the source offset and addresses the per-source weight rows, so a range reads clearest
    #[allow(clippy::needless_range_loop)]
    fn new(len: usize, stride: usize) -> Result<Self> {
        let up = UP * len;
        let outs = (up - 1) / stride + 1;
        let (mut cls, mut base, mut mats) = (Vec::new(), Vec::new(), Vec::<[[f64; J]; K7]>::new());
        for o in 0..outs {
            // weights[k][s] over the whole source axis; zero padding drops taps outside the upsampled axis
            let mut wts = vec![[0f64; K7]; len];
            for k in 0..K7 {
                let u = (o * stride) as i64 + k as i64 - PAD;
                if u < 0 || u >= up as i64 {
                    continue;
                }
                let (i0, i1, l0, l1) = interp(u as usize, len);
                wts[i0][k] += l0;
                wts[i1][k] += l1;
            }
            let nz: Vec<usize> = (0..len).filter(|&s| wts[s].iter().any(|&v| v != 0.0)).collect();
            let (lo, hi) = (nz[0], *nz.last().unwrap_or(&nz[0]));
            ensure!(hi - lo < J, "stem output {o} reads source {lo}..={hi}, more than {J} samples");
            let mut m = [[0f64; J]; K7];
            for s in lo..=hi {
                for (k, row) in m.iter_mut().enumerate() {
                    row[s - lo] = wts[s][k];
                }
            }
            let id = match mats
                .iter()
                .position(|q| q.iter().flatten().zip(m.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits()))
            {
                Some(i) => i,
                None => {
                    mats.push(m);
                    mats.len() - 1
                }
            };
            cls.push(id as u32);
            base.push(lo as i32);
        }
        Ok(Self { cls, base, mats })
    }
}

/// The composed stem `v8_stem` reads: the trilinear upsample and the 7 x 7 x 7 stride-(1, 2, 2) conv as
/// one 4 x 4 x 4 kernel per class triple, with the stem BatchNorm's scale folded in (float64, rounded to
/// float32 once) and its shift as `bias`
pub struct StemTables {
    /// `[class, tap, 64]`, class `(cd * nh + ch) * nh + cw`, tap `(jd * 4 + jh) * 4 + jw`
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
    pub dcls: Vec<u32>,
    pub dbase: Vec<i32>,
    pub hcls: Vec<u32>,
    pub hbase: Vec<i32>,
    pub nd: usize,
    pub nh: usize,
}

impl StemTables {
    /// `conv` is `backbone.conv1.weight` (`[64, 1, 7, 7, 7]`, float32), `scale` and `shift` the folded
    /// `backbone.bn1`
    // the separable composition indexes several kernel arrays by the same tap indices at once
    #[allow(clippy::needless_range_loop)]
    pub fn new(conv: &[f32], scale: &[f64], shift: &[f64]) -> Result<Self> {
        ensure!(conv.len() == SC * K7 * K7 * K7 && scale.len() == SC && shift.len() == SC, "stem weight shapes");
        let ad = Axis::new(D, 1)?;
        let ah = Axis::new(P, 2)?;
        ensure!(ad.cls.len() == SD && ah.cls.len() == SH, "stem axes {} and {}", ad.cls.len(), ah.cls.len());
        let (nd, nh) = (ad.mats.len(), ah.mats.len());
        // the kernel's interior classes are those of stem rows 64 and 65; check they are the interior's
        for o in 4..SH - 4 {
            ensure!(
                ah.cls[o] == ah.cls[64 + (o % 2)]
                    && ah.base[o] == ah.base[64 + (o % 2)] + (o as i32 - 64 - (o % 2) as i32) / 2,
                "stem row {o} breaks the interior pattern"
            );
        }
        // every block's source window fits the kernel's 12 x 12
        for p0 in (0..PH).step_by(PP) {
            let lo = (2 * p0 as i64 - 1).max(0) as usize;
            let hi = (2 * p0 + 2 * PP - 1).min(SH - 1);
            let r0 = ah.base[lo];
            for o in lo..=hi {
                let off = ah.base[o] - r0;
                ensure!((0..=WIN - J as i32).contains(&off), "stem patch {p0}: row {o} at window offset {off}");
            }
        }
        let mut w = vec![0f32; nd * nh * nh * TAPS * SC];
        for c in 0..SC {
            let wc = &conv[c * 343..(c + 1) * 343];
            for (cw, mw) in ah.mats.iter().enumerate() {
                // a[kd][kh][jw]
                let mut a = [[[0f64; J]; K7]; K7];
                for kd in 0..K7 {
                    for kh in 0..K7 {
                        for jw in 0..J {
                            a[kd][kh][jw] = (0..K7).map(|kw| wc[(kd * K7 + kh) * K7 + kw] as f64 * mw[kw][jw]).sum();
                        }
                    }
                }
                for (ch, mh) in ah.mats.iter().enumerate() {
                    let mut bm = [[[0f64; J]; J]; K7];
                    for kd in 0..K7 {
                        for jh in 0..J {
                            for jw in 0..J {
                                bm[kd][jh][jw] = (0..K7).map(|kh| a[kd][kh][jw] * mh[kh][jh]).sum();
                            }
                        }
                    }
                    for (cd, md) in ad.mats.iter().enumerate() {
                        let q = (cd * nh + ch) * nh + cw;
                        for jd in 0..J {
                            for jh in 0..J {
                                for jw in 0..J {
                                    let v: f64 = (0..K7).map(|kd| bm[kd][jh][jw] * md[kd][jd]).sum();
                                    let t = (jd * J + jh) * J + jw;
                                    w[(q * TAPS + t) * SC + c] = (v * scale[c]) as f32;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            w,
            bias: shift.iter().map(|&v| v as f32).collect(),
            dcls: ad.cls,
            dbase: ad.base,
            hcls: ah.cls,
            hbase: ah.base,
            nd,
            nh,
        })
    }
}

/// OpenCV's `getStructuringElement(MORPH_ELLIPSE, (7, 7))` as each row's half-width, rows -3..=3
const ELLIPSE7: [usize; 7] = [0, 2, 3, 3, 3, 2, 0];

/// `cv2.morphologyEx(mask, MORPH_CLOSE, ellipse 7 x 7)` with OpenCV's default border: a dilation in which
/// the outside counts as empty, then an erosion in which it counts as set
pub fn close7(mask: &[bool], h: usize, w: usize) -> Vec<bool> {
    use rayon::prelude::*;

    // per row, prefix counts of set pixels
    let prefix = |m: &[bool]| -> Vec<Vec<u32>> {
        m.par_chunks(w)
            .map(|row| {
                let mut p = vec![0u32; w + 1];
                for (x, &v) in row.iter().enumerate() {
                    p[x + 1] = p[x] + u32::from(v);
                }
                p
            })
            .collect()
    };
    let run = |p: &[u32], x: usize, dx: usize| {
        let (a, b) = (x.saturating_sub(dx), (x + dx + 1).min(w));
        (p[b] - p[a], (b - a) as u32)
    };
    let pd = prefix(mask);
    let dil: Vec<bool> = (0..h * w)
        .into_par_iter()
        .map(|i| {
            let (y, x) = (i / w, i % w);
            ELLIPSE7.iter().enumerate().any(|(r, &dx)| {
                let yy = y as i64 + r as i64 - 3;
                (0..h as i64).contains(&yy) && run(&pd[yy as usize], x, dx).0 > 0
            })
        })
        .collect();
    let pe = prefix(&dil);
    (0..h * w)
        .into_par_iter()
        .map(|i| {
            let (y, x) = (i / w, i % w);
            ELLIPSE7.iter().enumerate().all(|(r, &dx)| {
                let yy = y as i64 + r as i64 - 3;
                if !(0..h as i64).contains(&yy) {
                    return true;
                }
                let (set, len) = run(&pe[yy as usize], x, dx);
                set == len
            })
        })
        .collect()
}

/// The tile lattice `k * 21` along an axis of `len` pixels, tiles entirely inside it
pub fn lattice(len: usize) -> Vec<usize> {
    if len < P { Vec::new() } else { (0..=len - P).step_by(STRIDE).collect() }
}

/// Which lattice tiles a convention admits, row-major `[ny, nx]`
///
/// `inside` is the release's closed coverage mask (every pixel of the tile must be set); `ct` the 306B
/// rule's layer (at least a quarter of the tile set)
pub fn admit(cov: &[bool], h: usize, w: usize, convention: Convention) -> (Vec<usize>, Vec<usize>, Vec<bool>) {
    let (ys, xs) = (lattice(h), lattice(w));
    let mut integral = vec![0u32; (h + 1) * (w + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(cov[y * w + x]);
            integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row;
        }
    }
    let sum = |y: usize, x: usize| {
        let (y1, x1) = (y + P, x + P);
        integral[y1 * (w + 1) + x1] + integral[y * (w + 1) + x]
            - integral[y * (w + 1) + x1]
            - integral[y1 * (w + 1) + x]
    };
    let need = match convention {
        Convention::Release => PIX as u32,
        Convention::Ahn306b => (PIX / 4) as u32,
    };
    let keep = ys.iter().flat_map(|&y| xs.iter().map(move |&x| (y, x))).map(|(y, x)| sum(y, x) >= need).collect();
    (ys, xs, keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composed stem equals the upsample followed by the conv, in float64, on a random tile, at every
    /// border and interior class of a full stem output
    #[test]
    fn composed_stem_matches_upsample_then_conv() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let conv: Vec<f32> = (0..SC * 343).map(|_| rnd() as f32).collect();
        let scale: Vec<f64> = (0..SC).map(|_| 1.0 + rnd()).collect();
        let shift = vec![0f64; SC];
        let t = StemTables::new(&conv, &scale, &shift).unwrap();
        // a smaller source keeps the direct path cheap: the tables are built for 24 x 64 x 64, so test a
        // few outputs of the real size directly
        let src: Vec<f64> = (0..D * P * P).map(|_| rnd()).collect();
        let up = |d: usize, y: usize, x: usize| {
            let (d0, d1, a0, a1) = interp(d, D);
            let (y0, y1, b0, b1) = interp(y, P);
            let (x0, x1, c0, c1) = interp(x, P);
            let s = |a: usize, b: usize, c: usize| src[(a * P + b) * P + c];
            a0 * (b0 * (c0 * s(d0, y0, x0) + c1 * s(d0, y0, x1)) + b1 * (c0 * s(d0, y1, x0) + c1 * s(d0, y1, x1)))
                + a1 * (b0 * (c0 * s(d1, y0, x0) + c1 * s(d1, y0, x1)) + b1 * (c0 * s(d1, y1, x0) + c1 * s(d1, y1, x1)))
        };
        for &(od, oh, ow) in
            &[(0, 0, 0), (1, 1, 127), (2, 126, 2), (47, 64, 65), (93, 127, 1), (95, 3, 100), (50, 0, 64)]
        {
            for c in [0, 17, 63] {
                let mut direct = 0f64;
                for kd in 0..7 {
                    for kh in 0..7 {
                        for kw in 0..7 {
                            let (u, v, x) = (od as i64 + kd - 3, 2 * oh as i64 + kh - 3, 2 * ow as i64 + kw - 3);
                            if !(0..96).contains(&u) || !(0..256).contains(&v) || !(0..256).contains(&x) {
                                continue;
                            }
                            direct += conv[c * 343 + ((kd * 7 + kh) * 7 + kw) as usize] as f64
                                * up(u as usize, v as usize, x as usize);
                        }
                    }
                }
                direct *= scale[c];
                let q = (t.dcls[od] as usize * t.nh + t.hcls[oh] as usize) * t.nh + t.hcls[ow] as usize;
                let mut composed = 0f64;
                for jd in 0..J {
                    for jh in 0..J {
                        for jw in 0..J {
                            let (sd, sh, sw) =
                                (t.dbase[od] + jd as i32, t.hbase[oh] + jh as i32, t.hbase[ow] + jw as i32);
                            let wv = t.w[(q * TAPS + (jd * J + jh) * J + jw) * SC + c] as f64;
                            if wv == 0.0 {
                                continue;
                            }
                            assert!(
                                (0..D as i32).contains(&sd)
                                    && (0..P as i32).contains(&sh)
                                    && (0..P as i32).contains(&sw)
                            );
                            composed += wv * src[((sd as usize) * P + sh as usize) * P + sw as usize];
                        }
                    }
                }
                let tol = 1e-5 * direct.abs().max(1.0);
                assert!(
                    (direct - composed).abs() < tol,
                    "({od}, {oh}, {ow}) ch {c}: direct {direct} composed {composed}"
                );
            }
        }
        // 4 interior depth phases plus 10 border classes, 2 interior row phases plus 5 border classes
        assert_eq!((t.nd, t.nh), (14, 7));
    }

    #[test]
    fn close7_matches_opencv_on_a_small_case() {
        // a single pixel dilates to the ellipse and the erosion brings it back to one pixel
        let (h, w) = (11, 11);
        let mut m = vec![false; h * w];
        m[5 * w + 5] = true;
        assert_eq!(close7(&m, h, w), m);
        // closing fills a one-pixel hole in a block away from the border (checked against cv2)
        let n = 21;
        let mut m = vec![false; n * n];
        for y in 7..14 {
            for x in 7..14 {
                m[y * n + x] = (y, x) != (10, 10);
            }
        }
        let mut filled = m.clone();
        filled[10 * n + 10] = true;
        assert_eq!(close7(&m, n, n), filled);
        // near the border the erosion counts the outside as set, so the same block in 11 x 11 closes to the
        // whole image (cv2 agrees)
        let mut m = vec![false; h * w];
        for y in 2..9 {
            for x in 2..9 {
                m[y * w + x] = (y, x) != (5, 5);
            }
        }
        assert!(close7(&m, h, w).iter().all(|&v| v));
        // two pixels 5 apart stay apart: no ellipse translate covering the gap fits in the dilation
        let mut m = vec![false; h * w];
        m[5 * w + 2] = true;
        m[5 * w + 7] = true;
        assert_eq!(close7(&m, h, w), m);
        let full = vec![true; h * w];
        assert_eq!(close7(&full, h, w), full);
    }

    #[test]
    fn lattice_and_admission() {
        assert_eq!(lattice(63), Vec::<usize>::new());
        assert_eq!(lattice(64), [0]);
        assert_eq!(lattice(106), [0, 21, 42]);
        let (h, w) = (128, 106);
        let mut cov = vec![false; h * w];
        for y in 0..64 {
            for x in 0..32 {
                cov[y * w + x] = true;
            }
        }
        // tile (0, 0) is half covered: the 306B rule admits it, the release's does not
        let (_, _, keep) = admit(&cov, h, w, Convention::Ahn306b);
        assert!(keep[0] && !keep[2]);
        let (_, _, keep) = admit(&cov, h, w, Convention::Release);
        assert!(!keep[0]);
    }
}
