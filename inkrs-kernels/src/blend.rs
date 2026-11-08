//! The 1x1 head, villa's floored-Hann blend and the uint8 finish
//!
//! villa adds each tile's `sigmoid(logit) * w` and `w` into float32 sums in tile order and
//! writes `trunc(255 * clip(sum_pw / sum_w, 0, 1))`. [`blend_finish`] gathers instead of
//! scattering: each pixel visits the tiles covering it in villa's tile order, with a separate
//! product and add (no FMA), so the sums match villa's bit for bit

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{add_rn_f32, math, mul_rn_f32, narrow, r16, unpack_f16, widen};
use crate::tiles::DESC;

/// Pixels per tile
const PIXELS: u32 = 128 * 128;

/// Head logits of every real tile into its slot of `logits` (`[slots, PIXELS]` FP16)
///
/// `feat` is the last decoder stage, `[B, PIXELS, 32]` FP16 pairs; `w` the 32 head weights and
/// `bias` its bias, FP16-rounded by the host. One thread per (tile, pixel); padding tiles
/// (slot < 0) write nothing
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn head_logits(mut logits: DisjointSlice<u16>, feat: &[[u32; 16]], w: &[f32], bias: f32, desc: &[i32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, p) = (i / PIXELS, i % PIXELS);
    let d = b as usize * DESC;
    if d + DESC > desc.len() || w.len() < 32 || i as usize >= feat.len() {
        return;
    }
    let slot = desc[d + 4];
    if slot < 0 {
        return;
    }
    let f = feat[i as usize];
    let mut dot = 0f32;
    #[unroll]
    for k in 0..16 {
        let (a, c) = unpack_f16(f[k]);
        dot += a * w[2 * k] + c * w[2 * k + 1];
    }
    let o = slot as usize * PIXELS as usize + p as usize;
    if o < logits.len() {
        // SAFETY: the host gives every real tile its own slot, so (slot, pixel) is written by
        // this thread only, and `o` is in bounds
        unsafe { *logits.get_unchecked_mut(o) = narrow(r16(dot) + bias) };
    }
}

/// One map: blend the logits of its tiles and write villa's uint8 output `[height, width]`
///
/// `grid` is `[ny, nx]`, the slot of the tile at row position `iy` and column position `ix` of
/// villa's grid, or -1 where villa skipped it; `ys` and `xs` the positions; `y_span[y]` and
/// `x_span[x]` the half-open range of positions whose tile covers row `y` or column `x`;
/// `weight` the `[128, 128]` blend window. Pixels no tile covers get villa's 0
///
/// `cov`, when not empty, receives the coverage mask villa's finish divides on: 1 where the
/// weight sum is above 1e-6, else 0, so a gate can compare the engine's own coverage with
/// villa's schedule instead of inferring it from the output
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn blend_finish(
    mut out: DisjointSlice<u8>,
    mut cov: DisjointSlice<u8>,
    logits: &[u16],
    grid: &[i32],
    ys: &[u32],
    xs: &[u32],
    y_span: &[[u32; 2]],
    x_span: &[[u32; 2]],
    weight: &[f32],
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (y, x) = (i / width, i % width);
    let Some(mut out) = out.element_thread32(t) else { return };
    // its own witness, since a thread index is not Copy; none on an empty `cov`
    let cov = cov.element_thread32(thread::index_1d_u32(lc));
    let nx = xs.len() as u32;
    let [y_first, y_end] = y_span[y as usize];
    let [x_first, x_end] = x_span[x as usize];
    let (mut num, mut den) = (0f32, 0f32);
    let mut iy = y_first;
    while iy < y_end {
        let ly = y - ys[iy as usize];
        let mut ix = x_first;
        while ix < x_end {
            let slot = grid[(iy * nx + ix) as usize];
            if slot >= 0 {
                let lx = x - xs[ix as usize];
                let l = widen(logits[slot as usize * PIXELS as usize + (ly * 128 + lx) as usize]);
                let p = 1.0 / (1.0 + math::expf(-l));
                let w = weight[(ly * 128 + lx) as usize];
                num = add_rn_f32(num, mul_rn_f32(p, w));
                den = add_rn_f32(den, w);
            }
            ix += 1;
        }
        iy += 1;
    }
    let covered = den > 1e-6;
    let p = if covered { num / den } else { num };
    let p = math::fminf(math::fmaxf(p, 0.0), 1.0);
    out.write(mul_rn_f32(p, 255.0) as u8);
    if let Some(mut c) = cov {
        c.write(u8::from(covered));
    }
}
