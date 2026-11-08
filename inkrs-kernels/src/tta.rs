//! 8-fold dihedral test-time augmentation for the 2D-output readers, as the PHerc. 0306B recipe runs it
//! (`zoomlib.villa_map(..., tta8=True)` over `fastink.VillaFast.predict(..., tta="mirror4")`)
//!
//! The recipe reads the map twice, once on the window as stored and once on the window turned a quarter
//! (numpy `rot90(win, 1, axes=(1, 2))`), each with its own villa tiling. Inside each read every tile
//! runs four times, as stored and flipped in y, in x and in both; `tile_norm` writes the flipped copies
//! (descriptor field 5), and [`tta_blend_mirror4`] reads each copy's logits back through the same flip
//! and averages the four probabilities before villa's blend. Each read ends as villa's uint8 map, and
//! [`tta_combine`] turns the second back and averages the two in float32. Every kernel here takes the
//! tile edge at run time, so the 128 and 384 px readers share them

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{add_rn_f32, math, mul_rn_f32, widen};

/// Copies per tile of a mirror4 read: as stored, flipped in y, in x, in both
pub const MIRRORS: u32 = 4;

/// The volume turned a quarter, numpy `rot90(vol, 1, axes=(1, 2))`: `out[z, i, j] = vol[z, j, width - 1 - i]`,
/// so `out` is `[depth, width, height]`. One thread per byte
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn tta_rot90_u8(mut out: DisjointSlice<u8>, vol: &[u8], height: u32, width: u32) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let plane = height * width;
    let (z, r) = (e / plane, e % plane);
    let (i, j) = (r / height, r % height);
    let Some(mut out) = out.element_thread32(t) else { return };
    let src = (z * plane + j * width + (width - 1 - i)) as usize;
    if src < vol.len() {
        out.write(vol[src]);
    }
}

/// The logit copy `k` of a mirror4 read wrote for tile pixel `(ly, lx)`: copy `k` ran on the tile flipped
/// by its bits (1: y, 2: x), so its output is read through the same flip
#[inline(always)]
fn mirrored(base: usize, k: u32, ly: u32, lx: u32, edge: u32) -> usize {
    let fy = if k & 1 != 0 { edge - 1 - ly } else { ly };
    let fx = if k & 2 != 0 { edge - 1 - lx } else { lx };
    base + (k * edge * edge + fy * edge + fx) as usize
}

#[inline(always)]
fn sigmoid(bits: u16) -> f32 {
    1.0 / (1.0 + math::expf(-widen(bits)))
}

/// One map of a mirror4 read: villa's blend of each tile's mean probability over its four copies, and
/// villa's uint8 output `[height, width]`
///
/// As `blend_finish`, except that a tile's probability is `(((p0 + p1) + p2) + p3) / 4`, the sum in the
/// recipe's order (`TTA["mirror4"]`: none, y, x, both) of the four copies' float32 sigmoids, each read
/// back through its flip; copy `k` of the tile in grid slot `s` is logits slot `4 s + k`. `edge` is the
/// tile edge and `weight` its `[edge, edge]` blend window
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn tta_blend_mirror4(
    mut out: DisjointSlice<u8>,
    logits: &[u16],
    grid: &[i32],
    ys: &[u32],
    xs: &[u32],
    y_span: &[[u32; 2]],
    x_span: &[[u32; 2]],
    weight: &[f32],
    width: u32,
    edge: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (y, x) = (i / width, i % width);
    let Some(mut out) = out.element_thread32(t) else { return };
    let pixels = (edge * edge) as usize;
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
                let base = slot as usize * MIRRORS as usize * pixels;
                if base + MIRRORS as usize * pixels <= logits.len() {
                    let p0 = sigmoid(logits[mirrored(base, 0, ly, lx, edge)]);
                    let p1 = sigmoid(logits[mirrored(base, 1, ly, lx, edge)]);
                    let p2 = sigmoid(logits[mirrored(base, 2, ly, lx, edge)]);
                    let p3 = sigmoid(logits[mirrored(base, 3, ly, lx, edge)]);
                    let p = add_rn_f32(add_rn_f32(add_rn_f32(p0, p1), p2), p3) / 4.0;
                    let w = weight[(ly * edge + lx) as usize];
                    num = add_rn_f32(num, mul_rn_f32(p, w));
                    den = add_rn_f32(den, w);
                }
            }
            ix += 1;
        }
        iy += 1;
    }
    let p = if den > 1e-6 { num / den } else { num };
    let p = math::fminf(math::fmaxf(p, 0.0), 1.0);
    out.write(mul_rn_f32(p, 255.0) as u8);
}

/// The 8-fold map `[height, width]`: `(q0 / 255 + q1' / 255) / 2` in float32, the recipe's
/// `(a0 + rot90(a1, -1)) / 2` with `a = q.astype(float32) / 255.`, where `q0` is the upright read's
/// uint8 map and `q1` the quarter-turned read's `[width, height]`, turned back as
/// `q1'[y, x] = q1[width - 1 - x, y]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn tta_combine(mut out: DisjointSlice<f32>, q0: &[u8], q1: &[u8], height: u32, width: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (y, x) = (i / width, i % width);
    let Some(mut out) = out.element_thread32(t) else { return };
    let j = ((width - 1 - x) * height + y) as usize;
    if (i as usize) < q0.len() && j < q1.len() {
        let a = q0[i as usize] as f32 / 255.0;
        let b = q1[j] as f32 / 255.0;
        out.write(add_rn_f32(a, b) / 2.0);
    }
}
