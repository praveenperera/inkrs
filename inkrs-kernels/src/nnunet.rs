//! Patch input, test-time augmentation, blend and finalisation for nnU-Net models run as villa's
//! `vesuvius.predict` runs them (`inkrs nnunet`), around the fp16 forward of [`crate::unet3d::fp16`]
//!
//! Per patch the host runs one forward per flip. [`nn_prep`] writes the flipped normalised input,
//! [`nn_head_tta`] the head's fp16 logits un-flipped into a float32 sum over the flips, [`nn_check`] the
//! patch's status (any non-zero mean logit, any non-finite one), and [`nn_fold`] their fp16 mean times
//! villa's Gaussian into the float32 accumulators, unless every mean logit is 0. When a row of output chunks
//! has every patch it overlaps, [`nn_finalize`] writes its uint8 output and per-chunk flags, so the host
//! can skip the chunks villa skips. Index math is 32-bit: the host keeps every buffer under 2^32 elements
//!
//! Flip `f` has bit 1 for x, 2 for y, 4 for z: the forward sees voxel `(p-1-z, y, x)` at `(z, y, x)`
//! when bit 4 is set, and so on

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_contract, thread, warp};

use crate::common::{math, narrow, widen};

/// `x` as fp16 holds it, round to nearest even
#[inline(always)]
fn r16(x: f32) -> f32 {
    widen(narrow(x))
}

/// The coordinate the forward sees at `l` on an axis of `p` when the flip mirrors that axis
#[inline(always)]
fn mirror(l: u32, p: u32, on: bool) -> u32 {
    if on { p - 1 - l } else { l }
}

/// One flip of one patch's model input: `table[code]` (fp16 bits, the normalisation of each uint8 code
/// rounded as villa's dataset casts it) for in-volume voxels, 0 elsewhere (villa zero-pads outside the
/// volume after normalising). Writes channel 0 of `x`, `[p^3, cstride]` fp16 bits; the host zero-fills the
/// other channels once
///
/// The patch's corner is `(oz, oy, ox)` in a uint8 slab of dims `(_, sh, sw)`; patch coordinates in
/// `[v0, v1)` on each axis are inside the volume. `flip` mirrors the patch as above
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nn_prep(
    mut x: DisjointSlice<u16>,
    slab: &[u8],
    table: &[u16],
    sh: u32,
    sw: u32,
    oz: u32,
    oy: u32,
    ox: u32,
    p: u32,
    vz0: u32,
    vz1: u32,
    vy0: u32,
    vy1: u32,
    vx0: u32,
    vx1: u32,
    cstride: u32,
    flip: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    if e >= p * p * p || table.len() < 256 {
        return;
    }
    let (fx, fy, fz) = (e % p, (e / p) % p, e / (p * p));
    let lx = mirror(fx, p, flip & 1 != 0);
    let ly = mirror(fy, p, flip & 2 != 0);
    let lz = mirror(fz, p, flip & 4 != 0);
    let inside = lz >= vz0 && lz < vz1 && ly >= vy0 && ly < vy1 && lx >= vx0 && lx < vx1;
    let v = if inside { table[slab[(((oz + lz) * sh + oy + ly) * sw + ox + lx) as usize] as usize] } else { 0 };
    let o = (e * cstride) as usize;
    if o < x.len() {
        // SAFETY: in bounds; each thread writes its own voxel's channel 0
        unsafe { *x.get_unchecked_mut(o) = v };
    }
}

/// The head of one flip's forward, un-flipped, into the float32 sum over flips
///
/// Per patch voxel `e` (patch coordinates) and class `k`: the forward computed this voxel at its mirror
/// `s`, so `logit = r16(r16(sum_c x[s, c] * hw[k, c]) + hb[k])` (PyTorch adds the conv's bias in a separate
/// fp16 op), and `sum[k, e] = logit` on the first flip, `sum[k, e] + logit` after. `x` is the last decoder
/// stage `[p^3, cols]` fp16 pairs; `hw` `[classes, cols]` and `hb` `[classes]` are the head's fp16 weight and
/// bias as float32
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nn_head_tta(
    mut sum: DisjointSlice<f32>,
    x: &[u32],
    hw: &[f32],
    hb: &[f32],
    cols: u32,
    classes: u32,
    p: u32,
    flip: u32,
    first: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let pairs = cols / 2;
    let vox = p * p * p;
    if e >= vox || pairs == 0 || hw.len() < (classes * cols) as usize || hb.len() < classes as usize {
        return;
    }
    let (lx, ly, lz) = (e % p, (e / p) % p, e / (p * p));
    let sx = mirror(lx, p, flip & 1 != 0);
    let sy = mirror(ly, p, flip & 2 != 0);
    let sz = mirror(lz, p, flip & 4 != 0);
    let s = (sz * p + sy) * p + sx;
    let mut k = 0;
    while k < classes {
        let mut dot = 0f32;
        let mut cp = 0;
        while cp < pairs {
            let v = x[(s * pairs + cp) as usize];
            dot = fma_rn_f32(widen(v as u16), hw[(k * cols + 2 * cp) as usize], dot);
            dot = fma_rn_f32(widen((v >> 16) as u16), hw[(k * cols + 2 * cp + 1) as usize], dot);
            cp += 1;
        }
        let logit = r16(add_rn_f32(r16(dot), hb[k as usize]));
        let o = (k * vox + e) as usize;
        if o < sum.len() {
            // SAFETY: in bounds; each thread owns voxel `e` of every class
            unsafe {
                let a = sum.get_unchecked_mut(o);
                *a = if first != 0 { logit } else { add_rn_f32(*a, logit) };
            }
        }
        k += 1;
    }
}

/// Whether `v` is finite; false for NaN
#[inline(always)]
fn finite(v: f32) -> bool {
    abs(v) <= 3.402_823_5e38
}

/// Status bit of a patch with a mean logit other than 0: villa's `InMemoryBlender.add` skips a patch whose
/// logits are all 0, adding neither logits nor weight
pub const NN_NONZERO: u32 = 1;
/// Status bit of a patch with a non-finite flip sum or mean logit: the host fails the run before writing
pub const NN_NONFINITE: u32 = 2;

/// Patch `idx`'s status after its last flip, OR-ed into `stat[idx]` (the host zeroes `stat` once): bit
/// [`NN_NONZERO`] when one of its fp16 mean logits `r16(sum[k, e] * inv_n)` is not 0, bit [`NN_NONFINITE`]
/// when a sum or mean is not finite. Every finite mean is at most 65504 in magnitude and every weight in
/// [0, 1], so the float32 accumulators stay finite when every patch passes. Each warp reduces its lanes
/// and one lane ORs the bits in, so the patch's word sees one atomic per warp, not one per voxel
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nn_check(mut stat: DisjointSlice<u32>, sum: &[f32], inv_n: f32, classes: u32, p: u32, idx: u32) {
    let e = thread::index_1d_u32(lc).get();
    let vox = p * p * p;
    // no early return: every lane joins the shuffles below
    let live = e < vox && sum.len() >= (classes * vox) as usize;
    let (mut nonzero, mut bad) = (0f32, 0f32);
    let mut k = 0;
    while live && k < classes {
        let s = sum[(k * vox + e) as usize];
        let mean = r16(mul_rn_f32(s, inv_n));
        if mean != 0.0 {
            nonzero = 1.0;
        }
        if !finite(s) || !finite(mean) {
            bad = 1.0;
        }
        k += 1;
    }
    let mut m = 1;
    while m < 32 {
        nonzero = math::fmaxf(nonzero, warp::shuffle_xor_f32(nonzero, m));
        bad = math::fmaxf(bad, warp::shuffle_xor_f32(bad, m));
        m *= 2;
    }
    let bits = if nonzero != 0.0 { NN_NONZERO } else { 0 } | if bad != 0.0 { NN_NONFINITE } else { 0 };
    if warp::lane_id() != 0 || bits == 0 || (idx as usize) >= stat.len() {
        return;
    }
    // SAFETY: `idx < stat.len()`; device-scope atomics, only ever setting bits
    unsafe { DeviceAtomicU32::from_ptr(stat.as_mut_ptr().add(idx as usize)).fetch_or(bits, AtomicOrdering::Relaxed) };
}

/// One patch's mean logits blended into the accumulators, as villa's `InMemoryBlender.add` does it
///
/// Per patch voxel and class: `mean = r16(sum[k, e] * inv_n)` (the flips' fp16 mean, float32 accumulated:
/// `inv_n` is 1/8 with mirroring, 1 without), then `acc[k] += mean * w` and `wacc += w` with `w = gauss[e]`,
/// each product and sum rounded on its own as numpy's float32 ops are. The patch's corner is global `(pz,
/// py, px)`; the accumulators cover `[rz0, rz0 + rd) x [ry0, ry0 + rh) x [rx0, rx0 + rw)`, z stored modulo
/// `ring`, class `k` at `k * ring * rh * rw`. Voxels outside are skipped (outside the volume, or in chunk
/// rows already written), and the whole patch when `stat[idx]` lacks [`NN_NONZERO`] ([`nn_check`])
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nn_fold(
    mut acc: DisjointSlice<f32>,
    mut wacc: DisjointSlice<f32>,
    sum: &[f32],
    gauss: &[f32],
    stat: &[u32],
    idx: u32,
    classes: u32,
    inv_n: f32,
    p: u32,
    pz: u32,
    py: u32,
    px: u32,
    rz0: u32,
    ry0: u32,
    rx0: u32,
    rd: u32,
    rh: u32,
    rw: u32,
    ring: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let vox = p * p * p;
    if e >= vox || ring == 0 || classes == 0 || gauss.len() < vox as usize || sum.len() < (classes * vox) as usize {
        return;
    }
    if (idx as usize) >= stat.len() || stat[idx as usize] & NN_NONZERO == 0 {
        return;
    }
    let (lx, ly, lz) = (e % p, (e / p) % p, e / (p * p));
    let (gz, gy, gx) = (pz + lz, py + ly, px + lx);
    if gz < rz0 || gz >= rz0 + rd || gy < ry0 || gy >= ry0 + rh || gx < rx0 || gx >= rx0 + rw {
        return;
    }
    let w = gauss[e as usize];
    let plane = ring * rh * rw;
    let o = (((gz - rz0) % ring) * rh + gy - ry0) * rw + gx - rx0;
    if (o as usize) >= wacc.len() || ((classes - 1) * plane + o) as usize >= acc.len() {
        return;
    }
    let mut k = 0;
    while k < classes {
        let mean = r16(mul_rn_f32(sum[(k * vox + e) as usize], inv_n));
        // SAFETY: in bounds by the check above; one patch's voxels map to distinct slots, and one
        // patch's launch finishes before the next queues on the stream
        unsafe {
            let a = acc.get_unchecked_mut((k * plane + o) as usize);
            *a = add_rn_f32(*a, mul_rn_f32(mean, w));
        }
        k += 1;
    }
    // SAFETY: as above
    unsafe {
        let s = wacc.get_unchecked_mut(o as usize);
        *s = add_rn_f32(*s, w);
    }
}

/// Normalised logit of class `k` at accumulator slot `o`: villa divides by `w + 1e-8` where `w > 0` and
/// leaves the sum (0) elsewhere
#[inline(always)]
fn normalised(acc: &[f32], wacc: &[f32], k: u32, plane: u32, o: u32) -> f32 {
    let w = wacc[o as usize];
    let l = acc[(k * plane + o) as usize];
    if w > 0.0 { l / add_rn_f32(w, 1e-8) } else { l }
}

#[inline(always)]
fn abs(v: f32) -> f32 {
    if v < 0.0 { -v } else { v }
}

/// Whether `l` fails numpy's `isclose(l, first)` with tolerance `tol`; a NaN is never close
#[inline(always)]
fn far(l: f32, first: f32, tol: f32) -> bool {
    !(abs(l - first) <= tol)
}

/// One row of output chunks, finalised as villa's `_write_blended_chunk` and `apply_finalization` do for
/// a binary two-class model
///
/// The row is `rows` z-slices from accumulator z-slice `az` (ring index) by the accumulator's `rh x rw`
/// y-x extent, which is cut into `chunk`-voxel chunks on y and x (the extent starts on the chunk grid; the
/// last chunk may be short). Per voxel, with `l0`, `l1` the normalised logits:
/// - uncovered (both exactly 0): 0;
/// - `cutoff_on != 0`: 255 where `l1 - l0 > cutoff` (float32), else 0;
/// - otherwise `trunc(clip(softmax[1] * 255, 0, 255))`, softmax as `exp(l - max) / sum`.
///
/// `flags[2 * c]` gets 1 when chunk `c` has a voxel of weight above 0 and `flags[2 * c + 1]` when one of
/// its logits is not `allclose` (rtol 1e-6, atol 1e-8, float32) to the chunk's first, class 0 at its
/// corner; villa writes a chunk only when both hold. The host zeroes `flags`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nn_finalize(
    mut out: DisjointSlice<u8>,
    mut flags: DisjointSlice<u32>,
    acc: &[f32],
    wacc: &[f32],
    az: u32,
    rows: u32,
    rh: u32,
    rw: u32,
    ring: u32,
    chunk: u32,
    cutoff_on: u32,
    cutoff: f32,
) {
    let e = thread::index_1d_u32(lc).get();
    if e >= rows * rh * rw || chunk == 0 {
        return;
    }
    let (x, y, z) = (e % rw, (e / rw) % rh, e / (rw * rh));
    let plane = ring * rh * rw;
    let o = ((az + z) * rh + y) * rw + x;
    if (o as usize) >= wacc.len() || (plane + o) as usize >= acc.len() {
        return;
    }
    let l0 = normalised(acc, wacc, 0, plane, o);
    let l1 = normalised(acc, wacc, 1, plane, o);
    let v = if l0 == 0.0 && l1 == 0.0 {
        0
    } else if cutoff_on != 0 {
        if l1 - l0 > cutoff { 255 } else { 0 }
    } else {
        let m = math::fmaxf(l0, l1);
        let (e0, e1) = (math::expf(l0 - m), math::expf(l1 - m));
        let prob = e1 / add_rn_f32(e0, e1);
        math::fminf(math::fmaxf(mul_rn_f32(prob, 255.0), 0.0), 255.0) as u8
    };
    if (e as usize) < out.len() {
        // SAFETY: in bounds; each thread writes its own voxel
        unsafe { *out.get_unchecked_mut(e as usize) = v };
    }

    let (cy, cx) = (y / chunk, x / chunk);
    let c = cy * ((rw + chunk - 1) / chunk) + cx;
    let f = (2 * c) as usize;
    if f + 1 >= flags.len() {
        return;
    }
    let base = flags.as_mut_ptr();
    if wacc[o as usize] > 0.0 {
        // SAFETY: `f < flags.len()`; device-scope atomics, only ever setting the flag to 1
        unsafe { DeviceAtomicU32::from_ptr(base.add(f)).fetch_or(1, AtomicOrdering::Relaxed) };
    }
    let corner = (az * rh + cy * chunk) * rw + cx * chunk;
    let first = normalised(acc, wacc, 0, plane, corner);
    let tol = add_rn_f32(1e-8, mul_rn_f32(1e-6, abs(first)));
    if far(l0, first, tol) || far(l1, first, tol) {
        // SAFETY: as above
        unsafe { DeviceAtomicU32::from_ptr(base.add(f + 1)).fetch_or(1, AtomicOrdering::Relaxed) };
    }
}
