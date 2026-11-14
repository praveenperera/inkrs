//! The reference's Numba kernels (`core.py`), ported expression by expression
//!
//! Every expression keeps the float32/float64 typing Numba infers for it (as Numba's type dump
//! reports it), so results are bitwise the reference's: a float32 times a uint8
//! is float32, a float32 with a float64 literal, an int64 or a uint16 is float64, variables of mixed
//! type are float64, `int()` truncates, and two-argument `min`/`max` keep the first argument on ties.
//! Line numbers refer to `core.py` at bee798f7

use std::sync::OnceLock;

use color_eyre::eyre::{Result, bail};

use crate::pyfmt::{py_max, py_min};
use crate::vol::Vol;

/// cos(55°): the turn limit used throughout the tracer
pub const COS55: f64 = 0.5735764363510462;

/// Read-only access to a 3D field in z, y, x order
pub trait Grid<T: Copy>: Sync {
    /// Extent as D, H, W
    fn dims(&self) -> [i64; 3];
    /// The value at an in-bounds voxel
    fn at(&self, z: i64, y: i64, x: i64) -> T;
}

impl<T: Copy + PartialEq + Send + Sync> Grid<T> for Vol<T> {
    #[inline(always)]
    fn dims(&self) -> [i64; 3] {
        self.shape().map(|n| n as i64)
    }

    #[inline(always)]
    fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.get(z as usize, y as usize, x as usize)
    }
}

/// A dense C-order block, as handed to `stage1_centrality_and_ridge`
pub struct Dense<T> {
    pub dims: [usize; 3],
    pub data: Vec<T>,
}

impl<T: Copy + Sync> Grid<T> for Dense<T> {
    #[inline(always)]
    fn dims(&self) -> [i64; 3] {
        self.dims.map(|n| n as i64)
    }

    #[inline(always)]
    fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.data[((z as usize) * self.dims[1] + y as usize) * self.dims[2] + x as usize]
    }
}

/// `decode_dir` for every byte pair; the function is pure, so a table gives the same values
fn dir_table() -> &'static [[f64; 3]] {
    static TABLE: OnceLock<Vec<[f64; 3]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Vec::with_capacity(65536);
        for nx in 0..=255u8 {
            for ny in 0..=255u8 {
                t.push(decode_dir_calc(nx, ny));
            }
        }
        t
    })
}

/// `decode_dir` (core.py:8): all float64 after the float32 conversion of each byte
fn decode_dir_calc(nx: u8, ny: u8) -> [f64; 3] {
    let mut dx = (nx as f32 as f64 - 128.0) / 127.0;
    let mut dy = (ny as f32 as f64 - 128.0) / 127.0;
    let r2 = dx * dx + dy * dy;
    if r2 <= 1.0 {
        return [dx, dy, (1.0f32 as f64 - r2).sqrt()];
    }
    let inv = 1.0f32 as f64 / r2.sqrt();
    dx *= inv;
    dy *= inv;
    [dx, dy, 0.0f32 as f64]
}

#[inline(always)]
pub fn decode_dir(nx: u8, ny: u8) -> [f64; 3] {
    dir_table()[(nx as usize) << 8 | ny as usize]
}

/// `encode_dir` (core.py:23), with float32 tangents
#[inline]
pub fn encode_dir(mut ux: f32, mut uy: f32, uz: f32) -> (u8, u8) {
    if (uz as f64) < 0.0 {
        ux = -ux;
        uy = -uy;
    }
    let b_nx = py_min(py_max(ux as f64 * 127.0 + 128.5, 1.0), 255.0) as i64;
    let b_ny = py_min(py_max(uy as f64 * 127.0 + 128.5, 1.0), 255.0) as i64;
    (b_nx as u8, b_ny as u8)
}

/// The clamp and cell split shared by the trilinear helpers (core.py:36-53)
#[inline(always)]
fn cell(d: [i64; 3], mut x: f64, mut y: f64, mut z: f64) -> ([i64; 3], [f32; 3]) {
    let (dd, hh, ww) = (d[0] as f64, d[1] as f64, d[2] as f64);
    if x < 0.0 {
        x = 0.0;
    } else if x > ww - 1.001 {
        x = (ww - 1.001) as f32 as f64;
    }
    if y < 0.0 {
        y = 0.0;
    } else if y > hh - 1.001 {
        y = (hh - 1.001) as f32 as f64;
    }
    if z < 0.0 {
        z = 0.0;
    } else if z > dd - 1.001 {
        z = (dd - 1.001) as f32 as f64;
    }
    let ix = x as i64;
    let iy = y as i64;
    let iz = z as i64;
    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let fz = (z - iz as f64) as f32;
    ([iz, iy, ix], [fz, fy, fx])
}

/// `trilin_scalar_u8` (core.py:34): `fx * arr[..]` is a float32 product
#[inline]
pub fn trilin_u8<G: Grid<u8>>(arr: &G, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(arr.dims(), x, y, z);
    let a = |z: i64, y: i64, x: i64| arr.at(z, y, x);
    let lerp = |z: i64, y: i64| (1.0 - fx as f64) * a(z, y, ix) as f64 + (fx * a(z, y, ix + 1) as f32) as f64;
    let p00 = lerp(iz, iy);
    let p01 = lerp(iz, iy + 1);
    let p10 = lerp(iz + 1, iy);
    let p11 = lerp(iz + 1, iy + 1);
    let (fy, fz) = (fy as f64, fz as f64);
    (1.0 - fz) * ((1.0 - fy) * p00 + fy * p01) + fz * ((1.0 - fy) * p10 + fy * p11)
}

/// `trilin_scalar_u16` (core.py:62): float32 times uint16 is float64 in Numba
#[inline]
pub fn trilin_u16<G: Grid<u16>>(arr: &G, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(arr.dims(), x, y, z);
    let fx = fx as f64;
    let a = |z: i64, y: i64, x: i64| arr.at(z, y, x) as f64;
    let lerp = |z: i64, y: i64| (1.0 - fx) * a(z, y, ix) + fx * a(z, y, ix + 1);
    let p00 = lerp(iz, iy);
    let p01 = lerp(iz, iy + 1);
    let p10 = lerp(iz + 1, iy);
    let p11 = lerp(iz + 1, iy + 1);
    let (fy, fz) = (fy as f64, fz as f64);
    0.01f32 as f64 * ((1.0 - fz) * ((1.0 - fy) * p00 + fy * p01) + fz * ((1.0 - fy) * p10 + fy * p11))
}

/// The read-only fields one family traces over
pub struct Fields<'a, P, C, N> {
    pub pres: &'a P,
    pub cent: &'a C,
    pub nx: &'a N,
    pub ny: &'a N,
}

impl<P: Grid<u8>, C: Grid<u16>, N: Grid<u8>> Fields<'_, P, C, N> {
    #[inline(always)]
    fn dir_at(&self, z: i64, y: i64, x: i64) -> [f64; 3] {
        decode_dir(self.nx.at(z, y, x), self.ny.at(z, y, x))
    }

    /// `trilin_dir_aligned` (core.py:92)
    pub fn dir_aligned(&self, x: f64, y: f64, z: f64, rx: f64, ry: f64, rz: f64) -> [f64; 3] {
        let ([iz, iy, ix], [fz, fy, fx]) = cell(self.pres.dims(), x, y, z);
        let (fx, fy, fz) = (fx as f64, fy as f64, fz as f64);
        let (mut vx, mut vy, mut vz) = (0.0f64, 0.0f64, 0.0f64);
        for dz_i in 0..2 {
            let wz = if dz_i == 0 { 1.0 - fz } else { fz };
            let zz = iz + dz_i;
            for dy_i in 0..2 {
                let wy = wz * (if dy_i == 0 { 1.0 - fy } else { fy });
                let yy = iy + dy_i;
                for dx_i in 0..2 {
                    let w = wy * (if dx_i == 0 { 1.0 - fx } else { fx });
                    let xx = ix + dx_i;
                    let pw = w * (self.pres.at(zz, yy, xx) as f32 as f64 + 10.0);
                    let [mut ux, mut uy, mut uz] = self.dir_at(zz, yy, xx);
                    if ux * rx + uy * ry + uz * rz < 0.0 {
                        ux = -ux;
                        uy = -uy;
                        uz = -uz;
                    }
                    vx += pw * ux;
                    vy += pw * uy;
                    vz += pw * uz;
                }
            }
        }
        let n = (vx * vx + vy * vy + vz * vz).sqrt();
        if n > 1e-08 {
            vx /= n;
            vy /= n;
            vz /= n;
        }
        [vx, vy, vz]
    }

    /// `recenter_normal_plane` (core.py:162)
    #[allow(clippy::too_many_arguments)]
    pub fn recenter(
        &self,
        x: f64,
        y: f64,
        z: f64,
        tx: f64,
        ty: f64,
        tz: f64,
        max_shift: f64,
        n_iter: usize,
    ) -> [f64; 3] {
        let [d, h, w] = self.cent.dims().map(|n| n as f64);
        let [e1x, e1y, e1z, e2x, e2y, e2z] = ortho_basis(tx, ty, tz);
        let (mut cx, mut cy, mut cz) = (x, y, z);
        let ring = ring();
        let mut vals = [0.0f32; 13];
        for _ in 0..n_iter {
            let mut c_min = 1_000_000_000.0f32;
            let mut c_max = -1_000_000_000.0f32;
            for i in 0..13 {
                let (u1, u2) = (RING_U1[i] as f64, RING_U2[i] as f64);
                let px = cx + u1 * e1x + u2 * e2x;
                let py = cy + u1 * e1y + u2 * e2y;
                let pz = cz + u1 * e1z + u2 * e2z;
                if px < 0.0 || px >= w - 1.001 || py < 0.0 || py >= h - 1.001 || pz < 0.0 || pz >= d - 1.001 {
                    vals[i] = 0.0;
                } else {
                    let [ux, uy, uz] = self.dir_at((pz + 0.5) as i64, (py + 0.5) as i64, (px + 0.5) as i64);
                    let dot = (ux * tx + uy * ty + uz * tz).abs();
                    vals[i] = if dot < 0.82 { 0.0 } else { (trilin_u16(self.cent, px, py, pz) * dot) as f32 };
                }
                if vals[i] < c_min {
                    c_min = vals[i];
                }
                if vals[i] > c_max {
                    c_max = vals[i];
                }
            }
            if ((c_max - c_min) as f64) < 0.5 {
                break;
            }
            let (mut w_sum, mut du1, mut du2) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..13 {
                let ex = vals[i] - c_min;
                let wgt = (ex * ex) as f64 * ring[i];
                w_sum += wgt;
                du1 += wgt * RING_U1[i] as f64;
                du2 += wgt * RING_U2[i] as f64;
            }
            if w_sum > 1e-06 {
                du1 /= w_sum;
                du2 /= w_sum;
                cx += du1 * e1x + du2 * e2x;
                cy += du1 * e1y + du2 * e2y;
                cz += du1 * e1z + du2 * e2z;
            }
        }
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > max_shift {
            let s = max_shift / dist;
            cx = x + dx * s;
            cy = y + dy * s;
            cz = z + dz * s;
        }
        [cx, cy, cz]
    }
}

const RING_U1: [f32; 13] = [0.0, 0.55, 0.0, -0.55, 0.0, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317, 0.0, 0.81317];
const RING_U2: [f32; 13] = [0.0, 0.0, 0.55, 0.0, -0.55, 0.0, 0.81317, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317];

/// `exp_arr` of `recenter_normal_plane`: exp(-0.5 * r2) in float64 of a float32 r2
fn ring() -> &'static [f64; 13] {
    static RING: OnceLock<[f64; 13]> = OnceLock::new();
    RING.get_or_init(|| {
        std::array::from_fn(|i| {
            let r2 = RING_U1[i] * RING_U1[i] + RING_U2[i] * RING_U2[i];
            (-0.5 * r2 as f64).exp()
        })
    })
}

/// `ortho_basis` (core.py:142)
#[inline]
pub fn ortho_basis(ux: f64, uy: f64, uz: f64) -> [f64; 6] {
    let (mut e1x, mut e1y, mut e1z) = if uz.abs() < 0.85 { (-uy, ux, 0.0) } else { (0.0, -uz, uy) };
    let n1 = (e1x * e1x + e1y * e1y + e1z * e1z).sqrt();
    e1x /= n1;
    e1y /= n1;
    e1z /= n1;
    let e2x = uy * e1z - uz * e1y;
    let e2y = uz * e1x - ux * e1z;
    let e2z = ux * e1y - uy * e1x;
    [e1x, e1y, e1z, e2x, e2y, e2z]
}

/// Mutable coverage of one family: count (0, 1, 2) and the first marking direction
pub struct Coverage {
    pub cnt: Vol<u8>,
    pub nx: Vol<u8>,
    pub ny: Vol<u8>,
}

impl Coverage {
    pub fn new(shape: [usize; 3]) -> Self {
        Self { cnt: Vol::new(shape, 0), nx: Vol::new(shape, 0), ny: Vol::new(shape, 0) }
    }

    /// `is_covered_same_dir` (core.py:237)
    #[inline]
    pub fn covered_same_dir(&self, x: f64, y: f64, z: f64, tx: f64, ty: f64, tz: f64) -> bool {
        let [d, h, w] = self.cnt.dims();
        let ix = (x + 0.5) as i64;
        let iy = (y + 0.5) as i64;
        let iz = (z + 0.5) as i64;
        if ix < 0 || ix >= w || iy < 0 || iy >= h || iz < 0 || iz >= d {
            return false;
        }
        let c = self.cnt.at(iz, iy, ix);
        if c == 0 {
            return false;
        }
        if c >= 2 {
            return true;
        }
        let [ux, uy, uz] = decode_dir(self.nx.at(iz, iy, ix), self.ny.at(iz, iy, ix));
        (ux * tx + uy * ty + uz * tz).abs() > 0.86
    }

    /// `mark_curve_coverage` (core.py:527) with `core_r = 2.05`
    pub fn mark<P: Grid<u8>, C: Grid<u16>, N: Grid<u8>>(
        &mut self,
        f: &Fields<P, C, N>,
        pts: &[[f32; 3]],
        tans: &[[f32; 3]],
    ) {
        let [d, h, w] = self.cnt.dims();
        let core_r2 = (2.05f64 * 2.05) as f32 as f64;
        for (p, t) in pts.iter().zip(tans) {
            let [px, py, pz] = p.map(|v| v as f64);
            let [tx, ty, tz] = t.map(|v| v as f64);
            let (bnx, bny) = encode_dir(t[0], t[1], t[2]);
            let c0 = trilin_u16(f.cent, px, py, pz);
            let ix0 = (px + 0.5) as i64;
            let iy0 = (py + 0.5) as i64;
            let iz0 = (pz + 0.5) as i64;
            for zz in iz0 - 3..=iz0 + 3 {
                if zz < 0 || zz >= d {
                    continue;
                }
                for yy in iy0 - 3..=iy0 + 3 {
                    if yy < 0 || yy >= h {
                        continue;
                    }
                    for xx in ix0 - 3..=ix0 + 3 {
                        if xx < 0 || xx >= w {
                            continue;
                        }
                        let (vx, vy, vz) = (xx as f64 - px, yy as f64 - py, zz as f64 - pz);
                        let par = vx * tx + vy * ty + vz * tz;
                        if par.abs() > 0.9 {
                            continue;
                        }
                        let perp2 = vx * vx + vy * vy + vz * vz - par * par;
                        if perp2 > 3.5 * 3.5 {
                            continue;
                        }
                        let [ux, uy, uz] = f.dir_at(zz, yy, xx);
                        if (ux * tx + uy * ty + uz * tz).abs() <= 0.85 {
                            continue;
                        }
                        let suppress = perp2 <= core_r2 || {
                            let cq = (f.cent.at(zz, yy, xx) as f32 * 0.01f32) as f64;
                            let (xf, yf, zf) = (xx as f64, yy as f64, zz as f64);
                            let cm1 = trilin_u16(f.cent, 0.6 * px + 0.4 * xf, 0.6 * py + 0.4 * yf, 0.6 * pz + 0.4 * zf);
                            let cm2 =
                                trilin_u16(f.cent, 0.35 * px + 0.65 * xf, 0.35 * py + 0.65 * yf, 0.35 * pz + 0.65 * zf);
                            c0 + 2.0 >= cm1 && cm1 + 3.0 >= cm2 && cm2 + 3.0 >= cq
                        };
                        if !suppress {
                            continue;
                        }
                        let (zu, yu, xu) = (zz as usize, yy as usize, xx as usize);
                        if self.cnt.get(zu, yu, xu) == 0 {
                            self.cnt.set(zu, yu, xu, 1);
                            self.nx.set(zu, yu, xu, bnx);
                            self.ny.set(zu, yu, xu, bny);
                            continue;
                        }
                        let [ux0, uy0, uz0] = decode_dir(self.nx.get(zu, yu, xu), self.ny.get(zu, yu, xu));
                        if (ux0 * tx + uy0 * ty + uz0 * tz).abs() <= 0.85 {
                            self.cnt.set(zu, yu, xu, 2);
                        }
                    }
                }
            }
        }
    }
}

/// Centrality and ridge seeds of one tile (`stage1_centrality_and_ridge`, core.py:254, seed stride 2);
/// every coordinate and clamp is local to the tile, as in the reference
pub fn stage1(pres: &Dense<u8>, nx: &Dense<u8>, ny: &Dense<u8>) -> (Vec<u16>, Vec<u8>) {
    let [d, h, w] = pres.dims();
    let (df, hf, wf) = (d as f64, h as f64, w as f64);
    let n = pres.data.len();
    let mut cent = Dense { dims: pres.dims, data: vec![0u16; n] };
    // the reference's literals, rounded to float32 as Numba does; not the library constant
    #[allow(clippy::approx_constant)]
    const COS_A: [f32; 8] = [1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678, 0.0, 0.70710678];
    #[allow(clippy::approx_constant)]
    const SIN_A: [f32; 8] = [0.0, 0.70710678, 1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678];
    const RADII: [(f64, f64); 4] = [(1.0, 0.5), (1.8, 0.28), (2.7, 0.14), (3.6, 0.06)];
    let dir = |z: i64, y: i64, x: i64| decode_dir(nx.at(z, y, x), ny.at(z, y, x));
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let p0 = pres.at(z, y, x) as f32;
                if p0 < 64.0 {
                    continue;
                }
                let (xf, yf, zf) = (x as f64, y as f64, z as f64);
                let [ux, uy, uz] = dir(z, y, x);
                let p_p = trilin_u8(pres, xf + ux, yf + uy, zf + uz);
                let p_m = trilin_u8(pres, xf - ux, yf - uy, zf - uz);
                let p_par = 0.5 * p0 as f64 + 0.25 * p_p + 0.25 * p_m;
                let [e1x, e1y, e1z, e2x, e2y, e2z] = ortho_basis(ux, uy, uz);
                let mut c_sum = 0.0f64;
                for k in 0..8 {
                    let (ca, sa) = (COS_A[k] as f64, SIN_A[k] as f64);
                    let dx = ca * e1x + sa * e2x;
                    let dy = ca * e1y + sa * e2y;
                    let dz = ca * e1z + sa * e2z;
                    for (r, wr) in RADII {
                        let xx = py_min(py_max(xf + r * dx, 0.0), wf - 1.001);
                        let yy = py_min(py_max(yf + r * dy, 0.0), hf - 1.001);
                        let zz = py_min(py_max(zf + r * dz, 0.0), df - 1.001);
                        let pr = trilin_u8(pres, xx, yy, zz);
                        let [ux2, uy2, uz2] = dir((zz + 0.5) as i64, (yy + 0.5) as i64, (xx + 0.5) as i64);
                        let dr = (ux * ux2 + uy * uy2 + uz * uz2).abs();
                        if dr > 0.85 {
                            c_sum += wr / 8.0 * pr * ((dr - 0.85) / 0.15);
                        }
                    }
                }
                let c = py_min(py_max((p_par + c_sum) * 100.0 + 0.5, 0.0), 65535.0);
                cent.data[((z * h + y) * w + x) as usize] = c as u16;
            }
        }
    }
    let mut seeds = vec![0u8; n];
    for z in 1..d - 1 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if pres.at(z, y, x) < 79 || (x + y + z) & 1 != 0 {
                    continue;
                }
                let c0 = cent.at(z, y, x) as f32 * 0.01f32;
                let c0d = c0 as f64;
                let (xf, yf, zf) = (x as f64, y as f64, z as f64);
                let [ux, uy, uz] = dir(z, y, x);
                let [e1x, e1y, e1z, e2x, e2y, e2z] = ortho_basis(ux, uy, uz);
                let mut n_max = 0;
                for k in 0..4 {
                    let (ca, sa) = (COS_A[k] as f64, SIN_A[k] as f64);
                    let dx = ca * e1x + sa * e2x;
                    let dy = ca * e1y + sa * e2y;
                    let dz = ca * e1z + sa * e2z;
                    let cp = trilin_u16(&cent, xf + 1.15 * dx, yf + 1.15 * dy, zf + 1.15 * dz);
                    let cm = trilin_u16(&cent, xf - 1.15 * dx, yf - 1.15 * dy, zf - 1.15 * dz);
                    let cp2 = trilin_u16(&cent, xf + 2.0 * dx, yf + 2.0 * dy, zf + 2.0 * dz);
                    let cm2 = trilin_u16(&cent, xf - 2.0 * dx, yf - 2.0 * dy, zf - 2.0 * dz);
                    if c0d >= cp && c0d >= cm && c0d >= 0.5 * (cp2 + cm2) + 2.0 {
                        n_max += 1;
                    }
                }
                let i = ((z * h + y) * w + x) as usize;
                if n_max == 4 {
                    seeds[i] = 2;
                } else if n_max == 3 && c0d >= 185.0 {
                    seeds[i] = 1;
                }
            }
        }
    }
    (cent.data, seeds)
}

/// Half-path capacity of `trace_seed_range` (`pts_f`/`pts_b`)
pub const HALF: usize = 32768;

/// Scratch buffers for one tracing thread
pub struct Scratch {
    pts_f: Vec<[f32; 3]>,
    tans_f: Vec<[f32; 3]>,
    pts_b: Vec<[f32; 3]>,
    tans_b: Vec<[f32; 3]>,
    pts_c: Vec<[f32; 3]>,
    tans_c: Vec<[f32; 3]>,
}

impl Default for Scratch {
    fn default() -> Self {
        let buf = || vec![[0.0f32; 3]; HALF];
        Self {
            pts_f: buf(),
            tans_f: buf(),
            pts_b: buf(),
            tans_b: buf(),
            pts_c: vec![[0.0; 3]; 2 * HALF + 1],
            tans_c: vec![[0.0; 3]; 2 * HALF + 1],
        }
    }
}

/// One accepted raw curve; its tangents only matter for coverage, which is marked as it is accepted
pub struct RawCurve {
    pub pts: Vec<[f32; 3]>,
}

/// `trace_half_buf` (core.py:322) with `step_size = 0.75`, `max_low_steps = 3`
#[allow(clippy::too_many_arguments)]
fn trace_half<P: Grid<u8>, C: Grid<u16>, N: Grid<u8>>(
    f: &Fields<P, C, N>,
    cov: &Coverage,
    x0: f64,
    y0: f64,
    z0: f64,
    tx0: f64,
    ty0: f64,
    tz0: f64,
    pts: &mut [[f32; 3]],
    tans: &mut [[f32; 3]],
) -> usize {
    const STEP: f64 = 0.75;
    const MAX_LOW: i64 = 3;
    let [d, h, w] = f.pres.dims().map(|n| n as f64);
    let out =
        |x: f64, y: f64, z: f64| x < 0.25 || x >= w - 1.25 || y < 0.25 || y >= h - 1.25 || z < 0.25 || z >= d - 1.25;
    let max_steps = pts.len();
    let (mut x, mut y, mut z) = (x0, y0, z0);
    let (mut tx, mut ty, mut tz) = (tx0, ty0, tz0);
    let mut n_pts = 0usize;
    let mut low_pres_run = 0i64;
    let mut cov_run = 0i64;
    let p64 = |p: [f32; 3]| p.map(|v| v as f64);
    'steps: for _ in 0..max_steps {
        let mx = x + 0.5 * STEP * tx;
        let my = y + 0.5 * STEP * ty;
        let mz = z + 0.5 * STEP * tz;
        if out(mx, my, mz) {
            break;
        }
        let [mut ux, mut uy, mut uz] = f.dir_aligned(mx, my, mz, tx, ty, tz);
        let dot_u = ux * tx + uy * ty + uz * tz;
        let (nx_p, ny_p, nz_p, ntx, nty, ntz);
        if dot_u < COS55 {
            let mut crossed = false;
            for look_d in [1.5, 2.2] {
                let lx = x + look_d * tx;
                let ly = y + look_d * ty;
                let lz = z + look_d * tz;
                if lx >= 0.25 && lx < w - 1.25 && ly >= 0.25 && ly < h - 1.25 && lz >= 0.25 && lz < d - 1.25 {
                    let lp = trilin_u8(f.pres, lx, ly, lz);
                    let [lux, luy, luz] = f.dir_aligned(lx, ly, lz, tx, ty, tz);
                    if lp >= 72.5 && lux * tx + luy * ty + luz * tz >= COS55 {
                        crossed = true;
                        break;
                    }
                }
            }
            if !crossed {
                break;
            }
            nx_p = x + STEP * tx;
            ny_p = y + STEP * ty;
            nz_p = z + STEP * tz;
            (ntx, nty, ntz) = (tx, ty, tz);
            (ux, uy, uz) = (tx, ty, tz);
        } else {
            let mut ntx1 = 0.45 * tx + 0.55 * ux;
            let mut nty1 = 0.45 * ty + 0.55 * uy;
            let mut ntz1 = 0.45 * tz + 0.55 * uz;
            let nn = (ntx1 * ntx1 + nty1 * nty1 + ntz1 * ntz1).sqrt();
            ntx1 /= nn;
            nty1 /= nn;
            ntz1 /= nn;
            (ntx, nty, ntz) = (ntx1, nty1, ntz1);
            let px = x + STEP * ntx;
            let py = y + STEP * nty;
            let pz = z + STEP * ntz;
            if out(px, py, pz) {
                break;
            }
            [nx_p, ny_p, nz_p] = f.recenter(px, py, pz, ntx, nty, ntz, 0.42, 2);
        }
        let (mut dx, mut dy, mut dz) = (nx_p - x, ny_p - y, nz_p - z);
        let dl = (dx * dx + dy * dy + dz * dz).sqrt();
        #[allow(clippy::manual_range_contains)]
        if dl < 0.25 || dl > 1.45 {
            break;
        }
        dx /= dl;
        dy /= dl;
        dz /= dl;
        if dx * tx + dy * ty + dz * tz < COS55 {
            break;
        }
        if n_pts >= 4 {
            let p = p64(pts[n_pts - 4]);
            let (chx, chy, chz) = (nx_p - p[0], ny_p - p[1], nz_p - p[2]);
            let chl = (chx * chx + chy * chy + chz * chz).sqrt();
            if chl > 1e-05 && (chx * dx + chy * dy + chz * dz) / chl < COS55 {
                break;
            }
        } else {
            let (sx_d, sy_d, sz_d) = (nx_p - x0, ny_p - y0, nz_p - z0);
            let sl = (sx_d * sx_d + sy_d * sy_d + sz_d * sz_d).sqrt();
            if sl > 1e-05 && (sx_d * tx0 + sy_d * ty0 + sz_d * tz0) / sl < COS55 {
                break;
            }
        }
        if n_pts >= 4 {
            let p4 = pts[n_pts - 4];
            let q4 = p64(p4);
            let (v2x, v2y, v2z) = (nx_p - q4[0], ny_p - q4[1], nz_p - q4[2]);
            let l2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();
            // both operands float32 gives a float32 difference
            let [v1x, v1y, v1z] = if n_pts >= 9 {
                let p9 = pts[n_pts - 9];
                [0, 1, 2].map(|k| (p4[k] - p9[k]) as f64)
            } else {
                [q4[0] - x0, q4[1] - y0, q4[2] - z0]
            };
            let l1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
            if l1 > 1e-05 && l2 > 1e-05 && (v1x * v2x + v1y * v2y + v1z * v2z) / (l1 * l2) < COS55 {
                break;
            }
        }
        if n_pts >= 9 {
            let p9 = pts[n_pts - 9];
            let q9 = p64(p9);
            let (w2x, w2y, w2z) = (nx_p - q9[0], ny_p - q9[1], nz_p - q9[2]);
            let wl2 = (w2x * w2x + w2y * w2y + w2z * w2z).sqrt();
            let [w1x, w1y, w1z] = if n_pts >= 19 {
                let p19 = pts[n_pts - 19];
                [0, 1, 2].map(|k| (p9[k] - p19[k]) as f64)
            } else {
                [q9[0] - x0, q9[1] - y0, q9[2] - z0]
            };
            let wl1 = (w1x * w1x + w1y * w1y + w1z * w1z).sqrt();
            if wl1 > 1e-05 && wl2 > 1e-05 && (w1x * w2x + w1y * w2y + w1z * w2z) / (wl1 * wl2) < COS55 {
                break;
            }
        }
        let p_new = trilin_u8(f.pres, nx_p, ny_p, nz_p);
        let p_mid = trilin_u8(f.pres, 0.5 * (x + nx_p), 0.5 * (y + ny_p), 0.5 * (z + nz_p));
        if p_new < 76.5 || p_mid < 76.5 {
            if p_new < 52.5 || p_mid < 52.5 {
                break;
            }
            low_pres_run += 1;
            if low_pres_run > MAX_LOW {
                break;
            }
        } else {
            low_pres_run = 0;
        }
        if cov.covered_same_dir(nx_p, ny_p, nz_p, ntx, nty, ntz) {
            cov_run += 1;
            if cov_run >= 2 {
                break;
            }
        } else {
            cov_run = 0;
        }
        for p in &pts[..n_pts.saturating_sub(5)] {
            let (ddx, ddy, ddz) = (nx_p - p[0] as f64, ny_p - p[1] as f64, nz_p - p[2] as f64);
            if ddx * ddx + ddy * ddy + ddz * ddz < 9.0 {
                break 'steps;
            }
        }
        tx = 0.5 * ntx + 0.3 * dx + 0.2 * ux;
        ty = 0.5 * nty + 0.3 * dy + 0.2 * uy;
        tz = 0.5 * ntz + 0.3 * dz + 0.2 * uz;
        let tn = (tx * tx + ty * ty + tz * tz).sqrt();
        tx /= tn;
        ty /= tn;
        tz /= tn;
        (x, y, z) = (nx_p, ny_p, nz_p);
        pts[n_pts] = [x as f32, y as f32, z as f32];
        tans[n_pts] = [tx as f32, ty as f32, tz as f32];
        n_pts += 1;
    }
    while n_pts > 0 && cov_run > 0 {
        n_pts -= 1;
        cov_run -= 1;
    }
    while n_pts > 0 {
        let [px, py, pz] = p64(pts[n_pts - 1]);
        if trilin_u8(f.pres, px, py, pz) >= 76.5 {
            break;
        }
        n_pts -= 1;
    }
    n_pts
}

/// One iteration of `trace_seed_range` (core.py:882-1016) for seed `i`, with `min_raw_len = 7.5`;
/// accepted curves are marked into `cov` and appended to `out`
#[allow(clippy::too_many_arguments)]
pub fn trace_seed<P: Grid<u8>, C: Grid<u16>, N: Grid<u8>>(
    f: &Fields<P, C, N>,
    cov: &mut Coverage,
    s: &mut Scratch,
    seed: [i32; 3],
    primary: bool,
    out: &mut Vec<RawCurve>,
) -> Result<()> {
    let [sz, sy, sx] = seed.map(|v| v as i64);
    if !primary && cov.cnt.at(sz, sy, sx) > 0 {
        return Ok(());
    }
    let [ux, uy, uz] = f.dir_at(sz, sy, sx);
    let (sxf, syf, szf) = (sx as f64, sy as f64, sz as f64);
    if cov.covered_same_dir(sxf, syf, szf, ux, uy, uz) {
        return Ok(());
    }
    let [cx, cy, cz] = f.recenter(sx as f32 as f64, sy as f32 as f64, sz as f32 as f64, ux, uy, uz, 1.5, 4);
    if cov.covered_same_dir(cx, cy, cz, ux, uy, uz) {
        return Ok(());
    }
    if trilin_u8(f.pres, cx, cy, cz) < 76.5 {
        return Ok(());
    }
    let [ux, uy, uz] = f.dir_aligned(cx, cy, cz, ux, uy, uz);
    let nf = trace_half(f, cov, cx, cy, cz, ux, uy, uz, &mut s.pts_f, &mut s.tans_f);
    if nf == HALF {
        bail!("Full-scroll half-path buffer reached; refusing an artificial endpoint");
    }
    let (mut bux, mut buy, mut buz) = (-ux, -uy, -uz);
    if nf >= 3 {
        let k_dep = 3.min(nf - 1);
        let p = s.pts_f[k_dep];
        let (mut dfx, mut dfy, mut dfz) = (p[0] as f64 - cx, p[1] as f64 - cy, p[2] as f64 - cz);
        let dfn = (dfx * dfx + dfy * dfy + dfz * dfz).sqrt();
        if dfn > 1e-05 {
            dfx /= dfn;
            dfy /= dfn;
            dfz /= dfn;
            bux = -(0.55 * dfx + 0.45 * ux);
            buy = -(0.55 * dfy + 0.45 * uy);
            buz = -(0.55 * dfz + 0.45 * uz);
            let bn = (bux * bux + buy * buy + buz * buz).sqrt();
            bux /= bn;
            buy /= bn;
            buz /= bn;
        }
    }
    let nb = trace_half(f, cov, cx, cy, cz, bux, buy, buz, &mut s.pts_b, &mut s.tans_b);
    if nb == HALF {
        bail!("Full-scroll half-path buffer reached; refusing an artificial endpoint");
    }
    let ntot = nb + 1 + nf;
    if ntot < 8 {
        return Ok(());
    }
    let mut idx = 0;
    for j in (0..nb).rev() {
        s.pts_c[idx] = s.pts_b[j];
        s.tans_c[idx] = s.tans_b[j].map(|v| -v);
        idx += 1;
    }
    s.pts_c[idx] = [cx as f32, cy as f32, cz as f32];
    s.tans_c[idx] = [ux as f32, uy as f32, uz as f32];
    idx += 1;
    s.pts_c[idx..idx + nf].copy_from_slice(&s.pts_f[..nf]);
    s.tans_c[idx..idx + nf].copy_from_slice(&s.tans_f[..nf]);
    let req_len = if primary { 7.5 } else { py_max(7.5, 16.0f32 as f64) };
    for (a, b) in valid_curve_spans(&s.pts_c[..ntot], nb as i64)? {
        let n = b - a;
        if n < 8 {
            continue;
        }
        let mut arc_len = 0.0f32;
        for j in a..b - 1 {
            let (p, q) = (s.pts_c[j], s.pts_c[j + 1]);
            let (dx, dy, dz) = (q[0] - p[0], q[1] - p[1], q[2] - p[2]);
            arc_len += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        if (arc_len as f64) < req_len {
            continue;
        }
        cov.mark(f, &s.pts_c[a..b], &s.tans_c[a..b]);
        out.push(RawCurve { pts: s.pts_c[a..b].to_vec() });
    }
    Ok(())
}

#[inline(always)]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn norm2_f32(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

#[inline(always)]
fn dot_f32(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `np.dot` of two float32 3-vectors: OpenBLAS `sdot` rounds each product to float32 and sums them in
/// float64
#[inline(always)]
fn np_dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut acc = 0.0f64;
    for k in 0..3 {
        acc += (a[k] * b[k]) as f64;
    }
    acc as f32
}

/// `first_curve_split` (core.py:717), all float32
pub fn first_curve_split(pts: &[[f32; 3]]) -> i64 {
    let n = pts.len();
    if n < 2 {
        return 0;
    }
    for i in 0..n {
        for j in i + 7..n {
            if (norm2_f32(sub(pts[j], pts[i])) as f64) < 10.24 {
                return ((i + j) / 2) as i64;
            }
        }
    }
    let bend = |l: usize, c: usize, r: usize| {
        let (v1, v2) = (sub(pts[c], pts[l]), sub(pts[r], pts[c]));
        let (l1, l2) = (norm2_f32(v1).sqrt(), norm2_f32(v2).sqrt());
        (l1 as f64) > 1e-05 && (l2 as f64) > 1e-05 && ((dot_f32(v1, v2) / (l1 * l2)) as f64) < COS55
    };
    for i in 1..n.saturating_sub(1) {
        if bend(i - 1, i, i + 1) {
            return i as i64;
        }
    }
    for s in [5usize, 10] {
        if n > 2 * s {
            for i in s..n - s {
                if bend(i - s, i, i + s) {
                    return i as i64;
                }
            }
        }
    }
    let mut cum_s = vec![0.0f32; n];
    for i in 0..n - 1 {
        cum_s[i + 1] = cum_s[i] + norm2_f32(sub(pts[i + 1], pts[i])).sqrt();
    }
    const WIN: usize = 24;
    if n > WIN {
        for i in 0..n - WIN {
            let arc_w = cum_s[i + WIN] - cum_s[i];
            if arc_w as f64 > 1.0 {
                let ch_w = norm2_f32(sub(pts[i + WIN], pts[i])).sqrt();
                if (ch_w as f64) < 0.78 * arc_w as f64 {
                    return (i + WIN / 2) as i64;
                }
            }
        }
    }
    -1
}

/// `valid_curve_spans` (core.py:791): contiguous valid pieces in path order, as `[start, end)` pairs
pub fn valid_curve_spans(pts: &[[f32; 3]], seed_index: i64) -> Result<Vec<(usize, usize)>> {
    let mut pending = vec![(0usize, pts.len())];
    let mut spans = Vec::new();
    while let Some((a, b)) = pending.pop() {
        let n = b - a;
        if n < 2 {
            continue;
        }
        let piece = &pts[a..b];
        let mut cut = first_curve_split(piece);
        let local_seed = seed_index - a as i64;
        if cut < 0 && 0 < local_seed && local_seed < n as i64 - 1 {
            let ls = local_seed as usize;
            'outer: for j in 0..ls {
                for k in ls + 1..n {
                    if k - j < 6 {
                        continue;
                    }
                    let delta = sub(piece[j], piece[k]);
                    if (np_dot3(delta, delta) as f64) < 9.0 {
                        cut = ((j + k) / 2) as i64;
                        break 'outer;
                    }
                }
            }
        }
        if cut < 0 {
            spans.push((a, b));
            continue;
        }
        if !(0 < cut && cut < n as i64 - 1) {
            bail!("A rejected nontrivial curve must have an interior split");
        }
        let split = a + cut as usize;
        pending.push((split, b));
        pending.push((a, split + 1));
    }
    Ok(spans)
}

/// `fit_and_resample_splines` (core.py:590) for one curve with `target_step = 0.8`, `smooth_iters = 10`,
/// `max_dev = 0.3` and `min_final_len = 0`: the smoothed points and the resampled count (0 when the
/// curve has fewer than 4 points)
pub fn fit_curve<P: Grid<u8>>(pts: &[[f32; 3]], pres: &P) -> (Vec<[f32; 3]>, usize) {
    const MAX_DEV: f64 = 0.3;
    let m = pts.len();
    if m < 4 {
        return (pts.to_vec(), 0);
    }
    let mut buf = pts.to_vec();
    for _ in 0..10 {
        for i in 1..m - 1 {
            let mut p = [0.0f64; 3];
            if i == 1 || i == m - 2 {
                for k in 0..3 {
                    p[k] = 0.25 * buf[i - 1][k] as f64 + 0.5 * buf[i][k] as f64 + 0.25 * buf[i + 1][k] as f64;
                }
            } else {
                for k in 0..3 {
                    let lap = 0.5 * (buf[i - 1][k] + buf[i + 1][k]) as f64;
                    let bih = (-buf[i - 2][k] as f64 + 4.0 * buf[i - 1][k] as f64 + 4.0 * buf[i + 1][k] as f64
                        - buf[i + 2][k] as f64)
                        / 6.0;
                    p[k] = 0.5 * buf[i][k] as f64 + 0.3 * lap + 0.2 * bih;
                }
            }
            let o = pts[i];
            let [ox, oy, oz] = o.map(|v| v as f64);
            let (dx, dy, dz) = (p[0] - ox, p[1] - oy, p[2] - oz);
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > MAX_DEV {
                let s = MAX_DEV / d;
                p = [ox + dx * s, oy + dy * s, oz + dz * s];
            }
            let p_orig = trilin_u8(pres, ox, oy, oz);
            let p_new = trilin_u8(pres, p[0], p[1], p[2]);
            if p_orig >= 76.5 && p_new < py_max(77.0, p_orig - 6.0) {
                p = [0, 1, 2].map(|k| 0.5 * (o[k] + buf[i][k]) as f64);
            }
            buf[i] = p.map(|v| v as f32);
        }
    }
    let mut arc = 0.0f32;
    for i in 1..m {
        arc += norm2_f32(sub(buf[i], buf[i - 1])).sqrt();
    }
    let n_out = ((arc as f64 / 0.8 + 0.5) as i64 + 1).max(2) as usize;
    (buf, n_out)
}

/// `evaluate_catmull_rom_splines` (core.py:655) for one curve; `shape` is the tracing shape z, y, x
pub fn evaluate_curve(smoothed: &[[f32; 3]], n_out: usize, shape: [usize; 3]) -> Vec<[f32; 3]> {
    let m = smoothed.len();
    let mut cum_s = vec![0.0f32; m];
    for i in 0..m - 1 {
        cum_s[i + 1] = cum_s[i] + norm2_f32(sub(smoothed[i + 1], smoothed[i])).sqrt();
    }
    let tot_s = cum_s[m - 1];
    let mut out = vec![[0.0f32; 3]; n_out];
    out[0] = smoothed[0];
    out[n_out - 1] = smoothed[m - 1];
    let upper: [f64; 3] = [0, 1, 2].map(|d| (shape[2 - d] as i64 - 1) as f32 as f64);
    let mut seg_k = 0usize;
    for (j, slot) in out.iter_mut().enumerate().take(n_out - 1).skip(1) {
        let target_s = j as f32 / (n_out - 1) as f32 * tot_s;
        while seg_k + 2 < m && cum_s[seg_k + 1] < target_s {
            seg_k += 1;
        }
        let ds = cum_s[seg_k + 1] - cum_s[seg_k];
        let floor = 1e-06f32;
        let u = (target_s - cum_s[seg_k]) / (if floor > ds { floor } else { ds });
        let u2 = u * u;
        let u3 = u2 * u;
        let (u, u2, u3) = (u as f64, u2 as f64, u3 as f64);
        let i0 = seg_k.saturating_sub(1);
        let i1 = seg_k;
        let i2 = seg_k + 1;
        let i3 = (m - 1).min(seg_k + 2);
        let w0 = -0.5 * u3 + u2 - 0.5 * u;
        let w1 = 1.5 * u3 - 2.5 * u2 + 1.0;
        let w2 = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
        let w3 = 0.5 * u3 - 0.5 * u2;
        for d in 0..3 {
            let val = w0 * smoothed[i0][d] as f64
                + w1 * smoothed[i1][d] as f64
                + w2 * smoothed[i2][d] as f64
                + w3 * smoothed[i3][d] as f64;
            slot[d] = py_min(py_max(val, 0.0), upper[d]) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_matches_reference_formula() {
        // nx = ny = 128 is the straight-up direction; 255, 255 has r2 > 1 and is renormalized flat
        assert_eq!(decode_dir(128, 128), [0.0, 0.0, 1.0]);
        let [x, y, z] = decode_dir(255, 255);
        assert_eq!(z, 0.0);
        assert!(((x * x + y * y).sqrt() - 1.0).abs() < 1e-12);
        assert_eq!(encode_dir(0.0, 0.0, 1.0), (128, 128));
        assert_eq!(encode_dir(1.0, -1.0, -0.5), (1, 255));
    }

    #[test]
    fn spans_split_a_hairpin_at_its_vertex() {
        // a straight run that doubles back: the split keeps the vertex on both sides
        let mut pts = Vec::new();
        for i in 0..20 {
            pts.push([i as f32 * 0.75, 0.0, 0.0]);
        }
        for i in 1..20 {
            pts.push([(19 - i) as f32 * 0.75, 4.0, 0.0]);
        }
        let spans = valid_curve_spans(&pts, 5).unwrap();
        assert!(spans.len() >= 2);
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans.last().unwrap().1, pts.len());
        for w in spans.windows(2) {
            assert_eq!(w[0].1, w[1].0 + 1);
        }
    }
}
