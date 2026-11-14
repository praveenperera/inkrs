//! The direction field: `prepare.tensor_axes` (prepare.py:62) per voxel
//!
//! A voxel's structure tensor reaches 11 voxels (Gaussian σ 0.9 radius 4, gradient 1, Gaussian σ 1.5
//! radius 6), and the reference's 256³ tiles carry a 16-voxel halo, so no output voxel ever sees a
//! tile edge except at the volume edge, where tile and volume edges coincide. Any tiling with an
//! 11-voxel halo therefore gives the reference's values. The filters reproduce SciPy's `correlate1d`
//! (symmetric kernel, float64 sums, float32 after each axis, reflect at the edge) and `np.gradient`;
//! eigenvectors come from the `dsyevd` of numpy's own OpenBLAS on a float64 copy, cast back to float32, which
//! is what `np.linalg.eigh` does with float32 input (`_commonType` always computes in double)

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use color_eyre::eyre::{Result, bail, eyre};
use libloading::Library;
use rayon::prelude::*;

use crate::vol::{Shape, Vol};

/// Halo the tensor needs: 4 + 1 + 6
const HALO: usize = 11;
/// Tile edge for the parallel sweep; a multiple of the brick edge
const CORE: usize = 64;

/// SciPy's Gaussian weights for σ 0.9 (radius 4) and σ 1.5 (radius 6), outermost first, centre last;
/// computed by `scipy.ndimage._gaussian_kernel1d` and stored as their float64 bit patterns
const W09: [u64; 5] =
    [0x3ef7dff10d1ee066, 0x3f5c138a6b624c17, 0x3fa336a4e8e9e375, 0x3fce9aeb020a6f44, 0x3fdc5e85af5bea50];
const W15: [u64; 7] = [
    0x3f17638649a0fcd1,
    0x3f50d891ab53680a,
    0x3f7f1e741f80c9b8,
    0x3fa26dd9e7a284e1,
    0x3fbbfdc88289323c,
    0x3fcb42851d1c8a96,
    0x3fd1058ef29a59d7,
];

type Dsyevd = unsafe extern "C" fn(
    jobz: *const u8,
    uplo: *const u8,
    n: *const i64,
    a: *mut f64,
    lda: *const i64,
    w: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    iwork: *mut i64,
    liwork: *const i64,
    info: *mut i64,
    jobz_len: usize,
    uplo_len: usize,
) -> ();

/// `dsyevd` from numpy's bundled OpenBLAS (ILP64, `scipy_` prefix)
pub struct Lapack {
    _lib: Library,
    dsyevd: Dsyevd,
    lwork: i64,
    liwork: i64,
}

static LAPACK: OnceLock<Lapack> = OnceLock::new();

impl Lapack {
    /// Load the library at `path` once for the process
    pub fn init(path: &Path) -> Result<&'static Lapack> {
        if let Some(l) = LAPACK.get() {
            return Ok(l);
        }
        // SAFETY: loading a shared library runs its initializers; numpy's OpenBLAS is built to be loaded
        // into arbitrary processes
        let lib = unsafe { Library::new(path) }.map_err(|e| eyre!("{}: {e}", path.display()))?;
        // SAFETY: the symbol has the reference LAPACK signature with 64-bit integers
        let dsyevd: Dsyevd = unsafe { *lib.get::<Dsyevd>(b"scipy_dsyevd_64_\0")? };
        let mut lapack = Lapack { _lib: lib, dsyevd, lwork: -1, liwork: -1 };
        // workspace query, as numpy's init_evd does
        let mut a = [0.0f64; 9];
        let mut w = [0.0f64; 3];
        let mut work = [0.0f64; 1];
        let mut iwork = [0i64; 1];
        let info = lapack.call(&mut a, &mut w, &mut work, &mut iwork);
        if info != 0 {
            bail!("dsyevd workspace query failed ({info})");
        }
        lapack.lwork = work[0] as i64;
        lapack.liwork = iwork[0];
        Ok(LAPACK.get_or_init(|| lapack))
    }

    fn call(&self, a: &mut [f64; 9], w: &mut [f64; 3], work: &mut [f64], iwork: &mut [i64]) -> i64 {
        let (n, lda) = (3i64, 3i64);
        let (lwork, liwork) = (self.lwork, self.liwork);
        let mut info = 0i64;
        // SAFETY: buffers are at least as large as the workspace query asked for
        unsafe {
            (self.dsyevd)(
                b"V".as_ptr(),
                b"L".as_ptr(),
                &n,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &liwork,
                &mut info,
                1,
                1,
            );
        }
        info
    }

    /// The eigenvector of the smallest eigenvalue of a symmetric 3×3 matrix (row-major), as
    /// `np.linalg.eigh(m)[1][:, 0]`
    fn smallest(&self, m: [f32; 9], work: &mut Vec<f64>, iwork: &mut Vec<i64>) -> Result<[f32; 3]> {
        work.resize(self.lwork.max(1) as usize, 0.0);
        iwork.resize(self.liwork.max(1) as usize, 0);
        // column-major A(k, i) = m[k][i]; the matrix is symmetric, so this is m itself
        let mut a = [0.0f64; 9];
        for i in 0..3 {
            for k in 0..3 {
                a[k + 3 * i] = m[3 * k + i] as f64;
            }
        }
        let mut w = [0.0f64; 3];
        let info = self.call(&mut a, &mut w, work, iwork);
        if info != 0 {
            bail!("Eigenvalues did not converge");
        }
        // vt.astype(float32)
        Ok([a[0] as f32, a[1] as f32, a[2] as f32])
    }
}

/// Find numpy's OpenBLAS: `INKRS_AFV_LAPACK`, else the venv this executable was started from
/// (`VENV/bin/avf-spline-generator`), else `$VIRTUAL_ENV`
///
/// In each venv, numpy's bundled copy (`numpy.libs`, Linux wheels) comes first, then the `scipy-openblas64`
/// package's, which is the same OpenBLAS build as a standalone wheel. macOS numpy wheels link Accelerate instead
/// and bundle no OpenBLAS, so there the wheel of this tool depends on `scipy-openblas64`
pub fn find_lapack() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("INKRS_AFV_LAPACK") {
        return Ok(PathBuf::from(p));
    }
    let mut roots = Vec::new();
    if let Some(arg0) = std::env::args_os().next() {
        let arg0 = PathBuf::from(arg0);
        let arg0 = if arg0.components().count() > 1 { Some(arg0) } else { which(&arg0) };
        if let Some(bin) = arg0.as_deref().and_then(Path::parent) {
            roots.push(bin.join(".."));
        }
    }
    if let Some(v) = std::env::var_os("VIRTUAL_ENV") {
        roots.push(PathBuf::from(v));
    }
    for root in roots {
        let Ok(libs) = std::fs::read_dir(root.join("lib")) else { continue };
        for py in libs.flatten() {
            let site = py.path().join("site-packages");
            for dir in [site.join("numpy.libs"), site.join("scipy_openblas64/lib")] {
                if let Some(lib) = openblas_in(&dir) {
                    return Ok(lib);
                }
            }
        }
    }
    bail!(
        "cannot find numpy's OpenBLAS (numpy.libs/libscipy_openblas64_*.so, or the scipy-openblas64 package); \
         install this executable into the venv's bin directory or set INKRS_AFV_LAPACK"
    )
}

/// The `libscipy_openblas64_*` shared library in `dir`, if there is one
fn openblas_in(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries.flatten().map(|e| e.path()).find(|path| {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        name.starts_with("libscipy_openblas64_") && (name.ends_with(".so") || name.ends_with(".dylib"))
    })
}

fn which(name: &Path) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(name)).find(|p| p.is_file())
}

fn weights<const N: usize>(bits: [u64; N]) -> [f64; N] {
    bits.map(f64::from_bits)
}

/// A dense f32 block
struct Block {
    dims: Shape,
    data: Vec<f32>,
}

impl Block {
    fn stride(&self, axis: usize) -> usize {
        match axis {
            0 => self.dims[1] * self.dims[2],
            1 => self.dims[2],
            _ => 1,
        }
    }
}

/// One axis of `scipy.ndimage.correlate1d` with a symmetric kernel `w` (outermost first, centre last)
/// and mode "reflect", in place
fn correlate_axis(b: &mut Block, axis: usize, w: &[f64], line: &mut Vec<f64>) {
    let r = w.len() - 1;
    let n = b.dims[axis];
    let stride = b.stride(axis);
    let centre = w[r];
    line.resize(n + 2 * r, 0.0);
    let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
    let (d0, d1) = (b.dims[others[0]], b.dims[others[1]]);
    let (s0, s1) = (b.stride(others[0]), b.stride(others[1]));
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            let base = i0 * s0 + i1 * s1;
            for k in 0..n {
                line[r + k] = b.data[base + k * stride] as f64;
            }
            for k in 0..r {
                line[r - 1 - k] = line[r + reflect(-1 - k as isize, n)];
                line[r + n + k] = line[r + reflect((n + k) as isize, n)];
            }
            for k in 0..n {
                let c = r + k;
                let mut acc = line[c] * centre;
                for (j, &wj) in w[..r].iter().enumerate() {
                    let off = r - j;
                    acc += (line[c - off] + line[c + off]) * wj;
                }
                b.data[base + k * stride] = acc as f32;
            }
        }
    }
}

/// The sample index scipy's "reflect" mode reads for position `i` of a line of `n` samples: half-sample
/// symmetric extension, repeated with period `2n` (`NI_EXTEND_REFLECT`), so a kernel wider than the line
/// reflects more than once
fn reflect(i: isize, n: usize) -> usize {
    let m = i.rem_euclid(2 * n as isize) as usize;
    if m < n { m } else { 2 * n - 1 - m }
}

/// `gaussian_filter(block, sigma)` on float32: axes 0, 1, 2, each rounded to float32
fn gaussian(b: &mut Block, w: &[f64], line: &mut Vec<f64>) {
    for axis in 0..3 {
        correlate_axis(b, axis, w, line);
    }
}

/// `np.gradient` of a float32 block along one axis (unit spacing, first-order edges)
fn gradient(b: &Block, axis: usize) -> Block {
    let n = b.dims[axis];
    let s = b.stride(axis);
    let mut out = vec![0.0f32; b.data.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let k = (i / s) % n;
        let f = &b.data;
        *o = if k == 0 {
            (f[i + s] - f[i]) / 1.0
        } else if k == n - 1 {
            (f[i] - f[i - s]) / 1.0
        } else {
            (f[i + s] - f[i - s]) / 2.0
        };
    }
    Block { dims: b.dims, data: out }
}

/// One tile's core: origin, extent, nx and ny
type TileAxes = (Shape, Shape, Vec<u8>, Vec<u8>);

/// Compute nx, ny for the whole tracing volume from the two pooled channels
pub fn direction_field(tracing: &[Vol<u8>; 2], lapack: &Lapack) -> Result<(Vol<u8>, Vol<u8>)> {
    let shape = tracing[0].shape();
    let grid = shape.map(|n| n.div_ceil(CORE));
    let tiles: Vec<Shape> =
        (0..grid[0]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[2]).map(move |x| [z, y, x]))).collect();
    let results: Vec<Result<Option<TileAxes>>> = tiles.par_iter().map(|&t| tile_axes(tracing, t, lapack)).collect();
    let mut nx = Vol::new(shape, 128u8);
    let mut ny = Vol::new(shape, 128u8);
    for r in results {
        let Some((lo, dims, xx, yy)) = r? else { continue };
        nx.write_block(lo, dims, &xx);
        ny.write_block(lo, dims, &yy);
    }
    Ok((nx, ny))
}

fn tile_axes(tracing: &[Vol<u8>; 2], t: Shape, lapack: &Lapack) -> Result<Option<TileAxes>> {
    let shape = tracing[0].shape();
    let start = t.map(|i| i * CORE);
    let end = [0, 1, 2].map(|a| (start[a] + CORE).min(shape[a]));
    let lo = [0, 1, 2].map(|a| start[a].saturating_sub(HALO));
    let hi = [0, 1, 2].map(|a| (end[a] + HALO).min(shape[a]));
    let core_dims = [0, 1, 2].map(|a| end[a] - start[a]);
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
    // which family each core voxel belongs to, or none: presence = c0 + c1 >= 40, dominant = c1 > c0
    let mut ids: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
    for z in 0..core_dims[0] {
        for y in 0..core_dims[1] {
            for x in 0..core_dims[2] {
                let (gz, gy, gx) = (start[0] + z, start[1] + y, start[2] + x);
                let c0 = tracing[0].get(gz, gy, gx) as u16;
                let c1 = tracing[1].get(gz, gy, gx) as u16;
                if c0 + c1 < 40 {
                    continue;
                }
                let family = (c1 > c0) as usize;
                let local = ((z * core_dims[1]) + y) * core_dims[2] + x;
                let block = (((gz - lo[0]) * dims[1]) + gy - lo[1]) * dims[2] + gx - lo[2];
                ids[family].push((local, block));
            }
        }
    }
    if ids[0].is_empty() && ids[1].is_empty() {
        return Ok(None);
    }
    let n = core_dims.iter().product();
    let mut xx = vec![128u8; n];
    let mut yy = vec![128u8; n];
    let (w09, w15) = (weights(W09), weights(W15));
    let mut line = Vec::new();
    let (mut work, mut iwork) = (Vec::new(), Vec::new());
    for (family, ids) in ids.iter().enumerate() {
        if ids.is_empty() {
            continue;
        }
        let raw = tracing[family].read_block(lo, dims);
        let mut smooth = Block { dims, data: raw.iter().map(|&v| v as f32 / 255.0f32).collect() };
        gaussian(&mut smooth, &w09, &mut line);
        let grad = [0, 1, 2].map(|a| gradient(&smooth, a));
        drop(smooth);
        let mut tensor = vec![[0.0f32; 9]; ids.len()];
        for u in 0..3 {
            for v in u..3 {
                let data = grad[u].data.iter().zip(&grad[v].data).map(|(a, b)| a * b).collect();
                let mut prod = Block { dims, data };
                gaussian(&mut prod, &w15, &mut line);
                for (m, &(_, at)) in tensor.iter_mut().zip(ids) {
                    m[3 * u + v] = prod.data[at];
                    m[3 * v + u] = prod.data[at];
                }
            }
        }
        for (m, &(local, _)) in tensor.iter().zip(ids) {
            let v = lapack.smallest(*m, &mut work, &mut iwork)?;
            // xyz = v[::-1], flipped so z >= 0
            let mut xyz = [v[2], v[1], v[0]];
            if xyz[2] < 0.0 {
                xyz = xyz.map(|c| -c);
            }
            let q = |c: f32| (c * 127.0f32 + 128.0f32).round_ties_even().clamp(1.0, 255.0) as u8;
            xx[local] = q(xyz[0]);
            yy[local] = q(xyz[1]);
        }
    }
    Ok(Some((start, core_dims, xx, yy)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_weights_sum_to_one() {
        for w in [weights(W09).to_vec(), weights(W15).to_vec()] {
            let r = w.len() - 1;
            let total = w[r] + 2.0 * w[..r].iter().sum::<f64>();
            assert!((total - 1.0).abs() < 1e-12, "{total}");
        }
    }

    /// `scipy.ndimage.gaussian_filter1d(x, sigma, mode="reflect")` on float32 lines shorter than the
    /// kernel radius (4 voxels is the shortest tracing axis the reference accepts), as f32 bit patterns
    const SCIPY_SHORT_LINES: [(f64, &[u32], &[u32]); 6] = [
        (0.9, &[0x00000000, 0x3f666666, 0x3e800000, 0x3f100000], &[0x3e850000, 0x3ef6f5cd, 0x3ef6d393, 0x3efa036c]),
        (1.5, &[0x00000000, 0x3f666666, 0x3e800000, 0x3f100000], &[0x3eb65875, 0x3ed2ffe2, 0x3eecc526, 0x3ef6af4f]),
        (
            0.9,
            &[0x00000000, 0x3f666666, 0x3e23d70a, 0x3eb851ec, 0x3f23d70a],
            &[0x3e830340, 0x3ee81f81, 0x3ecb6f75, 0x3ed18bda, 0x3f0b4d20],
        ),
        (
            1.5,
            &[0x00000000, 0x3f666666, 0x3e23d70a, 0x3eb851ec, 0x3f23d70a],
            &[0x3eabab6b, 0x3ec01090, 0x3ed3ed9e, 0x3ee6adda, 0x3ef860de],
        ),
        (
            0.9,
            &[
                0x00000000, 0x3f666666, 0x3d4a4588, 0x3de38e39, 0x3e4a4588, 0x3e9e0652, 0x3ee38e39, 0x3f1add3c,
                0x3f4a4588,
            ],
            &[
                0x3e809059, 0x3ed567e6, 0x3e8b11f4, 0x3e1e208e, 0x3e5611b1, 0x3ea3270c, 0x3ee87be8, 0x3f1b2028,
                0x3f3977a5,
            ],
        ),
        (
            1.5,
            &[
                0x00000000, 0x3f666666, 0x3d4a4588, 0x3de38e39, 0x3e4a4588, 0x3e9e0652, 0x3ee38e39, 0x3f1add3c,
                0x3f4a4588,
            ],
            &[
                0x3e9bbe5e, 0x3e9c1c0e, 0x3e8b5652, 0x3e726099, 0x3e839b53, 0x3eae8017, 0x3eeb4e31, 0x3f13d8c4,
                0x3f273cdb,
            ],
        ),
    ];

    #[test]
    fn correlate_matches_scipy_on_short_lines() {
        let mut line = Vec::new();
        for (sigma, input, expected) in SCIPY_SHORT_LINES {
            let w = if sigma == 0.9 { weights(W09).to_vec() } else { weights(W15).to_vec() };
            let data = input.iter().map(|&b| f32::from_bits(b)).collect();
            let mut b = Block { dims: [1, 1, input.len()], data };
            correlate_axis(&mut b, 2, &w, &mut line);
            let got: Vec<u32> = b.data.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, expected, "sigma {sigma}, n {}", input.len());
        }
    }
}
