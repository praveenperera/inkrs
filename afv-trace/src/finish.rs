//! `tracing.finish`: split raw curves where predictions disagree, smooth, resample and re-validate
//!
//! `finish` treats every raw curve on its own and reads only immutable fields (pooled predictions,
//! presence, native predictions), so curves are finished in parallel once tracing is done. Sampling
//! reproduces `scipy.ndimage.map_coordinates(order=1, mode="nearest")` (`compressed.sample_tracing`)
//! and `storage.NativeSampler.sample` exactly

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use color_eyre::eyre::Result;

use crate::input::Predictions;
use crate::kernels::{evaluate_curve, fit_curve};
use crate::vol::{Shape, Vol};

/// One decoded native chunk, `(2, dims)` uint8 in V, H order
struct Chunk {
    lo: Shape,
    dims: Shape,
    data: Vec<u8>,
}

/// A bounded cache of decoded native chunks shared by the finishing threads, least recently used
/// first out
pub struct NativeCache<'a> {
    pred: &'a Predictions,
    edge: Shape,
    budget: usize,
    lru: Mutex<Lru>,
}

#[derive(Default)]
struct Lru {
    map: HashMap<Shape, (Arc<Chunk>, u64)>,
    bytes: usize,
    clock: u64,
    decoded: u64,
}

impl<'a> NativeCache<'a> {
    /// `budget_bytes` bounds the decoded chunks held at once; one chunk is always kept
    pub fn new(pred: &'a Predictions, budget_bytes: usize) -> Self {
        // whole Zarr chunks are the unit of decoding; NPY input uses the reference's 64³ blocks
        let edge = pred.chunk_shape().unwrap_or([64, 64, 64]);
        Self { pred, edge, budget: budget_bytes, lru: Mutex::default() }
    }

    /// The chunk grid key of a native voxel
    pub fn key(&self, z: usize, y: usize, x: usize) -> Shape {
        [z / self.edge[0], y / self.edge[1], x / self.edge[2]]
    }

    /// Chunks decoded so far, counting re-decodes after eviction
    pub fn decoded(&self) -> u64 {
        self.lru.lock().expect("cache lock").decoded
    }

    fn chunk(&self, key: Shape) -> Result<Arc<Chunk>> {
        {
            let mut s = self.lru.lock().expect("cache lock");
            s.clock += 1;
            let now = s.clock;
            if let Some((c, used)) = s.map.get_mut(&key) {
                *used = now;
                return Ok(c.clone());
            }
        }
        // decode outside the lock; two threads may decode the same chunk once each, which only costs time
        let (lo, dims, data) = self.pred.read_chunk_u8(key, self.edge)?;
        let chunk = Arc::new(Chunk { lo, dims, data });
        let mut s = self.lru.lock().expect("cache lock");
        s.clock += 1;
        s.decoded += 1;
        let now = s.clock;
        if let Some((old, _)) = s.map.insert(key, (chunk.clone(), now)) {
            s.bytes -= old.data.len();
        }
        s.bytes += chunk.data.len();
        while s.bytes > self.budget && s.map.len() > 1 {
            let oldest = s.map.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| *k).expect("nonempty");
            let (gone, _) = s.map.remove(&oldest).expect("present");
            s.bytes -= gone.data.len();
        }
        Ok(chunk)
    }
}

/// Recently used chunks each finishing thread keeps without touching the shared lock; a point's eight
/// corners span at most eight chunks
const RECENT: usize = 8;

/// Per-thread view of the native cache
pub struct NativeSampler<'c, 'a> {
    cache: &'c NativeCache<'a>,
    recent: Vec<(Shape, Arc<Chunk>)>,
}

impl<'c, 'a> NativeSampler<'c, 'a> {
    pub fn new(cache: &'c NativeCache<'a>) -> Self {
        Self { cache, recent: Vec::with_capacity(RECENT) }
    }

    fn value(&mut self, z: usize, y: usize, x: usize) -> Result<[u8; 2]> {
        let key = self.cache.key(z, y, x);
        let i = match self.recent.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                let c = self.cache.chunk(key)?;
                if self.recent.len() == RECENT {
                    self.recent.remove(0);
                }
                self.recent.push((key, c));
                self.recent.len() - 1
            }
        };
        let chunk = &self.recent[i].1;
        let (lz, ly, lx) = (z - chunk.lo[0], y - chunk.lo[1], x - chunk.lo[2]);
        let plane = chunk.dims.iter().product::<usize>();
        let at = (lz * chunk.dims[1] + ly) * chunk.dims[2] + lx;
        Ok([chunk.data[at], chunk.data[plane + at]])
    }

    /// `NativeSampler.sample` of one point given as float32 x, y, z in native voxels
    fn sample(&mut self, p: [f32; 3]) -> Result<[f32; 2]> {
        let shape = self.cache.pred.shape;
        // np.clip(points[:, ::-1], 0, shape - 1) is float64
        let q = [p[2], p[1], p[0]];
        let q: [f64; 3] = [0, 1, 2].map(|a| {
            let v = q[a].max(0.0);
            let hi = (shape[a] - 1) as f64;
            if hi < v as f64 { hi } else { v as f64 }
        });
        let lo = q.map(|v| v.floor() as i64);
        let frac = [0, 1, 2].map(|a| q[a] - lo[a] as f64);
        let mut values = [0.0f32; 2];
        for corner in 0..8 {
            let c = [(corner >> 2) & 1, (corner >> 1) & 1, corner & 1];
            let ij = [0, 1, 2].map(|a| ((lo[a] + c[a] as i64) as usize).min(shape[a] - 1));
            let w = [0, 1, 2].map(|a| if c[a] == 1 { frac[a] } else { 1.0 - frac[a] });
            let weight = w[0] * w[1] * w[2];
            let v = self.value(ij[0], ij[1], ij[2])?;
            for ch in 0..2 {
                values[ch] = (values[ch] as f64 + v[ch] as f64 * weight) as f32;
            }
        }
        Ok(values)
    }
}

/// `map_coordinates(field, q, order=1, mode="nearest", output="f4")` for both channels at a point
/// given as float32 x, y, z: SciPy keeps the unclamped coordinate for the weights and clamps only the
/// indices, and its second weight is `1 - (1 - t)`
fn sample_tracing(tracing: &[Vol<u8>; 2], p: [f32; 3]) -> [f32; 2] {
    let shape = tracing[0].shape();
    let cc = [p[2] as f64, p[1] as f64, p[0] as f64];
    let mut idx = [[0usize; 2]; 3];
    let mut wts = [[0.0f64; 2]; 3];
    for a in 0..3 {
        let start = cc[a].floor();
        let t = cc[a] - start;
        let w0 = 1.0 - t;
        wts[a] = [w0, 1.0 - w0];
        let s = start as i64;
        let len = shape[a] as i64;
        idx[a] = [s, s + 1].map(|i| i.clamp(0, len - 1) as usize);
    }
    let mut out = [0.0f32; 2];
    for (ch, field) in tracing.iter().enumerate() {
        let mut acc = 0.0f64;
        for kz in 0..2 {
            for ky in 0..2 {
                for kx in 0..2 {
                    let mut coeff = field.get(idx[0][kz], idx[1][ky], idx[2][kx]) as f64;
                    coeff *= wts[0][kz];
                    coeff *= wts[1][ky];
                    coeff *= wts[2][kx];
                    acc += coeff;
                }
            }
        }
        out[ch] = acc as f32;
    }
    out
}

/// What finishing reads: the pooled predictions, the family's presence and the native predictions
pub struct FinishCtx<'a> {
    pub tracing: &'a [Vol<u8>; 2],
    pub pres: &'a Vol<u8>,
    /// 1 = V, 2 = H
    pub family: usize,
}

impl FinishCtx<'_> {
    /// `valid_segments` (tracing.py:18): whether every sample along each segment favours the family in
    /// both the pooled and the native predictions; stops at the first failure when `all` is set
    fn valid_segments(&self, pts: &[[f32; 3]], native: &mut NativeSampler, all: bool) -> Result<Vec<bool>> {
        let (own, other) = (self.family - 1, 2 - self.family);
        let mut valid = vec![true; pts.len().saturating_sub(1)];
        for (i, seg) in pts.windows(2).enumerate() {
            let d = [seg[1][0] - seg[0][0], seg[1][1] - seg[0][1], seg[1][2] - seg[0][2]];
            // np.linalg.norm of float32 rows, then / 0.25 and ceil, all float32
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let n = ((len / 0.25f32).ceil() as i64).max(1);
            let step = 1.0 / n as f64;
            for j in 0..=n {
                // np.linspace(0, 1, n + 1, dtype="f4"): j * step in float64, the last exactly 1
                let t = if j == n { 1.0f32 } else { (j as f64 * step) as f32 };
                let q = [0, 1, 2].map(|k| seg[0][k] + d[k] * t);
                let tv = sample_tracing(self.tracing, q);
                let mut good = tv[own] + 1e-4f32 >= tv[other];
                if good {
                    let nv = native.sample(q.map(|v| v * 2.0f32 + 0.5f32))?;
                    good = nv[own] + 1e-4f32 >= nv[other];
                }
                if !good {
                    valid[i] = false;
                    break;
                }
            }
            if all && !valid[i] {
                break;
            }
        }
        Ok(valid)
    }

    /// `finish` for one raw curve: the finished pieces in order
    pub fn finish(&self, raw: &[[f32; 3]], native: &mut NativeSampler) -> Result<Vec<Vec<[f32; 3]>>> {
        let valid = self.valid_segments(raw, native, false)?;
        let mut pieces = Vec::new();
        let mut i = 0;
        while i < valid.len() {
            if !valid[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < valid.len() && valid[i] {
                i += 1;
            }
            pieces.push(raw[start..=i].to_vec());
        }
        let shape = self.pres.shape();
        for piece in pieces.iter_mut() {
            let (smoothed, count) = fit_curve(piece, self.pres);
            if count < 2 {
                continue;
            }
            let fitted = evaluate_curve(&smoothed, count, shape);
            if self.valid_segments(&fitted, native, true)?.iter().all(|&v| v) {
                *piece = fitted;
            }
        }
        Ok(pieces)
    }
}
