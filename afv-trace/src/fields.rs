//! Per-family fields and the seed catalogue (`prepare.prepare_family`, `prepare.prepare_seeds`)

use rayon::prelude::*;

use crate::kernels::{Dense, stage1};
use crate::pyfmt::argsort_u16;
use crate::vol::{Shape, Vol};

/// Tile edge and halo of the reference's centrality pass (prepare.py:148-150); the kernel clamps samples
/// to its tile, so these are part of the output, not a tuning choice
const TILE: usize = 128;
const TILE_HALO: usize = 8;

/// One family's fields: presence (0 where the other family dominates), centrality and seed priority
pub struct FamilyFields {
    pub pres: Vol<u8>,
    pub cent: Vol<u16>,
    pub seeds: Vol<u8>,
}

/// Presence of `family` (1 = V, 2 = H): `min(own + other, 255)`, zero where `own < other`
pub fn presence(tracing: &[Vol<u8>; 2], family: usize) -> Vol<u8> {
    let (own, other) = (&tracing[family - 1], &tracing[2 - family]);
    let mut pres = Vol::new(own.shape(), 0u8);
    pres.fill_bricks(|lo, dims, out| {
        let mut any = false;
        let mut i = 0;
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                for x in 0..dims[2] {
                    let (a, b) = (own.get(lo[0] + z, lo[1] + y, lo[2] + x), other.get(lo[0] + z, lo[1] + y, lo[2] + x));
                    let p = if a < b { 0 } else { (a as u16 + b as u16).min(255) as u8 };
                    out[i] = p;
                    any |= p != 0;
                    i += 1;
                }
            }
        }
        any
    });
    pres
}

/// The reference's `tiles(shape, size)` order: C order over tile origins
fn tiles(shape: Shape, size: usize) -> Vec<(Shape, Shape)> {
    let grid = shape.map(|n| n.div_ceil(size));
    let mut out = Vec::with_capacity(grid.iter().product());
    for z in 0..grid[0] {
        for y in 0..grid[1] {
            for x in 0..grid[2] {
                let lo = [z * size, y * size, x * size];
                let hi = [0, 1, 2].map(|a| (lo[a] + size).min(shape[a]));
                out.push((lo, hi));
            }
        }
    }
    out
}

/// One tile's core: origin, extent, centrality and seeds
type TileFields = (Shape, Shape, Vec<u16>, Vec<u8>);

/// Centrality and seeds over the reference's 128³ tiles with an 8-voxel halo, tiles in parallel
pub fn centrality(pres: Vol<u8>, nx: &Vol<u8>, ny: &Vol<u8>) -> FamilyFields {
    let shape = pres.shape();
    let jobs = tiles(shape, TILE);
    let results: Vec<Option<TileFields>> = jobs
        .par_iter()
        .map(|&(start, end)| {
            let lo = start.map(|v| v.saturating_sub(TILE_HALO));
            let hi = [0, 1, 2].map(|a| (end[a] + TILE_HALO).min(shape[a]));
            let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
            // a tile with no voxel at 64 or more has zero centrality and no seeds
            if !pres.any_set_in(lo, dims, |p| p >= 64) {
                return None;
            }
            let block = |v: &Vol<u8>| Dense { dims, data: v.read_block(lo, dims) };
            let (cent, seeds) = stage1(&block(&pres), &block(nx), &block(ny));
            let inner = [0, 1, 2].map(|a| end[a] - start[a]);
            let off = [0, 1, 2].map(|a| start[a] - lo[a]);
            let mut c = Vec::with_capacity(inner.iter().product());
            let mut s = Vec::with_capacity(c.capacity());
            for z in 0..inner[0] {
                for y in 0..inner[1] {
                    let at = ((z + off[0]) * dims[1] + y + off[1]) * dims[2] + off[2];
                    c.extend_from_slice(&cent[at..at + inner[2]]);
                    s.extend_from_slice(&seeds[at..at + inner[2]]);
                }
            }
            Some((start, inner, c, s))
        })
        .collect();
    let mut cent = Vol::new(shape, 0u16);
    let mut seeds = Vol::new(shape, 0u8);
    for (start, inner, c, s) in results.into_iter().flatten() {
        cent.write_block(start, inner, &c);
        seeds.write_block(start, inner, &s);
    }
    FamilyFields { pres, cent, seeds }
}

/// The seed catalogue: coordinates in tracing order, grouped by 1024³ region
pub struct Catalogue {
    pub coords: Vec<[i32; 3]>,
    /// `starts[r]..starts[r + 1]` are region r's seeds
    pub starts: Vec<usize>,
    /// The first `primary[r]` seeds of region r are priority 2
    pub primary: Vec<usize>,
}

/// `prepare_seeds` (prepare.py:162): priority 2 then 1, each in descending centrality (reversed
/// unstable argsort), then a stable sort by region
pub fn catalogue(seeds: &Vol<u8>, cent: &Vol<u16>) -> Catalogue {
    let shape = seeds.shape();
    let mut by_priority: [Vec<[i32; 3]>; 2] = [Vec::new(), Vec::new()];
    seeds.for_each_set(|z, y, x, v| {
        if v == 2 || v == 1 {
            by_priority[(v == 1) as usize].push([z as i32, y as i32, x as i32]);
        }
    });
    let n_primary = by_priority[0].len();
    let mut ordered = Vec::with_capacity(n_primary + by_priority[1].len());
    for raw in &by_priority {
        let values: Vec<u16> = raw.iter().map(|c| cent.get(c[0] as usize, c[1] as usize, c[2] as usize)).collect();
        let order = argsort_u16(&values);
        ordered.extend(order.iter().rev().map(|&i| raw[i]));
    }
    const REGION: usize = 1024;
    let grid = shape.map(|n| n.div_ceil(REGION));
    let key = |c: &[i32; 3]| {
        let r = c.map(|v| v as usize / REGION);
        (r[0] * grid[1] + r[1]) * grid[2] + r[2]
    };
    let keys: Vec<usize> = ordered.iter().map(key).collect();
    let n_regions = grid.iter().product();
    let mut counts = vec![0usize; n_regions];
    let mut primary = vec![0usize; n_regions];
    for (i, &k) in keys.iter().enumerate() {
        counts[k] += 1;
        if i < n_primary {
            primary[k] += 1;
        }
    }
    let mut idx: Vec<usize> = (0..ordered.len()).collect();
    idx.sort_by_key(|&i| keys[i]);
    let coords = idx.iter().map(|&i| ordered[i]).collect();
    let mut starts = vec![0usize; n_regions + 1];
    for r in 0..n_regions {
        starts[r + 1] = starts[r] + counts[r];
    }
    Catalogue { coords, starts, primary }
}
