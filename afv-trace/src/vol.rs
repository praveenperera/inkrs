//! Sparse in-memory volumes stored as 32³ bricks
//!
//! The band's fields are mostly air: a brick is allocated only once a value other than the volume's
//! default is written to it, and reads from an unallocated brick return the default. This replaces the
//! reference's compressed Zarr work store and its paged tile cache with plain memory

use std::ops::Range;

use rayon::prelude::*;

/// Brick edge, as a shift
pub const SHIFT: usize = 5;
/// Brick edge in voxels
pub const EDGE: usize = 1 << SHIFT;
const MASK: usize = EDGE - 1;
/// Voxels per brick
pub const BRICK: usize = EDGE * EDGE * EDGE;

/// A z, y, x shape
pub type Shape = [usize; 3];

/// A sparse 3D volume of `T` in z, y, x order
pub struct Vol<T> {
    shape: Shape,
    grid: Shape,
    default: T,
    bricks: Vec<Option<Box<[T]>>>,
}

impl<T: Copy + PartialEq + Send + Sync> Vol<T> {
    /// An empty volume where every voxel reads as `default`
    pub fn new(shape: Shape, default: T) -> Self {
        let grid = shape.map(|n| n.div_ceil(EDGE));
        let bricks = (0..grid.iter().product()).map(|_| None).collect();
        Self { shape, grid, default, bricks }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Brick grid size in z, y, x
    pub fn grid(&self) -> Shape {
        self.grid
    }

    #[inline(always)]
    fn brick_index(&self, z: usize, y: usize, x: usize) -> usize {
        ((z >> SHIFT) * self.grid[1] + (y >> SHIFT)) * self.grid[2] + (x >> SHIFT)
    }

    #[inline(always)]
    fn offset(z: usize, y: usize, x: usize) -> usize {
        ((z & MASK) << (2 * SHIFT)) | ((y & MASK) << SHIFT) | (x & MASK)
    }

    /// The value at an in-bounds voxel
    #[inline(always)]
    pub fn get(&self, z: usize, y: usize, x: usize) -> T {
        debug_assert!(z < self.shape[0] && y < self.shape[1] && x < self.shape[2]);
        match &self.bricks[self.brick_index(z, y, x)] {
            Some(b) => b[Self::offset(z, y, x)],
            None => self.default,
        }
    }

    /// The value at a voxel given as signed coordinates, which must be in bounds
    #[inline(always)]
    pub fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.get(z as usize, y as usize, x as usize)
    }

    /// Set one voxel, allocating its brick on first non-default write
    #[inline]
    pub fn set(&mut self, z: usize, y: usize, x: usize, value: T) {
        let bi = self.brick_index(z, y, x);
        let default = self.default;
        match &mut self.bricks[bi] {
            Some(b) => b[Self::offset(z, y, x)] = value,
            slot @ None => {
                if value != default {
                    let mut b = vec![default; BRICK].into_boxed_slice();
                    b[Self::offset(z, y, x)] = value;
                    *slot = Some(b);
                }
            }
        }
    }

    /// Bytes held by allocated bricks
    pub fn resident_bytes(&self) -> usize {
        self.bricks.iter().filter(|b| b.is_some()).count() * BRICK * std::mem::size_of::<T>()
    }

    /// Copy a dense C-order block with origin `lo` and extent `dims` into the volume; bricks that would
    /// stay all-default are not allocated
    pub fn write_block(&mut self, lo: Shape, dims: Shape, data: &[T]) {
        assert_eq!(data.len(), dims.iter().product::<usize>());
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                let row = &data[(z * dims[1] + y) * dims[2]..][..dims[2]];
                for (x, &v) in row.iter().enumerate() {
                    self.set(lo[0] + z, lo[1] + y, lo[2] + x, v);
                }
            }
        }
    }

    /// Read a dense C-order block with origin `lo` and extent `dims`
    pub fn read_block(&self, lo: Shape, dims: Shape) -> Vec<T> {
        let mut out = vec![self.default; dims.iter().product()];
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                let row = &mut out[(z * dims[1] + y) * dims[2]..][..dims[2]];
                for (x, v) in row.iter_mut().enumerate() {
                    *v = self.get(lo[0] + z, lo[1] + y, lo[2] + x);
                }
            }
        }
        out
    }

    /// Visit every voxel whose value differs from the default, in C order over the whole volume
    pub fn for_each_set(&self, mut f: impl FnMut(usize, usize, usize, T)) {
        for z in 0..self.shape[0] {
            for y in 0..self.shape[1] {
                for bx in 0..self.grid[2] {
                    let Some(b) = &self.bricks[((z >> SHIFT) * self.grid[1] + (y >> SHIFT)) * self.grid[2] + bx] else {
                        continue;
                    };
                    let row = Self::offset(z, y, 0);
                    let width = EDGE.min(self.shape[2] - bx * EDGE);
                    for (dx, &v) in b[row..row + width].iter().enumerate() {
                        if v != self.default {
                            f(z, y, bx * EDGE + dx, v);
                        }
                    }
                }
            }
        }
    }

    /// Whether any voxel of the block `[lo, lo + dims)` differs from the default
    pub fn any_set_in(&self, lo: Shape, dims: Shape, pred: impl Fn(T) -> bool) -> bool {
        let first = lo.map(|v| v >> SHIFT);
        let last = [0, 1, 2].map(|a| (lo[a] + dims[a] - 1) >> SHIFT);
        for bz in first[0]..=last[0] {
            for by in first[1]..=last[1] {
                for bx in first[2]..=last[2] {
                    if self.bricks[(bz * self.grid[1] + by) * self.grid[2] + bx].is_none() {
                        continue;
                    }
                    let blo = [bz * EDGE, by * EDGE, bx * EDGE];
                    let a0 = [0, 1, 2].map(|a| blo[a].max(lo[a]));
                    let a1 = [0, 1, 2].map(|a| (blo[a] + EDGE).min(lo[a] + dims[a]).min(self.shape[a]));
                    for z in a0[0]..a1[0] {
                        for y in a0[1]..a1[1] {
                            for x in a0[2]..a1[2] {
                                if pred(self.get(z, y, x)) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Fill bricks in parallel: `f(brick_lo, brick_dims, out)` writes one brick's voxels in C order
    /// (dims clipped at the volume edge) and returns whether any differs from the default
    pub fn fill_bricks<F>(&mut self, f: F)
    where
        F: Fn(Shape, Shape, &mut [T]) -> bool + Sync,
    {
        self.fill_brick_rows(0..self.grid[0], f);
    }

    /// `fill_bricks` for the bricks whose z index lies in `rows`; the others are left as they are
    pub fn fill_brick_rows<F>(&mut self, rows: Range<usize>, f: F)
    where
        F: Fn(Shape, Shape, &mut [T]) -> bool + Sync,
    {
        let grid = self.grid;
        let shape = self.shape;
        let default = self.default;
        let plane = grid[1] * grid[2];
        let slab = &mut self.bricks[rows.start * plane..rows.end * plane];
        slab.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let bi = rows.start * plane + i;
            let bz = bi / plane;
            let by = (bi / grid[2]) % grid[1];
            let bx = bi % grid[2];
            let lo = [bz * EDGE, by * EDGE, bx * EDGE];
            let dims = [0, 1, 2].map(|a| EDGE.min(shape[a] - lo[a]));
            let mut dense = vec![default; dims.iter().product()];
            if !f(lo, dims, &mut dense) {
                *slot = None;
                return;
            }
            let mut b = vec![default; BRICK].into_boxed_slice();
            for z in 0..dims[0] {
                for y in 0..dims[1] {
                    let src = &dense[(z * dims[1] + y) * dims[2]..][..dims[2]];
                    let at = Self::offset(z, y, 0);
                    b[at..at + dims[2]].copy_from_slice(src);
                }
            }
            *slot = Some(b);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_reads_default_and_round_trips() {
        let mut v = Vol::new([40, 70, 33], 128u8);
        assert_eq!(v.get(39, 69, 32), 128);
        v.set(39, 69, 32, 7);
        v.set(0, 0, 0, 128);
        assert_eq!(v.get(39, 69, 32), 7);
        assert_eq!(v.resident_bytes(), BRICK);
        let block: Vec<u8> = (0..3 * 4 * 5).map(|i| i as u8).collect();
        v.write_block([30, 31, 28], [3, 4, 5], &block);
        assert_eq!(v.read_block([30, 31, 28], [3, 4, 5]), block);
    }
}
