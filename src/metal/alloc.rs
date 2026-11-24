//! A global allocator for the macOS binaries that maps large blocks straight from the kernel
//!
//! macOS malloc keeps freed large blocks dirty in its large-entry cache, and they keep counting toward the process
//! footprint: after an nnU-Net load, 620 MB of freed checkpoint and host-weight copies stayed resident as
//! `Malloc Large (empty)` (`footprint`, `vmmap`) beside 2.2 GB of GPU buffers, and `malloc_zone_pressure_relief` did
//! not return them. Blocks of [`LargePages::THRESHOLD`] or more are mapped with `mmap` and unmapped on free, so their
//! pages leave the footprint at once; everything smaller goes to the system allocator as before

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

// <sys/mman.h> on Darwin
const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const MAP_PRIVATE: i32 = 0x2;
const MAP_ANON: i32 = 0x1000;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
/// The largest alignment a mapping guarantees: the 16 KiB page of Apple silicon
const PAGE: usize = 16 * 1024;

/// The system allocator, with blocks of at least [`LargePages::THRESHOLD`] bytes mapped from the kernel
pub struct LargePages;

impl LargePages {
    /// Blocks this large or larger are mapped: well past malloc's small and medium sizes, so the mapping cost
    /// (a syscall and zero-filled pages) only meets buffers that are filled anyway
    pub const THRESHOLD: usize = 1 << 20;

    fn mapped(layout: Layout) -> bool {
        layout.size() >= Self::THRESHOLD && layout.align() <= PAGE
    }
}

// SAFETY: mapped blocks are page aligned (at least `layout.align()`, checked by `mapped`), at least `size` long, and
// freed with the same length they were mapped with; every other block is the system allocator's
unsafe impl GlobalAlloc for LargePages {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !Self::mapped(layout) {
            // SAFETY: the caller's contract, passed through
            return unsafe { System.alloc(layout) };
        }
        // SAFETY: an anonymous private mapping takes no existing memory
        let p =
            unsafe { mmap(std::ptr::null_mut(), layout.size(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0) };
        if p == MAP_FAILED { std::ptr::null_mut() } else { p.cast() }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !Self::mapped(layout) {
            // SAFETY: as in alloc
            return unsafe { System.alloc_zeroed(layout) };
        }
        // SAFETY: fresh anonymous pages are zero
        unsafe { self.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !Self::mapped(layout) {
            // SAFETY: the block came from System.alloc with this layout
            return unsafe { System.dealloc(ptr, layout) };
        }
        // SAFETY: the block is a mapping of exactly layout.size() bytes made by alloc; a failure would leave it mapped,
        // which leaks but never corrupts
        unsafe { munmap(ptr.cast(), layout.size()) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract; the new layout keeps the alignment
        let new = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        if !Self::mapped(layout) && !Self::mapped(new) {
            // SAFETY: both blocks are the system allocator's
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        // SAFETY: a block of the new layout, then the old contents up to the shorter length, then the old block freed
        unsafe {
            let p = self.alloc(new);
            if !p.is_null() {
                std::ptr::copy_nonoverlapping(ptr, p, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
            p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LargePages;
    use std::alloc::{GlobalAlloc, Layout};

    /// Mapped and system blocks round-trip their contents through realloc across the threshold both ways, and a
    /// zeroed mapped block reads zero
    #[test]
    fn realloc_crosses_the_threshold() {
        let a = LargePages;
        let small = Layout::from_size_align(4096, 8).unwrap();
        // SAFETY: each block is used within its length and freed once with its own layout
        unsafe {
            let p = a.alloc(small);
            for i in 0..4096 {
                *p.add(i) = (i % 251) as u8;
            }
            let big = 3 * LargePages::THRESHOLD;
            let q = a.realloc(p, small, big);
            assert!((0..4096).all(|i| *q.add(i) == (i % 251) as u8));
            *q.add(big - 1) = 7;
            let r = a.realloc(q, Layout::from_size_align(big, 8).unwrap(), 100);
            assert!((0..100).all(|i| *r.add(i) == (i % 251) as u8));
            a.dealloc(r, Layout::from_size_align(100, 8).unwrap());

            let z = Layout::from_size_align(2 * LargePages::THRESHOLD, 64).unwrap();
            let p = a.alloc_zeroed(z);
            assert!(!p.is_null() && (p as usize).is_multiple_of(64));
            assert!(std::slice::from_raw_parts(p, z.size()).iter().all(|&b| b == 0));
            a.dealloc(p, z);
        }
    }
}
