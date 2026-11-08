//! Test-time augmentation for the 2D-output readers: the PHerc. 0306B recipe's 8-fold dihedral TTA
//!
//! The recipe (`zoomlib.villa_map(..., tta8=True)` over `fastink.VillaFast.predict(..., tta="mirror4")`, at
//! pherc0306b-first-letters 3f60827) reads each map twice: on the 17-layer window as stored, and on the
//! window turned a quarter (numpy `rot90(win, 1, axes=(1, 2))`) with a villa tiling of its own. Each read
//! runs every tile four times, as stored and flipped in y, in x and in both, and averages the four float32
//! sigmoids, read back through their flips, before villa's Hann blend; each read then ends as villa's
//! uint8 map. The map is `(q0 / 255 + rot90(q1, -1) / 255) / 2` in float32. Neither read runs villa's
//! low-resolution occupancy scan: the recipe hands villa the window as an array, so only tiles whose
//! window is all zero are skipped
//!
//! The engine runs both reads on the GPU with the volume and model resident: `tile_norm` writes the
//! flipped copies, a kernel turns the volume, and the blend and the final average are kernels
//! (`inkrs-kernels/src/tta.rs`)

use color_eyre::eyre::Result;

use crate::ensure_supported;

/// Copies per tile in a mirror4 read
pub const MIRRORS: usize = 4;

/// Test-time augmentation of a reader's maps
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tta {
    /// One read, villa's map
    #[default]
    Off,
    /// The 306B recipe's 8-fold dihedral TTA: two reads (as stored, turned a quarter) of four mirrored
    /// copies per tile; a float32 map
    Dihedral8,
}

impl Tta {
    /// From the CLI's and Python's variant count: 1 (off) or 8
    pub fn from_count(n: u32) -> Result<Self> {
        ensure_supported!(matches!(n, 1 | 8), "--tta {n}: 1 (off) or 8 (the 306B recipe's dihedral TTA)");
        Ok(if n == 8 { Self::Dihedral8 } else { Self::Off })
    }

    /// Network runs per tile position over the reads: 1 or 8
    pub const fn count(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::Dihedral8 => 8,
        }
    }

    pub const fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_round_trip_and_others_are_refused() {
        assert_eq!(Tta::from_count(1).unwrap(), Tta::Off);
        assert_eq!(Tta::from_count(8).unwrap(), Tta::Dihedral8);
        assert_eq!(Tta::Dihedral8.count(), 8);
        for n in [0, 2, 4, 16] {
            assert!(Tta::from_count(n).is_err(), "{n}");
        }
    }
}
