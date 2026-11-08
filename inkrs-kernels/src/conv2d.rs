//! inkrs's own 2D convolutions for the villa readers' UNet, replacing cuDNN: every 3x3 conv (stride 1
//! or 2, "same" padding) and the 1x1 skip convs, FP16 channels-last in and out, float32 accumulation on
//! the tensor cores (`mma.m16n8k16`, as cuDNN's tensor-op algorithms run them), no bias (every conv
//! feeds an InstanceNorm, which cancels it). Each conv also writes its InstanceNorm's statistics, so
//! the output is never read back for them
//!
//! [`conv_body`] is an implicit GEMM: M is the output pixels, N the output channels, K the taps times
//! the input channels. A block owns a patch of `SAMP` samples x `TH` x `TW` output pixels and `BN = 8
//! NT` output channels, and walks the input channels in chunks of `CK`. For each chunk it stages the
//! patch's input halo (`(TH - 1) S + KS` x `(TW - 1) S + KS` pixels per sample, zero outside the
//! image) in shared memory with `cp.async`, then runs every tap from it: the A fragments come from the
//! halo with `ldmatrix` at each tap's offset, so the input is read about 1.3x rather than once per tap.
//! The B fragments are host-ordered (`conv_frags` in the host crate: one 16-byte chunk per (16 input
//! channels, tap, 16 output channels, lane)) and come from global memory through L1, loaded one
//! (tap, 16 channels) step ahead into registers so their latency hides behind the step's MMAs. Chunks are
//! pipelined `NS` deep over a stream of (patch, chunk) items, so a block that strides over several
//! patches prefetches the next patch's halo while it computes the current one
//!
//! Each warp owns `MT` 16-pixel tiles of the patch and all `BN` channels. A pixel's 16-byte chunks sit
//! in shared memory XOR-swizzled by their 128-byte line (`swz`), so the eight row addresses of an
//! `ldmatrix` mostly fall in distinct banks whatever the tap's offset or the stride
//!
//! The epilogue writes the FP16-rounded output and, per (sample, channel), the patch's mean of the
//! FP16 values and their sum of squared deviations from it (two passes over the registers, summed in a
//! fixed order, so the result never depends on scheduling). [`conv_in_merge`] merges a sample's patch
//! partials in float32, in patch order, into `[mean, invstd]`, the `stats` `in_apply` takes. The products are summed in
//! the tensor core's order, not cuDNN's, so outputs can differ from cuDNN's in the last FP16 bit

use core::ptr::addr_of_mut;

use cuda_device::async_copy::{cp_async_ca_zfill_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{
    __unroll_config, blockDim_x, blockIdx_x, blockIdx_y, gridDim_x, sync_threads, threadIdx_x,
};
use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k16_f32_f16};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, warp};

use crate::common::{math, pack_f16, r16};
use crate::stem_conv2::Q;

/// The swizzled position of 16-byte chunk `lin` of a halo tile: its index within each 128-byte line is
/// XORed with the line's low three bits
#[inline(always)]
fn swz(lin: u32) -> u32 {
    lin ^ ((lin >> 3) & 7)
}

/// Chunks one ring slot holds: the halo of `units` chunks rounded up to whole 128-byte lines, since
/// `swz` permutes within a line of eight and a partial last line would spill into the next slot
#[inline(always)]
const fn slot_chunks(units: u32) -> u32 {
    units.div_ceil(8) * 8
}

/// Waits until at most `NS - 2` copy groups are pending: the item about to be computed has landed
///
/// # Safety
/// As `cp_async_wait_group`
#[inline(always)]
unsafe fn wait_stage<const NS: u32>() {
    // the PTX needs a literal count, so one arm per depth
    unsafe {
        match NS {
            2 => cp_async_wait_group(0),
            3 => cp_async_wait_group(1),
            _ => cp_async_wait_group(2),
        }
    }
}

/// One conv's runtime shape, shared by the helpers
#[derive(Clone, Copy)]
struct Shape {
    batch: u32,
    /// Input rows and columns
    h: u32,
    w: u32,
    cin: u32,
    cout: u32,
    /// Output rows and columns
    ho: u32,
    wo: u32,
}

/// Stages item `item` (patch `blockIdx.x + (item / chunks) gridDim.x`, input-channel chunk `item %
/// chunks`) into ring slot `item % NS`, or issues nothing past the last item. Every thread commits one
/// group either way, so the group count stays the same across the block
///
/// # Safety
/// `ring` is the block-shared ring of `NS` slots of `slot_chunks(SAMP * HR * HC * CK / 8)` chunks,
/// slot `item % NS` unread until a barrier; `x` holds every sample of the shape, checked by the caller
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn stage<
    const S: u32,
    const KS: u32,
    const TH: u32,
    const TW: u32,
    const SAMP: u32,
    const CK: u32,
    const NS: u32,
    const NWM: u32,
>(
    ring: *mut Q,
    x: &[Q],
    sh: Shape,
    item: u32,
    items: u32,
    chunks: u32,
    tid: u32,
) {
    let (hr, hc, nch) = ((TH - 1) * S + KS, (TW - 1) * S + KS, CK / 8);
    let units = SAMP * hr * hc * nch;
    if item < items {
        let patch = blockIdx_x() + (item / chunks) * gridDim_x();
        let ci = item % chunks;
        let (py, px) = (sh.ho / TH, sh.wo / TW);
        let (tx, rest) = (patch % px, patch / px);
        let (ty, bg) = (rest % py, rest / py);
        let pad = (KS - 1) / 2;
        let (iy0, ix0) = ((ty * TH * S) as i32 - pad as i32, (tx * TW * S) as i32 - pad as i32);
        let slot = ring.wrapping_add(((item % NS) * slot_chunks(units)) as usize);
        let cin8 = sh.cin / 8;
        let mut u = tid;
        while u < units {
            let (kc, pix) = (u % nch, u / nch);
            let (col, rest) = (pix % hc, pix / hc);
            let (row, s) = (rest % hr, rest / hr);
            let (iy, ix) = (iy0 + row as i32, ix0 + col as i32);
            let inside = (iy >= 0) & (iy < sh.h as i32) & (ix >= 0) & (ix < sh.w as i32);
            let o = if inside {
                (((bg * SAMP + s) * sh.h + iy as u32) * sh.w + ix as u32) * cin8 + ci * nch + kc
            } else {
                0
            };
            // SAFETY: `o` is in `x` (the caller's check) or 0 with a zero source size; the slot holds
            // `slot_chunks(units)` chunks and `swz` permutes within each line of eight
            unsafe {
                cp_async_ca_zfill_16(
                    slot.add(swz(pix * nch + kc) as usize) as *mut u32,
                    x.as_ptr().add(o as usize) as *const u8,
                    if inside { 16 } else { 0 },
                )
            };
            u += NWM * 32;
        }
    }
    // SAFETY: this thread's own copies
    unsafe { cp_async_commit_group() };
}

/// Sums each of `q` (per n-tile, element parity) over the lanes with this lane's `th`, and has lanes
/// `g == 0` write the warp totals to `red[wid][channel]`
///
/// # Safety
/// `red` holds `(wid + 1) * bn` floats; every lane of the warp calls this together
#[inline(always)]
unsafe fn reduce_put<const NT: usize>(q: &mut [[f32; 2]; NT], red: *mut f32, wid: u32, bn: u32, g: u32, th: u32) {
    let mut nt = 0;
    while nt < NT {
        __unroll_config::<0>();
        let mut e = 0;
        while e < 2 {
            __unroll_config::<0>();
            let mut v = q[nt][e];
            v += warp::shuffle_xor_f32(v, 4);
            v += warp::shuffle_xor_f32(v, 8);
            v += warp::shuffle_xor_f32(v, 16);
            q[nt][e] = v;
            if g == 0 {
                // SAFETY: the caller's; channel nt * 8 + 2 th + e < bn
                unsafe { *red.add((wid * bn + nt as u32 * 8 + 2 * th + e as u32) as usize) = v };
            }
            e += 1;
        }
        nt += 1;
    }
}

/// The patch partial `[mean, M2]` per (sample, channel) of the FP16-rounded `acc` into `partial`: the
/// sample's share of the patch reduced to its mean, then the squared deviations from that mean. Each
/// warp's pixels lie in one sample (`TH * TW` is a multiple of `MT * 16`). Three barriers: after the
/// sums, after the means are read, after the squares
///
/// # Safety
/// `red` holds `NWM * NT * 8` floats and `means` `SAMP * NT * 8`, block-shared and untouched outside
/// this call; `partial` holds every (sample, segment, channel) of the shape; every thread of the block
/// calls this together
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn stats<const TH: u32, const TW: u32, const SAMP: u32, const MT: usize, const NT: usize, const NWM: u32>(
    partial: &mut DisjointSlice<[f32; 2]>,
    acc: &[[[f32; 4]; NT]; MT],
    red: *mut f32,
    means: *mut f32,
    sh: Shape,
    patch: u32,
    n0: u32,
    tid: u32,
) {
    let bn = (NT * 8) as u32;
    let (wid, lane) = (tid / 32, tid % 32);
    let (g, th) = (lane / 4, lane % 4);
    let wps = TH * TW / (MT as u32 * 16);
    let count = (TH * TW) as f32;
    // the sum over sample `s`'s warps of channel `k`, in warp order
    let total = |s: u32, k: u32| {
        let mut t = 0f32;
        let mut i = 0;
        while i < wps {
            // SAFETY: (s * wps + i) < NWM
            t += unsafe { *red.add(((s * wps + i) * bn + k) as usize) };
            i += 1;
        }
        t
    };

    let mut q = [[0f32; 2]; NT];
    let mut mt = 0;
    while mt < MT {
        __unroll_config::<0>();
        let mut nt = 0;
        while nt < NT {
            __unroll_config::<0>();
            let c = acc[mt][nt];
            q[nt][0] += r16(c[0]) + r16(c[2]);
            q[nt][1] += r16(c[1]) + r16(c[3]);
            nt += 1;
        }
        mt += 1;
    }
    // SAFETY: the caller's
    unsafe { reduce_put::<NT>(&mut q, red, wid, bn, g, th) };
    sync_threads();
    let mut i = tid;
    while i < SAMP * bn {
        // SAFETY: i < SAMP * bn
        unsafe { *means.add(i as usize) = total(i / bn, i % bn) / count };
        i += NWM * 32;
    }
    sync_threads();
    let s = wid / wps;
    let mut q = [[0f32; 2]; NT];
    let mut mt = 0;
    while mt < MT {
        __unroll_config::<0>();
        let mut nt = 0;
        while nt < NT {
            __unroll_config::<0>();
            let c = acc[mt][nt];
            let k = s * bn + nt as u32 * 8 + 2 * th;
            // SAFETY: k + 1 < SAMP * bn
            let (m0, m1) = unsafe { (*means.add(k as usize), *means.add(k as usize + 1)) };
            let (d0, d1, d2, d3) = (r16(c[0]) - m0, r16(c[1]) - m1, r16(c[2]) - m0, r16(c[3]) - m1);
            q[nt][0] += d0 * d0 + d2 * d2;
            q[nt][1] += d1 * d1 + d3 * d3;
            nt += 1;
        }
        mt += 1;
    }
    // SAFETY: the caller's
    unsafe { reduce_put::<NT>(&mut q, red, wid, bn, g, th) };
    sync_threads();
    let (py, px) = (sh.ho / TH, sh.wo / TW);
    let segs = py * px;
    let (seg, bg) = (patch % segs, patch / segs);
    let mut i = tid;
    while i < SAMP * bn {
        let (s, k) = (i / bn, i % bn);
        let o = ((bg * SAMP + s) * segs + seg) * sh.cout + n0 + k;
        // SAFETY: in `partial` by the caller's check; thread i is the only writer of (sample, seg, k)
        unsafe { *partial.get_unchecked_mut(o as usize) = [*means.add(i as usize), total(s, k)] };
        i += NWM * 32;
    }
}

/// This lane's B fragments for step `st` (tap `st / (CK / 16)`, 16 input channels `st % (CK / 16)`) of
/// input-channel chunk `ci`: one 16-byte chunk per n16 group of the block's `NT / 2`
///
/// # Safety
/// `wf` holds `(cin / 16) * KS^2 * (cout / 16) * 32` chunks, `ci < cin / CK`, `st < KS^2 * CK / 16`
/// and `n0 + 8 NT <= cout`, checked by the caller
#[inline(always)]
unsafe fn load_b<const KS: u32, const CK: u32, const NT: usize>(
    wf: &[Q],
    cout: u32,
    ci: u32,
    st: u32,
    n0: u32,
    lane: u32,
) -> [[u32; 4]; NT] {
    let (t, ks) = (st / (CK / 16), st % (CK / 16));
    let q16 = ci * (CK / 16) + ks;
    let wbase = ((q16 * KS * KS + t) * (cout / 16) + n0 / 16) * 32 + lane;
    let mut b = [[0u32; 4]; NT];
    let mut j = 0;
    while j < NT / 2 {
        __unroll_config::<0>();
        // SAFETY: inside `wf` by the caller's bounds
        let Q(bw) = unsafe { *wf.get_unchecked((wbase + j as u32 * 32) as usize) };
        b[j] = bw;
        j += 1;
    }
    b
}

/// The conv body every `conv_*` kernel instantiates; see the module docs. `out` is `[batch, ho, wo,
/// cout]` FP16 pairs, `partial` `[batch, (ho / TH) (wo / TW), cout]`, `x` `[batch, h, w, cin]` as
/// 16-byte chunks, `wf` the B fragments. Grid `(any, cout / BN)`, block `NWM * 32`; block x strides
/// over the `batch / SAMP * (ho / TH) * (wo / TW)` patches
///
/// # Safety
/// `ring`, `red` and `means` are block-shared arrays of `NS * slot_chunks(SAMP * HR * HC * CK / 8)`
/// chunks, `NWM * NT * 8` and `SAMP * NT * 8` floats, used by nothing else
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv_body<
    const S: u32,
    const KS: u32,
    const TH: u32,
    const TW: u32,
    const SAMP: u32,
    const CK: u32,
    const MT: usize,
    const NT: usize,
    const NWM: u32,
    const NS: u32,
>(
    ring: *mut Q,
    red: *mut f32,
    means: *mut f32,
    out: &mut DisjointSlice<u32>,
    partial: &mut DisjointSlice<[f32; 2]>,
    x: &[Q],
    wf: &[Q],
    sh: Shape,
) {
    let threads = NWM * 32;
    let bn = (NT * 8) as u32;
    let taps = KS * KS;
    let (hr, hc, nch) = ((TH - 1) * S + KS, (TW - 1) * S + KS, CK / 8);
    let units = SAMP * hr * hc * nch;
    let tid = threadIdx_x();
    let (py, px) = (sh.ho / TH, sh.wo / TW);
    let patches = sh.batch / SAMP * py * px;
    let chunks = sh.cin / CK;
    let n0 = blockIdx_y() * bn;
    // block-uniform, so no block leaves before a barrier, shuffle, ldmatrix or mma that others reach
    if (blockDim_x() != threads)
        | (NWM * MT as u32 * 16 != SAMP * TH * TW)
        | ((TH * TW) % (MT as u32 * 16) != 0)
        | (chunks == 0)
        | (sh.cin % CK != 0)
        | (sh.cout % bn != 0)
        | (n0 + bn > sh.cout)
        | (sh.batch % SAMP != 0)
        | (sh.ho % TH != 0)
        | (sh.wo % TW != 0)
        | ((sh.batch * sh.h * sh.w * (sh.cin / 8)) as usize > x.len())
        | ((sh.batch * sh.ho * sh.wo * (sh.cout / 2)) as usize > out.len())
        | (((sh.cin / 16) * taps * (sh.cout / 16) * 32) as usize > wf.len())
        | ((sh.batch * py * px * sh.cout) as usize > partial.len())
    {
        return;
    }
    let mine = if blockIdx_x() < patches { (patches - blockIdx_x()).div_ceil(gridDim_x()) } else { 0 };
    let items = mine * chunks;
    // SAFETY: `ring` is block-shared
    let ring_at = unsafe { cvta_generic_to_shared_u32(ring as *const u8) };
    let (wid, lane) = (tid / 32, tid % 32);
    let (g, th) = (lane / 4, lane % 4);

    // this lane's ldmatrix row in each of its tiles: output pixel o of the patch, as its halo pixel at
    // tap (0, 0)
    let mut hp0 = [0u32; MT];
    let mut mt = 0;
    while mt < MT {
        __unroll_config::<0>();
        let o = (wid * MT as u32 + mt as u32) * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
        let (s, r, c) = (o / (TH * TW), (o / TW) % TH, o % TW);
        hp0[mt] = (s * hr + r * S) * hc + c * S;
        mt += 1;
    }
    let khalf = lane >> 4;

    let mut item = 0;
    while item < NS - 1 {
        // SAFETY: the check above; nothing reads the ring yet
        unsafe { stage::<S, KS, TH, TW, SAMP, CK, NS, NWM>(ring, x, sh, item, items, chunks, tid) };
        item += 1;
    }

    let steps = taps * (CK / 16);
    // the B fragments of the step about to run, loaded one step ahead
    // SAFETY: chunk 0, step 0 is in range by the check above
    let mut bnext = unsafe { load_b::<KS, CK, NT>(wf, sh.cout, 0, 0, n0, lane) };
    let mut acc = [[[0f32; 4]; NT]; MT];
    let mut i = 0;
    while i < items {
        // SAFETY: groups committed one per item by every thread
        unsafe { wait_stage::<NS>() };
        sync_threads();
        // slot (i + NS - 1) % NS last held item i - 1, which every thread finished before the barrier
        // SAFETY: the check above
        unsafe { stage::<S, KS, TH, TW, SAMP, CK, NS, NWM>(ring, x, sh, i + NS - 1, items, chunks, tid) };
        let ci = i % chunks;
        let slot = ring_at + (i % NS) * slot_chunks(units) * 16;
        let mut st = 0;
        while st < steps {
            __unroll_config::<0>();
            let (t, ks) = (st / (CK / 16), st % (CK / 16));
            let (dy, dx) = (t / KS, t % KS);
            let b = bnext;
            // the next step's, the first of the next item's chunk after the last; past the last item it
            // loads chunk (items % chunks) step 0, in range and unused
            let (nci, nst) = if st + 1 < steps { (ci, st + 1) } else { ((i + 1) % chunks, 0) };
            // SAFETY: nci < chunks and nst < steps, in range by the check above
            bnext = unsafe { load_b::<KS, CK, NT>(wf, sh.cout, nci, nst, n0, lane) };
            let kc = 2 * ks + khalf;
            let mut mt = 0;
            while mt < MT {
                __unroll_config::<0>();
                let hp = hp0[mt] + dy * hc + dx;
                // SAFETY: the halo pixel is inside the slot (rows r S + dy < HR, columns c S + dx < HC)
                let a = unsafe { ldmatrix_x4_shared_u32(slot + swz(hp * nch + kc) * 16) };
                let mut j = 0;
                while j < NT / 2 {
                    __unroll_config::<0>();
                    // SAFETY: the whole warp runs this, with fragments in the documented layout
                    unsafe {
                        acc[mt][2 * j] = mma_m16n8k16_f32_f16(acc[mt][2 * j], a, [b[j][0], b[j][1]]);
                        acc[mt][2 * j + 1] = mma_m16n8k16_f32_f16(acc[mt][2 * j + 1], a, [b[j][2], b[j][3]]);
                    }
                    j += 1;
                }
                mt += 1;
            }
            st += 1;
        }
        if ci + 1 == chunks {
            let patch = blockIdx_x() + (i / chunks) * gridDim_x();
            let (tx, rest) = (patch % px, patch / px);
            let (ty, bg) = (rest % py, rest / py);
            let mut mt = 0;
            while mt < MT {
                __unroll_config::<0>();
                let o = (wid * MT as u32 + mt as u32) * 16 + g;
                let mut half = 0;
                while half < 2 {
                    __unroll_config::<0>();
                    let o = o + 8 * half;
                    let (s, r, c) = (o / (TH * TW), (o / TW) % TH, o % TW);
                    let pix = ((bg * SAMP + s) * sh.ho + ty * TH + r) * sh.wo + tx * TW + c;
                    let mut nt = 0;
                    while nt < NT {
                        __unroll_config::<0>();
                        let v = acc[mt][nt];
                        let p = pix * (sh.cout / 2) + (n0 + nt as u32 * 8) / 2 + th;
                        // SAFETY: inside `out` by the check above; one lane writes each (pixel, pair)
                        unsafe {
                            *out.get_unchecked_mut(p as usize) =
                                pack_f16(v[2 * half as usize], v[2 * half as usize + 1])
                        };
                        nt += 1;
                    }
                    half += 1;
                }
                mt += 1;
            }
            // SAFETY: the check above; every thread of the block reaches this
            unsafe { stats::<TH, TW, SAMP, MT, NT, NWM>(partial, &acc, red, means, sh, patch, n0, tid) };
            acc = [[[0f32; 4]; NT]; MT];
        }
        i += 1;
    }
    // no copy may be left in flight when the block exits
    // SAFETY: this thread's own groups
    unsafe { cp_async_wait_group(0) };
}

/// One `conv_*` kernel: a monomorphisation of [`conv_body`] under a fixed PTX name, with its shared
/// memory declared here (a generic fn cannot own a `static`)
macro_rules! conv_kernel {
    ($(#[$doc:meta])* $name:ident: S $s:literal, KS $ks:literal, TH $th:literal, TW $tw:literal, SAMP $samp:literal,
     CK $ck:literal, MT $mt:literal, NT $nt:literal, NWM $nwm:literal, NS $ns:literal,
     BOUNDS $threads:literal, $minb:literal) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $minb)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            mut out: DisjointSlice<u32>,
            mut partial: DisjointSlice<[f32; 2]>,
            x: &[Q],
            wf: &[Q],
            batch: u32,
            h: u32,
            w: u32,
            cin: u32,
            cout: u32,
        ) {
            const HR: u32 = ($th - 1) * $s + $ks;
            const HC: u32 = ($tw - 1) * $s + $ks;
            // `swz` stays inside a slot only when the slot is whole lines of eight chunks
            const _: () = assert!(slot_chunks($samp * HR * HC * $ck / 8) % 8 == 0);
            const _: () = assert!(slot_chunks($samp * HR * HC * $ck / 8) >= $samp * HR * HC * $ck / 8);
            static mut RING: SharedArray<Q, { ($ns * slot_chunks($samp * HR * HC * $ck / 8)) as usize }> =
                SharedArray::UNINIT;
            const _: () = assert!($threads == $nwm * 32);
            static mut RED: SharedArray<f32, { $nwm * $nt * 8 }> = SharedArray::UNINIT;
            static mut MEANS: SharedArray<f32, { $samp * $nt * 8 }> = SharedArray::UNINIT;
            let sh = Shape { batch, h, w, cin, cout, ho: (h + $s - 1) / $s, wo: (w + $s - 1) / $s };
            // SAFETY: the three arrays are this kernel's own, sized as `conv_body` needs
            unsafe {
                conv_body::<$s, $ks, $th, $tw, $samp, $ck, $mt, $nt, $nwm, $ns>(
                    addr_of_mut!(RING) as *mut Q,
                    addr_of_mut!(RED) as *mut f32,
                    addr_of_mut!(MEANS) as *mut f32,
                    &mut out,
                    &mut partial,
                    x,
                    wf,
                    sh,
                )
            }
        }
    };
}

conv_kernel!(
    /// 3x3, stride 1, 16 x 16 patches, 32 output channels a block: C = 32 at 128 x 128
    conv_s1_p16_n32: S 1, KS 3, TH 16, TW 16, SAMP 1, CK 32, MT 2, NT 4, NWM 8, NS 2, BOUNDS 256, 1
);
conv_kernel!(
    /// 3x3, stride 1, 16 x 16 patches, 64 output channels a block: 64 x 64 down to 16 x 16
    conv_s1_p16_n64: S 1, KS 3, TH 16, TW 16, SAMP 1, CK 32, MT 2, NT 8, NWM 8, NS 2, BOUNDS 256, 1
);
conv_kernel!(
    /// 3x3, stride 1, one 8 x 8 sample a block, 64 output channels: the 8 x 8 stage
    conv_s1_p8_n64: S 1, KS 3, TH 8, TW 8, SAMP 1, CK 32, MT 1, NT 8, NWM 4, NS 3, BOUNDS 128, 1
);
conv_kernel!(
    /// 3x3, stride 1, four 4 x 4 samples a block, 32 output channels: the 4 x 4 stage
    conv_s1_p4_n32: S 1, KS 3, TH 4, TW 4, SAMP 4, CK 32, MT 1, NT 4, NWM 4, NS 3, BOUNDS 128, 1
);
conv_kernel!(
    /// 3x3, stride 1, one 4 x 4 sample a block, 32 output channels: the 4 x 4 stage at a batch that is
    /// not a multiple of four
    conv_s1_p4x1_n32: S 1, KS 3, TH 4, TW 4, SAMP 1, CK 32, MT 1, NT 4, NWM 1, NS 3, BOUNDS 32, 1
);
conv_kernel!(
    /// 3x3, stride 2, 8 x 16 output patches, 64 output channels a block: the strided convs down to 16 x 16
    conv_s2_p8x16_n64: S 2, KS 3, TH 8, TW 16, SAMP 1, CK 16, MT 1, NT 8, NWM 8, NS 2, BOUNDS 256, 1
);
conv_kernel!(
    /// 3x3, stride 2, one 8 x 8 output sample a block, 64 output channels: 16 x 16 to 8 x 8
    conv_s2_p8_n64: S 2, KS 3, TH 8, TW 8, SAMP 1, CK 32, MT 1, NT 8, NWM 4, NS 2, BOUNDS 128, 1
);
conv_kernel!(
    /// 3x3, stride 2, four 4 x 4 output samples a block, 32 output channels: 8 x 8 to 4 x 4
    conv_s2_p4_n32: S 2, KS 3, TH 4, TW 4, SAMP 4, CK 32, MT 1, NT 4, NWM 4, NS 2, BOUNDS 128, 1
);
conv_kernel!(
    /// 3x3, stride 2, one 4 x 4 output sample a block, 32 output channels: 8 x 8 to 4 x 4 at a batch that
    /// is not a multiple of four
    conv_s2_p4x1_n32: S 2, KS 3, TH 4, TW 4, SAMP 1, CK 32, MT 1, NT 4, NWM 1, NS 2, BOUNDS 32, 1
);
conv_kernel!(
    /// 1x1, 16 x 16 patches, 64 output channels a block: the skip convs at 64 x 64 down to 16 x 16
    conv_k1_p16_n64: S 1, KS 1, TH 16, TW 16, SAMP 1, CK 32, MT 2, NT 8, NWM 8, NS 2, BOUNDS 256, 1
);
conv_kernel!(
    /// 1x1, one 8 x 8 sample a block, 64 output channels: the skip conv at 8 x 8
    conv_k1_p8_n64: S 1, KS 1, TH 8, TW 8, SAMP 1, CK 32, MT 1, NT 8, NWM 4, NS 3, BOUNDS 128, 1
);

conv_kernel!(
    /// [`conv_s1_p16_n32`] held to 128 registers, so two blocks fit an SM
    conv_s1_p16_n32_m2: S 1, KS 3, TH 16, TW 16, SAMP 1, CK 32, MT 2, NT 4, NWM 8, NS 2, BOUNDS 256, 2
);
conv_kernel!(
    /// [`conv_s1_p8_n64`] held to 128 registers, so four blocks fit an SM
    conv_s1_p8_n64_m4: S 1, KS 3, TH 8, TW 8, SAMP 1, CK 32, MT 1, NT 8, NWM 4, NS 3, BOUNDS 128, 4
);
conv_kernel!(
    /// 3x3, stride 1, one 8 x 8 sample a block, 32 output channels: twice the blocks of
    /// [`conv_s1_p8_n64`] for the 8 x 8 stage
    conv_s1_p8_n32: S 1, KS 3, TH 8, TW 8, SAMP 1, CK 32, MT 1, NT 4, NWM 4, NS 3, BOUNDS 128, 1
);
conv_kernel!(
    /// 3x3, stride 1, four 4 x 4 samples a block, 16 output channels: twice the blocks of
    /// [`conv_s1_p4_n32`] for the 4 x 4 stage
    conv_s1_p4_n16: S 1, KS 3, TH 4, TW 4, SAMP 4, CK 32, MT 1, NT 2, NWM 4, NS 3, BOUNDS 128, 1
);

/// A compensated float32 sum: `c` carries the low bits each addition to `sum` dropped
#[derive(Default)]
struct KahanF32 {
    sum: f32,
    c: f32,
}

impl KahanF32 {
    #[inline(always)]
    fn add(&mut self, v: f32) {
        let y = v - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }
}

/// Per (sample, channel) `[mean, invstd]` from a conv's patch partials, merged in compensated float32 in
/// patch order: the partials have equal counts, so the mean is the mean of their means and `M2 = sum(M2_i)
/// + count * sum((mean_i - mean)^2)`. The variance is the population one, as `in_finalize` takes it.
/// One thread per (sample, channel), so a warp reads 32 neighbouring channels of each partial;
/// float64 would run at 1/64 rate on GeForce parts, and the fixed order already makes it deterministic
/// Grid `ceil(batch * cols / 128)`, block 128. `partial` is `[batch, segs, cols]`
#[kernel]
pub fn conv_in_merge(mut stats: DisjointSlice<[f32; 2]>, partial: &[[f32; 2]], segs: u32, cols: u32, count: u32, eps: f32) {
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    // no barriers or shuffles below, so threads may leave on their own
    if (cols == 0) | (segs == 0) | (i as usize >= stats.len()) {
        return;
    }
    let (b, k) = (i / cols, i % cols);
    if ((b + 1) * segs * cols) as usize > partial.len() {
        return;
    }
    // SAFETY: (b + 1) * segs partials of `cols` channels are in `partial`, checked above
    let at = |seg: u32| unsafe { *partial.get_unchecked(((b * segs + seg) * cols + k) as usize) };
    // both sums compensated (Kahan): up to 576 patches of near-equal means, or one large M2 followed by
    // many small ones, would otherwise drop the low bits that set the mean and the variance
    let mut kahan = KahanF32::default();
    let mut seg = 0;
    while seg < segs {
        kahan.add(at(seg)[0]);
        seg += 1;
    }
    let mean = kahan.sum / segs as f32;
    let mut kahan = KahanF32::default();
    let mut seg = 0;
    while seg < segs {
        let [m, q] = at(seg);
        let d = m - mean;
        kahan.add(q + count as f32 * d * d);
        seg += 1;
    }
    let m2 = kahan.sum;
    let var = m2 / (segs * count) as f32;
    let var = if var > 0.0 { var } else { 0.0 };
    // SAFETY: i < stats.len(); thread i is the only writer of (b, k)
    unsafe { *stats.get_unchecked_mut(i as usize) = [mean, 1.0 / math::sqrtf(var + eps)] };
}
