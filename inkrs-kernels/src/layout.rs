//! Channels-last FP16 glue of the 2D body: the residual skip's average pool and the decoder's
//! transposed-conv scatter plus skip concatenation

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{pack_f16, r16, unpack_f16};

/// 2x2 stride-2 average pool, `[B, H, W, C]` to `[B, H/2, W/2, C]`, FP16 pairs
///
/// PyTorch's channels-last FP16 `avg_pool2d` sums the window in float32 row by row, divides
/// by 4 and rounds once. `out_h`/`out_w` are the output size and `cols` the channel count
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn avgpool2(mut out: DisjointSlice<u32>, x: &[u32], out_h: u32, out_w: u32, cols: u32) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let pairs = cols / 2;
    let (pix, cp) = (e / pairs, e % pairs);
    let (ox, rest) = (pix % out_w, pix / out_w);
    let (oy, b) = (rest % out_h, rest / out_h);
    let Some(mut out) = out.element_thread32(t) else { return };
    let (h, w) = (2 * out_h, 2 * out_w);
    let row = ((b * h + 2 * oy) * w + 2 * ox) * pairs + cp;
    let (a0, a1) = unpack_f16(x[row as usize]);
    let (b0, b1) = unpack_f16(x[(row + pairs) as usize]);
    let (c0, c1) = unpack_f16(x[(row + w * pairs) as usize]);
    let (d0, d1) = unpack_f16(x[(row + w * pairs + pairs) as usize]);
    out.write(pack_f16((a0 + b0 + c0 + d0) / 4.0, (a1 + b1 + c1 + d1) / 4.0));
}

/// The decoder's `cat(transpconv(x) + bias, skip)` at the upsampled size
///
/// `g` is the transposed conv as one GEMM, `[B * H/2 * W/2, 4 * up]` FP16 with column
/// `(dy * 2 + dx) * up + c` for output pixel `(2i + dy, 2j + dx)`; `bias` its FP16-rounded bias;
/// `skip` the encoder's `[B, H, W, skip_cols]`. `out` is `[B, H, W, up + skip_cols]`. The FP16
/// bias add rounds once, as PyTorch adds the bias to the FP16 conv output
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn cat_up(
    mut out: DisjointSlice<u32>,
    g: &[u32],
    bias: &[f32],
    skip: &[u32],
    height: u32,
    width: u32,
    up: u32,
    skip_cols: u32,
) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let pairs = (up + skip_cols) / 2;
    let (pix, cp) = (e / pairs, e % pairs);
    let (x, rest) = (pix % width, pix / width);
    let (y, b) = (rest % height, rest / height);
    let Some(mut out) = out.element_thread32(t) else { return };
    let c = 2 * cp;
    if c < up {
        let m = (b * (height / 2) + y / 2) * (width / 2) + x / 2;
        let d = (y & 1) * 2 + (x & 1);
        let (v0, v1) = unpack_f16(g[((m * 4 * up + d * up + c) / 2) as usize]);
        out.write(pack_f16(r16(v0) + bias[c as usize], r16(v1) + bias[c as usize + 1]));
    } else {
        out.write(skip[(pix * (skip_cols / 2) + (c - up) / 2) as usize]);
    }
}
