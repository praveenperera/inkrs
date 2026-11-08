// Hecate 9.6 um on Metal: the MSL ports of inkrs-kernels/src/hecate.rs plus Hecate's convolutions, which cuDNN runs
// on CUDA. The reference is hecate.py at huggingface.co/scrollprize/hecate 9cb86e5 (MIT)
//
// Activations are channels-last (NDHWC) in T, float (`--precision fp32`) or bfloat (bf16 autocast). Under autocast
// the convs write bf16, and BatchNorm, ReLU and the encoder's residual adds round to bf16 again, while GroupNorm and
// everything after it in the decoder stay float32 until the next conv casts its input. These kernels keep those
// roundings: `rt<T>` rounds a float through T (round to nearest even), and FP contraction is off for the whole
// library (glue.metal's pragma), so products and sums round separately as on CUDA
//
// The convs fuse the eval BatchNorm (a per-channel scale and shift), the residual add and the ReLU into their
// epilogue, which CUDA runs as a separate `hec_affine` pass with the same arithmetic

constant uint HEC_P = 64;
constant uint HEC_PIX = HEC_P * HEC_P;
constant uint HEC_D = 16;
constant uint HEC_HEAD_C = 64;

template <typename T> static inline float rt(float x) { return float(T(x)); }

// conv epilogue flags
constant uint EPI_AFFINE = 1;    // y = x * ab[c].x + ab[c].y
constant uint EPI_RESIDUAL = 2;  // y += residual
constant uint EPI_RELU = 4;      // y = max(y, 0), keeping NaN
constant uint EPI_ROUND_MID = 8; // round the affine result to T before the add (autocast's BatchNorm writes bf16)
constant uint EPI_PAD = 16;      // write the output into hec_pad's zero-padded layout, for a tap conv to read

// Row of output voxel `row` of [B, d, h, w] in the padded layout [B, d + 2, h + 2, w + 2]. The host zeroes a padded
// buffer once and only interiors are ever written, so its borders stay zero
static inline ulong hec_padded_row(uint row, uint d, uint h, uint w) {
    uint x = row % w, r = row / w;
    uint y = r % h;
    r /= h;
    uint z = r % d, b = r / d;
    return (((ulong)b * (d + 2) + z + 1) * (h + 2) + y + 1) * (w + 2) + x + 1;
}

// The epilogue of one output element: the conv result rounds to T as cuDNN writes it, then hec_affine's arithmetic
template <typename T>
static inline T hec_epi(float acc, uint col, device const float2* ab, device const T* res, ulong o, uint flags) {
    float v = rt<T>(acc);
    if (flags & EPI_AFFINE) {
        float2 s = ab[col];
        v = v * s.x + s.y;
        if (flags & EPI_ROUND_MID) v = rt<T>(v);
    }
    if (flags & EPI_RESIDUAL) v = v + float(res[o]);
    // `<=` keeps a NaN, as PyTorch's ReLU does, so it reaches the finite check
    if ((flags & EPI_RELU) && v <= 0.0f) v = 0.0f;
    return T(v);
}

// y[m, n] = epi(x[m, k] . w[n, k]^T), x with leading dimension ldx: the 1x1 convs (and the stride-2 3x3 convs on an
// im2col matrix). With EPI_PAD the m rows are the voxels of [B, pd, ph, pw] and y is their padded layout. Grid (n
// tiles, m tiles), 128 threads
template <typename T, int BM, int BN>
kernel void hec_gemm(device T* p_x [[buffer(0)]], device T* p_w [[buffer(1)]], device T* p_y [[buffer(2)]],
                     device const float2* p_ab [[buffer(3)]], device const T* p_res [[buffer(4)]],
                     constant uint& m [[buffer(5)]], constant uint& n [[buffer(6)]], constant uint& k [[buffer(7)]],
                     constant uint& ldx [[buffer(8)]], constant uint& flags [[buffer(9)]],
                     constant uint& pd [[buffer(10)]], constant uint& ph [[buffer(11)]], constant uint& pw [[buffer(12)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
    int m0 = tg.y * BM, n0 = tg.x * BN;
    mat<T> tx(p_x, dextents<int32_t, 2>(k, m), array<int, 2>{1, (int)ldx});
    mat<T> tw(p_w, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    // a thread's elements share few rows, so the padded row (four integer divisions) is cached across them
    int last = -1;
    ulong prow = 0;
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], row = m0 + idx[1];
        if (row < (int)m && col < (int)n) {
            ulong o = (ulong)row * n + col;
            if ((flags & EPI_PAD) && row != last) {
                prow = hec_padded_row(row, pd, ph, pw);
                last = row;
            }
            ulong out = (flags & EPI_PAD) ? prow * n + col : o;
            p_y[out] = hec_epi<T>(acc[i], col, p_ab, p_res, o, flags);
        }
    }
}

// 3x3x3 conv, stride 1, pad 1, channels-last, over the zero-padded batch of hec_pad ([B, d + 2, h + 2, w + 2, c] plus
// tail rows): output row r of the padded volume reads padded rows r + dz plane + dy (w + 2) + dx. The three dx taps
// of each (dz, dy) are consecutive rows, and the weights [o, 27, c] hold them contiguously, so a K = 3c view with row
// stride c covers all three: 9 matmuls per tile
//
// Output (b, z, y, x) sits at row b per + z plane + y (w + 2) + x, so the outputs of one (b, z) are the h (w + 2)
// consecutive rows from b per + z plane, all kept but the 2 x-padding rows of each line. The grid tiles each of those
// row ranges separately rather than every padded row, which skips the z- and y-padding rows: up to 3x the kept work in
// the 2 x 8 x 8 layer. Grid (o tiles, B d tiles_per_plane), 128 threads
template <typename T, int BM, int BN>
kernel void hec_taps(device T* p_xp [[buffer(0)]], device T* p_w [[buffer(1)]], device T* p_y [[buffer(2)]],
                     device const float2* p_ab [[buffer(3)]], device const T* p_res [[buffer(4)]],
                     constant uint& batch [[buffer(5)]], constant uint& d [[buffer(6)]], constant uint& h [[buffer(7)]],
                     constant uint& w [[buffer(8)]], constant uint& c [[buffer(9)]], constant uint& o [[buffer(10)]],
                     constant uint& flags [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]]) {
    int wp = w + 2, plane = (h + 2) * wp, per = (d + 2) * plane;
    int rows = batch * per, xrows = rows + 2 * plane + 2 * wp + 2;
    int span = h * wp, per_plane = (span + BM - 1) / BM;
    int bz = tg.y / per_plane, chunk = tg.y % per_plane;
    // first row of output plane (b, z), then this tile's rows inside its span
    int base = (bz / d) * per + (bz % d) * plane;
    int m0 = base + chunk * BM, n0 = tg.x * BN;
    int end = base + span;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    // the last two rows would read past the buffer and are left out of the extent; no kept output reaches them
    mat<T> tx(p_xp, dextents<int32_t, 2>(3 * c, xrows - 2), array<int, 2>{1, (int)c});
    mat<T> tw0(p_w, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 27 * (int)c});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int shift = (t / 3) * plane + (t % 3) * wp;
        mat<T> tw(p_w + t * 3 * c, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 27 * (int)c});
        auto mx = tx.slice(0, m0 + shift);
        auto mw = tw.slice(0, n0);
        op.run(mx, mw, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], r = m0 + idx[1];
        int b = r / per, rem = r % per;
        int z = rem / plane, yx = rem % plane;
        int y = yx / wp, x = yx % wp;
        if (r < end && r < rows && z < (int)d && y < (int)h && x < (int)w && col < (int)o) {
            ulong oi = ((((ulong)b * d + z) * h + y) * w + x) * o + col;
            p_y[oi] = hec_epi<T>(acc[i], col, p_ab, p_res, oi, flags);
        }
    }
}

#define HEC_CONVS(T, NAME)                                                                                           \
    template [[host_name("hec_gemm_" NAME)]] kernel void hec_gemm<T, 64, 64>(                                       \
        device T*, device T*, device T*, device const float2*, device const T*, constant uint&, constant uint&,      \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2);      \
    template [[host_name("hec_taps_" NAME)]] kernel void hec_taps<T, 64, 64>(                                       \
        device T*, device T*, device T*, device const float2*, device const T*, constant uint&, constant uint&,      \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2); \
    template [[host_name("hec_taps32_" NAME)]] kernel void hec_taps<T, 32, 64>(                                       \
        device T*, device T*, device T*, device const float2*, device const T*, constant uint&, constant uint&,      \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2);
HEC_CONVS(float, "f32")
HEC_CONVS(bfloat, "bf16")

// Zero-pads x [B, d, h, w, c] into xp [B, d + 2, h + 2, w + 2, c] followed by zero tail rows up to `count` elements;
// one thread per element of xp
template <typename T>
kernel void hec_pad(device T* p_xp [[buffer(0)]], device const T* p_x [[buffer(1)]], constant uint& batch [[buffer(2)]],
                    constant uint& d [[buffer(3)]], constant uint& h [[buffer(4)]], constant uint& w [[buffer(5)]],
                    constant uint& c [[buffer(6)]], constant uint& count [[buffer(7)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint wp = w + 2, plane = (h + 2) * wp, per = (d + 2) * plane;
    uint ch = i % c, r = i / c;
    uint b = r / per, rem = r % per;
    uint z = rem / plane, yx = rem % plane;
    uint y = yx / wp, x = yx % wp;
    bool inside = b < batch && z >= 1 && z <= d && y >= 1 && y <= h && x >= 1 && x <= w;
    p_xp[i] = inside ? p_x[((((ulong)b * d + z - 1) * h + y - 1) * w + x - 1) * c + ch] : T(0.0f);
}

// im2col of a 3x3x3 stride-2 pad-1 conv: x [B, d, h, w, c] to col [B * d/2 * h/2 * w/2, 27c], column t * c + ch for
// tap t = (dz * 3 + dy) * 3 + dx, so a GEMM against the [o, 27, c] weights is the conv; one thread per element
template <typename T>
kernel void hec_im2col_s2(device T* p_col [[buffer(0)]], device const T* p_x [[buffer(1)]],
                          constant uint& d [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                          constant uint& c [[buffer(5)]], constant uint& count [[buffer(6)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint dout = d / 2, hout = h / 2, wout = w / 2;
    uint ch = i % c, rest = i / c;
    uint t = rest % 27, pix = rest / 27;
    uint ox = pix % wout, r = pix / wout;
    uint oy = r % hout;
    r /= hout;
    uint oz = r % dout, b = r / dout;
    int z = int(2 * oz + t / 9) - 1, y = int(2 * oy + (t / 3) % 3) - 1, x = int(2 * ox + t % 3) - 1;
    T v = T(0.0f);
    if (z >= 0 && z < int(d) && y >= 0 && y < int(h) && x >= 0 && x < int(w))
        v = p_x[((((ulong)b * d + uint(z)) * h + uint(y)) * w + uint(x)) * c + ch];
    p_col[i] = v;
}

// The stride-2 1x1 conv's input: x [B, d, h, w, c] sampled at even (z, y, x) into [B, d/2, h/2, w/2, c]
template <typename T>
kernel void hec_sub2(device T* p_y [[buffer(0)]], device const T* p_x [[buffer(1)]], constant uint& d [[buffer(2)]],
                     constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]], constant uint& c [[buffer(5)]],
                     constant uint& count [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint dout = d / 2, hout = h / 2, wout = w / 2;
    uint ch = i % c, pix = i / c;
    uint x = pix % wout, r = pix / wout;
    uint y = r % hout;
    r /= hout;
    uint z = r % dout, b = r / dout;
    p_y[i] = p_x[((((ulong)b * d + 2 * z) * h + 2 * y) * w + 2 * x) * c + ch];
}

// The 16-plane input of every tile, normalised: `uint8 / 255` in float32, then the eval BatchNorm3d(1) as
// `((x - mean) * invstd) * gamma + beta`, plus each column's support (any nonzero voxel in its 16 planes). One thread
// per (tile, pixel); x is [B, 16, 64, 64] in T. Pixels past the volume's edge read 0, as hecate.py zero-pads a short
// tile in uint8 before it normalises
template <typename T>
kernel void hec_gather(device T* p_x [[buffer(0)]], device uchar* p_support [[buffer(1)]],
                       device const uchar* p_vol [[buffer(2)]], device const int* p_desc [[buffer(3)]],
                       constant uint& height [[buffer(4)]], constant uint& width [[buffer(5)]],
                       constant float& mean [[buffer(6)]], constant float& invstd [[buffer(7)]],
                       constant float& gamma [[buffer(8)]], constant float& beta [[buffer(9)]],
                       constant uint& count [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint b = i / HEC_PIX, p = i % HEC_PIX;
    uint d = b * DESC;
    uint y = uint(p_desc[d]) + p / HEC_P, x = uint(p_desc[d + 1]) + p % HEC_P;
    int first = p_desc[d + 2], step = p_desc[d + 3];
    bool inside = y < height && x < width;
    bool any = false;
    for (uint z = 0; z < HEC_D; z++) {
        uchar v = 0;
        if (inside) {
            uint layer = uint(first + step * int(z));
            v = p_vol[((ulong)layer * height + y) * width + x];
        }
        any = any || v != 0;
        float norm = float(v) / 255.0f;
        float val = (norm - mean) * invstd * gamma + beta;
        p_x[((ulong)b * HEC_D + z) * HEC_PIX + p] = T(val);
    }
    p_support[i] = any ? 1 : 0;
}

// The stem conv3d 1 -> 64, 3x3x3, pad 1, on x [B, 16, 64, 64], with its BatchNorm and ReLU: one thread per (voxel,
// group of 16 output channels), FP32 accumulation, the conv result rounded to T. Weights [64, 27] already rounded to
// T on the host
template <typename T>
kernel void hec_stem(device T* p_y [[buffer(0)]], device const T* p_x [[buffer(1)]], device const float* p_w [[buffer(2)]],
                     device const float2* p_ab [[buffer(3)]], constant uint& flags [[buffer(4)]],
                     constant uint& count [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint g = i % 4, v = i / 4;
    uint x = v % HEC_P, r = v / HEC_P;
    uint y = r % HEC_P;
    r /= HEC_P;
    uint z = r % HEC_D, b = r / HEC_D;
    float acc[16];
    for (uint k = 0; k < 16; k++) acc[k] = 0.0f;
    device const T* xb = p_x + (ulong)b * HEC_D * HEC_PIX;
    for (uint t = 0; t < 27; t++) {
        int zz = int(z) + int(t / 9) - 1, yy = int(y) + int((t / 3) % 3) - 1, xx = int(x) + int(t % 3) - 1;
        if (zz < 0 || zz >= int(HEC_D) || yy < 0 || yy >= int(HEC_P) || xx < 0 || xx >= int(HEC_P)) continue;
        float xv = float(xb[(zz * HEC_P + yy) * HEC_P + xx]);
        for (uint k = 0; k < 16; k++) acc[k] = fma(xv, p_w[(g * 16 + k) * 27 + t], acc[k]);
    }
    ulong o = (ulong)v * 64 + g * 16;
    for (uint k = 0; k < 16; k++) p_y[o + k] = hec_epi<T>(acc[k], g * 16 + k, p_ab, p_y, o + k, flags);
}

#define HEC_PLAIN(T, NAME)                                                                                           \
    template [[host_name("hec_pad_" NAME)]] kernel void hec_pad<T>(device T*, device const T*, constant uint&,      \
                                                                    constant uint&, constant uint&, constant uint&, \
                                                                    constant uint&, constant uint&, uint);          \
    template [[host_name("hec_im2col_s2_" NAME)]] kernel void hec_im2col_s2<T>(                                     \
        device T*, device const T*, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,  \
        uint);                                                                                                       \
    template [[host_name("hec_sub2_" NAME)]] kernel void hec_sub2<T>(device T*, device const T*, constant uint&,    \
                                                                      constant uint&, constant uint&,               \
                                                                      constant uint&, constant uint&, uint);        \
    template [[host_name("hec_gather_" NAME)]] kernel void hec_gather<T>(                                           \
        device T*, device uchar*, device const uchar*, device const int*, constant uint&, constant uint&,           \
        constant float&, constant float&, constant float&, constant float&, constant uint&, uint);                  \
    template [[host_name("hec_stem_" NAME)]] kernel void hec_stem<T>(device T*, device const T*, device const float*, \
                                                                      device const float2*, constant uint&,         \
                                                                      constant uint&, uint);
HEC_PLAIN(float, "f32")
HEC_PLAIN(bfloat, "bf16")

// ---------------------------------------------------------------- GroupNorm

// GroupNorm's partial sums: per (sample, chunk of rows, channel), [sum(x - x0), sum((x - x0)^2)] around the
// channel's first value x0 = x[b, 0, c], so the sums don't cancel. x is [B, rows, cols]. Grid (chunks, B, cols / cpb)
// with cpb = min(cols, 256) channels per threadgroup and 256 / cpb lanes per channel
template <typename T>
kernel void hec_gn_partial(device float2* p_partial [[buffer(0)]], device const T* p_x [[buffer(1)]],
                           constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                           constant uint& cpb [[buffer(4)]], constant uint& chunk_rows [[buffer(5)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]],
                           uint3 tcount3 [[threads_per_threadgroup]]) {
    threadgroup float2 acc[256];
    uint tid = tid3.x, tcount = tcount3.x;
    uint chunk = tg.x, b = tg.y, cb = tg.z;
    // threadgroup-uniform, so a group that fails it never reaches a barrier
    if (cpb == 0 || tcount > 256 || tcount % cpb != 0 || (cb + 1) * cpb > cols) return;
    uint chunks = (rows + chunk_rows - 1) / chunk_rows;
    uint lanes = tcount / cpb;
    uint cl = tid % cpb, lane = tid / cpb;
    uint c = cb * cpb + cl;
    ulong sample = (ulong)b * rows * cols;
    float x0 = float(p_x[sample + c]);
    float s = 0.0f, q = 0.0f;
    uint end = min((chunk + 1) * chunk_rows, rows);
    for (uint r = chunk * chunk_rows + lane; r < end; r += lanes) {
        float dv = float(p_x[sample + (ulong)r * cols + c]) - x0;
        s += dv;
        q += dv * dv;
    }
    acc[tid] = float2(s, q);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float ts = 0.0f, tq = 0.0f;
        for (uint l = 0; l < lanes; l++) {
            float2 p = acc[l * cpb + cl];
            ts += p.x;
            tq += p.y;
        }
        p_partial[((ulong)b * chunks + chunk) * cols + c] = float2(ts, tq);
    }
}

// double-float helpers for gn_finalize's float64 sums (Metal has no float64)
static inline float2 df_two_prod(float a, float b) {
    float p = a * b;
    return float2(p, fma(a, b, -p));
}

static inline float2 df_add2(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);
    float lo = s.y + a.y + b.y;
    return two_sum(s.x, lo);
}

static inline float2 df_mulf(float2 a, float b) {
    float2 p = df_two_prod(a.x, b);
    p.y = fma(a.y, b, p.y);
    return two_sum(p.x, p.y);
}

// full double-float product, including the lo * lo term that matters when hi is small
static inline float2 df_mul(float2 a, float2 b) {
    float2 p = df_two_prod(a.x, b.x);
    p.y = fma(a.x, b.y, fma(a.y, b.x, fma(a.y, b.y, p.y)));
    return two_sum(p.x, p.y);
}

static inline float2 df_div(float2 a, float b) {
    float hi = a.x / b;
    float2 p = df_two_prod(hi, b);
    float lo = ((a.x - p.x) - p.y + a.y) / b;
    return two_sum(hi, lo);
}

// GroupNorm(groups) statistics to the epilogue's scale and shift, one thread per (sample, channel): the group's
// partial sums combined in double-float as CUDA does in float64, then as PyTorch's GroupNorm: rstd = rsqrt(var + eps)
// with the biased variance, a = rstd * gamma[c], b = beta[c] - a * mean
//
// Each channel's partials are centred on that channel's first value; they are re-centred on one shift for the whole
// group (its first channel's) before the moments combine. Rebuilding raw moments instead (n shift^2) overflows the
// float exponent range for large finite inputs, which float64 on CUDA does not
template <typename T>
kernel void hec_gn_finalize(device float2* p_ab [[buffer(0)]], device const float2* p_partial [[buffer(1)]],
                            device const T* p_x [[buffer(2)]], device const float* p_gamma [[buffer(3)]],
                            device const float* p_beta [[buffer(4)]], constant uint& rows [[buffer(5)]],
                            constant uint& cols [[buffer(6)]], constant uint& chunks [[buffer(7)]],
                            constant uint& groups [[buffer(8)]], constant float& eps [[buffer(9)]],
                            constant uint& count [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint b = i / cols, c = i % cols;
    uint cg = cols / max(groups, 1u);
    uint g0 = (c / cg) * cg;
    float n = float(rows);
    float2 sum = 0, sq = 0;
    float g_shift = float(p_x[(ulong)b * rows * cols + g0]);
    for (uint cc = g0; cc < g0 + cg; cc++) {
        float shift = float(p_x[(ulong)b * rows * cols + cc]);
        float2 s = 0, q = 0;
        for (uint k = 0; k < chunks; k++) {
            float2 p = p_partial[((ulong)b * chunks + k) * cols + cc];
            s = df_add(s, p.x);
            q = df_add(q, p.y);
        }
        // with d = shift - g_shift (exact as a double-float): sum += s + n d; sq += q + 2 d s + n d^2
        float2 d = two_sum(shift, -g_shift);
        float2 nd = df_mulf(d, n);
        sum = df_add2(sum, df_add2(s, nd));
        sq = df_add2(sq, df_add2(q, df_add2(df_mul(s, df_mulf(d, 2.0f)), df_mul(nd, d))));
    }
    float total = n * float(cg);
    // mean about g_shift, so mean^2 stays in range
    float2 mean_c = df_div(sum, total);
    float2 m2 = df_mul(mean_c, mean_c);
    float2 var2 = df_add2(df_div(sq, total), float2(-m2.x, -m2.y));
    float2 mean = df_add2(float2(g_shift, 0.0f), mean_c);
    float var = var2.x + var2.y;
    // `<` keeps a NaN variance, so a NaN input stays NaN through the norm
    var = var < 0.0f ? 0.0f : var;
    float rstd = rsqrt(var + eps);
    float a = rstd * p_gamma[c];
    float m = mean.x + mean.y;
    p_ab[i] = float2(a, p_beta[c] - a * m);
}

// GroupNorm's apply, as hec_affine with per-sample scales: y = relu?(x * a + b + residual?), out of place, x and the
// residual in T (conv outputs), y in U (float32, or T where the next conv reads it). With EPI_PAD, x is [B, d, h, w,
// cols] and y its padded layout
template <typename T, typename U>
kernel void hec_gn_apply(device U* p_y [[buffer(0)]], device const T* p_x [[buffer(1)]],
                         device const T* p_res [[buffer(2)]], device const float2* p_ab [[buffer(3)]],
                         constant uint& cols [[buffer(4)]], constant uint& per_sample [[buffer(5)]],
                         constant uint& flags [[buffer(6)]], constant uint& count [[buffer(7)]],
                         constant uint& d [[buffer(8)]], constant uint& h [[buffer(9)]], constant uint& w [[buffer(10)]],
                         uint e [[thread_position_in_grid]]) {
    if (e >= count) return;
    uint c = e % cols;
    uint k = (e / per_sample) * cols + c;
    float2 s = p_ab[k];
    float v = float(p_x[e]) * s.x + s.y;
    if (flags & EPI_RESIDUAL) v = v + float(p_res[e]);
    if ((flags & EPI_RELU) && v <= 0.0f) v = 0.0f;
    ulong out = (flags & EPI_PAD) ? hec_padded_row(e / cols, d, h, w) * cols + c : e;
    p_y[out] = U(v);
}

#define HEC_GN(T, NAME)                                                                                              \
    template [[host_name("hec_gn_partial_" NAME)]] kernel void hec_gn_partial<T>(                                   \
        device float2*, device const T*, constant uint&, constant uint&, constant uint&, constant uint&, uint3,      \
        uint3, uint3);                                                                                               \
    template [[host_name("hec_gn_finalize_" NAME)]] kernel void hec_gn_finalize<T>(                                 \
        device float2*, device const float2*, device const T*, device const float*, device const float*,            \
        constant uint&, constant uint&, constant uint&, constant uint&, constant float&, constant uint&, uint);      \
    template [[host_name("hec_gn_apply_" NAME "_f32")]] kernel void hec_gn_apply<T, float>(                         \
        device float*, device const T*, device const T*, device const float2*, constant uint&, constant uint&,      \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint);
HEC_GN(float, "f32")
HEC_GN(bfloat, "bf16")
// at fp32 the activation type is float32, so hec_gn_apply_f32_f32 above serves both outputs
template [[host_name("hec_gn_apply_bf16_bf16")]] kernel void hec_gn_apply<bfloat, bfloat>(
    device bfloat*, device const bfloat*, device const bfloat*, device const float2*, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint);

// ---------------------------------------------------------------- decoder upsample, head and blend

// PyTorch's align_corners=False source of output index o at a 2x upsample of an axis of `len`: the lower index, the
// offset to its neighbour (0 at the edge) and the two weights
static inline void up_src(uint o, uint len, thread uint& i0, thread uint& nb, thread float& w0, thread float& w1) {
    float r = 0.5f * (float(o) + 0.5f) - 0.5f;
    r = r > 0.0f ? r : 0.0f;
    i0 = uint(r);
    float l1 = r - float(i0);
    nb = i0 + 1 < len ? 1 : 0;
    w0 = 1.0f - l1;
    w1 = l1;
}

// cat([trilinear_x2(low), skip]) channels last, PyTorch's upsample_trilinear3d with align_corners=False at an exact
// 2x size. low [B, d, h, w, cl] and skip [B, 2d, 2h, 2w, cs] are float32; the output [B, 2d, 2h, 2w, cl + cs] is T,
// what the next conv casts it to. PyTorch's nesting of the eight products
template <typename T>
kernel void hec_up2cat(device T* p_y [[buffer(0)]], device const float* p_low [[buffer(1)]],
                       device const float* p_skip [[buffer(2)]], constant uint& d [[buffer(3)]],
                       constant uint& h [[buffer(4)]], constant uint& w [[buffer(5)]], constant uint& cl [[buffer(6)]],
                       constant uint& cs [[buffer(7)]], constant uint& count [[buffer(8)]],
                       uint e [[thread_position_in_grid]]) {
    if (e >= count) return;
    uint cols = cl + cs;
    uint c = e % cols, vox = e / cols;
    uint ow = 2 * w, oh = 2 * h, od = 2 * d;
    uint x = vox % ow, y = (vox / ow) % oh, z = (vox / (ow * oh)) % od, b = vox / (ow * oh * od);
    float v;
    if (c < cl) {
        uint z1, zp, y1, yp, x1, xp;
        float t0, t1, h0, h1, w0, w1;
        up_src(z, d, z1, zp, t0, t1);
        up_src(y, h, y1, yp, h0, h1);
        up_src(x, w, x1, xp, w0, w1);
        auto at = [&](uint zz, uint yy, uint xx) { return p_low[((((ulong)b * d + zz) * h + yy) * w + xx) * cl + c]; };
        float a00 = w0 * at(z1, y1, x1) + w1 * at(z1, y1, x1 + xp);
        float a01 = w0 * at(z1, y1 + yp, x1) + w1 * at(z1, y1 + yp, x1 + xp);
        float a10 = w0 * at(z1 + zp, y1, x1) + w1 * at(z1 + zp, y1, x1 + xp);
        float a11 = w0 * at(z1 + zp, y1 + yp, x1) + w1 * at(z1 + zp, y1 + yp, x1 + xp);
        v = t0 * (h0 * a00 + h1 * a01) + t1 * (h0 * a10 + h1 * a11);
    } else {
        v = p_skip[(ulong)vox * cs + c - cl];
    }
    p_y[e] = T(v);
}

template [[host_name("hec_up2cat_f32")]] kernel void hec_up2cat<float>(device float*, device const float*,
                                                                        device const float*, constant uint&,
                                                                        constant uint&, constant uint&, constant uint&,
                                                                        constant uint&, constant uint&, uint);
template [[host_name("hec_up2cat_bf16")]] kernel void hec_up2cat<bfloat>(device bfloat*, device const float*,
                                                                          device const float*, constant uint&,
                                                                          constant uint&, constant uint&,
                                                                          constant uint&, constant uint&,
                                                                          constant uint&, uint);

static inline float rbf_if(bool on, float x) { return on ? rt<bfloat>(x) : x; }

// The head: per column, the logit and attention 1x1 convs over the 16 depths, the masked depth softmax with the
// depth coordinate, the weighted logit sum and hecate.py's sigmoid(logit(clamp(sigmoid(.), eps, 1 - eps))), times
// the column's support. One thread per (tile, pixel); feat is the last decoder stage [B, 16, 64, 64, 64] float32.
// With bf16 the two convs run as autocast runs them. prob[slot, pixel] gets the probability, 0 off the support, NaN
// off the support when a logit is not finite (hecate.py's zero weights times the logits); padding tiles write nothing
kernel void hec_head(device float* p_prob [[buffer(0)]], device const float* p_feat [[buffer(1)]],
                     device const uchar* p_support [[buffer(2)]], device const float* p_wl [[buffer(3)]],
                     constant float& bl [[buffer(4)]], device const float* p_wa [[buffer(5)]],
                     constant float& ba [[buffer(6)]], constant float& dcs [[buffer(7)]],
                     device const int* p_desc [[buffer(8)]], constant uint& bf16 [[buffer(9)]],
                     constant uint& count [[buffer(10)]], constant uint& slots [[buffer(11)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint b = i / HEC_PIX, p = i % HEC_PIX;
    int slot = p_desc[b * DESC + 4];
    if (slot < 0 || uint(slot) >= slots) return;
    bool on = bf16 != 0;
    float logit[16], score[16];
    for (uint z = 0; z < HEC_D; z++) {
        ulong base = (((ulong)b * HEC_D + z) * HEC_PIX + p) * HEC_HEAD_C;
        float l = 0.0f, a = 0.0f;
        for (uint c = 0; c < HEC_HEAD_C; c++) {
            float f = rbf_if(on, p_feat[base + c]);
            l += f * rbf_if(on, p_wl[c]);
            a += f * rbf_if(on, p_wa[c]);
        }
        logit[z] = rbf_if(on, rbf_if(on, l) + rbf_if(on, bl));
        // the depth coordinate (z - 8) / 8 is exact in float32
        score[z] = rbf_if(on, rbf_if(on, a) + rbf_if(on, ba)) + dcs * ((float(z) - 8.0f) / 8.0f);
    }
    float value;
    if (p_support[i] == 0) {
        float zero = 0.0f;
        for (uint k = 0; k < HEC_D; k++) zero += 0.0f * logit[k];
        // -0 (all logits negative) becomes 0, NaN stays NaN
        value = zero == 0.0f ? 0.0f : zero;
    } else {
        float m = score[0];
        for (uint k = 1; k < HEC_D; k++) m = fmax(m, score[k]);
        float sum = 0.0f, e[16];
        for (uint k = 0; k < HEC_D; k++) {
            e[k] = exp(score[k] - m);
            sum += e[k];
        }
        // softmax, then hecate.py's renormalisation by the weights' own sum (clamped at 1e-8)
        float wsum = 0.0f;
        for (uint k = 0; k < HEC_D; k++) {
            e[k] /= sum;
            wsum += e[k];
        }
        // clamps by comparison, not fmax/fmin, so a NaN passes through as in torch.clamp
        wsum = wsum < 1e-8f ? 1e-8f : wsum;
        float acc = 0.0f;
        for (uint k = 0; k < HEC_D; k++) acc += (e[k] / wsum) * logit[k];
        float eps = 1.1920929e-7f;
        float pr = 1.0f / (1.0f + exp(-acc));
        float pc = pr < eps ? eps : (pr > 1.0f - eps ? 1.0f - eps : pr);
        float lg = log(pc / (1.0f - pc));
        value = 1.0f / (1.0f + exp(-lg));
    }
    p_prob[(ulong)slot * HEC_PIX + p] = value;
}

// One map: rint(255 * clip(sum(p * w) / sum(w), 0, 1)) per pixel, as hecate.py finishes. grid [ny, nx] holds the slot
// of the tile at each position, or -1 for an empty tile, which still adds its window to the denominator. Each pixel
// visits its tiles in hecate.py's order with a separate multiply and add, as numpy's float32 `+=` does
kernel void hec_blend(device uchar* p_out [[buffer(0)]], device const float* p_prob [[buffer(1)]],
                      device const int* p_grid [[buffer(2)]], device const uint* p_ys [[buffer(3)]],
                      device const uint* p_xs [[buffer(4)]], device const uint2* p_y_span [[buffer(5)]],
                      device const uint2* p_x_span [[buffer(6)]], device const float* p_weight [[buffer(7)]],
                      constant uint& width [[buffer(8)]], constant uint& nx [[buffer(9)]],
                      constant uint& count [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint y = i / width, x = i % width;
    uint2 ysp = p_y_span[y], xsp = p_x_span[x];
    float num = 0.0f, den = 0.0f;
    for (uint iy = ysp.x; iy < ysp.y; iy++) {
        uint ly = y - p_ys[iy];
        for (uint ix = xsp.x; ix < xsp.y; ix++) {
            uint lx = x - p_xs[ix];
            float w = p_weight[ly * HEC_P + lx];
            int slot = p_grid[iy * nx + ix];
            if (slot >= 0) {
                float pw = p_prob[(ulong)slot * HEC_PIX + ly * HEC_P + lx] * w;
                num = num + pw;
            }
            den = den + w;
        }
    }
    // the host has already refused non-finite probabilities; the comparison clamp still keeps a NaN
    float q = num / den;
    float pr = q < 0.0f ? 0.0f : (q > 1.0f ? 1.0f : q);
    p_out[i] = uchar(rint(pr * 255.0f));
}
