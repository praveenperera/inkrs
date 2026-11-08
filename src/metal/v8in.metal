// v8-in on Metal: the MSL ports of inkrs-kernels/src/v8in.rs plus the convolutions cuDNN runs on CUDA
//
// Activations are FP16, channels-last. As on CUDA, every BatchNorm is folded on the host into its conv's weights and
// a bias, and a conv's epilogue computes `relu(conv + residual + bias)` in float32 and rounds once (`v8_epi`), as
// cuDNN's fused bias-activation does. FP contraction is off for the whole library (glue.metal's pragma), so sums and
// products round separately as in the CUDA kernels

constant uint V8_P = 64;
constant uint V8_PIX = V8_P * V8_P;
constant uint V8_D = 24;
constant uint V8_VOX = V8_D * V8_PIX;
constant uint V8_DESC = 8;
constant uint V8_CLIP = 200;
constant float V8_INV255 = 1.0f / 255.0f;
// the stem's output edge before the pool, its depth, the pooled edge and channels, the composed taps, a stem block's
// pooled patch edge, its stem rows with the pool's halo, and its source window edge
constant uint V8_SH = 128;
constant uint V8_SD = 96;
constant uint V8_PH = 64;
constant uint V8_SC = 64;
constant uint V8_TAPS = 64;
constant uint V8_PP = 8;
constant uint V8_SR = 2 * V8_PP + 1;
constant uint V8_WIN = 12;
// a stem threadgroup's channels: two threadgroups share a patch, so the stem outputs fit 32 KB of threadgroup memory
constant uint V8_STEM_CH = 32;
constant uint V8_STEM_THREADS = 4 * V8_STEM_CH;

// conv epilogue flags
constant uint V8_BIAS = 1;  // y += bias[col]
constant uint V8_RES = 2;   // y += residual, the plain-layout element of the same index
constant uint V8_RELU = 4;  // y = max(y, 0), keeping NaN
constant uint V8_PAD = 8;   // write into the zero-padded layout a tap conv reads (v8_gemm only)

// One output element's epilogue in float32, rounded to FP16 once
static inline half v8_epi(float acc, uint col, device const float* bias, device const half* res, ulong o,
                          uint flags) {
    float v = acc;
    if (flags & V8_RES) v = v + float(res[o]);
    if (flags & V8_BIAS) v = v + bias[col];
    // `<` keeps a NaN, as PyTorch's ReLU does, so it reaches the finite check
    if ((flags & V8_RELU) && v < 0.0f) v = 0.0f;
    return half(v);
}

// ---------------------------------------------------------------- input and stem

// Each tile's InstanceNorm3d(1, affine) as the scale and shift that map `x = min(v, 200) * float32(1/255)` to
// `(x - mean) * invstd * gamma + beta`; one threadgroup of 256 per tile, ab is [B]. The sums of v and v^2 are exact
// integers and so is `n sum(v^2) - sum(v)^2`, so the variance has no cancellation. CUDA finishes in float64, which
// Metal lacks; float32 moves the scale and shift by a few ulp. A padding tile still gets its statistics, so the
// network never sees NaN from it
kernel void v8_stats(device float2* p_ab [[buffer(0)]], device const uchar* p_vol [[buffer(1)]],
                     device const int* p_desc [[buffer(2)]], constant uint& height [[buffer(3)]],
                     constant uint& width [[buffer(4)]], constant float& gamma [[buffer(5)]],
                     constant float& beta [[buffer(6)]], uint b [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint2 sums[256];
    device const int* d = p_desc + b * V8_DESC;
    uint y0 = d[0], x0 = d[1];
    int first = d[2], step = d[3];
    uint s1 = 0, s2 = 0;
    for (uint i = tid; i < V8_VOX; i += 256) {
        uint z = i / V8_PIX, y = y0 + (i / V8_P) % V8_P, x = x0 + i % V8_P;
        ulong layer = (ulong)(first + step * (int)z);
        uint v = (y < height && x < width) ? min((uint)p_vol[(layer * height + y) * width + x], V8_CLIP) : 0;
        s1 += v;
        s2 += v * v;
    }
    sums[tid] = uint2(s1, s2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    ulong t1 = 0, t2 = 0;
    for (uint k = 0; k < 256; k++) {
        t1 += sums[k].x;
        t2 += sums[k].y;
    }
    ulong n = V8_VOX;
    float c = V8_INV255;
    float mean = c * float(t1) / float(n);
    float var = c * c * float(n * t2 - t1 * t1) / float(n * n);
    float scale = gamma / sqrt(var + 1e-5f);
    p_ab[b] = float2(scale, beta - mean * scale);
}

// Stem: InstanceNorm, the trilinear upsample to 96 x 256 x 256, conv3d 1 -> 64 (7^3, stride (1, 2, 2), pad 3),
// BatchNorm, ReLU and max pool (1, 3, 3) stride (1, 2, 2) pad (0, 1, 1) in one pass, as CUDA's v8_stem: each stem
// output reads a 4 x 4 x 4 window of the normalised source tile through its class's composed kernel (the host's
// StemTables, BatchNorm scale folded in; w is [class, tap, 64]). Neither the upsampled input nor the stem output is
// ever written
//
// Grid (2 x 64 patches, 96 depths, B), 128 threads. A pair of threadgroups owns one depth and an 8 x 8 patch of the
// pooled output, so 17 x 17 stem outputs with the pool's halo, and each computes 32 of the 64 channels: CUDA's block
// holds all 64 in 37 KB, past Metal's 32 KB. Thread t owns channel t % 32 of its half and the outputs whose row and
// column parities are t / 32; it holds its interior class's weights in registers and reads a border output's from
// device memory. out is [B, 96, 64, 64, 64]
kernel void v8_stem(device half* p_out [[buffer(0)]], device const uchar* p_vol [[buffer(1)]],
                    device const int* p_desc [[buffer(2)]], device const float2* p_ab [[buffer(3)]],
                    device const float* p_w [[buffer(4)]], device const float* p_bias [[buffer(5)]],
                    device const uint* p_dcls [[buffer(6)]], device const int* p_dbase [[buffer(7)]],
                    device const uint* p_hcls [[buffer(8)]], device const int* p_hbase [[buffer(9)]],
                    constant uint& height [[buffer(10)]], constant uint& width [[buffer(11)]],
                    constant uint& nh [[buffer(12)]], uint3 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float src[4 * V8_WIN * V8_WIN];
    threadgroup half act[V8_SR * V8_SR * V8_STEM_CH];
    uint patch = tg.x / 2, part = tg.x % 2, od = tg.y, b = tg.z;
    device const int* d = p_desc + b * V8_DESC;
    uint y0 = d[0], x0 = d[1];
    int first = d[2], step = d[3];
    float2 ab = p_ab[b];
    uint ph0 = (patch / (V8_PH / V8_PP)) * V8_PP, pw0 = (patch % (V8_PH / V8_PP)) * V8_PP;
    // stem rows and columns 2 p0 - 1 ..= 2 p0 + 15; the first can be -1
    int oh_lo = 2 * (int)ph0 - 1, ow_lo = 2 * (int)pw0 - 1;
    int r0 = p_hbase[max(oh_lo, 0)], c0 = p_hbase[max(ow_lo, 0)];
    int z0 = p_dbase[od];

    // the normalised source window, clamped into the tile: a clamped index only ever carries a zero weight
    for (uint i = tid; i < 4 * V8_WIN * V8_WIN; i += V8_STEM_THREADS) {
        int jd = i / (V8_WIN * V8_WIN), r = (i / V8_WIN) % V8_WIN, cc = i % V8_WIN;
        uint z = (uint)clamp(z0 + jd, 0, (int)V8_D - 1);
        uint y = y0 + (uint)clamp(r0 + r, 0, (int)V8_P - 1);
        uint x = x0 + (uint)clamp(c0 + cc, 0, (int)V8_P - 1);
        ulong layer = (ulong)(first + step * (int)z);
        uint v = (y < height && x < width) ? min((uint)p_vol[(layer * height + y) * width + x], V8_CLIP) : 0;
        float xv = float(v) * V8_INV255;
        src[i] = xv * ab.x + ab.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cl = tid % V8_STEM_CH, par = tid / V8_STEM_CH;
    uint ch = part * V8_STEM_CH + cl;
    uint rh = par / 2, rw = par % 2;
    uint cd = p_dcls[od];
    // the interior class of this thread's outputs: local row i is stem row 2 p0 - 1 + i, so local parity rh holds stem
    // rows of parity 1 - rh, whose interior representative is row 65 - rh (rows 64 and 65 are interior on both axes)
    uint mine = (cd * nh + p_hcls[65 - rh]) * nh + p_hcls[65 - rw];
    float wr[V8_TAPS];
    for (uint t = 0; t < V8_TAPS; t++) wr[t] = p_w[(mine * V8_TAPS + t) * V8_SC + ch];
    float bc = p_bias[ch];
    for (uint i = rh; i < V8_SR; i += 2) {
        int oh = oh_lo + (int)i;
        if (oh < 0 || oh >= (int)V8_SH) continue;
        uint hb = (uint)clamp(p_hbase[oh] - r0, 0, (int)V8_WIN - 4);
        uint h_cls = p_hcls[oh];
        for (uint j = rw; j < V8_SR; j += 2) {
            int ow = ow_lo + (int)j;
            if (ow < 0 || ow >= (int)V8_SH) continue;
            uint wb = (uint)clamp(p_hbase[ow] - c0, 0, (int)V8_WIN - 4);
            uint cls = (cd * nh + h_cls) * nh + p_hcls[ow];
            float acc = 0.0f;
            if (cls == mine) {
                for (uint t = 0; t < V8_TAPS; t++) {
                    uint jd = t / 16, jh = (t / 4) % 4, jw = t % 4;
                    acc += wr[t] * src[(jd * V8_WIN + hb + jh) * V8_WIN + wb + jw];
                }
            } else {
                device const float* wc = p_w + (ulong)cls * V8_TAPS * V8_SC + ch;
                for (uint t = 0; t < V8_TAPS; t++) {
                    uint jd = t / 16, jh = (t / 4) % 4, jw = t % 4;
                    acc += wc[t * V8_SC] * src[(jd * V8_WIN + hb + jh) * V8_WIN + wb + jw];
                }
            }
            float y = acc + bc;
            // ReLU; NaN propagates, as PyTorch's does
            if (y < 0.0f) y = 0.0f;
            act[(i * V8_SR + j) * V8_STEM_CH + cl] = half(y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // max pool: the window's stem rows 2p - 1 ..= 2p + 1 are local rows 2 (p - p0) ..= + 2
    for (uint k = tid; k < V8_PP * V8_PP * V8_STEM_CH; k += V8_STEM_THREADS) {
        uint c = k % V8_STEM_CH, pp = k / V8_STEM_CH;
        uint pi = pp / V8_PP, pj = pp % V8_PP;
        float m = -INFINITY;
        for (uint di = 0; di < 3; di++) {
            uint li = 2 * pi + di;
            int oh = oh_lo + (int)li;
            for (uint dj = 0; dj < 3; dj++) {
                uint lj = 2 * pj + dj;
                int ow = ow_lo + (int)lj;
                if (oh >= 0 && oh < (int)V8_SH && ow >= 0 && ow < (int)V8_SH) {
                    float v = float(act[(li * V8_SR + lj) * V8_STEM_CH + c]);
                    if (v > m || isnan(v)) m = v;
                }
            }
        }
        ulong o = ((((ulong)b * V8_SD + od) * V8_PH + ph0 + pi) * V8_PH + pw0 + pj) * V8_SC + part * V8_STEM_CH + c;
        p_out[o] = half(m);
    }
}

// ---------------------------------------------------------------- convolutions

// Row of voxel `row` of [B, d, h, w] in the padded layout [B, d + 2, h + 2, w + 2] (d = 0: pixel `row` of [B, h, w]
// in the decoder's [B, h + 2, w + 2]). The host zeroes each padded buffer once and kernels write only its interior,
// so the borders stay zero; every padded layout has its own buffer
static inline ulong v8_padded_row(uint row, uint d, uint h, uint w) {
    uint x = row % w, r = row / w;
    uint y = r % h;
    r /= h;
    if (d == 0) return ((ulong)r * (h + 2) + y + 1) * (w + 2) + x + 1;
    uint z = r % d, b = r / d;
    return (((ulong)b * (d + 2) + z + 1) * (h + 2) + y + 1) * (w + 2) + x + 1;
}

// A matmul tile's float32 accumulators, staged in threadgroup memory as [BM, BN] so the epilogue reads and writes
// contiguous channels: a cooperative tensor's own elements are scattered, and storing FP16 outputs one at a time from
// them cost more than the matmul itself at K = 64
using v8_stage = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

// v8_epi on 8 consecutive outputs of a staged tile, in the same order: y[out ..] = epi(s[0 ..]), the residual at
// res[ro ..] and the bias at bias[col ..]. out, ro and col are multiples of 8 (the host requires o % 8 == 0), so
// the vector accesses are aligned
static inline void v8_epi8(threadgroup const float* s, device half* y, ulong out, device const float* bias, uint col,
                           device const half* res, ulong ro, uint flags) {
    float4 a = *(threadgroup const float4*)s, e = *(threadgroup const float4*)(s + 4);
    if (flags & V8_RES) {
        a = a + float4(*(device const half4*)(res + ro));
        e = e + float4(*(device const half4*)(res + ro + 4));
    }
    if (flags & V8_BIAS) {
        a = a + *(device const float4*)(bias + col);
        e = e + *(device const float4*)(bias + col + 4);
    }
    // `<` keeps a NaN, as v8_epi does
    if (flags & V8_RELU) {
        a = select(a, float4(0.0f), a < 0.0f);
        e = select(e, float4(0.0f), e < 0.0f);
    }
    *(device half4*)(y + out) = half4(a);
    *(device half4*)(y + out + 4) = half4(e);
}

// y[m, n] = epi(x[m, k] . w[n, k]^T): the 1x1x1 convs, w [n, k]. With V8_PAD the m rows are the voxels of
// [B, pd, ph, pw] and y is their padded layout. Each tile's views start at its first row, so the int32 tensor
// extents stay small at any batch. Grid (n tiles, m tiles), 32 SG threads
template <int BM, int BN, int SG>
kernel void v8_gemm(device half* p_x [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                    device const float* p_bias [[buffer(3)]], device const half* p_res [[buffer(4)]],
                    constant uint& m [[buffer(5)]], constant uint& n [[buffer(6)]], constant uint& k [[buffer(7)]],
                    constant uint& flags [[buffer(8)]], constant uint& pd [[buffer(9)]],
                    constant uint& ph [[buffer(10)]], constant uint& pw [[buffer(11)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint m0 = tg.y * BM;
    int n0 = tg.x * BN;
    int rows = min((int)(m - m0), BM);
    mat<half> tx(p_x + (ulong)m0 * k, dextents<int32_t, 2>(k, rows), array<int, 2>{1, (int)k});
    mat<half> tw(p_w, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mx = tx.slice(0, 0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    threadgroup float stage[BM * BN];
    v8_stage ts(stage, dextents<int32_t, 2>(BN, BM), array<int, 2>{1, BN});
    acc.store(ts);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < BM * BN / 8; q += 32 * SG) {
        int r = q / (BN / 8), c8 = (q % (BN / 8)) * 8, col = n0 + c8;
        if (r >= rows || col >= (int)n) continue;
        uint row = m0 + r;
        ulong o = (ulong)row * n + col;
        ulong out = (flags & V8_PAD) ? v8_padded_row(row, pd, ph, pw) * n + col : o;
        v8_epi8(stage + r * BN + c8, p_y, out, p_bias, col, p_res, o, flags);
    }
}

// 3x3x3 (D3) or 3x3 conv, pad 1, stride S (1 or 2), channels-last, over the zero-padded input of an input volume
// [d, h, w] ([B, d + 2, h + 2, w + 2, c]; 2D: [B, h + 2, w + 2, c], with d = 1); output [B, d / S, h / S, w / S, o],
// plain. The three dx taps of each (dz, dy) are consecutive voxels and the weights [o, taps, c] hold them
// contiguously, so a tensor view with K extent 3c covers them: 9 matmuls per tile (3 in 2D), no im2col. Each view
// starts at its tile's plane or line, so the int32 tensor extents stay small
//
// - S = 1: a tile is BM consecutive rows of one output plane (b, oz) in the padded row layout, virtual row
//   r = oy (w + 2) + ox, whose tap input is row r + dz plane + dy (w + 2) + dx (row stride c). Rows with ox >= w are
//   x padding and are dropped; the last plane's reach 2 voxels past its sample, so padded buffers carry a tail
// - S = 2: a tile is up to BM outputs of one output line (b, oz, oy); output ox reads padded voxel 2 ox + dx of line
//   2 oy + dy of plane 2 oz + dz (row stride 2c). A view can't stride across lines, so tiles never span two
//
// Grid (o tiles, B (d / S) tiles_per_plane), 32 SG threads
template <int BM, int BN, int SG, int S, bool D3>
kernel void v8_taps(device half* p_xp [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                    device const float* p_bias [[buffer(3)]], device const half* p_res [[buffer(4)]],
                    constant uint& batch [[buffer(5)]], constant uint& d [[buffer(6)]], constant uint& h [[buffer(7)]],
                    constant uint& w [[buffer(8)]], constant uint& c [[buffer(9)]], constant uint& o [[buffer(10)]],
                    constant uint& flags [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]]) {
    constexpr int TAPS = D3 ? 27 : 9;
    int wp = w + 2, plane = (h + 2) * wp, per = D3 ? (d + 2) * plane : plane;
    int dout = D3 ? d / S : 1, hout = h / S, wout = w / S;
    // rows of one view, tiles per output line or plane, and this tile's output plane and first row
    int span = S == 1 ? hout * wp : wout;
    int per_line = (span + BM - 1) / BM, per_plane = S == 1 ? per_line : hout * per_line;
    int bz = tg.y / per_plane, chunk = tg.y % per_plane;
    int b = bz / dout, oz = bz % dout;
    if (b >= (int)batch) return;
    int oy0 = S == 1 ? 0 : chunk / per_line;
    int m0 = (chunk % per_line) * BM, n0 = tg.x * BN;
    device half* base = p_xp + ((ulong)b * per + (ulong)(S * oz) * plane + (ulong)(S * oy0) * wp) * c;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    mat<half> tx0(base, dextents<int32_t, 2>(3 * c, span), array<int, 2>{1, S * (int)c});
    mat<half> tw0(p_w, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, TAPS * (int)c});
    auto acc =
        op.template get_destination_cooperative_tensor<decltype(tx0.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < TAPS / 3; t++) {
        int shift = D3 ? (t / 3) * plane + (t % 3) * wp : t * wp;
        mat<half> tx(base + (ulong)shift * c, dextents<int32_t, 2>(3 * c, span), array<int, 2>{1, S * (int)c});
        mat<half> tw(p_w + t * 3 * c, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, TAPS * (int)c});
        auto mx = tx.slice(0, m0);
        auto mw = tw.slice(0, n0);
        op.run(mx, mw, acc);
    }
    // no staging here: at K = 3c x 9 the scattered stores are a small share of the tile, and the 16 KB stage cost
    // 5-10% on every taps shape (bench smoke5), likely occupancy
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], r = m0 + idx[1];
        int oy = S == 1 ? r / wp : oy0, ox = S == 1 ? r % wp : r;
        if (r < span && ox < wout && col < (int)o) {
            ulong oi = ((((ulong)b * dout + oz) * hout + oy) * wout + ox) * o + col;
            p_y[oi] = v8_epi(acc[i], col, p_bias, p_res, oi, flags);
        }
    }
}

#define V8_GEMM(BM, BN, SG)                                                                                          \
    template [[host_name("v8_gemm_" #BM "x" #BN "_sg" #SG)]] kernel void v8_gemm<BM, BN, SG>(                       \
        device half*, device half*, device half*, device const float*, device const half*, constant uint&,           \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint);
#define V8_TAPS(BM, BN, SG, S, D3, NAME)                                                                             \
    template [[host_name("v8_taps" NAME "_" #BM "x" #BN "_sg" #SG "_s" #S)]] kernel void                            \
    v8_taps<BM, BN, SG, S, D3>(device half*, device half*, device half*, device const float*, device const half*,    \
                               constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,      \
                               constant uint&, constant uint&, uint2);
// a staged GEMM tile takes BM BN 4 bytes of Metal's 32 KB of threadgroup memory, so 128 x 128 is gone; the taps
// kept theirs only while it was never their fastest (bench2)
V8_GEMM(64, 64, 4)
V8_GEMM(128, 64, 4)
V8_GEMM(128, 64, 8)
V8_GEMM(64, 128, 4)
V8_TAPS(64, 64, 4, 1, true, "3")
V8_TAPS(128, 64, 4, 1, true, "3")
V8_TAPS(128, 64, 8, 1, true, "3")
V8_TAPS(64, 128, 4, 1, true, "3")
V8_TAPS(32, 64, 4, 2, true, "3")
V8_TAPS(32, 128, 4, 2, true, "3")
V8_TAPS(64, 64, 4, 2, true, "3")
V8_TAPS(64, 64, 4, 1, false, "2")
V8_TAPS(128, 64, 4, 1, false, "2")
V8_TAPS(64, 128, 4, 1, false, "2")

// x [B, d, h, w, c] at even (z, y, x) into [B, d/2, h/2, w/2, c]: the stride-2 downsample's input. One thread per
// channel pair
kernel void v8_sub2(device half2* p_y [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                    constant uint& d [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                    constant uint& c [[buffer(5)]], constant uint& count [[buffer(6)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint c2 = c / 2, dout = d / 2, hout = h / 2, wout = w / 2;
    uint k = i % c2, pix = i / c2;
    uint x = pix % wout, r = pix / wout;
    uint y = r % hout;
    r /= hout;
    uint z = r % dout, b = r / dout;
    p_y[i] = p_x[((((ulong)b * d + 2 * z) * h + 2 * y) * w + 2 * x) * c2 + k];
}

// ---------------------------------------------------------------- decoder and head

// y[b, p, off + c] = max over depth of x[b, z, p, c]: a layer's depth max into channels off .. off + c of the
// decoder's concatenation buffer (ystride channels per pixel, the zero-padded 2D layout when pad != 0) or into a plain
// map. torch.max's NaN propagation, FP16 in and out. One thread per (sample, pixel, channel pair); x is
// [B, depth, h, w, c]
kernel void v8_dmax(device half2* p_y [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                    constant uint& depth [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                    constant uint& c [[buffer(5)]], constant uint& ystride [[buffer(6)]],
                    constant uint& off [[buffer(7)]], constant uint& pad [[buffer(8)]],
                    constant uint& count [[buffer(9)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint c2 = c / 2, pixels = h * w;
    uint b = i / (pixels * c2), p = (i / c2) % pixels, k = i % c2;
    float2 m = float2(-INFINITY);
    for (uint z = 0; z < depth; z++) {
        float2 v = float2(p_x[(((ulong)b * depth + z) * pixels + p) * c2 + k]);
        if (v.x > m.x || isnan(v.x)) m.x = v.x;
        if (v.y > m.y || isnan(v.y)) m.y = v.y;
    }
    ulong row = pad ? v8_padded_row(b * pixels + p, 0, h, w) : (ulong)b * pixels + p;
    p_y[(row * ystride + off) / 2 + k] = half2(m);
}

// PyTorch's bilinear source index for output d of a x2 upsample (align_corners=False): the first input, the step to
// the second (0 at the edge) and both weights, in float32
static inline void v8_src2(uint d, uint len, thread uint& i0, thread uint& step, thread float& l0, thread float& l1) {
    float r = 0.5f * (float(d) + 0.5f) - 0.5f;
    r = r < 0.0f ? 0.0f : r;
    i0 = uint(r);
    step = i0 < len - 1 ? 1 : 0;
    l1 = r - float(i0);
    l0 = 1.0f - l1;
}

// The decoder's F.interpolate(low, scale_factor=2, mode="bilinear") (align_corners=False) into channels off .. off + c
// of the zero-padded concatenation buffer y (ystride channels per pixel), FP16 in and out with PyTorch's float32 index
// and weight formulas. One thread per (sample, output pixel, channel pair); low is [B, h, w, c]
kernel void v8_up2cat(device half2* p_y [[buffer(0)]], device const half2* p_low [[buffer(1)]],
                      constant uint& h [[buffer(2)]], constant uint& w [[buffer(3)]], constant uint& c [[buffer(4)]],
                      constant uint& ystride [[buffer(5)]], constant uint& off [[buffer(6)]],
                      constant uint& count [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint c2 = c / 2, h2 = 2 * h, w2 = 2 * w;
    uint b = i / (h2 * w2 * c2), p = (i / c2) % (h2 * w2), k = i % c2;
    uint y0, ys, x0, xs;
    float hl0, hl1, wl0, wl1;
    v8_src2(p / w2, h, y0, ys, hl0, hl1);
    v8_src2(p % w2, w, x0, xs, wl0, wl1);
    device const half2* lb = p_low + (ulong)b * h * w * c2 + k;
    float2 a = float2(lb[(y0 * w + x0) * c2]), e = float2(lb[(y0 * w + x0 + xs) * c2]);
    float2 f = float2(lb[((y0 + ys) * w + x0) * c2]), g = float2(lb[((y0 + ys) * w + x0 + xs) * c2]);
    float2 v = hl0 * (wl0 * a + wl1 * e) + hl1 * (wl0 * f + wl1 * g);
    ulong row = v8_padded_row(b * h2 * w2 + p, 0, h2, w2);
    p_y[(row * ystride + off) / 2 + k] = half2(v);
}

static inline float v8_r16(float x) { return float(half(x)); }

// The head: the 1 x 1 logit conv (256 -> 1) on the decoder's last map, then sigmoid, into each real tile's slot of
// prob ([slots, 64 * 64] float32), as autocast runs it: the FP16 dot with float32 accumulation rounded to FP16, plus
// the FP16-rounded bias, rounded again. w holds the weights already rounded to FP16. sigmoid16 rounds the probability
// to FP16 (torch.sigmoid of the FP16 logit, as ink8um takes it); otherwise it is sigmoid(logit.float()), as the 306B
// runners take it. feat is [B, 64, 64, c]; one thread per (tile, pixel)
kernel void v8_head(device float* p_prob [[buffer(0)]], device const half* p_feat [[buffer(1)]],
                    device const float* p_w [[buffer(2)]], constant float& bias [[buffer(3)]],
                    device const int* p_desc [[buffer(4)]], constant uint& c [[buffer(5)]],
                    constant uint& sigmoid16 [[buffer(6)]], constant uint& count [[buffer(7)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint tile = i / V8_PIX, p = i % V8_PIX;
    int slot = p_desc[tile * V8_DESC + 4];
    if (slot < 0) return;
    device const half* f = p_feat + (ulong)i * c;
    float acc = 0.0f;
    for (uint k = 0; k < c; k++) acc += float(f[k]) * p_w[k];
    float logit = v8_r16(v8_r16(acc) + v8_r16(bias));
    float s = 1.0f / (1.0f + precise::exp(-logit));
    p_prob[(ulong)slot * V8_PIX + p] = sigmoid16 ? v8_r16(s) : s;
}

// The stitched map: per pixel, over the kept tiles that cover it in tile order (row-major on the stride grid),
// num += p * w and den += w (mode 1, the 306B runners' weight sum, NaN where no tile) or den += 1 (mode 0, the
// release's tile count, 0 where no tile, clipped to 0..1), in float32 with no contraction, as numpy and PyTorch add
// them. grid is [ny, nx] slots (-1 for a tile not kept) of the lattice k * stride; out is [height, width]. One thread
// per pixel
kernel void v8_blend(device float* p_out [[buffer(0)]], device const float* p_prob [[buffer(1)]],
                     device const int* p_grid [[buffer(2)]], device const float* p_weight [[buffer(3)]],
                     constant uint& ny [[buffer(4)]], constant uint& nx [[buffer(5)]],
                     constant uint& stride [[buffer(6)]], constant uint& width [[buffer(7)]],
                     constant uint& mode [[buffer(8)]], constant uint& count [[buffer(9)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint y = i / width, x = i % width;
    // tiles k with k stride <= v < k stride + 64
    uint ky0 = y >= V8_P ? (y - (V8_P - 1) + stride - 1) / stride : 0, ky1 = min(y / stride + 1, ny);
    uint kx0 = x >= V8_P ? (x - (V8_P - 1) + stride - 1) / stride : 0, kx1 = min(x / stride + 1, nx);
    float num = 0.0f, den = 0.0f;
    for (uint ky = ky0; ky < ky1; ky++) {
        uint ly = y - ky * stride;
        for (uint kx = kx0; kx < kx1; kx++) {
            uint lx = x - kx * stride;
            int slot = p_grid[ky * nx + kx];
            if (slot < 0) continue;
            float wt = p_weight[ly * V8_P + lx];
            float p = p_prob[(ulong)slot * V8_PIX + ly * V8_P + lx];
            num = num + p * wt;
            den = den + (mode == 0 ? 1.0f : wt);
        }
    }
    float v;
    if (mode == 0) {
        float q = den != 0.0f ? precise::divide(num, den) : 0.0f;
        // np.clip(np.nan_to_num(q), 0, 1)
        v = isnan(q) ? 0.0f : clamp(q, 0.0f, 1.0f);
    } else {
        v = den > 1e-9f ? precise::divide(num, den) : NAN;
    }
    p_out[i] = v;
}
