// Convolutions and GEMMs of hybrid_3d2d on Metal 4 tensor ops (matmul2d), FP16 operands with FP32 accumulation and
// one FP16 rounding of the result, as cuDNN's FP16 channels-last convolutions under autocast. Convs carry no bias:
// every conv feeds an InstanceNorm, which removes it exactly (see src/model.rs). Each conv's epilogue also writes the
// InstanceNorm partials of its FP16 outputs (`tile_stats`), so the norm never re-reads the conv output for its
// statistics
//
// A stride-1 conv runs as one accumulating matmul2d per tap over row windows of a zero-padded input: in the padded
// layout every tap is a constant row shift, so the whole batch is one tall GEMM per tap. Output rows that land on
// padding are computed and dropped

// x [m, k] . w[n, k]^T for the tile at (m0, n0), with x and w strided, handed to `epi(row, col, value)` for every
// in-bounds element
template <int BM, int BN, typename Epi>
inline void gemm_tile(device half* x, device half* w, int m, int n, int k, int ldx, int ldw, int m0, int n0, Epi epi) {
    mat<half> tx(x, dextents<int32_t, 2>(k, m), array<int, 2>{1, ldx});
    mat<half> tw(w, dextents<int32_t, 2>(k, n), array<int, 2>{1, ldw});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], row = m0 + idx[1];
        if (row < m && col < n) epi(row, col, acc[i]);
    }
}

// y[m, n] = x[m, k] . w[n, k]^T in FP16, x with leading dimension ldx; grid (n tiles, m tiles), 128 threads
template <int BM, int BN>
kernel void gemm_nt(device half* p_x [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                    constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                    constant uint& ldx [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
    gemm_tile<BM, BN>(p_x, p_w, m, n, k, ldx, k, tg.y * BM, tg.x * BN,
                      [&](int row, int col, float v) { p_y[(long)row * n + col] = half(v); });
}

#define GEMM_NT(BM, BN)                                                                                              \
    template [[host_name("gemm_nt_" #BM "x" #BN)]] kernel void gemm_nt<BM, BN>(                                     \
        device half*, device half*, device half*, constant uint&, constant uint&, constant uint&, constant uint&,    \
        uint2);
GEMM_NT(64, 64)
GEMM_NT(64, 32)
GEMM_NT(64, 16)

// gemm_nt for a conv whose output feeds an InstanceNorm: the m rows are B samples of `rps` rows each, every tile
// stays inside one sample (grid (n tiles, B * ceil(rps / BM))), and each tile writes its columns' IN partials to
// part [B, ceil(rps / BM), n]
template <int BM, int BN>
kernel void gemm_nt_in(device half* p_x [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                       constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                       constant uint& ldx [[buffer(6)]], constant uint& rps [[buffer(7)]],
                       device float4* p_part [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * BN];
    threadgroup float4 part[128];
    int tps = (rps + BM - 1) / BM;
    int b = tg.y / tps, chunk = tg.y % tps;
    int m0 = b * rps + chunk * BM, n0 = tg.x * BN;
    int end = min(m0 + BM, int((b + 1) * rps));
    gemm_tile<BM, BN>(p_x, p_w, m, n, k, ldx, k, m0, n0, [&](int row, int col, float v) {
        if (row >= end) return;
        half h = half(v);
        p_y[(long)row * n + col] = h;
        tile[(row - m0) * BN + col - n0] = h;
    });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<BM, BN, 128>(tile, part, tid, [&](int j) { return m0 + j < end; },
                            p_part + ((long)b * tps + chunk) * n + n0);
}

#define GEMM_NT_IN(BM, BN)                                                                                           \
    template [[host_name("gemm_nt_in_" #BM "x" #BN)]] kernel void gemm_nt_in<BM, BN>(                               \
        device half*, device half*, device half*, constant uint&, constant uint&, constant uint&, constant uint&,    \
        constant uint&, device float4*, uint2, uint);
GEMM_NT_IN(64, 64)
GEMM_NT_IN(64, 32)
GEMM_NT_IN(64, 16)
GEMM_NT_IN(32, 64)
GEMM_NT_IN(32, 32)
GEMM_NT_IN(32, 16)

// Zero-pads x [B, h, w, c] into xp [B, h + 2, w + 2, c] followed by `tail` zero rows of c, so every tap's row window
// of the last sample stays inside the buffer; one thread per FP16 pair of xp
kernel void pad2d(device half2* p_xp [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                  constant uint& batch [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                  constant uint& c [[buffer(5)]], constant uint& count [[buffer(6)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint pairs = c / 2, wp = w + 2, hp = h + 2;
    uint cp = i % pairs, r = i / pairs;
    uint per = hp * wp;
    uint b = r / per, rem = r % per;
    uint y = rem / wp, x = rem % wp;
    // rows past the last sample are the tail, zero
    bool inside = b < batch && y >= 1 && y <= h && x >= 1 && x <= w;
    half2 v = half2(0.0h);
    if (inside) v = p_x[((b * h + y - 1) * w + x - 1) * pairs + cp];
    p_xp[i] = v;
}

// 3x3 conv, stride 1, pad 1, channels-last, over the padded batch xp of pad2d: output row r of the padded map
// (b = r / P, y = (r % P) / (w + 2), x = (r % P) % (w + 2), P = (h + 2)(w + 2)) reads xp rows r + dy (w + 2) + dx.
// Weights are inkrs's [o, 9, c]; tap t is the strided view at offset t * c with row stride 9c. Grid (o tiles, rows
// tiles) of 128 threads
//
// The outputs of sample b are the h (w + 2) consecutive rows from b P, all kept but the 2 x-padding rows of each line;
// the grid tiles each sample's run separately (B * ceil(h (w + 2) / BM) row tiles), which skips the y-padding rows and
// keeps every tile inside one sample, so each tile also writes its columns' InstanceNorm partials to part [B, tiles,
// o]
template <int BM, int BN>
kernel void conv3x3(device half* p_xp [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                    constant uint& batch [[buffer(3)]], constant uint& h [[buffer(4)]], constant uint& w [[buffer(5)]],
                    constant uint& c [[buffer(6)]], constant uint& o [[buffer(7)]],
                    device float4* p_part [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * BN];
    threadgroup float4 part[128];
    int wp = w + 2, per = (h + 2) * wp;
    int rows = batch * per, xrows = rows + 2 * wp + 2;
    int span = h * wp, tps = (span + BM - 1) / BM;
    int b = tg.y / tps, chunk = tg.y % tps;
    int base = b * per, end = base + span;
    int m0 = base + chunk * BM, n0 = tg.x * BN;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    // overlapping rows: row r of this K = 3c view is xp rows r, r + 1, r + 2 back to back, the three dx taps of one
    // dy, and the matching weights [o, dy, 0..3, c] are contiguous too, so each dy is one matmul with K = 3c. The
    // last two rows would read past the buffer and are left out of the extent; no kept output reaches them
    mat<half> tx(p_xp, dextents<int32_t, 2>(3 * c, xrows - 2), array<int, 2>{1, (int)c});
    mat<half> tw0(p_w, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 9 * (int)c});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int dy = 0; dy < 3; dy++) {
        int shift = dy * wp;
        mat<half> tw(p_w + dy * 3 * c, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 9 * (int)c});
        auto mx = tx.slice(0, m0 + shift);
        auto mw = tw.slice(0, n0);
        op.run(mx, mw, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], r = m0 + idx[1];
        int rem = r - base;
        int y = rem / wp, x = rem % wp;
        if (r < end && x < (int)w && col < (int)o) {
            half v = half(acc[i]);
            p_y[((long)(b * h + y) * w + x) * o + col] = v;
            tile[idx[1] * BN + idx[0]] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<BM, BN, 128>(tile, part, tid, [&](int j) { return m0 + j < end && (m0 + j - base) % wp < (int)w; },
                            p_part + ((long)b * tps + chunk) * o + n0);
}

#define CONV3X3(BM, BN)                                                                                              \
    template [[host_name("conv3x3_" #BM "x" #BN)]] kernel void conv3x3<BM, BN>(                                     \
        device half*, device half*, device half*, constant uint&, constant uint&, constant uint&, constant uint&,    \
        constant uint&, device float4*, uint2, uint);
CONV3X3(64, 64)
CONV3X3(64, 32)
CONV3X3(64, 16)
CONV3X3(32, 64)
CONV3X3(32, 32)
CONV3X3(32, 16)

// im2col of a 3x3 stride-2 pad-1 conv: x [B, h, w, c] to col [B * h/2 * w/2, 9c], column t * c + ch for tap t, so a
// GEMM against inkrs's [o, 9, c] weights is the conv; one thread per FP16 pair of col
kernel void im2col_s2(device half2* p_col [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                      constant uint& h [[buffer(2)]], constant uint& w [[buffer(3)]], constant uint& c [[buffer(4)]],
                      constant uint& count [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint pairs = c / 2, ho = h / 2, wo = w / 2;
    uint cp = i % pairs, rest = i / pairs;
    uint t = rest % 9, pix = rest / 9;
    uint ox = pix % wo, r2 = pix / wo;
    uint oy = r2 % ho, b = r2 / ho;
    int y = int(2 * oy + t / 3) - 1, x = int(2 * ox + t % 3) - 1;
    half2 v = half2(0.0h);
    if (y >= 0 && y < int(h) && x >= 0 && x < int(w)) v = p_x[((b * h + uint(y)) * w + uint(x)) * pairs + cp];
    p_col[i] = v;
}

// The stem's first conv3d, 1 -> 16 channels, 3x3x3, pad 1, on the normalised input x [B, 17, 128, 128]: one thread
// per output voxel, 16 FP32 accumulators, one FP16 rounding each. Weights [16, 27] FP16. Threadgroups of 256 voxels,
// which divide a sample's 17 x edge x edge, each write their channels' InstanceNorm partials to part [B * tiles, 16]
kernel void conv3d_c1(device half* p_y [[buffer(0)]], device const half* p_x [[buffer(1)]],
                      device const half* p_w [[buffer(2)]], device float4* p_part [[buffer(3)]],
                      constant uint& edge [[buffer(4)]], uint i [[thread_position_in_grid]],
                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[256 * 16];
    threadgroup float4 part[256];
    const uint D = 17;
    uint H = edge, W = edge;
    uint x = i % W, r = i / W;
    uint y = r % H;
    r /= H;
    uint z = r % D, b = r / D;
    float acc[16];
    for (uint k = 0; k < 16; k++) acc[k] = 0.0f;
    device const half* xb = p_x + (long)b * D * H * W;
    for (uint t = 0; t < 27; t++) {
        int zz = int(z) + int(t / 9) - 1, yy = int(y) + int((t / 3) % 3) - 1, xx = int(x) + int(t % 3) - 1;
        if (zz < 0 || zz >= int(D) || yy < 0 || yy >= int(H) || xx < 0 || xx >= int(W)) continue;
        float v = float(xb[(zz * H + yy) * W + xx]);
        for (uint k = 0; k < 16; k++) acc[k] = fma(v, float(p_w[k * 27 + t]), acc[k]);
    }
    device half* yo = p_y + (long)i * 16;
    for (uint k = 0; k < 16; k++) {
        half v = half(acc[k]);
        yo[k] = v;
        tile[tid * 16 + k] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<256, 16, 256>(tile, part, tid, [](int) { return true; }, p_part + (long)tg * 16);
}

// Zero-pads the stem activation x [B, 17, 128, 128, 16] into xp [B, 19, 130, 130, 16] plus `tail` zero rows; one
// thread per FP16 pair of xp
kernel void pad3d(device half2* p_xp [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                  constant uint& batch [[buffer(2)]], constant uint& count [[buffer(3)]],
                  constant uint& edge [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint D = 17, PAIRS = 8;
    uint H = edge, W = edge;
    const uint WP = W + 2, HP = H + 2, DP = D + 2;
    uint cp = i % PAIRS, r = i / PAIRS;
    uint per = DP * HP * WP;
    uint b = r / per, rem = r % per;
    uint z = rem / (HP * WP), yx = rem % (HP * WP);
    uint y = yx / WP, x = yx % WP;
    bool inside = b < batch && z >= 1 && z <= D && y >= 1 && y <= H && x >= 1 && x <= W;
    half2 v = half2(0.0h);
    if (inside) v = p_x[(((b * D + z - 1) * H + y - 1) * W + x - 1) * PAIRS + cp];
    p_xp[i] = v;
}

// The stem's second conv3d, 16 -> 16, 3x3x3, pad 1, as 9 accumulating matmul2d over the padded layout of pad3d.
// Weights [16, 27, 16]
//
// Output (b, z, y, x) sits at padded row b P + z PLANE + y WP + x (P = 19 PLANE), so sample b's outputs lie in the 17
// PLANE rows from b P. The grid tiles each sample's run separately (grid (1, B * ceil(17 PLANE / BM))), which skips
// the two z-padding planes the outputs never use and keeps every tile inside one sample, so each tile also writes its
// channels' InstanceNorm partials to part [B, tiles, 16]
template <int BM>
kernel void conv3d_c16(device half* p_xp [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                       constant uint& batch [[buffer(3)]], device float4* p_part [[buffer(4)]],
                       constant uint& edge [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * 16];
    threadgroup float4 part[128];
    const int D = 17, C = 16;
    const int H = edge, W = edge;
    const int WP = W + 2, HP = H + 2, PLANE = HP * WP, per = (D + 2) * PLANE;
    int rows = batch * per, xrows = rows + 2 * PLANE + 2 * WP + 2;
    const int span = D * PLANE, tps = (span + BM - 1) / BM;
    int b = tg.y / tps, chunk = tg.y % tps;
    int base = b * per, end = base + span;
    int m0 = base + chunk * BM;
    constexpr auto desc = matmul2d_descriptor(BM, 16, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    // the dx taps fold into K = 48 through overlapping rows, as in conv3x3: 9 matmuls instead of 27
    mat<half> tx(p_xp, dextents<int32_t, 2>(3 * C, xrows - 2), array<int, 2>{1, C});
    mat<half> tw0(p_w, dextents<int32_t, 2>(3 * C, 16), array<int, 2>{1, 27 * C});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int shift = (t / 3) * PLANE + (t % 3) * WP;
        mat<half> tw(p_w + t * 3 * C, dextents<int32_t, 2>(3 * C, 16), array<int, 2>{1, 27 * C});
        auto mx = tx.slice(0, m0 + shift);
        auto mw = tw.slice(0, 0);
        op.run(mx, mw, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = idx[0], r = m0 + idx[1];
        int rem = r - base;
        int z = rem / PLANE, yx = rem % PLANE;
        int y = yx / WP, x = yx % WP;
        if (r < end && y < H && x < W && col < 16) {
            half v = half(acc[i]);
            p_y[((((long)b * D + z) * H + y) * W + x) * 16 + col] = v;
            tile[idx[1] * 16 + col] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<BM, 16, 128>(tile, part, tid,
                            [&](int j) {
                                int rem = m0 + j - base, yx = rem % PLANE;
                                return m0 + j < end && yx / WP < H && yx % WP < W;
                            },
                            p_part + ((long)b * tps + chunk) * 16);
}

template [[host_name("conv3d_c16_64")]] kernel void conv3d_c16<64>(device half*, device half*, device half*,
                                                                    constant uint&, device float4*, constant uint&,
                                                                    uint2, uint);
