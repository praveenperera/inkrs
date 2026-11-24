// The nnU-Net ResEnc-L forward (m7) on Metal 4 tensor ops: fp16 channels-last activations, every 3x3x3 conv as
// accumulating matmul2d over its taps with FP32 accumulation, and the glue that repeats the CUDA engine's u3h_* and
// nn_* rounding (inkrs-kernels unet3d.rs and nnunet.rs). Always half, whatever `real` is: the nnU-Net engine runs
// villa's fp16 autocast only
//
// Layouts. A conv input lives in a zero-padded buffer [e + 2, e + 2, e + 2, c] whose one-voxel border no kernel
// writes, so every tap is a constant row shift and no conv bounds-checks its reads. Conv outputs, the transposed
// conv's GEMM input and the head's input are unpadded [e, e, e, c]. `pad` arguments say which of the two a kernel
// reads or writes: 1 for padded, 0 for not
//
// Conv outputs are stored as the CUDA engine's u3h_partial sees them: `r16(r16(acc) + bias)`, the conv rounded to
// fp16 and PyTorch's separate bias add rounded again, so the norm statistics and the apply read the biased value

// `x` rounded to fp16 and widened back, round to nearest even
static inline float r16(float x) { return float(half(x)); }

// Voxel (z, y, x) of an [e, e, e] grid, padded (pad = 1) or not
static inline long vox_at(int z, int y, int x, int e, int pad) {
    int ep = e + 2 * pad;
    return ((long)(z + pad) * ep + (y + pad)) * ep + (x + pad);
}

template <int BM, int BN>
using conv_desc_t = matmul2d<matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                                 matmul2d_descriptor::mode::multiply_accumulate),
                             execution_simdgroups<4>>;

// The epilogue every conv tile runs: for each in-bounds accumulator element whose row `row_of(m)` maps to an output
// voxel (>= 0), store r16(r16(acc) + bias) into p_y [voxels, cout] and stage it for the IN partial of the tile's
// columns; then every thread joins tile_stats, which writes part[col] for the tile's BN columns
template <int BM, int BN, typename Acc, typename RowOf>
inline void conv_epilogue(thread Acc& acc, RowOf row_of, int n0, int cout, device const float* p_bias, int has_bias,
                          device half* p_y, threadgroup half* tile, threadgroup float4* part, uint tid,
                          device float4* p_part, int has_part) {
    // the M4 Pro allows 32 KB of threadgroup memory, and Metal validation charged a 128x64 tile 36,864 bytes, twice
    // its 18 KB of tile and part, so a tile over 4096 elements fails validation even though it ran without it
    static_assert(BM * BN <= 4096, "conv tile over the 32 KB threadgroup budget");
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0];
        long v = row_of(idx[1]);
        if (v < 0 || col >= cout) continue;
        float o = r16(acc[i]);
        if (has_bias) o = r16(o + p_bias[col]);
        half h = half(o);
        p_y[v * cout + col] = h;
        tile[idx[1] * BN + idx[0]] = h;
    }
    if (!has_part) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<BM, BN, 128>(tile, part, tid, [&](int j) { return row_of(j) >= 0; }, p_part);
}

// 3x3x3 stride-1 pad-1 conv of up to two padded sources (the decoder's concat [up, skip] without a concat buffer):
// p_x0 [e+2]^3 x c0 with weights p_w0 [cout, 27, c0], and when c1 > 0 p_x1 with p_w1 [cout, 27, c1]
//
// In the padded layout the output (z, y, x) has span row r = z PLANE + y WP + x (PLANE = WP^2, WP = e + 2), and tap
// (dz, dy, dx) reads padded row r + dz PLANE + dy WP + dx. Rows r, r + 1, r + 2 sit back to back, so the 3 dx taps
// of one (dz, dy) are one K = 3c matmul over overlapping rows, as conv3x3 does: 9 accumulating matmuls a source.
// The grid tiles each output plane's e WP rows separately (grid (cout tiles, e * ceil(e WP / BM))), skipping the
// y-padding rows; the 2 x-padding rows of each line are computed and dropped. Tile (z, chunk) writes its columns'
// IN partials to p_part [tile, cout]
template <int BM, int BN>
kernel void nn_conv3(device half* p_x0 [[buffer(0)]], device half* p_w0 [[buffer(1)]],
                     device half* p_x1 [[buffer(2)]], device half* p_w1 [[buffer(3)]],
                     device half* p_y [[buffer(4)]], device const float* p_bias [[buffer(5)]],
                     device float4* p_part [[buffer(6)]], constant uint& e [[buffer(7)]],
                     constant uint& c0 [[buffer(8)]], constant uint& c1 [[buffer(9)]],
                     constant uint& cout [[buffer(10)]], constant uint& has_bias [[buffer(11)]],
                     constant uint& has_part [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * BN];
    threadgroup float4 part[128];
    const int E = e, WP = E + 2, PLANE = WP * WP;
    const int span = E * WP, tpp = (span + BM - 1) / BM;
    const int z = tg.y / tpp, chunk = tg.y % tpp;
    const int m0 = z * PLANE + chunk * BM, n0 = tg.x * BN;
    // the last row any tap may start a K = 3c read at: the final padded row minus the 2 dx rows it spans
    const int xrows = WP * PLANE - 2;
    conv_desc_t<BM, BN> op;
    mat<half> tx0(p_x0, dextents<int32_t, 2>(3 * c0, xrows), array<int, 2>{1, (int)c0});
    mat<half> tw00(p_w0, dextents<int32_t, 2>(3 * c0, cout), array<int, 2>{1, 27 * (int)c0});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx0.slice(0, 0)), decltype(tw00.slice(0, 0)),
                                                             float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int shift = (t / 3) * PLANE + (t % 3) * WP;
        mat<half> tw(p_w0 + t * 3 * c0, dextents<int32_t, 2>(3 * c0, cout), array<int, 2>{1, 27 * (int)c0});
        auto mx = tx0.slice(0, m0 + shift);
        auto mw = tw.slice(0, n0);
        op.run(mx, mw, acc);
    }
    if (c1 > 0) {
        mat<half> tx1(p_x1, dextents<int32_t, 2>(3 * c1, xrows), array<int, 2>{1, (int)c1});
        for (int t = 0; t < 9; t++) {
            int shift = (t / 3) * PLANE + (t % 3) * WP;
            mat<half> tw(p_w1 + t * 3 * c1, dextents<int32_t, 2>(3 * c1, cout), array<int, 2>{1, 27 * (int)c1});
            auto mx = tx1.slice(0, m0 + shift);
            auto mw = tw.slice(0, n0);
            op.run(mx, mw, acc);
        }
    }
    const int end = z * PLANE + span;
    auto row_of = [&](int m) -> long {
        int r = m0 + m;
        int yx = r - z * PLANE;
        int y = yx / WP, x = yx % WP;
        if (r >= end || x >= E) return -1;
        return ((long)z * E + y) * E + x;
    };
    conv_epilogue<BM, BN>(acc, row_of, n0, cout, p_bias, has_bias, p_y, tile, part, tid,
                          p_part + (long)tg.y * cout + n0, has_part);
}

// 3x3x3 stride-2 pad-1 conv, the encoder's downsampling conv1: p_x [2e+2]^3 x c padded, weights [cout, 27, c],
// output [e]^3 x cout. Output (z, y, x) reads padded input rows (2z + dz) PLANE + (2y + dy) WP + 2x + dx, so along one
// output line the rows step by 2 and the dx taps are again one K = 3c read: a view with row stride 2c. Tiles run
// along output lines (grid (cout tiles, e * e * ceil(e / BM))) and the view's extent ends at the line, so rows past it
// read as zero and are dropped
template <int BM, int BN>
kernel void nn_conv3s2(device half* p_x [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                       device const float* p_bias [[buffer(3)]], device float4* p_part [[buffer(4)]],
                       constant uint& e [[buffer(5)]], constant uint& c [[buffer(6)]],
                       constant uint& cout [[buffer(7)]], constant uint& has_bias [[buffer(8)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * BN];
    threadgroup float4 part[128];
    const int E = e, WP = 2 * E + 2, PLANE = WP * WP;
    const int tpl = (E + BM - 1) / BM;
    const int line = tg.y / tpl, x0 = (tg.y % tpl) * BM;
    const int z = line / E, y = line % E;
    const int n0 = tg.x * BN;
    const int rows = E - x0;
    conv_desc_t<BM, BN> op;
    mat<half> tw00(p_w, dextents<int32_t, 2>(3 * c, cout), array<int, 2>{1, 27 * (int)c});
    mat<half> tx00(p_x, dextents<int32_t, 2>(3 * c, rows), array<int, 2>{1, 2 * (int)c});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx00.slice(0, 0)), decltype(tw00.slice(0, 0)),
                                                             float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int dz = t / 3, dy = t % 3;
        long base = ((long)(2 * z + dz) * PLANE + (2 * y + dy) * WP + 2 * x0) * c;
        mat<half> tx(p_x + base, dextents<int32_t, 2>(3 * c, rows), array<int, 2>{1, 2 * (int)c});
        mat<half> tw(p_w + t * 3 * c, dextents<int32_t, 2>(3 * c, cout), array<int, 2>{1, 27 * (int)c});
        auto mx = tx.slice(0, 0);
        auto mw = tw.slice(0, n0);
        op.run(mx, mw, acc);
    }
    auto row_of = [&](int m) -> long {
        if (m >= rows) return -1;
        return ((long)z * E + y) * E + x0 + m;
    };
    conv_epilogue<BM, BN>(acc, row_of, n0, cout, p_bias, has_bias, p_y, tile, part, tid,
                          p_part + (long)tg.y * cout + n0, 1u);
}

#define NN_CONV3(BM, BN)                                                                                             \
    template [[host_name("nn_conv3_" #BM "x" #BN)]] kernel void nn_conv3<BM, BN>(                                   \
        device half*, device half*, device half*, device half*, device half*, device const float*, device float4*,   \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint);
NN_CONV3(64, 32)
NN_CONV3(128, 32)
NN_CONV3(64, 64)
NN_CONV3(32, 64)
NN_CONV3(32, 128)

#define NN_CONV3S2(BM, BN)                                                                                           \
    template [[host_name("nn_conv3s2_" #BM "x" #BN)]] kernel void nn_conv3s2<BM, BN>(                               \
        device half*, device half*, device half*, device const float*, device float4*, constant uint&,               \
        constant uint&, constant uint&, constant uint&, uint2, uint);
NN_CONV3S2(32, 64)
NN_CONV3S2(64, 64)
NN_CONV3S2(16, 64)
NN_CONV3S2(32, 128)
NN_CONV3S2(16, 128)

// Per column [mean, invstd] of a conv's IN partials p_part [tiles, cols] (n, mean, M2, 0), merged by Chan's rule in a
// fixed order: thread t merges tiles t, t + 256, ... in order, then a tree over the 256 threads in a fixed pairing,
// so a second run gives the same bits. invstd = 1 / sqrt(var + eps) with the biased variance, as ATen's statistics
// give it. Grid (cols), 256 threads
kernel void nn_in_stats(device float2* p_stats [[buffer(0)]], device const float4* p_part [[buffer(1)]],
                        constant uint& tiles [[buffer(2)]], constant uint& cols [[buffer(3)]],
                        constant float& eps [[buffer(4)]], uint col [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 acc[256];
    float4 a = float4(0.0f);
    for (uint t = tid; t < tiles; t += 256) a = chan_merge(a, p_part[(long)t * cols + col]);
    acc[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_n = 128; half_n > 0; half_n /= 2) {
        if (tid < half_n) acc[tid] = chan_merge(acc[tid], acc[tid + half_n]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float4 r = acc[0];
        float var = r.x > 0.0f ? r.z / r.x : 0.0f;
        var = var > 0.0f ? var : 0.0f;
        p_stats[col] = float2(r.y, 1.0f / sqrt(var + eps));
    }
}

// ---- glue: the CUDA engine's u3h_* and nn_* steps, same rounding points ----

// ATen's InstanceNorm transform rounded to fp16: gamma * (x - mean) * invstd + beta, the trailing multiply-add fused
static inline float in_affine(float v, float2 st, float g, float b) { return r16(fma(g * (v - st.x), st.y, b)); }

// PyTorch's LeakyReLU on an fp16 value
static inline float leaky16(float v, float slope) { return v > 0.0f ? v : r16(v * slope); }

// One flip of one patch's input into channel 0 of the padded x [p+2]^3 x cstride (nn_prep): table[code] for in-volume
// voxels, 0 elsewhere. The forward sees voxel (p-1-z, y, x) at (z, y, x) when flip bit 4 is set, and so on (x = 1,
// y = 2). One thread per patch voxel
kernel void nn_prep(device half* p_x [[buffer(0)]], device const uchar* p_slab [[buffer(1)]],
                    device const half* p_table [[buffer(2)]], constant uint& sh [[buffer(3)]],
                    constant uint& sw [[buffer(4)]], constant uint& oz [[buffer(5)]], constant uint& oy [[buffer(6)]],
                    constant uint& ox [[buffer(7)]], constant uint& p [[buffer(8)]], constant uint& vz [[buffer(9)]],
                    constant uint& vy [[buffer(10)]], constant uint& vx [[buffer(11)]],
                    constant uint& cstride [[buffer(12)]], constant uint& flip [[buffer(13)]],
                    uint e [[thread_position_in_grid]]) {
    if (e >= p * p * p) return;
    uint fx = e % p, fy = (e / p) % p, fz = e / (p * p);
    uint lx = (flip & 1) ? p - 1 - fx : fx;
    uint ly = (flip & 2) ? p - 1 - fy : fy;
    uint lz = (flip & 4) ? p - 1 - fz : fz;
    half v = 0.0h;
    if (lz < vz && ly < vy && lx < vx) v = p_table[p_slab[((long)(oz + lz) * sh + oy + ly) * sw + ox + lx]];
    p_x[vox_at(fz, fy, fx, p, 1) * cstride] = v;
}

// InstanceNorm apply, eight channels a thread: out = leaky?(r16(affine(x) + residual?)) (u3h_apply)
//
// - p_x: the biased conv output [e^3, c] unpadded; p_st, p_g, p_b its [mean, invstd] and float32 affine
// - res_mode 0: none; 1: plain, p_r [e^3, c] laid out per res_pad; 2: normed, p_r a bias-free raw conv output
//   [e^3, c] unpadded, normalised on load with p_rst, p_rg, p_rb (the BasicBlockD skip projection)
// - act != 0: LeakyReLU with slope
// - the result goes to p_out laid out per out_pad
//
// c is a multiple of 8. Eight channels a thread, as two half4, makes every access 16 bytes and spends the voxel's
// index arithmetic once per 8 channels: at two channels a thread the applies ran at 60% of their bandwidth floor
kernel void nn_apply(device half4* p_out [[buffer(0)]], device const half4* p_x [[buffer(1)]],
                     device const float2* p_st [[buffer(2)]], device const float* p_g [[buffer(3)]],
                     device const float* p_b [[buffer(4)]], device const half4* p_r [[buffer(5)]],
                     device const float2* p_rst [[buffer(6)]], device const float* p_rg [[buffer(7)]],
                     device const float* p_rb [[buffer(8)]], constant uint& e [[buffer(9)]],
                     constant uint& c [[buffer(10)]], constant uint& res_mode [[buffer(11)]],
                     constant uint& res_pad [[buffer(12)]], constant uint& act [[buffer(13)]],
                     constant float& slope [[buffer(14)]], constant uint& out_pad [[buffer(15)]],
                     uint i [[thread_position_in_grid]]) {
    uint groups = c / 8, quads = c / 4;
    uint vox = e * e * e;
    if (i >= vox * groups) return;
    uint v = i / groups, g = i % groups;
    int x = v % e, y = (v / e) % e, z = v / (e * e);
    long src = (long)v * quads + 2 * g;
    float4 h[2] = {float4(p_x[src]), float4(p_x[src + 1])};
    float4 r[2] = {float4(0.0f), float4(0.0f)};
    if (res_mode == 1) {
        long rs = vox_at(z, y, x, e, res_pad) * quads + 2 * g;
        r[0] = float4(p_r[rs]);
        r[1] = float4(p_r[rs + 1]);
    } else if (res_mode == 2) {
        r[0] = float4(p_r[src]);
        r[1] = float4(p_r[src + 1]);
    }
    half4 o[2];
    for (int q = 0; q < 2; q++) {
        float4 out;
        for (int j = 0; j < 4; j++) {
            uint ch = 8 * g + 4 * q + j;
            float yv = in_affine(h[q][j], p_st[ch], p_g[ch], p_b[ch]);
            if (res_mode == 1) yv = r16(yv + r[q][j]);
            else if (res_mode == 2) yv = r16(yv + in_affine(r[q][j], p_rst[ch], p_rg[ch], p_rb[ch]));
            if (act != 0) yv = leaky16(yv, slope);
            out[j] = yv;
        }
        o[q] = half4(out);
    }
    long dst = vox_at(z, y, x, e, out_pad) * quads + 2 * g;
    p_out[dst] = o[0];
    p_out[dst + 1] = o[1];
}

// AvgPool3d(2, 2) of the padded x [2e+2]^3 x c into the unpadded out [e^3, c] (u3h_pool2): the 2x2x2 block summed in
// float32 in ATen's order (z, y, x), times 0.125, rounded once. One thread per output pair
kernel void nn_pool2(device half2* p_out [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                     constant uint& e [[buffer(2)]], constant uint& c [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    uint pairs = c / 2;
    if (i >= e * e * e * pairs) return;
    uint v = i / pairs, cp = i % pairs;
    int x = v % e, y = (v / e) % e, z = v / (e * e);
    float2 s = float2(0.0f);
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) s += float2(p_x[vox_at(2 * z + dz, 2 * y + dy, 2 * x + dx, 2 * e, 1) * pairs + cp]);
    p_out[i] = half2(s * 0.125f);
}

// A bias-free 1x1 conv as a GEMM, y [m, n] = x [m, k] . w [n, k]^T in fp16, with each tile's IN partials of its
// columns in p_part [m tiles, n] (the skip projection). Grid (n tiles, m tiles), 128 threads
template <int BM, int BN>
kernel void nn_gemm_in(device half* p_x [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_y [[buffer(2)]],
                       device float4* p_part [[buffer(3)]], constant uint& m [[buffer(4)]],
                       constant uint& n [[buffer(5)]], constant uint& k [[buffer(6)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * BN];
    threadgroup float4 part[128];
    const int m0 = tg.y * BM, n0 = tg.x * BN;
    mat<half> tx(p_x, dextents<int32_t, 2>(k, m), array<int, 2>{1, (int)k});
    mat<half> tw(p_w, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    auto row_of = [&](int r) -> long { return m0 + r < (int)m ? (long)(m0 + r) : -1; };
    device const float* no_bias = nullptr;
    conv_epilogue<BM, BN>(acc, row_of, n0, n, no_bias, 0, p_y, tile, part, tid, p_part + (long)tg.y * n + n0, 1);
}

template [[host_name("nn_gemm_in_64x64")]] kernel void nn_gemm_in<64, 64>(device half*, device half*, device half*,
                                                                          device float4*, constant uint&,
                                                                          constant uint&, constant uint&, uint2, uint);
template [[host_name("nn_gemm_in_64x32")]] kernel void nn_gemm_in<64, 32>(device half*, device half*, device half*,
                                                                          device float4*, constant uint&,
                                                                          constant uint&, constant uint&, uint2, uint);

// The k2 s2 transposed conv as one GEMM with its scatter in the epilogue: g [m, 8 cout] = x [m, k] . w [8 cout, k]^T
// with x the unpadded lower level [e^3, k] and row (tap, co) of w the weight of tap 4a + 2b + c, channel co; element
// (voxel (z, y, x), tap, co) goes to the padded out [2e+2]^3 x cout at (2z + a, 2y + b, 2x + c) as
// r16(r16(g) + bias[co]), the GEMM's fp16 output plus PyTorch's separate bias add (u3h_up). Grid (8 cout / BN, m
// tiles), 128 threads; cout is a multiple of 8
//
// The tile is staged in threadgroup memory and stored 8 channels a thread: 8 adjacent columns share a tap, so each
// store is 16 contiguous bytes of one output voxel, where storing from the accumulator's own layout scattered 2-byte
// writes (18 ms a forward against about 7 ms for the GEMM at the roof)
template <int BM, int BN>
kernel void nn_up(device half* p_x [[buffer(0)]], device half* p_w [[buffer(1)]], device half* p_out [[buffer(2)]],
                  device const float* p_bias [[buffer(3)]], constant uint& e [[buffer(4)]],
                  constant uint& k [[buffer(5)]], constant uint& cout [[buffer(6)]],
                  uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[BM * BN];
    const int m = e * e * e, n = 8 * cout;
    const int m0 = tg.y * BM, n0 = tg.x * BN;
    mat<half> tx(p_x, dextents<int32_t, 2>(k, m), array<int, 2>{1, (int)k});
    mat<half> tw(p_w, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0];
        if (col >= n) continue;
        tile[idx[1] * BN + idx[0]] = half(r16(r16(acc[i]) + p_bias[col % cout]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tid; j < BM * BN / 8; j += 128) {
        int lr = j / (BN / 8), lc = 8 * (j % (BN / 8));
        int row = m0 + lr, col = n0 + lc;
        if (row >= m || col >= n) continue;
        int tap = col / cout, co = col % cout;
        int x = row % e, y = (row / e) % e, z = row / (e * e);
        long o = vox_at(2 * z + tap / 4, 2 * y + (tap / 2) % 2, 2 * x + tap % 2, 2 * e, 1);
        threadgroup const half4* src = (threadgroup const half4*)(tile + lr * BN + lc);
        device half4* dst = (device half4*)(p_out + o * cout + co);
        dst[0] = src[0];
        dst[1] = src[1];
    }
}

template [[host_name("nn_up_64x64")]] kernel void nn_up<64, 64>(device half*, device half*, device half*,
                                                                device const float*, constant uint&, constant uint&,
                                                                constant uint&, uint2, uint);

// The head of one flip's forward, un-flipped, into the float32 sum over flips (nn_head_tta): per patch voxel e and
// class k, the forward computed e at its mirror s, logit = r16(r16(sum_c x[s, c] hw[k, c]) + hb[k]); sum[k, e] = logit
// on the first flip, sum + logit after. x is the last decoder output [p^3, cols] unpadded
kernel void nn_head(device float* p_sum [[buffer(0)]], device const half2* p_x [[buffer(1)]],
                    device const float* p_hw [[buffer(2)]], device const float* p_hb [[buffer(3)]],
                    constant uint& cols [[buffer(4)]], constant uint& classes [[buffer(5)]],
                    constant uint& p [[buffer(6)]], constant uint& flip [[buffer(7)]],
                    constant uint& first [[buffer(8)]], uint e [[thread_position_in_grid]]) {
    uint vox = p * p * p, pairs = cols / 2;
    if (e >= vox) return;
    uint lx = e % p, ly = (e / p) % p, lz = e / (p * p);
    uint sx = (flip & 1) ? p - 1 - lx : lx;
    uint sy = (flip & 2) ? p - 1 - ly : ly;
    uint sz = (flip & 4) ? p - 1 - lz : lz;
    uint s = (sz * p + sy) * p + sx;
    for (uint k = 0; k < classes; k++) {
        float dot = 0.0f;
        for (uint cp = 0; cp < pairs; cp++) {
            float2 v = float2(p_x[(long)s * pairs + cp]);
            dot = fma(v.x, p_hw[k * cols + 2 * cp], dot);
            dot = fma(v.y, p_hw[k * cols + 2 * cp + 1], dot);
        }
        float logit = r16(r16(dot) + p_hb[k]);
        long o = (long)k * vox + e;
        p_sum[o] = first ? logit : p_sum[o] + logit;
    }
}

// Patch idx's status after its last flip, OR-ed into stat[idx] (nn_check): bit 1 when one of its fp16 mean logits
// r16(sum * inv_n) is not 0, bit 2 when a sum or mean is not finite. Each SIMD group reduces its lanes first and
// writes only bits the word lacks
kernel void nn_check(device atomic_uint* p_stat [[buffer(0)]], device const float* p_sum [[buffer(1)]],
                     constant float& inv_n [[buffer(2)]], constant uint& classes [[buffer(3)]],
                     constant uint& p [[buffer(4)]], constant uint& idx [[buffer(5)]],
                     uint e [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint vox = p * p * p;
    uint bits = 0;
    if (e < vox) {
        for (uint k = 0; k < classes; k++) {
            float s = p_sum[(long)k * vox + e];
            float mean = r16(s * inv_n);
            if (mean != 0.0f) bits |= 1u;
            if (!isfinite(s) || !isfinite(mean)) bits |= 2u;
        }
    }
    bits = simd_or(bits);
    // the word is set within the first few groups, so read it before writing: a SIMD group whose bits are already
    // there issues no atomic (7M voxels a patch otherwise send 221K atomics to one word, 5.6 ms)
    if (lane == 0 && bits != 0 && (atomic_load_explicit(&p_stat[idx], memory_order_relaxed) & bits) != bits)
        atomic_fetch_or_explicit(&p_stat[idx], bits, memory_order_relaxed);
}

// One patch's mean logits blended into the logit-difference accumulator: acc += m1 g - m0 g with m_k = r16(sum[k, e]
// inv_n) and g = gauss[e], each product, the difference and the sum rounded on their own. A binary softmax needs only
// l1 - l0, so one float32 plane replaces the CUDA engine's two class sums (half the accumulator memory, about 0.5 GB
// for a 512^3 zone); the weight sum is not stored either (nn_finalize recomputes it). The accumulator covers the
// canvas [c0, c0 + cd) on each axis, z stored modulo `ring` slices. Skipped when the patch's status lacks bit 1
kernel void nn_fold(device float* p_acc [[buffer(0)]], device const float* p_sum [[buffer(1)]],
                    device const float* p_gauss [[buffer(2)]], device const uint* p_stat [[buffer(3)]],
                    constant uint& idx [[buffer(4)]], constant float& inv_n [[buffer(5)]], constant uint& p [[buffer(6)]],
                    constant uint& pz [[buffer(7)]], constant uint& py [[buffer(8)]], constant uint& px [[buffer(9)]],
                    constant uint& cz0 [[buffer(10)]], constant uint& cy0 [[buffer(11)]],
                    constant uint& cx0 [[buffer(12)]], constant uint& cd [[buffer(13)]],
                    constant uint& ch [[buffer(14)]], constant uint& cw [[buffer(15)]],
                    constant uint& ring [[buffer(16)]], uint e [[thread_position_in_grid]]) {
    uint vox = p * p * p;
    if (e >= vox || (p_stat[idx] & 1u) == 0) return;
    uint lx = e % p, ly = (e / p) % p, lz = e / (p * p);
    uint gz = pz + lz, gy = py + ly, gx = px + lx;
    if (gz < cz0 || gz >= cz0 + cd || gy < cy0 || gy >= cy0 + ch || gx < cx0 || gx >= cx0 + cw) return;
    float w = p_gauss[e];
    long o = ((long)((gz - cz0) % ring) * ch + gy - cy0) * cw + gx - cx0;
    float m0 = r16(p_sum[e] * inv_n), m1 = r16(p_sum[(long)vox + e] * inv_n);
    p_acc[o] = p_acc[o] + (m1 * w - m0 * w);
}

// The plan's geometry for nn_finalize: the patch starts per axis (z, y, x, each ascending) and each patch's status,
// index (iz * ny + iy) * nx + ix in villa's order
struct Grid {
    uint nz, ny, nx, p;
};

// The blend weight at global voxel (gz, gy, gx): villa's wacc, the Gaussian of every blended patch that covers the
// voxel added in villa's patch order (z, then y, then x), so it is the CUDA engine's float32 sum bit for bit
static inline float nn_weight(uint gz, uint gy, uint gx, thread const Grid& g, device const uint* sz,
                              device const uint* sy, device const uint* sx, device const uint* stat,
                              device const float* gauss) {
    float w = 0.0f;
    uint p = g.p;
    for (uint iz = 0; iz < g.nz; iz++) {
        if (gz < sz[iz] || gz >= sz[iz] + p) continue;
        for (uint iy = 0; iy < g.ny; iy++) {
            if (gy < sy[iy] || gy >= sy[iy] + p) continue;
            for (uint ix = 0; ix < g.nx; ix++) {
                if (gx < sx[ix] || gx >= sx[ix] + p) continue;
                if ((stat[(iz * g.ny + iy) * g.nx + ix] & 1u) == 0) continue;
                w = w + gauss[((gz - sz[iz]) * p + gy - sy[iy]) * p + gx - sx[ix]];
            }
        }
    }
    return w;
}

// The stem: a 3x3x3 conv of the padded 1-channel input [e+2]^3 into the unpadded y [e^3, 32] with fp32 weights
// p_w [32, 27] (the fp16 values), stored r16(r16(acc) + bias), and the IN partials of each 128-voxel tile in
// p_part [tile, 32]. One thread per voxel: 27 x 32 FMAs a voxel, 0.25% of the forward's FLOPs, so not a GEMM
kernel void nn_stem(device half* p_y [[buffer(0)]], device const half* p_x [[buffer(1)]],
                    device const float* p_w [[buffer(2)]], device const float* p_bias [[buffer(3)]],
                    device float4* p_part [[buffer(4)]], constant uint& e [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int C = 32;
    threadgroup half tile[128 * C];
    threadgroup float4 part[128];
    uint vox = e * e * e;
    uint v = tg * 128 + tid;
    if (v < vox) {
        int x = v % e, y = (v / e) % e, z = v / (e * e);
        float in[27];
        for (int t = 0; t < 27; t++) in[t] = float(p_x[vox_at(z + t / 9 - 1, y + (t / 3) % 3 - 1, x + t % 3 - 1, e, 1)]);
        for (int co = 0; co < C; co++) {
            float acc = 0.0f;
            for (int t = 0; t < 27; t++) acc = fma(in[t], p_w[co * 27 + t], acc);
            half h = half(r16(r16(acc) + p_bias[co]));
            p_y[(long)v * C + co] = h;
            tile[tid * C + co] = h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = tg * 128;
    tile_stats<128, C, 128>(tile, part, tid, [&](int j) { return base + j < vox; }, p_part + (long)tg * C);
}

// Zeroes `slices` ring slots of the logit accumulators from slot `slot0` (wrapping at `ring`), every class: one
// thread per float of a slot's plane `elems`
kernel void nn_zero_slices(device float* p_acc [[buffer(0)]], constant uint& classes [[buffer(1)]],
                           constant uint& ring [[buffer(2)]], constant uint& elems [[buffer(3)]],
                           constant uint& slot0 [[buffer(4)]], constant uint& slices [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= slices * elems) return;
    uint s = (slot0 + i / elems) % ring;
    long o = (long)s * elems + i % elems;
    for (uint k = 0; k < classes; k++) p_acc[(long)k * ring * elems + o] = 0.0f;
}

// The logit accumulators, their canvas and the output row they finalise into
struct Fin {
    uint cz0, cy0, cx0;  // canvas corner (global)
    uint cd, ch, cw;     // canvas extent
    uint ring;           // accumulator z slots
    uint y0, x0;         // the output row's corner on y and x (on the chunk grid)
    uint rh, rw;         // the output row's extent on y and x
    uint chunk;
    uint cutoff_on;
    float cutoff;
};

// The normalised logit difference l1 - l0 at global voxel (gz, gy, gx) and its weight: villa divides the blended sums
// by `w + 1e-8` where `w > 0` and leaves them (0) elsewhere; voxels off the canvas have neither
static inline float2 nn_logits(uint gz, uint gy, uint gx, thread const Fin& f, thread const Grid& g, device const float* acc,
                               device const uint* sz, device const uint* sy, device const uint* sx,
                               device const uint* stat, device const float* gauss) {
    if (gz < f.cz0 || gz >= f.cz0 + f.cd || gy < f.cy0 || gy >= f.cy0 + f.ch || gx < f.cx0 || gx >= f.cx0 + f.cw)
        return float2(0.0f);
    long o = ((long)((gz - f.cz0) % f.ring) * f.ch + gy - f.cy0) * f.cw + gx - f.cx0;
    float w = nn_weight(gz, gy, gx, g, sz, sy, sx, stat, gauss);
    float d = acc[o];
    return float2(w > 0.0f ? d / (w + 1e-8f) : d, w);
}

// The logit difference at each output chunk's corner of the row starting at global slice gz (nn_finalize's `first`,
// the value the chunk's allclose test compares against): one thread per chunk of the row
kernel void nn_first(device float* p_first [[buffer(0)]], device const float* p_acc [[buffer(1)]],
                     device const uint* p_sz [[buffer(2)]], device const uint* p_sy [[buffer(3)]],
                     device const uint* p_sx [[buffer(4)]], device const uint* p_stat [[buffer(5)]],
                     device const float* p_gauss [[buffer(6)]], device const Fin& p_fin [[buffer(7)]],
                     device const Grid& p_grid [[buffer(8)]], constant uint& gz [[buffer(9)]],
                     uint i [[thread_position_in_grid]]) {
    Fin f = p_fin;
    Grid g = p_grid;
    uint nx = (f.rw + f.chunk - 1) / f.chunk, ny = (f.rh + f.chunk - 1) / f.chunk;
    if (i >= ny * nx) return;
    uint cy = i / nx, cx = i % nx;
    p_first[i] = nn_logits(gz, f.y0 + cy * f.chunk, f.x0 + cx * f.chunk, f, g, p_acc, p_sz, p_sy, p_sx, p_stat, p_gauss).x;
}

// `slices` z-slices of an output chunk row from global slice gz0, finalised as the CUDA engine's nn_finalize does
// (villa's _write_blended_chunk and apply_finalization for a binary model) into p_out [row slice, rh, rw] at row
// slice gz0 - rz0, with the row's chunk flags OR-ed into p_flags [2 chunk]: 1 when the chunk has weight above 0,
// and 1 at 2 chunk + 1 when the chunk's logits are not all allclose to its corner's. The weight is recomputed, not
// read. The CUDA engine keeps both class sums and compares both with the corner's class 0; this accumulator holds
// only l1 - l0, so a voxel is not allclose when its difference is not allclose to the corner's (p_first) or is off
// 0 by more than the tolerance's absolute 1e-8 (the classes differ). Two cases still differ from the CUDA engine: a
// covered voxel whose blended logits are both exactly 0 gives 127 here (0 there), and a chunk whose difference is
// constant either way of the tolerance's relative part (1e-6 of |l0|, unknown here), or within 1e-8 of 0 while l0
// varies, is kept or skipped on the difference alone. Real predictions have neither outside one-voxel chunks, which
// the host refuses, and the gate's chunk diff would show either
kernel void nn_finalize(device uchar* p_out [[buffer(0)]], device atomic_uint* p_flags [[buffer(1)]],
                        device const float* p_first [[buffer(2)]], device const float* p_acc [[buffer(3)]],
                        device const uint* p_sz [[buffer(4)]], device const uint* p_sy [[buffer(5)]],
                        device const uint* p_sx [[buffer(6)]], device const uint* p_stat [[buffer(7)]],
                        device const float* p_gauss [[buffer(8)]], device const Fin& p_fin [[buffer(9)]],
                        device const Grid& p_grid [[buffer(10)]], constant uint& gz0 [[buffer(11)]],
                        constant uint& slices [[buffer(12)]], constant uint& rz0 [[buffer(13)]],
                        uint e [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    Fin f = p_fin;
    Grid g = p_grid;
    // no early return: every lane joins the SIMD-group flag reduction below
    bool live = e < slices * f.rh * f.rw;
    uint i = live ? e : 0;
    uint x = i % f.rw, y = (i / f.rw) % f.rh, s = i / (f.rw * f.rh);
    uint gz = gz0 + s;
    float2 l = nn_logits(gz, f.y0 + y, f.x0 + x, f, g, p_acc, p_sz, p_sy, p_sx, p_stat, p_gauss);
    float d = l.x;
    uchar v;
    if (d == 0.0f && l.y == 0.0f) {
        // uncovered: no weight, so villa's logits are both the zeros it started from
        v = 0;
    } else if (f.cutoff_on != 0) {
        v = d > f.cutoff ? 255 : 0;
    } else {
        // villa's softmax exp(l - max) / sum with the larger logit's term exactly 1
        float e = exp(-fabs(d));
        float prob = d >= 0.0f ? 1.0f / (e + 1.0f) : e / (1.0f + e);
        v = uchar(fmin(fmax(prob * 255.0f, 0.0f), 255.0f));
    }
    if (live) p_out[((long)(gz - rz0) * f.rh + y) * f.rw + x] = v;
    uint nx = (f.rw + f.chunk - 1) / f.chunk;
    uint c = (y / f.chunk) * nx + x / f.chunk;
    float first = p_first[c];
    float tol = 1e-8f + 1e-6f * fabs(first);
    uint bits = 0;
    if (live && l.y > 0.0f) bits |= 1u;
    if (live && !(fabs(d - first) <= tol && fabs(d) <= 1e-8f)) bits |= 2u;
    // one atomic per SIMD group when its lanes share a chunk (all but the lanes at a chunk edge), so 7M voxels do
    // not contend on a few dozen words
    if (simd_all(c == simd_broadcast_first(c))) {
        bits = simd_or(bits);
        if (lane != 0) bits = 0;
    }
    if (bits & 1u) atomic_fetch_or_explicit(&p_flags[2 * c], 1u, memory_order_relaxed);
    if (bits & 2u) atomic_fetch_or_explicit(&p_flags[2 * c + 1], 1u, memory_order_relaxed);
}

// Zeroes the one-voxel border of a padded [e+2]^3 x c buffer, two channels a thread: its two z faces, then the y
// faces of the interior planes, then the x faces of the interior lines. A buffer that other levels' layouts wrote
// into gets this before it serves as a conv input
kernel void nn_border(device half2* p_x [[buffer(0)]], constant uint& e [[buffer(1)]], constant uint& c [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    uint pairs = c / 2, ep = e + 2;
    uint fz = ep * ep, fy = e * ep, fx = e * e;
    uint v = i / pairs, cp = i % pairs;
    if (v >= 2 * (fz + fy + fx)) return;
    uint z, y, x;
    if (v < 2 * fz) {
        z = v < fz ? 0 : ep - 1;
        y = (v % fz) / ep;
        x = v % ep;
    } else if ((v -= 2 * fz) < 2 * fy) {
        z = 1 + (v % fy) / ep;
        y = v < fy ? 0 : ep - 1;
        x = v % ep;
    } else {
        v -= 2 * fy;
        z = 1 + (v % fx) / e;
        y = 1 + (v % fx) % e;
        x = v < fx ? 0 : ep - 1;
    }
    p_x[((long)(z * ep + y) * ep + x) * pairs + cp] = half2(0.0h);
}
