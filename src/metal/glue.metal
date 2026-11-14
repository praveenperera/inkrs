// inkrs glue kernels for Metal: the MSL ports of inkrs-kernels/src/{tiles,inorm,stem,layout,blend}.rs
//
// Activations are `real` channels-last (NHWC, NDHWC; common.metal), FP16 as on CUDA or FP32. Every rounding the CUDA
// kernels make to copy PyTorch FP16 autocast is kept: `rnd` rounds a float to `real` (through half, round to nearest
// even, at FP16; exact at FP32, where every op runs in float as villa's eager PyTorch on MPS), and products and sums
// that villa or numpy round separately stay separate, so FP contraction is off for the whole file
//
// Buffers carry no length on Metal, so every kernel takes the element counts it bounds-checks against

#pragma METAL fp contract(off)

// The tile edge (128 for reader-v2, 384 for Reader V3) is a kernel argument: Metal compiles at load, so one kernel
// serves both edges, where the CUDA build keeps `_384` copies to leave its 128 PTX unchanged
constant uint DESC = 8;
constant uint DEPTH = 17;

static inline float rnd(float x) { return float(real(x)); }
static inline float leaky(float x) { return x > 0.0f ? x : rnd(x * 0.01f); }

// ---------------------------------------------------------------- tiles: gather and villa normalize_robust

kernel void zero_u32(device uint* p_x [[buffer(0)]], constant uint& count [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < count) p_x[i] = 0;
}

// Histogram of every tile's voxels; grid (DEPTH, tiles) threadgroups of 256, as tile_hist on CUDA
kernel void tile_hist(device atomic_uint* p_hist [[buffer(0)]],
                      device const uchar* p_vol [[buffer(1)]],
                      device const int* p_desc [[buffer(2)]],
                      constant uint& height [[buffer(3)]],
                      constant uint& width [[buffer(4)]],
                      constant uint& tiles [[buffer(5)]],
                      constant uint& edge [[buffer(6)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint bins[256];
    uint z = tg.x, b = tg.y;
    if (b >= tiles) return;
    uint d = b * DESC;
    uint y0 = uint(p_desc[d]), x0 = uint(p_desc[d + 1]);
    int l0 = p_desc[d + 2], step = p_desc[d + 3];
    uint layer = uint(l0 + step * int(z));
    uint plane = layer * height * width;
    atomic_store_explicit(&bins[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < edge * edge; i += 256) {
        uint y = i / edge, x = i % edge;
        uint v = p_vol[plane + (y0 + y) * width + x0 + x];
        atomic_fetch_add_explicit(&bins[v], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = atomic_load_explicit(&bins[tid], memory_order_relaxed);
    if (count != 0) atomic_fetch_add_explicit(&p_hist[b * 256 + tid], count, memory_order_relaxed);
}

static inline float order_stat(device const uint* h, uint k) {
    uint cum = 0;
    for (uint v = 0; v < 256; v++) {
        cum += h[v];
        if (cum > k) return float(v);
    }
    return 255.0f;
}

// numpy's float32 `_lerp`, see tiles.rs lerp32
static inline float lerp32(float a, float b, float t, float omt) {
    float d = b - a;
    if (t >= 0.5f) {
        float p = d * omt;
        return b - p;
    }
    float p = d * t;
    return a + p;
}

static inline float clipf(float v, float lo, float hi) { return fmin(fmax(v, lo), hi); }

// Per tile [lo, hi, median, divisor] of villa normalize_robust, one thread per tile, as tile_stats on CUDA
kernel void tile_stats(device float4* p_stats [[buffer(0)]],
                       device const uint* p_hist [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant uint& lo_index [[buffer(3)]],
                       constant float& lo_t [[buffer(4)]],
                       constant float& lo_omt [[buffer(5)]],
                       constant uint& hi_index [[buffer(6)]],
                       constant float& hi_t [[buffer(7)]],
                       constant float& hi_omt [[buffer(8)]],
                       constant uint& tiles [[buffer(9)]],
                       uint b [[thread_position_in_grid]]) {
    if (b >= tiles || n < 2) return;
    device const uint* h = p_hist + b * 256;
    float lo = lerp32(order_stat(h, lo_index), order_stat(h, lo_index + 1), lo_t, lo_omt);
    float hi = lerp32(order_stat(h, hi_index), order_stat(h, hi_index + 1), hi_t, hi_omt);
    uint k1 = (n % 2 == 0) ? n / 2 - 1 : n / 2;
    uint k2 = n / 2;
    float m1 = clipf(order_stat(h, k1), lo, hi);
    float m2 = clipf(order_stat(h, k2), lo, hi);
    float median = (k1 == k2) ? m1 : (m1 + m2) / 2.0f;

    uint right = 0;
    while (right < 256 && clipf(float(right), lo, hi) <= median) right++;
    int left = int(right) - 1;
    uint cum = 0;
    float d1 = 0.0f, d2 = 0.0f;
    bool found1 = false;
    for (;;) {
        float dl = left >= 0 ? median - clipf(float(left), lo, hi) : INFINITY;
        float dr = right < 256 ? clipf(float(right), lo, hi) - median : INFINITY;
        if (isinf(dl) && isinf(dr)) break;
        float dev;
        uint count;
        if (dl <= dr) {
            count = h[left];
            left--;
            dev = dl;
        } else {
            count = h[right];
            right++;
            dev = dr;
        }
        cum += count;
        if (!found1 && cum > k1) {
            d1 = dev;
            found1 = true;
        }
        if (cum > k2) {
            d2 = dev;
            break;
        }
    }
    float mad = (k1 == k2) ? d2 : (d1 + d2) / 2.0f;
    // villa takes 1.4826 * mad in float64 and casts to float32; Metal has no float64, so the product is formed
    // exactly as the float32 pair c_hi + c_lo times mad with fma, and rounded once at the end. The double-rounding
    // difference against float64 is at most one float32 ulp of the divisor
    const float c_hi = 1.48259997367858886719f;  // float32 nearest to 1.4826
    const float c_lo = 2.6321411e-08f;           // 1.4826 - c_hi
    float scaled = fma(mad, c_hi, mad * c_lo);
    float divisor;
    if (scaled >= 1e-6f) {
        divisor = scaled;
    } else {
        float span = (hi - lo) / 2.0f;
        divisor = -(span >= 1e-6f ? span : 1.0f);
    }
    p_stats[b] = float4(lo, hi, median, divisor);
}

// numpy's pairwise-sum tree on one tile has 2048 nodes at both edges: at 128 a node is 136 voxels (a 64- and a
// 72-voxel leaf); at 384 it is 1224, which numpy splits at multiples of 8 as (304 + 304) + (304 + (152 + 160)), with
// leaves 152 = 72 + 80 and 160 = 80 + 80 (CUDA's tiles384.rs, checked there against the host's pairwise_f32)
constant uint NODES = 2048;

// The voxel source of one tile sum: the volume, the tile's first plane offset and plane step, its first row offset,
// the volume width and the tile edge
struct Src {
    device const uchar* vol;
    int p0, sp;
    uint row0, width, edge;
};

static inline float voxel(Src s, uint e) {
    uint plane_px = s.edge * s.edge;
    uint z = e / plane_px, yx = e % plane_px;
    uint plane = uint(s.p0 + s.sp * int(z));
    return float(s.vol[plane + s.row0 + (yx / s.edge) * s.width + (yx % s.edge)]);
}

static inline float leaf(Src s, uint start, uint len, float mean, bool square) {
    float r[8];
    for (uint j = 0; j < 8; j++) {
        float v = voxel(s, start + j);
        if (square) {
            float d = v - mean;
            v = d * d;
        }
        r[j] = v;
    }
    for (uint i = 8; i < len; i += 8) {
        for (uint j = 0; j < 8; j++) {
            float v = voxel(s, start + i + j);
            if (square) {
                float d = v - mean;
                v = d * d;
            }
            r[j] = r[j] + v;
        }
    }
    return ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
}

static inline float s152(Src s, uint a, float m, bool q) { return leaf(s, a, 72, m, q) + leaf(s, a + 72, 80, m, q); }
static inline float s160(Src s, uint a, float m, bool q) { return leaf(s, a, 80, m, q) + leaf(s, a + 80, 80, m, q); }
static inline float s304(Src s, uint a, float m, bool q) { return s152(s, a, m, q) + s152(s, a + 152, m, q); }
static inline float s312(Src s, uint a, float m, bool q) { return s152(s, a, m, q) + s160(s, a + 152, m, q); }

// One node of the tree: 136 voxels at a 128 edge, 1224 at 384
static inline float node(Src s, uint k, float m, bool q) {
    if (s.edge == 128) {
        uint a = k * 136;
        return leaf(s, a, 64, m, q) + leaf(s, a + 64, 72, m, q);
    }
    uint a = k * 1224;
    return (s304(s, a, m, q) + s304(s, a + 304, m, q)) + (s304(s, a + 608, m, q) + s312(s, a + 912, m, q));
}

static float tile_sum(threadgroup float* nodes, Src src, float mean, bool square, uint tid) {
    for (uint k = tid; k < NODES; k += 256) nodes[k] = node(src, k, mean, square);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint m = NODES; m > 1; m /= 2) {
        uint half_m = m / 2;
        float r[4];
        for (uint q = 0; q < 4; q++) {
            uint i = tid + 256 * q;
            r[q] = i < half_m ? nodes[2 * i] + nodes[2 * i + 1] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0; q < 4; q++) {
            uint i = tid + 256 * q;
            if (i < half_m) nodes[i] = r[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return nodes[0];
}

// numpy's exact float32 np.std for the tiles tile_stats marked; grid (tiles) threadgroups of 256, edge 128 or 384
kernel void tile_std(device float4* p_stats [[buffer(0)]],
                     device const uchar* p_vol [[buffer(1)]],
                     device const int* p_desc [[buffer(2)]],
                     constant uint& height [[buffer(3)]],
                     constant uint& width [[buffer(4)]],
                     constant uint& tiles [[buffer(5)]],
                     constant uint& edge [[buffer(6)]],
                     uint b [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tree[2048];
    if (b >= tiles) return;
    float marker = p_stats[b].w;
    if (marker >= 0.0f) return;
    uint d = b * DESC;
    uint y0 = uint(p_desc[d]), x0 = uint(p_desc[d + 1]);
    int l0 = p_desc[d + 2], step = p_desc[d + 3];
    int hw = int(height * width);
    Src src = {p_vol, l0 * hw, step * hw, y0 * width + x0, width, edge};
    float n = float(DEPTH * edge * edge);
    float mean = tile_sum(tree, src, 0.0f, false, tid) / n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = tile_sum(tree, src, mean, true, tid) / n;
    float sd = sqrt(var);
    if (tid == 0) p_stats[b].w = sd >= 1e-6f ? sd : -marker;
}

// Normalised `real` input [tiles, DEPTH, edge, edge], two voxels per thread, as tile_norm on CUDA
kernel void tile_norm(device real2* p_out [[buffer(0)]],
                      device const uchar* p_vol [[buffer(1)]],
                      device const int* p_desc [[buffer(2)]],
                      device const float4* p_stats [[buffer(3)]],
                      constant uint& height [[buffer(4)]],
                      constant uint& width [[buffer(5)]],
                      constant uint& count [[buffer(6)]],
                      constant uint& edge [[buffer(7)]],
                      uint e [[thread_position_in_grid]]) {
    if (e >= count) return;
    uint half_plane = edge * edge / 2;
    uint per_tile = DEPTH * half_plane;
    uint b = e / per_tile, r = e % per_tile;
    uint z = r / half_plane, yx = r % half_plane;
    uint y = yx / (edge / 2), x = (yx % (edge / 2)) * 2;
    uint d = b * DESC;
    float4 s = p_stats[b];
    uint y0 = uint(p_desc[d]), x0 = uint(p_desc[d + 1]);
    int l0 = p_desc[d + 2], step = p_desc[d + 3];
    uint layer = uint(l0 + step * int(z));
    uint i = layer * height * width + (y0 + y) * width + x0 + x;
    float a = (clipf(float(p_vol[i]), s.x, s.y) - s.z) / s.w;
    float c = (clipf(float(p_vol[i + 1]), s.x, s.y) - s.z) / s.w;
    p_out[e] = real2(real(a), real(c));
}

// ---------------------------------------------------------------- InstanceNorm, as inorm.rs

// Error-free sum of two floats (Knuth TwoSum), for the float64 accumulations of the CUDA kernels (Hecate's GroupNorm)
static inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return float2(s, err);
}

// A double-float accumulator: hi + lo
static inline float2 df_add(float2 acc, float v) {
    float2 s = two_sum(acc.x, v);
    float lo = s.y + acc.y;
    return two_sum(s.x, lo);
}

// Per (sample, channel) [mean, invstd] from the conv epilogue's partials [B, tiles, cols] (`tile_stats`). Grid (cols /
// 16, B) of 256 threads: each of 16 threads per column merges a contiguous stripe of tiles in order, then the first
// merges the stripes in order, so the result is the same on every pass. CUDA forms these in float64 from shifted sums;
// the per-tile two-pass partials keep float's error at about one ulp of the variance: well under the apply's rounding
// at FP16, and at FP32 the size of PyTorch's own float statistics' error
kernel void in_finalize(device float2* p_stats [[buffer(0)]],
                        device const float4* p_part [[buffer(1)]],
                        constant uint& tiles [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant float& eps [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 part[256];
    uint c = tg.x * 16 + tid % 16, k = tid / 16, b = tg.y;
    uint stripe = (tiles + 15) / 16;
    uint lo = min(k * stripe, tiles), hi = min(lo + stripe, tiles);
    device const float4* p = p_part + (ulong)b * tiles * cols + c;
    float4 a = 0.0f;
    for (uint t = lo; t < hi; t++) a = chan_merge(a, p[(ulong)t * cols]);
    part[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0) {
        for (uint q = 1; q < 16; q++) a = chan_merge(a, part[q * 16 + tid]);
        float var = a.z / a.x;
        var = var > 0.0f ? var : 0.0f;
        p_stats[b * cols + c] = float2(a.y, 1.0f / sqrt(var + eps));
    }
}

// One normalised channel value as CUDA's in_apply rounds it: rnd(norm(v) * w + bias)
static inline float in_one(float v, float2 s, float w, float bias) {
    float t = (v - s.x) * s.y;
    t = t * w;
    return rnd(t + bias);
}

// in_apply's arithmetic for the 8 channels from c0 of sample b: then the optional residual add and LeakyReLU
static inline void in_eight(thread real4* v, uint b, uint c0, uint cols, device const float2* p_stats,
                            device const float* p_w, device const float* p_bias, device const real4* res, uint flags) {
    for (uint h = 0; h < 2; h++) {
        float4 y;
        for (uint j = 0; j < 4; j++) {
            uint c = c0 + 4 * h + j;
            y[j] = in_one(float(v[h][j]), p_stats[b * cols + c], p_w[c], p_bias[c]);
        }
        if (flags & 2) {
            float4 r = float4(res[h]);
            for (uint j = 0; j < 4; j++) y[j] = rnd(y[j] + r[j]);
        }
        if (flags & 1) {
            for (uint j = 0; j < 4; j++) y[j] = leaky(y[j]);
        }
        v[h] = real4(y);
    }
}

// x = leaky?(rnd(rnd(norm(x) * w + bias) + residual?)) in place, eight channels per thread (cols a multiple of 8)
kernel void in_apply(device real4* p_x [[buffer(0)]],
                     device const real4* p_residual [[buffer(1)]],
                     device const float2* p_stats [[buffer(2)]],
                     device const float* p_w [[buffer(3)]],
                     device const float* p_bias [[buffer(4)]],
                     constant uint& rows [[buffer(5)]],
                     constant uint& cols [[buffer(6)]],
                     constant uint& flags [[buffer(7)]],  // 1 LeakyReLU, 2 residual add
                     constant uint& count [[buffer(8)]],
                     uint e [[thread_position_in_grid]]) {
    if (e >= count) return;
    uint oct = cols / 8;
    uint pix = e / oct, q = e % oct;
    real4 v[2] = {p_x[2 * e], p_x[2 * e + 1]};
    in_eight(v, pix / rows, 8 * q, cols, p_stats, p_w, p_bias, p_residual + 2 * e, flags);
    p_x[2 * e] = v[0];
    p_x[2 * e + 1] = v[1];
}

// in_apply written into the zero-padded layout the next tap conv reads, so that conv needs no pad pass: x [B, depth,
// h, w, cols] to xp [B, depth + 2 dpad, h + 2, w + 2, cols] followed by zero rows up to `count` octets (dpad 1 pads
// depth too, for the stem's conv3d). Flags as in_apply (1 LeakyReLU, 2 residual add) plus 4: also write the result
// back into x, for activations other layers read unpadded (residuals, pools). One thread per 8 channels of xp
kernel void in_apply_pad(device real4* p_xp [[buffer(0)]],
                         device real4* p_x [[buffer(1)]],
                         device const real4* p_residual [[buffer(2)]],
                         device const float2* p_stats [[buffer(3)]],
                         device const float* p_w [[buffer(4)]],
                         device const float* p_bias [[buffer(5)]],
                         constant uint& batch [[buffer(6)]],
                         constant uint& depth [[buffer(7)]],
                         constant uint& h [[buffer(8)]],
                         constant uint& w [[buffer(9)]],
                         constant uint& cols [[buffer(10)]],
                         constant uint& dpad [[buffer(11)]],
                         constant uint& flags [[buffer(12)]],
                         constant uint& count [[buffer(13)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint oct = cols / 8, wp = w + 2, hp = h + 2, dp = depth + 2 * dpad;
    uint q = i % oct, r = i / oct;
    uint plane = hp * wp, per = dp * plane;
    uint b = r / per, rem = r % per;
    uint z = rem / plane, yx = rem % plane;
    uint y = yx / wp, x = yx % wp;
    // rows past the last sample are the tail, zero
    bool inside = b < batch && z >= dpad && z < depth + dpad && y >= 1 && y <= h && x >= 1 && x <= w;
    real4 v[2] = {real4(0), real4(0)};
    if (inside) {
        uint e = (((b * depth + z - dpad) * h + y - 1) * w + x - 1) * oct + q;
        v[0] = p_x[2 * e];
        v[1] = p_x[2 * e + 1];
        // the same operation order as in_apply, so both round identically
        in_eight(v, b, 8 * q, cols, p_stats, p_w, p_bias, p_residual + 2 * e, flags);
        if (flags & 4) {
            p_x[2 * e] = v[0];
            p_x[2 * e + 1] = v[1];
        }
    }
    p_xp[2 * i] = v[0];
    p_xp[2 * i + 1] = v[1];
}

// ---------------------------------------------------------------- stem tail, as stem.rs

// cat(sum_z softmax_z(att) * f, max_z f), 32 `real` channels per pixel; one thread per (sample, pixel)
kernel void depth_pool(device real* p_out [[buffer(0)]],
                       device const real* p_feat [[buffer(1)]],
                       device const float* p_w [[buffer(2)]],
                       constant float& bias [[buffer(3)]],
                       constant uint& pixels [[buffer(4)]],
                       constant uint& count [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    const uint Z = 17, C = 16;
    if (i >= count) return;
    uint b = i / pixels, p = i % pixels;
    uint base = b * Z * pixels + p;
    float logits[17];
    float top = -INFINITY;
    for (uint z = 0; z < Z; z++) {
        device const real* f = p_feat + (base + z * pixels) * C;
        float dot = 0.0f;
        for (uint k = 0; k < C / 2; k++) {
            float a = float(f[2 * k]), c = float(f[2 * k + 1]);
            float pa = a * p_w[2 * k];
            float pc = c * p_w[2 * k + 1];
            dot = dot + (pa + pc);
        }
        float l = rnd(rnd(dot) + bias);
        logits[z] = l;
        top = fmax(top, l);
    }
    float sum = 0.0f;
    for (uint z = 0; z < Z; z++) {
        float e = exp(logits[z] - top);
        logits[z] = e;
        sum += e;
    }
    float att[16], mx[16];
    for (uint k = 0; k < C; k++) {
        att[k] = 0.0f;
        mx[k] = -INFINITY;
    }
    for (uint z = 0; z < Z; z++) {
        float wz = rnd(logits[z] / sum);
        device const real* f = p_feat + (base + z * pixels) * C;
        for (uint k = 0; k < C; k++) {
            float a = float(f[k]);
            att[k] += rnd(a * wz);
            mx[k] = fmax(mx[k], a);
        }
    }
    device real* o = p_out + i * 32;
    for (uint k = 0; k < C; k++) {
        o[k] = real(att[k]);
        o[C + k] = real(mx[k]);
    }
}

// ---------------------------------------------------------------- 2D glue, as layout.rs

// 2x2 stride-2 average pool, [B, H, W, C] to [B, H/2, W/2, C], in pairs
kernel void avgpool2(device real2* p_out [[buffer(0)]],
                     device const real2* p_x [[buffer(1)]],
                     constant uint& out_h [[buffer(2)]],
                     constant uint& out_w [[buffer(3)]],
                     constant uint& cols [[buffer(4)]],
                     constant uint& count [[buffer(5)]],
                     uint e [[thread_position_in_grid]]) {
    if (e >= count) return;
    uint pairs = cols / 2;
    uint pix = e / pairs, cp = e % pairs;
    uint ox = pix % out_w, rest = pix / out_w;
    uint oy = rest % out_h, b = rest / out_h;
    uint h = 2 * out_h, w = 2 * out_w;
    uint row = ((b * h + 2 * oy) * w + 2 * ox) * pairs + cp;
    float2 a = float2(p_x[row]), bb = float2(p_x[row + pairs]);
    float2 c = float2(p_x[row + w * pairs]), d = float2(p_x[row + w * pairs + pairs]);
    p_out[e] = real2(real((((a.x + bb.x) + c.x) + d.x) / 4.0f), real((((a.y + bb.y) + c.y) + d.y) / 4.0f));
}

// The decoder's cat(transpconv(x) + bias, skip) at the upsampled size, written straight into the zero-padded layout
// of the tap conv that reads it ([B, height + 2, width + 2, up + skip_cols], then zero tail rows up to `count` pairs);
// g is [B*H/2*W/2, 4*up] with column (dy*2+dx)*up + c
kernel void cat_up(device real2* p_out [[buffer(0)]],
                   device const real* p_g [[buffer(1)]],
                   device const float* p_bias [[buffer(2)]],
                   device const real2* p_skip [[buffer(3)]],
                   constant uint& batch [[buffer(4)]],
                   constant uint& height [[buffer(5)]],
                   constant uint& width [[buffer(6)]],
                   constant uint& up [[buffer(7)]],
                   constant uint& skip_cols [[buffer(8)]],
                   constant uint& count [[buffer(9)]],
                   uint e [[thread_position_in_grid]]) {
    if (e >= count) return;
    uint pairs = (up + skip_cols) / 2;
    uint pix = e / pairs, cp = e % pairs;
    uint wp = width + 2, per = (height + 2) * wp;
    uint b = pix / per, rem = pix % per;
    uint yp = rem / wp, xp = rem % wp;
    // padding and the tail past the last sample are zero
    if (b >= batch || yp < 1 || yp > height || xp < 1 || xp > width) {
        p_out[e] = real2(0);
        return;
    }
    uint y = yp - 1, x = xp - 1;
    uint c = 2 * cp;
    if (c < up) {
        uint m = (b * (height / 2) + y / 2) * (width / 2) + x / 2;
        uint d = (y & 1) * 2 + (x & 1);
        uint gi = m * 4 * up + d * up + c;
        float v0 = float(p_g[gi]), v1 = float(p_g[gi + 1]);
        p_out[e] = real2(real(rnd(v0) + p_bias[c]), real(rnd(v1) + p_bias[c + 1]));
    } else {
        uint q = (b * height + y) * width + x;
        p_out[e] = p_skip[q * (skip_cols / 2) + (c - up) / 2];
    }
}

// ---------------------------------------------------------------- head and blend, as blend.rs

// Head logits of every real tile into its slot; one thread per (tile, pixel); padding tiles (slot < 0) write nothing
kernel void head_logits(device real* p_logits [[buffer(0)]],
                        device const real* p_feat [[buffer(1)]],
                        device const float* p_w [[buffer(2)]],
                        constant float& bias [[buffer(3)]],
                        device const int* p_desc [[buffer(4)]],
                        constant uint& count [[buffer(5)]],
                        constant uint& slots [[buffer(6)]],
                        constant uint& pixels [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint b = i / pixels, p = i % pixels;
    int slot = p_desc[b * DESC + 4];
    if (slot < 0) return;
    device const real* f = p_feat + i * 32;
    float dot = 0.0f;
    for (uint k = 0; k < 16; k++) {
        float a = float(f[2 * k]), c = float(f[2 * k + 1]);
        float pa = a * p_w[2 * k];
        float pc = c * p_w[2 * k + 1];
        dot = dot + (pa + pc);
    }
    // 64-bit: a large sweep's logits pass 2^32 elements (17x8192x8192, 18 maps is 4.8e9)
    if (uint(slot) < slots) p_logits[ulong(slot) * pixels + p] = real(rnd(dot) + bias);
}

// One map: blend its tiles' logits in villa's tile order and write villa's uint8 output
kernel void blend_finish(device uchar* p_out [[buffer(0)]],
                         device uchar* p_cov [[buffer(1)]],
                         device const real* p_logits [[buffer(2)]],
                         device const int* p_grid [[buffer(3)]],
                         device const uint* p_ys [[buffer(4)]],
                         device const uint* p_xs [[buffer(5)]],
                         device const uint2* p_y_span [[buffer(6)]],
                         device const uint2* p_x_span [[buffer(7)]],
                         device const float* p_weight [[buffer(8)]],
                         constant uint& width [[buffer(9)]],
                         constant uint& count [[buffer(10)]],
                         constant uint& nx [[buffer(11)]],
                         constant uint& has_cov [[buffer(12)]],
                         constant uint& edge [[buffer(13)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint y = i / width, x = i % width;
    uint2 ysp = p_y_span[y], xsp = p_x_span[x];
    float num = 0.0f, den = 0.0f;
    for (uint iy = ysp.x; iy < ysp.y; iy++) {
        uint ly = y - p_ys[iy];
        for (uint ix = xsp.x; ix < xsp.y; ix++) {
            int slot = p_grid[iy * nx + ix];
            if (slot >= 0) {
                uint lx = x - p_xs[ix];
                float l = float(p_logits[ulong(slot) * edge * edge + ly * edge + lx]);
                float p = 1.0f / (1.0f + exp(-l));
                float w = p_weight[ly * edge + lx];
                float pw = p * w;
                num = num + pw;
                den = den + w;
            }
        }
    }
    bool covered = den > 1e-6f;
    float p = covered ? num / den : num;
    p = fmin(fmax(p, 0.0f), 1.0f);
    float scaled = p * 255.0f;
    p_out[i] = uchar(scaled);
    if (has_cov) p_cov[i] = covered ? 1 : 0;
}
