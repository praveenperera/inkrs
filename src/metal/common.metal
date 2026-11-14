// Shared header of the inkrs Metal kernels, compiled first. Pointer parameters are named `p_*`: the host's argument
// check reads the names to tell a pointer from a `constant uint&` scalar
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename T>
using mat = tensor<device T, dextents<int32_t, 2>, tensor_inline>;

// The storage type of the hybrid_3d2d reader's activations, conv weights and logits (conv.metal, glue.metal): half, as
// PyTorch's FP16 autocast stores them on CUDA, or float when the host builds the fp32 reader library with INKRS_FP32,
// as villa runs on MPS. Accumulation and every statistic stay float in both
#ifdef INKRS_FP32
typedef float real;
typedef float2 real2;
typedef float4 real4;
#else
typedef half real;
typedef half2 real2;
typedef half4 real4;
#endif

// InstanceNorm partials are (n, mean, M2, 0) over a run of rows of one sample and channel; two runs merge as Chan et
// al. do, so every conv tile's partial combines in a fixed order and a second pass reproduces the first bit for bit
static inline float4 chan_merge(float4 a, float4 b) {
    if (b.x == 0.0f) return a;
    if (a.x == 0.0f) return b;
    float n = a.x + b.x;
    float d = b.y - a.y;
    float f = b.x / n;
    return float4(n, a.y + d * f, a.z + b.z + d * d * a.x * f, 0.0f);
}

// The partial of every column of one conv tile, from its stored outputs staged row-major in `tile` [ROWS, COLS]:
// THREADS / COLS threads per column each take a contiguous run of rows (two passes: mean, then squared deviations),
// then the first merges the runs in order and writes dst[col]. valid(j) says whether tile row j is a kept output.
// Every thread of the threadgroup must call it
template <int ROWS, int COLS, int THREADS, typename Valid>
inline void tile_stats(threadgroup const real* tile, threadgroup float4* part, uint tid, Valid valid,
                       device float4* dst) {
    static_assert(THREADS % COLS == 0 && ROWS % (THREADS / COLS) == 0, "tile_stats layout");
    constexpr int TPC = THREADS / COLS, SPAN = ROWS / TPC;
    int col = tid % COLS, k = tid / COLS;
    float n = 0.0f, s = 0.0f;
    for (int j = k * SPAN; j < (k + 1) * SPAN; j++) {
        if (valid(j)) {
            s += float(tile[j * COLS + col]);
            n += 1.0f;
        }
    }
    float mean = n > 0.0f ? s / n : 0.0f;
    float m2 = 0.0f;
    for (int j = k * SPAN; j < (k + 1) * SPAN; j++) {
        if (valid(j)) {
            float d = float(tile[j * COLS + col]) - mean;
            m2 += d * d;
        }
    }
    part[tid] = float4(n, mean, m2, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0) {
        float4 a = part[col];
        for (int q = 1; q < TPC; q++) a = chan_merge(a, part[q * COLS + col]);
        dst[col] = a;
    }
}
