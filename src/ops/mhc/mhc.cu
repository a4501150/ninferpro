// infernix::ops — mhc: manifold-constrained hyper-connection residual streams.
//
// Wrapper (validation, workspace scope, dispatch on the stream count), launchers and kernels in one
// translation unit, like hyper_connection. See include/infernix/ops/mhc.h for the contract.

#include "infernix/ops/mhc.h"

#include "core/pdl.cuh"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
// The mixer's split-K shape, fixed for every T so a column's bits never depend on the call's width:
// each chunk covers kChunkD features across every stream (warp 0 / lower 32 and warp 1 / upper 32
// reduced in identical butterfly order, then added); the epilogue sums the chunks in ascending
// order.
constexpr int kChunkD = 64;
constexpr int kCols   = 64;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("mhc: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("mhc ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 &&
           t.ne[3] == 1;
}

bool contiguous_1d(const Tensor& t, DType dtype, int length) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[0] == length &&
           t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

bool supported_streams(int streams) { return streams == 1 || streams == 2 || streams == 4; }

int gate_rows(int streams) { return (2 + streams) * streams; }

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a.data);
    const auto pb = reinterpret_cast<std::uintptr_t>(b.data);
    return pa < pb + b.bytes() && pb < pa + a.bytes();
}

__device__ __forceinline__ float bf16_value(bf16 v) { return __bfloat162float(v); }

__device__ __forceinline__ bf16 rounded_bf16(float v) { return __float2bfloat16_rn(v); }

bool aligned_bytes(const void* p, std::size_t align) {
    return (reinterpret_cast<std::uintptr_t>(p) & (align - 1)) == 0;
}

// Bit-exact pair butterfly reduction: computes warp_reduce_sum(a0) + warp_reduce_sum(a1) on every
// lane with 7 shuffles instead of 10, using FP32 addition commutativity after the 16-lane fold.
__device__ __forceinline__ float warp_reduce_pair_sum(float a0, float a1, int lane) {
    a0 += __shfl_xor_sync(0xFFFFFFFFU, a0, 16);
    a1 += __shfl_xor_sync(0xFFFFFFFFU, a1, 16);
    float z = lane < 16 ? a0 : a1;
    z += __shfl_xor_sync(0xFFFFFFFFU, z, 8);
    z += __shfl_xor_sync(0xFFFFFFFFU, z, 4);
    z += __shfl_xor_sync(0xFFFFFFFFU, z, 2);
    z += __shfl_xor_sync(0xFFFFFFFFU, z, 1);
    return z + __shfl_xor_sync(0xFFFFFFFFU, z, 16);
}

// Bit-exact dual-pair butterfly reduction: reduces two (a0, a1) pairs simultaneously in 10 shuffles
// instead of 20. Returns pair 0's sum on lanes 0..7 and pair 1's sum on lanes 8..15.
__device__ __forceinline__ float warp_reduce_two_pairs(float a0, float a1, float b0, float b1,
                                                       int lane) {
    a0 += __shfl_xor_sync(0xFFFFFFFFU, a0, 16);
    a1 += __shfl_xor_sync(0xFFFFFFFFU, a1, 16);
    b0 += __shfl_xor_sync(0xFFFFFFFFU, b0, 16);
    b1 += __shfl_xor_sync(0xFFFFFFFFU, b1, 16);
    float p = lane < 16 ? a0 : a1;
    float q = lane < 16 ? b0 : b1;
    p += __shfl_xor_sync(0xFFFFFFFFU, p, 8);
    q += __shfl_xor_sync(0xFFFFFFFFU, q, 8);
    float r = (lane & 8) == 0 ? p : q;
    r += __shfl_xor_sync(0xFFFFFFFFU, r, 4);
    r += __shfl_xor_sync(0xFFFFFFFFU, r, 2);
    r += __shfl_xor_sync(0xFFFFFFFFU, r, 1);
    return r + __shfl_xor_sync(0xFFFFFFFFU, r, 16);
}

// One output stream's update contribution: post_s * y + sum_j comb[j][s] * R[j], in ascending
// stream order. Shared verbatim by mhc_update's kernel and the fused mixer's first stage, so both
// routes round the same FP32 arithmetic to the same bits; the explicit fmaf pins the contraction
// (fma(x, y, 0) is x * y's correctly rounded product) against per-kernel scheduling choices.
template <int kStreams>
__device__ __forceinline__ float update_stream(const float* post_column, const float* comb_column,
                                               float y_value, const float (&residual_old)[kStreams],
                                               int s) {
    float acc = __fmaf_rn(post_column[s], y_value, 0.0F);
#pragma unroll
    for (int j = 0; j < kStreams; ++j) {
        acc = __fmaf_rn(comb_column[s * kStreams + j], residual_old[j], acc);
    }
    return acc;
}

// The mixer's first stage: per column and chunk (kChunkD = 64 features across every stream), the
// fixed-order partials of f = fn . R and of the squared RMS, from the rounded BF16 values of the
// (updated) residual. With kUpdate the stage first applies the residual update elementwise (the
// same shared arithmetic as mhc_update's kernel, so the fused call keeps the separate calls' bits).
//
// Each CTA (8 warps = 256 threads) owns one feature chunk across up to kCols columns:
// 1. Each warp loads its gate rows' fn weights for this chunk once into registers (lane l holds
//    features l and l + 32, preserving the exact 32+32 butterfly reduction order).
// 2. Columns advance in batches of kTileT = 8: warp ii cooperatively loads (and, with kUpdate,
//    updates and stores once) column ii's 64 features using coalesced bfloat162 accesses into
//    shared memory sh_r[ii][s][64], and writes column ii's squared-sum partial.
// 3. Every warp accumulates its gate rows across the batch's live columns from sh_r and writes
//    partials[(t * (kRows + 1) + row) * chunks + chunk], contiguous per column for the epilogue.
template <int kStreams, bool kUpdate>
__global__ __launch_bounds__(kThreads, kStreams == 4 ? 3 : 4) void mhc_partial_kernel(
    const bf16* __restrict__ fn, const bf16* __restrict__ residual_in,
    bf16* __restrict__ residual_out, const bf16* __restrict__ y, const float* __restrict__ post,
    const float* __restrict__ comb, int hidden, int chunks, int columns, bool vec2_ok,
    float* __restrict__ partials) {
    pdl::enter();
    constexpr int kRows        = (2 + kStreams) * kStreams;
    constexpr int kWarps       = kThreads / 32; // 8
    constexpr int kRowsPerWarp = (kRows + kWarps - 1) / kWarps;
    constexpr int kTileT       = kWarps; // 8 columns per shared-memory batch

    // Double-buffered BF16 residual staging (8 KiB total, 0 store bank conflicts on bfloat162
    // writes) so warps finishing Step 2 can immediately issue the next batch's global loads.
    __shared__ bf16 sh_r[2][kTileT][kStreams][kChunkD];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int chunk  = static_cast<int>(blockIdx.x);
    const int d_base = chunk * kChunkD;
    const int d0     = d_base + lane;
    const int d1     = d_base + 32 + lane;
    const bool act0  = d0 < hidden;
    const bool act1  = d1 < hidden;
    const bool full  = (d_base + kChunkD <= hidden);

    // Load this warp's gate rows of fn once for all kCols columns, packed as (d0, d1) BF16 pairs
    // to cut register pressure in half while preserving exact BF16 values.
    __nv_bfloat162 fn_p[kRowsPerWarp][kStreams];
    const int row_lo = warp * kRowsPerWarp;
#pragma unroll
    for (int v = 0; v < kRowsPerWarp; ++v) {
        const int i = row_lo + v;
        if (i < kRows) {
            const bf16* row = fn + static_cast<std::size_t>(i) * (kStreams * hidden);
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const bf16* sp = row + static_cast<std::size_t>(s) * hidden;
                const bf16 a   = act0 ? sp[d0] : rounded_bf16(0.0F);
                const bf16 b   = act1 ? sp[d1] : rounded_bf16(0.0F);
                fn_p[v][s]     = __halves2bfloat162(a, b);
            }
        } else {
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                fn_p[v][s] = __halves2bfloat162(rounded_bf16(0.0F), rounded_bf16(0.0F));
            }
        }
    }

    const int col_start = static_cast<int>(blockIdx.y) * kCols;
    const int col_end   = min(columns, col_start + kCols);

    int stage = 0;
    for (int col_base = col_start; col_base < col_end; col_base += kTileT, stage ^= 1) {
        const int live_cols = min(kTileT, col_end - col_base);
        // Step 1: warp `warp` loads (and optionally updates + stores once) column `col_base + warp`.
        if (warp < live_cols) {
            const int t              = col_base + warp;
            const std::size_t column = static_cast<std::size_t>(t) * (kStreams * hidden);
            if (full && vec2_ok) {
                const int d2 = d_base + 2 * lane;
                if constexpr (kUpdate) {
                    float old0[kStreams], old1[kStreams];
#pragma unroll
                    for (int j = 0; j < kStreams; ++j) {
                        const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(
                            residual_in + column + static_cast<std::size_t>(j) * hidden + d2);
                        const float2 f = __bfloat1622float2(pair);
                        old0[j]        = f.x;
                        old1[j]        = f.y;
                    }
                    const __nv_bfloat162 ypair = *reinterpret_cast<const __nv_bfloat162*>(
                        y + static_cast<std::size_t>(t) * hidden + d2);
                    const float2 yf          = __bfloat1622float2(ypair);
                    const float* post_column = post + static_cast<std::size_t>(t) * kStreams;
                    const float* comb_column =
                        comb + static_cast<std::size_t>(t) * kStreams * kStreams;
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) {
                        const bf16 u0 = rounded_bf16(
                            update_stream<kStreams>(post_column, comb_column, yf.x, old0, s));
                        const bf16 u1 = rounded_bf16(
                            update_stream<kStreams>(post_column, comb_column, yf.y, old1, s));
                        const __nv_bfloat162 up = __halves2bfloat162(u0, u1);
                        *reinterpret_cast<__nv_bfloat162*>(
                            residual_out + column + static_cast<std::size_t>(s) * hidden + d2) = up;
                        *reinterpret_cast<__nv_bfloat162*>(&sh_r[stage][warp][s][2 * lane]) = up;
                    }
                } else {
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) {
                        const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(
                            residual_in + column + static_cast<std::size_t>(s) * hidden + d2);
                        *reinterpret_cast<__nv_bfloat162*>(&sh_r[stage][warp][s][2 * lane]) = pair;
                    }
                }
            } else {
                const float* post_column =
                    kUpdate ? post + static_cast<std::size_t>(t) * kStreams : nullptr;
                const float* comb_column =
                    kUpdate ? comb + static_cast<std::size_t>(t) * kStreams * kStreams : nullptr;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int idx  = 2 * lane + k;
                    const int d    = d_base + idx;
                    const bool act = d < hidden;
                    if constexpr (kUpdate) {
                        float old_r[kStreams];
#pragma unroll
                        for (int j = 0; j < kStreams; ++j) {
                            old_r[j] = act ? bf16_value(residual_in[column +
                                                                    static_cast<std::size_t>(j) *
                                                                        hidden +
                                                                    d])
                                           : 0.0F;
                        }
                        const float yv =
                            act ? bf16_value(y[static_cast<std::size_t>(t) * hidden + d]) : 0.0F;
#pragma unroll
                        for (int s = 0; s < kStreams; ++s) {
                            const bf16 u = rounded_bf16(
                                update_stream<kStreams>(post_column, comb_column, yv, old_r, s));
                            if (act) {
                                residual_out[column + static_cast<std::size_t>(s) * hidden + d] = u;
                            }
                            sh_r[stage][warp][s][idx] = act ? u : rounded_bf16(0.0F);
                        }
                    } else {
#pragma unroll
                        for (int s = 0; s < kStreams; ++s) {
                            sh_r[stage][warp][s][idx] =
                                act ? residual_in[column + static_cast<std::size_t>(s) * hidden + d]
                                    : rounded_bf16(0.0F);
                        }
                    }
                }
            }
            __syncwarp();
            float sq0 = 0.0F, sq1 = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const float v0 = bf16_value(sh_r[stage][warp][s][lane]);
                const float v1 = bf16_value(sh_r[stage][warp][s][32 + lane]);
                sq0 += v0 * v0;
                sq1 += v1 * v1;
            }
            const float sq = warp_reduce_pair_sum(sq0, sq1, lane);
            if (lane == 0) {
                partials[(static_cast<std::size_t>(col_base) * (kRows + 1) +
                          static_cast<std::size_t>(kRows) * live_cols) *
                             chunks +
                         static_cast<std::size_t>(chunk) * live_cols + warp] = sq;
            }
        }
        __syncthreads();

        // Step 2: every warp reduces its gate rows across the live columns of this batch.
        // Reading sh_r once per column and sharing across all kRowsPerWarp rows cuts LDS traffic
        // 3x; pairing two columns for kRowsPerWarp == 3 halves fn unpacks and folds 6 sums into 3
        // dual-pair reductions (15 shuffles/col, bit-exact 32+32 order).
        const std::size_t group_base =
            static_cast<std::size_t>(col_base) * (kRows + 1) * chunks;
        float row_out[kRowsPerWarp];
#pragma unroll
        for (int v = 0; v < kRowsPerWarp; ++v) { row_out[v] = 0.0F; }
        if constexpr (kRowsPerWarp == 3) {
            int ii = 0;
            for (; ii + 1 < live_cols; ii += 2) {
                float a0[3] = {0.0F, 0.0F, 0.0F}, a1[3] = {0.0F, 0.0F, 0.0F};
                float b0[3] = {0.0F, 0.0F, 0.0F}, b1[3] = {0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const float ra0 = bf16_value(sh_r[stage][ii][s][lane]);
                    const float ra1 = bf16_value(sh_r[stage][ii][s][32 + lane]);
                    const float rb0 = bf16_value(sh_r[stage][ii + 1][s][lane]);
                    const float rb1 = bf16_value(sh_r[stage][ii + 1][s][32 + lane]);
#pragma unroll
                    for (int v = 0; v < 3; ++v) {
                        const float2 wf = __bfloat1622float2(fn_p[v][s]);
                        a0[v] += wf.x * ra0;
                        a1[v] += wf.y * ra1;
                        b0[v] += wf.x * rb0;
                        b1[v] += wf.y * rb1;
                    }
                }
                const float p0 = warp_reduce_two_pairs(a0[0], a1[0], a0[1], a1[1], lane);
                const float p1 = warp_reduce_two_pairs(a0[2], a1[2], b0[0], b1[0], lane);
                const float p2 = warp_reduce_two_pairs(b0[1], b1[1], b0[2], b1[2], lane);
                const float p0_hi = __shfl_xor_sync(0xFFFFFFFFU, p0, 8);
                const float p1_hi = __shfl_xor_sync(0xFFFFFFFFU, p1, 8);
                const float p2_hi = __shfl_xor_sync(0xFFFFFFFFU, p2, 8);
                if (lane == ii) {
                    row_out[0] = p0;
                    row_out[1] = p0_hi;
                    row_out[2] = p1;
                } else if (lane == ii + 1) {
                    row_out[0] = p1_hi;
                    row_out[1] = p2;
                    row_out[2] = p2_hi;
                }
            }
            if (ii < live_cols) {
                float a0[3] = {0.0F, 0.0F, 0.0F}, a1[3] = {0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const float ra0 = bf16_value(sh_r[stage][ii][s][lane]);
                    const float ra1 = bf16_value(sh_r[stage][ii][s][32 + lane]);
#pragma unroll
                    for (int v = 0; v < 3; ++v) {
                        const float2 wf = __bfloat1622float2(fn_p[v][s]);
                        a0[v] += wf.x * ra0;
                        a1[v] += wf.y * ra1;
                    }
                }
                const float p0    = warp_reduce_two_pairs(a0[0], a1[0], a0[1], a1[1], lane);
                const float p0_hi = __shfl_xor_sync(0xFFFFFFFFU, p0, 8);
                const float s2    = warp_reduce_pair_sum(a0[2], a1[2], lane);
                if (lane == ii) {
                    row_out[0] = p0;
                    row_out[1] = p0_hi;
                    row_out[2] = s2;
                }
            }
        } else {
            for (int ii = 0; ii < live_cols; ++ii) {
                float acc0[kRowsPerWarp], acc1[kRowsPerWarp];
#pragma unroll
                for (int v = 0; v < kRowsPerWarp; ++v) {
                    acc0[v] = 0.0F;
                    acc1[v] = 0.0F;
                }
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const float r0 = bf16_value(sh_r[stage][ii][s][lane]);
                    const float r1 = bf16_value(sh_r[stage][ii][s][32 + lane]);
#pragma unroll
                    for (int v = 0; v < kRowsPerWarp; ++v) {
                        const float2 wf = __bfloat1622float2(fn_p[v][s]);
                        acc0[v] += wf.x * r0;
                        acc1[v] += wf.y * r1;
                    }
                }
#pragma unroll
                for (int v = 0; v < kRowsPerWarp; ++v) {
                    const float sv = warp_reduce_pair_sum(acc0[v], acc1[v], lane);
                    if (lane == ii) { row_out[v] = sv; }
                }
            }
        }
        if (lane < live_cols) {
#pragma unroll
            for (int v = 0; v < kRowsPerWarp; ++v) {
                const int i = row_lo + v;
                if (i < kRows) {
                    partials[group_base + (static_cast<std::size_t>(i) * chunks + chunk) * live_cols +
                             lane] = row_out[v];
                }
            }
        }
    }
}

// The mixer's second stage: one CTA per column sums the column's partials in ascending chunk order
// once, computes pre/post/Sinkhorn comb on warp 0, and streams vectorized uint4 residual loads to
// write x = bf16(sum_s pre_s * R[s*H + d]) across all 256 threads.
template <int kStreams>
__global__ __launch_bounds__(kThreads, 4) void mhc_epilogue_kernel(
    const float* __restrict__ partials, const bf16* __restrict__ residual,
    const float* __restrict__ base, const float* __restrict__ scale, int hidden, int chunks,
    int columns, int sinkhorn_iters, float eps_norm, float eps, bool vec8_ok, bf16* __restrict__ x,
    float* __restrict__ post, float* __restrict__ comb) {
    pdl::enter();
    constexpr int kRows  = (2 + kStreams) * kStreams;
    constexpr int kS2    = kStreams * kStreams;
    constexpr int kTileT = kThreads / 32;
    const int kWidth     = kStreams * hidden;
    __shared__ float sh_f[kRows + 1];
    __shared__ float sh_pre[kStreams];
    const int t    = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    if (tid < 32) {
        const int col_base = (t / kTileT) * kTileT;
        const int w        = min(kTileT, columns - col_base);
        const int off      = t - col_base;
        const std::size_t group_base =
            static_cast<std::size_t>(col_base) * (kRows + 1) * chunks;
        if (lane <= kRows) {
            const float* row_partial =
                partials + group_base + static_cast<std::size_t>(lane) * chunks * w + off;
            float sum = 0.0F;
            if (w == 1 && (chunks & 3) == 0) {
                const auto* p4 = reinterpret_cast<const float4*>(row_partial);
                const int n4   = chunks >> 2;
                for (int c = 0; c < n4; ++c) {
                    const float4 v = p4[c];
                    sum += v.x;
                    sum += v.y;
                    sum += v.z;
                    sum += v.w;
                }
            } else {
                // Issue 4 independent loads per iteration while summing in strict ascending order.
                int c = 0;
                for (; c + 3 < chunks; c += 4) {
                    const float v0 = row_partial[static_cast<std::size_t>(c) * w];
                    const float v1 = row_partial[static_cast<std::size_t>(c + 1) * w];
                    const float v2 = row_partial[static_cast<std::size_t>(c + 2) * w];
                    const float v3 = row_partial[static_cast<std::size_t>(c + 3) * w];
                    sum            = (((sum + v0) + v1) + v2) + v3;
                }
                for (; c < chunks; ++c) {
                    sum += row_partial[static_cast<std::size_t>(c) * w];
                }
            }
            sh_f[lane] = sum;
        }
        __syncwarp();
        const float inverse_rms = rsqrtf(sh_f[kRows] / static_cast<float>(kWidth) + eps_norm);
        if (lane < kRows) { sh_f[lane] *= inverse_rms; }
        __syncwarp();
        if (lane < kStreams) {
            sh_pre[lane] = sigmoid(sh_f[lane] * scale[0] + base[lane]) + eps;
            if (post != nullptr) {
                post[static_cast<std::size_t>(t) * kStreams + lane] =
                    2.0F * sigmoid(sh_f[kStreams + lane] * scale[1] + base[kStreams + lane]);
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        float logit = 0.0F;
        if (lane < kS2) {
            const int f_at = 2 * kStreams + lane;
            logit          = sh_f[f_at] * scale[2] + base[f_at];
        }
        float m = logit;
#pragma unroll
        for (int delta = 1; delta < kStreams; delta <<= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, delta, kStreams));
        }
        float e   = lane < kS2 ? expf(logit - m) : 0.0F;
        float den = e;
#pragma unroll
        for (int delta = 1; delta < kStreams; delta <<= 1) {
            den += __shfl_xor_sync(0xFFFFFFFFU, den, delta, kStreams);
        }
        float v = lane < kS2 ? e / den + eps : 0.0F;
        v       = __shfl_sync(0xFFFFFFFFU, v, lane & (kS2 - 1));
        for (int it = 0; it < sinkhorn_iters; ++it) {
            if (it > 0) {
                float row_sum = v;
#pragma unroll
                for (int delta = 1; delta < kStreams; delta <<= 1) {
                    row_sum += __shfl_xor_sync(0xFFFFFFFFU, row_sum, delta, kStreams);
                }
                v /= (row_sum + eps);
            }
            float column_sum = v;
#pragma unroll
            for (int delta = kStreams; delta < kS2; delta <<= 1) {
                column_sum += __shfl_xor_sync(0xFFFFFFFFU, column_sum, delta, kS2);
            }
            v /= (column_sum + eps);
        }
        if (comb != nullptr && lane < kS2) {
            comb[(static_cast<std::size_t>(t) * kStreams + (lane % kStreams)) * kStreams +
                 (lane / kStreams)] = v;
        }
    }
    // All warps stream x from the (updated) residual; warps 1..7 overlap warp 0's Sinkhorn.
    float pre_r[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) { pre_r[s] = sh_pre[s]; }
    const std::size_t column = static_cast<std::size_t>(t) * kWidth;
    if (vec8_ok) {
        const int hidden_vec = hidden >> 3;
        auto* x_vec          = reinterpret_cast<uint4*>(x + static_cast<std::size_t>(t) * hidden);
        for (int v = tid; v < hidden_vec; v += kThreads) {
            float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    residual + column + static_cast<std::size_t>(s) * hidden +
                    static_cast<std::size_t>(v) * 8);
                const auto* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                const float ps = pre_r[s];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 f = __bfloat1622float2(p2[k]);
                    acc[2 * k] += ps * f.x;
                    acc[2 * k + 1] += ps * f.y;
                }
            }
            uint4 out_v;
            auto* out_p2 = reinterpret_cast<__nv_bfloat162*>(&out_v);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                out_p2[k] = __halves2bfloat162(rounded_bf16(acc[2 * k]),
                                               rounded_bf16(acc[2 * k + 1]));
            }
            x_vec[v] = out_v;
        }
    } else {
        for (int d = tid; d < hidden; d += kThreads) {
            float acc = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                acc += pre_r[s] *
                       bf16_value(residual[column + static_cast<std::size_t>(s) * hidden + d]);
            }
            x[static_cast<std::size_t>(t) * hidden + d] = rounded_bf16(acc);
        }
    }
}

// mhc_update's kernel: vectorized uint4 over hidden when 8-element aligned, with an elementwise
// fallback for arbitrary hidden sizes. Old streams are read before any write so residual_out may
// alias residual_in.
template <int kStreams>
__global__ __launch_bounds__(kThreads) void mhc_update_kernel(const bf16* __restrict__ y,
                                                              const float* __restrict__ post,
                                                              const float* __restrict__ comb,
                                                              const bf16* __restrict__ residual_in,
                                                              bf16* __restrict__ residual_out,
                                                              int hidden, bool vec8_ok) {
    pdl::enter();
    const int t              = static_cast<int>(blockIdx.y);
    const std::size_t column = static_cast<std::size_t>(t) * (kStreams * hidden);
    const float* post_column = post + static_cast<std::size_t>(t) * kStreams;
    const float* comb_column = comb + static_cast<std::size_t>(t) * kStreams * kStreams;
    float post_r[kStreams];
    float comb_r[kStreams * kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) { post_r[s] = post_column[s]; }
#pragma unroll
    for (int k = 0; k < kStreams * kStreams; ++k) { comb_r[k] = comb_column[k]; }

    if (vec8_ok) {
        const int v = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
        if (v >= (hidden >> 3)) { return; }
        const std::size_t off = static_cast<std::size_t>(v) * 8;
        float old_r[8][kStreams];
#pragma unroll
        for (int j = 0; j < kStreams; ++j) {
            const uint4 raw = *reinterpret_cast<const uint4*>(
                residual_in + column + static_cast<std::size_t>(j) * hidden + off);
            const auto* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f        = __bfloat1622float2(p2[k]);
                old_r[2 * k][j]       = f.x;
                old_r[2 * k + 1][j]   = f.y;
            }
        }
        const uint4 y_raw = *reinterpret_cast<const uint4*>(
            y + static_cast<std::size_t>(t) * hidden + off);
        const auto* yp2 = reinterpret_cast<const __nv_bfloat162*>(&y_raw);
        float yv[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(yp2[k]);
            yv[2 * k]      = f.x;
            yv[2 * k + 1]  = f.y;
        }
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            uint4 out_v;
            auto* out_p2 = reinterpret_cast<__nv_bfloat162*>(&out_v);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const bf16 u0 =
                    rounded_bf16(update_stream<kStreams>(post_r, comb_r, yv[2 * k], old_r[2 * k], s));
                const bf16 u1 = rounded_bf16(
                    update_stream<kStreams>(post_r, comb_r, yv[2 * k + 1], old_r[2 * k + 1], s));
                out_p2[k] = __halves2bfloat162(u0, u1);
            }
            *reinterpret_cast<uint4*>(residual_out + column +
                                      static_cast<std::size_t>(s) * hidden + off) = out_v;
        }
        return;
    }

    const int d = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (d >= hidden) { return; }
    float residual_old[kStreams];
#pragma unroll
    for (int j = 0; j < kStreams; ++j) {
        residual_old[j] =
            bf16_value(residual_in[column + static_cast<std::size_t>(j) * hidden + d]);
    }
    const float y_value = bf16_value(y[static_cast<std::size_t>(t) * hidden + d]);
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        residual_out[column + static_cast<std::size_t>(s) * hidden + d] = rounded_bf16(
            update_stream<kStreams>(post_r, comb_r, y_value, residual_old, s));
    }
}

// The two mixer stages, shared by mhc_mix (no pending update) and mhc_update_mix (the first stage
// applies y's update exactly as mhc_update's kernel does). residual_out is null for mhc_mix.
template <int kStreams>
void run_mix(const Tensor& residual_in, const Tensor* residual_out, const Tensor* y,
             const Tensor* post_in, const Tensor* comb_in, const Tensor& fn, const Tensor& base,
             const Tensor& scale, std::int32_t sinkhorn_iters, float eps_norm, float eps, Tensor& x,
             Tensor* post_out, Tensor* comb_out, WorkspaceArena& workspace, cudaStream_t stream) {
    const int hidden  = x.ne[0];
    const int columns = x.ne[1];
    const int rows    = (2 + kStreams) * kStreams;
    const int chunks  = (hidden + kChunkD - 1) / kChunkD;
    auto scope        = workspace.scope();
    const DeviceSpan partials =
        workspace.alloc_bytes(static_cast<std::size_t>(rows + 1) *
                              static_cast<std::size_t>(columns) * chunks * sizeof(float));
    const auto* fn_p       = static_cast<const bf16*>(fn.data);
    const auto* residual_p = static_cast<const bf16*>(residual_in.data);
    auto* out_p      = residual_out != nullptr ? static_cast<bf16*>(residual_out->data) : nullptr;
    const int groups = (columns + kCols - 1) / kCols;
    const bool vec2_ok =
        (hidden & 1) == 0 && aligned_bytes(residual_p, 4) &&
        (out_p == nullptr || aligned_bytes(out_p, 4)) &&
        (y == nullptr || aligned_bytes(y->data, 4));
    if (residual_out != nullptr) {
        if (pdl::launch_consumer({dim3(chunks, groups), dim3(kThreads), 0, stream},
                                 mhc_partial_kernel<kStreams, true>, fn_p, residual_p, out_p,
                                 static_cast<const bf16*>(y->data),
                                 static_cast<const float*>(post_in->data),
                                 static_cast<const float*>(comb_in->data), hidden, chunks, columns,
                                 vec2_ok, static_cast<float*>(partials.data)) != cudaSuccess) {
            check_launch("update mix partial");
        }
    } else {
        if (pdl::launch_consumer({dim3(chunks, groups), dim3(kThreads), 0, stream},
                                 mhc_partial_kernel<kStreams, false>, fn_p, residual_p,
                                 static_cast<bf16*>(nullptr), static_cast<const bf16*>(nullptr),
                                 static_cast<const float*>(nullptr),
                                 static_cast<const float*>(nullptr), hidden, chunks, columns,
                                 vec2_ok, static_cast<float*>(partials.data)) != cudaSuccess) {
            check_launch("mix partial");
        }
    }
    const bf16* mix_residual = residual_out != nullptr ? out_p : residual_p;
    const bool vec8_ok =
        (hidden & 7) == 0 && aligned_bytes(mix_residual, 16) && aligned_bytes(x.data, 16);
    if (pdl::launch_consumer(
            {dim3(columns), dim3(kThreads), 0, stream}, mhc_epilogue_kernel<kStreams>,
            static_cast<const float*>(partials.data), mix_residual,
            static_cast<const float*>(base.data), static_cast<const float*>(scale.data), hidden,
            chunks, columns, sinkhorn_iters, eps_norm, eps, vec8_ok, static_cast<bf16*>(x.data),
            post_out != nullptr ? static_cast<float*>(post_out->data) : nullptr,
            comb_out != nullptr ? static_cast<float*>(comb_out->data) : nullptr) != cudaSuccess) {
        check_launch("mix epilogue");
    }
}

void validate_mix(const Tensor& residual, const Tensor& fn, const Tensor& base, const Tensor& scale,
                  std::int32_t streams, std::int32_t sinkhorn_iters, float eps_norm, float eps,
                  const Tensor& x, const Tensor* post, const Tensor* comb) {
    require(supported_streams(streams), "streams must be 1, 2 or 4");
    require(sinkhorn_iters >= 1, "sinkhorn_iters must be at least 1");
    require(eps_norm > 0.0F && eps > 0.0F, "eps_norm and eps must be positive");
    const int columns = residual.ne[1];
    const int hidden  = x.ne[0];
    const int rows    = gate_rows(streams);
    require(columns > 0 && hidden > 0, "extents must be positive");
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(x, DType::BF16),
            "residual and x must be contiguous BF16 tensors");
    require(residual.ne[0] == streams * hidden && x.ne[1] == columns,
            "residual and x shapes disagree");
    require(contiguous_2d(fn, DType::BF16) && fn.ne[0] == residual.ne[0] && fn.ne[1] == rows,
            "fn must be BF16 [rows, S*H] with rows = (2+S)*S");
    require(contiguous_1d(base, DType::FP32, rows), "base must be FP32 [rows]");
    require(contiguous_1d(scale, DType::FP32, 3), "scale must be FP32 [3]");
    if (post != nullptr) {
        require(contiguous_2d(*post, DType::FP32) && post->ne[0] == streams &&
                    post->ne[1] == columns,
                "post must be FP32 [S, T]");
    }
    if (comb != nullptr) {
        require(comb->data != nullptr && comb->dtype == DType::FP32 && comb->is_contiguous() &&
                    comb->ne[0] == streams && comb->ne[1] == streams && comb->ne[2] == columns &&
                    comb->ne[3] == 1,
                "comb must be FP32 [S, S, T]");
    }
    require(!overlaps(x, residual) && !overlaps(x, fn), "x must not alias the residual or fn");
}

void validate_update(const Tensor& y, const Tensor& post, const Tensor& comb,
                     const Tensor& residual_in, const Tensor& residual_out) {
    const int streams = post.ne[0];
    const int columns = y.ne[1];
    const int hidden  = y.ne[0];
    require(supported_streams(streams), "post's stream count must be 1, 2 or 4");
    require(contiguous_2d(y, DType::BF16) && columns > 0 && hidden > 0,
            "y must be a contiguous BF16 [H, T] tensor");
    require(contiguous_2d(post, DType::FP32) && post.ne[1] == columns, "post must be FP32 [S, T]");
    require(comb.data != nullptr && comb.dtype == DType::FP32 && comb.is_contiguous() &&
                comb.ne[0] == streams && comb.ne[1] == streams && comb.ne[2] == columns &&
                comb.ne[3] == 1,
            "comb must be FP32 [S, S, T]");
    require(contiguous_2d(residual_in, DType::BF16) && contiguous_2d(residual_out, DType::BF16) &&
                residual_in.ne[0] == streams * hidden && residual_in.ne[1] == columns &&
                residual_out.ne[0] == residual_in.ne[0] && residual_out.ne[1] == columns,
            "the residuals must be contiguous BF16 [S*H, T] tensors");
    if (residual_in.data != residual_out.data) {
        require(!overlaps(residual_in, residual_out), "the residuals must not partially overlap");
    }
    require(!overlaps(y, residual_out), "y must not alias the updated residual");
}

} // namespace

void mhc_mix(const Tensor& residual, const Tensor& fn, const Tensor& base, const Tensor& scale,
             std::int32_t streams, std::int32_t sinkhorn_iters, float eps_norm, float eps,
             Tensor& x, Tensor* post, Tensor* comb, WorkspaceArena& workspace,
             cudaStream_t stream) {
    validate_mix(residual, fn, base, scale, streams, sinkhorn_iters, eps_norm, eps, x, post, comb);
    switch (streams) {
    case 1:
        run_mix<1>(residual, nullptr, nullptr, nullptr, nullptr, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post, comb, workspace, stream);
        break;
    case 2:
        run_mix<2>(residual, nullptr, nullptr, nullptr, nullptr, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post, comb, workspace, stream);
        break;
    default:
        run_mix<4>(residual, nullptr, nullptr, nullptr, nullptr, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post, comb, workspace, stream);
        break;
    }
}

std::size_t mhc_mix_workspace_capacity_bytes(std::int32_t streams, std::int32_t hidden,
                                             std::int32_t min_tokens, std::int32_t max_tokens) {
    require(
        supported_streams(streams) && hidden > 0 && min_tokens > 0 && min_tokens <= max_tokens,
        "mhc workspace needs streams in {1, 2, 4}, a positive hidden width and a valid interval");
    const auto rows         = static_cast<std::size_t>(gate_rows(streams)) + 1;
    const auto chunks       = static_cast<std::size_t>((hidden + kChunkD - 1) / kChunkD);
    const std::size_t bytes = rows * static_cast<std::size_t>(max_tokens) * chunks * sizeof(float);
    return (bytes + 255) / 256 * 256;
}

void mhc_update(const Tensor& y, const Tensor& post, const Tensor& comb, const Tensor& residual_in,
                Tensor& residual_out, cudaStream_t stream) {
    validate_update(y, post, comb, residual_in, residual_out);
    const int streams  = post.ne[0];
    const int hidden   = y.ne[0];
    const int columns  = y.ne[1];
    const bool vec8_ok = (hidden & 7) == 0 && aligned_bytes(y.data, 16) &&
                         aligned_bytes(residual_in.data, 16) &&
                         aligned_bytes(residual_out.data, 16);
    const int work_x = vec8_ok ? (hidden >> 3) : hidden;
    const dim3 grid((work_x + kThreads - 1) / kThreads, columns);
    switch (streams) {
    case 1:
        if (pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, mhc_update_kernel<1>,
                                 static_cast<const bf16*>(y.data),
                                 static_cast<const float*>(post.data),
                                 static_cast<const float*>(comb.data),
                                 static_cast<const bf16*>(residual_in.data),
                                 static_cast<bf16*>(residual_out.data), hidden,
                                 vec8_ok) != cudaSuccess) {
            check_launch("update");
        }
        break;
    case 2:
        if (pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, mhc_update_kernel<2>,
                                 static_cast<const bf16*>(y.data),
                                 static_cast<const float*>(post.data),
                                 static_cast<const float*>(comb.data),
                                 static_cast<const bf16*>(residual_in.data),
                                 static_cast<bf16*>(residual_out.data), hidden,
                                 vec8_ok) != cudaSuccess) {
            check_launch("update");
        }
        break;
    default:
        if (pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, mhc_update_kernel<4>,
                                 static_cast<const bf16*>(y.data),
                                 static_cast<const float*>(post.data),
                                 static_cast<const float*>(comb.data),
                                 static_cast<const bf16*>(residual_in.data),
                                 static_cast<bf16*>(residual_out.data), hidden,
                                 vec8_ok) != cudaSuccess) {
            check_launch("update");
        }
        break;
    }
}

void mhc_update_mix(const Tensor& y, const Tensor& post, const Tensor& comb,
                    const Tensor& residual_in, Tensor& residual_out, const Tensor& fn,
                    const Tensor& base, const Tensor& scale, std::int32_t streams,
                    std::int32_t sinkhorn_iters, float eps_norm, float eps, Tensor& x,
                    Tensor* post_out, Tensor* comb_out, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    validate_mix(residual_out, fn, base, scale, streams, sinkhorn_iters, eps_norm, eps, x, post_out,
                 comb_out);
    validate_update(y, post, comb, residual_in, residual_out);
    switch (streams) {
    case 1:
        run_mix<1>(residual_in, &residual_out, &y, &post, &comb, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post_out, comb_out, workspace, stream);
        break;
    case 2:
        run_mix<2>(residual_in, &residual_out, &y, &post, &comb, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post_out, comb_out, workspace, stream);
        break;
    default:
        run_mix<4>(residual_in, &residual_out, &y, &post, &comb, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post_out, comb_out, workspace, stream);
        break;
    }
}

} // namespace infernix::ops
