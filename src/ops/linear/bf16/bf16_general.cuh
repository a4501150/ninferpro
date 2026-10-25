#pragma once

// General-purpose BF16 x BF16 GEMM fallback for shapes the specialised bf16 linear path does not
// tile (for example a full-precision vocab projection kept as BF16 by a QAT checkpoint). Computes
// out[T, N] = x[T, K] * weight[N, K]^T with an fp32 accumulation and a BF16 store, matching the
// contiguous token-major layout of the specialised kernels. Tiling is fixed and the extents are
// runtime, so any (N, K, T) is handled without a per-shape instantiation.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {
constexpr std::int32_t kTileN         = 32;
constexpr std::int32_t kTileT         = 32;
constexpr std::int32_t kTileK         = 32;
constexpr std::int32_t kThreads       = 256;
constexpr std::int32_t kThreadCols    = kThreads / 16;      // 16
constexpr std::int32_t kRowsPerThread = kTileN / 16;        // 2
constexpr std::int32_t kColsPerThread = kTileT / kThreadCols;  // 2

__global__ void bf16_general_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ weight,
                                         __nv_bfloat16* __restrict__ out, std::int32_t N,
                                         std::int32_t K, std::int32_t T) {
    __shared__ __nv_bfloat16 ws[kTileN][kTileK + 1];
    __shared__ __nv_bfloat16 wx[kTileT][kTileK + 1];

    const std::int32_t out_row0 = static_cast<std::int32_t>(blockIdx.x) * kTileN;
    const std::int32_t token0   = static_cast<std::int32_t>(blockIdx.y) * kTileT;
    const std::int32_t tid      = static_cast<std::int32_t>(threadIdx.x);

    const std::int32_t trow = tid / kThreadCols;  // 0..15
    const std::int32_t tcol = tid % kThreadCols;  // 0..15

    float accum[kRowsPerThread][kColsPerThread] = {};

    for (std::int32_t k0 = 0; k0 < K; k0 += kTileK) {
        for (std::int32_t i = tid; i < kTileN * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = out_row0 + r;
            const std::int32_t gc = k0 + c;
            ws[r][c + 1] = (gr < N && gc < K)
                               ? weight[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        for (std::int32_t i = tid; i < kTileT * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = token0 + r;
            const std::int32_t gc = k0 + c;
            wx[r][c + 1] = (gr < T && gc < K)
                               ? x[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
            const std::int32_t row = trow * kRowsPerThread + rr;
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = tcol * kColsPerThread + cc;
                float sum = accum[rr][cc];
#pragma unroll
                for (std::int32_t kk = 0; kk < kTileK; ++kk) {
                    sum = fmaf(static_cast<float>(wx[col][kk + 1]),
                               static_cast<float>(ws[row][kk + 1]), sum);
                }
                accum[rr][cc] = sum;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
        const std::int32_t row = out_row0 + trow * kRowsPerThread + rr;
        if (row < N) {
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = token0 + tcol * kColsPerThread + cc;
                if (col < T) {
                    out[static_cast<std::int64_t>(col) * N + row] =
                        __float2bfloat16_rn(accum[rr][cc]);
                }
            }
        }
    }
}

// The same arithmetic in 64 x 64 tiles: each thread owns ThreadRows rows x ThreadTokens tokens and, as
// the kernel above, accumulates every output as one fmaf chain over k = 0, 1, ..., K - 1 from zero,
// with K zero-padded to a multiple of 32 (kTileK), and rounds it to BF16 once. It multiplies the BF16
// operands without widening them (fma.rn.f32.bf16 equals fmaf on their exact FP32 values) and stages
// each 32-wide K slice as k-pair words ([k / 2][row]), so one 16-byte shared load feeds four rows or
// tokens. Requires K % 8 == 0 and 16-byte aligned x and weight.
constexpr std::int32_t kTiledThreads = 256;
constexpr std::int32_t kTiledPairs   = kTileK / 2; // k-pair words per row per slice

__device__ __forceinline__ void fma_bf16_half(float& acc, unsigned w, unsigned x, bool high) {
    const unsigned short wh = static_cast<unsigned short>(high ? w >> 16 : w & 0xFFFFU);
    const unsigned short xh = static_cast<unsigned short>(high ? x >> 16 : x & 0xFFFFU);
    asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(acc) : "h"(wh), "h"(xh));
}

// [640, 2560] (the QSA indexer) on RTX PRO 6000, 2026-10-24: 74-84 us from 65 to 1024 columns against
// 227-384 for the kernel above, 909 against 3,858 us at 16,384 (59 TFLOP/s); 128 x 128 tiles of 8 x 8
// were as fast or slower at every measured shape (979 us at 16,384, 305-330 us below 2,048 columns).
template <int TileRows, int TileTokens, int ThreadRows, int ThreadTokens>
__global__ void __launch_bounds__(kTiledThreads, 2)
    bf16_general_tiled_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                              __nv_bfloat16* __restrict__ out, std::int32_t N, std::int32_t K, std::int32_t T) {
    constexpr int kThreadColumns = TileTokens / ThreadTokens;
    static_assert((TileRows / ThreadRows) * kThreadColumns == kTiledThreads);
    static_assert(ThreadRows % 4 == 0 && ThreadTokens % 4 == 0 && TileRows == TileTokens);
    constexpr int kLoads = TileRows * 4 / kTiledThreads; // 16-byte chunks per thread per operand
    static_assert(kLoads >= 1 && TileRows * 4 % kTiledThreads == 0);
    __shared__ __align__(16) unsigned sw[2][kTiledPairs][TileRows];
    __shared__ __align__(16) unsigned sx[2][kTiledPairs][TileTokens];
    const std::int32_t row0 = static_cast<std::int32_t>(blockIdx.x) * TileRows;
    const std::int32_t tok0 = static_cast<std::int32_t>(blockIdx.y) * TileTokens;
    const std::int32_t tid  = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t ty = tid / kThreadColumns, tx = tid % kThreadColumns;
    // Staging: thread tid loads 16-byte chunks tid / TileRows + u * (threads / TileRows) of row/token
    // tid % TileRows.
    const std::int32_t line = tid % TileRows, chunk = tid / TileRows;
    uint4 wr[kLoads], xr[kLoads];
    const auto load = [&](std::int32_t k0) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const std::int32_t k   = k0 + 8 * (chunk + u * (kTiledThreads / TileRows));
            const std::int32_t row = row0 + line, tok = tok0 + line;
            wr[u] = row < N && k < K ? *reinterpret_cast<const uint4*>(weight + static_cast<std::int64_t>(row) * K + k)
                                     : make_uint4(0, 0, 0, 0);
            xr[u] = tok < T && k < K ? *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(tok) * K + k)
                                     : make_uint4(0, 0, 0, 0);
        }
    };
    const auto store = [&](int buffer) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int p = 4 * (chunk + u * (kTiledThreads / TileRows));
            sw[buffer][p + 0][line] = wr[u].x;
            sw[buffer][p + 1][line] = wr[u].y;
            sw[buffer][p + 2][line] = wr[u].z;
            sw[buffer][p + 3][line] = wr[u].w;
            sx[buffer][p + 0][line] = xr[u].x;
            sx[buffer][p + 1][line] = xr[u].y;
            sx[buffer][p + 2][line] = xr[u].z;
            sx[buffer][p + 3][line] = xr[u].w;
        }
    };
    float acc[ThreadRows][ThreadTokens] = {};
    const std::int32_t slices = (K + kTileK - 1) / kTileK;
    load(0);
    store(0);
    __syncthreads();
    for (std::int32_t s = 0; s < slices; ++s) {
        const int buffer = s & 1;
        if (s + 1 < slices) { load((s + 1) * kTileK); }
#pragma unroll 4
        for (int p = 0; p < kTiledPairs; ++p) {
            unsigned wv[ThreadRows], xv[ThreadTokens];
#pragma unroll
            for (int i = 0; i < ThreadRows; i += 4) {
                const uint4 w = *reinterpret_cast<const uint4*>(&sw[buffer][p][ty * ThreadRows + i]);
                wv[i] = w.x, wv[i + 1] = w.y, wv[i + 2] = w.z, wv[i + 3] = w.w;
            }
#pragma unroll
            for (int i = 0; i < ThreadTokens; i += 4) {
                const uint4 v = *reinterpret_cast<const uint4*>(&sx[buffer][p][tx * ThreadTokens + i]);
                xv[i] = v.x, xv[i + 1] = v.y, xv[i + 2] = v.z, xv[i + 3] = v.w;
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) { // k = 2p, then 2p + 1
#pragma unroll
                for (int r = 0; r < ThreadRows; ++r) {
#pragma unroll
                    for (int c = 0; c < ThreadTokens; ++c) { fma_bf16_half(acc[r][c], wv[r], xv[c], half != 0); }
                }
            }
        }
        if (s + 1 < slices) {
            store(buffer ^ 1);
            __syncthreads();
        }
    }
#pragma unroll
    for (int c = 0; c < ThreadTokens; ++c) {
        const std::int32_t tok = tok0 + tx * ThreadTokens + c;
        if (tok >= T) { continue; }
#pragma unroll
        for (int r = 0; r < ThreadRows; ++r) {
            const std::int32_t row = row0 + ty * ThreadRows + r;
            if (row < N) { out[static_cast<std::int64_t>(tok) * N + row] = __float2bfloat16_rn(acc[r][c]); }
        }
    }
}

} // namespace infernix::ops::detail
