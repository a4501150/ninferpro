#pragma once

// Compile-time weight formats for the floating-point-code A16 kernels (E4M3 and NVFP4 weights,
// BF16 activations). The shared bodies in this directory own staging, the MMA main loop and the
// output; a trait pins what a stored format changes: its code and scale staging, the exact
// widening of codes to BF16 operands, the shared-tile swizzle and where its scale meets the
// FP32 reduction. A new stored format adds a trait, not a new kernel.

#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_shared.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// E4M3 codes with one represented BF16 scale per output row. The codes widen exactly to BF16;
// the row scale multiplies the complete FP32 reduction.
struct Fp8A16Format {
    using Scale       = __nv_bfloat16;
    using A16Operands = Fp8A16Operands;

    template <class Schedule>
    static constexpr bool kPingPong =
        Schedule::kFragmentPipeline == Fp8MmaFragmentPipeline::PingPong;

    // No collective epilogue pairs C-fragment rows for this format.
    static constexpr bool kThreadPairedEpilogue = false;

    static __device__ __forceinline__ int shared_col_64(int row, int col) {
        return fp8_a16_shared_col_64(row, col);
    }

    // Codes only; the row scales are read at the output.
    template <class Schedule>
    static __device__ __forceinline__ std::uint8_t* mma_scale_shared(std::uint8_t*) {
        return nullptr;
    }

    template <class Schedule, class RowPolicy>
    static __device__ __forceinline__ void
    mma_stage_codes(int tid, std::uint8_t* code_shared, std::uint8_t* /*scale_shared*/,
                    const std::uint8_t* __restrict__ weight_codes,
                    const Scale* __restrict__ /*scales*/, const RowPolicy& row_policy,
                    int row_begin, int M, int K, int k_tile) {
        constexpr int BM  = Schedule::kBlockRows;
        constexpr int BK  = Schedule::kBlockK;
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += Schedule::kThreads) {
            const int local_row = item / (BK / 16);
            const int chunk     = item - local_row * (BK / 16);
            cp_async<16, Schedule::kWeightCache>(
                &code_shared[local_row * BK + chunk * 16],
                weight_codes +
                    static_cast<std::int64_t>(row_policy.weight_row(row_begin, local_row, M)) * K +
                    k_begin + chunk * 16);
        }
    }

    template <class Schedule>
    static __device__ __forceinline__ unsigned mma_widen_pair(const std::uint8_t* code_shared,
                                                              const std::uint8_t* /*scale_shared*/,
                                                              int row, int col) {
        constexpr int BK      = Schedule::kBlockK;
        const unsigned packed = static_cast<unsigned>(
            *reinterpret_cast<const std::uint16_t*>(&code_shared[row * BK + col]));
        return fp8_e4m3x2_to_bf16x2_bits(packed);
    }

    template <class RowPolicy>
    static __device__ __forceinline__ float
    finish_tile_scale(const Scale* __restrict__ scales, float /*alpha*/,
                      const RowPolicy& row_policy, int row_begin, int local_row, int M) {
        return __bfloat162float(scales[row_policy.weight_row(row_begin, local_row, M)]);
    }

    static __device__ __forceinline__ float store_scale(const Scale* __restrict__ scales,
                                                        float /*alpha*/, int row) {
        return __bfloat162float(__ldg(scales + row));
    }

    // --- sliced-K family ---

    // The kernel streams its weights through the main loop: it waits for the producer first and
    // lets dependents launch once that loop is done.
    static constexpr bool kSlicedStreamingPdl = true;

    template <class Schedule>
    static constexpr bool kSlicedPaddedActivations =
        Schedule::kActivationStage == Fp8ActivationStage::PaddedZero;

    template <int Bytes>
    static __device__ __forceinline__ unsigned char* sliced_shared_storage() {
        return fp8_shared_storage<Bytes>();
    }

    template <class Schedule>
    struct SlicedStaging {
        std::uint8_t codes[Schedule::kStages][Schedule::kBlockRows][Schedule::kBlockK];
        __nv_bfloat16 activations[Schedule::kStages][Schedule::kKWarps]
                                 [Schedule::kBlockTokens * Schedule::kTileKPerWarp];
    };

    template <class Schedule, class Staging, class RowPolicy>
    static __device__ __forceinline__ void
    sliced_stage_codes(Staging& staging, int stage, int group_k0, int warp, int lane,
                       const std::uint8_t* __restrict__ weight_codes,
                       const Scale* __restrict__ /*scales*/, const RowPolicy& row_policy, int row0,
                       int rows, int hidden) {
        constexpr auto kWeightCache = Schedule::kWeightCache;
        constexpr int kBlockK       = Schedule::kBlockK;
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row = warp * Schedule::kRowsPerLoaderWarp + row_item;
            for (int chunk = lane; chunk < kBlockK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, kWeightCache>(
                    &staging.codes[stage][row][swizzled_chunk * 16],
                    weight_codes +
                        static_cast<std::int64_t>(row_policy.weight_row(row0, row, rows)) * hidden +
                        group_k0 + chunk * 16);
            }
        }
    }

    // The BF16 operand pair of staged row `row` at warp-tile column `col`.
    template <class Schedule, class Staging>
    static __device__ __forceinline__ unsigned sliced_operand(const Staging& staging, int stage,
                                                              int warp_k0, int row, int col) {
        const int chunk  = (warp_k0 + col) >> 4;
        const int offset = (chunk ^ (row & 7)) * 16 + (col & 15);
        return fp8_e4m3x2_to_bf16x2_bits(static_cast<unsigned>(
            *reinterpret_cast<const std::uint16_t*>(&staging.codes[stage][row][offset])));
    }

    // The scales of C-fragment rows gid and gid + 8 of a 16-row tile: lanes 0 and 1 of each quad
    // load them and the quad shares them.
    template <class Operands, class RowPolicy>
    static __device__ __forceinline__ void
    sliced_tile_scales(const Operands& operands, const RowPolicy& row_policy, int row0,
                       int tile_row, int gid, int lid, int lane, float& top_scale,
                       float& bottom_scale) {
        constexpr unsigned kMask = 0xffffffffU;
        unsigned lane_scale      = 0;
        if (lid < 2) {
            lane_scale = static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(
                operands
                    .scales)[row_policy.weight_row(row0, tile_row + gid + lid * 8, operands.rows)]);
        }
        const unsigned top_scale_bits    = __shfl_sync(kMask, lane_scale, lane & ~3);
        const unsigned bottom_scale_bits = __shfl_sync(kMask, lane_scale, (lane & ~3) + 1);
        top_scale =
            __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(top_scale_bits)));
        bottom_scale =
            __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(bottom_scale_bits)));
    }
};

// NVFP4 (E2M1) codes with raw E4M3 scales per 16 K elements and one FP32 divisor per tensor.
// A code times its group scale is exact in BF16; the divisor (alpha) multiplies the complete
// FP32 reduction.
struct Nvfp4A16Format {
    using Scale       = std::uint8_t;
    using A16Operands = Nvfp4A16Operands;

    template <class Schedule>
    static constexpr bool kPingPong =
        Schedule::kFragmentPipeline == Nvfp4MmaFragmentPipeline::PingPong;

    // Thread-paired row policies (gate g, up g+8 of one output row) finish through apply_pair.
    static constexpr bool kThreadPairedEpilogue = true;

    static __device__ __forceinline__ int shared_col_64(int row, int col) {
        return nvfp4_a16_shared_col_64(row, col);
    }

    // The tile's group scales follow its packed codes.
    template <class Schedule>
    static __device__ __forceinline__ std::uint8_t* mma_scale_shared(std::uint8_t* code_shared) {
        return code_shared + Schedule::kBlockRows * (Schedule::kBlockK / 2);
    }

    template <class Schedule, class RowPolicy>
    static __device__ __forceinline__ void
    mma_stage_codes(int tid, std::uint8_t* code_shared, std::uint8_t* scale_shared,
                    const std::uint8_t* __restrict__ weight_codes,
                    const Scale* __restrict__ weight_scales, const RowPolicy& row_policy,
                    int row_begin, int M, int K, int k_tile) {
        constexpr int BM  = Schedule::kBlockRows;
        constexpr int BK  = Schedule::kBlockK;
        const int k_begin = k_tile * BK;
        for (int item = tid; item < BM * (BK / 32); item += Schedule::kThreads) {
            const int row = item / (BK / 32), chunk = item % (BK / 32);
            const int parent = row_policy.weight_row(row_begin, row, M);
            cp_async<16, Schedule::kWeightCache>(code_shared + row * (BK / 2) + chunk * 16,
                                                 weight_codes +
                                                     static_cast<std::int64_t>(parent) * (K / 2) +
                                                     k_begin / 2 + chunk * 16);
        }
        for (int item = tid; item < BM * (BK / 64); item += Schedule::kThreads) {
            const int row = item / (BK / 64), tile = item % (BK / 64);
            const int parent = row_policy.weight_row(row_begin, row, M);
            cp_async<4>(scale_shared + row * (BK / 16) + tile * 4,
                        weight_scales +
                            nvfp4_scale_byte_offset(parent, k_begin / 16 + tile * 4, K));
        }
    }

    template <class Schedule>
    static __device__ __forceinline__ unsigned mma_widen_pair(const std::uint8_t* code_shared,
                                                              const std::uint8_t* scale_shared,
                                                              int row, int col) {
        constexpr int BK = Schedule::kBlockK;
        return nvfp4_scaled_pair_bf16(code_shared[row * (BK / 2) + col / 2],
                                      scale_shared[row * (BK / 16) + col / 16]);
    }

    template <class RowPolicy>
    static __device__ __forceinline__ float
    finish_tile_scale(const Scale* __restrict__, float alpha, const RowPolicy&, int, int, int) {
        return alpha;
    }

    static __device__ __forceinline__ float store_scale(const Scale* __restrict__, float alpha,
                                                        int) {
        return alpha;
    }

    // --- sliced-K family ---

    static constexpr bool kSlicedStreamingPdl = false;

    template <class Schedule>
    static constexpr bool kSlicedPaddedActivations =
        Schedule::kActivationStage == Nvfp4ActivationStage::PaddedZero;

    template <int Bytes>
    static __device__ __forceinline__ unsigned char* sliced_shared_storage() {
        return nvfp4_shared_storage<Bytes>();
    }

    template <class Schedule>
    struct SlicedStaging {
        std::uint8_t codes[Schedule::kStages][Schedule::kBlockRows][Schedule::kBlockK / 2];
        std::uint8_t scales[Schedule::kStages][Schedule::kBlockRows][Schedule::kBlockK / 16];
        __nv_bfloat16 activations[Schedule::kStages][Schedule::kKWarps]
                                 [Schedule::kBlockTokens * Schedule::kTileKPerWarp];
    };

    // 16-byte code chunks are swizzled over at most eight chunks per row.
    template <class Schedule>
    static constexpr int kSlicedCodeSwizzle =
        Schedule::kBlockK / 32 < 8 ? Schedule::kBlockK / 32 - 1 : 7;

    template <class Schedule, class Staging, class RowPolicy>
    static __device__ __forceinline__ void
    sliced_stage_codes(Staging& staging, int stage, int group_k0, int warp, int lane,
                       const std::uint8_t* __restrict__ weight_codes,
                       const Scale* __restrict__ weight_scales, const RowPolicy& row_policy,
                       int row0, int rows, int hidden) {
        constexpr int kBlockK      = Schedule::kBlockK;
        constexpr int kCodeChunks  = kBlockK / 32;
        constexpr int kCodeSwizzle = kSlicedCodeSwizzle<Schedule>;
#pragma unroll
        for (int ri = 0; ri < Schedule::kRowsPerLoaderWarp; ++ri) {
            const int row    = warp * Schedule::kRowsPerLoaderWarp + ri;
            const int parent = row_policy.weight_row(row0, row, rows);
            for (int chunk = lane; chunk < kCodeChunks; chunk += 32) {
                cp_async<16, Schedule::kWeightCache>(
                    &staging.codes[stage][row][(chunk ^ (row & kCodeSwizzle)) * 16],
                    weight_codes + static_cast<std::int64_t>(parent) * (hidden / 2) + group_k0 / 2 +
                        chunk * 16);
            }
            for (int tile = lane; tile < kBlockK / 64; tile += 32) {
                cp_async<4>(&staging.scales[stage][row][tile * 4],
                            weight_scales +
                                nvfp4_scale_byte_offset(parent, group_k0 / 16 + tile * 4, hidden));
            }
        }
    }

    template <class Schedule, class Staging>
    static __device__ __forceinline__ unsigned sliced_operand(const Staging& staging, int stage,
                                                              int warp_k0, int row, int col) {
        constexpr int kCodeSwizzle = kSlicedCodeSwizzle<Schedule>;
        const int byte             = (warp_k0 + col) / 2;
        const int offset           = ((byte / 16) ^ (row & kCodeSwizzle)) * 16 + (byte & 15);
        return nvfp4_scaled_pair_bf16(staging.codes[stage][row][offset],
                                      staging.scales[stage][row][(warp_k0 + col) / 16]);
    }

    template <class Operands, class RowPolicy>
    static __device__ __forceinline__ void
    sliced_tile_scales(const Operands& operands, const RowPolicy&, int, int, int, int, int,
                       float& top_scale, float& bottom_scale) {
        top_scale    = operands.alpha;
        bottom_scale = operands.alpha;
    }
};

} // namespace infernix::ops::detail
