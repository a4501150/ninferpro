#pragma once

// Shared small-T E4M3/NVFP4 weight x BF16 activation Tensor Core mainloop.
//
// A CTA owns sixteen output rows per row tile and splits K across compile-time-selected warps.
// Staged codes are widened exactly to BF16 MMA operands; the format's remaining scale (the FP8 row
// scale, the NVFP4 global divisor) is applied once to the complete FP32 dot product. The public
// activation is never quantized. The format trait (format_trait.cuh) supplies the staging layout,
// the widening and the scale.

#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp_a16/format_trait.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// One kernel template rather than per-format entry points over a shared __device__ body: the
// extra inline boundary changes how the split-K reduction's shared addresses are scheduled.
template <class Format, class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp_a16_sliced_k_mma_kernel(
    typename Format::A16Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset) {
    if constexpr (Format::kSlicedStreamingPdl) { pdl::enter_streaming(); }
    const auto* __restrict__ x             = operands.x;
    const auto* __restrict__ weight_codes  = operands.codes;
    const auto* __restrict__ weight_scales = operands.scales;
    const int kHidden                      = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    constexpr int ActiveTokens =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    constexpr bool MaskedColumns = !Schedule::kExactTokens;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kBlockRows     = Schedule::kBlockRows;
    constexpr int kRowTiles      = Schedule::kRowTiles;
    static_assert(kRowTiles == 1 || !RowPolicy::kPaired, "two row tiles store unpaired rows only");
    constexpr int kBlockK      = Schedule::kBlockK;
    const int kGroups          = kHidden / kBlockK;
    constexpr int kBlockTokens = Schedule::kBlockTokens;
    constexpr int kTokenMmas   = kBlockTokens / 8;
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kBlockTokens);
    static_assert((kWarps & 1) == 0);

    union SharedStorage {
        typename Format::template SlicedStaging<Schedule> staging;

        float partial[kWarps * kRowTiles * kTokenMmas * 32 * 4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared = *reinterpret_cast<SharedStorage*>(
        Format::template sliced_shared_storage<Schedule::kSharedBytes>());
    auto& x_shared = shared.staging.activations;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * (kBlockRows / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * ActiveTokens;
    const int live_columns =
        MaskedColumns ? min(ActiveTokens, operands.tokens - token_begin) : ActiveTokens;

    const auto stage_activation = [&](int stage, int group_k0) {
        constexpr auto kActivationCache = Schedule::kActivationCache;
        constexpr bool kPadded          = Format::template kSlicedPaddedActivations<Schedule>;
        constexpr int kStageTokens      = kPadded ? kBlockTokens : ActiveTokens;
        constexpr int kItems            = kStageTokens * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int token = item / (kTileK / 8);
            const int k8    = item - token * (kTileK / 8);
            auto* destination =
                &x_shared[stage][warp][token * kTileK + Format::shared_col_64(token, k8 * 8)];
            if constexpr (!MaskedColumns && (!kPadded || ActiveTokens == kBlockTokens)) {
                cp_async<16, kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token_begin + token) * kHidden +
                                     group_k0 + warp * kTileK + k8 * 8);
            } else {
                const int source_token = token < live_columns ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(token_begin + source_token) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8,
                    token < live_columns ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int stage, int group_k0) {
        Format::template sliced_stage_codes<Schedule>(shared.staging, stage, group_k0, warp, lane,
                                                      weight_codes, weight_scales, row_policy, row0,
                                                      operands.rows, kHidden);
    };

    const int b_row                              = lane & 7;
    const int b_k_offset                         = ((lane >> 3) & 1) << 3;
    const int warp_k0                            = warp * kTileK;
    float accumulators[kRowTiles][kTokenMmas][4] = {};

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_codes(stage, stage * kBlockK);
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const int code_col   = k_step * 16 + lid * 2;
            const auto load_pair = [&](int row, int col) {
                return Format::template sliced_operand<Schedule>(shared.staging, stage, warp_k0,
                                                                 row, col);
            };
            unsigned a[kRowTiles][4];
#pragma unroll
            for (int tile = 0; tile < kRowTiles; ++tile) {
                const int top = tile * 16 + gid;
                a[tile][0]    = load_pair(top, code_col);
                a[tile][1]    = load_pair(top + 8, code_col);
                a[tile][2]    = load_pair(top, code_col + 8);
                a[tile][3]    = load_pair(top + 8, code_col + 8);
            }
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b0;
                unsigned b1;
                const int row = token_mma * 8 + b_row;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(
                        &x_shared[stage][warp][row * kTileK + Format::shared_col_64(
                                                                  row, k_step * 16 + b_k_offset)]));
#pragma unroll
                for (int tile = 0; tile < kRowTiles; ++tile) {
                    mma_bf16(accumulators[tile][token_mma][0], accumulators[tile][token_mma][1],
                             accumulators[tile][token_mma][2], accumulators[tile][token_mma][3],
                             a[tile][0], a[tile][1], a[tile][2], a[tile][3], b0, b1);
                }
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }
    if constexpr (Format::kSlicedStreamingPdl) { pdl::trigger_dependents(); }

    __syncthreads();
    auto* partial            = shared.partial;
    const auto partial_index = [&](int split, int tile, int token_mma) {
        return (((split * kRowTiles + tile) * kTokenMmas + token_mma) * 32 + lane) * 4;
    };
    if ((warp & 1) != 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const auto& acc = accumulators[tile][token_mma];
                store_vec(partial + partial_index(warp, tile, token_mma),
                          make_float4(acc[0], acc[1], acc[2], acc[3]));
            }
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                auto& acc = accumulators[tile][token_mma];
                const float4 partner =
                    load_vec<float4>(partial + partial_index(warp + 1, tile, token_mma));
                acc[0] += partner.x;
                acc[1] += partner.y;
                acc[2] += partner.z;
                acc[3] += partner.w;
                if (warp != 0) {
                    store_vec(partial + partial_index(warp, tile, token_mma),
                              make_float4(acc[0], acc[1], acc[2], acc[3]));
                }
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        const auto destination =
            linear_output_tile<kBlockRows / (RowPolicy::kPaired ? 2 : 1)>(output, row0);
#pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
            const int tile_row = tile * 16;
            float top_scale, bottom_scale;
            Format::sliced_tile_scales(operands, row_policy, row0, tile_row, gid, lid, lane,
                                       top_scale, bottom_scale);

#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const auto& acc = accumulators[tile][token_mma];
                float4 sum      = make_float4(acc[0], acc[1], acc[2], acc[3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value =
                        load_vec<float4>(partial + partial_index(split, tile, token_mma));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int local_token = token_mma * 8 + 2 * lid;
                const int token0      = token_begin + local_token;
                const int row_a       = row_policy.weight_row(row0, tile_row + gid, operands.rows);
                const int row_b = row_policy.weight_row(row0, tile_row + gid + 8, operands.rows);
                if constexpr (RowPolicy::kPaired) {
                    if (local_token < live_columns)
                        epilogue.apply_pair(destination, row_a, token0, sum.x * top_scale,
                                            sum.z * bottom_scale);
                    if (local_token + 1 < live_columns)
                        epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * top_scale,
                                            sum.w * bottom_scale);
                } else {
                    if (local_token < live_columns) {
                        destination.store(row_a, token0,
                                          epilogue.apply(row_a, token0, sum.x * top_scale));
                        destination.store(row_b, token0,
                                          epilogue.apply(row_b, token0, sum.z * bottom_scale));
                    }
                    if (local_token + 1 < live_columns) {
                        destination.store(row_a, token0 + 1,
                                          epilogue.apply(row_a, token0 + 1, sum.y * top_scale));
                        destination.store(row_b, token0 + 1,
                                          epilogue.apply(row_b, token0 + 1, sum.w * bottom_scale));
                    }
                }
            }
        }
    }
}

} // namespace infernix::ops::detail
