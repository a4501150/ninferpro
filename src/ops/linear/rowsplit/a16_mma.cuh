#pragma once

// Shared RowSplit x BF16 MMA kernel. Scaled weights are materialized as BF16;
// accumulation is FP32. N/K/T stay runtime values and the launcher owns bounds.

#include "ops/common/mma.cuh"
#include "ops/linear/rowsplit/format_trait.cuh"
#include "ops/linear/rowsplit/schedule.cuh"

#include <type_traits>

namespace infernix::ops::detail {

// XOR swizzle for a [rows][64] BF16 tile. The eight 16-byte column groups are
// permuted by the low row bits so ldmatrix reads do not repeatedly hit the same
// shared-memory bank group.
__device__ __forceinline__ int rowsplit_mma_swizzle_k64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// Views of one CTA's staged state for the format traits. The arrays themselves are separate
// __shared__ declarations in the body (As, Bs, Cr, [Hr], Sr): a struct would place them at fixed
// offsets of one symbol and change the generated addressing.
template <class Schedule, class Format, bool HasHigh = Format::kHasHigh>
struct RowSplitMmaShared;

template <class Schedule, class Format>
struct RowSplitMmaShared<Schedule, Format, true> {
    __nv_bfloat16 (&As)[Schedule::kBlockRows * Schedule::kBlockK];
    __nv_bfloat16 (&Bs)[Schedule::kActivationStages][Schedule::kBlockTokens * Schedule::kBlockK];
    std::uint8_t (&Cr)[Schedule::kStages][Schedule::kBlockRows * Schedule::kGroupsPerK *
                                          Format::Storage::kCodeBytesPerGroup];
    std::uint8_t (&Hr)[Schedule::kStages][Schedule::kBlockRows * Schedule::kGroupsPerK *
                                          Format::Storage::kHighBytesPerGroup];
    std::uint8_t (&Sr)[Schedule::kStages]
                      [Schedule::kBlockRows * Schedule::kGroupsPerK * Schedule::kScaleBytes];
};

template <class Schedule, class Format>
struct RowSplitMmaShared<Schedule, Format, false> {
    __nv_bfloat16 (&As)[Schedule::kBlockRows * Schedule::kBlockK];
    __nv_bfloat16 (&Bs)[Schedule::kActivationStages][Schedule::kBlockTokens * Schedule::kBlockK];
    std::uint8_t (&Cr)[Schedule::kStages][Schedule::kBlockRows * Schedule::kGroupsPerK *
                                          Format::Storage::kCodeBytesPerGroup];
    std::uint8_t (&Sr)[Schedule::kStages]
                      [Schedule::kBlockRows * Schedule::kGroupsPerK * Schedule::kScaleBytes];
};

// clang-format off
template <class Schedule, class Format, bool Full, class Output, class Epilogue>
__device__ __forceinline__
void rowsplit_a16_mma_body(
    int tid,
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales,
    Output output, Epilogue epilogue,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k, std::int32_t token_begin) {
    // clang-format on
    static_assert(std::is_same_v<typename Schedule::RowSplitFormat, Format>);
    constexpr bool kFull = Full;
    constexpr int BM     = Schedule::kBlockRows;
    constexpr int BN     = Schedule::kBlockTokens;
    constexpr int BK     = Schedule::kBlockK;
    constexpr int WM     = Schedule::kWarpRows;
    constexpr int WN     = Schedule::kWarpTokens;
    constexpr int MT     = Schedule::kMmaRows;
    constexpr int NT     = Schedule::kMmaTokens;
    constexpr int KSUB   = Schedule::kMmaKSteps;
    constexpr int S      = Schedule::kStages;
    constexpr int BS     = Schedule::kActivationStages;
    constexpr int GPB    = Schedule::kGroupsPerK;
    constexpr int SB     = Schedule::kScaleBytes;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[BS][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * GPB * Format::Storage::kCodeBytesPerGroup];
    // Unreferenced, and so not allocated, for formats without a high plane.
    __shared__ __align__(16)
        std::uint8_t Hr[S][BM * GPB * (Format::kHasHigh ? Format::kHighBytesPerGroup : 1)];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * GPB * SB];
    using Shared        = RowSplitMmaShared<Schedule, Format>;
    const Shared shared = [&]() -> Shared {
        if constexpr (Format::kHasHigh) {
            return {As, Bs, Cr, Hr, Sr};
        } else {
            return {As, Bs, Cr, Sr};
        }
    }();

    const int groups_per_row = padded_k / Format::Storage::kGroupK;
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridTokens;
    const int warp_col       = warp % Schedule::kWarpGridTokens;
    const int mma_row        = lane >> 2;
    const int mma_col        = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    float accum[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            accum[mi][ni][0] = 0.0f;
            accum[mi][ni][1] = 0.0f;
            accum[mi][ni][2] = 0.0f;
            accum[mi][ni][3] = 0.0f;
        }
    }

    const int k_tiles = padded_k / BK;

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst =
                &shared.Bs[stage][local_col * BK + rowsplit_mma_swizzle_k64(local_col, k8 * 8)];
            if constexpr (kFull) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                if (col < cols && kk + 8 <= k) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group0 = (k_tile * BK) / Format::Storage::kGroupK;
#pragma unroll 1
        for (int item = tid; item < BM * GPB * 2; item += Schedule::kThreads) {
            const int row_group = item >> 1;
            const int half      = item & 1;
            const int local_row = row_group / GPB;
            const int group     = row_group - local_row * GPB;
            const int row       = row0 + local_row;
            auto* dst =
                &shared.Cr[stage][row_group * Format::Storage::kCodeBytesPerGroup + half * 16];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                cp_async<16, Schedule::kWeightCache>(
                    dst, &codes[group_index * Format::Storage::kCodeBytesPerGroup + half * 16]);
            } else {
                if (Format::kBoundGroups ? (row < rows && group0 + group < (k + 63) / 64)
                                         : row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                    cp_async<16, Schedule::kWeightCache>(
                        dst, &codes[group_index * Format::Storage::kCodeBytesPerGroup + half * 16]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }

        if constexpr (Format::kHasHigh) {
            Format::template mma_stage_high<Schedule, kFull>(shared, stage, high, row0, rows, k,
                                                             groups_per_row, group0, tid);
        }

#pragma unroll 1
        for (int row_group = tid; row_group < BM * GPB; row_group += Schedule::kThreads) {
            const int local_row   = row_group / GPB;
            const int group       = row_group - local_row * GPB;
            const int row         = row0 + local_row;
            const int scale_group = group0 + group;
            auto* dst             = &shared.Sr[stage][row_group * SB];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                if constexpr (Schedule::kScaleLoadMode == RowSplitScaleLoad::Pair32) {
                    const int aligned_group = scale_group & ~1;
                    const std::int64_t aligned_index =
                        static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                    if (aligned_group + 1 < groups_per_row) {
                        cp_async<4>(dst,
                                    &scales[aligned_index * Format::Storage::kScaleBytesPerGroup]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Format::Storage::kScaleBytesPerGroup]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(
                            &scales[group_index * Format::Storage::kScaleBytesPerGroup]);
                }
            } else {
                if (Format::kBoundGroups ? (row < rows && group0 + group < (k + 63) / 64)
                                         : row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                    if constexpr (Schedule::kScaleLoadMode == RowSplitScaleLoad::Pair32) {
                        const int aligned_group = scale_group & ~1;
                        const std::int64_t aligned_index =
                            static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                        if (aligned_group + 1 < groups_per_row) {
                            cp_async<4>(
                                dst, &scales[aligned_index * Format::Storage::kScaleBytesPerGroup]);
                        } else {
                            *reinterpret_cast<std::uint16_t*>(dst) =
                                *reinterpret_cast<const std::uint16_t*>(
                                    &scales[group_index * Format::Storage::kScaleBytesPerGroup]);
                            *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                        }
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Format::Storage::kScaleBytesPerGroup]);
                    }
                } else {
                    dst[0] = 0;
                    dst[1] = 0;
                    if constexpr (Schedule::kScaleLoadMode == RowSplitScaleLoad::Pair32) {
                        dst[2] = 0;
                        dst[3] = 0;
                    }
                }
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    auto decode_weight = [&](int stage, int k_tile) {
        const int scale_group = (k_tile * BK) / Format::Storage::kGroupK;
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            auto* dst = &shared.As[local_row * BK];
#pragma unroll
            for (int group = 0; group < GPB; ++group) {
                const int staged_group = local_row * GPB + group;
                const std::uint8_t* scale_ptr =
                    &shared.Sr[stage][staged_group * SB +
                                      (Schedule::kScaleLoadMode == RowSplitScaleLoad::Pair32
                                           ? ((scale_group + group) & 1) *
                                                 Format::Storage::kScaleBytesPerGroup
                                           : 0)];
                const __nv_bfloat162 weights =
                    Format::mma_decode_pair(shared, stage, scale_ptr, staged_group, lane);
                const int shared_col = rowsplit_mma_swizzle_k64(
                    local_row, group * Format::Storage::kGroupK + 2 * lane);
                store_vec(&dst[shared_col], weights);
            }
        }
    };

    if constexpr (BS == S) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            if (stage < k_tiles) { stage_inputs(stage, stage); }
            cp_commit();
        }
    } else {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            if (stage < k_tiles) { stage_quant(stage, stage); }
            cp_commit();
        }
        if (k_tiles > 0) { stage_activation(0, 0); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        if constexpr (BS == S) {
            cp_wait<S - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        decode_weight(stage, k_tile);
        __syncthreads();

        if constexpr (Format::mma_quant_prefetch(BS, S)) {
            const int prefetch_quant_tile = k_tile + S;
            if (prefetch_quant_tile < k_tiles) { stage_quant(stage, prefetch_quant_tile); }
            cp_commit();
        }

        auto load_fragments = [&](int k_step, unsigned (&a_frag)[MT][4],
                                  unsigned (&b_frag)[NT][2]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_row * WM + mi * 16 + a_row_offset;
                const int col = k_step + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&shared.As[row * BK + rowsplit_mma_swizzle_k64(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_col * WN + ni * 8 + b_inner_row;
                const int col = k_step + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&shared.Bs[BS == 1 ? 0 : stage]
                                                [row * BK + rowsplit_mma_swizzle_k64(row, col)]));
            }
        };

        if constexpr (Schedule::kFragmentPipeline == RowSplitMmaFragmentPipeline::PingPong) {
            unsigned a_frag[2][MT][4];
            unsigned b_frag[2][NT][2];
            load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int current = ki & 1;
                const int next    = (ki + 1) & 1;
                if (ki + 1 < KSUB) { load_fragments((ki + 1) * 16, a_frag[next], b_frag[next]); }
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                 accum[mi][ni][3], a_frag[current][mi][0], a_frag[current][mi][1],
                                 a_frag[current][mi][2], a_frag[current][mi][3],
                                 b_frag[current][ni][0], b_frag[current][ni][1]);
                    }
                }
            }
        } else {
            unsigned a_frag[MT][4];
            unsigned b_frag[NT][2];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                load_fragments(ki * 16, a_frag, b_frag);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                 accum[mi][ni][3], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2],
                                 a_frag[mi][3], b_frag[ni][0], b_frag[ni][1]);
                    }
                }
            }
        }

        __syncthreads();
        if constexpr (BS == S) {
            const int prefetch_tile = k_tile + S;
            if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
            cp_commit();
        } else {
            const int next_tile = k_tile + 1;
            if (next_tile < k_tiles) { stage_activation(0, next_tile); }
            cp_commit();
        }
    }

    if constexpr (Format::kFinishTile && requires {
                      epilogue.template finish_tile<Schedule, Full>(output, shared.Bs[0], accum,
                                                                    row0, token_begin + col0, rows,
                                                                    token_begin + cols);
                  }) {
        epilogue.template finish_tile<Schedule, Full>(output, shared.Bs[0], accum, row0,
                                                      token_begin + col0, rows, token_begin + cols);
    } else {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row0 = static_cast<int>(blockIdx.x) * BM + warp_row * WM + mi * 16 + mma_row;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token0 = col0 + warp_col * WN + ni * 8 + 2 * mma_col;
                const auto store = [&](int row, int token, float value) {
                    if (kFull || (row < rows && token < cols)) {
                        const int global_token = token_begin + token;
                        output.store(row, global_token, epilogue.apply(row, global_token, value));
                    }
                };
                store(row0, token0, accum[mi][ni][0]);
                store(row0, token0 + 1, accum[mi][ni][1]);
                store(row0 + 8, token0, accum[mi][ni][2]);
                store(row0 + 8, token0 + 1, accum[mi][ni][3]);
            }
        }
    }
}

// Launch entry points over the shared body. Formats with a high plane take
// the historical high pointer; a format without one (Q4) keeps its historical
// parameter list, so the compiled parameter layout is unchanged.
// clang-format off
template <class Schedule, class Format, bool Full, class Output, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void rowsplit_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales,
    Output output, Epilogue epilogue,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k, std::int32_t token_begin) {
    // clang-format on
    static_assert(Format::kHasHigh);
    rowsplit_a16_mma_body<Schedule, Format, Full, Output, Epilogue>(
        static_cast<int>(threadIdx.x), x, codes, high, scales, output, epilogue, rows, k, cols,
        padded_k, token_begin);
}

// clang-format off
template <class Schedule, class Format, bool Full, class Output, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void rowsplit_a16_mma_kernel_no_high(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    Output output, Epilogue epilogue,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k, std::int32_t token_begin) {
    // clang-format on
    static_assert(!Format::kHasHigh);
    rowsplit_a16_mma_body<Schedule, Format, Full, Output, Epilogue>(
        static_cast<int>(threadIdx.x), x, codes, nullptr, scales, output, epilogue, rows, k, cols,
        padded_k, token_begin);
}

} // namespace infernix::ops::detail
