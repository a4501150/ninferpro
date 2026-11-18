#pragma once

// Shared RowSplit x BF16 SIMT kernel. Warps split a row at scale-pair
// boundaries; the weight format (Q4/Q5/Q6 trait) supplies the decode atom, the
// staged planes and the tail handling.

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/rowsplit/format_trait.cuh"
#include "ops/linear/rowsplit/schedule.cuh"

#include <type_traits>

namespace infernix::ops::detail {

// Staged codes, high plane and scale pairs for one warp pipeline. Formats
// without a high plane omit the member rather than sizing it to zero.
template <class Schedule, bool HasHigh>
struct RowSplitSimtStaging;

template <class Schedule>
struct RowSplitSimtStaging<Schedule, true> {
    alignas(16) uint4 codes[Schedule::kWarps][Schedule::kStages][Schedule::kCodeVecsPerStage];
    alignas(16) uint4 high[Schedule::kWarps][Schedule::kStages][Schedule::kHighVecsPerStage];
    alignas(16)
        std::uint32_t scales[Schedule::kWarps][Schedule::kStages][Schedule::kScalePairsPerStage];
};

template <class Schedule>
struct RowSplitSimtStaging<Schedule, false> {
    alignas(16) uint4 codes[Schedule::kWarps][Schedule::kStages][Schedule::kCodeVecsPerStage];
    alignas(16)
        std::uint32_t scales[Schedule::kWarps][Schedule::kStages][Schedule::kScalePairsPerStage];
};

// The staged high plane of one warp pipeline slot, or null for formats without
// one; unused by those formats' decode.
template <bool HasHigh, class Staging>
__device__ __forceinline__ uint4* rowsplit_simt_high_plane(Staging& staging, int warp, int slot) {
    if constexpr (HasHigh) {
        return staging.high[warp][slot];
    } else {
        return nullptr;
    }
}

template <class Schedule>
__device__ __forceinline__ void rowsplit_simt_copy_code(uint4* shared_dst,
                                                        const std::uint8_t* global_src) {
    if constexpr (Schedule::kCodeCache == Cache::cg) {
        cp_async<16, Cache::cg>(shared_dst, global_src);
    } else {
        cp_async<16, Cache::ca>(shared_dst, global_src);
    }
}

template <class Schedule, class Format, bool FullStage>
__device__ __forceinline__ void rowsplit_simt_issue_stage(
    uint4* __restrict__ shared_codes, uint4* __restrict__ shared_high,
    std::uint32_t* __restrict__ shared_scales, const std::uint8_t* __restrict__ code_row,
    const std::uint8_t* __restrict__ high_row, const std::uint8_t* __restrict__ scale_row,
    int stage, int active_groups, int lane) {
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0 = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerWarpStage;
    const std::uint8_t* stage_codes = code_row + group0 * Format::Storage::kCodeBytesPerGroup;
    [[maybe_unused]] const std::uint8_t* stage_high = [&] {
        if constexpr (Format::kHasHigh && !Format::kSimtHighByPairs)
            return high_row + group0 * Format::Storage::kHighBytesPerGroup;
        else
            return static_cast<const std::uint8_t*>(nullptr);
    }();
    const std::uint8_t* stage_scales = scale_row + group0 * Format::Storage::kScaleBytesPerGroup;

    const int active_code_vecs = FullStage ? kCodeVecs
                                           : active_groups * Format::Storage::kCodeBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_code_vecs) {
            rowsplit_simt_copy_code<Schedule>(&shared_codes[vec],
                                              stage_codes + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    if constexpr (Format::kSimtHighByPairs) {
        // One high vector per scale pair, addressed from the row as the Q5 kernel always has.
        for (int pair = lane; pair < Schedule::kHighVecsPerStage; pair += 32) {
            if (FullStage || pair < active_groups / 2)
                cp_async<16, Schedule::kCodeCache>(
                    &shared_high[pair],
                    high_row + group0 * Format::Storage::kHighBytesPerGroup + pair * 16);
            else
                shared_high[pair] = uint4{0, 0, 0, 0};
        }
    } else if constexpr (Format::kHasHigh) {
        const int active_high_vecs = FullStage
                                         ? Schedule::kHighVecsPerStage
                                         : active_groups * Format::Storage::kHighBytesPerGroup /
                                               static_cast<int>(sizeof(uint4));
        for (int vec = lane; vec < Schedule::kHighVecsPerStage; vec += 32) {
            if (FullStage || vec < active_high_vecs) {
                rowsplit_simt_copy_code<Schedule>(&shared_high[vec],
                                                  stage_high + static_cast<std::int64_t>(vec) * 16);
            } else {
                shared_high[vec] = uint4{0u, 0u, 0u, 0u};
            }
        }
    }

    const int active_scale_pairs = FullStage ? kScalePairs : active_groups / 2;
    for (int pair = lane; pair < kScalePairs; pair += 32) {
        if (FullStage || pair < active_scale_pairs) {
            cp_async<4>(&shared_scales[pair], stage_scales + static_cast<std::int64_t>(pair) * 4);
        } else {
            shared_scales[pair] = 0u;
        }
    }
    cp_commit();
}

template <class Schedule, class Format, bool FullStage, bool FullCols>
__device__ __forceinline__ void rowsplit_simt_consume_stage(
    const __nv_bfloat16* __restrict__ x, std::int32_t k, int col0, int active_cols, int stage,
    int active_groups, const uint4* __restrict__ shared_codes,
    const uint4* __restrict__ shared_high, const std::uint32_t* __restrict__ shared_scales,
    int lane, float (&acc)[Schedule::kBlockTokens]) {
    constexpr int kCols       = Schedule::kBlockTokens;
    constexpr int kCodePhases = Schedule::kCodePhases;

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group =
            phase * Format::kSimtGroupsPerPhase + (lane >> Format::kSimtLaneGroupShift);
        const int stage_groups = FullStage ? Schedule::kGroupsPerWarpStage : active_groups;
        if (group < stage_groups) {
            const typename Format::SimtCodes packed =
                reinterpret_cast<const typename Format::SimtCodes*>(
                    shared_codes)[phase * 32 + lane];
            const auto high_bits           = Format::simt_preload_high(shared_high, phase, lane);
            const std::uint32_t scale_pair = shared_scales[group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));

            float weights[8];
            Format::simt_decode_eight(shared_high, high_bits, packed, scale_bits, phase, lane,
                                      weights);

            const std::int64_t xk = static_cast<std::int64_t>(stage) * Schedule::kStageK +
                                    static_cast<std::int64_t>(phase) * 256 + lane * 8;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                if (FullCols || col < active_cols) {
                    const uint4 values =
                        load_vec<uint4>(x + static_cast<std::int64_t>(col0 + col) * k + xk);
                    const float2 x0 = bf16x2_bits_to_float2(values.x);
                    const float2 x1 = bf16x2_bits_to_float2(values.y);
                    const float2 x2 = bf16x2_bits_to_float2(values.z);
                    const float2 x3 = bf16x2_bits_to_float2(values.w);
                    acc[col]        = fmaf(weights[0], x0.x, acc[col]);
                    acc[col]        = fmaf(weights[1], x0.y, acc[col]);
                    acc[col]        = fmaf(weights[2], x1.x, acc[col]);
                    acc[col]        = fmaf(weights[3], x1.y, acc[col]);
                    acc[col]        = fmaf(weights[4], x2.x, acc[col]);
                    acc[col]        = fmaf(weights[5], x2.y, acc[col]);
                    acc[col]        = fmaf(weights[6], x3.x, acc[col]);
                    acc[col]        = fmaf(weights[7], x3.y, acc[col]);
                }
            }
        }
    }
}

// Warps split a row at scale-pair boundaries. With one warp per row the
// original warp-local pipeline is retained and the CTA reduction compiles out.
template <class Schedule, class Format, bool Full, bool FullK, class Output, class Epilogue,
          bool TriggerPdl, bool JoinPdl>
__device__ __forceinline__ void rowsplit_a16_simt_body(
    int tid, const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    static_assert(std::is_same_v<typename Schedule::RowSplitFormat, Format>);
    if constexpr (TriggerPdl) {
        if (tid == 0) pdl::trigger_dependents();
    }
    constexpr int R = Schedule::kBlockRows;
    constexpr int W = Schedule::kWarpsPerRow;
    constexpr int T = Schedule::kBlockTokens;
    constexpr int S = Schedule::kStages;
    constexpr int G = Schedule::kGroupsPerWarpStage;

    union SharedStorage {
        RowSplitSimtStaging<Schedule, Format::kHasHigh> staging;
        float partial[R][W][T];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    __shared__ SharedStorage shared;
    const int lane      = tid & 31;
    const int warp      = tid >> 5;
    const int local_row = warp / W;
    const int split     = warp % W;
    const int row       = static_cast<int>(blockIdx.x) * R + local_row;
    if constexpr (W == 1 && !Full) {
        if (row >= rows) return;
    }
    const int safe_row      = row < rows ? row : 0;
    int group_begin         = 0;
    const int vector_groups = Format::simt_vector_groups(k);
    int groups              = vector_groups;
    if constexpr (W > 1) {
        const int pairs = vector_groups / 2;
        group_begin     = (pairs * split / W) * 2;
        groups          = row < rows ? (pairs * (split + 1) / W) * 2 - group_begin : 0;
    }
    const int stages        = FullK ? groups / G : (groups + G - 1) / G;
    const int token0        = static_cast<int>(blockIdx.y) * T;
    const int active_tokens = Full ? T : min(T, tokens - token0);
    const std::int64_t group0 =
        static_cast<std::int64_t>(safe_row) * (padded_k / Format::Storage::kGroupK) + group_begin;
    const auto* code_row = codes + group0 * 32;
    const auto* high_row = [&] {
        if constexpr (Format::kHasHigh)
            return high + group0 * Format::Storage::kHighBytesPerGroup;
        else
            return static_cast<const std::uint8_t*>(nullptr);
    }();
    const auto* scale_row = scales + group0 * 2;
    const auto* warp_x    = x + group_begin * Format::Storage::kGroupK;
    auto& stage_codes     = shared.staging.codes;
    auto& stage_scales    = shared.staging.scales;
    float acc[T]          = {};
#pragma unroll
    for (int prefetch = 0; prefetch < S - 1; ++prefetch) {
        if (prefetch < stages) {
            if constexpr (Format::kSimtScalesBeforeHigh) {
                uint4* const slot_codes          = stage_codes[warp][prefetch];
                std::uint32_t* const slot_scales = stage_scales[warp][prefetch];
                rowsplit_simt_issue_stage<Schedule, Format, FullK>(
                    slot_codes,
                    rowsplit_simt_high_plane<Format::kHasHigh>(shared.staging, warp, prefetch),
                    slot_scales, code_row, high_row, scale_row, prefetch,
                    min(G, groups - prefetch * G), lane);
            } else {
                rowsplit_simt_issue_stage<Schedule, Format, FullK>(
                    stage_codes[warp][prefetch],
                    rowsplit_simt_high_plane<Format::kHasHigh>(shared.staging, warp, prefetch),
                    stage_scales[warp][prefetch], code_row, high_row, scale_row, prefetch,
                    min(G, groups - prefetch * G), lane);
            }
        } else {
            cp_commit();
        }
    }
#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + S - 1;
        if (fetch < stages) {
            if constexpr (Format::kSimtScalesBeforeHigh) {
                uint4* const slot_codes          = stage_codes[warp][fetch % S];
                std::uint32_t* const slot_scales = stage_scales[warp][fetch % S];
                rowsplit_simt_issue_stage<Schedule, Format, FullK>(
                    slot_codes,
                    rowsplit_simt_high_plane<Format::kHasHigh>(shared.staging, warp, fetch % S),
                    slot_scales, code_row, high_row, scale_row, fetch, min(G, groups - fetch * G),
                    lane);
            } else {
                rowsplit_simt_issue_stage<Schedule, Format, FullK>(
                    stage_codes[warp][fetch % S],
                    rowsplit_simt_high_plane<Format::kHasHigh>(shared.staging, warp, fetch % S),
                    stage_scales[warp][fetch % S], code_row, high_row, scale_row, fetch,
                    min(G, groups - fetch * G), lane);
            }
        } else {
            cp_commit();
        }
        cp_wait<S - 1>();
        __syncwarp();
        if constexpr (Format::kSimtScalesBeforeHigh) {
            const int stage_groups                 = min(G, groups - stage * G);
            const auto* stage_code_words           = stage_codes[warp][stage % S];
            const std::uint32_t* stage_scale_pairs = stage_scales[warp][stage % S];
            rowsplit_simt_consume_stage<Schedule, Format, FullK, Full>(
                warp_x, k, token0, active_tokens, stage, stage_groups, stage_code_words,
                rowsplit_simt_high_plane<Format::kHasHigh>(shared.staging, warp, stage % S),
                stage_scale_pairs, lane, acc);
        } else {
            rowsplit_simt_consume_stage<Schedule, Format, FullK, Full>(
                warp_x, k, token0, active_tokens, stage, min(G, groups - stage * G),
                stage_codes[warp][stage % S],
                rowsplit_simt_high_plane<Format::kHasHigh>(shared.staging, warp, stage % S),
                stage_scales[warp][stage % S], lane, acc);
        }
        __syncwarp();
    }
    if constexpr (Format::kScalarRemainder) {
        // Warp zero owns the remainder; other K warps only own complete scale pairs.
        if (!FullK && split == 0 && row < rows) {
            constexpr int kGroupK = Format::Storage::kGroupK;
            for (int group = vector_groups; group < (k + kGroupK - 1) / kGroupK; ++group) {
                const int kk = group * kGroupK + lane * 2;
                // A group narrower than the warp's 64 elements leaves the upper lanes idle.
                if ((kGroupK >= 64 || lane < kGroupK / 2) && kk < k) {
                    float w0, w1;
                    Format::simt_remainder_pair(codes, high, scales,
                                                std::int64_t(row) * (padded_k / kGroupK) + group,
                                                lane, w0, w1);
#pragma unroll
                    for (int token = 0; token < T; ++token) {
                        if (Full || token < active_tokens) {
                            const auto index = std::int64_t(token0 + token) * k + kk;
                            acc[token]       = fmaf(w0, __bfloat162float(x[index]), acc[token]);
                            if (kk + 1 < k)
                                acc[token] = fmaf(w1, __bfloat162float(x[index + 1]), acc[token]);
                        }
                    }
                }
            }
        }
    }
    if constexpr (W > 1) {
        cp_wait<0>();
        __syncthreads(); // All staged reads and asynchronous writes finish before reuse.
    }
    if constexpr (Format::kSimtApplyRow) {
        float sums[T];
#pragma unroll
        for (int token = 0; token < T; ++token) {
            sums[token] = warp_reduce_sum(acc[token]);
            if constexpr (W > 1) {
                if (lane == 0) shared.partial[local_row][split][token] = sums[token];
            }
        }
        if constexpr (W > 1) {
            __syncthreads();
            if (split == 0 && lane == 0 && row < rows) {
#pragma unroll
                for (int token = 0; token < T; ++token) {
                    sums[token] = shared.partial[local_row][0][token];
#pragma unroll
                    for (int part = 1; part < W; ++part)
                        sums[token] += shared.partial[local_row][part][token];
                }
                linear_finish_row(output, epilogue, row, token_begin + token0, sums, active_tokens);
            }
        } else if (lane == 0) {
            linear_finish_row(output, epilogue, row, token_begin + token0, sums, active_tokens);
        }
    } else {
#pragma unroll
        for (int token = 0; token < T; ++token) {
            const float sum = warp_reduce_sum(acc[token]);
            if constexpr (W == 1) {
                if (lane == 0 && token < active_tokens) {
                    const int global_token = token_begin + token0 + token;
                    output.store(row, global_token, epilogue.apply(row, global_token, sum));
                }
            } else {
                if (lane == 0) shared.partial[local_row][split][token] = sum;
            }
        }
        if constexpr (W > 1) {
            __syncthreads();
            if (split == 0 && lane == 0 && row < rows) {
#pragma unroll
                for (int token = 0; token < T; ++token) {
                    if (token < active_tokens) {
                        float sum = shared.partial[local_row][0][token];
#pragma unroll
                        for (int part = 1; part < W; ++part)
                            sum += shared.partial[local_row][part][token];
                        const int global_token = token_begin + token0 + token;
                        output.store(row, global_token, epilogue.apply(row, global_token, sum));
                    }
                }
            }
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

// Launch entry points over the shared body. Formats with a high plane take
// the historical high pointer; a format without one (Q4) keeps its historical
// parameter list, so the compiled parameter layout is unchanged.
template <class Schedule, class Format, bool Full, bool FullK, class Output, class Epilogue,
          bool TriggerPdl = false, bool JoinPdl = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void rowsplit_a16_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    static_assert(Format::kHasHigh);
    rowsplit_a16_simt_body<Schedule, Format, Full, FullK, Output, Epilogue, TriggerPdl, JoinPdl>(
        static_cast<int>(threadIdx.x), x, codes, high, scales, output, epilogue, rows, k, tokens,
        padded_k, token_begin);
}

template <class Schedule, class Format, bool Full, bool FullK, class Output, class Epilogue,
          bool TriggerPdl = false, bool JoinPdl = false>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kMinBlocksPerSm) void rowsplit_a16_simt_kernel_no_high(const __nv_bfloat16* __restrict__ x,
                                                               const std::
                                                                   uint8_t* __restrict__ codes,
                                                               const std::
                                                                   uint8_t* __restrict__ scales,
                                                               Output output, Epilogue epilogue,
                                                               int rows, int k, int tokens,
                                                               int padded_k, int token_begin) {
    static_assert(!Format::kHasHigh);
    rowsplit_a16_simt_body<Schedule, Format, Full, FullK, Output, Epilogue, TriggerPdl, JoinPdl>(
        static_cast<int>(threadIdx.x), x, codes, nullptr, scales, output, epilogue, rows, k, tokens,
        padded_k, token_begin);
}

} // namespace infernix::ops::detail
