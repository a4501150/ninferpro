// GdnChunkArithmetic::Recurrent: the recurrent arithmetic for calls of any width (see launch.h).
//
// recurrent_bf16_direct_kernel waits on each token's global loads and on its two Q/K normalizations
// inside the serial chain. This kernel stages kStageTokens tokens at a time in shared memory (the next
// stage's raw inputs load into registers while the current one runs): every warp normalizes its share
// of the stage's queries and keys with the sequence's own lane mapping (normalize_qk_lane), then each
// warp runs the serial chain over the stage from shared memory, a token's readout reduced together
// with the next token's prediction. Every dot product, reduction and update is the direct kernel's
// (row_partials, warp_sum_rows, update_state), so every output and the final state carry its bits.

#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail::gated_delta_net {
namespace {

constexpr int kStageTokens   = 16;
constexpr int kThreads       = kWarpSize * kNumWarps;
constexpr int kTokensPerWarp = kStageTokens / kNumWarps;
static_assert(kStageTokens % kNumWarps == 0 && kStageTokens <= kThreads);

// A CTA's warps each own Rows consecutive state rows (value dimensions) of one value head; every lane
// holds the same four key dimensions of each row as the direct kernel's lanes do.
template <int Rows>
struct Tile {
    static constexpr int kRows         = Rows * kNumWarps;
    static constexpr int kTiles        = kStateDim / kRows;
    static constexpr int kValuesPerThread = (kStageTokens * kRows + kThreads - 1) / kThreads;
};

template <int Rows>
struct alignas(16) PrefillStage {
    float key[kStageTokens][kStateDim];
    float query[kStageTokens][kStateDim];
    float value[kStageTokens][Tile<Rows>::kRows];
    float g[kStageTokens];
    float beta[kStageTokens];
};

// One stage's raw inputs in registers: each warp's share of the keys and queries (its lanes' four
// elements), the CTA's value rows and the gates.
template <int Rows>
struct StageLoad {
    Bf16x4Pack key[kTokensPerWarp];
    Bf16x4Pack query[kTokensPerWarp];
    __nv_bfloat16 value[Tile<Rows>::kValuesPerThread];
    float g;
    float beta;
};

struct Coordinates {
    int lane, warp, tid;
    std::uint32_t value_head, qk_head, tile_dv, dv_base, dqk_base;
};

template <int Rows>
__device__ __forceinline__ void load_stage(StageLoad<Rows>& load, const DirectAccess& access, const Coordinates& c,
                                           std::int32_t base, int n) {
    const auto column = [&](int t) { return static_cast<std::int64_t>(base + t); };
#pragma unroll
    for (int i = 0; i < kTokensPerWarp; ++i) {
        const int t = c.warp + i * kNumWarps;
        if (t < n) {
            const std::int64_t qk = (column(t) * access.heads.H_qk + c.qk_head) * kStateDim + c.dqk_base;
            load.key[i]           = load_vec<Bf16x4Pack>(access.k + qk);
            load.query[i]         = load_vec<Bf16x4Pack>(access.q + qk);
        }
    }
#pragma unroll
    for (int i = 0; i < Tile<Rows>::kValuesPerThread; ++i) {
        const int e = c.tid + i * kThreads, t = e / Tile<Rows>::kRows, j = e % Tile<Rows>::kRows;
        if (t < n) {
            load.value[i] = access.v[(column(t) * access.heads.H_v + c.value_head) * kStateDim + c.tile_dv + j];
        }
    }
    if (c.tid < n) {
        const RawGatePair gate = load_source_gate(access.g, access.beta, column(c.tid) * access.heads.H_v + c.value_head);
        load.g                 = gate.g;
        load.beta              = gate.beta;
    }
}

// The stage's keys and queries normalized as the sequence normalizes them (each lane its four
// elements), the values widened, into shared memory.
template <bool Normalize, int Rows>
__device__ __forceinline__ void store_stage(PrefillStage<Rows>& stage, const StageLoad<Rows>& load,
                                            const Coordinates& c, int n) {
#pragma unroll
    for (int i = 0; i < kTokensPerWarp; ++i) {
        const int t = c.warp + i * kNumWarps;
        if (t < n) {
            RawQkLane key   = raw_qk_lane(load.key[i]);
            RawQkLane query = raw_qk_lane(load.query[i]);
            normalize_qk_lane<Normalize>(key.value, c.lane);
            normalize_qk_lane<Normalize>(query.value, c.lane);
            store_qk_lane(key.value, stage.key[t], c.dqk_base);
            store_qk_lane(query.value, stage.query[t], c.dqk_base);
        }
    }
#pragma unroll
    for (int i = 0; i < Tile<Rows>::kValuesPerThread; ++i) {
        const int e = c.tid + i * kThreads, t = e / Tile<Rows>::kRows, j = e % Tile<Rows>::kRows;
        if (t < n) { stage.value[t][j] = __bfloat162float(load.value[i]); }
    }
    if (c.tid < n) {
        stage.g[c.tid]    = load.g;
        stage.beta[c.tid] = load.beta;
    }
}

template <bool Normalize, int Rows>
__global__ void __launch_bounds__(kThreads)
    recurrent_prefill_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                             const float* __restrict__ beta, const float* __restrict__ state_read,
                             float* __restrict__ state_write, __nv_bfloat16* __restrict__ out, std::int32_t width,
                             head_map heads, float scale) {
    __shared__ PrefillStage<Rows> stage;
    const DirectAccess access{q, k, v, g, beta, state_read, state_write, out, heads, width, scale};
    Coordinates c;
    c.lane       = static_cast<int>(threadIdx.x);
    c.warp       = static_cast<int>(threadIdx.y);
    c.tid        = c.warp * kWarpSize + c.lane;
    c.value_head = blockIdx.x;
    c.qk_head    = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(c.value_head)));
    c.tile_dv    = blockIdx.z * Tile<Rows>::kRows;
    c.dv_base    = c.tile_dv + c.warp * Rows;
    c.dqk_base   = static_cast<std::uint32_t>(c.lane * kQkPerLane);
    const std::int64_t head_offset = static_cast<std::int64_t>(c.value_head) * kStateDim * kStateDim;
    __align__(16) float state[Rows][kQkPerLane];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        load_qk_lane(state[r], state_read + head_offset + static_cast<std::int64_t>(c.dv_base + r) * kStateDim,
                     c.dqk_base);
    }
    constexpr int kShift = kRowLaneShift<2 * Rows>;

    StageLoad<Rows> load;
    load_stage(load, access, c, 0, min(kStageTokens, width));
    for (std::int32_t base = 0; base < width; base += kStageTokens) {
        const int n = min(kStageTokens, width - base);
        store_stage<Normalize>(stage, load, c, n);
        __syncthreads();
        if (base + kStageTokens < width) {
            load_stage(load, access, c, base + kStageTokens, min(kStageTokens, width - base - kStageTokens));
        }
        // Token t's readout and token t + 1's prediction both read the state after t: their reductions
        // run together (rows 0..Rows-1 the readout, Rows..2 Rows-1 the prediction; lane l ends with
        // row l >> kShift). Each is the arithmetic of apply_gdn_transition and readout_values_and_store.
        float key[kQkPerLane], query[kQkPerLane], prediction[Rows];
        load_qk_lane(key, stage.key[0], c.dqk_base);
        {
            row_partials<Rows>(prediction, state, key);
            const float sum = warp_reduce_scatter_rows(prediction, c.lane);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                prediction[r] = __shfl_sync(kFullWarpMask, sum, r << kRowLaneShift<Rows>, kWarpSize);
            }
        }
        for (int t = 0; t < n; ++t) {
            float value[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) { value[r] = stage.value[t][c.warp * Rows + r]; }
            update_state_rows<Rows>(state, prediction, key, value, stage.g[t], stage.beta[t]);
            load_qk_lane(query, stage.query[t], c.dqk_base);
            float sums[2 * Rows] = {};
            row_partials<Rows>(*reinterpret_cast<float(*)[Rows]>(sums), state, query);
            const bool next = t + 1 < n;
            if (next) {
                load_qk_lane(key, stage.key[t + 1], c.dqk_base);
                row_partials<Rows>(*reinterpret_cast<float(*)[Rows]>(sums + Rows), state, key);
            }
            const float sum = warp_reduce_scatter_rows(sums, c.lane);
            if (next) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    prediction[r] = __shfl_sync(kFullWarpMask, sum, (Rows + r) << kShift, kWarpSize);
                }
            }
            if ((c.lane & ((1 << kShift) - 1)) == 0 && (c.lane >> kShift) < Rows) {
                out[((base + t) * static_cast<std::int64_t>(heads.H_v) + c.value_head) * kStateDim + c.dv_base +
                    (c.lane >> kShift)] = __float2bfloat16(sum * scale);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        store_qk_lane(state[r], state_write + head_offset + static_cast<std::int64_t>(c.dv_base + r) * kStateDim,
                      c.dqk_base);
    }
}

// Rows per warp, measured at T = 2048 on the RTX PRO 6000 (the bits do not depend on it).
constexpr int kRowsPerWarp = 4;

template <bool Normalize>
void launch_fixed(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
                  float scale, const Tensor& state_read, Tensor& state_write, Tensor& out, cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(Tile<kRowsPerWarp>::kTiles));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_prefill_kernel<Normalize, kRowsPerWarp><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const float*>(state_read.data),
        static_cast<float*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2], heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent_prefill(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                              cudaStream_t stream) {
    if (normalize_qk) {
        launch_fixed<true>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
    } else {
        launch_fixed<false>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
    }
}

} // namespace infernix::ops::detail::gated_delta_net
