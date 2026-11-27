#pragma once

// Implements: include/infernix/ops/top_logprobs.h
// Match: contiguous BF16 or FP32 [physical_rows,C], I32 [K,C], and FP32 [K,C].
// Algorithm assumptions: one 1024-thread CTA per column fuses single-pass online logsumexp with a
// per-thread top-4 buffer and tracks the maximum dropped key; ranks strictly above that floor are
// extracted directly from registers with exact tie-breaking, and any remaining ranks fall back to
// exact column scans.

#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kTopLogprobsBlock = 1024;

// Larger keys rank first: descending value, then ascending row. Equal values share their high
// half (zero of either sign included), and every valid row's key is nonzero.
__device__ __forceinline__ unsigned long long top_logprobs_key(float value, std::int32_t row) {
    const unsigned bits    = __float_as_uint(value == 0.0f ? 0.0f : value);
    const unsigned ordered = (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
    return (static_cast<unsigned long long>(ordered) << 32) |
           (0xffffffffu - static_cast<unsigned>(row));
}

template <int BlockSize>
__device__ __forceinline__ unsigned long long top_logprobs_block_max(unsigned long long value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ unsigned long long warp_maxima[kWarps];
    __shared__ unsigned long long result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = max(value, __shfl_xor_sync(kFullWarpMask, value, offset));
    }
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : 0ULL;
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            value = max(value, __shfl_xor_sync(kFullWarpMask, value, offset));
        }
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

template <int BlockSize, class Logit>
__launch_bounds__(BlockSize) __global__
    void top_logprobs_kernel(const Logit* logits, std::int32_t* ids, float* output,
                             std::int32_t valid_rows, std::int32_t physical_rows,
                             std::int32_t count) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const Logit* values       = logits + static_cast<std::int64_t>(column) * physical_rows;

    // Single pass: online logsumexp + per-thread top-4 keys + max dropped key.
    float local_max              = -CUDART_INF_F;
    float local_sum              = 0.0f;
    unsigned long long top[4]    = {0ULL, 0ULL, 0ULL, 0ULL};
    unsigned long long dropped   = 0ULL;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value   = logprob_logit(values[row]);
        const float new_max = fmaxf(local_max, value);
        if (new_max > local_max) { local_sum *= expf(local_max - new_max); }
        local_max = new_max;
        local_sum += expf(value - local_max);

        const unsigned long long key = top_logprobs_key(value, row);
        if (key > top[3]) {
            if (top[3] > dropped) { dropped = top[3]; }
            if (key > top[0]) {
                top[3] = top[2];
                top[2] = top[1];
                top[1] = top[0];
                top[0] = key;
            } else if (key > top[1]) {
                top[3] = top[2];
                top[2] = top[1];
                top[1] = key;
            } else if (key > top[2]) {
                top[3] = top[2];
                top[2] = key;
            } else {
                top[3] = key;
            }
        } else if (key > dropped) {
            dropped = key;
        }
    }

    const float maximum = target_logprobs_block_max<BlockSize>(local_max);
    local_sum *= expf(local_max - maximum);
    __shared__ float warp_sums[BlockSize / kWarpSize];
    __shared__ float log_sum_s;
    const float sum = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    if (threadIdx.x == 0) { log_sum_s = logf(sum); }
    __syncthreads();
    const float log_sum = log_sum_s;

    // Every unretained key in the column is <= global_floor, so any best > global_floor is exact.
    const unsigned long long global_floor = top_logprobs_block_max<BlockSize>(dropped);
    unsigned long long previous           = ~0ULL;
    int top_idx                           = 0;
    std::int32_t rank                     = 0;
    for (; rank < count; ++rank) {
        const unsigned long long cand = top_idx < 4 ? top[top_idx] : 0ULL;
        const unsigned long long best = top_logprobs_block_max<BlockSize>(cand);
        if (best <= global_floor) { break; }
        if (threadIdx.x == 0) {
            const auto row           = static_cast<std::int32_t>(0xffffffffu - static_cast<unsigned>(best));
            const std::int64_t index = static_cast<std::int64_t>(column) * count + rank;
            ids[index]               = row;
            output[index]            = logprob_logit(values[row]) - maximum - log_sum;
        }
        if (top_idx < 4 && top[top_idx] == best) { ++top_idx; }
        previous = best;
    }

    // Exact fallback for any ranks <= global_floor.
    for (; rank < count; ++rank) {
        unsigned long long best = 0ULL;
        for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
             row += BlockSize) {
            const unsigned long long key = top_logprobs_key(logprob_logit(values[row]), row);
            if (key < previous && key > best) { best = key; }
        }
        best = top_logprobs_block_max<BlockSize>(best);
        if (threadIdx.x == 0) {
            const auto row           = static_cast<std::int32_t>(0xffffffffu - static_cast<unsigned>(best));
            const std::int64_t index = static_cast<std::int64_t>(column) * count + rank;
            ids[index]               = row;
            output[index]            = logprob_logit(values[row]) - maximum - log_sum;
        }
        previous = best;
    }
}

} // namespace infernix::ops
