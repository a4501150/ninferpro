#pragma once

#include "ops/common/rope_d128_theta1e7_table.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace infernix::ops {

// D128 full-head 1-D RoPE frequencies for theta=1e7.
static __device__ __constant__ double kRopeD128Theta1e7InvFrequency[64] = {
    INFERNIX_ROPE_D128_THETA1E7_INV_FREQUENCY};

__device__ __forceinline__ void rope_d128_theta1e7_sincos(const std::int32_t* positions, int token,
                                                          int pair, float* sine, float* cosine) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    const double angle =
        static_cast<double>(positions[token]) * kRopeD128Theta1e7InvFrequency[pair];
    const double turns  = angle * kInvTwoPi;
    const float reduced = static_cast<float>(angle - nearbyint(turns) * kTwoPi);
    sincosf(reduced, sine, cosine);
}

} // namespace infernix::ops
