#pragma once
// The sliced-K column-invariant route family (A16ColumnInvariant): KWarps K warps of 64 at every width.
// Each output's K is split the same way and its FP32 partials reduced in the same order whatever the
// token tile, row tiles or stages, so a column's bits do not depend on the call's width. Included by
// the shape files that register it; each instantiates its own K.
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

#include <cstdint>

namespace infernix::ops::detail {

template <int K, int KWarps, int Tokens, int RowTiles = 1, int Stages = (Tokens <= 16 && RowTiles == 1 ? 2 : 1)>
using Q8SlicedKInvariant = Q8A16SlicedKMmaSchedule<Tokens, KWarps, Stages, 1, Q8ScaleAccess::Shared, Cache::ca,
                                                   Cache::cg, Q8ActivationStage::PaddedZero, K, Tokens, false,
                                                   RowTiles>;

template <class Schedule>
void launch_q8_sliced_k_invariant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(q8_linear_operands(x, weight),
                                         LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                         LinearIdentityEpilogue{}, stream);
}

// 8-, 16- and 32-token tiles up to 64 columns, then two 16-row tiles per CTA sharing the staged
// activation.
template <int K, int KWarps>
Q8Launch select_q8_sliced_k_invariant(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_sliced_k_invariant<Q8SlicedKInvariant<K, KWarps, 8>>;
    if (tokens <= 16) return launch_q8_sliced_k_invariant<Q8SlicedKInvariant<K, KWarps, 16>>;
    if (tokens <= 64) return launch_q8_sliced_k_invariant<Q8SlicedKInvariant<K, KWarps, 32>>;
    return launch_q8_sliced_k_invariant<Q8SlicedKInvariant<K, KWarps, 32, 2>>;
}

} // namespace infernix::ops::detail
