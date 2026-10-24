#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_invariant.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next MTP embedding and hidden projections: the register-streamed route up to 8 columns but 4
// (SIMT there), the MMA tiles beyond (M1 sweep). K1: 2 row(s) per warp, x staged in shared memory per pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;
} // namespace

Q8Launch select_q8_n2560_k2560(std::int32_t tokens) {
    if (tokens <= 3) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_stream_linear<Stream<8, 8>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

// A16ColumnInvariant (the MTP embedding and hidden projections): the 8-K-warp sliced-K contraction at
// every T (2026-10-24, RTX PRO 6000: 8.2 against 20.3 us at T = 1 for the MMA tiles' single K chain
// over 40 CTAs (the width route 8.0); 236 against 154 us for the width route at T = 2048, where only
// the MTP's prompt pass runs it).
Q8Launch select_q8_n2560_k2560_invariant(std::int32_t tokens) {
    return select_q8_sliced_k_invariant<2560, 8>(tokens);
}

} // namespace infernix::ops::detail
