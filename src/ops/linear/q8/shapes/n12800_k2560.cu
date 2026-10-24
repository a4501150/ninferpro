#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next PLE key/value projection: the register-streamed route up to 8 columns (6-28 % faster than
// SIMT), the MMA tiles beyond (M1 sweep). K1: 2 row(s) per warp, x staged in shared memory per pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;
// The column-invariant route's decode tile: four warps of 16 rows x 8 columns over 128-wide K steps
// (2026-10-23 sweep, RTX PRO 6000: 28.7 us at T = 1-16 against 38.9 for the 32 x 64 tile).
using Narrow = Q8A16MmaSchedule<64, 8, 128, 16, 8, 1, 2>;

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n12800_k2560(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 8) return launch_q8_a16_stream_linear<Stream<8, 8>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    if (tokens <= 128) return launch_q8_a16_mma_r48_t64;
    return launch_q8_a16_mma_r64_t128;
}

// A16ColumnInvariant: the MMA tiles at every T (each accumulates an output over K in the same order;
// at decode widths the smallest tile runs with its unused columns masked).
Q8Launch select_q8_n12800_k2560_invariant(std::int32_t tokens) {
    if (tokens <= 16) return mma<Narrow>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    return select_q8_n12800_k2560(tokens);
}

} // namespace infernix::ops::detail
