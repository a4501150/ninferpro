#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_invariant.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next hyper-connection down + inject: the register-streamed route at 1-3 and 9-64 columns (2-7x
// faster than the MMA tile there), the few-row SIMT route at 4-8 (M1 sweep). K1: 2 row(s) per warp, 8 warps
// splitting each row's K, x read through L1.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<10240, 2, 8, 8, MaxCols, ColsPerPass, Q8StreamX::Global, 2>;
} // namespace

Q8Launch select_q8_n324_k10240(std::int32_t tokens) {
    if (tokens <= 3) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r1_t4_w8;
    if (tokens <= 8) return launch_q8_a16_simt_r1_t8_w8;
    if (tokens <= 64) return launch_q8_a16_stream_linear<Stream<16, 8>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    if (tokens <= 1024) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r48_t64;
}

// A16ColumnInvariant: the 16-K-warp sliced-K contraction at every T. The MMA tiles' single K chain
// (640 dependent k16 steps over 11 to 21 CTAs) took 104 us at T = 1, 17 times the A16 route; sixteen K
// warps against eight (2026-10-23, RTX PRO 6000) took 10.2 against 14.3 us at T = 1, 152 against 125 us
// at T = 2048. Two row tiles above 64 columns spill 8-48 bytes of stack; one row tile does not, but
// took 61 against 41 us at T = 512 and 201 against 152 us at T = 2048.
Q8Launch select_q8_n324_k10240_invariant(std::int32_t tokens) {
    return select_q8_sliced_k_invariant<10240, 16>(tokens);
}

} // namespace infernix::ops::detail
