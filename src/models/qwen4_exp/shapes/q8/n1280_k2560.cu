#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_invariant.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next shared expert gate/up: the register-streamed route up to 48 columns but 4 (SIMT there),
// the MMA tiles beyond (the sweep's measured winners are in RESEARCH_NOTES.md, "Q8 MMA tile choices").
// K1: 2 row(s) per warp, x staged in shared memory per pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 4, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;
} // namespace

Q8Launch select_q8_n1280_k2560(std::int32_t tokens) {
    if (tokens <= 3) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_stream_linear<Stream<8, 8>>;
    if (tokens <= 48) return launch_q8_a16_stream_linear<Stream<16, 8>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the swept MMA tiles (RESEARCH_NOTES.md, "Q8 MMA tile
    // choices"). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical; only speed differs.
    if (tokens <= 512) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r64_t128;
}

// A16ColumnInvariant (the MTP shared expert gate/up): the 8-K-warp sliced-K contraction at every T,
// which measured faster than the MMA tiles' single K chain at the draft widths it runs (the numbers
// are in RESEARCH_NOTES.md, "Q8 MMA tile choices").
Q8Launch select_q8_n1280_k2560_invariant(std::int32_t tokens) {
    return select_q8_sliced_k_invariant<2560, 8>(tokens);
}

} // namespace infernix::ops::detail
