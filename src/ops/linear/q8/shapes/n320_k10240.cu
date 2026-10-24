#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next final mixer down: the register-streamed route at 1-3 and 9-64 columns (2-7x faster than
// the MMA tile there), the few-row SIMT route at 4-8 (M1 sweep). K1: 2 row(s) per warp, 8 warps splitting each
// row's K, x read through L1.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<10240, 2, 8, 8, MaxCols, ColsPerPass, Q8StreamX::Global, 2>;

// The column-invariant route: sixteen K warps of 64 at every width, as the hyper-connection down + inject
// projection (n324_k10240.cu).
template <int Tokens, int RowTiles = 1, int Stages = (Tokens <= 16 && RowTiles == 1 ? 2 : 1)>
using Invariant = Q8A16SlicedKMmaSchedule<Tokens, 16, Stages, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 10240, Tokens, false, RowTiles>;

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(q8_linear_operands(x, weight),
                                         LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                         LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n320_k10240(std::int32_t tokens) {
    if (tokens <= 3) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r1_t4_w8;
    if (tokens <= 8) return launch_q8_a16_simt_r1_t8_w8;
    if (tokens <= 64) return launch_q8_a16_stream_linear<Stream<16, 8>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

// A16ColumnInvariant: the 16-K-warp sliced-K contraction at every T (the MMA tiles' single K chain took
// 104 us at T = 1).
Q8Launch select_q8_n320_k10240_invariant(std::int32_t tokens) {
    if (tokens <= 8) return sliced<Invariant<8>>;
    if (tokens <= 16) return sliced<Invariant<16>>;
    if (tokens <= 64) return sliced<Invariant<32>>;
    return sliced<Invariant<32, 2>>;
}

} // namespace infernix::ops::detail
