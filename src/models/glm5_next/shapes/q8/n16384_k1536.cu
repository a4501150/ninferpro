#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace infernix::ops::detail {
namespace {

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(q8_linear_operands(x, weight),
                                         LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                         LinearIdentityEpilogue{}, stream);
}

} // namespace

// Glm5Next: the MLA query up projection, T = a call's columns.
// bench/ops/q8_sweep.cu --n 16384 --k 1536 (RTX PRO 6000, trimmed mean of 21 cold-L2 graphs,
// 2 % tolerance to T = 128 and 5 % beyond; T <= 8 from one column-invariant class):
// T = 1 16.4 us (91% of the DRAM floor), T = 1024 230 us, T = 4096 899 us.
Q8Launch select_q8_n16384_k1536(std::int32_t tokens) {
    if (tokens <= 8) return sliced<Q8A16SlicedKMmaSchedule<8, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 16) return sliced<Q8A16SlicedKMmaSchedule<16, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 24) return sliced<Q8A16SlicedKMmaSchedule<24, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 48) return launch_q8_a16_mma_r64x16_t48_k128_a1;
    if (tokens <= 63) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 64) return launch_q8_a16_mma_r64x32_t64_k128_a1;
    if (tokens <= 96) return launch_q8_a16_mma_r96_t96;
    if (tokens <= 128) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 192) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
