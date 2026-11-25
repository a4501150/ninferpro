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

// Glm5Next: the shared expert's gate and up projections, T = a call's columns.
// bench/ops/q8_sweep.cu --n 4096 --k 4096 (RTX PRO 6000, trimmed mean of 21 cold-L2 graphs,
// 2 % tolerance to T = 128 and 5 % beyond; T <= 8 from one column-invariant class):
// T = 1 12.6 us (79% of the DRAM floor), T = 1024 165 us, T = 4096 587 us.
Q8Launch select_q8_n4096_k4096(std::int32_t tokens) {
    if (tokens <= 9) return sliced<Q8A16SlicedKMmaSchedule<8, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 24) return sliced<Q8A16SlicedKMmaSchedule<24, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 40) return sliced<Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 52) return sliced<Q8A16SlicedKMmaSchedule<64, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 64) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return sliced<Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>;
    if (tokens <= 102) return launch_q8_a16_mma_r64x16_t48_k128_a1;
    if (tokens <= 128) return launch_q8_a16_mma_r64x32_t64_k128_a1;
    if (tokens <= 256) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 384) return launch_q8_a16_mma_r96_t96;
    if (tokens <= 512) return launch_q8_a16_mma_r64_t128;
    if (tokens <= 768) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
