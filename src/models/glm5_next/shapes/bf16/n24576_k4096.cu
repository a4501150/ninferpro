#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Glm5Next: the KDA qkv projection and the dense MLP's gate and up projections, T = a call's
// columns. T <= 64 (decode, DFlash2 verify, small calls) keeps the general route's chunked skinny
// GEMV; wider calls take the fastest measured route per interval
// (bench/ops/bf16_sweep.cu --n 24576 --k 4096, RTX PRO 6000, 2 % tolerance, seams at the last
// measured T: 60.3 -> 2.25 ms at T = 4096).
Bf16Launch select_bf16_n24576_k4096(std::int32_t tokens) {
    if (tokens <= 64) return select_bf16_general_launch(tokens);
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>>; // tma_r128t64k64s4
    if (tokens <= 2048)
        return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8>>; // tma_r128t128k64s3_w64x32_group8
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>>; // tma_r128t128k64s3_w64x32
}

} // namespace infernix::ops::detail
