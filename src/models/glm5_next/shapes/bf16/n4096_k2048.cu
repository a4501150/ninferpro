#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Glm5Next: the shared expert's down projection, T = a call's columns. T <= 64 keeps the general
// route's chunked skinny GEMV; wider calls take the fastest measured route per interval
// (bench/ops/bf16_sweep.cu --n 4096 --k 2048, RTX PRO 6000, 2 % tolerance: 215 -> 16.4 us at
// T = 65, 4.88 -> 0.22 ms at T = 4096).
Bf16Launch select_bf16_n4096_k2048(std::int32_t tokens) {
    if (tokens <= 64) return select_bf16_general_launch(tokens);
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>>; // tma_r128t64k64s4
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>>; // tma_r128t128k64s3_w64x32
}

} // namespace infernix::ops::detail
