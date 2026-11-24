#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Glm5Next: the KDA control projection (forget down, gate down and beta in one 320-row group),
// T = a call's columns. T <= 64 keeps the general route's chunked skinny GEMV; wider calls take
// the fastest measured route per interval (bench/ops/bf16_sweep.cu --n 320 --k 4096, RTX PRO 6000,
// 2 % tolerance: 362 -> 10.3 us at T = 65, 956 -> 45 us at T = 4096).
Bf16Launch select_bf16_n320_k4096(std::int32_t tokens) {
    if (tokens <= 64) return select_bf16_general_launch(tokens);
    if (tokens <= 256) return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3>>; // tma_r64t128k64s3
}

} // namespace infernix::ops::detail
