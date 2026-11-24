#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Glm5Next: the KDA output projection, T = a call's columns. T <= 64 keeps the general route's
// chunked skinny GEMV; wider calls take the fastest measured route per interval
// (bench/ops/bf16_sweep.cu --n 4096 --k 8192, RTX PRO 6000, 2 % tolerance: 843.8 -> 49.2 us at
// T = 65, 19.29 -> 0.79 ms at T = 4096).
Bf16Launch select_bf16_n4096_k8192(std::int32_t tokens) {
    if (tokens <= 64) return select_bf16_general_launch(tokens);
    if (tokens <= 128) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4>>; // tma_r128t64k64s4
    return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<128, 128, 32, 64, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r128t128s3_w32x64_rows
}

} // namespace infernix::ops::detail
