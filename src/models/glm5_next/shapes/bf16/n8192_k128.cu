#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Glm5Next: the KDA forget and gate up projections (8192 x 128 each), T = a call's columns.
// T <= 64 keeps the general route's chunked skinny GEMV; wider calls take the fastest measured
// route per interval (bench/ops/bf16_sweep.cu --n 8192 --k 128, RTX PRO 6000, 2 % tolerance:
// 26.7 -> 6.1 us at T = 65, 725 -> 51 us at T = 4096).
Bf16Launch select_bf16_n8192_k128(std::int32_t tokens) {
    if (tokens <= 64) return select_bf16_general_launch(tokens);
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>>; // tma_r64t128k64s2_rows
}

} // namespace infernix::ops::detail
