#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Glm5Next: the DSA indexer's input projection (index key, gate and head weights, 288 rows in one
// group), T = a call's columns. T <= 64 keeps the general route's chunked skinny GEMV; wider calls
// take the fastest measured route per interval (bench/ops/bf16_sweep.cu --n 288 --k 4096, RTX PRO
// 6000, 2 % tolerance: 362.5 -> 10.2 us at T = 65, 952 -> 43 us at T = 4096). 288 rows fit no
// 64-row tile, so the tensor-core routes are the 32-row sliced-K schedule and the predicated tails.
Bf16Launch select_bf16_n288_k4096(std::int32_t tokens) {
    if (tokens <= 64) return select_bf16_general_launch(tokens);
    if (tokens <= 256) return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>>; // tail_r64t32s3
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>>; // tail_r64t64s3
    return launch_bf16_tma_mma<Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tail_r64t128s3_rows
}

} // namespace infernix::ops::detail
