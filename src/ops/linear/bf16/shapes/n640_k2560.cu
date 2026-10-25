#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next: the QSA index projection (both recipes), T = a call's columns. The runtime-shape
// GEMM serves it as before this entry.
Bf16Launch select_bf16_n640_k2560(std::int32_t tokens) { return select_bf16_general_launch(tokens); }

// A16ColumnInvariant: one sliced-K family at every T, as the GDN a/b projection's (n96_k2560.cu).
Bf16Launch select_bf16_n640_k2560_invariant(std::int32_t tokens) {
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 2560>>;
    return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>, 2560>>;
}
// LinearRoutes::Bf16IndexMma: the BF16 tensor-core tiles from 9 columns, 4-7x faster than the
// runtime-shape GEMM (bench/ops/bf16_flash_next_sweep.cu --shape qsaidx, RTX PRO 6000, 2026-10-25:
// 8 against 33-74 us at 32-65 columns, 18 against 84 at 1,024, 178 against 926 at 16,384).
Bf16Launch select_bf16_n640_k2560_index_mma(std::int32_t tokens) {
    if (tokens <= 8) return select_bf16_general_launch(tokens);
    if (tokens <= 128) return launch_bf16_sliced_k_mma<Bf16A16SlicedR32T16W4>; // sliced_r32t16w4
    if (tokens <= 256) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>>; // tma_r64t32k64s3
    if (tokens <= 1024) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>>; // tma_r64t64k64s3
    if (tokens <= 2048) return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3>>; // tma_r64t128k64s3
    return launch_bf16_tma_mma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>; // tma_r128t128k64s3_w64x32_rows
}
} // namespace infernix::ops::detail
