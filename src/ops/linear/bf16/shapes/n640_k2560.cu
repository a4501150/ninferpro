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
} // namespace infernix::ops::detail
