#include "ops/linear/q8/q8_instance_launch.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {
namespace {
using Geometry = Q8N2560K12800;

// The DFlash2 feature projection's sliced-K schedules: 160 CTAs of one 16-row tile stream the 35 MB
// weight, so each double-buffers its stages (48- and 64-column tiles with 8 K warps exceed the 99 KB
// shared-memory budget, so 4 K warps serve the wider tiles). RTX PRO 6000 sweep of 2026-11-03, cold,
// graph-timed, min/median of 60 (production serving concurrently): at 8 columns 24.3 us against the
// runtime-shape SIMT route's 46.8, at 16 26.4 against 129, at 48 48.9 against 129, at 64 63.5
// against 124.9, at 96 86.0 against 133 (single stage: the double buffer stops paying once the
// kernel loops token blocks), at 128 110.4; the r32_t64 MMA tile takes 256 columns (137 against the
// generic r32_t128's 139 at 256), r32_t96 384 (163.6 against 169.7) and r64_t128 beyond (176.1/286.7/
// 569.3 at 512/1024/2048 against 190.5/376.4/720.9).
template <int Columns, int KWarps, int Stages = 2>
using Sliced = Q8A16SlicedKMmaSchedule<Columns, KWarps, Stages, KWarps == 8 ? 2 : 3, Q8ScaleAccess::Shared,
                                       Cache::ca, Cache::cg, Q8ActivationStage::ActiveOnly>;
} // namespace

// The exact route keeps the runtime-shape routes this problem ran before its table entry (the
// predicated SIMT pair to 8 columns, the r32 MMA tiles beyond): identical launches at identical
// token counts, so the exact route's bits are unchanged.
Q8Launch select_q8_n2560_k12800(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

// The q8-sliced-k fast route: the sliced-K tiles of the sweep above. A column's bits depend on the
// call's width (8 K warps through 32 columns, 4 beyond; the stages seam at 64), so the fast path is
// width-keyed like the flag's other problems; up to 8 columns the single 8-column tile holds a
// column's bits independent of how many share the call.
Q8Launch select_q8_n2560_k12800_fast(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, Sliced<8, 8>>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, Sliced<16, 8>>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, Sliced<24, 8>>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, Sliced<32, 8>>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, Sliced<40, 4>>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, Sliced<48, 4>>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 64, Sliced<64, 4>>;
    if (tokens <= 128) return launch_q8_a16_sliced<Geometry, 64, Sliced<64, 4, 1>>;
    if (tokens <= 256) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 384) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
