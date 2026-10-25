#pragma once

// Decode-width schedules of the BF16 shapes that ops::linear_group also composes: the shape's own
// route and the composed launch take the same type, so they produce the same bits.

#include "ops/linear/bf16/bf16_schedule.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next GDN a/b [96, 2560] at T = 1 (NInfer 81c8ce09's route).
using Bf16N96K2560Gemv = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>;
// ... and at T = 2 .. 8.
using Bf16N96K2560Simt = Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked, Bf16WeightCache::Default,
                                             Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>;
// The skinny GEMV instance of decode-sized BF16 calls of more than 256 rows (bf16_general.cu), such
// as the QSA index projection [640, 2560]; every skinny instance has the same bits.
inline constexpr int kBf16SkinnyWarps       = 8;
inline constexpr int kBf16SkinnyRowsPerWarp = 2;
} // namespace infernix::ops::detail
