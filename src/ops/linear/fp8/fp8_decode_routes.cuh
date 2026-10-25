#pragma once

// Decode-width schedules of the FP8 shapes that ops::linear_group also composes: the shape's own
// route and the composed launch take the same type, so they produce the same bits.

#include "ops/linear/fp8/fp8_schedule.cuh"

namespace infernix::ops::detail {
// Qwen3.8-Flash-Next GDN query/key/value/z [16384, 2560] at T = 1.
using Fp8N16384K2560Gemv = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
// ... and at T = 2 .. 8 (the SIMT tile of capacity 8).
using Fp8N16384K2560Simt = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming,
                                              1, Fp8SimtBlockOrder::RowsContiguous, 1>;
// Qwen3.8-Flash-Next attention query/gate/key/value [13312, 2560] at T = 1 ...
using Fp8N13312K2560Gemv = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
// ... and at T = 2 .. 8.
using Fp8N13312K2560Simt = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default,
                                              1, Fp8SimtBlockOrder::RowsContiguous, 1>;
} // namespace infernix::ops::detail
