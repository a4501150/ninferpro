// The Qwen4Exp (Qwen3.8-Flash-Next, recipe C) family's tuned FP8 linear shapes: one row per
// shape file in this directory. The table's external linkage (fp8_shapes.h declares it extern;
// the dispatch references it) forces the linker to extract this translation unit and, through
// its rows, every shape file below.
#include "ops/linear/fp8/fp8_shapes.h"

namespace infernix::ops::detail {

const Fp8LinearShape* const qwen4_exp_fp8_shapes[] = {
    &kFp8N16384K2560, &kFp8N13312K2560, &kFp8N2560K6144,
    &kFp8N1280K2560,  &kFp8N2560K640,
};
const std::size_t qwen4_exp_fp8_shapes_count =
    sizeof(qwen4_exp_fp8_shapes) / sizeof(qwen4_exp_fp8_shapes[0]);

} // namespace infernix::ops::detail
