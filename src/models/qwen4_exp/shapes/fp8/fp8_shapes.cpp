// The Qwen4Exp (Qwen3.8-Flash-Next, recipe C) family's tuned FP8 linear shapes: one row per shape
// file in this directory. src/models/shapes.cpp registers this table with the dispatch, which
// forces the linker to extract this translation unit and, through its rows, every shape file below.
#include "models/qwen4_exp/shapes/shapes.h"

namespace infernix::models::qwen4_exp {

using namespace ops::detail;

const Fp8LinearShape* const fp8_shapes[] = {
    &kFp8N16384K2560, &kFp8N13312K2560, &kFp8N2560K6144,
    &kFp8N1280K2560,  &kFp8N2560K640,
};
const std::size_t fp8_shapes_count = sizeof(fp8_shapes) / sizeof(fp8_shapes[0]);

} // namespace infernix::models::qwen4_exp
