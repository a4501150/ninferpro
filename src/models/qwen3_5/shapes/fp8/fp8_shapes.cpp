// The Qwen3.5 family's tuned FP8 linear shapes: one row per shape file in this directory.
// src/models/shapes.cpp registers this table with the dispatch, which forces the linker to extract
// this translation unit and, through its rows, every shape file below.
#include "models/qwen3_5/shapes/shapes.h"

namespace infernix::models::qwen3_5 {

using namespace ops::detail;

const Fp8LinearShape* const fp8_shapes[] = {
    &kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
    &kFp8N5120K6144,  &kFp8N5120K17408,  &kFp8N248320K5120,
};
const std::size_t fp8_shapes_count = sizeof(fp8_shapes) / sizeof(fp8_shapes[0]);

} // namespace infernix::models::qwen3_5
