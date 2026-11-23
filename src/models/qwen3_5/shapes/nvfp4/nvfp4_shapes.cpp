// The Qwen3.5 family's tuned NVFP4 linear shapes: one row per shape file in this directory.
// src/models/shapes.cpp registers this table with the dispatch, which forces the linker to extract
// this translation unit and, through its rows, every shape file below.
#include "models/qwen3_5/shapes/shapes.h"

namespace infernix::models::qwen3_5 {

using namespace ops::detail;

const Nvfp4LinearShape* const nvfp4_shapes[] = {
    &kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
    &kNvfp4N5120K6144,  &kNvfp4N5120K17408,
};
const std::size_t nvfp4_shapes_count = sizeof(nvfp4_shapes) / sizeof(nvfp4_shapes[0]);

} // namespace infernix::models::qwen3_5
