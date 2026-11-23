// The Qwen3.5 family's tuned q6 linear shapes: one row per shape file in this directory.
// src/models/shapes.cpp registers this table with the dispatch, which forces the linker to extract
// this translation unit and, through its rows, every shape file below.
#include "models/qwen3_5/shapes/shapes.h"

namespace infernix::models::qwen3_5 {

using namespace ops::detail;

const Q6ShapeEntry q6_shapes[] = {
    {248320, 5120, select_q6_n248320_k5120},
    {34816, 5120, select_q6_n34816_k5120},
    {248320, 2048, select_q6_n248320_k2048},
    {1152, 1536, select_q6_n1152_k1536},
};
const std::size_t q6_shapes_count = sizeof(q6_shapes) / sizeof(q6_shapes[0]);

} // namespace infernix::models::qwen3_5
