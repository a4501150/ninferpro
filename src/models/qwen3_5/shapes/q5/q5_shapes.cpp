// The Qwen3.5 family's tuned q5 linear shapes: one row per shape file in this directory.
// src/models/shapes.cpp registers this table with the dispatch, which forces the linker to extract
// this translation unit and, through its rows, every shape file below.
#include "models/qwen3_5/shapes/shapes.h"

namespace infernix::models::qwen3_5 {

using namespace ops::detail;

const Q5ShapeEntry q5_shapes[] = {
    {1024, 5120, select_q5_n1024_k5120},
    {6144, 5120, select_q5_n6144_k5120},
    {7168, 5120, select_q5_n7168_k5120},
    {5120, 6144, select_q5_n5120_k6144},
    {5120, 17408, select_q5_n5120_k17408},
    {1152, 1152, select_q5_n1152_k1152},
    {1152, 4304, select_q5_n1152_k4304},
};
const std::size_t q5_shapes_count = sizeof(q5_shapes) / sizeof(q5_shapes[0]);

} // namespace infernix::models::qwen3_5
