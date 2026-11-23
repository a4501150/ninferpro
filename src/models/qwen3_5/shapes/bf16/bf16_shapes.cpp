// The Qwen3.5 family's tuned BF16 linear shapes: one row per shape file in this directory.
// src/models/shapes.cpp registers this table with the dispatch, which forces the linker to extract
// this translation unit and, through its rows, every shape file below.
#include "models/qwen3_5/shapes/shapes.h"

namespace infernix::models::qwen3_5 {

using namespace ops::detail;

const Bf16ShapeEntry bf16_shapes[] = {
    {14336, 5120, select_bf16_n14336_k5120},
    {5120, 6144, select_bf16_n5120_k6144},
    {1152, 1536, select_bf16_n1152_k1536},
    {3456, 1152, select_bf16_n3456_k1152},
    {1152, 1152, select_bf16_n1152_k1152},
    {4304, 1152, select_bf16_n4304_k1152},
    {1152, 4304, select_bf16_n1152_k4304},
    {4608, 4608, select_bf16_n4608_k4608},
    {256, 5120, select_bf16_n256_k5120},
    {5120, 4608, select_bf16_n5120_k4608},
};
const std::size_t bf16_shapes_count = sizeof(bf16_shapes) / sizeof(bf16_shapes[0]);

} // namespace infernix::models::qwen3_5
