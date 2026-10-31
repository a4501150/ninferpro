// The Qwen3.5 family's tuned BF16 linear shapes: one row per shape file in this directory. The
// table's external linkage (bf16_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/bf16/bf16_shapes.h"

namespace infernix::ops::detail {

const Bf16ShapeEntry qwen3_5_bf16_shapes[] = {
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
const std::size_t qwen3_5_bf16_shapes_count =
    sizeof(qwen3_5_bf16_shapes) / sizeof(qwen3_5_bf16_shapes[0]);

} // namespace infernix::ops::detail
