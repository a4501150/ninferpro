// The Qwen3.5 family's tuned q4 linear shapes: one row per shape file in this directory. The
// table's external linkage (q4_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/q4/q4_shapes.h"

namespace infernix::ops::detail {

const Q4ShapeEntry qwen3_5_q4_shapes[] = {
    {1024, 5120, select_q4_n1024_k5120},
    {4096, 5120, select_q4_n4096_k5120},
    {5120, 6144, select_q4_n5120_k6144},
    {6144, 5120, select_q4_n6144_k5120},
    {7168, 5120, select_q4_n7168_k5120},
    {34816, 5120, select_q4_n34816_k5120},
    {131072, 5120, select_q4_n131072_k5120},
    {131072, 2048, select_q4_n131072_k2048},
    {3456, 1152, select_q4_n3456_k1152},
    {4304, 1152, select_q4_n4304_k1152},
};
const std::size_t qwen3_5_q4_shapes_count = sizeof(qwen3_5_q4_shapes) / sizeof(qwen3_5_q4_shapes[0]);

} // namespace infernix::ops::detail
