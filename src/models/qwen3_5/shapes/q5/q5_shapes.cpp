// The Qwen3.5 family's tuned q5 linear shapes: one row per shape file in this directory. The
// table's external linkage (q5_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/q5/q5_shapes.h"

namespace infernix::ops::detail {

const Q5ShapeEntry qwen3_5_q5_shapes[] = {
    {1024, 5120, select_q5_n1024_k5120},
    {6144, 5120, select_q5_n6144_k5120},
    {7168, 5120, select_q5_n7168_k5120},
    {5120, 6144, select_q5_n5120_k6144},
    {5120, 17408, select_q5_n5120_k17408},
    {1152, 1152, select_q5_n1152_k1152},
    {1152, 4304, select_q5_n1152_k4304},
};
const std::size_t qwen3_5_q5_shapes_count = sizeof(qwen3_5_q5_shapes) / sizeof(qwen3_5_q5_shapes[0]);

} // namespace infernix::ops::detail
