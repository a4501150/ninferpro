// The Qwen3.5 family's tuned q6 linear shapes: one row per shape file in this directory. The
// table's external linkage (q6_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/q6/q6_shapes.h"

namespace infernix::ops::detail {

const Q6ShapeEntry qwen3_5_q6_shapes[] = {
    {248320, 5120, select_q6_n248320_k5120},
    {34816, 5120, select_q6_n34816_k5120},
    {248320, 2048, select_q6_n248320_k2048},
    {1152, 1536, select_q6_n1152_k1536},
};
const std::size_t qwen3_5_q6_shapes_count = sizeof(qwen3_5_q6_shapes) / sizeof(qwen3_5_q6_shapes[0]);

} // namespace infernix::ops::detail
