// The Qwen3.5 family's tuned q8 linear shapes: one row per shape file in this directory. The
// table's external linkage (q8_shapes.h declares it extern; the dispatch references it) forces
// the linker to extract this translation unit and, through its rows, every shape file below.
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {

const Q8ShapeEntry qwen3_5_q8_shapes[] = {
    {1024, 2048, select_q8_n1024_k2048},
    {1024, 5120, select_q8_n1024_k5120},
    {2048, 4096, select_q8_n2048_k4096},
    {2048, 4608, select_q8_n2048_k4608},
    {2048, 16384, select_q8_n2048_k16384},
    {4608, 4608, select_q8_n4608_k4608},
    {5120, 4608, select_q8_n5120_k4608},
    {5120, 6144, select_q8_n5120_k6144},
    {5120, 10240, select_q8_n5120_k10240},
    {5120, 17408, select_q8_n5120_k17408},
    {5120, 25600, select_q8_n5120_k25600},
    {6144, 5120, select_q8_n6144_k5120},
    {9216, 2048, select_q8_n9216_k2048},
    {12288, 2048, select_q8_n12288_k2048},
    {14336, 5120, select_q8_n14336_k5120},
    {17408, 5120, select_q8_n17408_k5120},
    {34816, 5120, select_q8_n34816_k5120},
    {248320, 5120, select_q8_n248320_k5120, nullptr, select_q8_n248320_k5120_invariant},
};
const std::size_t qwen3_5_q8_shapes_count =
    sizeof(qwen3_5_q8_shapes) / sizeof(qwen3_5_q8_shapes[0]);

} // namespace infernix::ops::detail
