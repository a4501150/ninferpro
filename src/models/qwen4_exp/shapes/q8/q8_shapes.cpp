// The Qwen4Exp (Qwen3.8-Flash-Next, recipe B) family's tuned q8 linear shapes: one row per
// shape file in this directory. The table's external linkage (q8_shapes.h declares it extern;
// the dispatch references it) forces the linker to extract this translation unit and, through
// its rows, every shape file below.
#include "ops/linear/q8/q8_shapes.h"

namespace infernix::ops::detail {

const Q8ShapeEntry qwen4_exp_q8_shapes[] = {
    {16384, 2560, select_q8_n16384_k2560, select_q8_n16384_k2560_fast},
    {12800, 2560, select_q8_n12800_k2560, nullptr, select_q8_n12800_k2560_invariant},
    {13952, 2560, select_q8_n13952_k2560, nullptr, select_q8_n13952_k2560_invariant},
    {2560, 2560, select_q8_n2560_k2560, nullptr, select_q8_n2560_k2560_invariant},
    {1280, 2560, select_q8_n1280_k2560, nullptr, select_q8_n1280_k2560_invariant},
    {2560, 6144, select_q8_n2560_k6144, select_q8_n2560_k6144_fast,
     select_q8_n2560_k6144_invariant},
    {324, 10240, select_q8_n324_k10240, nullptr, select_q8_n324_k10240_invariant,
     select_q8_n324_k10240_fast},
    {320, 10240, select_q8_n320_k10240, nullptr, select_q8_n320_k10240_invariant,
     select_q8_n320_k10240_fast},
    {10240, 320, select_q8_n10240_k320, nullptr, select_q8_n10240_k320_invariant},
    {2560, 640, select_q8_n2560_k640, nullptr, select_q8_n2560_k640_invariant},
    // NInfer's tunings for its own 2560-wide groupings.
    {6144, 2560, select_q8_n6144_k2560},
    {10240, 2560, select_q8_n10240_k2560},
    {12288, 2560, select_q8_n12288_k2560},
};
const std::size_t qwen4_exp_q8_shapes_count =
    sizeof(qwen4_exp_q8_shapes) / sizeof(qwen4_exp_q8_shapes[0]);

} // namespace infernix::ops::detail
