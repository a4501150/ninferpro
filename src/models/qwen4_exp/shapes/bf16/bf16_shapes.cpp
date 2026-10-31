// The Qwen4Exp (Qwen3.8-Flash-Next, recipe A) family's tuned BF16 linear shapes: one row per
// shape file in this directory. The table's external linkage (bf16_shapes.h declares it extern;
// the dispatch references it) forces the linker to extract this translation unit and, through
// its rows, every shape file below.
#include "ops/linear/bf16/bf16_shapes.h"

namespace infernix::ops::detail {

const Bf16ShapeEntry qwen4_exp_bf16_shapes[] = {
    {48, 2560, select_bf16_n48_k2560},
    {96, 2560, select_bf16_n96_k2560, select_bf16_n96_k2560_invariant},
    {640, 2560, select_bf16_n640_k2560, select_bf16_n640_k2560_invariant,
     select_bf16_n640_k2560_index_mma},
    {1664, 2560, select_bf16_n1664_k2560},
    {324, 10240, select_bf16_n324_k10240},
    {320, 10240, select_bf16_n320_k10240},
    {10240, 320, select_bf16_n10240_k320},
    {12800, 2560, select_bf16_n12800_k2560},
    {248320, 2560, select_bf16_n248320_k2560},
    {13952, 2560, select_bf16_n13952_k2560},
    {2560, 6144, select_bf16_n2560_k6144},
    {2560, 2560, select_bf16_n2560_k2560},
    {16384, 2560, select_bf16_n16384_k2560},
    {1280, 2560, select_bf16_n1280_k2560},
    {2560, 640, select_bf16_n2560_k640},
    {2560, 4608, select_bf16_n2560_k4608},
};
const std::size_t qwen4_exp_bf16_shapes_count =
    sizeof(qwen4_exp_bf16_shapes) / sizeof(qwen4_exp_bf16_shapes[0]);

} // namespace infernix::ops::detail
