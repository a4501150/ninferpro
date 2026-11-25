// The Glm5Next (GLM-5.3-Flash) family's tuned Q8 linear shapes: one row per shape file in this
// directory. src/models/shapes.cpp registers this table with the dispatch, which forces the linker
// to extract this translation unit and, through its rows, every shape file below. The MLA query rank
// and latent parent [2048,4096] is not listed: Qwen3.5's table already registers that problem.
#include "models/glm5_next/shapes/shapes.h"

namespace infernix::models::glm5_next {

using namespace ops::detail;

const Q8ShapeEntry q8_shapes[] = {
    {24576, 4096, select_q8_n24576_k4096},  // KDA qkv; dense MLP gate and up
    {4096, 8192, select_q8_n4096_k8192},    // KDA output
    {16384, 1536, select_q8_n16384_k1536},  // MLA query up
    {4096, 16384, select_q8_n4096_k16384},  // MLA output
    {4096, 12288, select_q8_n4096_k12288},  // dense MLP down
    {4096, 4096, select_q8_n4096_k4096},    // shared expert gate and up
    {4096, 2048, select_q8_n4096_k2048},    // shared expert down
};
const std::size_t q8_shapes_count = sizeof(q8_shapes) / sizeof(q8_shapes[0]);

} // namespace infernix::models::glm5_next
