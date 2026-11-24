// The Glm5Next (GLM-5.3-Flash) family's tuned BF16 linear shapes: one row per shape file in this
// directory. src/models/shapes.cpp registers this table with the dispatch, which forces the linker
// to extract this translation unit and, through its rows, every shape file below.
#include "models/glm5_next/shapes/shapes.h"

namespace infernix::models::glm5_next {

using namespace ops::detail;

const Bf16ShapeEntry bf16_shapes[] = {
    {24576, 4096, select_bf16_n24576_k4096},  // KDA qkv; dense MLP gate and up
    {320, 4096, select_bf16_n320_k4096},      // KDA control (forget down, gate down, beta)
    {8192, 128, select_bf16_n8192_k128},      // KDA forget up; gate up
    {4096, 8192, select_bf16_n4096_k8192},    // KDA output
    {2048, 4096, select_bf16_n2048_k4096},    // MLA query rank and latent
    {288, 4096, select_bf16_n288_k4096},      // MLA indexer inputs (key, gate, head weights)
    {4096, 1536, select_bf16_n4096_k1536},    // MLA indexer query
    {16384, 1536, select_bf16_n16384_k1536},  // MLA query up
    {4096, 16384, select_bf16_n4096_k16384},  // MLA output
    {4096, 12288, select_bf16_n4096_k12288},  // dense MLP down
    {4096, 4096, select_bf16_n4096_k4096},    // shared expert gate and up
    {4096, 2048, select_bf16_n4096_k2048},    // shared expert down
};
const std::size_t bf16_shapes_count = sizeof(bf16_shapes) / sizeof(bf16_shapes[0]);

} // namespace infernix::models::glm5_next
