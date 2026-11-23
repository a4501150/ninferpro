// Registers every model family's tuned linear shape tables with the linear dispatch. The
// dispatch references these registries, so linking it extracts this translation unit and, through
// it, each family's tables and shape files from the archive.
#include "models/qwen3_5/shapes/shapes.h"
#include "models/qwen4_exp/shapes/shapes.h"

#include <iterator>

namespace infernix::ops::detail {

const LinearShapeSet<Bf16ShapeEntry> bf16_linear_shape_sets[] = {
    {models::qwen3_5::bf16_shapes, &models::qwen3_5::bf16_shapes_count},
    {models::qwen4_exp::bf16_shapes, &models::qwen4_exp::bf16_shapes_count},
};
const std::size_t bf16_linear_shape_set_count = std::size(bf16_linear_shape_sets);

const LinearShapeSet<const Fp8LinearShape*> fp8_linear_shape_sets[] = {
    {models::qwen3_5::fp8_shapes, &models::qwen3_5::fp8_shapes_count},
    {models::qwen4_exp::fp8_shapes, &models::qwen4_exp::fp8_shapes_count},
};
const std::size_t fp8_linear_shape_set_count = std::size(fp8_linear_shape_sets);

const LinearShapeSet<const Nvfp4LinearShape*> nvfp4_linear_shape_sets[] = {
    {models::qwen3_5::nvfp4_shapes, &models::qwen3_5::nvfp4_shapes_count},
};
const std::size_t nvfp4_linear_shape_set_count = std::size(nvfp4_linear_shape_sets);

const LinearShapeSet<Q4ShapeEntry> q4_linear_shape_sets[] = {
    {models::qwen3_5::q4_shapes, &models::qwen3_5::q4_shapes_count},
};
const std::size_t q4_linear_shape_set_count = std::size(q4_linear_shape_sets);

const LinearShapeSet<Q5ShapeEntry> q5_linear_shape_sets[] = {
    {models::qwen3_5::q5_shapes, &models::qwen3_5::q5_shapes_count},
};
const std::size_t q5_linear_shape_set_count = std::size(q5_linear_shape_sets);

const LinearShapeSet<Q6ShapeEntry> q6_linear_shape_sets[] = {
    {models::qwen3_5::q6_shapes, &models::qwen3_5::q6_shapes_count},
};
const std::size_t q6_linear_shape_set_count = std::size(q6_linear_shape_sets);

const LinearShapeSet<Q8ShapeEntry> q8_linear_shape_sets[] = {
    {models::qwen3_5::q8_shapes, &models::qwen3_5::q8_shapes_count},
    {models::qwen4_exp::q8_shapes, &models::qwen4_exp::q8_shapes_count},
};
const std::size_t q8_linear_shape_set_count = std::size(q8_linear_shape_sets);

} // namespace infernix::ops::detail
