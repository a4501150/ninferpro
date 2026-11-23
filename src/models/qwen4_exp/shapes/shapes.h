#pragma once

// The family's tuned linear shape tables (one <dtype>/<dtype>_shapes.cpp each), registered with
// the linear dispatch by src/models/shapes.cpp.

#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/q8/q8_shapes.h"

#include <cstddef>

namespace infernix::models::qwen4_exp {

extern const ops::detail::Bf16ShapeEntry bf16_shapes[];
extern const std::size_t bf16_shapes_count;
extern const ops::detail::Fp8LinearShape* const fp8_shapes[];
extern const std::size_t fp8_shapes_count;
extern const ops::detail::Q8ShapeEntry q8_shapes[];
extern const std::size_t q8_shapes_count;

} // namespace infernix::models::qwen4_exp
