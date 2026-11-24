#pragma once

// The family's tuned linear shape tables (one <dtype>/<dtype>_shapes.cpp each), registered with
// the linear dispatch by src/models/shapes.cpp.

#include "ops/linear/bf16/bf16_shapes.h"

#include <cstddef>

namespace infernix::models::glm5_next {

extern const ops::detail::Bf16ShapeEntry bf16_shapes[];
extern const std::size_t bf16_shapes_count;

} // namespace infernix::models::glm5_next
