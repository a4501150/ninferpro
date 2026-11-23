#pragma once

// The family's tuned linear shape tables (one <dtype>/<dtype>_shapes.cpp each), registered with
// the linear dispatch by src/models/shapes.cpp.

#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q6/q6_shapes.h"
#include "ops/linear/q8/q8_shapes.h"

#include <cstddef>

namespace infernix::models::qwen3_5 {

extern const ops::detail::Bf16ShapeEntry bf16_shapes[];
extern const std::size_t bf16_shapes_count;
extern const ops::detail::Fp8LinearShape* const fp8_shapes[];
extern const std::size_t fp8_shapes_count;
extern const ops::detail::Nvfp4LinearShape* const nvfp4_shapes[];
extern const std::size_t nvfp4_shapes_count;
extern const ops::detail::Q4ShapeEntry q4_shapes[];
extern const std::size_t q4_shapes_count;
extern const ops::detail::Q5ShapeEntry q5_shapes[];
extern const std::size_t q5_shapes_count;
extern const ops::detail::Q6ShapeEntry q6_shapes[];
extern const std::size_t q6_shapes_count;
extern const ops::detail::Q8ShapeEntry q8_shapes[];
extern const std::size_t q8_shapes_count;

} // namespace infernix::models::qwen3_5
