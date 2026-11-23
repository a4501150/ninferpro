#pragma once

#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/common/shape_set.h"

namespace infernix::ops::detail {

struct Q6ShapeEntry {
    std::int32_t n, k;
    Q6Launch (*select)(std::int32_t);
};

[[nodiscard]] Q6Launch select_q6_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2048(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n1152_k1536(std::int32_t tokens);

// The tuned shape tables, owned by the model families (src/models/<family>/shapes/) and listed
// by src/models/shapes.cpp. The dispatch references this registry, which forces the linker to
// extract the registry's translation unit and, through it, every table and shape file it
// references from the archive.
extern const LinearShapeSet<Q6ShapeEntry> q6_linear_shape_sets[];
extern const std::size_t q6_linear_shape_set_count;

} // namespace infernix::ops::detail
