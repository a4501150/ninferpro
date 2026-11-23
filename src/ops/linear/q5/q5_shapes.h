#pragma once

#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/common/shape_set.h"

namespace infernix::ops::detail {

struct Q5ShapeEntry {
    std::int32_t n, k;
    Q5Launch (*select)(std::int32_t);
};

[[nodiscard]] Q5Launch select_q5_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1152_k1152(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1152_k4304(std::int32_t tokens);

// The tuned shape tables, owned by the model families (src/models/<family>/shapes/) and listed
// by src/models/shapes.cpp. The dispatch references this registry, which forces the linker to
// extract the registry's translation unit and, through it, every table and shape file it
// references from the archive.
extern const LinearShapeSet<Q5ShapeEntry> q5_linear_shape_sets[];
extern const std::size_t q5_linear_shape_set_count;

} // namespace infernix::ops::detail
