#pragma once

#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/common/shape_set.h"

namespace infernix::ops::detail {

struct Q4ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

[[nodiscard]] Q4Launch select_q4_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2048(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4304_k1152(std::int32_t tokens);

// The tuned shape tables, owned by the model families (src/models/<family>/shapes/) and listed
// by src/models/shapes.cpp. The dispatch references this registry, which forces the linker to
// extract the registry's translation unit and, through it, every table and shape file it
// references from the archive.
extern const LinearShapeSet<Q4ShapeEntry> q4_linear_shape_sets[];
extern const std::size_t q4_linear_shape_set_count;

} // namespace infernix::ops::detail
