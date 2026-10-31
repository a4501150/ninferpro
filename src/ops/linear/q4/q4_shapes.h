#pragma once

#include "ops/linear/q4/q4_launch.h"

namespace infernix::ops::detail {

// Family shape tables: the model families own their tuned shapes (src/models/<family>/shapes/),
// and each family's table lists its whole set in one visible place. The tables have external
// linkage — the dispatch declares them extern, which forces the linker to extract the table
// translation unit and, through its rows, every shape file it references from the archive.
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

// The Qwen3.5 family's tuned shapes (src/models/qwen3_5/shapes/q4/q4_shapes.cpp).
extern const Q4ShapeEntry qwen3_5_q4_shapes[];
extern const std::size_t qwen3_5_q4_shapes_count;

} // namespace infernix::ops::detail
