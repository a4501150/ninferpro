#pragma once

#include "ops/linear/q5/q5_launch.h"

namespace infernix::ops::detail {

// Family shape tables: the model families own their tuned shapes (src/models/<family>/shapes/),
// and each family's table lists its whole set in one visible place. The tables have external
// linkage — the dispatch declares them extern, which forces the linker to extract the table
// translation unit and, through its rows, every shape file it references from the archive.
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

// The Qwen3.5 family's tuned shapes (src/models/qwen3_5/shapes/q5/q5_shapes.cpp).
extern const Q5ShapeEntry qwen3_5_q5_shapes[];
extern const std::size_t qwen3_5_q5_shapes_count;

} // namespace infernix::ops::detail
