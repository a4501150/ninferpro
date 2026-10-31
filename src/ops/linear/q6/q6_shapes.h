#pragma once

#include "ops/linear/q6/q6_launch.h"

namespace infernix::ops::detail {

// Family shape tables: the model families own their tuned shapes (src/models/<family>/shapes/),
// and each family's table lists its whole set in one visible place. The tables have external
// linkage — the dispatch declares them extern, which forces the linker to extract the table
// translation unit and, through its rows, every shape file it references from the archive.
struct Q6ShapeEntry {
    std::int32_t n, k;
    Q6Launch (*select)(std::int32_t);
};

[[nodiscard]] Q6Launch select_q6_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2048(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n1152_k1536(std::int32_t tokens);

// The Qwen3.5 family's tuned shapes (src/models/qwen3_5/shapes/q6/q6_shapes.cpp).
extern const Q6ShapeEntry qwen3_5_q6_shapes[];
extern const std::size_t qwen3_5_q6_shapes_count;

} // namespace infernix::ops::detail
