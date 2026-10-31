#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace infernix::ops::detail {

// Family shape tables: the model families own their tuned shapes (src/models/<family>/shapes/),
// and each family's table lists its whole set in one visible place. The tables have external
// linkage — the dispatch declares them extern, which forces the linker to extract the table
// translation unit and, through its rows, every shape file it references from the archive.
struct Bf16ShapeEntry {
    std::int32_t n, k;
    Bf16Launch (*select)(std::int32_t);
    Bf16Launch (*invariant)(std::int32_t) = nullptr; // A16ColumnInvariant, or null where none is registered
    Bf16Launch (*index_mma)(std::int32_t) = nullptr; // LinearRoutes::Bf16IndexMma, or null
};

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n48_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n96_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1664_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4608_k4608(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k4608(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n13952_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n324_k10240(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n320_k10240(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n10240_k320(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n12800_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens);
// Qwen3.8-Flash-Next projections NVIDIA stores BF16 (the bit-exact artifact): GDN q/k/v/z and the
// shared expert.
[[nodiscard]] Bf16Launch select_bf16_n16384_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1280_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k640(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n640_k2560(std::int32_t tokens);
// A16ColumnInvariant routes (one reduction for every T).
[[nodiscard]] Bf16Launch select_bf16_n96_k2560_invariant(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n640_k2560_invariant(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n640_k2560_index_mma(std::int32_t tokens);

// The Qwen3.5 family's tuned shapes (src/models/qwen3_5/shapes/bf16/bf16_shapes.cpp).
extern const Bf16ShapeEntry qwen3_5_bf16_shapes[];
extern const std::size_t qwen3_5_bf16_shapes_count;

// The Qwen4Exp (Qwen3.8-Flash-Next) family's tuned shapes
// (src/models/qwen4_exp/shapes/bf16/bf16_shapes.cpp).
extern const Bf16ShapeEntry qwen4_exp_bf16_shapes[];
extern const std::size_t qwen4_exp_bf16_shapes_count;

// Runtime-shape GEMM fallback for any (n, k) the tables above do not specialise.
[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens);

} // namespace infernix::ops::detail
