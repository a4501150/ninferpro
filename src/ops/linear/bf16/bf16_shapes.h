#pragma once

#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/common/shape_set.h"

namespace infernix::ops::detail {

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

// The tuned shape tables, owned by the model families (src/models/<family>/shapes/) and listed
// by src/models/shapes.cpp. The dispatch references this registry, which forces the linker to
// extract the registry's translation unit and, through it, every table and shape file it
// references from the archive.
extern const LinearShapeSet<Bf16ShapeEntry> bf16_linear_shape_sets[];
extern const std::size_t bf16_linear_shape_set_count;

// Runtime-shape GEMM fallback for any (n, k) the tables above do not specialise.
[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens);

} // namespace infernix::ops::detail
