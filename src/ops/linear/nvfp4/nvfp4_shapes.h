#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace infernix::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
};

// Family shape tables: the model families own their tuned shapes (src/models/<family>/shapes/),
// and each family's table lists its whole set in one visible place. The tables have external
// linkage — the dispatch declares them extern, which forces the linker to extract the table
// translation unit and, through its rows, every shape file it references from the archive.

// The Qwen3.5 family's tuned shapes (src/models/qwen3_5/shapes/nvfp4/nvfp4_shapes.cpp).
extern const Nvfp4LinearShape* const qwen3_5_nvfp4_shapes[];
extern const std::size_t qwen3_5_nvfp4_shapes_count;

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
} // namespace infernix::ops::detail
