#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/common/shape_set.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace infernix::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
};

// The tuned shape tables, owned by the model families (src/models/<family>/shapes/) and listed
// by src/models/shapes.cpp. The dispatch references this registry, which forces the linker to
// extract the registry's translation unit and, through it, every table and shape file it
// references from the archive.
extern const LinearShapeSet<const Nvfp4LinearShape*> nvfp4_linear_shape_sets[];
extern const std::size_t nvfp4_linear_shape_set_count;

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
} // namespace infernix::ops::detail
