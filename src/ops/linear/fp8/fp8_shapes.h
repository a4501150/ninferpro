#pragma once
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/common/row_parts.h"

namespace infernix::ops::detail {
struct Fp8LinearShape {
    std::int32_t n, k;
    Fp8Launch a16;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t);
    bool (*uses_a8)(std::int32_t min_tokens, std::int32_t max_tokens);
    std::size_t (*partial_capacity_bytes)(std::int32_t max_tokens) = nullptr;
    // A16ColumnInvariant: one schedule for every T; null where the shape registers none.
    Fp8Launch invariant = nullptr;
    // fp8-a8-pair (LinearRoutes::Fp8A8Pair): the two-term E4M3 activation route, selected at
    // T >= pair_min_tokens when the flag is set and no single-term A8 route is selected; null
    // where the problem registers none.
    void (*pair)(const Tensor&, const Weight&, Tensor&, Fp8A8PairWorkspace, cudaStream_t) = nullptr;
    std::int32_t pair_min_tokens = 0;
    // The pair route writing row ranges to separate tensors (linear_rows); null where callers split
    // the problem's output with a copy.
    void (*pair_rows)(const Tensor&, const Weight&, const LinearBf16RowParts&, Fp8A8PairWorkspace,
                      cudaStream_t) = nullptr;
};

// Family shape tables: the model families own their tuned shapes (src/models/<family>/shapes/),
// and each family's table lists its whole set in one visible place. The tables have external
// linkage — the dispatch declares them extern, which forces the linker to extract the table
// translation unit and, through its rows, every shape file it references from the archive.

// The Qwen3.5 family's tuned shapes (src/models/qwen3_5/shapes/fp8/fp8_shapes.cpp).
extern const Fp8LinearShape* const qwen3_5_fp8_shapes[];
extern const std::size_t qwen3_5_fp8_shapes_count;

// The Qwen4Exp (Qwen3.8-Flash-Next) family's tuned shapes
// (src/models/qwen4_exp/shapes/fp8/fp8_shapes.cpp).
extern const Fp8LinearShape* const qwen4_exp_fp8_shapes[];
extern const std::size_t qwen4_exp_fp8_shapes_count;

extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
// Qwen3.8-Flash-Next (recipe C, W8A16: A16 routes, plus the fp8-a8-pair route).
extern const Fp8LinearShape kFp8N16384K2560;
extern const Fp8LinearShape kFp8N13312K2560;
extern const Fp8LinearShape kFp8N2560K6144;
extern const Fp8LinearShape kFp8N1280K2560;
extern const Fp8LinearShape kFp8N2560K640;
} // namespace infernix::ops::detail
