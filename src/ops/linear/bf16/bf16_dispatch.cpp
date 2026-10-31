#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Scans each owning family's table in turn; the first (n, k) row wins.
const Bf16ShapeEntry* find_shape(std::int32_t n, std::int32_t k) {
    for (std::size_t i = 0; i < qwen3_5_bf16_shapes_count; ++i) {
        const Bf16ShapeEntry& entry = qwen3_5_bf16_shapes[i];
        if (entry.n == n && entry.k == k) return &entry;
    }
    for (std::size_t i = 0; i < qwen4_exp_bf16_shapes_count; ++i) {
        const Bf16ShapeEntry& entry = qwen4_exp_bf16_shapes[i];
        if (entry.n == n && entry.k == k) return &entry;
    }
    return nullptr;
}
} // namespace

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearRoutes routes) {
    if (t <= 0) throw std::invalid_argument("bf16 linear: T must be positive");
    const Bf16ShapeEntry* entry = find_shape(n, k);
    if (entry != nullptr) {
        if (has_route(routes, LinearRoutes::Bf16IndexMma) && entry->index_mma != nullptr)
            return entry->index_mma(t);
        return entry->select(t);
    }
    // Shapes without a specialised kernel (for example a full-precision vocab head) fall back to the
    // general runtime-shape GEMM.
    return select_bf16_general_launch(t);
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy,
                              LinearRoutes routes) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("bf16 linear: unsupported policy");
    if (column_invariant(policy)) {
        if (t <= 0) throw std::invalid_argument("bf16 linear: T must be positive");
        const Bf16ShapeEntry* entry = find_shape(n, k);
        if (entry != nullptr && entry->invariant != nullptr) return entry->invariant(t);
        throw std::invalid_argument("bf16 linear: no column-invariant route for this problem");
    }
    return select_bf16_a16_launch(n, k, t, routes);
}

bool bf16_column_invariant_supported(std::int32_t n, std::int32_t k) noexcept {
    const Bf16ShapeEntry* entry = find_shape(n, k);
    return entry != nullptr && entry->invariant != nullptr;
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   LinearRoutes routes, cudaStream_t stream) {
    select_bf16_launch(weight.n, weight.k, x.ne[1], policy, routes)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
