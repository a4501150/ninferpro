#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_format.h"
#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Scans each owning family's table in turn; the first (n, k) row wins.
const Fp8LinearShape* find_shape(std::int32_t n, std::int32_t k) {
    for (std::size_t i = 0; i < qwen3_5_fp8_shapes_count; ++i) {
        const Fp8LinearShape* shape = qwen3_5_fp8_shapes[i];
        if (shape->n == n && shape->k == k) return shape;
    }
    for (std::size_t i = 0; i < qwen4_exp_fp8_shapes_count; ++i) {
        const Fp8LinearShape* shape = qwen4_exp_fp8_shapes[i];
        if (shape->n == n && shape->k == k) return shape;
    }
    return nullptr;
}

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    const Fp8LinearShape* shape = find_shape(n, k);
    if (shape == nullptr) throw std::invalid_argument("fp8 linear: unsupported shape");
    return *shape;
}

// The single-term A8 route takes precedence where the policy permits it; the pair route only
// replaces an A16 route.
bool uses_pair(const Fp8LinearShape& shape, LinearPolicy policy, LinearRoutes routes,
               std::int32_t tokens) {
    return has_route(routes, LinearRoutes::Fp8A8Pair) && shape.pair != nullptr &&
               tokens >= shape.pair_min_tokens && !(allows_a8(policy) && shape.uses_a8(tokens, tokens));
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    if (column_invariant(policy) && shape.invariant == nullptr)
        throw std::invalid_argument("fp8 linear: the shape registers no column-invariant route");
    if (column_invariant(policy)) return 0; // one A16 schedule, no route flag applies
    const std::size_t single = allows_a8(policy) && shape.uses_a8(min_tokens, max_tokens)
                                   ? fp8_a8_workspace_capacity_bytes(
                                         max_tokens, k,
                                         shape.partial_capacity_bytes
                                             ? shape.partial_capacity_bytes(max_tokens)
                                             : 0)
                                   : 0;
    // The query does not see the route flags, so it also covers the pair route wherever the
    // interval reaches it.
    const std::size_t pair = shape.pair != nullptr && max_tokens >= shape.pair_min_tokens
                                 ? fp8_a8_pair_workspace_capacity_bytes(max_tokens, k)
                                 : 0;
    return std::max(single, pair);
}

bool fp8_column_invariant_supported(std::int32_t n, std::int32_t k) noexcept {
    const Fp8LinearShape* shape = find_shape(n, k);
    return shape != nullptr && shape->invariant != nullptr;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (column_invariant(policy)) {
        if (shape.invariant == nullptr)
            throw std::invalid_argument("fp8 linear: the shape registers no column-invariant route");
        return shape.invariant(x, weight, out, stream);
    }
    if (uses_pair(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr)
            throw std::invalid_argument("fp8 A8 pair linear requires caller workspace");
        auto scope         = workspace->scope();
        const auto scratch = allocate_fp8_a8_pair_workspace(*workspace, x.ne[1], weight.k);
        return shape.pair(x, weight, out, scratch, stream);
    }
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}
bool fp8_dispatch_rows(const Tensor& x, const Weight& weight, const LinearBf16RowParts& parts, LinearPolicy policy,
                       LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (column_invariant(policy) || shape.pair_rows == nullptr || !uses_pair(shape, policy, routes, x.ne[1]))
        return false;
    if (workspace == nullptr) throw std::invalid_argument("fp8 A8 pair linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_pair_workspace(*workspace, x.ne[1], weight.k);
    shape.pair_rows(x, weight, parts, scratch, stream);
    return true;
}
} // namespace infernix::ops::detail
