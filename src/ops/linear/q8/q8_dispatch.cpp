#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Scans each owning family's table in turn; the first (n, k) row wins.
const Q8ShapeEntry* find_shape(std::int32_t n, std::int32_t k) {
    for (std::size_t i = 0; i < qwen3_5_q8_shapes_count; ++i) {
        const Q8ShapeEntry& entry = qwen3_5_q8_shapes[i];
        if (entry.n == n && entry.k == k) return &entry;
    }
    for (std::size_t i = 0; i < qwen4_exp_q8_shapes_count; ++i) {
        const Q8ShapeEntry& entry = qwen4_exp_q8_shapes[i];
        if (entry.n == n && entry.k == k) return &entry;
    }
    return nullptr;
}

// Shapes without a tuned entry use the runtime-shape templates: predicated SIMT for decode and
// verification widths, MMA tiles beyond, each covering any row count and any K padded to 128.
// Both SIMT widths of a schedule pair reduce a column identically, so a column's output does not
// depend on how many columns (up to 8) share the call.
Q8Launch select_q8_generic(std::int32_t n, std::int32_t k, std::int32_t t) {
    (void)n;
    (void)k;
    if (t <= 4) return launch_q8_a16_simt_r8_t4;
    if (t <= 8) return launch_q8_a16_simt_r8_t8;
    if (t <= 64) return launch_q8_a16_mma_r32_t64;
    if (t <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}
} // namespace

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearRoutes routes) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    if (n <= 0 || k <= 0) throw std::invalid_argument("q8 linear: unsupported shape");
    const Q8ShapeEntry* entry = find_shape(n, k);
    if (entry != nullptr) {
        if (has_route(routes, LinearRoutes::Q8SlicedK) && entry->fast != nullptr) return entry->fast(t);
        if (has_route(routes, LinearRoutes::Q8HcSlicedK) && entry->hc_fast != nullptr) return entry->hc_fast(t);
        return entry->select(t);
    }
    return select_q8_generic(n, k, t);
}

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy,
                          LinearRoutes routes) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    if (column_invariant(policy)) {
        if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
        const Q8ShapeEntry* entry = find_shape(n, k);
        if (entry == nullptr || entry->invariant == nullptr)
            throw std::invalid_argument("q8 linear: no column-invariant route for this problem");
        return entry->invariant(t);
    }
    return select_q8_a16_launch(n, k, t, routes);
}

bool q8_column_invariant_supported(std::int32_t n, std::int32_t k) noexcept {
    const Q8ShapeEntry* entry = find_shape(n, k);
    return entry != nullptr && entry->invariant != nullptr;
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                  LinearRoutes routes, cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy, routes)(x, w, out, stream);
}

} // namespace infernix::ops::detail
