#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q6/q6_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q6 linear: T must be positive");
    for (std::size_t s = 0; s < q6_linear_shape_set_count; ++s) {
        const auto& set = q6_linear_shape_sets[s];
        for (std::size_t i = 0; i < *set.count; ++i) {
            const Q6ShapeEntry& entry = set.rows[i];
            if (entry.n == n && entry.k == k) return entry.select(t);
        }
    }
    throw std::invalid_argument("q6 linear: unsupported shape");
}

Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy) || column_invariant(policy))
        throw std::invalid_argument("q6 linear: unsupported policy");
    return select_q6_a16_launch(n, k, t);
}

void q6_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q6_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
