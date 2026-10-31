#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_shapes.h"
#include <stdexcept>

namespace infernix::ops::detail {

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (std::size_t i = 0; i < qwen3_5_q4_shapes_count; ++i) {
        const Q4ShapeEntry& entry = qwen3_5_q4_shapes[i];
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q4 linear: unsupported shape");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy) || column_invariant(policy))
        throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
