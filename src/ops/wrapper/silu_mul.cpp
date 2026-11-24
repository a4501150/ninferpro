// infernix::ops — silu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "infernix/ops/silu_mul.h"

#include "ops/launcher/silu_and_mul.h" // detail::silu_and_mul_launch

#include <cmath>
#include <stdexcept>

namespace infernix::ops {
namespace {

void check_elementwise_problem(const Tensor& gate, const Tensor& up, const Tensor& out,
                               const char* op) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument(std::string(op) + ": gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": out must be contiguous");
    }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": gate/up/out data must be non-null");
    }
}

} // namespace

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    check_elementwise_problem(gate, up, out, "silu_mul");
    detail::silu_and_mul_launch(gate, up, out, stream); // single variant -> direct dispatch
}

void silu_mul_clamped(const Tensor& gate, const Tensor& up, float limit, Tensor& out,
                      cudaStream_t stream) {
    check_elementwise_problem(gate, up, out, "silu_mul_clamped");
    if (!std::isfinite(limit) || limit <= 0.0F) {
        throw std::invalid_argument("silu_mul_clamped: limit must be finite and positive");
    }
    detail::silu_and_mul_clamped_launch(gate, up, limit, out, stream);
}

} // namespace infernix::ops
