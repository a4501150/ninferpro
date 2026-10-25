#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/rows.h"

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    case LinearPolicy::A16ColumnInvariant:
    case LinearPolicy::AllowA4Residual:
    case LinearPolicy::AllowA8Residual:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy, LinearRoutes routes,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, routes, stream);
        return;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, routes, workspace, stream);
        return;
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy, LinearRoutes::Exact);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy, LinearRoutes::Exact);
        return 0;
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

bool linear_column_invariant_supported(QType qtype, std::int32_t output_rows, std::int32_t input_rows) noexcept {
    switch (qtype) {
    case QType::Q8_G32_FP16: return detail::q8_column_invariant_supported(output_rows, input_rows);
    case QType::BF16: return detail::bf16_column_invariant_supported(output_rows, input_rows);
    case QType::FP8_E4M3FN_ROW_BF16: return detail::fp8_column_invariant_supported(output_rows, input_rows);
    default: break;
    }
    // NVFP4 registers its text problems; the other formats register none.
    try {
        (void)linear_workspace_capacity_bytes(qtype, output_rows, input_rows, LinearPolicy::A16ColumnInvariant, 1, 1);
        return true;
    } catch (...) { return false; }
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes) {
    validate_linear_semantics(x, w, out, policy);
    if (!valid_linear_routes(routes)) {
        throw std::invalid_argument("linear: unsupported route set");
    }
    dispatch_linear(x, w, out, policy, routes, &workspace, stream);
}

void linear_rows(const Tensor& x, const Weight& w, std::span<Tensor* const> parts, LinearPolicy policy,
                 WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes) {
    if (parts.empty()) { throw std::invalid_argument("linear_rows: no parts"); }
    if (!valid_linear_routes(routes)) { throw std::invalid_argument("linear: unsupported route set"); }
    const std::int32_t T = x.ne[1];
    std::int32_t rows    = 0;
    bool direct          = w.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                  parts.size() <= static_cast<std::size_t>(detail::LinearBf16RowParts::kMaxParts);
    detail::LinearBf16RowParts targets{};
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const Tensor& part = *parts[i];
        if (part.dtype != DType::BF16 || !part.is_contiguous() || part.ne[1] != T || part.ne[2] != 1 ||
            part.ne[3] != 1 || part.ne[0] <= 0 || !aligned_to(part.data, 16)) {
            throw std::invalid_argument("linear_rows: parts must be contiguous, 16-byte aligned BF16 [r_i, T]");
        }
        rows += part.ne[0];
        direct = direct && part.ne[0] % 8 == 0;
        if (i < static_cast<std::size_t>(detail::LinearBf16RowParts::kMaxParts)) {
            targets.data[i] = static_cast<__nv_bfloat16*>(part.data);
            targets.end[i]  = rows;
        }
    }
    if (rows != w.n) { throw std::invalid_argument("linear_rows: parts must cover the output rows"); }
    for (std::size_t i = parts.size(); i < static_cast<std::size_t>(detail::LinearBf16RowParts::kMaxParts); ++i) {
        targets.data[i] = targets.data[parts.size() - 1];
        targets.end[i]  = rows;
    }
    if (direct) {
        // The complete output's checks, on a stand-in with its shape (it is never written).
        validate_linear_semantics(x, w, Tensor(parts[0]->data, DType::BF16, {w.n, T}), policy);
        if (detail::fp8_dispatch_rows(x, w, targets, policy, routes, &workspace, stream)) { return; }
    }
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {w.n, T});
    linear(x, w, out, policy, workspace, stream, routes);
    split_rows(out, parts, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, LinearRoutes::Exact, nullptr, stream);
}

} // namespace infernix::ops
