#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "ops/linear/common/row_parts.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// The capacity of every route the problem may select over the interval, including the
// fp8-a8-pair route (the query takes no route flags).
[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

// Whether [n, k] registers an A16ColumnInvariant route.
[[nodiscard]] bool fp8_column_invariant_supported(std::int32_t n, std::int32_t k) noexcept;
void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream);
// linear_rows's direct route: true when the call runs a route that writes the row ranges to their
// tensors itself (the fp8-a8-pair route of a problem that registers it), false when the caller must
// split a complete output instead.
[[nodiscard]] bool fp8_dispatch_rows(const Tensor& x, const Weight& weight, const LinearBf16RowParts& parts,
                                     LinearPolicy policy, LinearRoutes routes, WorkspaceArena* workspace,
                                     cudaStream_t stream);

} // namespace infernix::ops::detail
