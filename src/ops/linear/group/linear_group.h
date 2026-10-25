#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "ops/linear/common/row_parts.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// The registered compositions of ops::linear_group: two independent projections of x in one launch,
// each with the bits of its own A16 route, writing their row ranges to their parts. False (nothing
// launched) where the pair, the width or the policy has no composition.
bool linear_group_composed(const Tensor& x, const Weight& first, const LinearBf16RowParts& first_parts,
                           const Weight& second, const LinearBf16RowParts& second_parts, LinearPolicy policy,
                           cudaStream_t stream);

} // namespace infernix::ops::detail
