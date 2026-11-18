#pragma once

// NVFP4 weight x BF16 activation Tensor Core GEMM. NVFP4 codes multiplied by their raw E4M3 G16
// scales are exactly representable in BF16; the global weight divisor (alpha) is applied to the
// complete FP32 reduction. The body is the shared floating-point-code A16 GEMM
// (fp_a16/a16_mma.cuh) with Nvfp4A16Format.

#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"
#include "ops/linear/fp_a16/a16_mma.cuh"

namespace infernix::ops::detail {

// Keep restricted pointers in the device ABI. Putting them in an aggregate loses
// NVCC alias information and increases register pressure in the K128 mainloop.
template <class Schedule, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales, float alpha, Output output, Epilogue epilogue,
    RowPolicy row_policy, int rows, int input_rows, int token_offset, int count) {
    fp_a16_mma_body<Nvfp4A16Format, Schedule, FullTokens, Output, Epilogue, RowPolicy>(
        static_cast<int>(threadIdx.x), x, weight_codes, weight_scales, alpha, output, epilogue,
        row_policy, rows, input_rows, token_offset, count);
}

} // namespace infernix::ops::detail
