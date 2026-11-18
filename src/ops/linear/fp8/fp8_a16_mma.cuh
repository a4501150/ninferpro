#pragma once

// Large-T row-scaled E4M3 weight x BF16 activation Tensor Core GEMM:
//
//   out[M,T] = (scale[M] * e4m3_codes[M,K]) * x[K,T]
//
// The body is the shared floating-point-code A16 GEMM (fp_a16/a16_mma.cuh) with Fp8A16Format.

#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_shared.cuh"
#include "ops/linear/fp_a16/a16_mma.cuh"

namespace infernix::ops::detail {

// Keep restricted pointers in the device ABI. Putting them in an aggregate loses
// NVCC alias information and increases register pressure in the K128 mainloop.
template <class Schedule, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, Epilogue epilogue,
    RowPolicy row_policy, int rows, int input_rows, int token_offset, int count) {
    fp_a16_mma_body<Fp8A16Format, Schedule, FullTokens, Output, Epilogue, RowPolicy>(
        static_cast<int>(threadIdx.x), x, weight_codes, row_scales, 0.0f, output, epilogue,
        row_policy, rows, input_rows, token_offset, count);
}

} // namespace infernix::ops::detail
