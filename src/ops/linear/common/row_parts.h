#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// Up to kMaxParts consecutive row ranges of one contraction, each written to its own contiguous
// [rows_i, T] tensor (what split_rows would copy them into). Every boundary is a multiple of eight
// rows, so a vector store lies in one part. Unused parts repeat the last end.
struct LinearBf16RowParts {
    static constexpr int kMaxParts = 4;
    __nv_bfloat16* data[kMaxParts];
    std::int32_t end[kMaxParts]; // exclusive, ascending

#ifdef __CUDACC__
    // Constant indices keep the fields in registers (a runtime index would copy them to local memory).
    __device__ __forceinline__ __nv_bfloat16* at(int row, int token) const {
        __nv_bfloat16* base = data[0];
        int begin = 0, rows = end[0];
#pragma unroll
        for (int i = 1; i < kMaxParts; ++i) {
            if (row >= end[i - 1]) {
                base  = data[i];
                begin = end[i - 1];
                rows  = end[i] - end[i - 1];
            }
        }
        return base + static_cast<std::int64_t>(token) * rows + (row - begin);
    }

    __device__ __forceinline__ void store(int row, int token, float value) const {
        *at(row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(int row, int token, uint4 values) const {
        *reinterpret_cast<uint4*>(at(row, token)) = values;
    }
#endif
};

} // namespace infernix::ops::detail
