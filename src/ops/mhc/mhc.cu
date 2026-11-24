// infernix::ops — mhc: manifold-constrained hyper-connection residual streams.
//
// Wrapper (validation, workspace scope, dispatch on the stream count), launchers and kernels in one
// translation unit, like hyper_connection. See include/infernix/ops/mhc.h for the contract.

#include "infernix/ops/mhc.h"

#include "core/pdl.cuh"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
// The mixer's split-K shape, fixed for every T so a column's bits never depend on the call's width:
// one CTA of kChunkD threads reduces kChunkD features across every stream, then walks kCols columns
// in sequence; the epilogue sums the chunks in ascending order.
constexpr int kChunkD = 64;
constexpr int kCols   = 8;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("mhc: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("mhc ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 &&
           t.ne[3] == 1;
}

bool contiguous_1d(const Tensor& t, DType dtype, int length) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[0] == length &&
           t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

bool supported_streams(int streams) { return streams == 1 || streams == 2 || streams == 4; }

int gate_rows(int streams) { return (2 + streams) * streams; }

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a.data);
    const auto pb = reinterpret_cast<std::uintptr_t>(b.data);
    return pa < pb + b.bytes() && pb < pa + a.bytes();
}

__device__ __forceinline__ float bf16_value(bf16 v) { return __bfloat162float(v); }

__device__ __forceinline__ bf16 rounded_bf16(float v) { return __float2bfloat16_rn(v); }

// One output stream's update contribution: post_s * y + sum_j comb[j][s] * R[j], in ascending
// stream order. Shared verbatim by mhc_update's kernel and the fused mixer's first stage, so both
// routes round the same FP32 arithmetic to the same bits; the explicit fmaf pins the contraction
// (fma(x, y, 0) is x * y's correctly rounded product) against per-kernel scheduling choices.
template <int kStreams>
__device__ __forceinline__ float update_stream(const float* post_column, const float* comb_column,
                                               float y_value, const float (&residual_old)[kStreams],
                                               int s) {
    float acc = __fmaf_rn(post_column[s], y_value, 0.0F);
#pragma unroll
    for (int j = 0; j < kStreams; ++j) {
        acc = __fmaf_rn(comb_column[s * kStreams + j], residual_old[j], acc);
    }
    return acc;
}

// The mixer's first stage: per column and chunk (kChunkD features across every stream), the
// fixed-order partials of f = fn . R and of the squared RMS, from the rounded BF16 values of the
// (updated) residual. With kUpdate the stage first applies the residual update elementwise (the
// same shared arithmetic as mhc_update's kernel, so the fused call keeps the separate calls' bits).
//
// The block's kThreads threads split into kRowGroups groups of kChunkD threads, one feature d each;
// group g reduces gate rows [g * rows_per_group, ...) of its chunk, so a thread holds only a few
// fn rows' accumulators and the chunk streams its fn slice in one memory wave. Group 0's slot
// rows_per_group carries the chunk's squared sum (the other groups contribute zeros there), so the
// reduced layout matches the epilogue's: value v of column t, chunk c at
// partials[(v * columns + t) * chunks + c], v < gate rows naming f's rows and v == gate rows the
// squared sum.
template <int kStreams, bool kUpdate>
__global__ __launch_bounds__(kThreads) void mhc_partial_kernel(
    const bf16* __restrict__ fn, const bf16* __restrict__ residual_in,
    bf16* __restrict__ residual_out, const bf16* __restrict__ y, const float* __restrict__ post,
    const float* __restrict__ comb, int hidden, int chunks, int columns,
    float* __restrict__ partials) {
    pdl::enter();
    constexpr int kRows       = (2 + kStreams) * kStreams;
    constexpr int kRowGroups  = kThreads / kChunkD;
    constexpr int kRowsGroup  = (kRows + kRowGroups - 1) / kRowGroups;
    constexpr int kWarpsGroup = kChunkD / 32;
    constexpr int kVals       = kRowsGroup + 1;
    constexpr int kInner      = 4; // columns reduced per barrier pair
    __shared__ float red[kInner * kVals * kRowGroups * kWarpsGroup];
    const int tid     = static_cast<int>(threadIdx.x);
    const int group   = tid / kChunkD;
    const int d       = static_cast<int>(blockIdx.x) * kChunkD + (tid % kChunkD);
    const bool active = d < hidden;
    const int row_lo  = group * kRowsGroup;
    const int row_hi  = min(kRows, row_lo + kRowsGroup);
    float values[kVals][kInner];
    for (int base = 0; base < kCols; base += kInner) {
        // Column base's group starts at or before the last live column, so the batch is not empty.
        if (static_cast<int>(blockIdx.y) * kCols + base >= columns) { break; }
        float r[kInner][kStreams];
#pragma unroll
        for (int ii = 0; ii < kInner; ++ii) {
            const int t              = static_cast<int>(blockIdx.y) * kCols + base + ii;
            const bool live          = t < columns;
            const std::size_t column = static_cast<std::size_t>(t) * (kStreams * hidden);
            if (kUpdate) {
                float residual_old[kStreams];
#pragma unroll
                for (int j = 0; j < kStreams; ++j) {
                    residual_old[j] =
                        live && active
                            ? bf16_value(
                                  residual_in[column + static_cast<std::size_t>(j) * hidden + d])
                            : 0.0F;
                }
                const float y_value =
                    live && active ? bf16_value(y[static_cast<std::size_t>(t) * hidden + d]) : 0.0F;
                const float* post_column = post + static_cast<std::size_t>(t) * kStreams;
                const float* comb_column = comb + static_cast<std::size_t>(t) * kStreams * kStreams;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const bf16 updated = rounded_bf16(update_stream<kStreams>(
                        post_column, comb_column, y_value, residual_old, s));
                    if (live && active && group == 0) {
                        residual_out[column + static_cast<std::size_t>(s) * hidden + d] = updated;
                    }
                    r[ii][s] = bf16_value(updated);
                }
            } else {
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    r[ii][s] =
                        live && active
                            ? bf16_value(
                                  residual_in[column + static_cast<std::size_t>(s) * hidden + d])
                            : 0.0F;
                }
            }
        }
        // One fn read per (row, stream) serves every column of the batch; each column's partial
        // accumulates its kStreams products in ascending stream order, then the batch's
        // kInner * kVals values reduce through shared memory in one barrier pair.
#pragma unroll
        for (int v = 0; v < kRowsGroup; ++v) {
            const int i = row_lo + v;
            float acc[kInner];
#pragma unroll
            for (int ii = 0; ii < kInner; ++ii) { acc[ii] = 0.0F; }
            if (i < row_hi) {
                const bf16* row = fn + static_cast<std::size_t>(i) * (kStreams * hidden);
#pragma unroll
                for (int s = 0; s < kStreams; ++s) {
                    const float fnv =
                        active ? bf16_value(row[static_cast<std::size_t>(s) * hidden + d]) : 0.0F;
#pragma unroll
                    for (int ii = 0; ii < kInner; ++ii) { acc[ii] += fnv * r[ii][s]; }
                }
            }
#pragma unroll
            for (int ii = 0; ii < kInner; ++ii) { values[v][ii] = acc[ii]; }
        }
#pragma unroll
        for (int ii = 0; ii < kInner; ++ii) {
            float squared = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { squared += r[ii][s] * r[ii][s]; }
            values[kRowsGroup][ii] = group == 0 ? squared : 0.0F;
        }
        const int lane          = tid & 31;
        const int warp_in_group = (tid % kChunkD) / 32;
#pragma unroll
        for (int v = 0; v < kVals; ++v) {
#pragma unroll
            for (int ii = 0; ii < kInner; ++ii) {
                float x = values[v][ii];
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    x += __shfl_xor_sync(0xFFFFFFFFU, x, offset);
                }
                if (lane == 0) {
                    red[((v * kInner + ii) * kRowGroups + group) * kWarpsGroup + warp_in_group] = x;
                }
            }
        }
        __syncthreads();
        if (tid % kChunkD < kInner * kVals) {
            const int local      = tid % kChunkD;
            const int v          = local / kInner;
            const int ii         = local % kInner;
            const bool is_square = v == kRowsGroup;
            const int global_row = group * kRowsGroup + v;
            const int t_i        = static_cast<int>(blockIdx.y) * kCols + base + ii;
            const bool writes    = t_i < columns && (is_square ? group == 0 : global_row < kRows);
            if (writes) {
                const int row = is_square ? kRows : global_row;
                float x       = 0.0F;
#pragma unroll
                for (int w = 0; w < kWarpsGroup; ++w) {
                    x += red[((v * kInner + ii) * kRowGroups + group) * kWarpsGroup + w];
                }
                partials[(static_cast<std::size_t>(row) * columns + t_i) * chunks +
                         static_cast<int>(blockIdx.x)] = x;
            }
        }
        // The next batch's warp partials overwrite red.
        __syncthreads();
    }
}

// The mixer's second stage: per column, the fixed-order sum of the chunks gives f and the squared
// RMS; warp 0 turns them into pre, post and the Sinkhorn-normalized comb (the S x S matrix lives on
// one warp's lanes: lane = i * S + j, input i, output j), and every CTA writes its feature slice of
// x = bf16(sum_s pre_s * R[s*H + d]) from the (updated) residual.
template <int kStreams>
__global__ __launch_bounds__(kThreads) void mhc_epilogue_kernel(
    const float* __restrict__ partials, const bf16* __restrict__ residual,
    const float* __restrict__ base, const float* __restrict__ scale, int hidden, int chunks,
    int columns, int sinkhorn_iters, float eps_norm, float eps, bf16* __restrict__ x,
    float* __restrict__ post, float* __restrict__ comb) {
    pdl::enter();
    constexpr int kRows = (2 + kStreams) * kStreams;
    constexpr int kS2   = kStreams * kStreams;
    const int kWidth    = kStreams * hidden;
    __shared__ float sh_f[kRows + 1];
    __shared__ float sh_pre[kStreams];
    const int t    = static_cast<int>(blockIdx.y);
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    if (tid < 32) {
        // f and the RMS: one lane per gate row, one lane for the squared sum, chunks in ascending
        // order. f = (fn . R) * rsqrt(mean(R^2) + eps_norm).
        if (lane < kRows) {
            const float* row_partial = partials +
                                       static_cast<std::size_t>(lane) * columns * chunks +
                                       static_cast<std::size_t>(t) * chunks;
            float sum                = 0.0F;
            for (int c = 0; c < chunks; ++c) { sum += row_partial[c]; }
            sh_f[lane] = sum;
        }
        if (lane == kRows) {
            const float* sq_partial = partials +
                                      static_cast<std::size_t>(kRows) * columns * chunks +
                                      static_cast<std::size_t>(t) * chunks;
            float sum               = 0.0F;
            for (int c = 0; c < chunks; ++c) { sum += sq_partial[c]; }
            sh_f[kRows] = sum;
        }
        __syncwarp();
        const float inverse_rms = rsqrtf(sh_f[kRows] / static_cast<float>(kWidth) + eps_norm);
        if (lane < kRows) { sh_f[lane] *= inverse_rms; }
        __syncwarp();
        // pre and post (independent of the Sinkhorn below); the first CTA of the column publishes
        // post, so the other warps can compute x while warp 0 normalizes comb.
        if (lane < kStreams) {
            sh_pre[lane] = sigmoid(sh_f[lane] * scale[0] + base[lane]) + eps;
            if (blockIdx.x == 0 && post != nullptr) {
                post[static_cast<std::size_t>(t) * kStreams + lane] =
                    2.0F * sigmoid(sh_f[kStreams + lane] * scale[1] + base[kStreams + lane]);
            }
        }
    }
    __syncthreads();
    if (tid < 32 && blockIdx.x == 0) {
        // comb: row-softmax over the output index (only the publishing CTA runs it; the others go
        // straight to x, which needs only pre), + eps, then Sinkhorn's alternating
        // column/row normalizations; lanes >= S*S carry lane & (S*S - 1)'s value so the full-warp
        // butterflies stay consistent.
        float logit = 0.0F;
        if (lane < kS2) {
            const int f_at = 2 * kStreams + lane; // row i = lane / S, column j = lane % S
            logit          = sh_f[f_at] * scale[2] + base[f_at];
        }
        float m = logit;
#pragma unroll
        for (int delta = 1; delta < kStreams; delta <<= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, delta, kStreams));
        }
        float e   = lane < kS2 ? expf(logit - m) : 0.0F;
        float den = e;
#pragma unroll
        for (int delta = 1; delta < kStreams; delta <<= 1) {
            den += __shfl_xor_sync(0xFFFFFFFFU, den, delta, kStreams);
        }
        float v = lane < kS2 ? e / den + eps : 0.0F;
        v       = __shfl_sync(0xFFFFFFFFU, v, lane & (kS2 - 1));
        for (int it = 0; it < sinkhorn_iters; ++it) {
            if (it > 0) {
                float row_sum = v;
#pragma unroll
                for (int delta = 1; delta < kStreams; delta <<= 1) {
                    row_sum += __shfl_xor_sync(0xFFFFFFFFU, row_sum, delta, kStreams);
                }
                v /= (row_sum + eps);
            }
            float column_sum = v;
#pragma unroll
            for (int delta = kStreams; delta < kS2; delta <<= 1) {
                column_sum += __shfl_xor_sync(0xFFFFFFFFU, column_sum, delta, kS2);
            }
            v /= (column_sum + eps);
        }
        if (blockIdx.x == 0 && comb != nullptr && lane < kS2) {
            comb[(static_cast<std::size_t>(t) * kStreams + (lane % kStreams)) * kStreams +
                 (lane / kStreams)] = v;
        }
    }
    // x from the (updated) residual and the published pre; warp 0 joins once its Sinkhorn is done.
    const int d = static_cast<int>(blockIdx.x) * kThreads + tid;
    if (d < hidden) {
        const std::size_t column = static_cast<std::size_t>(t) * kWidth;
        float acc                = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            acc +=
                sh_pre[s] * bf16_value(residual[column + static_cast<std::size_t>(s) * hidden + d]);
        }
        x[static_cast<std::size_t>(t) * hidden + d] = rounded_bf16(acc);
    }
}

// mhc_update's kernel: one feature d of every stream per thread, the old streams read before any
// write so residual_out may alias residual_in.
template <int kStreams>
__global__ __launch_bounds__(kChunkD) void mhc_update_kernel(const bf16* __restrict__ y,
                                                             const float* __restrict__ post,
                                                             const float* __restrict__ comb,
                                                             const bf16* __restrict__ residual_in,
                                                             bf16* __restrict__ residual_out,
                                                             int hidden, int columns) {
    pdl::enter();
    const int d = static_cast<int>(blockIdx.x) * kChunkD + static_cast<int>(threadIdx.x);
    const int t = static_cast<int>(blockIdx.y);
    if (d >= hidden) { return; }
    const std::size_t column = static_cast<std::size_t>(t) * (kStreams * hidden);
    float residual_old[kStreams];
#pragma unroll
    for (int j = 0; j < kStreams; ++j) {
        residual_old[j] =
            bf16_value(residual_in[column + static_cast<std::size_t>(j) * hidden + d]);
    }
    const float y_value      = bf16_value(y[static_cast<std::size_t>(t) * hidden + d]);
    const float* post_column = post + static_cast<std::size_t>(t) * kStreams;
    const float* comb_column = comb + static_cast<std::size_t>(t) * kStreams * kStreams;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        residual_out[column + static_cast<std::size_t>(s) * hidden + d] = rounded_bf16(
            update_stream<kStreams>(post_column, comb_column, y_value, residual_old, s));
    }
}

// The two mixer stages, shared by mhc_mix (no pending update) and mhc_update_mix (the first stage
// applies y's update exactly as mhc_update's kernel does). residual_out is null for mhc_mix.
template <int kStreams>
void run_mix(const Tensor& residual_in, const Tensor* residual_out, const Tensor* y,
             const Tensor* post_in, const Tensor* comb_in, const Tensor& fn, const Tensor& base,
             const Tensor& scale, std::int32_t sinkhorn_iters, float eps_norm, float eps, Tensor& x,
             Tensor* post_out, Tensor* comb_out, WorkspaceArena& workspace, cudaStream_t stream) {
    const int hidden  = x.ne[0];
    const int columns = x.ne[1];
    const int rows    = (2 + kStreams) * kStreams;
    const int chunks  = (hidden + kChunkD - 1) / kChunkD;
    auto scope        = workspace.scope();
    const DeviceSpan partials =
        workspace.alloc_bytes(static_cast<std::size_t>(rows + 1) *
                              static_cast<std::size_t>(columns) * chunks * sizeof(float));
    const auto* fn_p       = static_cast<const bf16*>(fn.data);
    const auto* residual_p = static_cast<const bf16*>(residual_in.data);
    auto* out_p      = residual_out != nullptr ? static_cast<bf16*>(residual_out->data) : nullptr;
    const int groups = (columns + kCols - 1) / kCols;
    if (residual_out != nullptr) {
        if (pdl::launch_consumer({dim3(chunks, groups), dim3(kThreads), 0, stream},
                                 mhc_partial_kernel<kStreams, true>, fn_p, residual_p, out_p,
                                 static_cast<const bf16*>(y->data),
                                 static_cast<const float*>(post_in->data),
                                 static_cast<const float*>(comb_in->data), hidden, chunks, columns,
                                 static_cast<float*>(partials.data)) != cudaSuccess) {
            check_launch("update mix partial");
        }
    } else {
        if (pdl::launch_consumer({dim3(chunks, groups), dim3(kThreads), 0, stream},
                                 mhc_partial_kernel<kStreams, false>, fn_p, residual_p,
                                 static_cast<bf16*>(nullptr), static_cast<const bf16*>(nullptr),
                                 static_cast<const float*>(nullptr),
                                 static_cast<const float*>(nullptr), hidden, chunks, columns,
                                 static_cast<float*>(partials.data)) != cudaSuccess) {
            check_launch("mix partial");
        }
    }
    const bf16* mix_residual = residual_out != nullptr ? out_p : residual_p;
    if (pdl::launch_consumer(
            {dim3((hidden + kThreads - 1) / kThreads, columns), dim3(kThreads), 0, stream},
            mhc_epilogue_kernel<kStreams>, static_cast<const float*>(partials.data), mix_residual,
            static_cast<const float*>(base.data), static_cast<const float*>(scale.data), hidden,
            chunks, columns, sinkhorn_iters, eps_norm, eps, static_cast<bf16*>(x.data),
            post_out != nullptr ? static_cast<float*>(post_out->data) : nullptr,
            comb_out != nullptr ? static_cast<float*>(comb_out->data) : nullptr) != cudaSuccess) {
        check_launch("mix epilogue");
    }
}

void validate_mix(const Tensor& residual, const Tensor& fn, const Tensor& base, const Tensor& scale,
                  std::int32_t streams, std::int32_t sinkhorn_iters, float eps_norm, float eps,
                  const Tensor& x, const Tensor* post, const Tensor* comb) {
    require(supported_streams(streams), "streams must be 1, 2 or 4");
    require(sinkhorn_iters >= 1, "sinkhorn_iters must be at least 1");
    require(eps_norm > 0.0F && eps > 0.0F, "eps_norm and eps must be positive");
    const int columns = residual.ne[1];
    const int hidden  = x.ne[0];
    const int rows    = gate_rows(streams);
    require(columns > 0 && hidden > 0, "extents must be positive");
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(x, DType::BF16),
            "residual and x must be contiguous BF16 tensors");
    require(residual.ne[0] == streams * hidden && x.ne[1] == columns,
            "residual and x shapes disagree");
    require(contiguous_2d(fn, DType::BF16) && fn.ne[0] == residual.ne[0] && fn.ne[1] == rows,
            "fn must be BF16 [rows, S*H] with rows = (2+S)*S");
    require(contiguous_1d(base, DType::FP32, rows), "base must be FP32 [rows]");
    require(contiguous_1d(scale, DType::FP32, 3), "scale must be FP32 [3]");
    if (post != nullptr) {
        require(contiguous_2d(*post, DType::FP32) && post->ne[0] == streams &&
                    post->ne[1] == columns,
                "post must be FP32 [S, T]");
    }
    if (comb != nullptr) {
        require(comb->data != nullptr && comb->dtype == DType::FP32 && comb->is_contiguous() &&
                    comb->ne[0] == streams && comb->ne[1] == streams && comb->ne[2] == columns &&
                    comb->ne[3] == 1,
                "comb must be FP32 [S, S, T]");
    }
    require(!overlaps(x, residual) && !overlaps(x, fn), "x must not alias the residual or fn");
}

void validate_update(const Tensor& y, const Tensor& post, const Tensor& comb,
                     const Tensor& residual_in, const Tensor& residual_out) {
    const int streams = post.ne[0];
    const int columns = y.ne[1];
    const int hidden  = y.ne[0];
    require(supported_streams(streams), "post's stream count must be 1, 2 or 4");
    require(contiguous_2d(y, DType::BF16) && columns > 0 && hidden > 0,
            "y must be a contiguous BF16 [H, T] tensor");
    require(contiguous_2d(post, DType::FP32) && post.ne[1] == columns, "post must be FP32 [S, T]");
    require(comb.data != nullptr && comb.dtype == DType::FP32 && comb.is_contiguous() &&
                comb.ne[0] == streams && comb.ne[1] == streams && comb.ne[2] == columns &&
                comb.ne[3] == 1,
            "comb must be FP32 [S, S, T]");
    require(contiguous_2d(residual_in, DType::BF16) && contiguous_2d(residual_out, DType::BF16) &&
                residual_in.ne[0] == streams * hidden && residual_in.ne[1] == columns &&
                residual_out.ne[0] == residual_in.ne[0] && residual_out.ne[1] == columns,
            "the residuals must be contiguous BF16 [S*H, T] tensors");
    if (residual_in.data != residual_out.data) {
        require(!overlaps(residual_in, residual_out), "the residuals must not partially overlap");
    }
    require(!overlaps(y, residual_out), "y must not alias the updated residual");
}

} // namespace

void mhc_mix(const Tensor& residual, const Tensor& fn, const Tensor& base, const Tensor& scale,
             std::int32_t streams, std::int32_t sinkhorn_iters, float eps_norm, float eps,
             Tensor& x, Tensor* post, Tensor* comb, WorkspaceArena& workspace,
             cudaStream_t stream) {
    validate_mix(residual, fn, base, scale, streams, sinkhorn_iters, eps_norm, eps, x, post, comb);
    switch (streams) {
    case 1:
        run_mix<1>(residual, nullptr, nullptr, nullptr, nullptr, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post, comb, workspace, stream);
        break;
    case 2:
        run_mix<2>(residual, nullptr, nullptr, nullptr, nullptr, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post, comb, workspace, stream);
        break;
    default:
        run_mix<4>(residual, nullptr, nullptr, nullptr, nullptr, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post, comb, workspace, stream);
        break;
    }
}

std::size_t mhc_mix_workspace_capacity_bytes(std::int32_t streams, std::int32_t hidden,
                                             std::int32_t min_tokens, std::int32_t max_tokens) {
    require(
        supported_streams(streams) && hidden > 0 && min_tokens > 0 && min_tokens <= max_tokens,
        "mhc workspace needs streams in {1, 2, 4}, a positive hidden width and a valid interval");
    const auto rows         = static_cast<std::size_t>(gate_rows(streams)) + 1;
    const auto chunks       = static_cast<std::size_t>((hidden + kChunkD - 1) / kChunkD);
    const std::size_t bytes = rows * static_cast<std::size_t>(max_tokens) * chunks * sizeof(float);
    return (bytes + 255) / 256 * 256;
}

void mhc_update(const Tensor& y, const Tensor& post, const Tensor& comb, const Tensor& residual_in,
                Tensor& residual_out, cudaStream_t stream) {
    validate_update(y, post, comb, residual_in, residual_out);
    const int streams = post.ne[0];
    const int hidden  = y.ne[0];
    const int columns = y.ne[1];
    switch (streams) {
    case 1:
        if (pdl::launch_consumer(
                {dim3((hidden + kChunkD - 1) / kChunkD, columns), dim3(kChunkD), 0, stream},
                mhc_update_kernel<1>, static_cast<const bf16*>(y.data),
                static_cast<const float*>(post.data), static_cast<const float*>(comb.data),
                static_cast<const bf16*>(residual_in.data), static_cast<bf16*>(residual_out.data),
                hidden, columns) != cudaSuccess) {
            check_launch("update");
        }
        break;
    case 2:
        if (pdl::launch_consumer(
                {dim3((hidden + kChunkD - 1) / kChunkD, columns), dim3(kChunkD), 0, stream},
                mhc_update_kernel<2>, static_cast<const bf16*>(y.data),
                static_cast<const float*>(post.data), static_cast<const float*>(comb.data),
                static_cast<const bf16*>(residual_in.data), static_cast<bf16*>(residual_out.data),
                hidden, columns) != cudaSuccess) {
            check_launch("update");
        }
        break;
    default:
        if (pdl::launch_consumer(
                {dim3((hidden + kChunkD - 1) / kChunkD, columns), dim3(kChunkD), 0, stream},
                mhc_update_kernel<4>, static_cast<const bf16*>(y.data),
                static_cast<const float*>(post.data), static_cast<const float*>(comb.data),
                static_cast<const bf16*>(residual_in.data), static_cast<bf16*>(residual_out.data),
                hidden, columns) != cudaSuccess) {
            check_launch("update");
        }
        break;
    }
}

void mhc_update_mix(const Tensor& y, const Tensor& post, const Tensor& comb,
                    const Tensor& residual_in, Tensor& residual_out, const Tensor& fn,
                    const Tensor& base, const Tensor& scale, std::int32_t streams,
                    std::int32_t sinkhorn_iters, float eps_norm, float eps, Tensor& x,
                    Tensor* post_out, Tensor* comb_out, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    validate_mix(residual_out, fn, base, scale, streams, sinkhorn_iters, eps_norm, eps, x, post_out,
                 comb_out);
    validate_update(y, post, comb, residual_in, residual_out);
    switch (streams) {
    case 1:
        run_mix<1>(residual_in, &residual_out, &y, &post, &comb, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post_out, comb_out, workspace, stream);
        break;
    case 2:
        run_mix<2>(residual_in, &residual_out, &y, &post, &comb, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post_out, comb_out, workspace, stream);
        break;
    default:
        run_mix<4>(residual_in, &residual_out, &y, &post, &comb, fn, base, scale, sinkhorn_iters,
                   eps_norm, eps, x, post_out, comb_out, workspace, stream);
        break;
    }
}

} // namespace infernix::ops
