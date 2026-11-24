#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Op: mhc_mix / mhc_update / mhc_update_mix — manifold-constrained hyper-connection residual
 * streams.
 *
 * Math / indexing:
 *   A residual R holds S streams of H features per column, stream-major: R[s*H + d, t] is feature
 *   d of stream s. Every block mixer (one before the attention, one before the MLP) computes, per
 *   column t, over the column's flat values R_flat[m] = R[m, t], m in [0, S*H):
 *
 *     r      = R_flat / sqrt(mean(R_flat^2) + eps_norm)   unweighted RMS over all S*H values
 *     f      = fn . r                                     fn: BF16 [rows, S*H], rows = (2+S)*S
 *     pre_s  = sigmoid(f[s] * scale[0] + base[s]) + eps                 s < S
 *     post_s = 2 * sigmoid(f[S+s] * scale[1] + base[S+s])
 *     comb   = softmax over the output index j of
 *              (f[2S + i*S + j] * scale[2] + base[2S + i*S + j])
 *              taken as an S x S matrix [input i][output j], then + eps
 *     comb   = comb / (column sums + eps)   (column sum: sum over the input index i, per output j)
 *     repeat (sinkhorn_iters - 1) times:
 *       comb = comb / (row sums + eps)     (row sum: sum over the output index j, per input i)
 *       comb = comb / (column sums + eps)
 *     x[d]   = bf16(sum_s pre_s * R[s*H + d])
 *
 *   comb[i][j] weighs input stream i feeding output stream j; the softmax and the row
 *   normalizations act over the output index, the column normalizations over the input index. The
 *   mixer returns x (BF16 [H, T], the block input) and keeps post (FP32 [S, T]) and comb
 *   (FP32 [S, S, T]; element (i, j, t) at (t*S + j)*S + i) for the update. After the block produced
 *   y (BF16 [H, T]):
 *
 *     R'[s*H + d] = bf16(post_s * y[d] + sum_j comb[j][s] * R[j*H + d])
 *
 *   with the sum over the input index j in ascending stream order.
 *
 * Logical shapes:
 *   residual: contiguous BF16 [S*H, T]. fn: contiguous BF16 [rows, S*H] (element (i, m) at
 *   i*S*H + m). base: contiguous FP32 [rows]; scale: contiguous FP32 [3]. x: contiguous BF16 [H,
 * T]. post: contiguous FP32 [S, T]; comb: contiguous FP32 [S, S, T]. y: contiguous BF16 [H, T]. T
 * is any positive extent; H any positive feature width; S the stream count.
 *
 * Supported domain:
 *   S in {1, 2, 4} (rows = (2+S)*S <= 24 and the S x S matrix stay warp-resident). fn's row count
 *   must be exactly (2+S)*S. sinkhorn_iters >= 1; eps_norm > 0; eps > 0 (the eps floor keeps every
 *   Sinkhorn divisor positive). All tensors BF16 or FP32 as listed above, contiguous.
 *
 * Numeric:
 *   Every entry evaluates in FP32 from the represented BF16 inputs and the FP32 base/scale, and
 *   rounds each BF16 output (x, the updated residual) exactly once; post and comb stay FP32. f is
 *   evaluated as a fixed-shape FP32 reduction of fn . R divided by the column's inverse RMS
 *   (rsqrt(mean + eps_norm)); the reduction's shape is private and identical for every column.
 *   post and comb therefore carry the FP32 reduction error of a 16384-term product sum (bounded by
 *   the sigmoid/softmax Lipschitz 1/4 against f); x and the residual carry one BF16 rounding on
 *   top. The independent oracle is the formula above in FP64 from the represented BF16 inputs.
 *
 *   Column invariance: a column's x, post, comb and updated residual bits depend only on that
 *   column's inputs — not on T, on the column's position, or on whether it was produced by
 *   mhc_update followed by mhc_mix or by the fused mhc_update_mix (which runs the same two-stage
 *   kernel pair on the same rounded values). This holds at every T.
 *
 * Effects:
 *   mhc_mix writes all of x, and all of post/comb when they are not null; it reads residual and
 *   does not modify it. mhc_update writes all of residual_out; residual_out may alias residual_in
 *   (the update is computed from the old streams before each element is written; the kernel
 *   partitions columns by feature d, so in-place update reads only elements it owns). x must not
 *   alias residual, y, or the updated residual (mix and update read the full residual per column
 *   while writing x). mhc_update_mix is mhc_update followed by mhc_mix on the updated residual,
 *   bit for bit; residual_out may alias residual_in, and post_out/comb_out may alias post/comb
 *   (their inputs are consumed by the update stage before the mix stage writes).
 *
 * Workspace:
 *   mhc_mix and mhc_update_mix need caller-owned transient storage of
 *   mhc_mix_workspace_capacity_bytes(streams, hidden, 1, T) scoped to the call (per-column split-K
 *   partials of f and of the squared RMS); it must not overlap any operand. mhc_update needs none.
 *   There is no persistent state side effect.
 *
 * Execution:
 *   Stream-ordered; safe under CUDA graph capture. The kernels are programmatic-dependent consumers
 *   of whatever precedes them on the stream and of each other (each calls the consumer protocol
 *   before its first dependent access), so a captured sequence overlaps launch latency without
 *   changing results.
 */

/// The mixer of one block: x, and post/comb for the update, from the residual R (BF16 [S*H, T]).
void mhc_mix(const Tensor& residual, const Tensor& fn, const Tensor& base, const Tensor& scale,
             std::int32_t streams, std::int32_t sinkhorn_iters, float eps_norm, float eps,
             Tensor& x, Tensor* post, Tensor* comb, WorkspaceArena& workspace, cudaStream_t stream);

/// The transient capacity mhc_mix and mhc_update_mix need for every T in the inclusive
/// [min_tokens, max_tokens] interval at this geometry. Invalid streams or intervals throw.
[[nodiscard]] std::size_t mhc_mix_workspace_capacity_bytes(std::int32_t streams,
                                                           std::int32_t hidden,
                                                           std::int32_t min_tokens,
                                                           std::int32_t max_tokens);

/// residual_out[s*H+d,t] = bf16(post[s,t] * y[d,t] + sum_j comb[j][s,t] * residual_in[j*H+d,t]);
/// the sum in ascending stream order. residual_out may alias residual_in. No workspace.
void mhc_update(const Tensor& y, const Tensor& post, const Tensor& comb, const Tensor& residual_in,
                Tensor& residual_out, cudaStream_t stream);

/// mhc_update(y, post, comb, residual_in, residual_out) followed by
/// mhc_mix(residual_out, fn, base, scale, ..., x, post_out, comb_out), bit for bit.
void mhc_update_mix(const Tensor& y, const Tensor& post, const Tensor& comb,
                    const Tensor& residual_in, Tensor& residual_out, const Tensor& fn,
                    const Tensor& base, const Tensor& scale, std::int32_t streams,
                    std::int32_t sinkhorn_iters, float eps_norm, float eps, Tensor& x,
                    Tensor* post_out, Tensor* comb_out, WorkspaceArena& workspace,
                    cudaStream_t stream);

} // namespace infernix::ops
