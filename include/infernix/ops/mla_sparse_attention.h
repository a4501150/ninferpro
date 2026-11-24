#pragma once

#include "core/tensor.h"
#include "infernix/types.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Op family: sparse multi-head latent attention over a paged shared latent cache, with its two
 * absorbed projections and the latent append. A layer caches one shared latent per token
 * (width `latent_dim`, BF16) that serves as both key and value for every query head; the
 * absorbed form scores the latent directly against the per-head projected queries.
 *
 * The reference attention of a query head h over a selected token set J is
 *
 *   o_h = sum_{j in J} softmax_j(scale * <q_h, k_h,j>) v_h,j,      k_h,j = W_uk,h c_j,
 *   v_h,j = W_uv,h c_j,   (c_j: the shared latent of token j)
 *
 * with W_uk,h [head_dim x latent_dim], W_uv,h [head_dim x latent_dim]. In the absorbed form,
 *
 *   qa_h      = W_uk,h^T q_h                    (latent_dim per head; mla_absorbed_q_projection)
 *   score_h,j = scale * <qa_h, c_j>             (scale = 1/sqrt(head_dim); softmax in FP32)
 *   ol_h      = sum_j p_h,j c_j                 (latent_dim per head)
 *   o_h       = W_uv,h ol_h                     (head_dim per head; the attention's epilogue)
 *
 * so the Op is attention of `heads` query heads of width latent_dim against one shared
 * key = value latent per token over the selected tokens, followed by the per-head output
 * projection. The mathematical oracle of every entry is the formula above evaluated naively in
 * FP64 over the represented inputs (BF16 inputs promoted, an INT8 cache's latents decoded with
 * their stored scales, FP32 softmax of the FP64 scores, FP64 value reduction); BF16 outputs are
 * promoted for comparison and storage rounding belongs to each entry's criterion, not to the
 * oracle.
 */

struct MlaAttentionGeometry {
    std::int32_t heads      = 0; // query heads, a multiple of 16
    std::int32_t head_dim   = 0; // per-head query and output width
    std::int32_t latent_dim = 0; // shared key = value latent width per token
};

/// The call's columns: `batch` sequences of `width` consecutive positions each. Column
/// c = b * width + t is column t of sequence b; positions[c] is its absolute (cache) position.
/// Shared by the latent attention and the pooled-key indexer (dsa_indexer.h), whose caches are
/// token-paged with the same page tables.
struct MlaSparseBatch {
    Tensor block_tables; // I32 [pages per row, rows]: pool-local physical page per logical page
    Tensor table_rows;   // I32 [batch]: active table row of each sequence
    Tensor positions;    // I32 [batch * width]
    std::int32_t batch = 0;
    std::int32_t width = 0;
};

/// One layer's paged latent cache in the engine's page-major plane order: logical token p lives at
/// page table[p >> 6], slot p & 63. Pages hold 64 tokens; the caller materializes every page the
/// call reads. Two storages:
///   BF16      pages BF16 [latent_dim, 64, 1, pages], scales empty: the latent's 512 contiguous
///             BF16 elements, cached bit-for-bit (1024 B per token).
///   INT8-G64  pages I8 [latent_dim, 64, 1, pages] and scales FP16 [latent_dim / 64, 64, 1, pages]:
///             per contiguous group of 64 latent values x the persistent INT8 group encoding of
///             kv_cache_append.h,
///               a = max_i |FP32(x[i])|, scale_bits = FP16_RNE(a / 127), s = FP32(scale_bits),
///               code[i] = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * FP32(1 / s)), -127, 127)),
///             decoded as FP32(code[i]) * s (528 B per token). Readers use the decoded values.
struct MlaLatentBatchView {
    Tensor pages;
    Tensor scales; // INT8-G64 only
};

/// Appends whole columns of latent rows to the batched paged latent cache. latents is contiguous
/// BF16 [latent_dim, width, batch]; column c's row latents[:, c % width, c / width] is stored at
/// logical position positions[c] through table row table_rows[c / width]: copied bit-for-bit
/// (BF16), or encoded group by group with its scales (INT8-G64). Every addressed row is
/// overwritten; no other byte changes. The Op owns no frontier or commit authority, and a
/// column's stored bytes depend only on that column.
void mla_latent_append(const Tensor& latents, const MlaSparseBatch& batch,
                       const MlaLatentBatchView& cache, cudaStream_t stream);

/**
 * Op: absorbed query-to-latent projection. qa[h, d, c] = BF16(sum_d0 w_uk[h, d, d0] *
 * FP32(q[d0, h, c])) for every head h and column c: the per-head batched product of the stored
 * per-head transposed up-projection with the layer's queries.
 *
 * Logical shapes:
 *   q BF16 [head_dim, heads, T] contiguous; qa BF16 [latent_dim, heads, T] contiguous; w_uk BF16
 *   [head_dim, latent_dim, heads] contiguous, element (d0, d, h) = (W_uk,h)^T[d][d0]: per head
 *   the transpose of the natural [head_dim, latent_dim] up-projection, each stored row the
 *   head_dim query dimensions of one latent dimension, contiguous (the converter writes this
 *   layout). T > 0.
 *
 * Numeric:
 *   FP32 accumulation in dimension order, one BF16 rounding of the result. A column's outputs
 *   depend only on that column's inputs.
 *
 * Effects:
 *   Writes qa completely; inputs unchanged; no aliasing between inputs and qa.
 *
 * Workspace / Execution:
 *   none; stream-ordered without host synchronization.
 */
void mla_absorbed_q_projection(const Tensor& q, const Tensor& w_uk, Tensor& qa,
                               cudaStream_t stream);

/**
 * Op: latent-to-output projection. o[h, e, c] = BF16(sum_d w_uv[h, e, d] * FP32(ol[d, h, c])):
 * the per-head batched down-projection of the unnormalized latent-space attention output.
 * Composing it after a call that produced ol in BF16 adds one BF16 rounding of ol; the sparse
 * attention Op below applies this projection to its FP32 latent sums instead, so its output
 * carries no ol rounding.
 *
 * Logical shapes:
 *   ol BF16 [latent_dim, heads, T] contiguous; o BF16 [head_dim, heads, T] contiguous; w_uv BF16
 *   [latent_dim, head_dim, heads] contiguous, element (d, e, h) = W_uv,h[e][d]: per head the
 *   natural down-projection, each stored row the latent_dim contracted elements of one output
 *   dimension, contiguous (the converter writes this layout). T > 0.
 *
 * Numeric / Effects / Workspace / Execution:
 *   as mla_absorbed_q_projection, with the same FP32-accumulate one-rounding profile.
 */
void mla_latent_output_projection(const Tensor& ol, const Tensor& w_uv, Tensor& o,
                                  cudaStream_t stream);

/// The selected token set of every column. pools is I32 [top_pools, columns] (slot (k, c) at
/// k + c * top_pools): ascending selected pool ids; counts is I32 [columns]: the number of
/// selected pools, or -1 when the column attends every visible token. Either tensor null (or
/// pools empty) selects the dense causal set for every column: the model's per-call dense mode,
/// which skips its indexer when every visible token is selected anyway. A sparse column attends
/// the tokens of its selected pools (pool p covers positions pool*p .. pool*p + pool - 1 of its
/// sequence, pools starting at the sequence's first token) unioned with the visible incomplete
/// tail: positions pool * floor(n/pool) .. n - 1 for its visible count n = position + 1 (at most
/// pool - 1 tokens). Selected sets are the indexer's; this Op never selects.
struct MlaSelectedPools {
    const Tensor* pools  = nullptr;
    const Tensor* counts = nullptr;
    std::int32_t pool    = 4; // tokens per pool, as the indexer's geometry
};

/// Execution envelope: the maximum number of key splits of a launch (flash decoding over the
/// selected tokens). 0 selects the automatic route: splits = clamp(ceil(max_context / 32), 1, 65)
/// per launch, and each column uses min(splits, ceil(attended tokens / 32)) of them, so a
/// column's bits depend only on its own inputs, its sequence's cache and `max_context`, never on
/// the call's other columns, width or batch. A positive value fixes the launch's splits (the
/// model's per-call choice, e.g. 1 for prefill); a column's bits then also depend on that host
/// value, and stay independent of the batch within it. The workspace query and the execution
/// must use the same envelope value.
///
/// tensor_cores selects the tensor-core route (heads 64 only; splits is then ignored): one pass
/// per column over all of its attended tokens in 16-token tiles, scores and value sums on BF16
/// MMAs with FP32 accumulation (see the Op's Numeric). Meant for wide prompt calls; a column's
/// bits depend only on its own inputs and its sequence's cache.
struct MlaAttentionExecutionEnvelope {
    std::int32_t splits = 0;
    bool tensor_cores   = false;
};

/// Caller-owned transient capacity for every legal call of up to `columns` columns at
/// `max_context` under `envelope`: the split partials, the merged latent sums and the column
/// chunking the execution uses. Invalid profiles throw.
[[nodiscard]] std::size_t
mla_sparse_attention_workspace_bytes(const MlaAttentionGeometry& geometry, std::int32_t columns,
                                     std::int32_t max_context,
                                     MlaAttentionExecutionEnvelope envelope);

/**
 * Op: sparse absorbed latent attention with fused output projection. For every column c
 * (position p = positions[c], visible count n = p + 1) over its selected token set J(c)
 * (MlaSelectedPools; the union of the selected pools' tokens and the visible incomplete tail, in
 * ascending position order):
 *
 *   score[h, j]  = scale * <FP32(qa[:, h, c]), c_j>       (c_j: the cached latent, decoded)
 *   p[h, j]      = FP32 softmax over J(c) of score[h, j]         (stable: exact tile maxima)
 *   ol[h, :]     = sum_j p[h, j] * c_j
 *   out[e, h, c] = BF16(sum_d w_uv[h, e, d] * ol[h, d])
 *
 * The latent sums accumulate in FP32 across the call's splits and are normalized before the
 * projection; no ol rounding occurs. Keys and values are the same latent rows, read once per
 * (column, split). The weights are read once per call, not per column. Empty splits contribute
 * nothing; a column with a nonempty selected set (or the dense set) has l > 0, and its output is
 * fully determined by its own inputs, its sequence's cache and the envelope.
 *
 * Logical shapes:
 *   qa BF16 [latent_dim, heads, columns] contiguous (mla_absorbed_q_projection's output);
 *   out BF16 [head_dim, heads, columns] contiguous; w_uv as in mla_latent_output_projection;
 *   cache.pages as in MlaLatentBatchView with latent_dim rows; batch as in MlaSparseBatch with
 *   columns = batch * width; selected pools/counts sized for `columns` (the call's flat column
 *   count). columns > 0, every position in [0, max_context).
 *
 * Supported domain:
 *   BF16 inputs and a BF16 or INT8-G64 cache; heads a multiple of 16; scale > 0. The registered
 *   geometry is heads 64, head_dim 256, latent_dim 512 (scale 1/16).
 *
 * Numeric:
 *   FP32 scores and online softmax (exact per-tile maxima, exponentials and in-order value fold
 *   per 32-token tile), FP32 split merge (max, weight exp, weighted sums in split order), FP32
 *   projection accumulation, one BF16 rounding of out. Selection belongs to the indexer Op.
 *   The tensor-core route: the cached latents enter the MMAs exactly (BF16 as stored, INT8 codes
 *   with each 64-dimension group's dot scaled in FP32), qa as stored; FP32 online softmax per
 *   16-token tile; each probability (times the INT8 group scale) enters the value MMA as a BF16
 *   high part plus the BF16 rounding of its remainder (about 16 significant bits); MMA
 *   accumulation order is the hardware's.
 *
 * Effects:
 *   Writes out completely; reads the cache; nothing else changes. Inputs, cache and out pairwise
 *   non-overlapping; workspace is caller-owned transient scratch (the capacity query above).
 *
 * Execution:
 *   stream-ordered without host synchronization; every split, position and selection is read on
 *   the device, so a captured call serves any device values of the same shapes and envelope.
 */
void mla_sparse_attention(const Tensor& qa, const Tensor& w_uv, const MlaLatentBatchView& cache,
                          const MlaSparseBatch& batch, const MlaSelectedPools& selected,
                          const MlaAttentionGeometry& geometry, float scale,
                          std::int32_t max_context, MlaAttentionExecutionEnvelope envelope,
                          void* workspace, std::size_t workspace_bytes, Tensor& out,
                          cudaStream_t stream);

} // namespace infernix::ops
