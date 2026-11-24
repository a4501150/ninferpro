#pragma once

#include "infernix/ops/mla_sparse_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Op family: the pooled-key indexer of sparse latent attention. Per layer, a 128-wide index key
 * is computed per token, tokens are compressed into pools of `pool` consecutive tokens (pool p
 * covers positions 4p..4p+3 of its sequence; pools start at the sequence's first token, so only
 * complete pools have keys), and every query column selects the top `top_pools` pools by a
 * ReLU-of-dot score. The selected pools' tokens, unioned with the visible incomplete tail, are
 * the token set the latent attention Op attends (MlaSelectedPools).
 *
 * Index keys and pooling (the update Op; the model computes the pre-projection vectors):
 *
 *   k_t   = LayerNorm(x_t; gamma, beta, eps)     per channel, x_t = W_k x_t of the model,
 *                                                 evaluated in FP32, stored BF16
 *   z_t   = the model's W_g x_t                  BF16 [head_dim], cached as given
 *   key_p = BF16(sum_{i < pool} prob_i * k_{4p+i}),  per channel, summed in FP32;
 *   prob  = softmax over i (FP32) of FP32(z_{4p+i}) + ape[i],   ape FP32 [pool, head_dim]
 *
 * The stored key is the BF16 rounding of the FP32 sum. The reference implementation casts prob
 * to BF16 and accumulates the products in BF16; this contract defines the stored key as the
 * FP32 sum instead (one BF16 rounding, no per-product rounding), which is the value every
 * consumer reads.
 *
 * Scoring and selection (the select Op):
 *
 *   score_p    = sum_h w_h * relu(scale * <q_h, key_p>)     FP32; w BF16 [heads] per column
 *   selected   = the top min(top_pools, P) pools by score, P the column's complete pool count;
 *                equal scores take the lower pool id; the ids are written in ascending order
 *   count      = min(top_pools, P), or -1 when P <= top_pools (the dense causal set: every
 *                visible token is selected, so the model may skip the indexer and the attention
 *                Op takes its dense mode)
 *
 * The mathematical oracle of the update is the formula above in FP64 (the stored BF16 key
 * compared after its rounding); the select's oracle is the exact top-k of the Op's own FP32
 * scores (whose arithmetic the contract fixes below) plus, on scores whose top gaps exceed the
 * FP32 evaluation error, the FP64 formula's selection.
 */

struct DsaIndexerGeometry {
    std::int32_t heads     = 0; // index query heads, a multiple of 32
    std::int32_t head_dim  = 0; // index key, gate and pooled-key width
    std::int32_t pool      = 0; // tokens per pooled key
    std::int32_t top_pools = 0; // selection budget in pools
    float eps              = 0; // LayerNorm epsilon
};

/// One layer's index cache. Only the pooled keys are kept for the whole context; a token's own
/// key and gate are needed only until its pool completes, so they live in a short per-row ring:
///   k_recent     BF16 [head_dim, 64, rows]: the normalized index keys of table row r's 64 most
///                recently appended positions, position p at slot p & 63 of row r
///   z_recent     BF16 [head_dim, 64, rows]: their gate vectors, as given
///   pooled_pages BF16 [head_dim / pool, 64, 1, pages]: token-paged with the batch's page tables
///                (MlaSparseBatch) and the engine's page-major plane order, logical token p at
///                table[p >> 6], slot p & 63: pool p's key spread across its `pool` token slots,
///                `head_dim / pool` contiguous elements per slot (slot 4p + i holds key channels
///                32i..32i+31 for pool 4; a pool's key is one 256 B run inside a page).
/// rows covers every table row the batches name. A call whose first position p0 is not a pool
/// boundary reads the ring for positions pool * floor(p0 / pool) .. p0 - 1: those positions'
/// latest appends to the row, with fewer than 64 - pool + 1 newer positions appended to the row
/// since (a rollback to p0 rewinds at most 60 positions past it). A row's ring needs no clearing
/// between sequences: a sequence starting at position 0 reads nothing before it.
struct DsaIndexerCacheView {
    Tensor k_recent;
    Tensor z_recent;
    Tensor pooled_pages;
};

/**
 * Op: indexer update. Appends the call's columns to the index cache and completes their pools.
 * For column c (position p = positions[c], sequence c / width, table row r = table_rows[c /
 * width]):
 *
 *   k_t = BF16(LayerNorm(FP32(raw_k[:, c]); gamma, beta, eps))   (per channel), z_t = z[:, c]
 *   when the column completes a pool (p ≡ pool - 1 mod pool), the pooled key of pool p / pool
 *   at pooled_pages as in the family formula above, over its tokens' k and z: the call's own
 *   columns' values, and the ring's for tokens before the call;
 *   then k_recent/z_recent of row r at slot p & 63 hold k_t and z_t bit-for-bit, for the last 64
 *   columns of each sequence (earlier columns of a wider call are not needed by any later call).
 *
 * LayerNorm is the reference elementwise-affine layer norm: mean and biased variance over the
 * whole vector in FP32, (x - mean) / sqrt(var + eps) * gamma + beta.
 *
 * Logical shapes:
 *   raw_k, z BF16 [head_dim, columns] contiguous (columns = batch * width); gamma, beta BF16
 *   [head_dim]; ape FP32 [pool, head_dim] contiguous; the cache as in DsaIndexerCacheView;
 *   batch as in MlaSparseBatch.
 *
 * Supported domain:
 *   head_dim 128 and pool 4 (one warp reads a whole pooled key; the pool is a compile-time
 *   constant of the pooling kernel); heads a multiple of 32; eps > 0.
 *
 * Numeric:
 *   FP32 normalization, softmax and pooling; stored index keys and pooled keys round once to BF16;
 * z is stored bit-for-bit. A column's stored bytes (and the pooled keys of the pools it completes)
 *   depend only on its own inputs and its sequence's cache state, never on the call's other
 *   columns, width or batch; the same tokens appended in one call or several give the same
 *   stored bytes.
 *
 * Effects:
 *   Writes the ring slots of each sequence's last 64 positions and the pooled keys of the pools
 *   the call's positions complete; nothing else. Inputs unchanged. The caller owns commit and
 *   rollback policy (a rewritten position is simply overwritten).
 *
 * Workspace / Execution:
 *   none; two stream-ordered launches (pooling, then the ring append), no host synchronization.
 */
void dsa_indexer_update(const Tensor& raw_k, const Tensor& z, const Tensor& gamma,
                        const Tensor& beta, const Tensor& ape, const DsaIndexerCacheView& cache,
                        const MlaSparseBatch& batch, const DsaIndexerGeometry& geometry,
                        cudaStream_t stream);

/// Caller-owned transient capacity of dsa_indexer_select for every legal call of up to
/// `columns` columns at `max_context` (the score rows and selection state of one column group).
[[nodiscard]] std::size_t dsa_indexer_select_workspace_bytes(const DsaIndexerGeometry& geometry,
                                                             std::int32_t columns,
                                                             std::int32_t max_context);

/**
 * Op: indexer selection. Scores every complete pool the columns see and selects the top
 * `top_pools` per column (the family formula; ties to the lower pool id, ids ascending). A
 * column at position p sees the P = (p + 1) / pool complete pools 0..P-1; when P <= top_pools
 * its count is -1 (dense) and its pool slots are not written. The FP32 arithmetic of a score is
 * fixed so selections are reproducible: a warp step reads `pool`-slot groups of 8 pooled keys
 * and, per column, the heads' lane partials (each lane accumulates its four channel elements of
 * a head's dot in channel order) are summed in the xor-butterfly order over head groups of 4;
 * the ReLU and the per-head weight apply to each head's sum, the heads add in head order, and
 * the total is multiplied by `scale` once. Scores, and therefore selections, are bitwise
 * independent of the launch's slicing, and a column's selection depends only on its own inputs
 * and its sequence's pooled keys.
 *
 * Logical shapes:
 *   q BF16 [head_dim, heads, columns] contiguous; w BF16 [heads, columns] contiguous; cache and
 *   batch as in dsa_indexer_update; pools out I32 [top_pools, columns] (slot (k, c) at
 *   k + c * top_pools), counts out I32 [columns]. columns = batch * width > 0; every position in
 *   [0, max_context).
 *
 * Supported domain:
 *   as dsa_indexer_update; scale > 0. The registered geometry is heads 32, head_dim 128,
 *   pool 4, top_pools 512 (scale 1/sqrt(128)).
 *
 * Numeric:
 *   FP32 scores as fixed above; the selection is the exact top-k over the 64-bit (score,
 *   lower-id-first) order key (a 2048-bin histogram of the scores' bits [30:20] and an 8-bit
 *   radix select over the boundary bin, ops/common/exact_topk_select.cuh). Non-finite scores
 *   are rejected by validation of the inputs' domain, not by the kernel.
 *
 * Effects:
 *   Writes pools (only slots below a column's count) and counts completely; reads the cache;
 *   nothing else. Inputs and outputs pairwise non-overlapping; workspace is caller-owned
 *   transient scratch (the capacity query above).
 *
 * Execution:
 *   stream-ordered without host synchronization; positions are read on the device, so a captured
 *   call serves any device values of the same shapes.
 */
void dsa_indexer_select(const Tensor& q, const Tensor& w, const DsaIndexerCacheView& cache,
                        const MlaSparseBatch& batch, const DsaIndexerGeometry& geometry,
                        float scale, std::int32_t max_context, void* workspace,
                        std::size_t workspace_bytes, Tensor& pools, Tensor& counts,
                        cudaStream_t stream);

} // namespace infernix::ops
