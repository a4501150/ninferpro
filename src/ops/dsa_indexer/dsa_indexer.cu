// The pooled-key indexer of sparse latent attention (include/infernix/ops/dsa_indexer.h): the
// LayerNorm-fused append and pool completion, and the ReLU-of-dot scoring whose exact top-pool
// selection runs on the shared exact top-k facility (ops/common/exact_topk_select.h).

#include "infernix/ops/dsa_indexer.h"

#include "core/device.h"
#include "ops/common/exact_topk_select.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

// The registered geometry: 32 index heads of 128 channels, pools of 4 tokens, a 512-pool budget.
constexpr int kIdxHeadDim = 128;

// Scoring: four warps per CTA, each warp owns one page (16 pools) per step, scoring up to
// kScoreTileColumns query columns with m16n8k16 BF16 Tensor Core MMAs (FP32 accumulation).
constexpr int kScoreThreads     = 128;
constexpr int kScoreWarps       = kScoreThreads / 32;
constexpr int kScoreWarpPools   = 16; // one KV page (64 tokens / pool 4) per warp
constexpr int kScoreStepPools   = kScoreWarps * kScoreWarpPools; // 64 pools per CTA step
constexpr int kScoreTileColumns = 4;                             // columns of one row per CTA tile
constexpr int kScoreMaxSlices   = 256;
constexpr int kScoreGroupCols   = 64; // columns per select pass

static_assert(kScoreWarpPools * 4 == kPagedKVPageSize);

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("dsa indexer: ") + message); }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

int ceil_div(std::int64_t a, std::int64_t b) { return static_cast<int>((a + b - 1) / b); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
}

__device__ __forceinline__ int swizzled_128(int row, int chunk) {
    return row * kIdxHeadDim + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ unsigned smem_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ bf16* pooled_element(bf16* pooled, const std::int32_t* table, int token,
                                                int slot_width, int lane) {
    const int page = table[token >> kPagedKVPageShift];
    return pooled + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + lane;
}

// ------------------------------------------------------------------------- append and pooling

// The recent-token ring: a table row's last kRecent positions, slot position & (kRecent - 1).
constexpr int kRecent = 64;
static_assert((kRecent & (kRecent - 1)) == 0, "a power-of-two ring");

// One thread per channel of a CTA of head_dim threads: channel d of the column's normalized
// index key, in FP32 (every thread of the CTA calls it).
__device__ __forceinline__ float index_key(const bf16* __restrict__ raw, const bf16* __restrict__ gamma,
                                           const bf16* __restrict__ beta, float eps, int head_dim,
                                           float* spot) {
    const int d = static_cast<int>(threadIdx.x), lane = d % 32, warp = d / 32;
    const float x = __bfloat162float(raw[d]);
    float v       = warp_sum(x);
    if (lane == 0) { spot[warp] = v; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < head_dim / 32; ++w) { total += spot[w]; }
    __syncthreads();
    const float mean     = total / static_cast<float>(head_dim);
    const float centered = x - mean;
    v                    = warp_sum(centered * centered);
    if (lane == 0) { spot[warp] = v; }
    __syncthreads();
    total = 0.0F;
    for (int w = 0; w < head_dim / 32; ++w) { total += spot[w]; }
    __syncthreads();
    return centered * rsqrtf(total / static_cast<float>(head_dim) + eps) *
               __bfloat162float(gamma[d]) +
           __bfloat162float(beta[d]);
}

// One CTA per column, one thread per channel: the pooled key of the pool this column completes.
// The pool's tokens inside the call come from the call's inputs (their keys normalized here, as
// the append stores them); earlier ones from the row's recent ring, which the append of this
// call has not touched yet. The pool is a compile-time constant so its weights stay in registers.
template <int pool>
__global__ void indexer_pool_kernel(const bf16* __restrict__ raw_k, const bf16* __restrict__ z,
                                    const bf16* __restrict__ gamma, const bf16* __restrict__ beta,
                                    float eps, const bf16* __restrict__ k_recent,
                                    const bf16* __restrict__ z_recent, const float* __restrict__ ape,
                                    const std::int32_t* __restrict__ positions,
                                    const std::int32_t* __restrict__ tables, int table_stride,
                                    const std::int32_t* __restrict__ table_rows, int width,
                                    int head_dim, bf16* __restrict__ pooled) {
    __shared__ float spot[32];
    const int slot_width = head_dim / pool;
    const int column = static_cast<int>(blockIdx.x), d = static_cast<int>(threadIdx.x);
    const int p = positions[column];
    if ((p + 1) % pool != 0) { return; }
    const int b = column / width, row = table_rows[b];
    const int call_first = positions[b * width]; // the sequence's positions are consecutive
    const int first      = p + 1 - pool;
    float kv[pool], zv[pool];
#pragma unroll
    for (int i = 0; i < pool; ++i) {
        const int token = first + i;
        if (token >= call_first) {
            const std::int64_t c = static_cast<std::int64_t>(b) * width + (token - call_first);
            kv[i] = __bfloat162float(__float2bfloat16_rn(
                index_key(raw_k + c * head_dim, gamma, beta, eps, head_dim, spot)));
            zv[i] = __bfloat162float(z[d + c * head_dim]);
        } else {
            const std::int64_t at =
                (static_cast<std::int64_t>(row) * kRecent + (token & (kRecent - 1))) * head_dim + d;
            kv[i] = __bfloat162float(k_recent[at]);
            zv[i] = __bfloat162float(z_recent[at]);
        }
    }
    float e[pool];
    float total = 0.0F;
#pragma unroll
    for (int i = 0; i < pool; ++i) {
        e[i] = expf(zv[i] + ape[i * head_dim + d]);
        total += e[i];
    }
    float key = 0.0F;
#pragma unroll
    for (int i = 0; i < pool; ++i) { key += e[i] / total * kv[i]; }
    const std::int32_t* table =
        tables + static_cast<std::int64_t>(row) * table_stride;
    const int token                                                   = first + d / slot_width;
    *pooled_element(pooled, table, token, slot_width, d % slot_width) = __float2bfloat16_rn(key);
}

// One CTA per column of each sequence's last kRecent columns (the others would be overwritten in
// the ring by later columns of the same call), one thread per channel: the normalized index key
// and the gate vector into the row's recent ring.
__global__ void indexer_append_kernel(const bf16* __restrict__ raw_k, const bf16* __restrict__ z,
                                      const bf16* __restrict__ gamma, const bf16* __restrict__ beta,
                                      float eps, const std::int32_t* __restrict__ positions,
                                      const std::int32_t* __restrict__ table_rows, int width,
                                      int head_dim, bf16* __restrict__ k_recent,
                                      bf16* __restrict__ z_recent) {
    __shared__ float spot[32];
    const int column = static_cast<int>(blockIdx.x), d = static_cast<int>(threadIdx.x);
    if (column % width < width - kRecent) { return; }
    const int p = positions[column];
    const std::int64_t slot =
        (static_cast<std::int64_t>(table_rows[column / width]) * kRecent + (p & (kRecent - 1))) *
        head_dim;
    const std::int64_t c = column;
    const float y        = index_key(raw_k + c * head_dim, gamma, beta, eps, head_dim, spot);
    k_recent[slot + d]   = __float2bfloat16_rn(y);
    z_recent[slot + d]   = z[d + c * head_dim];
}

// ------------------------------------------------------------------------------------- scoring

// Grid (slices, row tiles of up to 4 columns). Each CTA stages its tile's queries in swizzled
// BF16 shared memory. Each warp independently loads one KV page (16 contiguous pooled keys of
// 128 BF16 channels) into its private shared-memory slab and computes Q_c * K_page^T using
// m16n8k16 BF16 MMAs with FP32 accumulation, then applies ReLU, multiplies by w[h, c], and reduces
// across heads in fixed group order. A column's score for any pool b uses identical MMA K-steps
// and head-reduction order at every call width and batch layout (strict bitwise column invariance).
__global__ void __launch_bounds__(kScoreThreads)
    indexer_score_kernel(const bf16* __restrict__ q, const bf16* __restrict__ w, int heads,
                         const bf16* __restrict__ pooled, const std::int32_t* __restrict__ tables,
                         int table_stride, const std::int32_t* __restrict__ table_rows,
                         const std::int32_t* __restrict__ positions, int width, int pool, int top,
                         float scale, int group_begin, int group_columns, int tile_base,
                         float* __restrict__ scores, int score_stride) {
    constexpr int kChunks128 = kIdxHeadDim / 8; // 16 uint4 chunks per 128-BF16 row
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_s = reinterpret_cast<bf16*>(smem); // [kScoreTileColumns][heads][128] swizzled
    auto* k_s = q_s + static_cast<std::size_t>(kScoreTileColumns) * heads * kIdxHeadDim; // [64][128]
    auto* w_s = reinterpret_cast<float*>(k_s + kScoreStepPools * kIdxHeadDim); // [4][heads]
    auto* column_pools = reinterpret_cast<int*>(w_s + kScoreTileColumns * heads); // [4]

    const int tiles_per_row = (width + kScoreTileColumns - 1) / kScoreTileColumns;
    const int tile          = tile_base + static_cast<int>(blockIdx.y);
    const int row = tile / tiles_per_row, col_in = (tile % tiles_per_row) * kScoreTileColumns;
    const int c0   = row * width + col_in;
    const int nc   = min(min(kScoreTileColumns, width - col_in), group_begin + group_columns - c0);
    const int skip = max(0, group_begin - c0);
    if (nc <= skip) { return; }

    const int tid = static_cast<int>(threadIdx.x);
    if (tid < nc) { column_pools[tid] = (positions[c0 + tid] + 1) / pool; }
    __syncthreads();

    int tile_pools = 0;
    for (int c = skip; c < nc; ++c) { tile_pools = max(tile_pools, column_pools[c]); }
    const int warp = tid >> 5, lane = tid & 31;
    if (tile_pools <= top) { return; }
    const int steps = (tile_pools + kScoreStepPools - 1) / kScoreStepPools;
    if (static_cast<int>(blockIdx.x) >= steps) { return; }

    // Stage only the active columns [skip, nc) of Q (swizzled BF16) and w (FP32).
    const int live_cols    = nc - skip;
    const int total_q_vecs = live_cols * heads * kChunks128;
    for (int i = tid; i < total_q_vecs; i += kScoreThreads) {
        const int ch = i & (kChunks128 - 1);
        const int hr = i >> 4;
        const int c  = skip + hr / heads;
        const int h  = hr % heads;
        const bf16* src = q + static_cast<std::int64_t>(c0 + c) * heads * kIdxHeadDim +
                          static_cast<std::int64_t>(h) * kIdxHeadDim + ch * 8;
        *reinterpret_cast<uint4*>(q_s + static_cast<std::size_t>(c) * heads * kIdxHeadDim +
                                  swizzled_128(h, ch)) = *reinterpret_cast<const uint4*>(src);
    }
    for (int i = tid; i < live_cols * heads; i += kScoreThreads) {
        const int c = skip + i / heads, h = i % heads;
        w_s[c * heads + h] = __bfloat162float(w[h + static_cast<std::int64_t>(c0 + c) * heads]);
    }
    __syncthreads();

    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[row]) * table_stride;
    bf16* const warp_k_s      = k_s + static_cast<std::size_t>(warp) * kScoreWarpPools * kIdxHeadDim;
    const int grp             = lane >> 2; // 0..7 -> rows grp and grp + 8
    const int tid4            = lane & 3;  // 0..3 -> pools 2*tid4 and 2*tid4 + 1
    const int b_row_local     = 8 * (lane >> 4) + (lane & 7);
    const int b_ch_off        = (lane >> 3) & 1;
    const int a_row_off       = lane & 15;
    const int a_ch_off        = lane >> 4;
    const bool full_tile      = (skip == 0 && nc == kScoreTileColumns);

    for (int step = static_cast<int>(blockIdx.x); step < steps;
         step += static_cast<int>(gridDim.x)) {
        const int page_pool_base = step * kScoreStepPools + warp * kScoreWarpPools;
        if (page_pool_base >= tile_pools) { continue; }
        const int page_pools = min(kScoreWarpPools, tile_pools - page_pool_base);
        const int page       = table[page_pool_base >> 4];
        const bf16* page_ptr =
            pooled + static_cast<std::int64_t>(page) * (kScoreWarpPools * kIdxHeadDim);

        // Warp-private coalesced load of 16 pooled keys (1 KV page) into swizzled shared memory.
#pragma unroll
        for (int i = lane; i < kScoreWarpPools * kChunks128; i += 32) {
            const int lp = i >> 4;
            const int ch = i & 15;
            *reinterpret_cast<uint4*>(warp_k_s + swizzled_128(lp, ch)) =
                lp < page_pools ? *reinterpret_cast<const uint4*>(page_ptr + lp * kIdxHeadDim + ch * 8)
                                : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();

        if (full_tile) {
            float sum0[kScoreTileColumns][2] = {};
            float sum1[kScoreTileColumns][2] = {};
            for (int hg = 0; hg < heads; hg += 16) {
                float acc[kScoreTileColumns][2][4] = {};
#pragma unroll
                for (int ks = 0; ks < 8; ++ks) {
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4(b0, b1, b2, b3,
                                smem_address(warp_k_s + swizzled_128(b_row_local, 2 * ks + b_ch_off)));
#pragma unroll
                    for (int c = 0; c < kScoreTileColumns; ++c) {
                        unsigned a0, a1, a2, a3;
                        ldmatrix_x4(a0, a1, a2, a3,
                                    smem_address(q_s + static_cast<std::size_t>(c) * heads * kIdxHeadDim +
                                                 swizzled_128(hg + a_row_off, 2 * ks + a_ch_off)));
                        mma_bf16(acc[c][0][0], acc[c][0][1], acc[c][0][2], acc[c][0][3], a0, a1, a2,
                                 a3, b0, b1);
                        mma_bf16(acc[c][1][0], acc[c][1][1], acc[c][1][2], acc[c][1][3], a0, a1, a2,
                                 a3, b2, b3);
                    }
                }
#pragma unroll
                for (int c = 0; c < kScoreTileColumns; ++c) {
                    const float w0 = w_s[c * heads + hg + grp];
                    const float w1 = w_s[c * heads + hg + grp + 8];
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        sum0[c][nt] += w0 * fmaxf(acc[c][nt][0], 0.0F);
                        sum0[c][nt] += w1 * fmaxf(acc[c][nt][2], 0.0F);
                        sum1[c][nt] += w0 * fmaxf(acc[c][nt][1], 0.0F);
                        sum1[c][nt] += w1 * fmaxf(acc[c][nt][3], 0.0F);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < kScoreTileColumns; ++c) {
                const int blocks = column_pools[c];
                if (page_pool_base >= blocks) { continue; }
                float* const out_row =
                    scores + static_cast<std::size_t>(c0 + c - group_begin) * score_stride;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    float s0 = sum0[c][nt];
                    float s1 = sum1[c][nt];
                    s0 += __shfl_xor_sync(0xFFFFFFFFU, s0, 4);
                    s0 += __shfl_xor_sync(0xFFFFFFFFU, s0, 8);
                    s0 += __shfl_xor_sync(0xFFFFFFFFU, s0, 16);
                    s1 += __shfl_xor_sync(0xFFFFFFFFU, s1, 4);
                    s1 += __shfl_xor_sync(0xFFFFFFFFU, s1, 8);
                    s1 += __shfl_xor_sync(0xFFFFFFFFU, s1, 16);
                    if (grp == 0) {
                        const int p0 = page_pool_base + 8 * nt + 2 * tid4;
                        if (p0 < blocks) { out_row[p0] = s0 * scale; }
                        if (p0 + 1 < blocks) { out_row[p0 + 1] = s1 * scale; }
                    }
                }
            }
        } else {
            for (int c = skip; c < nc; ++c) {
                const int blocks = column_pools[c];
                if (page_pool_base >= blocks) { continue; }
                float sum0[2] = {0.0F, 0.0F};
                float sum1[2] = {0.0F, 0.0F};
                for (int hg = 0; hg < heads; hg += 16) {
                    float acc[2][4] = {};
#pragma unroll
                    for (int ks = 0; ks < 8; ++ks) {
                        unsigned b0, b1, b2, b3;
                        ldmatrix_x4(b0, b1, b2, b3,
                                    smem_address(warp_k_s +
                                                 swizzled_128(b_row_local, 2 * ks + b_ch_off)));
                        unsigned a0, a1, a2, a3;
                        ldmatrix_x4(a0, a1, a2, a3,
                                    smem_address(q_s +
                                                 static_cast<std::size_t>(c) * heads * kIdxHeadDim +
                                                 swizzled_128(hg + a_row_off, 2 * ks + a_ch_off)));
                        mma_bf16(acc[0][0], acc[0][1], acc[0][2], acc[0][3], a0, a1, a2, a3, b0,
                                 b1);
                        mma_bf16(acc[1][0], acc[1][1], acc[1][2], acc[1][3], a0, a1, a2, a3, b2,
                                 b3);
                    }
                    const float w0 = w_s[c * heads + hg + grp];
                    const float w1 = w_s[c * heads + hg + grp + 8];
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        sum0[nt] += w0 * fmaxf(acc[nt][0], 0.0F);
                        sum0[nt] += w1 * fmaxf(acc[nt][2], 0.0F);
                        sum1[nt] += w0 * fmaxf(acc[nt][1], 0.0F);
                        sum1[nt] += w1 * fmaxf(acc[nt][3], 0.0F);
                    }
                }
                float* const out_row =
                    scores + static_cast<std::size_t>(c0 + c - group_begin) * score_stride;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    float s0 = sum0[nt];
                    float s1 = sum1[nt];
                    s0 += __shfl_xor_sync(0xFFFFFFFFU, s0, 4);
                    s0 += __shfl_xor_sync(0xFFFFFFFFU, s0, 8);
                    s0 += __shfl_xor_sync(0xFFFFFFFFU, s0, 16);
                    s1 += __shfl_xor_sync(0xFFFFFFFFU, s1, 4);
                    s1 += __shfl_xor_sync(0xFFFFFFFFU, s1, 8);
                    s1 += __shfl_xor_sync(0xFFFFFFFFU, s1, 16);
                    if (grp == 0) {
                        const int p0 = page_pool_base + 8 * nt + 2 * tid4;
                        if (p0 < blocks) { out_row[p0] = s0 * scale; }
                        if (p0 + 1 < blocks) { out_row[p0 + 1] = s1 * scale; }
                    }
                }
            }
        }
        __syncwarp();
    }
}

// ------------------------------------------------------------------------------------------ host

void require_geometry(const DsaIndexerGeometry& g) {
    require(g.heads > 0 && g.heads % 32 == 0 && g.heads <= 64,
            "heads must be a positive multiple of 32 of at most 64");
    require(g.head_dim > 0 && g.head_dim % 32 == 0, "head_dim must be a positive multiple of 32");
    require(g.pool == 4 && g.head_dim == 128,
            "pool must be 4 and head_dim 128 (one warp reads a whole pooled key)");
    require(g.top_pools > 0, "top_pools must be positive");
    require(g.eps > 0.0F, "eps must be positive");
}

void require_batch(const MlaSparseBatch& b, int columns) {
    require(b.batch > 0 && b.width > 0 && b.batch * b.width == columns, "batch and width disagree");
    require(contiguous(b.positions, DType::I32) && b.positions.numel() == columns,
            "positions must be I32 [batch * width]");
    require(contiguous(b.table_rows, DType::I32) && b.table_rows.numel() == b.batch,
            "table_rows must be I32 [batch]");
    require(contiguous(b.block_tables, DType::I32) && b.block_tables.ne[0] > 0,
            "block_tables must be I32 [pages per row, rows]");
}

void require_cache(const DsaIndexerCacheView& cache, const DsaIndexerGeometry& g, int slot_width) {
    require(contiguous(cache.k_recent, DType::BF16) && cache.k_recent.ne[0] == g.head_dim &&
                cache.k_recent.ne[1] == kRecent && cache.k_recent.ne[2] > 0 &&
                cache.k_recent.ne[3] == 1,
            "k_recent must be BF16 [head_dim, 64, rows]");
    require(contiguous(cache.z_recent, DType::BF16) && cache.z_recent.ne[0] == g.head_dim &&
                cache.z_recent.ne[1] == kRecent && cache.z_recent.ne[2] == cache.k_recent.ne[2] &&
                cache.z_recent.ne[3] == 1,
            "z_recent must be BF16 [head_dim, 64, rows]");
    require(contiguous(cache.pooled_pages, DType::BF16) && cache.pooled_pages.ne[0] == slot_width &&
                cache.pooled_pages.ne[1] == kPagedKVPageSize && cache.pooled_pages.ne[2] == 1,
            "pooled_pages must be BF16 [head_dim / pool, 64, 1, pages]");
}

} // namespace

void dsa_indexer_update(const Tensor& raw_k, const Tensor& z, const Tensor& gamma,
                        const Tensor& beta, const Tensor& ape, const DsaIndexerCacheView& cache,
                        const MlaSparseBatch& batch, const DsaIndexerGeometry& geometry,
                        cudaStream_t stream) {
    require_geometry(geometry);
    const int columns = batch.batch * batch.width;
    require(columns > 0, "the call needs at least one column");
    const int slot_width = geometry.head_dim / geometry.pool;
    require_batch(batch, columns);
    require(contiguous(raw_k, DType::BF16) && raw_k.ne[0] == geometry.head_dim &&
                raw_k.ne[1] == columns,
            "raw_k must be BF16 [head_dim, columns]");
    require(contiguous(z, DType::BF16) && z.ne[0] == geometry.head_dim && z.ne[1] == columns,
            "z must be BF16 [head_dim, columns]");
    require(contiguous(gamma, DType::BF16) && gamma.numel() == geometry.head_dim &&
                contiguous(beta, DType::BF16) && beta.numel() == geometry.head_dim,
            "gamma and beta must be BF16 [head_dim]");
    require(contiguous(ape, DType::FP32) && ape.ne[0] == geometry.head_dim &&
                ape.ne[1] == geometry.pool,
            "ape must be FP32 [pool, head_dim]");
    require_cache(cache, geometry, slot_width);
    const auto* tables = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int stride   = batch.block_tables.ne[0];
    const auto* rows   = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* pos    = static_cast<const std::int32_t*>(batch.positions.data);
    const auto* rk = static_cast<const bf16*>(raw_k.data);
    const auto* zz = static_cast<const bf16*>(z.data);
    const auto* g  = static_cast<const bf16*>(gamma.data);
    const auto* bt = static_cast<const bf16*>(beta.data);
    // Pooling first: it reads the ring's earlier tokens before the append rewrites their slots.
    indexer_pool_kernel<4><<<dim3(columns), dim3(geometry.head_dim), 0, stream>>>(
        rk, zz, g, bt, geometry.eps, static_cast<const bf16*>(cache.k_recent.data),
        static_cast<const bf16*>(cache.z_recent.data), static_cast<const float*>(ape.data), pos,
        tables, stride, rows, batch.width, geometry.head_dim,
        static_cast<bf16*>(cache.pooled_pages.data));
    CUDA_CHECK(cudaGetLastError());
    indexer_append_kernel<<<dim3(columns), dim3(geometry.head_dim), 0, stream>>>(
        rk, zz, g, bt, geometry.eps, pos, rows, batch.width, geometry.head_dim,
        static_cast<bf16*>(cache.k_recent.data), static_cast<bf16*>(cache.z_recent.data));
    CUDA_CHECK(cudaGetLastError());
}

namespace {

struct SelectLayout {
    int group = 0, stride = 0;
    std::size_t scores = 0, total = 0;
};

SelectLayout select_layout(const DsaIndexerGeometry& geometry, std::int32_t columns,
                           std::int32_t max_context) {
    SelectLayout l;
    l.group  = std::min<int>(columns, kScoreGroupCols);
    l.stride = (max_context / geometry.pool + 4 + 31) / 32 * 32;
    l.scores = align256(sizeof(float) * static_cast<std::size_t>(l.group) * l.stride);
    l.total  = l.scores + detail::topk_select_state_bytes(l.group);
    return l;
}

} // namespace

std::size_t dsa_indexer_select_workspace_bytes(const DsaIndexerGeometry& geometry,
                                               std::int32_t columns, std::int32_t max_context) {
    require_geometry(geometry);
    require(columns > 0 && max_context > 0, "columns and max_context must be positive");
    return select_layout(geometry, columns, max_context).total;
}

void dsa_indexer_select(const Tensor& q, const Tensor& w, const DsaIndexerCacheView& cache,
                        const MlaSparseBatch& batch, const DsaIndexerGeometry& geometry,
                        float scale, std::int32_t max_context, void* workspace,
                        std::size_t workspace_bytes, Tensor& pools, Tensor& counts,
                        cudaStream_t stream) {
    require_geometry(geometry);
    const int columns = batch.batch * batch.width;
    require(columns > 0, "the call needs at least one column");
    const int slot_width = geometry.head_dim / geometry.pool;
    require_batch(batch, columns);
    require(contiguous(q, DType::BF16) && q.ne[0] == geometry.head_dim &&
                q.ne[1] == geometry.heads && q.ne[2] == columns,
            "q must be BF16 [head_dim, heads, columns]");
    require(contiguous(w, DType::BF16) && w.ne[0] == geometry.heads && w.ne[1] == columns,
            "w must be BF16 [heads, columns]");
    require_cache(cache, geometry, slot_width);
    require(contiguous(pools, DType::I32) && pools.ne[0] == geometry.top_pools &&
                pools.ne[1] == columns,
            "pools must be I32 [top_pools, columns]");
    require(contiguous(counts, DType::I32) && counts.numel() == columns,
            "counts must be I32 [columns]");
    require(scale > 0.0F && max_context > 0, "scale and max_context must be positive");
    const SelectLayout l = select_layout(geometry, columns, max_context);
    require(workspace != nullptr && workspace_bytes >= l.total, "workspace is too small");
    auto* scores = static_cast<float*>(workspace);
    auto* state =
        reinterpret_cast<std::uint32_t*>(static_cast<unsigned char*>(workspace) + l.scores);

    // Dynamic shared memory of the score kernel: swizzled BF16 queries, per-warp pooled-key slabs,
    // head weights, and column pool counts.
    const std::size_t smem = sizeof(bf16) * kScoreTileColumns * geometry.heads * kIdxHeadDim +
                             sizeof(bf16) * kScoreStepPools * kIdxHeadDim +
                             sizeof(float) * kScoreTileColumns * geometry.heads +
                             sizeof(int) * kScoreTileColumns;
    static std::atomic<std::size_t> reserved{0};
    if (smem > 48 * 1024 && smem > reserved.load(std::memory_order_acquire)) {
        int device = 0, limit = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        require(smem <= static_cast<std::size_t>(limit),
                "heads need more shared memory than a CTA has");
        CUDA_CHECK(cudaFuncSetAttribute(indexer_score_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
        std::size_t seen = reserved.load(std::memory_order_relaxed);
        while (seen < smem &&
               !reserved.compare_exchange_weak(seen, smem, std::memory_order_acq_rel)) {}
    }

    const auto* qp          = static_cast<const bf16*>(q.data);
    const auto* wp          = static_cast<const bf16*>(w.data);
    const auto* pooled      = static_cast<const bf16*>(cache.pooled_pages.data);
    const auto* tables      = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int table_stride  = batch.block_tables.ne[0];
    const auto* rows        = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* pos         = static_cast<const std::int32_t*>(batch.positions.data);
    const int max_pools     = max_context / geometry.pool;
    const int tiles_per_row = ceil_div(batch.width, kScoreTileColumns);
    const int slices =
        std::clamp(ceil_div(static_cast<std::int64_t>(max_pools), kScoreStepPools), 1,
                   kScoreMaxSlices);
    auto* pools_out  = reinterpret_cast<std::int32_t*>(pools.data);
    auto* counts_out = reinterpret_cast<std::int32_t*>(counts.data);
    for (int begin = 0; begin < columns; begin += l.group) {
        const int n = std::min<int>(l.group, columns - begin);
        const int tile_lo =
            (begin / batch.width) * tiles_per_row + (begin % batch.width) / kScoreTileColumns;
        const int last = begin + n - 1;
        const int tile_hi =
            (last / batch.width) * tiles_per_row + (last % batch.width) / kScoreTileColumns;
        indexer_score_kernel<<<dim3(static_cast<unsigned>(slices),
                                    static_cast<unsigned>(tile_hi - tile_lo + 1)),
                               dim3(kScoreThreads), smem, stream>>>(
            qp, wp, geometry.heads, pooled, tables, table_stride, rows, pos, batch.width,
            geometry.pool, geometry.top_pools, scale, begin, n, tile_lo, scores, l.stride);
        CUDA_CHECK(cudaGetLastError());
        detail::exact_topk_select(pos, geometry.pool, geometry.top_pools, begin, n, scores,
                                  l.stride, state, pools_out, counts_out, max_pools, false, stream);
    }
}

} // namespace infernix::ops
