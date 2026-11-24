// The pooled-key indexer of sparse latent attention (include/infernix/ops/dsa_indexer.h): the
// LayerNorm-fused append and pool completion, and the ReLU-of-dot scoring whose exact top-pool
// selection runs on the shared exact top-k facility (ops/common/exact_topk_select.h).

#include "infernix/ops/dsa_indexer.h"

#include "core/device.h"
#include "ops/common/exact_topk_select.h"
#include "ops/common/memory.cuh"
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

// Scoring: a warp step reads 8 pooled keys and, per column, reduce-scatters the 32 lane partials
// of its dots (8 pools x 4 heads) once per head group of 4.
constexpr int kScoreThreads     = 256;
constexpr int kScoreWarps       = kScoreThreads / 32;
constexpr int kScoreStepPools   = 8;
constexpr int kScoreHeadSlots   = 4; // heads per reduce-scatter round
constexpr int kScoreTileColumns = 4; // columns of one row per CTA tile
constexpr int kScoreMaxSlices   = 128;
constexpr int kScoreGroupCols   = 64; // columns per select pass

static_assert(kScoreStepPools * kScoreHeadSlots == 32);

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

// One stage of the reduce-scatter: lanes whose `Half` bit is clear keep values [0, Half), the
// others [Half, 2 Half); each kept value adds its partner lane's copy (own + partner, as
// warp_sum's `v += __shfl_xor_sync(v, Half)`).
template <int Half>
__device__ __forceinline__ void reduce_scatter_stage(float (&v)[32], int lane) {
    const bool upper = (lane & Half) != 0;
#pragma unroll
    for (int i = 0; i < Half; ++i) {
        const float keep = upper ? v[i + Half] : v[i];
        const float send = upper ? v[i] : v[i + Half];
        v[i]             = keep + __shfl_xor_sync(0xFFFFFFFFU, send, Half);
    }
}

// Lane l returns the 32-lane sum of its value l, bitwise equal to warp_sum's result for that
// value (31 shuffles for 32 sums instead of 160).
__device__ __forceinline__ float reduce_scatter_32(float (&v)[32], int lane) {
    reduce_scatter_stage<16>(v, lane);
    reduce_scatter_stage<8>(v, lane);
    reduce_scatter_stage<4>(v, lane);
    reduce_scatter_stage<2>(v, lane);
    reduce_scatter_stage<1>(v, lane);
    return v[0];
}

// The pooled key's element for token slot `token` (of pool `token / pool`), lane coordinate
// `lane`: channel `token_in_pool * slot_width + lane` of the key.
__device__ __forceinline__ const bf16* pooled_element_const(const bf16* pooled,
                                                            const std::int32_t* table, int token,
                                                            int slot_width, int lane) {
    const int page = table[token >> kPagedKVPageShift];
    return pooled + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + lane;
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

// Grid (slices, row tiles of up to 4 columns). A warp step reads 8 pooled keys once and scores
// them for every column of its tile; the heads run in groups of 4, one reduce-scatter each, and
// the four lane totals of a pool add by two xor shuffles. A column's scores are bitwise
// independent of the slicing. The score of pool b of column c is
//   scale * sum_h w[h, c] * relu(<q[:, h, c], key_b>)   (FP32; heads added in head order).
__global__ void __launch_bounds__(kScoreThreads)
    indexer_score_kernel(const bf16* __restrict__ q, const bf16* __restrict__ w, int heads,
                         const bf16* __restrict__ pooled, const std::int32_t* __restrict__ tables,
                         int table_stride, const std::int32_t* __restrict__ table_rows,
                         const std::int32_t* __restrict__ positions, int width, int pool, int top,
                         float scale, int group_begin, int group_columns, int tile_base,
                         float* __restrict__ scores, int score_stride) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* qs = reinterpret_cast<float4*>(smem); // [tile columns][heads][32]: lane l, q[h][l + 32 j]
    float* w_s =
        reinterpret_cast<float*>(qs + kScoreTileColumns * heads * 32); // [tile columns][heads]
    int* column_pools = reinterpret_cast<int*>(w_s + kScoreTileColumns * heads); // [tile columns]

    const int tiles_per_row = (width + kScoreTileColumns - 1) / kScoreTileColumns;
    const int tile          = tile_base + static_cast<int>(blockIdx.y);
    const int row = tile / tiles_per_row, col_in = (tile % tiles_per_row) * kScoreTileColumns;
    const int c0 = group_begin + row * width + col_in;
    const int nc = min(min(kScoreTileColumns, width - col_in), group_begin + group_columns - c0);
    if (nc <= 0) { return; }
    const int slot_width = kIdxHeadDim / pool;

    for (int i = static_cast<int>(threadIdx.x); i < nc * heads * kIdxHeadDim; i += kScoreThreads) {
        const int c = i / (heads * kIdxHeadDim), r = i % (heads * kIdxHeadDim), h = r / kIdxHeadDim;
        const int d = r % kIdxHeadDim;
        const float value =
            __bfloat162float(q[d + static_cast<std::int64_t>(h) * kIdxHeadDim +
                               static_cast<std::int64_t>(c0 + c) * kIdxHeadDim * heads]);
        reinterpret_cast<float*>(&qs[(c * heads + h) * 32 + d % slot_width])[d / slot_width] =
            value;
    }
    for (int i = static_cast<int>(threadIdx.x); i < nc * heads; i += kScoreThreads) {
        const int c = i / heads, h = i % heads;
        w_s[c * heads + h] = __bfloat162float(w[h + static_cast<std::int64_t>(c0 + c) * heads]);
    }
    if (static_cast<int>(threadIdx.x) < nc) {
        column_pools[threadIdx.x] = (positions[c0 + static_cast<int>(threadIdx.x)] + 1) / pool;
    }
    __syncthreads();
    int tile_pools = 0;
    for (int c = 0; c < nc; ++c) { tile_pools = max(tile_pools, column_pools[c]); }
    const int steps = (tile_pools + kScoreStepPools - 1) / kScoreStepPools;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int first_step = static_cast<int>(blockIdx.x) * kScoreWarps + warp;
    if (tile_pools <= top || first_step >= steps) { return; }
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[row]) * table_stride;

    for (int step = first_step; step < steps; step += static_cast<int>(gridDim.x) * kScoreWarps) {
        const int b0 = step * kScoreStepPools;
        float key[kScoreStepPools][kScoreHeadSlots];
#pragma unroll
        for (int bb = 0; bb < kScoreStepPools; ++bb) {
            const int b = b0 + bb;
            if (b >= tile_pools) {
#pragma unroll
                for (int j = 0; j < kScoreHeadSlots; ++j) { key[bb][j] = 0.0F; }
                continue;
            }
#pragma unroll
            for (int j = 0; j < kScoreHeadSlots; ++j) {
                // The key's channels this lane owns: slot j of the pool's token run.
                key[bb][j] = __bfloat162float(
                    *pooled_element_const(pooled, table, pool * b + j, slot_width, lane));
            }
        }
        for (int c = 0; c < nc; ++c) {
            const int blocks = column_pools[c];
            if (b0 >= blocks) { continue; } // warp-uniform: this column sees none of the step
            float total = 0.0F;
            // Head groups of 4; lane 4 bb + h ends each round with pool b0 + bb's head dot.
#pragma unroll
            for (int g = 0; g < heads / kScoreHeadSlots; ++g) {
                float4 qv[kScoreHeadSlots];
#pragma unroll
                for (int h = 0; h < kScoreHeadSlots; ++h) {
                    qv[h] = qs[(c * heads + 4 * g + h) * 32 + lane];
                }
                float v[32];
#pragma unroll
                for (int bb = 0; bb < kScoreStepPools; ++bb) {
#pragma unroll
                    for (int h = 0; h < kScoreHeadSlots; ++h) {
                        float dot = 0.0F;
                        dot += qv[h].x * key[bb][0];
                        dot += qv[h].y * key[bb][1];
                        dot += qv[h].z * key[bb][2];
                        dot += qv[h].w * key[bb][3];
                        v[bb * kScoreHeadSlots + h] = dot;
                    }
                }
                const float relu = fmaxf(reduce_scatter_32(v, lane), 0.0F);
                total += w_s[c * heads + 4 * g + lane % kScoreHeadSlots] * relu;
            }
            // The pool's four lane totals (one per head residue) add; lane 4 bb writes pool b0 +
            // bb.
            total += __shfl_xor_sync(0xFFFFFFFFU, total, 1);
            total += __shfl_xor_sync(0xFFFFFFFFU, total, 2);
            const int b = b0 + lane / kScoreHeadSlots;
            if (lane % kScoreHeadSlots == 0 && b < blocks) {
                // The group-relative column of this tile entry.
                scores[static_cast<std::size_t>(row * width + col_in + c) * score_stride + b] =
                    total * scale;
            }
        }
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

    // Dynamic shared memory of the score kernel: the staged queries, the weights, the column
    // pool counts.
    const std::size_t smem = sizeof(float4) * kScoreTileColumns * geometry.heads * 32 +
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
        std::clamp(ceil_div(static_cast<std::int64_t>(max_pools), kScoreWarps * kScoreStepPools), 1,
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
