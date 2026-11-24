// Sparse multi-head latent attention over a paged shared latent cache
// (include/infernix/ops/mla_sparse_attention.h): the latent append, the absorbed query and output
// projections, and the split attention with fused output projection.

#include "infernix/ops/mla_sparse_attention.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

// One attention CTA: 16 warps, one query head per warp; a 32-token tile's positions and page
// offsets are staged in shared memory, the latent rows themselves are read through L2 (score and
// value phases read the same rows, so the second pass hits the cache). The launch's splits cover
// a column's attended tokens in 32-token tiles.
constexpr int kThreads     = 512;
constexpr int kHeadGroup   = 16; // query heads per attention CTA
constexpr int kTokenTile   = 32; // latent tokens per staged tile
constexpr int kMaxSplits   = 65; // automatic splits: (budget tokens + tail) / tile at the top
constexpr int kProjectCols = 32; // projection columns per CTA pass

constexpr std::size_t kChunkBudget = std::size_t{32} << 20; // partials + latent sums per chunk

// The registered geometry (heads may be any multiple of 16).
constexpr int kLatent  = 512;
constexpr int kHeadDim = 256;
// INT8-G64 cache: one FP16 scale per 64 latent values.
constexpr int kGroup  = 64;
constexpr int kGroups = kLatent / kGroup;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("mla attention: ") + message); }
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

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    return v;
}

// Attended tokens of a column: dense (count < 0 or no pools) is 0..n-1; sparse is count pools
// (pool tokens each, in ascending pool order) plus the n % pool visible tail tokens.
__device__ __forceinline__ int attended_total(int count, int pool, int n) {
    return count < 0 ? n : count * pool + n % pool;
}

// The token position of index i of a sparse column's attended list.
__device__ __forceinline__ int attended_position(int i, int count, int pool,
                                                 const std::int32_t* pools, int top,
                                                 std::int64_t column, int tail_base) {
    if (i < count * pool) { return pools[i / pool + column * top] * pool + i % pool; }
    return tail_base + (i - count * pool);
}

// Splits of a column: bounded by its own attended tokens and the launch's grid splits. Shared by
// the split kernel and the merge, so both skip the same empty splits.
__device__ __forceinline__ int column_splits(int total, int splits_grid) {
    return min(splits_grid, (total + kTokenTile - 1) / kTokenTile);
}

// Shared memory of the split kernel: the tile's latent rows as stored (staged once, read by every
// head's warp from shared memory instead of walking L2 per head), an INT8 tile's decoded scales,
// its probabilities, its token page-table resolved element offsets, and the running softmax
// statistics.
template <bool kInt8>
struct AttentionShared {
    static constexpr std::size_t kRows   = (kInt8 ? 1 : sizeof(bf16)) * kTokenTile * kLatent;
    static constexpr std::size_t kScales =
        kRows + (kInt8 ? sizeof(float) * kTokenTile * kGroups : 0);
    static constexpr std::size_t kProbs  = kScales + sizeof(float) * kHeadGroup * kTokenTile;
    static constexpr std::size_t kStats  = kProbs + sizeof(float) * 3 * kHeadGroup;
    static constexpr std::size_t kTokens = kStats + sizeof(std::int64_t) * kTokenTile;
    static constexpr std::size_t kBytes  = kTokens;
};

// The 16 values of one lane's latent dimensions of a staged row, decoded to FP32: BF16 as stored,
// INT8 as codes (the caller applies the group's scale once per row).
template <bool kInt8>
__device__ __forceinline__ void load_lane_row(const unsigned char* row, int dim0, float (&v)[16]) {
    if constexpr (kInt8) {
        const uint4 raw  = *reinterpret_cast<const uint4*>(row + dim0);
        const auto* code = reinterpret_cast<const std::int8_t*>(&raw);
#pragma unroll
        for (int e = 0; e < 16; ++e) { v[e] = static_cast<float>(code[e]); }
    } else {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint4 raw = *reinterpret_cast<const uint4*>(row + sizeof(bf16) * (dim0 + 8 * r));
            const auto* b   = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
            for (int e = 0; e < 8; ++e) { v[8 * r + e] = __bfloat162float(b[e]); }
        }
    }
}

// One CTA per (split, head group, column): the column's head group over one slice of its attended
// tokens. Warp w owns head w of the group; lane l owns the 16 latent dimensions 16 l..16 l + 15 of
// its head's queries and of every latent row (within one INT8 scale group, shared by 4 lanes).
// Each tile's rows are staged in shared memory once (the tile spans at most two scattered pages),
// then every head scores and folds them from there.
// Online softmax per head and tile: the tile's exact maximum, exponentials in parallel, then the
// sum folded in token order. The unnormalized latent sums accumulate in registers across the
// split's tiles and are written once, with the running maximum and sum.
template <bool kInt8>
__global__ void __launch_bounds__(kThreads)
    attention_split_kernel(const bf16* __restrict__ qa, int heads,
                           const unsigned char* __restrict__ pages,
                           const __half* __restrict__ scales,
                           const std::int32_t* __restrict__ tables, int table_stride,
                           const std::int32_t* __restrict__ table_rows,
                           const std::int32_t* __restrict__ positions, int width,
                           const std::int32_t* __restrict__ pools,
                           const std::int32_t* __restrict__ counts, int top, int pool,
                           int splits_grid, int groups, float scale, int column_begin,
                           float* __restrict__ partial) {
    constexpr int kDims = kLatent / 32; // latent dimensions per lane
    static_assert(kDims == 16, "a lane's dimensions are one 16-byte INT8 or two BF16 loads");
    using Shared = AttentionShared<kInt8>;
    constexpr std::size_t kRowBytes = (kInt8 ? 1 : sizeof(bf16)) * kLatent;
    pdl::enter();
    const int split = static_cast<int>(blockIdx.x), group = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.z), column = column_begin + t;
    const int count  = counts == nullptr ? -1 : counts[column];
    const bool dense = pools == nullptr || counts == nullptr || count < 0;
    const int n      = positions[column] + 1;
    const int total  = attended_total(dense ? -1 : count, pool, n);
    const int splits = column_splits(total, splits_grid);
    if (split >= splits) { return; }
    const int per_split = (total + splits - 1) / splits;
    const int begin = split * per_split, end = min(total, begin + per_split);
    const int tiles     = (end - begin + kTokenTile - 1) / kTokenTile;
    const int tail_base = n - n % pool;
    const std::int32_t* table =
        tables + static_cast<std::int64_t>(table_rows[column / width]) * table_stride;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* rows_s = smem; // the tile's latent rows, kRowBytes each
    auto* group_s = reinterpret_cast<float (*)[kGroups]>(smem + Shared::kRows); // INT8 scales
    auto* probs   = reinterpret_cast<float (*)[kTokenTile]>(smem + Shared::kScales);
    float* m_run  = reinterpret_cast<float*>(smem + Shared::kProbs);
    float* l_run  = m_run + kHeadGroup;
    float* scale_ = l_run + kHeadGroup; // the tile's rescale, computed by lane 0
    auto* offset  = reinterpret_cast<std::int64_t*>(smem + Shared::kStats);

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int head = group * kHeadGroup + warp;
    const int dim0 = lane * kDims; // this lane's first latent dimension
    const int lane_group = dim0 / kGroup;

    // The head's query, the lane's dimensions only (two 16-byte loads, not sixteen scalars).
    float q[kDims];
    {
        const bf16* qrow = qa + static_cast<std::int64_t>(head) * kLatent +
                           static_cast<std::int64_t>(column) * kLatent * heads + dim0;
#pragma unroll
        for (int r = 0; r < kDims / 8; ++r) {
            const uint4 raw = *reinterpret_cast<const uint4*>(qrow + 8 * r);
            const auto* v   = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
            for (int e = 0; e < 8; ++e) { q[8 * r + e] = __bfloat162float(v[e]); }
        }
    }
    if (threadIdx.x < kHeadGroup) {
        m_run[threadIdx.x] = -INFINITY;
        l_run[threadIdx.x] = 0.0F;
    }
    __syncthreads(); // the statistics are initialized before any warp reads them

    float acc[kDims];
#pragma unroll
    for (int e = 0; e < kDims; ++e) { acc[e] = 0.0F; }

    for (int i = 0; i < tiles; ++i) {
        const int first = begin + i * kTokenTile;
        const int n_tok = min(kTokenTile, end - first);
        // The tile's latent rows: element offsets into the cache plane, resolved once.
        if (static_cast<int>(threadIdx.x) < n_tok) {
            const int index = first + static_cast<int>(threadIdx.x);
            const int position =
                dense ? index
                      : attended_position(index, count, pool, pools, top, column, tail_base);
            offset[threadIdx.x] = (static_cast<std::int64_t>(table[position >> kPagedKVPageShift]) *
                                       kPagedKVPageSize +
                                   (position & kPagedKVPageMask)) *
                                  kLatent;
        }
        __syncthreads();
        // Stage the tile's rows once (and an INT8 tile's scales, decoded): every head's warp then
        // reads them from shared memory. offset[j] counts elements: one byte each for INT8.
        constexpr int kChunks = static_cast<int>(kRowBytes / 16);
        for (int i = static_cast<int>(threadIdx.x); i < n_tok * kChunks; i += kThreads) {
            const int j = i / kChunks, ch = i % kChunks;
            reinterpret_cast<uint4*>(rows_s + j * kRowBytes)[ch] = *reinterpret_cast<const uint4*>(
                pages + offset[j] * static_cast<std::int64_t>(kRowBytes / kLatent) + 16 * ch);
        }
        if constexpr (kInt8) {
            if (static_cast<int>(threadIdx.x) < n_tok * kGroups) {
                const int j = static_cast<int>(threadIdx.x) / kGroups;
                const int g = static_cast<int>(threadIdx.x) % kGroups;
                group_s[j][g] = __half2float(scales[offset[j] / kGroup + g]);
            }
        }
        __syncthreads();
        // Scores: this warp's head against every token of the tile.
        for (int j = 0; j < n_tok; ++j) {
            float v[kDims];
            load_lane_row<kInt8>(rows_s + j * kRowBytes, dim0, v);
            float dot = 0.0F;
#pragma unroll
            for (int e = 0; e < kDims; ++e) { dot += q[e] * v[e]; }
            if constexpr (kInt8) { dot *= group_s[j][lane_group]; }
            const float total = warp_sum(dot);
            if (lane == 0) { probs[warp][j] = total * scale; }
        }
        __syncwarp();
        // Online softmax per head: the tile's exact maximum, then l in token order. Lane 0 owns
        // the running statistics and the rescale; the warp syncs before anyone reads them.
        const float a = lane < n_tok ? probs[warp][lane] : -INFINITY;
        const float m = fmaxf(m_run[warp], warp_max(a));
        float added   = 0.0F;
        if (lane < n_tok) {
            const float e     = expf(a - m);
            probs[warp][lane] = e; // the value phase folds the exponentials
            added             = e;
        }
        added = warp_sum(added);
        if (lane == 0) {
            scale_[warp] = expf(m_run[warp] - m);
            m_run[warp]  = m;
            l_run[warp]  = l_run[warp] * scale_[warp] + added;
        }
        __syncwarp();
        const float r = scale_[warp];
#pragma unroll
        for (int e = 0; e < kDims; ++e) { acc[e] *= r; }
        // Values: the lane's dimensions of this warp's head, folded in token order.
        for (int j = 0; j < n_tok; ++j) {
            float p = probs[warp][j];
            if constexpr (kInt8) { p *= group_s[j][lane_group]; }
            float v[kDims];
            load_lane_row<kInt8>(rows_s + j * kRowBytes, dim0, v);
#pragma unroll
            for (int e = 0; e < kDims; ++e) { acc[e] += p * v[e]; }
        }
        __syncthreads(); // the tile's probability and offset slots are free once every warp is done
    }
    // The split's partials: unnormalized sums, running maximum and sum, per head.
    float* slot = partial +
                  (static_cast<std::size_t>(t) * splits_grid * groups +
                   static_cast<std::size_t>(split) * groups + group) *
                      (kHeadGroup * (kLatent + 2)) +
                  static_cast<std::size_t>(warp) * (kLatent + 2);
#pragma unroll
    for (int e = 0; e < kDims; ++e) { slot[dim0 + e] = acc[e]; }
    if (lane == 0) {
        slot[kLatent]     = m_run[warp];
        slot[kLatent + 1] = l_run[warp];
    }
}

// Combines a column's splits into its normalized latent sums: grid (heads, columns).
__global__ void attention_merge_kernel(const float* __restrict__ partial, int heads,
                                       int splits_grid, int groups,
                                       const std::int32_t* __restrict__ positions,
                                       const std::int32_t* __restrict__ counts,
                                       const std::int32_t* __restrict__ pools, int top, int pool,
                                       int column_begin, float* __restrict__ ol) {
    pdl::enter();
    const int h = static_cast<int>(blockIdx.x), t = static_cast<int>(blockIdx.y);
    const int column     = column_begin + t;
    const int count      = counts == nullptr ? -1 : counts[column];
    const bool dense     = pools == nullptr || counts == nullptr || count < 0;
    const int n          = positions[column] + 1;
    const int total      = attended_total(dense ? -1 : count, pool, n);
    const int splits     = column_splits(total, splits_grid);
    constexpr int kBlock = kHeadGroup * (kLatent + 2);
    const int group = h / kHeadGroup, h_local = h % kHeadGroup, d = static_cast<int>(threadIdx.x);
    const float* base = partial +
                        (static_cast<std::size_t>(t) * splits_grid * groups + group) * kBlock +
                        static_cast<std::size_t>(h_local) * (kLatent + 2);
    float m           = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        m = fmaxf(m, base[static_cast<std::size_t>(s) * groups * kBlock + kLatent]);
    }
    float l = 0.0F, acc = 0.0F;
    for (int s = 0; s < splits; ++s) {
        const float* slot = base + static_cast<std::size_t>(s) * groups * kBlock;
        const float ms    = slot[kLatent];
        if (ms == -INFINITY) { continue; }
        const float w = expf(ms - m);
        l += w * slot[kLatent + 1];
        acc += w * slot[d];
    }
    ol[d + static_cast<std::size_t>(h) * kLatent + static_cast<std::size_t>(t) * kLatent * heads] =
        acc / l;
}

// One warp per (head, output) row; the row loops the chunk's columns. The weight row is read once
// per warp and reused across columns; the latent sums are read per column. Ol is BF16 (the public
// projection's stored latent sums) or FP32 (the attention's merged sums).
template <typename Ol>
__global__ void project_kernel(const Ol* __restrict__ ol, const bf16* __restrict__ w_uv, int heads,
                               int columns, int column_begin, bf16* __restrict__ out) {
    constexpr int kPer   = kLatent / 32; // latent elements per lane
    constexpr int kWarps = 256 / 32;
    pdl::enter();
    const int row = static_cast<int>(blockIdx.x) * kWarps + static_cast<int>(threadIdx.x) / 32;
    if (row >= heads * kHeadDim) { return; }
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int h = row / kHeadDim, e = row % kHeadDim;
    // w_uv element (d, e, h) = W_uv,h[e][d]: the row's latent dimensions are contiguous.
    const bf16* wrow = w_uv + static_cast<std::int64_t>(h) * kHeadDim * kLatent + e * kLatent;
    float w[kPer];
#pragma unroll
    for (int r = 0; r < kPer / 8; ++r) {
        const uint4 raw = *reinterpret_cast<const uint4*>(wrow + lane * kPer + 8 * r);
        const auto* v   = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int u = 0; u < 8; ++u) { w[8 * r + u] = __bfloat162float(v[u]); }
    }
    for (int c = 0; c < columns; ++c) {
        const Ol* rowp = ol + lane * kPer + static_cast<std::int64_t>(h) * kLatent +
                         static_cast<std::int64_t>(c) * kLatent * heads;
        float acc      = 0.0F;
#pragma unroll
        for (int r = 0; r < kPer / 8; ++r) {
            if constexpr (std::is_same_v<Ol, bf16>) {
                const uint4 raw = *reinterpret_cast<const uint4*>(rowp + 8 * r);
                const auto* v   = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
                for (int u = 0; u < 8; ++u) { acc += w[8 * r + u] * __bfloat162float(v[u]); }
            } else {
                const float4 lo = *reinterpret_cast<const float4*>(rowp + 8 * r);
                const float4 hi = *reinterpret_cast<const float4*>(rowp + 8 * r + 4);
                acc += w[8 * r + 0] * lo.x + w[8 * r + 1] * lo.y + w[8 * r + 2] * lo.z +
                       w[8 * r + 3] * lo.w;
                acc += w[8 * r + 4] * hi.x + w[8 * r + 5] * hi.y + w[8 * r + 6] * hi.z +
                       w[8 * r + 7] * hi.w;
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            out[e + static_cast<std::int64_t>(h) * kHeadDim +
                static_cast<std::int64_t>(column_begin + c) * kHeadDim * heads] =
                __float2bfloat16_rn(acc);
        }
    }
}

// Absorbed query projection: CTA (column block, head, latent-dim chunk of 64), the query tile
// staged in shared memory, one warp per latent dimension d of the chunk looping the block's
// columns. The chunking fills the machine at decode's one-to-four columns: the weight stream,
// not the launch width, decides the time.
__global__ void qa_project_kernel(const bf16* __restrict__ q, const bf16* __restrict__ w_uk,
                                  int heads, int columns, bf16* __restrict__ qa) {
    constexpr int kQPer     = kHeadDim / 32; // query elements per lane
    constexpr int kWarps    = 256 / 32;
    constexpr int kDimChunk = 64; // latent dimensions per CTA
    __shared__ bf16 tile[kProjectCols][kHeadDim];
    pdl::enter();
    const int c0 = static_cast<int>(blockIdx.x) * kProjectCols, h = static_cast<int>(blockIdx.y);
    const int cols = min(kProjectCols, columns - c0);
    for (int i = static_cast<int>(threadIdx.x); i < cols * kHeadDim; i += 256) {
        const int c = i / kHeadDim, d0 = i % kHeadDim;
        tile[c][d0] = q[d0 + static_cast<std::int64_t>(h) * kHeadDim +
                        static_cast<std::int64_t>(c0 + c) * kHeadDim * heads];
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int chunk = static_cast<int>(blockIdx.z) * kDimChunk;
    for (int d = chunk + warp; d < chunk + kDimChunk; d += kWarps) {
        float w[kQPer];
        // w_uk element (d0, d, h) = (W_uk,h)^T[d][d0]: the row's query dimensions are contiguous.
        const uint4 raw = *reinterpret_cast<const uint4*>(
            w_uk + d * kHeadDim + static_cast<std::int64_t>(h) * kHeadDim * kLatent + lane * kQPer);
        const auto* v = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int u = 0; u < kQPer; ++u) { w[u] = __bfloat162float(v[u]); }
        for (int c = 0; c < cols; ++c) {
            float acc = 0.0F;
#pragma unroll
            for (int u = 0; u < kQPer; ++u) {
                acc += w[u] * __bfloat162float(tile[c][lane * kQPer + u]);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                qa[d + static_cast<std::int64_t>(h) * kLatent +
                   static_cast<std::int64_t>(c0 + c) * kLatent * heads] = __float2bfloat16_rn(acc);
            }
        }
    }
}

// Latent append: grid (columns), one uint4 chunk per thread.
__global__ void latent_append_kernel(const bf16* __restrict__ latents,
                                     const std::int32_t* __restrict__ positions,
                                     const std::int32_t* __restrict__ tables, int table_stride,
                                     const std::int32_t* __restrict__ table_rows, int width,
                                     bf16* __restrict__ pages) {
    const int column = static_cast<int>(blockIdx.x);
    const int b = column / width, w = column % width;
    const int p = positions[column];
    const int page =
        tables[static_cast<std::int64_t>(table_rows[b]) * table_stride + (p >> kPagedKVPageShift)];
    const bf16* src = latents + static_cast<std::int64_t>(w) * kLatent +
                      static_cast<std::int64_t>(b) * kLatent * width;
    bf16* dst =
        pages +
        (static_cast<std::int64_t>(page) * kPagedKVPageSize + (p & kPagedKVPageMask)) * kLatent;
    for (int ch = static_cast<int>(threadIdx.x); ch < kLatent * 2 / 16; ch += 64) {
        *reinterpret_cast<uint4*>(dst + 8 * ch) = *reinterpret_cast<const uint4*>(src + 8 * ch);
    }
}

// INT8-G64 latent append: grid (columns), 64 threads of 8 values each; the 8 threads of a group
// reduce its maximum with shuffles, and the group's first thread writes the scale.
__global__ void latent_append_int8_kernel(const bf16* __restrict__ latents,
                                          const std::int32_t* __restrict__ positions,
                                          const std::int32_t* __restrict__ tables, int table_stride,
                                          const std::int32_t* __restrict__ table_rows, int width,
                                          std::int8_t* __restrict__ pages,
                                          __half* __restrict__ scales) {
    constexpr int kPerThread = 8, kGroupThreads = kGroup / kPerThread;
    const int column = static_cast<int>(blockIdx.x), t = static_cast<int>(threadIdx.x);
    const int b = column / width, w = column % width;
    const int p = positions[column];
    const int page =
        tables[static_cast<std::int64_t>(table_rows[b]) * table_stride + (p >> kPagedKVPageShift)];
    const std::int64_t slot =
        static_cast<std::int64_t>(page) * kPagedKVPageSize + (p & kPagedKVPageMask);
    const bf16* src = latents + static_cast<std::int64_t>(w) * kLatent +
                      static_cast<std::int64_t>(b) * kLatent * width + kPerThread * t;
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const auto* v   = reinterpret_cast<const bf16*>(&raw);
    float x[kPerThread];
    float a = 0.0F;
#pragma unroll
    for (int e = 0; e < kPerThread; ++e) {
        x[e] = __bfloat162float(v[e]);
        a    = fmaxf(a, fabsf(x[e]));
    }
#pragma unroll
    for (int o = kGroupThreads / 2; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFU, a, o));
    }
    const __half scale_bits = __float2half_rn(__fdiv_rn(a, 127.0F));
    const float s           = __half2float(scale_bits);
    const float inv         = s == 0.0F ? 0.0F : __frcp_rn(s);
    alignas(8) std::int8_t code[kPerThread];
#pragma unroll
    for (int e = 0; e < kPerThread; ++e) {
        code[e] = static_cast<std::int8_t>(
            s == 0.0F ? 0.0F : fminf(fmaxf(rintf(__fmul_rn(x[e], inv)), -127.0F), 127.0F));
    }
    *reinterpret_cast<uint2*>(pages + slot * kLatent + kPerThread * t) =
        *reinterpret_cast<const uint2*>(code);
    if (t % kGroupThreads == 0) { scales[slot * kGroups + t / kGroupThreads] = scale_bits; }
}

int resolve_splits(std::int32_t max_context, MlaAttentionExecutionEnvelope envelope) {
    if (envelope.splits > 0) { return envelope.splits; }
    return std::clamp(ceil_div(max_context, kTokenTile), 1, kMaxSplits);
}

struct Carve {
    int splits = 0, groups = 0, chunk = 0;
    std::size_t bytes = 0;
    float* partial    = nullptr; // [chunk, splits, groups, 16 * (latent + 2)]
    float* ol         = nullptr; // [latent, heads, chunk] FP32, the merged latent sums
};

Carve carve(void* base, const MlaAttentionGeometry& geometry, int columns, int splits) {
    Carve out;
    out.splits = splits;
    out.groups = geometry.heads / kHeadGroup;
    const std::size_t per_column =
        static_cast<std::size_t>(splits) * out.groups * kHeadGroup * (kLatent + 2) +
        static_cast<std::size_t>(geometry.heads) * kLatent;
    out.chunk = std::clamp<std::size_t>(kChunkBudget / (per_column * sizeof(float)), 1,
                                        static_cast<std::size_t>(columns));
    out.bytes = align256(out.chunk * per_column * sizeof(float));
    if (base != nullptr) {
        auto* p     = static_cast<unsigned char*>(base);
        out.partial = reinterpret_cast<float*>(p);
        out.ol =
            reinterpret_cast<float*>(p + out.chunk * static_cast<std::size_t>(splits) * out.groups *
                                             kHeadGroup * (kLatent + 2) * sizeof(float));
    }
    return out;
}

void require_geometry(const MlaAttentionGeometry& g) {
    require(g.heads > 0 && g.heads % kHeadGroup == 0, "heads must be a positive multiple of 16");
    require(g.head_dim == kHeadDim && g.latent_dim == kLatent,
            "the registered geometry is head_dim 256, latent_dim 512");
}

// The cache's storage: BF16 pages without scales, or INT8-G64 pages with their FP16 scales.
bool require_cache(const MlaLatentBatchView& cache) {
    const bool int8 = cache.pages.dtype == DType::I8;
    require((contiguous(cache.pages, DType::BF16) || contiguous(cache.pages, DType::I8)) &&
                cache.pages.ne[0] == kLatent && cache.pages.ne[1] == kPagedKVPageSize &&
                cache.pages.ne[2] == 1,
            "cache pages must be BF16 or I8 [latent_dim, 64, 1, pages]");
    if (int8) {
        require(contiguous(cache.scales, DType::FP16) && cache.scales.ne[0] == kGroups &&
                    cache.scales.ne[1] == kPagedKVPageSize && cache.scales.ne[2] == 1 &&
                    cache.scales.ne[3] == cache.pages.ne[3],
                "an INT8 cache's scales must be FP16 [latent_dim / 64, 64, 1, pages]");
    } else {
        require(cache.scales.data == nullptr, "a BF16 cache has no scales");
    }
    return int8;
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

} // namespace

void mla_latent_append(const Tensor& latents, const MlaSparseBatch& batch,
                       const MlaLatentBatchView& cache, cudaStream_t stream) {
    const int columns = batch.batch * batch.width;
    require(columns > 0, "the call needs at least one column");
    require_batch(batch, columns);
    require(contiguous(latents, DType::BF16) && latents.ne[0] == kLatent &&
                latents.ne[1] == batch.width && latents.ne[2] == batch.batch,
            "latents must be BF16 [latent_dim, width, batch]");
    if (require_cache(cache)) {
        latent_append_int8_kernel<<<dim3(columns), kLatent / 8, 0, stream>>>(
            static_cast<const bf16*>(latents.data),
            static_cast<const std::int32_t*>(batch.positions.data),
            static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
            static_cast<const std::int32_t*>(batch.table_rows.data), batch.width,
            static_cast<std::int8_t*>(cache.pages.data), static_cast<__half*>(cache.scales.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    latent_append_kernel<<<dim3(columns), 64, 0, stream>>>(
        static_cast<const bf16*>(latents.data),
        static_cast<const std::int32_t*>(batch.positions.data),
        static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
        static_cast<const std::int32_t*>(batch.table_rows.data), batch.width,
        static_cast<bf16*>(cache.pages.data));
    CUDA_CHECK(cudaGetLastError());
}

void mla_absorbed_q_projection(const Tensor& q, const Tensor& w_uk, Tensor& qa,
                               cudaStream_t stream) {
    const MlaAttentionGeometry geometry{
        .heads = q.ne[1], .head_dim = q.ne[0], .latent_dim = qa.ne[0]};
    require_geometry(geometry);
    const int columns = q.ne[2];
    require(columns > 0 && contiguous(q, DType::BF16) && q.ne[0] == kHeadDim &&
                contiguous(qa, DType::BF16) && qa.ne[0] == kLatent && qa.ne[1] == geometry.heads &&
                qa.ne[2] == columns && contiguous(w_uk, DType::BF16) && w_uk.ne[0] == kHeadDim &&
                w_uk.ne[1] == kLatent && w_uk.ne[2] == geometry.heads,
            "projection shapes disagree (w_uk is BF16 [head_dim, latent_dim, heads])");
    const int blocks         = ceil_div(columns, kProjectCols);
    constexpr int kDimChunks = kLatent / 64; // the kernel's latent-dim chunks per head
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(geometry.heads), kDimChunks),
         dim3(256), 0, stream},
        qa_project_kernel, static_cast<const bf16*>(q.data), static_cast<const bf16*>(w_uk.data),
        geometry.heads, columns, static_cast<bf16*>(qa.data)));
    CUDA_CHECK(cudaGetLastError());
}

void mla_latent_output_projection(const Tensor& ol, const Tensor& w_uv, Tensor& o,
                                  cudaStream_t stream) {
    const MlaAttentionGeometry geometry{
        .heads = ol.ne[1], .head_dim = o.ne[0], .latent_dim = ol.ne[0]};
    require_geometry(geometry);
    const int columns = ol.ne[2];
    require(columns > 0 && contiguous(ol, DType::BF16) && ol.ne[0] == kLatent &&
                contiguous(o, DType::BF16) && o.ne[0] == kHeadDim && o.ne[1] == geometry.heads &&
                o.ne[2] == columns && contiguous(w_uv, DType::BF16) && w_uv.ne[0] == kLatent &&
                w_uv.ne[1] == kHeadDim && w_uv.ne[2] == geometry.heads,
            "projection shapes disagree (w_uv is BF16 [latent_dim, head_dim, heads])");
    const int rows = geometry.heads * kHeadDim;
    CUDA_CHECK(
        pdl::launch_consumer({dim3(static_cast<unsigned>(ceil_div(rows, 8))), dim3(256), 0, stream},
                             project_kernel<bf16>, static_cast<const bf16*>(ol.data),
                             static_cast<const bf16*>(w_uv.data), geometry.heads, columns, 0,
                             static_cast<bf16*>(o.data)));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t mla_sparse_attention_workspace_bytes(const MlaAttentionGeometry& geometry,
                                                 std::int32_t columns, std::int32_t max_context,
                                                 MlaAttentionExecutionEnvelope envelope) {
    require_geometry(geometry);
    require(columns > 0 && max_context > 0 && envelope.splits >= 0, "invalid workspace profile");
    return carve(nullptr, geometry, columns, resolve_splits(max_context, envelope)).bytes;
}

void mla_sparse_attention(const Tensor& qa, const Tensor& w_uv, const MlaLatentBatchView& cache,
                          const MlaSparseBatch& batch, const MlaSelectedPools& selected,
                          const MlaAttentionGeometry& geometry, float scale,
                          std::int32_t max_context, MlaAttentionExecutionEnvelope envelope,
                          void* workspace, std::size_t workspace_bytes, Tensor& out,
                          cudaStream_t stream) {
    require_geometry(geometry);
    const int columns = batch.batch * batch.width;
    require(columns > 0, "the call needs at least one column");
    require_batch(batch, columns);
    require(contiguous(qa, DType::BF16) && qa.ne[0] == kLatent && qa.ne[1] == geometry.heads &&
                qa.ne[2] == columns && contiguous(out, DType::BF16) && out.ne[0] == kHeadDim &&
                out.ne[1] == geometry.heads && out.ne[2] == columns,
            "attention shapes disagree");
    require(contiguous(w_uv, DType::BF16) && w_uv.ne[0] == kLatent && w_uv.ne[1] == kHeadDim &&
                w_uv.ne[2] == geometry.heads,
            "w_uv must be BF16 [latent_dim, head_dim, heads]");
    const bool int8 = require_cache(cache);
    require(scale > 0.0F && max_context > 0, "scale and max_context must be positive");
    const Tensor* pools  = selected.pools;
    const Tensor* counts = selected.counts;
    const bool sparse    = pools != nullptr && counts != nullptr && pools->data != nullptr;
    if (sparse) {
        require(contiguous(*pools, DType::I32) && pools->ne[0] > 0 && pools->ne[1] == columns &&
                    contiguous(*counts, DType::I32) && counts->numel() == columns,
                "pools must be I32 [top_pools, columns] and counts I32 [columns]");
        require(selected.pool > 0 && selected.pool < kPagedKVPageSize, "pool must be in (0, 64)");
    }
    const int splits = resolve_splits(max_context, envelope);
    Carve ws         = carve(workspace, geometry, columns, splits);
    require(workspace != nullptr && workspace_bytes >= ws.bytes, "workspace is too small");
    const auto* pages  = static_cast<const unsigned char*>(cache.pages.data);
    const auto* scales = static_cast<const __half*>(cache.scales.data);
    const auto* tables = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int stride   = batch.block_tables.ne[0];
    const auto* rows   = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* pos    = static_cast<const std::int32_t*>(batch.positions.data);
    const auto* pool_p = sparse ? static_cast<const std::int32_t*>(pools->data) : nullptr;
    const auto* cnt_p  = sparse ? static_cast<const std::int32_t*>(counts->data) : nullptr;
    const int top      = sparse ? pools->ne[0] : 0;
    const int pool     = sparse ? selected.pool : 4;
    for (int begin = 0; begin < columns; begin += static_cast<int>(ws.chunk)) {
        const int n = std::min<int>(static_cast<int>(ws.chunk), columns - begin);
        const dim3 grid(static_cast<unsigned>(splits), static_cast<unsigned>(ws.groups),
                        static_cast<unsigned>(n));
        if (int8) {
            CUDA_CHECK(pdl::launch_consumer(
                {grid, dim3(kThreads), AttentionShared<true>::kBytes, stream},
                attention_split_kernel<true>, static_cast<const bf16*>(qa.data), geometry.heads,
                pages, scales, tables, stride, rows, pos, batch.width, pool_p, cnt_p, top, pool,
                splits, ws.groups, scale, begin, ws.partial));
        } else {
            CUDA_CHECK(pdl::launch_consumer(
                {grid, dim3(kThreads), AttentionShared<false>::kBytes, stream},
                attention_split_kernel<false>, static_cast<const bf16*>(qa.data), geometry.heads,
                pages, scales, tables, stride, rows, pos, batch.width, pool_p, cnt_p, top, pool,
                splits, ws.groups, scale, begin, ws.partial));
        }
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(static_cast<unsigned>(geometry.heads), static_cast<unsigned>(n)), dim3(kLatent),
             0, stream},
            attention_merge_kernel, ws.partial, geometry.heads, splits, ws.groups, pos, cnt_p,
            pool_p, top, pool, begin, ws.ol));
        const int row_blocks = ceil_div(static_cast<std::int64_t>(geometry.heads) * kHeadDim, 8);
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(static_cast<unsigned>(row_blocks)), dim3(256), 0, stream}, project_kernel<float>,
            static_cast<const float*>(ws.ol), static_cast<const bf16*>(w_uv.data), geometry.heads,
            n, begin, static_cast<bf16*>(out.data)));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
