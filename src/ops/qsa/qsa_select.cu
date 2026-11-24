// QSA block selection: multi-CTA scoring and exact radix selection (qsa_select.h,
// docs/maintainer/qwen3_8-flash-next-design.md §3.1).

#include "ops/qsa/qsa_select.h"

#include "ops/common/exact_topk_select.h"
#include "ops/qsa/page_spaces.cuh"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/pdl.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::ops::detail {
namespace {

using bf16   = __nv_bfloat16;

constexpr int kIndexDim       = 128;
constexpr unsigned kFullMask  = 0xFFFFFFFFU;
constexpr int kLanePartials   = kIndexDim / 32; // products each lane accumulates per dot

// Scoring: a warp step reads kScoreStepBlocks pooled keys and, per column, reduce-scatters the
// kScoreStepBlocks x kScoreHeadSlots = 32 lane partials of its dots.
constexpr int kScoreThreads     = 256;
constexpr int kScoreWarps       = kScoreThreads / 32;
constexpr int kScoreCtasPerSm   = 2;
constexpr int kScoreStepBlocks  = 8;
constexpr int kScoreHeadSlots   = 4;  // index heads (1..4; unused slots hold zero queries)
constexpr int kScoreTileColumns = 16; // columns of one row scored from one key read
static_assert(kScoreStepBlocks * kScoreHeadSlots == 32);

// Selection state: one 2048-bin histogram plus one arrival counter per column, the layout
// exact_topk_select (ops/common/exact_topk_select.h) owns; this file only sizes its scratch.
constexpr int kScoreBins = kTopkScoreBins;

struct ScoreTiles {
    std::int32_t count = 0;
    std::uint8_t first[kQsaSelectGroupColumns];   // first column, counted from the group's first
    std::uint8_t columns[kQsaSelectGroupColumns]; // 1..kScoreTileColumns, all of one row
};
static_assert(kQsaSelectGroupColumns <= 256, "ScoreTiles stores group-relative columns in bytes");

// ------------------------------------------------------------------------------------- scoring

// Same addressing as qsa.cu's pool and attention kernels: the pooled key of block b fills the R
// token slots of its block, slot_width = Di / R elements each.
// A page id's space is resolved (design §14.4): the pool's pooled plane or the spaces' array.
__device__ __forceinline__ const bf16* pooled_element(const bf16* pool, const QsaPageSpaces& spaces,
                                                      const std::int32_t* table, int token, int slot_width, int slot) {
    int page;
    const bf16* plane = qsa_pooled_plane(spaces, pool, table[token >> kPagedKVPageShift], page);
    return plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + slot;
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
        v[i]             = keep + __shfl_xor_sync(kFullMask, send, Half);
    }
}

// Lane l returns the 32-lane sum of its value l, bitwise equal to warp_sum's result for that value
// (31 shuffles for 32 sums instead of 160). After the stages at offsets 16..o, each value lane l
// keeps is the sum over the lanes l ^ m, m a combination of those offsets, formed by exactly the
// additions the xor butterfly performs at lane l; the butterfly ends with that sum in every lane.
__device__ __forceinline__ float reduce_scatter_32(float (&v)[32], int lane) {
    reduce_scatter_stage<16>(v, lane);
    reduce_scatter_stage<8>(v, lane);
    reduce_scatter_stage<4>(v, lane);
    reduce_scatter_stage<2>(v, lane);
    reduce_scatter_stage<1>(v, lane);
    return v[0];
}

// Grid (CTAs per tile, tiles). A tile is up to 16 consecutive columns of one row; its CTAs split
// the row's blocks in steps of 8, one step per warp at a time. Each pooled key is read once per
// tile and scored for every column of the tile that sees it. With `clear`, the CTAs first zero
// `clear_words` words of selection state (histograms and arrival counters).
__global__ void __launch_bounds__(kScoreThreads, kScoreCtasPerSm)
    qsa_score_kernel(const bf16* __restrict__ index_q, int index_heads, const bf16* __restrict__ pooled,
                     const std::int32_t* __restrict__ tables, int table_stride,
                     const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                     int width, int ratio, int top_blocks, int group_begin, const ScoreTiles tiles,
                     float* __restrict__ scores, int score_stride, std::uint32_t* __restrict__ clear,
                     int clear_words, QsaPageSpaces spaces) {
    pdl::enter_streaming();
    if (clear != nullptr) {
        const std::int64_t ctas = static_cast<std::int64_t>(gridDim.x) * gridDim.y;
        const std::int64_t cta  = static_cast<std::int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        const int begin         = static_cast<int>(clear_words * cta / ctas);
        const int end           = static_cast<int>(clear_words * (cta + 1) / ctas);
        for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += kScoreThreads) { clear[i] = 0; }
    }
    __shared__ float4 qs[kScoreTileColumns][kScoreHeadSlots][32]; // lane l: q[h][l + 32 j], j = 0..3
    __shared__ int column_blocks[kScoreTileColumns];
    const int tile = static_cast<int>(blockIdx.y);
    const int c0 = tiles.first[tile], nc = tiles.columns[tile];
    const int t0 = group_begin + c0;
    int tile_blocks = 0;
    for (int c = 0; c < nc; ++c) { tile_blocks = max(tile_blocks, (positions[t0 + c] + 1) / ratio); }
    const int steps      = (tile_blocks + kScoreStepBlocks - 1) / kScoreStepBlocks;
    const int first_step = static_cast<int>(blockIdx.x) * kScoreWarps;
    if (tile_blocks <= top_blocks || first_step >= steps) {
        pdl::trigger_dependents();
        return; // every column of the tile is dense, or this CTA has no step
    }
    if (static_cast<int>(threadIdx.x) < nc) {
        column_blocks[threadIdx.x] = (positions[t0 + static_cast<int>(threadIdx.x)] + 1) / ratio;
    }
    for (int i = static_cast<int>(threadIdx.x); i < nc * kScoreHeadSlots * kIndexDim; i += kScoreThreads) {
        const int c = i / (kScoreHeadSlots * kIndexDim), h = (i / kIndexDim) % kScoreHeadSlots, d = i % kIndexDim;
        const float value =
            h < index_heads
                ? __bfloat162float(index_q[(static_cast<std::size_t>(t0 + c) * index_heads + h) * kIndexDim + d])
                : 0.0F;
        reinterpret_cast<float*>(&qs[c][h][d % 32])[d / 32] = value;
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t0 / width]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    int token_offset[kLanePartials], slot[kLanePartials];
#pragma unroll
    for (int j = 0; j < kLanePartials; ++j) {
        token_offset[j] = (lane + 32 * j) / slot_width;
        slot[j]         = (lane + 32 * j) % slot_width;
    }
    const bool block_in_page = ratio <= kPagedKVPageSize; // a block's tokens share one page
    const float scale        = rsqrtf(static_cast<float>(kIndexDim));
    for (int step = first_step + warp; step < steps; step += static_cast<int>(gridDim.x) * kScoreWarps) {
        const int b0 = step * kScoreStepBlocks;
        float key[kScoreStepBlocks][kLanePartials];
#pragma unroll
        for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
            const int b = b0 + bb;
            if (b >= tile_blocks) {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) { key[bb][j] = 0.0F; }
            } else if (block_in_page) {
                int local;
                const bf16* plane = qsa_pooled_plane(spaces, pooled, table[(ratio * b) >> kPagedKVPageShift], local);
                const std::int64_t page = static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * local;
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) {
                    const int token = ratio * b + token_offset[j];
                    key[bb][j] = __bfloat162float(plane[page + static_cast<std::int64_t>(slot_width) *
                                                                   (token & kPagedKVPageMask) + slot[j]]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) {
                    key[bb][j] = __bfloat162float(
                        *pooled_element(pooled, spaces, table, ratio * b + token_offset[j], slot_width, slot[j]));
                }
            }
        }
        for (int c = 0; c < nc; ++c) {
            const int blocks = column_blocks[c];
            if (b0 >= blocks) { continue; } // warp-uniform: this column sees none of the step
            float4 q[kScoreHeadSlots];
#pragma unroll
            for (int h = 0; h < kScoreHeadSlots; ++h) { q[h] = qs[c][h][lane]; }
            // Value bb * 4 + h: this lane's partial of block b0 + bb, head h, accumulated in j order
            // from 0 like the one-warp-per-block kernel's `dot += q[h][lane + 32 j] * key[j]`.
            float v[32];
#pragma unroll
            for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
#pragma unroll
                for (int h = 0; h < kScoreHeadSlots; ++h) {
                    float dot = 0.0F;
                    dot += q[h].x * key[bb][0];
                    dot += q[h].y * key[bb][1];
                    dot += q[h].z * key[bb][2];
                    dot += q[h].w * key[bb][3];
                    v[bb * kScoreHeadSlots + h] = dot;
                }
            }
            // Lane 4 bb + h now holds head h's dot of block b0 + bb; heads join in head order.
            const float relu  = fmaxf(reduce_scatter_32(v, lane), 0.0F);
            const float relu1 = __shfl_down_sync(kFullMask, relu, 1);
            const float relu2 = __shfl_down_sync(kFullMask, relu, 2);
            const float relu3 = __shfl_down_sync(kFullMask, relu, 3);
            const int b       = b0 + lane / kScoreHeadSlots;
            if (lane % kScoreHeadSlots == 0 && b < blocks) {
                float score = 0.0F;
                score += relu;
                if (index_heads > 1) { score += relu1; }
                if (index_heads > 2) { score += relu2; }
                if (index_heads > 3) { score += relu3; }
                scores[static_cast<std::size_t>(c0 + c) * score_stride + b] = score * scale;
            }
        }
    }
    pdl::trigger_dependents();
}

// --------------------------------------------------------------------------- wide-row scoring

// Prompt rows (width >= kWideMinWidth): the same scores as qsa_score_kernel, register-tiled for
// the FP32 pipes instead of shuffles. A thread owns one column's four head dots against two
// blocks (8 dots). It forms each dot's 32 lane partials itself, the chain over j from 0 exactly as
// lane l would, and adds them in the butterfly's tree: the partial of lane l pairs with lane
// l ^ 16 first, then the pairs at offset 8, 4, 2 and 1. Taking the leaves in bit-reversed lane
// order turns that tree into a binary counter, so a six-entry stack per dot holds every open
// subtree. FP32 addition is commutative, so each sum is bitwise the butterfly's.
// A CTA scores a 16-column row tile against key tiles of 32 blocks; the next key tile travels in
// registers while the current one is scored. Queries are staged as FP32 (rows of 32 float4 lanes plus
// one pad), keys as BF16 in the same lane order (rows of 32 four-element lanes plus one pad), widened
// to FP32 where they are used (exactly the values the FP32 staging held). Two CTAs fit an SM (42 KiB
// of shared memory and at most 128 registers per thread each): the kernel is latency-bound on its
// dependent adds, and 16 warps per SM hide twice what 8 did.
constexpr int kWideThreads      = 256;
constexpr int kWideCtasPerSm    = 2;
constexpr int kWideBlocks       = 32;                    // blocks per key tile
constexpr int kWideLaneBlocks   = 8;                     // consecutive blocks across a warp's lanes
constexpr int kWideThreadBlocks = 2;                     // a thread's blocks: g + 8 i, i = 0..1
constexpr int kWideRow          = 32 + 1;                // lanes per staged row (one pad)
constexpr int kWideMinWidth     = 2 * kScoreTileColumns; // rows this narrow keep qsa_score_kernel
constexpr int kWideStack        = 6;                     // open subtrees per dot, plus the new leaf
constexpr std::size_t kWideQueryFloat4 = static_cast<std::size_t>(kScoreHeadSlots) * kScoreTileColumns * kWideRow;
constexpr std::size_t kWideKeyLanes    = static_cast<std::size_t>(kWideBlocks) * kWideRow; // uint2: 4 BF16 each
constexpr std::size_t kWideSharedBytes = sizeof(float4) * kWideQueryFloat4 + sizeof(uint2) * kWideKeyLanes;
static_assert(kWideThreads == kScoreTileColumns * kWideBlocks / kWideThreadBlocks, "one thread per 2 blocks");
static_assert(kWideThreads * 16 == kWideBlocks * kIndexDim, "a thread loads 16 BF16 of one block's key");

__host__ __device__ constexpr int bit_reverse5(int i) {
    return ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4);
}
__host__ __device__ constexpr int popcount5(int i) {
    return (i & 1) + ((i >> 1) & 1) + ((i >> 2) & 1) + ((i >> 3) & 1) + ((i >> 4) & 1);
}
__host__ __device__ constexpr int trailing_zeros(int n) {
    int count = 0;
    while (count < 6 && (n & 1) == 0) {
        n >>= 1;
        ++count;
    }
    return count;
}

using WideStacks = float[kScoreHeadSlots][kWideThreadBlocks][kWideStack];

// Four BF16 (lane l's elements l + 32 j, j = 0..3) as FP32, exactly.
__device__ __forceinline__ float4 widen_lane(uint2 raw) {
    return make_float4(__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xFFFF0000U), __uint_as_float(raw.y << 16),
                       __uint_as_float(raw.y & 0xFFFF0000U));
}

// Leaf I (in bit-reversed order: lane l = bit_reverse5(I)) of all 8 dots, pushed and folded.
template <int I>
__device__ __forceinline__ void wide_leaves(const float4* __restrict__ query, const uint2* __restrict__ key,
                                            WideStacks& stack) {
    if constexpr (I < 32) {
        constexpr int l      = bit_reverse5(I);
        constexpr int depth  = popcount5(I);         // open subtrees before this leaf
        constexpr int merges = trailing_zeros(I + 1); // subtrees this leaf completes
        float4 q[kScoreHeadSlots], k[kWideThreadBlocks];
#pragma unroll
        for (int h = 0; h < kScoreHeadSlots; ++h) { q[h] = query[h * kScoreTileColumns * kWideRow + l]; }
#pragma unroll
        for (int i = 0; i < kWideThreadBlocks; ++i) { k[i] = widen_lane(key[i * kWideLaneBlocks * kWideRow + l]); }
#pragma unroll
        for (int h = 0; h < kScoreHeadSlots; ++h) {
#pragma unroll
            for (int i = 0; i < kWideThreadBlocks; ++i) {
                // Lane l's partial, as qsa_score_kernel forms it.
                float dot = 0.0F;
                dot += q[h].x * k[i].x;
                dot += q[h].y * k[i].y;
                dot += q[h].z * k[i].z;
                dot += q[h].w * k[i].w;
                stack[h][i][depth] = dot;
#pragma unroll
                for (int m = 0; m < merges; ++m) {
                    stack[h][i][depth - 1 - m] = stack[h][i][depth - 1 - m] + stack[h][i][depth - m];
                }
            }
        }
        wide_leaves<I + 1>(query, key, stack);
    }
}

// Grid (CTAs per tile, tiles), kWideThreads threads, kWideSharedBytes of dynamic shared memory.
// Warp w scores columns 4 (w % 4) .. + 3 (lane / 8) against blocks 16 (w / 4) + lane % 8 + 8 i of
// each key tile. Arguments as qsa_score_kernel's.
__global__ void __launch_bounds__(kWideThreads, kWideCtasPerSm)
    qsa_score_wide_kernel(const bf16* __restrict__ index_q, int index_heads, const bf16* __restrict__ pooled,
                          const std::int32_t* __restrict__ tables, int table_stride,
                          const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                          int width, int ratio, int top_blocks, int group_begin, const ScoreTiles tiles,
                          float* __restrict__ scores, int score_stride, std::uint32_t* __restrict__ clear,
                          int clear_words, QsaPageSpaces spaces) {
    pdl::enter_streaming();
    const int tid = static_cast<int>(threadIdx.x);
    if (clear != nullptr) {
        const std::int64_t ctas = static_cast<std::int64_t>(gridDim.x) * gridDim.y;
        const std::int64_t cta  = static_cast<std::int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        const int begin         = static_cast<int>(clear_words * cta / ctas);
        const int end           = static_cast<int>(clear_words * (cta + 1) / ctas);
        for (int i = begin + tid; i < end; i += kWideThreads) { clear[i] = 0; }
    }
    extern __shared__ float4 wide_shared[];
    float4* qs = wide_shared;                                   // [head][column][kWideRow]: lane l = q[l + 32 j], j = 0..3
    uint2* ks  = reinterpret_cast<uint2*>(wide_shared + kWideQueryFloat4); // [block][kWideRow]: BF16 k[l + 32 j]
    __shared__ int column_blocks[kScoreTileColumns];

    const int tile = static_cast<int>(blockIdx.y);
    const int c0 = tiles.first[tile], nc = tiles.columns[tile];
    const int t0 = group_begin + c0;
    int tile_blocks = 0;
    for (int c = 0; c < nc; ++c) { tile_blocks = max(tile_blocks, (positions[t0 + c] + 1) / ratio); }
    const int key_tiles = (tile_blocks + kWideBlocks - 1) / kWideBlocks;
    if (tile_blocks <= top_blocks || static_cast<int>(blockIdx.x) >= key_tiles) {
        pdl::trigger_dependents();
        return; // every column of the tile is dense, or this CTA has no key tile
    }
    if (tid < kScoreTileColumns) { column_blocks[tid] = tid < nc ? (positions[t0 + tid] + 1) / ratio : 0; }
    for (int i = tid; i < kScoreTileColumns * kScoreHeadSlots * kIndexDim; i += kWideThreads) {
        const int c = i / (kScoreHeadSlots * kIndexDim), h = (i / kIndexDim) % kScoreHeadSlots, d = i % kIndexDim;
        const float value =
            c < nc && h < index_heads
                ? __bfloat162float(index_q[(static_cast<std::size_t>(t0 + c) * index_heads + h) * kIndexDim + d])
                : 0.0F;
        reinterpret_cast<float*>(&qs[(h * kScoreTileColumns + c) * kWideRow + d % 32])[d / 32] = value;
    }

    // Key staging: thread t carries elements 16 (t % 8) .. + 15 of block t / 8 of a tile (two
    // 16-byte loads; a block's key is 128 contiguous BF16 because its tokens share a page). Element e
    // goes to lane e % 32 of the block's row, BF16 e / 32 of the lane, as the FP32 rows ordered it.
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t0 / width]) * table_stride;
    const int slot_width = kIndexDim / ratio;
    const int load_block = tid / 8, load_part = tid % 8;
    auto load_tile = [&](int key_tile, uint4 (&raw)[2]) {
        const int b = key_tile * kWideBlocks + load_block;
        if (b < tile_blocks) {
            int local;
            const bf16* plane = qsa_pooled_plane(spaces, pooled, table[(ratio * b) >> kPagedKVPageShift], local);
            const auto* src   = reinterpret_cast<const uint4*>(
                plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * local +
                static_cast<std::int64_t>(slot_width) * ((ratio * b) & kPagedKVPageMask) + 16 * load_part);
            raw[0] = src[0];
            raw[1] = src[1];
        } else {
            raw[0] = raw[1] = make_uint4(0U, 0U, 0U, 0U);
        }
    };
    auto store_tile = [&](const uint4 (&raw)[2], uint2* buffer) {
        auto* row          = reinterpret_cast<std::uint16_t*>(buffer + load_block * kWideRow);
        const auto* half   = reinterpret_cast<const std::uint16_t*>(raw);
        const int lane0    = 16 * (load_part % 2), j = load_part / 2;
#pragma unroll
        for (int k = 0; k < 16; ++k) { row[(lane0 + k) * 4 + j] = half[k]; }
    };

    const int warp = tid / 32, lane = tid % 32;
    const int column     = 4 * (warp % 4) + lane / kWideLaneBlocks;
    const int warp_first = 16 * (warp / 4);                       // first block of the warp in a tile
    const int first      = warp_first + lane % kWideLaneBlocks;    // this thread's first block in a tile
    const float scale    = rsqrtf(static_cast<float>(kIndexDim));

    uint4 raw[2];
    int key_tile = static_cast<int>(blockIdx.x);
    load_tile(key_tile, raw);
    store_tile(raw, ks);
    __syncthreads();
    int warp_blocks = 0; // the most blocks a column of the warp sees
    for (int c = 0; c < 4; ++c) { warp_blocks = max(warp_blocks, column_blocks[4 * (warp % 4) + c]); }
    const int blocks = column_blocks[column];
    const float4* query = qs + column * kWideRow;
    for (; key_tile < key_tiles; key_tile += static_cast<int>(gridDim.x)) {
        const int next = key_tile + static_cast<int>(gridDim.x);
        if (next < key_tiles) { load_tile(next, raw); }
        const int b0 = key_tile * kWideBlocks;
        if (b0 + warp_first < warp_blocks) { // warp-uniform: some column of the warp sees a block
            WideStacks stack;
            wide_leaves<0>(query, ks + first * kWideRow, stack);
#pragma unroll
            for (int i = 0; i < kWideThreadBlocks; ++i) {
                const int b = b0 + first + kWideLaneBlocks * i;
                if (b < blocks) {
                    float score = 0.0F;
                    score += fmaxf(stack[0][i][0], 0.0F);
                    if (index_heads > 1) { score += fmaxf(stack[1][i][0], 0.0F); }
                    if (index_heads > 2) { score += fmaxf(stack[2][i][0], 0.0F); }
                    if (index_heads > 3) { score += fmaxf(stack[3][i][0], 0.0F); }
                    scores[static_cast<std::size_t>(c0 + column) * score_stride + b] = score * scale;
                }
            }
        }
        if (next < key_tiles) {
            __syncthreads(); // every warp is done with this tile's keys
            store_tile(raw, ks);
            __syncthreads();
        }
    }
    pdl::trigger_dependents();
}

// ------------------------------------------------------------------------------------------ host

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa select: ") + message); }
}

int ceil_div(std::int64_t a, std::int64_t b) { return static_cast<int>((a + b - 1) / b); }

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

int multiprocessors() {
    static const int count = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

struct Layout {
    std::int32_t group  = 0; // columns per score pass
    std::int32_t stride = 0; // floats per column score row
    std::size_t scores = 0, state = 0, total = 0;
};

Layout layout(const QsaGeometry& g, std::int32_t columns, std::int32_t max_context) {
    Layout l;
    l.group  = std::min(columns, kQsaSelectGroupColumns);
    l.stride = (max_context / g.ratio + 1 + 31) / 32 * 32;
    l.scores = align256(sizeof(float) * static_cast<std::size_t>(l.group) * l.stride);
    // kScoreBins histogram words, then one arrival counter, per column.
    l.state = align256(sizeof(std::uint32_t) * static_cast<std::size_t>(l.group) * (kScoreBins + 1));
    l.total = l.scores + l.state;
    return l;
}

ScoreTiles score_tiles(std::int32_t begin, std::int32_t columns, std::int32_t width) {
    ScoreTiles tiles;
    for (std::int32_t c = 0; c < columns;) {
        const std::int32_t t       = begin + c;
        const std::int32_t row_end = (t / width + 1) * width;
        const std::int32_t n       = std::min({kScoreTileColumns, columns - c, row_end - t});
        tiles.first[tiles.count]   = static_cast<std::uint8_t>(c);
        tiles.columns[tiles.count] = static_cast<std::uint8_t>(n);
        ++tiles.count;
        c += n;
    }
    return tiles;
}

// Opts a kernel into `bytes` of dynamic shared memory beyond the default 48 KiB, once per size.
template <class Kernel>
void reserve_shared(Kernel kernel, std::size_t bytes, std::atomic<std::size_t>& reserved) {
    if (bytes <= 48 * 1024 || bytes <= reserved.load(std::memory_order_acquire)) { return; }
    int device = 0, limit = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    require(bytes <= static_cast<std::size_t>(limit), "max_context needs more shared memory than a CTA has");
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    std::size_t seen = reserved.load(std::memory_order_relaxed);
    while (seen < bytes && !reserved.compare_exchange_weak(seen, bytes, std::memory_order_acq_rel)) {}
}

std::atomic<std::size_t> g_wide_reserved{0};

} // namespace

std::size_t qsa_select_scratch_bytes(const QsaGeometry& geometry, std::int32_t columns, std::int32_t max_context) {
    require(columns > 0 && max_context > 0 && geometry.ratio > 0, "columns and max_context must be positive");
    return layout(geometry, columns, max_context).total;
}

QsaSelectScores qsa_select_scores(const void* scratch, const QsaGeometry& geometry, std::int32_t max_context) {
    return {static_cast<const float*>(scratch), layout(geometry, 1, max_context).stride};
}

void qsa_select(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
                const QsaGeometry& geometry, std::int32_t max_context, void* scratch,
                std::size_t scratch_bytes, const QsaSelectOutput& output, cudaStream_t stream,
                const QsaPageSpaces& spaces) {
    const std::int32_t columns = batch.batch * batch.width;
    require(geometry.index_head_dim == kIndexDim && geometry.index_heads >= 1 &&
                geometry.index_heads <= kScoreHeadSlots && geometry.ratio > 1 && kIndexDim % geometry.ratio == 0 &&
                geometry.budget > 0 && geometry.budget % geometry.ratio == 0,
            "unsupported geometry");
    require(columns > 0 && index_q.data != nullptr && index_q.dtype == DType::BF16 && index_q.is_contiguous() &&
                index_q.ne[0] == kIndexDim && index_q.ne[1] == geometry.index_heads && index_q.ne[2] == columns,
            "index queries disagree with the batch");
    require(pooled_pages.data != nullptr && pooled_pages.dtype == DType::BF16 && pooled_pages.is_contiguous() &&
                pooled_pages.ne[0] == kIndexDim / geometry.ratio,
            "pooled pages must be the layer's BF16 plane");
    require(output.selected != nullptr && output.counts != nullptr, "selection outputs are missing");
    const Layout l = layout(geometry, columns, max_context);
    require(scratch != nullptr && scratch_bytes >= l.total, "scratch is too small");

    const std::int32_t top_blocks = geometry.budget / geometry.ratio;
    const std::int32_t max_blocks = max_context / geometry.ratio;
    auto* base                    = static_cast<unsigned char*>(scratch);
    auto* scores                  = reinterpret_cast<float*>(base);
    auto* state                   = reinterpret_cast<std::uint32_t*>(base + l.scores);

    const auto* index      = static_cast<const bf16*>(index_q.data);
    const auto* pooled     = static_cast<const bf16*>(pooled_pages.data);
    const auto* tables     = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int table_stride = static_cast<int>(batch.block_tables.ne[0]);
    const auto* rows       = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* positions  = static_cast<const std::int32_t*>(batch.positions.data);
    const int sms          = multiprocessors();
    const int max_steps    = ceil_div(max_blocks + 1, kScoreStepBlocks);
    // Wide rows take the register-tiled kernel when a block's key is 16-byte aligned in one page.
    const bool wide = batch.width >= kWideMinWidth && geometry.ratio <= kPagedKVPageSize &&
                      (kIndexDim / geometry.ratio) % 8 == 0;
    if (wide) { reserve_shared(qsa_score_wide_kernel, kWideSharedBytes, g_wide_reserved); }
    for (std::int32_t begin = 0; begin < columns; begin += l.group) {
        const std::int32_t n   = std::min(l.group, columns - begin);
        const ScoreTiles tiles = score_tiles(begin, n, batch.width);
        if (wide) {
            const int ctas = std::clamp(ceil_div(std::int64_t{kWideCtasPerSm} * sms, tiles.count), 1,
                                        std::max(1, ceil_div(max_blocks + 1, kWideBlocks)));
            CUDA_CHECK(pdl::launch_consumer(
                {dim3(static_cast<unsigned>(ctas), static_cast<unsigned>(tiles.count)), dim3(kWideThreads),
                 kWideSharedBytes, stream},
                qsa_score_wide_kernel, index, geometry.index_heads, pooled, tables, table_stride, rows, positions,
                batch.width, geometry.ratio, top_blocks, begin, tiles, scores, l.stride, state,
                l.group * (kScoreBins + 1), spaces));
        } else {
            const int ctas = std::clamp(ceil_div(std::int64_t{kScoreCtasPerSm} * sms, tiles.count), 1,
                                        std::max(1, ceil_div(max_steps, kScoreWarps)));
            CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(ctas), static_cast<unsigned>(tiles.count)),
                                             dim3(kScoreThreads), 0, stream},
                                            qsa_score_kernel, index, geometry.index_heads, pooled, tables,
                                            table_stride, rows, positions, batch.width, geometry.ratio, top_blocks,
                                            begin, tiles, scores, l.stride, state, l.group * (kScoreBins + 1),
                                            spaces));
        }
        exact_topk_select(positions, geometry.ratio, top_blocks, begin, n,
                          static_cast<const float*>(scores), l.stride, state, output.selected,
                          output.counts, max_blocks, true, stream);
    }
}

} // namespace infernix::ops::detail
