// Sparse multi-head latent attention over a paged shared latent cache
// (include/infernix/ops/mla_sparse_attention.h): the latent append, the absorbed query and output
// projections, and the split attention with fused output projection.

#include "infernix/ops/mla_sparse_attention.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
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

// Split attention CTA: 8 warps (256 threads) over kHeadGroup = 16 heads and 16-token MMA tiles.
// Each warp owns one 64-dim latent group (one INT8 scale group) in both the score and value MMA
// phases, using m16n8k16 BF16 Tensor Core MMAs with FP32 accumulation and split-BF16 probabilities.
constexpr int kThreads        = 256;
constexpr int kHeadGroup      = 16; // query heads per split-attention CTA
constexpr int kTokenTile      = 32; // split granularity (shared with attention_merge_kernel)
constexpr int kSplitMmaTokens = 16; // tokens per MMA tile inside a split
constexpr int kMaxSplits      = 65; // automatic splits: (budget tokens + tail) / tile at the top

constexpr std::size_t kChunkBudget = std::size_t{32} << 20; // partials + latent sums per chunk
constexpr std::size_t kMmaChunkBudget = std::size_t{96} << 20; // the tensor-core route's latent sums

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

// Element offset of (row, dimension) in a swizzled BF16 row block: the 16-byte chunks of a row
// are XORed with the row's low three bits, so ldmatrix's eight rows hit eight banks.
__device__ __forceinline__ int swizzled(int row, int chunk) {
    return row * kLatent + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ int swizzled_64(int row, int chunk) {
    return row * 64 + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ unsigned smem_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

// The BF16 high part of (a, b) and the BF16 rounding of what it leaves.
__device__ __forceinline__ void split_bf16(float a, float b, unsigned& hi, unsigned& lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    hi                     = *reinterpret_cast<const unsigned*>(&h);
    lo = pack_bf16(a - __bfloat162float(h.x), b - __bfloat162float(h.y));
}

// Two signed int8 bytes (the low 16 bits of `bytes`, all values -128..127) as an exact BF16 pair.
__device__ __forceinline__ unsigned int8_to_bf16_pair(unsigned bytes) {
    union Bits {
        __nv_bfloat162 pair;
        unsigned bits;
    };
    Bits biased, bias, out;
    biased.bits = __byte_perm(bytes, 0x43004300U, 0x7150) & 0xFF7FFF7FU;
    bias.bits   = 0x43004300U | (bytes & 0x80U) | ((bytes & 0x8000U) << 8);
    out.pair    = __hsub2_rn(biased.pair, bias.pair);
    return out.bits;
}

__device__ __forceinline__ unsigned pack_f16(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ unsigned bf16_pair_to_f16_pair(unsigned v) {
    const float2 f  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    const __half2 h = __float22half2_rn(f);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Shared memory of the split-KV Tensor Core kernel: swizzled BF16 Q [16, 512] and K [16, 512],
// optional INT8 scales [16, 8], per-warp score partials [8][16][17], softmax weights [16][17],
// running softmax stats (m, l, alpha), and the tile's page-resolved offsets. Fits in 48 KiB.
template <bool kInt8>
struct AttentionShared {
    static constexpr int kScorePitch       = kSplitMmaTokens + 1;
    static constexpr std::size_t kQ        = sizeof(bf16) * kHeadGroup * kLatent;
    static constexpr std::size_t kK        = kQ + sizeof(bf16) * kSplitMmaTokens * kLatent;
    static constexpr std::size_t kScales   = kK + (kInt8 ? sizeof(float) * kSplitMmaTokens * kGroups : 0);
    static constexpr std::size_t kPart     = kScales + sizeof(float) * kGroups * kHeadGroup * kScorePitch;
    static constexpr std::size_t kScores   = kPart + sizeof(float) * kHeadGroup * kScorePitch;
    static constexpr std::size_t kStats    = kScores + sizeof(float) * 3 * kHeadGroup;
    static constexpr std::size_t kBytes    = kStats + sizeof(std::int64_t) * kSplitMmaTokens;
};

static_assert(AttentionShared<true>::kBytes <= 48 * 1024, "split MMA shared memory must fit 48 KiB");

// One CTA per (split, head group of 16, column): scores and value-folds one slice of the column's
// attended tokens with m16n8k16 BF16 Tensor Core MMAs (FP32 accumulation, split-BF16 probabilities),
// writing unnormalized latent sums plus running (m, l) into ws.partial for attention_merge_kernel.
template <bool kInt8>
__global__ void __launch_bounds__(kThreads, 2)
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
    using Shared          = AttentionShared<kInt8>;
    constexpr int kChunks = kLatent / 8; // 64 uint4 chunks per BF16 row
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
    const int tail_base = n - n % pool;
    const std::int32_t* table =
        tables + static_cast<std::int64_t>(table_rows[column / width]) * table_stride;

    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_s     = reinterpret_cast<bf16*>(smem);
    auto* k_s     = reinterpret_cast<bf16*>(smem + Shared::kQ);
    auto* scale_s = reinterpret_cast<float (*)[kGroups]>(smem + Shared::kK);
    auto* part_s =
        reinterpret_cast<float (*)[kHeadGroup][Shared::kScorePitch]>(smem + Shared::kScales);
    auto* score_s = reinterpret_cast<float (*)[Shared::kScorePitch]>(smem + Shared::kPart);
    float* m_s    = reinterpret_cast<float*>(smem + Shared::kScores);
    float* l_s    = m_s + kHeadGroup;
    float* a_s    = l_s + kHeadGroup;
    auto* offset  = reinterpret_cast<std::int64_t*>(smem + Shared::kStats);

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int head_base = group * kHeadGroup;

    // Stage the 16 query heads in swizzled BF16 layout once per split.
    {
        const bf16* q = qa + static_cast<std::int64_t>(column) * kLatent * heads +
                        static_cast<std::int64_t>(head_base) * kLatent;
        for (int i = tid; i < kHeadGroup * kChunks; i += kThreads) {
            const int h = i / kChunks, c = i % kChunks;
            *reinterpret_cast<uint4*>(q_s + swizzled(h, c)) =
                *reinterpret_cast<const uint4*>(q + static_cast<std::int64_t>(h) * kLatent + 8 * c);
        }
    }
    if (tid < kHeadGroup) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0F;
    }

    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[nt][i] = 0.0F; }
    }

    const int grp       = lane >> 2;
    const int tid4      = lane & 3;
    const int a_row_off = lane & 15;
    const int a_ch_off  = lane >> 4;
    const int b_row_off = 8 * (lane >> 4) + (lane & 7);
    const int b_ch_off  = (lane >> 3) & 1;
    const int val_r0    = lane >> 2;
    const int val_k0    = 2 * (lane & 3);
    const int sm_h      = tid >> 4;
    const int sm_tok    = tid & 15;

    for (int first = begin; first < end; first += kSplitMmaTokens) {
        const int n_tok = min(kSplitMmaTokens, end - first);
        if (tid < kSplitMmaTokens) {
            const int index = first + tid;
            std::int64_t at = -1;
            if (tid < n_tok) {
                const int position =
                    dense ? index
                          : attended_position(index, count, pool, pools, top, column, tail_base);
                at = (static_cast<std::int64_t>(table[position >> kPagedKVPageShift]) *
                          kPagedKVPageSize +
                      (position & kPagedKVPageMask)) *
                     kLatent;
            }
            offset[tid] = at;
        }
        __syncthreads();

        // Stage the 16 latent rows into swizzled BF16 (INT8 codes convert losslessly to BF16).
        if constexpr (kInt8) {
            for (int i = tid; i < kSplitMmaTokens * (kLatent / 16); i += kThreads) {
                const int j = i / (kLatent / 16), ch = i % (kLatent / 16);
                uint4 lo8 = make_uint4(0, 0, 0, 0), hi8 = make_uint4(0, 0, 0, 0);
                if (offset[j] >= 0) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(pages + offset[j] + 16 * ch);
                    lo8 = make_uint4(int8_to_bf16_pair(raw.x), int8_to_bf16_pair(raw.x >> 16),
                                     int8_to_bf16_pair(raw.y), int8_to_bf16_pair(raw.y >> 16));
                    hi8 = make_uint4(int8_to_bf16_pair(raw.z), int8_to_bf16_pair(raw.z >> 16),
                                     int8_to_bf16_pair(raw.w), int8_to_bf16_pair(raw.w >> 16));
                }
                *reinterpret_cast<uint4*>(k_s + swizzled(j, 2 * ch))     = lo8;
                *reinterpret_cast<uint4*>(k_s + swizzled(j, 2 * ch + 1)) = hi8;
            }
            if (tid < kSplitMmaTokens * kGroups) {
                const int j   = tid / kGroups, g = tid % kGroups;
                scale_s[j][g] = offset[j] >= 0 ? __half2float(scales[offset[j] / kGroup + g]) : 0.0F;
            }
        } else {
            for (int i = tid; i < kSplitMmaTokens * kChunks; i += kThreads) {
                const int j = i / kChunks, ch = i % kChunks;
                *reinterpret_cast<uint4*>(k_s + swizzled(j, ch)) =
                    offset[j] >= 0
                        ? *reinterpret_cast<const uint4*>(pages + sizeof(bf16) * offset[j] + 16 * ch)
                        : make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Score phase: warp `warp` computes group `warp` (64 dims) of Q [16, 64] * K [16, 64]^T.
        {
            const int g = warp;
            float p0[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float p1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int chunk0 = g * 8 + 2 * ks;
                unsigned a0, a1, a2, a3, b0, b1, b2, b3;
                ldmatrix_x4(a0, a1, a2, a3,
                            smem_address(q_s + swizzled(a_row_off, chunk0 + a_ch_off)));
                ldmatrix_x4(b0, b1, b2, b3,
                            smem_address(k_s + swizzled(b_row_off, chunk0 + b_ch_off)));
                mma_bf16(p0[0], p0[1], p0[2], p0[3], a0, a1, a2, a3, b0, b1);
                mma_bf16(p1[0], p1[1], p1[2], p1[3], a0, a1, a2, a3, b2, b3);
            }
            const int t0 = 2 * tid4;
            if constexpr (kInt8) {
                const float s0 = scale_s[t0][g], s1 = scale_s[t0 + 1][g];
                const float s8 = scale_s[8 + t0][g], s9 = scale_s[8 + t0 + 1][g];
                p0[0] *= s0;
                p0[1] *= s1;
                p0[2] *= s0;
                p0[3] *= s1;
                p1[0] *= s8;
                p1[1] *= s9;
                p1[2] *= s8;
                p1[3] *= s9;
            }
            part_s[g][grp][t0]         = p0[0];
            part_s[g][grp][t0 + 1]     = p0[1];
            part_s[g][grp + 8][t0]     = p0[2];
            part_s[g][grp + 8][t0 + 1] = p0[3];
            part_s[g][grp][8 + t0]     = p1[0];
            part_s[g][grp][9 + t0]     = p1[1];
            part_s[g][grp + 8][8 + t0] = p1[2];
            part_s[g][grp + 8][9 + t0] = p1[3];
        }
        __syncthreads();

        // Reduce the 8 latent-group partials in fixed group order and update online softmax:
        // 16 threads per head (one token per thread), 256 threads total.
        {
            float dot = 0.0F;
#pragma unroll
            for (int g = 0; g < kGroups; ++g) { dot += part_s[g][sm_h][sm_tok]; }
            const float sc = sm_tok < n_tok ? dot * scale : -INFINITY;
            float tmax     = sc;
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 1, 16));
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 2, 16));
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 4, 16));
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 8, 16));
            const float m_old     = m_s[sm_h];
            const float m_new     = fmaxf(m_old, tmax);
            const float prob      = sc == -INFINITY ? 0.0F : expf(sc - m_new);
            score_s[sm_h][sm_tok] = prob;
            float sum             = prob;
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1, 16);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2, 16);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4, 16);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 8, 16);
            if (sm_tok == 0) {
                const float alpha = m_old == -INFINITY ? 0.0F : expf(m_old - m_new);
                a_s[sm_h]         = alpha;
                l_s[sm_h]         = l_s[sm_h] * alpha + sum;
                m_s[sm_h]         = m_new;
            }
        }
        __syncthreads();

        // Value phase: warp `warp` folds latent group `warp` (64 dims) of all 16 heads.
        {
            const int g        = warp;
            const float alpha0 = a_s[val_r0], alpha1 = a_s[val_r0 + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[nt][0] *= alpha0;
                acc[nt][1] *= alpha0;
                acc[nt][2] *= alpha1;
                acc[nt][3] *= alpha1;
            }
            if constexpr (kInt8) {
                // Normalize group scales by s_max so P * (scale / s_max) fits [-1, 1] in FP16
                // without underflow or overflow; unscaled INT8 codes [-128, 127] are exact in FP16.
                float s_max = lane < 16 ? fabsf(scale_s[lane][g]) : 0.0F;
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 1));
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 2));
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 4));
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 8));
                s_max       = __shfl_sync(0xFFFFFFFFU, s_max, 0);
                const float inv_smax = s_max > 0.0F ? (1.0F / s_max) : 0.0F;
                const float w0       = scale_s[val_k0][g] * inv_smax;
                const float w1       = scale_s[val_k0 + 1][g] * inv_smax;
                const float w8       = scale_s[val_k0 + 8][g] * inv_smax;
                const float w9       = scale_s[val_k0 + 9][g] * inv_smax;
                const unsigned a0 =
                    pack_f16(score_s[val_r0][val_k0] * w0, score_s[val_r0][val_k0 + 1] * w1);
                const unsigned a1 =
                    pack_f16(score_s[val_r0 + 8][val_k0] * w0, score_s[val_r0 + 8][val_k0 + 1] * w1);
                const unsigned a2 =
                    pack_f16(score_s[val_r0][val_k0 + 8] * w8, score_s[val_r0][val_k0 + 9] * w9);
                const unsigned a3 = pack_f16(score_s[val_r0 + 8][val_k0 + 8] * w8,
                                             score_s[val_r0 + 8][val_k0 + 9] * w9);
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4_t(
                        b0, b1, b2, b3,
                        smem_address(k_s + swizzled(lane & 15, g * 8 + 2 * np + (lane >> 4))));
                    b0 = bf16_pair_to_f16_pair(b0);
                    b1 = bf16_pair_to_f16_pair(b1);
                    b2 = bf16_pair_to_f16_pair(b2);
                    b3 = bf16_pair_to_f16_pair(b3);
                    const int nt0 = 2 * np;
                    const int nt1 = 2 * np + 1;
                    float v0 = 0.0F, v1 = 0.0F, v2 = 0.0F, v3 = 0.0F;
                    mma_f16(v0, v1, v2, v3, a0, a1, a2, a3, b0, b1);
                    acc[nt0][0] = fmaf(s_max, v0, acc[nt0][0]);
                    acc[nt0][1] = fmaf(s_max, v1, acc[nt0][1]);
                    acc[nt0][2] = fmaf(s_max, v2, acc[nt0][2]);
                    acc[nt0][3] = fmaf(s_max, v3, acc[nt0][3]);
                    v0 = v1 = v2 = v3 = 0.0F;
                    mma_f16(v0, v1, v2, v3, a0, a1, a2, a3, b2, b3);
                    acc[nt1][0] = fmaf(s_max, v0, acc[nt1][0]);
                    acc[nt1][1] = fmaf(s_max, v1, acc[nt1][1]);
                    acc[nt1][2] = fmaf(s_max, v2, acc[nt1][2]);
                    acc[nt1][3] = fmaf(s_max, v3, acc[nt1][3]);
                }
            } else {
                unsigned a_hi[4], a_lo[4];
                split_bf16(score_s[val_r0][val_k0], score_s[val_r0][val_k0 + 1], a_hi[0], a_lo[0]);
                split_bf16(score_s[val_r0 + 8][val_k0], score_s[val_r0 + 8][val_k0 + 1], a_hi[1],
                           a_lo[1]);
                split_bf16(score_s[val_r0][val_k0 + 8], score_s[val_r0][val_k0 + 9], a_hi[2],
                           a_lo[2]);
                split_bf16(score_s[val_r0 + 8][val_k0 + 8], score_s[val_r0 + 8][val_k0 + 9],
                           a_hi[3], a_lo[3]);
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4_t(
                        b0, b1, b2, b3,
                        smem_address(k_s + swizzled(lane & 15, g * 8 + 2 * np + (lane >> 4))));
                    const int nt0 = 2 * np;
                    const int nt1 = 2 * np + 1;
                    mma_bf16(acc[nt0][0], acc[nt0][1], acc[nt0][2], acc[nt0][3], a_hi[0], a_hi[1],
                             a_hi[2], a_hi[3], b0, b1);
                    mma_bf16(acc[nt0][0], acc[nt0][1], acc[nt0][2], acc[nt0][3], a_lo[0], a_lo[1],
                             a_lo[2], a_lo[3], b0, b1);
                    mma_bf16(acc[nt1][0], acc[nt1][1], acc[nt1][2], acc[nt1][3], a_hi[0], a_hi[1],
                             a_hi[2], a_hi[3], b2, b3);
                    mma_bf16(acc[nt1][0], acc[nt1][1], acc[nt1][2], acc[nt1][3], a_lo[0], a_lo[1],
                             a_lo[2], a_lo[3], b2, b3);
                }
            }
        }
        __syncthreads();
    }

    // Write this split's unnormalized latent sums plus (m, l) for attention_merge_kernel.
    float* const base = partial + (static_cast<std::size_t>(t) * splits_grid * groups +
                                   static_cast<std::size_t>(split) * groups + group) *
                                      (kHeadGroup * (kLatent + 2));
    const int d0      = 64 * warp + 2 * tid4;
    float* const row0 = base + static_cast<std::size_t>(val_r0) * (kLatent + 2);
    float* const row1 = base + static_cast<std::size_t>(val_r0 + 8) * (kLatent + 2);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<float2*>(row0 + d0 + 8 * nt) = make_float2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<float2*>(row1 + d0 + 8 * nt) = make_float2(acc[nt][2], acc[nt][3]);
    }
    if (tid < kHeadGroup) {
        float* const hslot = base + static_cast<std::size_t>(tid) * (kLatent + 2);
        hslot[kLatent]     = m_s[tid];
        hslot[kLatent + 1] = l_s[tid];
    }
}

// The tensor-core prompt route (MlaAttentionExecutionEnvelope::tensor_cores): one CTA per column
// (all 64 heads sharing one KV decode and one L2 fetch), 16 warps (512 threads), and 16-token KV
// tiles. For INT8 caches, each warp prefetches one token's raw INT8 codes and FP16 scales via
// cp.async into double-buffered raw storage while the current tile's score, softmax and value
// phases execute. In the score phase, 16 warps cover 4 head tiles x 2 token tiles x 2 K-halves
// (4 latent groups each); in the value phase, 16 warps split the 8 latent groups into 32-dim
// halves (64 accumulator registers per thread).
constexpr int kMmaThreads    = 512;
constexpr int kMmaTokens     = 16;
constexpr int kMmaHeads      = 64;
constexpr int kMmaHeadBlocks = 1;
constexpr int kMmaChunks     = kLatent / 8; // 16-byte BF16 chunks per row

// Pair-column index (0..7) for float2 access into a [64][16] score plane: XORing bit 2 with
// ((row >> 1) & 1) gives 2-way (minimal for 32-lane LDS.64) shared-memory traffic with no padding.
__device__ __forceinline__ int score_pair_col(int row, int pair) {
    return (pair ^ ((row & 2) << 1)) << 1;
}

template <bool kInt8>
struct MmaShared {
    static constexpr int kScalePitch         = kMmaTokens + 1;
    static constexpr std::size_t kQ          = sizeof(bf16) * kMmaHeads * kLatent;
    static constexpr std::size_t kK          = kQ + sizeof(bf16) * kMmaTokens * kLatent;
    static constexpr std::size_t kKRaw       = kK + (kInt8 ? sizeof(unsigned char) * kMmaTokens * kLatent : 0);
    static constexpr std::size_t kScaleRaw   = kKRaw + (kInt8 ? sizeof(__half) * kMmaTokens * kGroups : 0);
    static constexpr std::size_t kScales     =
        kScaleRaw + (kInt8 ? sizeof(float) * kGroups * kScalePitch : 0);
    static constexpr std::size_t kScoreParts =
        ((kScales + 15) / 16 * 16) + sizeof(float) * 2 * kMmaHeads * kMmaTokens;
    static constexpr std::size_t kScoreOff   = (kScales + 15) / 16 * 16;
    static constexpr std::size_t kBytes      = kScoreParts + sizeof(float) * 3 * kMmaHeads;
};

static_assert(MmaShared<true>::kBytes <= 99 * 1024, "MMA attention shared memory must fit 99 KiB");
static_assert(MmaShared<false>::kBytes <= 99 * 1024, "MMA attention shared memory must fit 99 KiB");

template <bool kInt8>
__global__ void __launch_bounds__(kMmaThreads, 1)
    attention_mma_kernel(const bf16* __restrict__ qa, const unsigned char* __restrict__ pages,
                         const __half* __restrict__ scales, const std::int32_t* __restrict__ tables,
                         int table_stride, const std::int32_t* __restrict__ table_rows,
                         const std::int32_t* __restrict__ positions, int width,
                         const std::int32_t* __restrict__ pools,
                         const std::int32_t* __restrict__ counts, int top, int pool, float scale,
                         int column_begin, float* __restrict__ ol) {
    using Shared = MmaShared<kInt8>;
    pdl::enter();
    const int t      = static_cast<int>(blockIdx.x);
    const int column = column_begin + t;
    const int count  = counts == nullptr ? -1 : counts[column];
    const bool dense = pools == nullptr || counts == nullptr || count < 0;
    const int n      = positions[column] + 1;
    const int total  = attended_total(dense ? -1 : count, pool, n);
    const int tail_base = n - n % pool;
    const std::int32_t* table =
        tables + static_cast<std::int64_t>(table_rows[column / width]) * table_stride;

    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_s          = reinterpret_cast<bf16*>(smem);
    auto* k_s          = reinterpret_cast<bf16*>(smem + Shared::kQ);
    auto* k_raw_s      = reinterpret_cast<unsigned char*>(smem + Shared::kK);
    auto* scale_raw_s  = reinterpret_cast<__half*>(smem + Shared::kKRaw);
    auto* scale_s      = reinterpret_cast<float (*)[Shared::kScalePitch]>(smem + Shared::kScaleRaw);
    auto* score_part_s =
        reinterpret_cast<float (*)[kMmaHeads][kMmaTokens]>(smem + Shared::kScoreOff);
    auto* score_s = score_part_s[0];
    float* m_s    = reinterpret_cast<float*>(smem + Shared::kScoreParts);
    float* l_s    = m_s + kMmaHeads;
    float* a_s    = l_s + kMmaHeads;

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;

    // The column's queries, every head, swizzled.
    {
        const bf16* q = qa + static_cast<std::int64_t>(column) * kLatent * kMmaHeads;
        for (int i = tid; i < kMmaHeads * kMmaChunks; i += kMmaThreads) {
            const int h = i / kMmaChunks, c = i % kMmaChunks;
            *reinterpret_cast<uint4*>(q_s + swizzled(h, c)) =
                *reinterpret_cast<const uint4*>(q + static_cast<std::int64_t>(h) * kLatent + 8 * c);
        }
    }
    if (tid < kMmaHeads) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0F;
    }

    // Score tiling: 4 head tiles x 16 tokens/warp (2x Q reuse, 2 independent MMA chains) x 4 K-quarters (2 groups each).
    const int score_mt = warp & 3;
    const int score_kq = warp >> 2;
    // Value tiling: 8 latent groups x 2 32-dim halves (4 n-tiles per warp) across all 64 heads.
    const int val_g    = warp >> 1;
    const int val_half = warp & 1;

    float acc[4][4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[mt][nt][i] = 0.0F; }
        }
    }

    // Issue cp.async for 16 tokens starting at `tile_first`: 16 warps = 1 token per warp.
    const auto issue_tile = [&](int tile_first) {
        const int j      = warp;
        const int index  = tile_first + j;
        const bool valid = index < total;
        std::int64_t at  = 0;
        if (valid) {
            const int position =
                dense ? index : attended_position(index, count, pool, pools, top, column, tail_base);
            at = (static_cast<std::int64_t>(table[position >> kPagedKVPageShift]) * kPagedKVPageSize +
                  (position & kPagedKVPageMask)) *
                 kLatent;
        }
        if constexpr (kInt8) {
            cp_async_zfill<16>(&k_raw_s[j * kLatent + 16 * lane],
                               valid ? (pages + at + 16 * lane) : pages, valid ? 16 : 0);
            if (lane == 0) {
                cp_async_zfill<16>(&scale_raw_s[j * kGroups],
                                   valid ? (scales + at / kGroup) : scales, valid ? 16 : 0);
            }
        } else {
            const unsigned char* base = valid ? (pages + sizeof(bf16) * at) : pages;
            cp_async_zfill<16>(k_s + swizzled(j, lane), base + 16 * lane, valid ? 16 : 0);
            cp_async_zfill<16>(k_s + swizzled(j, lane + 32), base + 16 * (lane + 32),
                               valid ? 16 : 0);
        }
        cp_commit();
    };

    if constexpr (kInt8) {
        if (total > 0) { issue_tile(0); }
        cp_wait<0>();
    }
    __syncthreads();

    for (int first = 0; first < total; first += kMmaTokens) {
        const int n_tok = min(kMmaTokens, total - first);
        if constexpr (kInt8) {
            // Interleave even/odd token parity across 4-lane groups so every 8-lane STS.128 wave
            // covers all 32 banks of swizzled k_s with zero store bank conflicts, while k_raw_s
            // reads stay conflict-free at 16 * lane.
            const int dj    = (warp & ~1) | (((lane >> 2) ^ warp) & 1);
            const uint4 raw = *reinterpret_cast<const uint4*>(&k_raw_s[dj * kLatent + 16 * lane]);
            const uint4 lo8 = make_uint4(int8_to_bf16_pair(raw.x), int8_to_bf16_pair(raw.x >> 16),
                                         int8_to_bf16_pair(raw.y), int8_to_bf16_pair(raw.y >> 16));
            const uint4 hi8 = make_uint4(int8_to_bf16_pair(raw.z), int8_to_bf16_pair(raw.z >> 16),
                                         int8_to_bf16_pair(raw.w), int8_to_bf16_pair(raw.w >> 16));
            *reinterpret_cast<uint4*>(k_s + swizzled(dj, 2 * lane))     = lo8;
            *reinterpret_cast<uint4*>(k_s + swizzled(dj, 2 * lane + 1)) = hi8;
            if (lane < kGroups) {
                scale_s[lane][warp] = __half2float(scale_raw_s[warp * kGroups + lane]);
            }
            __syncthreads();
            if (first + kMmaTokens < total) { issue_tile(first + kMmaTokens); }
        } else {
            issue_tile(first);
            cp_wait<0>();
            __syncthreads();
        }

        // Scores: 4 head tiles x 16 tokens/warp (both n-tiles, 2x Q reuse) x 4 K-quarters (2 groups).
        {
            const int pair0 = lane & 3;
            const int pair1 = pair0 + 4;
            const int tok0  = 2 * pair0;
            float s0[4]     = {0.0F, 0.0F, 0.0F, 0.0F};
            float s1[4]     = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
            for (int lg = 0; lg < 2; ++lg) {
                const int g = 2 * score_kq + lg;
                float p0[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                float p1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int kp = 0; kp < 2; ++kp) {
                    const int chunk0 = g * 8 + kp * 4;
                    unsigned b0[4], b1[4], a0[4], a1[4];
                    ldmatrix_x4(b0[0], b0[1], b0[2], b0[3],
                                smem_address(k_s + swizzled(lane & 7, chunk0 + (lane >> 3))));
                    ldmatrix_x4(b1[0], b1[1], b1[2], b1[3],
                                smem_address(k_s + swizzled(8 + (lane & 7), chunk0 + (lane >> 3))));
                    ldmatrix_x4(a0[0], a0[1], a0[2], a0[3],
                                smem_address(q_s + swizzled(16 * score_mt + (lane & 15),
                                                            chunk0 + (lane >> 4))));
                    ldmatrix_x4(a1[0], a1[1], a1[2], a1[3],
                                smem_address(q_s + swizzled(16 * score_mt + (lane & 15),
                                                            chunk0 + 2 + (lane >> 4))));
                    mma_bf16(p0[0], p0[1], p0[2], p0[3], a0[0], a0[1], a0[2], a0[3], b0[0], b0[1]);
                    mma_bf16(p1[0], p1[1], p1[2], p1[3], a0[0], a0[1], a0[2], a0[3], b1[0], b1[1]);
                    mma_bf16(p0[0], p0[1], p0[2], p0[3], a1[0], a1[1], a1[2], a1[3], b0[2], b0[3]);
                    mma_bf16(p1[0], p1[1], p1[2], p1[3], a1[0], a1[1], a1[2], a1[3], b1[2], b1[3]);
                }
                if constexpr (kInt8) {
                    const float sc0 = scale_s[g][tok0], sc1 = scale_s[g][tok0 + 1];
                    const float sc8 = scale_s[g][tok0 + 8], sc9 = scale_s[g][tok0 + 9];
                    s0[0] = fmaf(p0[0], sc0, s0[0]);
                    s0[1] = fmaf(p0[1], sc1, s0[1]);
                    s0[2] = fmaf(p0[2], sc0, s0[2]);
                    s0[3] = fmaf(p0[3], sc1, s0[3]);
                    s1[0] = fmaf(p1[0], sc8, s1[0]);
                    s1[1] = fmaf(p1[1], sc9, s1[1]);
                    s1[2] = fmaf(p1[2], sc8, s1[2]);
                    s1[3] = fmaf(p1[3], sc9, s1[3]);
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s0[i] += p0[i];
                        s1[i] += p1[i];
                    }
                }
            }
            const int row = 16 * score_mt + (lane >> 2);
            const int c0a = score_pair_col(row, pair0), c0b = score_pair_col(row + 8, pair0);
            const int c1a = score_pair_col(row, pair1), c1b = score_pair_col(row + 8, pair1);
            const int slot = score_kq & 1;
            if (score_kq < 2) {
                *reinterpret_cast<float2*>(&score_part_s[slot][row][c0a])     = make_float2(s0[0], s0[1]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row + 8][c0b]) = make_float2(s0[2], s0[3]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row][c1a])     = make_float2(s1[0], s1[1]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row + 8][c1b]) = make_float2(s1[2], s1[3]);
            }
            __syncthreads();
            if (score_kq >= 2) {
                const float2 a0 = *reinterpret_cast<const float2*>(&score_part_s[slot][row][c0a]);
                const float2 b0 = *reinterpret_cast<const float2*>(&score_part_s[slot][row + 8][c0b]);
                const float2 a1 = *reinterpret_cast<const float2*>(&score_part_s[slot][row][c1a]);
                const float2 b1 = *reinterpret_cast<const float2*>(&score_part_s[slot][row + 8][c1b]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row][c0a]) =
                    make_float2(a0.x + s0[0], a0.y + s0[1]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row + 8][c0b]) =
                    make_float2(b0.x + s0[2], b0.y + s0[3]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row][c1a]) =
                    make_float2(a1.x + s1[0], a1.y + s1[1]);
                *reinterpret_cast<float2*>(&score_part_s[slot][row + 8][c1b]) =
                    make_float2(b1.x + s1[2], b1.y + s1[3]);
            }
        }
        __syncthreads();

        // Online softmax: 8 threads per head, 2 tokens (one float2 pair) per thread.
        {
            const int row  = tid >> 3;
            const int pair = tid & 7;
            const int col  = score_pair_col(row, pair);
            const float2 a = *reinterpret_cast<const float2*>(&score_part_s[0][row][col]);
            const float2 b = *reinterpret_cast<const float2*>(&score_part_s[1][row][col]);
            const int t0 = 2 * pair, t1 = t0 + 1;
            const float v0 = t0 < n_tok ? (a.x + b.x) * scale : -INFINITY;
            const float v1 = t1 < n_tok ? (a.y + b.y) * scale : -INFINITY;
            float tmax     = fmaxf(v0, v1);
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 1, 8));
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 2, 8));
            tmax           = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFU, tmax, 4, 8));
            const float m_old = m_s[row];
            const float m_new = fmaxf(m_old, tmax);
            const float p0    = v0 == -INFINITY ? 0.0F : expf(v0 - m_new);
            const float p1    = v1 == -INFINITY ? 0.0F : expf(v1 - m_new);
            *reinterpret_cast<float2*>(&score_s[row][col]) = make_float2(p0, p1);
            float sum = p0 + p1;
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1, 8);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2, 8);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4, 8);
            if (pair == 0) {
                const float alpha = m_old == -INFINITY ? 0.0F : expf(m_old - m_new);
                a_s[row]          = alpha;
                l_s[row]          = l_s[row] * alpha + sum;
                m_s[row]          = m_new;
            }
        }
        __syncthreads();

        // Values: half `val_half` (32 dims) of latent group `val_g` for all 64 heads.
        {
            const int g = val_g;
            if constexpr (kInt8) {
                float s_max = lane < 16 ? fabsf(scale_s[g][lane]) : 0.0F;
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 1));
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 2));
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 4));
                s_max       = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFFU, s_max, 8));
                s_max       = __shfl_sync(0xFFFFFFFFU, s_max, 0);
                const float inv_smax = s_max > 0.0F ? (1.0F / s_max) : 0.0F;
                const int p_lo       = lane & 3;
                const int p_hi       = p_lo + 4;
                const int k0         = 2 * p_lo;
                const float w0       = scale_s[g][k0] * inv_smax;
                const float w1       = scale_s[g][k0 + 1] * inv_smax;
                const float w8       = scale_s[g][k0 + 8] * inv_smax;
                const float w9       = scale_s[g][k0 + 9] * inv_smax;
                unsigned a_f16[4][4];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int r0       = 16 * mt + (lane >> 2);
                    const float alpha0 = a_s[r0], alpha1 = a_s[r0 + 8];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        acc[mt][nt][0] *= alpha0;
                        acc[mt][nt][1] *= alpha0;
                        acc[mt][nt][2] *= alpha1;
                        acc[mt][nt][3] *= alpha1;
                    }
                    const float2 q0 =
                        *reinterpret_cast<const float2*>(&score_s[r0][score_pair_col(r0, p_lo)]);
                    const float2 q1 = *reinterpret_cast<const float2*>(
                        &score_s[r0 + 8][score_pair_col(r0 + 8, p_lo)]);
                    const float2 q2 =
                        *reinterpret_cast<const float2*>(&score_s[r0][score_pair_col(r0, p_hi)]);
                    const float2 q3 = *reinterpret_cast<const float2*>(
                        &score_s[r0 + 8][score_pair_col(r0 + 8, p_hi)]);
                    a_f16[mt][0] = pack_f16(q0.x * w0, q0.y * w1);
                    a_f16[mt][1] = pack_f16(q1.x * w0, q1.y * w1);
                    a_f16[mt][2] = pack_f16(q2.x * w8, q2.y * w9);
                    a_f16[mt][3] = pack_f16(q3.x * w8, q3.y * w9);
                }
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    const int chunk_base = g * 8 + val_half * 4 + 2 * np;
                    unsigned b[4];
                    ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                                  smem_address(k_s + swizzled(lane & 15, chunk_base + (lane >> 4))));
                    b[0] = bf16_pair_to_f16_pair(b[0]);
                    b[1] = bf16_pair_to_f16_pair(b[1]);
                    b[2] = bf16_pair_to_f16_pair(b[2]);
                    b[3] = bf16_pair_to_f16_pair(b[3]);
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int nt = 2 * np + half;
                        float v[4][4];
#pragma unroll
                        for (int mt = 0; mt < 4; ++mt) {
                            v[mt][0] = v[mt][1] = v[mt][2] = v[mt][3] = 0.0F;
                            mma_f16(v[mt][0], v[mt][1], v[mt][2], v[mt][3], a_f16[mt][0],
                                    a_f16[mt][1], a_f16[mt][2], a_f16[mt][3], b[2 * half],
                                    b[2 * half + 1]);
                        }
#pragma unroll
                        for (int mt = 0; mt < 4; ++mt) {
                            acc[mt][nt][0] = fmaf(s_max, v[mt][0], acc[mt][nt][0]);
                            acc[mt][nt][1] = fmaf(s_max, v[mt][1], acc[mt][nt][1]);
                            acc[mt][nt][2] = fmaf(s_max, v[mt][2], acc[mt][nt][2]);
                            acc[mt][nt][3] = fmaf(s_max, v[mt][3], acc[mt][nt][3]);
                        }
                    }
                }
            } else {
                const int p_lo = lane & 3;
                const int p_hi = p_lo + 4;
                unsigned a_hi[4][4], a_lo[4][4];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int r0       = 16 * mt + (lane >> 2);
                    const float alpha0 = a_s[r0], alpha1 = a_s[r0 + 8];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        acc[mt][nt][0] *= alpha0;
                        acc[mt][nt][1] *= alpha0;
                        acc[mt][nt][2] *= alpha1;
                        acc[mt][nt][3] *= alpha1;
                    }
                    const float2 q0 =
                        *reinterpret_cast<const float2*>(&score_s[r0][score_pair_col(r0, p_lo)]);
                    const float2 q1 = *reinterpret_cast<const float2*>(
                        &score_s[r0 + 8][score_pair_col(r0 + 8, p_lo)]);
                    const float2 q2 =
                        *reinterpret_cast<const float2*>(&score_s[r0][score_pair_col(r0, p_hi)]);
                    const float2 q3 = *reinterpret_cast<const float2*>(
                        &score_s[r0 + 8][score_pair_col(r0 + 8, p_hi)]);
                    split_bf16(q0.x, q0.y, a_hi[mt][0], a_lo[mt][0]);
                    split_bf16(q1.x, q1.y, a_hi[mt][1], a_lo[mt][1]);
                    split_bf16(q2.x, q2.y, a_hi[mt][2], a_lo[mt][2]);
                    split_bf16(q3.x, q3.y, a_hi[mt][3], a_lo[mt][3]);
                }
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    const int chunk_base = g * 8 + val_half * 4 + 2 * np;
                    unsigned b[4];
                    ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                                  smem_address(k_s + swizzled(lane & 15, chunk_base + (lane >> 4))));
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int nt = 2 * np + half;
#pragma unroll
                        for (int mt = 0; mt < 4; ++mt) {
                            mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                                     a_hi[mt][0], a_hi[mt][1], a_hi[mt][2], a_hi[mt][3],
                                     b[2 * half], b[2 * half + 1]);
                            mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                                     a_lo[mt][0], a_lo[mt][1], a_lo[mt][2], a_lo[mt][3],
                                     b[2 * half], b[2 * half + 1]);
                        }
                    }
                }
            }
        }
        if constexpr (kInt8) {
            if (first + kMmaTokens < total) { cp_wait<0>(); }
        }
        __syncthreads();
    }

    // The normalized latent sums: ol (d, h, t) FP32.
    float* out = ol + static_cast<std::size_t>(t) * kLatent * kMmaHeads;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int r0     = 16 * mt + (lane >> 2);
        const float inv0 = 1.0F / l_s[r0], inv1 = 1.0F / l_s[r0 + 8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int d = 64 * val_g + 32 * val_half + 8 * nt + 2 * (lane & 3);
            *reinterpret_cast<float2*>(out + d + static_cast<std::size_t>(r0) * kLatent) =
                make_float2(acc[mt][nt][0] * inv0, acc[mt][nt][1] * inv0);
            *reinterpret_cast<float2*>(out + d + static_cast<std::size_t>(r0 + 8) * kLatent) =
                make_float2(acc[mt][nt][2] * inv1, acc[mt][nt][3] * inv1);
        }
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

// Per-head batched BF16 Tensor Core GEMM used by both mla_absorbed_q_projection (M=512, K=256,
// In=bf16) and mla_latent_output_projection / attention output projection (M=256, K=512, In=bf16
// or float). Each output element (m, h, c) executes the exact same sequence of K/16 m16n8k16
// MMAs (with split-BF16 hi+lo when In=float) regardless of call width, guaranteeing strict
// bitwise column invariance between kNtile=8 (decode/verify) and kNtile=64 (prefill).
template <typename In, int kM, int kK, int kNtile>
__global__ void __launch_bounds__(128, 4)
    mla_head_gemm_kernel(const In* __restrict__ x, const bf16* __restrict__ weight, int heads,
                         int columns, int column_begin, bf16* __restrict__ out) {
    static_assert(kM % 64 == 0 && kK % 64 == 0 && (kNtile == 8 || kNtile == 64));
    constexpr int kMtile       = 64;
    constexpr int kKtile       = 64;
    constexpr int kChunks64    = kKtile / 8; // 8 uint4 chunks per 64-BF16 row
    constexpr int kOutStride   = kMtile + 8; // padded to avoid smem bank conflicts
    constexpr bool kSplitInput = std::is_same_v<In, float>;

    pdl::enter();
    const int c0    = static_cast<int>(blockIdx.x) * kNtile;
    const int h     = static_cast<int>(blockIdx.y);
    const int m0    = static_cast<int>(blockIdx.z) * kMtile;
    const int n_col = min(kNtile, columns - c0);
    if (n_col <= 0) { return; }

    __shared__ __align__(16) bf16 a_s[kMtile * kKtile];
    __shared__ __align__(16) bf16 b_hi_s[kNtile * kKtile];
    __shared__ __align__(16) bf16 b_lo_s[kSplitInput ? (kNtile * kKtile) : 1];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;
    const int tid4 = lane & 3;
    const int a_r  = lane & 15;
    const int a_c  = lane >> 4;

    const bf16* const w_head = weight + static_cast<std::int64_t>(h) * kM * kK +
                               static_cast<std::int64_t>(m0) * kK;
    const In* const x_head = x + static_cast<std::int64_t>(h) * kK;

    if constexpr (kNtile == 8) {
        // 4 warps split M (16 rows each) x 8 columns (1 m16n8k16 tile per warp).
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        const int wm = warp;
        const int b_r = lane & 7;
        const int b_c = (lane >> 3) & 1;

        for (int k_base = 0; k_base < kK; k_base += kKtile) {
            // Load A [64, 64] BF16 (4 uint4 per thread).
#pragma unroll
            for (int i = tid; i < kMtile * kChunks64; i += 128) {
                const int r = i >> 3, ch = i & 7;
                *reinterpret_cast<uint4*>(a_s + swizzled_64(r, ch)) =
                    *reinterpret_cast<const uint4*>(w_head + static_cast<std::int64_t>(r) * kK +
                                                    k_base + ch * 8);
            }
            // Load B [8, 64] (bf16 or split_bf16 from float).
            if (tid < kNtile * kChunks64) {
                const int n = tid >> 3, ch = tid & 7;
                if constexpr (!kSplitInput) {
                    *reinterpret_cast<uint4*>(b_hi_s + swizzled_64(n, ch)) =
                        n < n_col ? *reinterpret_cast<const uint4*>(
                                        x_head + static_cast<std::int64_t>(c0 + n) * kK * heads +
                                        k_base + ch * 8)
                                  : make_uint4(0, 0, 0, 0);
                } else {
                    uint4 hi_v = make_uint4(0, 0, 0, 0);
                    uint4 lo_v = make_uint4(0, 0, 0, 0);
                    if (n < n_col) {
                        const float* xp = x_head + static_cast<std::int64_t>(c0 + n) * kK * heads +
                                          k_base + ch * 8;
                        const float4 f0 = *reinterpret_cast<const float4*>(xp);
                        const float4 f1 = *reinterpret_cast<const float4*>(xp + 4);
                        split_bf16(f0.x, f0.y, hi_v.x, lo_v.x);
                        split_bf16(f0.z, f0.w, hi_v.y, lo_v.y);
                        split_bf16(f1.x, f1.y, hi_v.z, lo_v.z);
                        split_bf16(f1.z, f1.w, hi_v.w, lo_v.w);
                    }
                    *reinterpret_cast<uint4*>(b_hi_s + swizzled_64(n, ch)) = hi_v;
                    *reinterpret_cast<uint4*>(b_lo_s + swizzled_64(n, ch)) = lo_v;
                }
            }
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldmatrix_x4(a0, a1, a2, a3,
                            smem_address(a_s + swizzled_64(16 * wm + a_r, 2 * ks + a_c)));
                ldmatrix_x2(b0, b1, smem_address(b_hi_s + swizzled_64(b_r, 2 * ks + b_c)));
                mma_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
                if constexpr (kSplitInput) {
                    unsigned bl0, bl1;
                    ldmatrix_x2(bl0, bl1, smem_address(b_lo_s + swizzled_64(b_r, 2 * ks + b_c)));
                    mma_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, bl0, bl1);
                }
            }
            __syncthreads();
        }

        // Stage [8][64] BF16 output in a_s for coalesced uint4 writes.
        auto* out_s = a_s;
        const int r0 = 16 * wm + grp;
        const int n0 = 2 * tid4;
        out_s[n0 * kOutStride + r0]           = __float2bfloat16_rn(acc[0]);
        out_s[(n0 + 1) * kOutStride + r0]     = __float2bfloat16_rn(acc[1]);
        out_s[n0 * kOutStride + r0 + 8]       = __float2bfloat16_rn(acc[2]);
        out_s[(n0 + 1) * kOutStride + r0 + 8] = __float2bfloat16_rn(acc[3]);
        __syncthreads();

        if (tid < n_col * kChunks64) {
            const int n = tid >> 3, ch = tid & 7;
            bf16* dst = out + m0 + ch * 8 + static_cast<std::int64_t>(h) * kM +
                        static_cast<std::int64_t>(column_begin + c0 + n) * kM * heads;
            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(out_s + n * kOutStride + ch * 8);
        }
    } else {
        // kNtile == 64: 2 x 2 warp grid, each warp computes 32 rows (2 mt) x 32 cols (4 nt).
        float acc[2][4][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[mt][nt][i] = 0.0F; }
            }
        }
        const int wm  = warp >> 1;
        const int wn  = warp & 1;
        const int b_r = 8 * (lane >> 4) + (lane & 7);
        const int b_c = (lane >> 3) & 1;

        for (int k_base = 0; k_base < kK; k_base += kKtile) {
#pragma unroll
            for (int i = tid; i < kMtile * kChunks64; i += 128) {
                const int r = i >> 3, ch = i & 7;
                *reinterpret_cast<uint4*>(a_s + swizzled_64(r, ch)) =
                    *reinterpret_cast<const uint4*>(w_head + static_cast<std::int64_t>(r) * kK +
                                                    k_base + ch * 8);
            }
#pragma unroll
            for (int i = tid; i < kNtile * kChunks64; i += 128) {
                const int n = i >> 3, ch = i & 7;
                if constexpr (!kSplitInput) {
                    *reinterpret_cast<uint4*>(b_hi_s + swizzled_64(n, ch)) =
                        n < n_col ? *reinterpret_cast<const uint4*>(
                                        x_head + static_cast<std::int64_t>(c0 + n) * kK * heads +
                                        k_base + ch * 8)
                                  : make_uint4(0, 0, 0, 0);
                } else {
                    uint4 hi_v = make_uint4(0, 0, 0, 0);
                    uint4 lo_v = make_uint4(0, 0, 0, 0);
                    if (n < n_col) {
                        const float* xp = x_head + static_cast<std::int64_t>(c0 + n) * kK * heads +
                                          k_base + ch * 8;
                        const float4 f0 = *reinterpret_cast<const float4*>(xp);
                        const float4 f1 = *reinterpret_cast<const float4*>(xp + 4);
                        split_bf16(f0.x, f0.y, hi_v.x, lo_v.x);
                        split_bf16(f0.z, f0.w, hi_v.y, lo_v.y);
                        split_bf16(f1.x, f1.y, hi_v.z, lo_v.z);
                        split_bf16(f1.z, f1.w, hi_v.w, lo_v.w);
                    }
                    *reinterpret_cast<uint4*>(b_hi_s + swizzled_64(n, ch)) = hi_v;
                    *reinterpret_cast<uint4*>(b_lo_s + swizzled_64(n, ch)) = lo_v;
                }
            }
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                unsigned a[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                                smem_address(a_s + swizzled_64(32 * wm + 16 * mt + a_r,
                                                               2 * ks + a_c)));
                }
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    const int nb = 32 * wn + 16 * np;
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4(b0, b1, b2, b3,
                                smem_address(b_hi_s + swizzled_64(nb + b_r, 2 * ks + b_c)));
                    const int nt0 = 2 * np, nt1 = 2 * np + 1;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        mma_bf16(acc[mt][nt0][0], acc[mt][nt0][1], acc[mt][nt0][2], acc[mt][nt0][3],
                                 a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1);
                        mma_bf16(acc[mt][nt1][0], acc[mt][nt1][1], acc[mt][nt1][2], acc[mt][nt1][3],
                                 a[mt][0], a[mt][1], a[mt][2], a[mt][3], b2, b3);
                    }
                    if constexpr (kSplitInput) {
                        unsigned bl0, bl1, bl2, bl3;
                        ldmatrix_x4(bl0, bl1, bl2, bl3,
                                    smem_address(b_lo_s + swizzled_64(nb + b_r, 2 * ks + b_c)));
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            mma_bf16(acc[mt][nt0][0], acc[mt][nt0][1], acc[mt][nt0][2],
                                     acc[mt][nt0][3], a[mt][0], a[mt][1], a[mt][2], a[mt][3], bl0,
                                     bl1);
                            mma_bf16(acc[mt][nt1][0], acc[mt][nt1][1], acc[mt][nt1][2],
                                     acc[mt][nt1][3], a[mt][0], a[mt][1], a[mt][2], a[mt][3], bl2,
                                     bl3);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Reuse a_s + b_hi_s (16 KiB total) to stage [64][64+8] BF16 (9,216 B) for coalesced stores.
        // Write 32 columns at a time using a_s (8 KiB >= 32 * 72 * 2 = 4,608 B) so the buffer is
        // guaranteed contiguous inside a_s without relying on adjacent shared array placement.
        auto* out_s = a_s;
        for (int half = 0; half < 2; ++half) {
            if (wn == half) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int r0 = 32 * wm + 16 * mt + grp;
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const int n0 = 8 * nt + 2 * tid4;
                        out_s[n0 * kOutStride + r0]           = __float2bfloat16_rn(acc[mt][nt][0]);
                        out_s[(n0 + 1) * kOutStride + r0]     = __float2bfloat16_rn(acc[mt][nt][1]);
                        out_s[n0 * kOutStride + r0 + 8]       = __float2bfloat16_rn(acc[mt][nt][2]);
                        out_s[(n0 + 1) * kOutStride + r0 + 8] = __float2bfloat16_rn(acc[mt][nt][3]);
                    }
                }
            }
            __syncthreads();
            const int n_base = 32 * half;
            const int n_take = min(32, max(0, n_col - n_base));
            for (int i = tid; i < n_take * kChunks64; i += 128) {
                const int n = i >> 3, ch = i & 7;
                bf16* dst = out + m0 + ch * 8 + static_cast<std::int64_t>(h) * kM +
                            static_cast<std::int64_t>(column_begin + c0 + n_base + n) * kM * heads;
                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(out_s + n * kOutStride + ch * 8);
            }
            __syncthreads();
        }
    }
}

template <typename In, int kM, int kK>
void launch_mla_head_gemm(const In* x, const bf16* weight, int heads, int columns, int column_begin,
                          bf16* out, cudaStream_t stream) {
    constexpr int kMblocks = kM / 64;
    if (columns <= 8) {
        const dim3 grid(1U, static_cast<unsigned>(heads), static_cast<unsigned>(kMblocks));
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(128), 0, stream},
                                        mla_head_gemm_kernel<In, kM, kK, 8>, x, weight, heads,
                                        columns, column_begin, out));
    } else {
        const dim3 grid(static_cast<unsigned>(ceil_div(columns, 64)),
                        static_cast<unsigned>(heads), static_cast<unsigned>(kMblocks));
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(128), 0, stream},
                                        mla_head_gemm_kernel<In, kM, kK, 64>, x, weight, heads,
                                        columns, column_begin, out));
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

// The tensor-core route keeps only the FP32 latent sums of a chunk of columns, rounded down to a
// whole wave of CTAs so full chunks avoid short tail waves while keeping workspace bytes monotonic.
Carve carve_mma(void* base, int columns) {
    Carve out;
    const std::size_t per_column = sizeof(float) * kMmaHeads * kLatent;
    std::size_t max_chunk        = std::max<std::size_t>(1, kMmaChunkBudget / per_column);
    static const int sms         = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            count > 0) {
            return count;
        }
        return 0;
    }();
    const std::size_t wave_cols =
        sms > 0 ? static_cast<std::size_t>((sms + kMmaHeadBlocks - 1) / kMmaHeadBlocks) : 0;
    if (wave_cols > 0 && max_chunk >= wave_cols) {
        max_chunk = (max_chunk / wave_cols) * wave_cols;
    }
    out.chunk = std::clamp<std::size_t>(max_chunk, 1, static_cast<std::size_t>(columns));
    out.bytes = align256(out.chunk * per_column);
    out.ol    = static_cast<float*>(base);
    return out;
}

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
    launch_mla_head_gemm<bf16, kLatent, kHeadDim>(
        static_cast<const bf16*>(q.data), static_cast<const bf16*>(w_uk.data), geometry.heads,
        columns, 0, static_cast<bf16*>(qa.data), stream);
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
    launch_mla_head_gemm<bf16, kHeadDim, kLatent>(
        static_cast<const bf16*>(ol.data), static_cast<const bf16*>(w_uv.data), geometry.heads,
        columns, 0, static_cast<bf16*>(o.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t mla_sparse_attention_workspace_bytes(const MlaAttentionGeometry& geometry,
                                                 std::int32_t columns, std::int32_t max_context,
                                                 MlaAttentionExecutionEnvelope envelope) {
    require_geometry(geometry);
    require(columns > 0 && max_context > 0 && envelope.splits >= 0, "invalid workspace profile");
    if (envelope.tensor_cores) {
        require(geometry.heads == kMmaHeads, "the tensor-core route needs 64 heads");
        return carve_mma(nullptr, columns).bytes;
    }
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
    if (envelope.tensor_cores) { require(geometry.heads == kMmaHeads, "the tensor-core route needs 64 heads"); }
    Carve ws = envelope.tensor_cores ? carve_mma(workspace, columns)
                                     : carve(workspace, geometry, columns, splits);
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
    if (envelope.tensor_cores) {
        static const bool attributes = [] {
            CUDA_CHECK(cudaFuncSetAttribute(attention_mma_kernel<false>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(MmaShared<false>::kBytes)));
            CUDA_CHECK(cudaFuncSetAttribute(attention_mma_kernel<true>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(MmaShared<true>::kBytes)));
            return true;
        }();
        (void)attributes;
        for (int begin = 0; begin < columns; begin += static_cast<int>(ws.chunk)) {
            const int n = std::min<int>(static_cast<int>(ws.chunk), columns - begin);
            const dim3 grid(static_cast<unsigned>(n));
            if (int8) {
                const pdl::LaunchConfig launch{grid, dim3(kMmaThreads), MmaShared<true>::kBytes, stream};
                CUDA_CHECK(pdl::launch_consumer(launch, attention_mma_kernel<true>, static_cast<const bf16*>(qa.data),
                                                pages, scales, tables, stride, rows, pos, batch.width, pool_p, cnt_p,
                                                top, pool, scale, begin, ws.ol));
            } else {
                const pdl::LaunchConfig launch{grid, dim3(kMmaThreads), MmaShared<false>::kBytes, stream};
                CUDA_CHECK(pdl::launch_consumer(launch, attention_mma_kernel<false>, static_cast<const bf16*>(qa.data),
                                                pages, scales, tables, stride, rows, pos, batch.width, pool_p, cnt_p,
                                                top, pool, scale, begin, ws.ol));
            }
            launch_mla_head_gemm<float, kHeadDim, kLatent>(
                static_cast<const float*>(ws.ol), static_cast<const bf16*>(w_uv.data),
                geometry.heads, n, begin, static_cast<bf16*>(out.data), stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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
        launch_mla_head_gemm<float, kHeadDim, kLatent>(
            static_cast<const float*>(ws.ol), static_cast<const bf16*>(w_uv.data),
            geometry.heads, n, begin, static_cast<bf16*>(out.data), stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
