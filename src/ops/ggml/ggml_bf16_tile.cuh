#pragma once

// The ggml-bf16-prefill tiles (docs/maintainer/qwen3_8-flash-next-design.md §13.5): BF16 tensor-core
// tiles over decoded GGUF planes (ggml_tile.cuh's CodeStage images, the ones the exact route's tiles
// read). ops::ggml_linear runs them for wide dense projections (LinearRoutes::GgmlBf16Wide) and the
// GGUF expert banks for wide jobs (offloaded_moe::WideA16::kGgmlBf16, ggml_expert.cuh).
//
// Each code is converted exactly to BF16 (|code| <= 127) and multiplied by the BF16 operand as
// stored on m16n8k16 MMAs with FP32 accumulation, so every product is exact and only sums round. Per
// 32-element slice two MMAs give the sums over elements 0-15 and 16-31 (one chained sum where one
// sub-scale covers the slice); the slice adds d * sub * sum (d * sub is exact in binary32) and, for
// the K-quants, - dmin * min * sum(x) (a per-slice column sum the Columns policy supplies) to each output's FP32
// accumulator, slice by slice in K order. An output's bits depend only on its row, its column and K,
// not on how many columns share the call; they are not the exact route's (no A16 encoding, FP32 sums).
//
// The plane stores a thread's codes in m16n8k32 A-fragment order (row g's elements 4t .. 4t + 3 and
// 16 + 4t .. 16 + 4t + 3). An MMA's sum over K does not depend on how K is labelled as long as A and
// B agree, so the m16n8k16 fragments take those elements as their K slots: A's registers 0/2 (K 2t,
// 2t + 1 and 2t + 8, 2t + 9 of row g) carry elements 4t, 4t + 1 and 4t + 2, 4t + 3, and B's two
// registers x[4t .. 4t + 3] of column g. The B operand is staged in shared memory by 8-byte units
// (four elements), unit u of a column's slice at position 2 (u % 4) + u / 4, so one 16-byte load gives
// a thread both halves' B registers.
//
// A CTA computes kRows = 128 rows (two plane tiles, `images`) x kColumns = 128 columns with 8 warps of
// 64 x 32 (four m16 by four n8 tiles; warp row wr reads image wr), stages of 64 elements of K in a
// ring of kStages by cp.async. Column c's K elements come from a Columns policy (contiguous columns
// of a dense call, or an expert's gathered token rows).

#include "core/ggml_format.h"
#include "ops/common/memory.cuh"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace infernix::ggml::bf16 {

constexpr int kRows        = 2 * tile::kTileRows; // two plane tiles
constexpr int kColumns     = 128;
constexpr int kThreads     = 256;                 // 8 warps: 2 (rows) x 4 (columns)
constexpr int kStages      = 3;
constexpr int kSlices      = tile::kStageSlices;  // 32-element slices per stage
constexpr int kWarpColumns = 32;
constexpr int kMTiles      = tile::kTileRows / 16; // a warp's m16 tiles: its plane tile's rows
constexpr int kNTiles      = kWarpColumns / 8;
static_assert(kThreads / 32 == (kRows / tile::kTileRows) * (kColumns / kWarpColumns));

// How a format's slice is scaled: one sub-scale per 32 elements (kPlain), one per 16 (kHalf: Q6_K,
// IQ2_S, Q3_K), none (kUnit: IQ4_NL, Q2_0, Q8_0), one per 32 with the K-quants' min term (kMin:
// Q4_K, Q5_K), or one sub-scale and one min per 16 (kHalfMin: Q2_K).
enum class Scaling : int { kPlain, kHalf, kUnit, kMin, kHalfMin };

template <QType F>
inline constexpr Scaling kScaling = tile::kHalfSub<F> && fast::kMinHalf<F> ? Scaling::kHalfMin
                                    : tile::kHalfSub<F>                    ? Scaling::kHalf
                                    : fast::kMin<F>                        ? Scaling::kMin
                                    : tile::kUnitSub<F>                    ? Scaling::kUnit
                                                                           : Scaling::kPlain;

// Bytes of a stage image the tiles copy (the min terms only where the scaling reads them).
template <Scaling S>
inline constexpr int kImageBytes = S == Scaling::kMin ? static_cast<int>(sizeof(tile::CodeStage))
                                                      : static_cast<int>(offsetof(tile::CodeStage, dmin));

struct Shared {
    alignas(16) std::uint8_t image[kStages][2][sizeof(tile::CodeStage)];
    alignas(16) std::uint16_t x[kStages][kSlices][kColumns][32]; // BF16 bits, 8-byte units permuted
    float xsum[kStages][kSlices][kColumns];                      // kMin only
};

// The accumulators of a CTA's thread: its warp's four m16 by four n8 tiles.
struct Accumulators {
    float r[kMTiles][kNTiles][4];
};

// Output row (within the CTA's 128) and column (within its 128) of accumulator (mi, nt, q).
__device__ __forceinline__ int out_row(int warp, int lane, int mi, int q) {
    return (warp / (kColumns / kWarpColumns)) * tile::kTileRows + 16 * mi + lane / 4 + 8 * (q >> 1);
}
__device__ __forceinline__ int out_column(int warp, int lane, int nt, int q) {
    return (warp % (kColumns / kWarpColumns)) * kWarpColumns + 8 * nt + 2 * (lane % 4) + (q & 1);
}

// Two signed bytes (the low 16 bits of `bytes`) as an exact BF16 pair (low byte in the low half):
// 128 + (b & 127) less 128 or 256 by the byte's sign.
__device__ __forceinline__ unsigned bf16_pair(unsigned bytes) {
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

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float f16_value(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

// The B operand of a dense call: columns col0 .. of x (BF16 [k, columns]), input element e at
// x[perm(e)], and their slice sums (xsum [columns, k / 32]; kMin only).
struct DenseColumns {
    const std::uint16_t* x;
    int k;
    const std::int32_t* perm;
    int perm_block;
    const float* xsum;
    int col0, columns;

    // Column c's elements (null past the call's columns: zero-filled).
    __device__ __forceinline__ const std::uint16_t* column(int c) const {
        return col0 + c < columns ? x + static_cast<std::size_t>(col0 + c) * k : nullptr;
    }
    __device__ __forceinline__ int element(int e) const {
        return perm == nullptr ? e : perm[e / perm_block] * perm_block + e % perm_block;
    }
    __device__ __forceinline__ const float* slice_sum(int c, int s) const {
        return col0 + c < columns ? xsum + static_cast<std::size_t>(col0 + c) * (k / 32) + s : nullptr;
    }
};

// The CTA's K loop: `images` the two plane tiles' stage images (stage st at images[h] + st *
// sizeof(CodeStage)), `stages` stages of kSlices slices, B columns from `cols`. Every thread calls it;
// it leaves the thread's accumulators and returns with every copy landed and every thread past its
// last read of shared memory.
template <Scaling S, class Columns>
__device__ __forceinline__ void compute(Shared& sm, const std::uint8_t* const (&images)[2], int stages, const Columns& cols,
                                        Accumulators& acc) {
    constexpr int kImage          = kImageBytes<S>;
    constexpr std::size_t kStride = sizeof(tile::CodeStage);
    // Stage st's copies into ring slot `buf`: both row tiles' images, the columns' units (zero past the
    // columns) and, for kMin, their slice sums.
    const auto issue = [&](int st, int buf) {
        for (int i = static_cast<int>(threadIdx.x); i < 2 * (kImage / 16); i += kThreads) {
            const int h = i / (kImage / 16), c = i % (kImage / 16);
            tile::cp_async16(&sm.image[buf][h][16 * c], images[h] + st * kStride + 16 * c, 16);
        }
        for (int i = static_cast<int>(threadIdx.x); i < kSlices * kColumns * 8; i += kThreads) {
            const int j = i / (kColumns * 8), c = (i / 8) % kColumns, u = i % 8;
            const std::uint16_t* column = cols.column(c);
            const std::uint16_t* from   = column != nullptr ? column + cols.element((st * kSlices + j) * 32 + 4 * u)
                                                            : cols.column(0);
            ops::cp_async_zfill<8>(&sm.x[buf][j][c][4 * (2 * (u % 4) + u / 4)], from, column != nullptr ? 8 : 0);
        }
        if constexpr (S == Scaling::kMin) {
            for (int i = static_cast<int>(threadIdx.x); i < kSlices * kColumns; i += kThreads) {
                const int j = i / kColumns, c = i % kColumns;
                const float* from = cols.slice_sum(c, st * kSlices + j);
                ops::cp_async_zfill<4>(&sm.xsum[buf][j][c], from != nullptr ? from : cols.slice_sum(0, 0), from != nullptr ? 4 : 0);
            }
        }
    };

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4;
    const int wr = warp / (kColumns / kWarpColumns), wc = warp % (kColumns / kWarpColumns);
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc.r[mi][nt][q] = 0.0F; }
        }
    }
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < stages) { issue(s, s); }
        ops::cp_commit();
    }
    for (int st = 0; st < stages; ++st) {
        ops::cp_wait<kStages - 2>(); // stage st has landed
        __syncthreads();             // and every thread finished stage st - 1, whose slot the next issue fills
        if (st + kStages - 1 < stages) { issue(st + kStages - 1, (st + kStages - 1) % kStages); }
        ops::cp_commit();
        const int buf = st % kStages;
        const auto& cs = *reinterpret_cast<const tile::CodeStage*>(sm.image[buf][wr]);
#pragma unroll
        for (int j = 0; j < kSlices; ++j) {
            uint4 b[kNTiles];
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                b[nt] = *reinterpret_cast<const uint4*>(&sm.x[buf][j][wc * kWarpColumns + 8 * nt + g][8 * t]);
            }
            float xs[kNTiles][2] = {};
            if constexpr (S == Scaling::kMin) {
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    xs[nt][0] = sm.xsum[buf][j][wc * kWarpColumns + 8 * nt + 2 * t];
                    xs[nt][1] = sm.xsum[buf][j][wc * kWarpColumns + 8 * nt + 2 * t + 1];
                }
            }
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                const int4 a = *reinterpret_cast<const int4*>(cs.a[j][mi][g][t]);
                // Elements 0-15 (lo) and 16-31 (hi): rows g, g + 8, then their elements 4t + 2, + 3.
                const unsigned lo[4] = {bf16_pair(static_cast<unsigned>(a.x)), bf16_pair(static_cast<unsigned>(a.y)),
                                        bf16_pair(static_cast<unsigned>(a.x) >> 16), bf16_pair(static_cast<unsigned>(a.y) >> 16)};
                const unsigned hi[4] = {bf16_pair(static_cast<unsigned>(a.z)), bf16_pair(static_cast<unsigned>(a.w)),
                                        bf16_pair(static_cast<unsigned>(a.z) >> 16), bf16_pair(static_cast<unsigned>(a.w) >> 16)};
                const int r = 16 * mi + g; // rows r and r + 8 of the warp's plane tile
                const float d0 = f16_value(cs.d[j][r]), d1 = f16_value(cs.d[j][r + 8]);
                // d * sub per row (0: r, 1: r + 8) and half: exact (an 11-bit mantissa by |sub| <= 127).
                float s00 = d0, s01 = d0, s10 = d1, s11 = d1;
                if constexpr (S != Scaling::kUnit) {
                    const int4 sc = *reinterpret_cast<const int4*>(cs.sub[j][mi][g]);
                    s00 = d0 * static_cast<float>(sc.x);
                    s01 = d0 * static_cast<float>(sc.y);
                    s10 = d1 * static_cast<float>(sc.z);
                    s11 = d1 * static_cast<float>(sc.w);
                }
                float m0 = 0.0F, m1 = 0.0F;
                if constexpr (S == Scaling::kMin) {
                    m0 = f16_value(cs.dmin[j][r]) * static_cast<float>(cs.min[j][r]);
                    m1 = f16_value(cs.dmin[j][r + 8]) * static_cast<float>(cs.min[j][r + 8]);
                }
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    float(&o)[4] = acc.r[mi][nt];
                    if constexpr (S == Scaling::kHalf) {
                        float p[4] = {0.0F, 0.0F, 0.0F, 0.0F}, q[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                        mma(p, lo, b[nt].x, b[nt].y);
                        mma(q, hi, b[nt].z, b[nt].w);
                        o[0] = fmaf(s01, q[0], fmaf(s00, p[0], o[0]));
                        o[1] = fmaf(s01, q[1], fmaf(s00, p[1], o[1]));
                        o[2] = fmaf(s11, q[2], fmaf(s10, p[2], o[2]));
                        o[3] = fmaf(s11, q[3], fmaf(s10, p[3], o[3]));
                    } else {
                        float p[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                        mma(p, lo, b[nt].x, b[nt].y);
                        mma(p, hi, b[nt].z, b[nt].w);
                        o[0] = fmaf(s00, p[0], o[0]);
                        o[1] = fmaf(s00, p[1], o[1]);
                        o[2] = fmaf(s10, p[2], o[2]);
                        o[3] = fmaf(s10, p[3], o[3]);
                    }
                    if constexpr (S == Scaling::kMin) {
                        o[0] = fmaf(-m0, xs[nt][0], o[0]);
                        o[1] = fmaf(-m0, xs[nt][1], o[1]);
                        o[2] = fmaf(-m1, xs[nt][0], o[2]);
                        o[3] = fmaf(-m1, xs[nt][1], o[3]);
                    }
                }
            }
        }
    }
    ops::cp_wait<0>();
    __syncthreads(); // the caller may reuse shared memory
}

// The decode-in-loop tiles' raw units: 256 elements of K (128 for formats of blocks under 256
// elements, so a Q8_0 row's unit stays within 160 bytes).
template <QType F>
inline constexpr int kUnitElems = ggml_block_elems(F) >= 256 ? 256 : 128;
template <QType F>
inline constexpr int kUnitBytes = kUnitElems<F> / static_cast<int>(ggml_block_elems(F)) * static_cast<int>(ggml_block_bytes(F));
template <QType F>
inline constexpr int kUnitChunks = (kUnitBytes<F> + 15 + 15) / 16;
// compute_decode places a kFold row's unit up to 15 bytes into its raw row (a 12-byte bank shift and
// the unit's offset in its first word) and copies whole words.
template <QType F>
inline constexpr bool kUnitFits = 12 + 3 + kUnitBytes<F> + 3 <= kUnitChunks<F> * 16;

// Formats whose code times sub-scale is an integer of at most 8 significant bits (Q3_K: -4..3 by
// -32..31; Q2_K: 0..3 by 0..15), so compute_decode stores the product exact in BF16 and its MMAs sum a
// whole block (one d) before scaling it.
template <QType F>
inline constexpr bool kFold = F == QType::GGML_Q3_K || F == QType::GGML_Q2_K;

// The four little-endian words at p..p + 15 (any alignment), as tile::words2_at.
__device__ __forceinline__ uint4 words4_at(const std::uint8_t* p) {
    const auto a     = reinterpret_cast<std::uintptr_t>(p);
    const auto* w    = reinterpret_cast<const std::uint32_t*>(a & ~std::uintptr_t{3});
    const unsigned s = static_cast<unsigned>(a & 3) * 8;
    const std::uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = s != 0 ? w[4] : 0;
    return make_uint4(__funnelshift_r(w0, w1, s), __funnelshift_r(w1, w2, s), __funnelshift_r(w2, w3, s),
                      __funnelshift_r(w3, w4, s));
}

// Half h16 of slice sb of a kFold block b: its 16 codes (byte j of word w is element 4w + j), its
// sub-scale and (Q2_K) its min sub-scale; the integers of tile::tile_chunk.
template <QType F>
__device__ __forceinline__ void fold_half(const std::uint8_t* b, int sb, int h16, int (&code)[4], int& sub, int& mn) {
    const int half = sb / 4, plane = sb % 4, l = 16 * h16;
    if constexpr (F == QType::GGML_Q3_K) {
        const uint4 q = words4_at(b + 32 + 32 * half + l), m = words4_at(b + l);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, mw[4] = {m.x, m.y, m.z, m.w};
        const int bit = 4 * half + plane;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            code[w] = fast::sub_bytes4(((qw[w] >> (2 * plane)) & 0x03030303U) | (((mw[w] >> bit) & 0x01010101U) << 2),
                                       0x04040404U);
        }
        const int i = 8 * half + 2 * plane + h16;
        sub = (((b[96 + i % 8] >> (4 * (i / 8))) & 0xF) | (((b[104 + i % 4] >> (2 * (i / 4))) & 3) << 4)) - 32;
    } else {
        static_assert(F == QType::GGML_Q2_K);
        const uint4 q = words4_at(b + 16 + 32 * half + l);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) { code[w] = static_cast<int>((qw[w] >> (2 * plane)) & 0x03030303U); }
        const int s = b[8 * half + 2 * plane + h16];
        sub = s & 0xF;
        mn  = s >> 4;
    }
}

// A kFold block's d word.
template <QType F>
__device__ __forceinline__ std::uint16_t fold_d(const std::uint8_t* b) {
    return u16_at(b + (F == QType::GGML_Q3_K ? 108 : 80));
}

// One stage of a 64-row tile of a kFold format: per slice and m16 tile, lane (g, t)'s BF16 A registers
// in 16-byte chunks, chunk (2 t + half) ^ g of row group g (so neither the decode's stores nor the
// MMAs' loads conflict much) holding the k16 half's four words (row g's elements 4t, 4t + 1 and
// 4t + 2, 4t + 3 of the half, then row g + 8's: the K order of the permuted x units, so one thread
// stores a row's pair of words at once); each row's d (FP32, the block's, which spans the stage).
struct FoldStage {
    alignas(16) unsigned a[kSlices][kMTiles][8][8][4];
    float d[tile::kTileRows];
};
// ... and Q2_K's min terms: dmin * min of each row's two halves per slice (FP32, exact).
struct FoldMinStage : FoldStage {
    float2 m[kSlices][tile::kTileRows];
};

// Dynamic shared memory of compute_decode: the stage images of both 64-row tiles for two stages
// (the one the MMAs read, the next being decoded), two stages of x units, two raw units.
template <class Stage, QType F>
struct DecodeSharedOf {
    alignas(16) Stage code[2][2];
    alignas(16) std::uint16_t x[2][kSlices][kColumns][32]; // BF16 bits, 8-byte units permuted
    alignas(16) std::uint8_t raw[2][kRows][kUnitChunks<F> * 16];
    std::uint8_t raw_offset[2][kRows]; // where row r's unit bytes start in raw[.][r]
    const std::uint16_t* column[kColumns]; // the tile's columns (Columns::column, null past them)
};
template <QType F>
using DecodeShared = DecodeSharedOf<std::conditional_t<kFold<F>, std::conditional_t<fast::kMinHalf<F>, FoldMinStage, FoldStage>,
                                                       tile::CodeStage>,
                                    F>;

// compute_decode's CTA: 16 warps of 32 rows x 32 columns (twice compute's warps, for its decode's
// latency), each with two m16 by four n8 accumulator tiles.
constexpr int kDecodeThreads = 512;
constexpr int kDecodeMTiles  = 2;
static_assert(kDecodeThreads / 32 == (kRows / 32) * (kColumns / kWarpColumns));

struct DecodeAccumulators {
    float r[kDecodeMTiles][kNTiles][4];
};

// Output row (within the CTA's 128) of compute_decode's accumulator (mi, nt, q); its column is
// out_column's.
__device__ __forceinline__ int decode_out_row(int warp, int lane, int mi, int q) {
    return (warp / (kColumns / kWarpColumns)) * 32 + 16 * mi + lane / 4 + 8 * (q >> 1);
}

// compute's K loop over raw GGUF rows instead of decoded planes (the expert banks whose jobs are too
// narrow to amortize a plane; tile h's row r at row_of(h, r)). Each iteration, after one barrier,
// issues the next stage's x units (and, at a unit's first stage, the unit after next's raw bytes by
// cp.async), decodes the next stage from the raw copy into its image, then runs this stage's MMAs, so
// the copies overlap a stage of decode and MMAs. A kFold format's thread
// decodes one 16-element half-slice (fold_half), the others' threads two 8-element chunks
// (tile_chunk). The min formats' X sums (each
// 16-element half of a column's slice, FP32 in a fixed order) are formed from the B registers the
// MMAs read, so Columns supplies no slice sums.
template <QType F, class RowOf, class Columns>
__device__ __forceinline__ void compute_decode(DecodeShared<F>& sm, RowOf row_of, int stages, const Columns& cols,
                                               DecodeAccumulators& acc) {
    constexpr Scaling S = kScaling<F>;
    constexpr bool kSums = S == Scaling::kMin || S == Scaling::kHalfMin;
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kUnitStages = kUnitElems<F> / (kSlices * 32);
    constexpr int kDecodeChunks = kRows * kSlices * 4 / kDecodeThreads; // 8-element chunks per thread and stage
    static_assert(kUnitStages >= 2 && kDecodeChunks * kDecodeThreads == kRows * kSlices * 4);
    static_assert(!kFold<F> || (kUnitFits<F> && kUnitElems<F> == static_cast<int>(ggml_block_elems(F)) && kRows * kSlices * 2 == kDecodeThreads));
    const int units = stages / kUnitStages;
    // The columns once per tile (a gathered column's token is a division by top_k).
    if (threadIdx.x < kColumns) { sm.column[threadIdx.x] = cols.column(static_cast<int>(threadIdx.x)); }
    const auto issue_x = [&](int st, int buf) {
        for (int i = static_cast<int>(threadIdx.x); i < kSlices * kColumns * 8; i += kDecodeThreads) {
            const int j = i / (kColumns * 8), c = (i / 8) % kColumns, u = i % 8;
            const std::uint16_t* column = sm.column[c];
            const std::uint16_t* from   = column != nullptr ? column + cols.element((st * kSlices + j) * 32 + 4 * u)
                                                            : sm.column[0];
            ops::cp_async_zfill<8>(&sm.x[buf][j][c][4 * (2 * (u % 4) + u / 4)], from, column != nullptr ? 8 : 0);
        }
    };
    // A row's unit: a kFold format's by 4-byte copies from its aligned-down start, placed 4 (r % 4)
    // bytes into its raw row (every row's unit starts at the same alignment, so this spreads the rows
    // a warp decodes, eight at a time, over four banks); the others' by 16-byte copies.
    const auto issue_raw = [&](int buf, int unit) {
        const std::size_t begin = static_cast<std::size_t>(unit) * kUnitBytes<F>;
        if constexpr (!kFold<F>) {
            for (int i = static_cast<int>(threadIdx.x); i < kRows * kUnitChunks<F>; i += kDecodeThreads) {
                const int r = i / kUnitChunks<F>, chunk = i % kUnitChunks<F>;
                const std::uint8_t* src = row_of(r / tile::kTileRows, r % tile::kTileRows) + begin;
                const auto* aligned =
                    reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{15});
                const int offset = static_cast<int>(src - aligned);
                const int need   = offset + kUnitBytes<F> - 16 * chunk;
                if (chunk == 0) { sm.raw_offset[buf][r] = static_cast<std::uint8_t>(offset); }
                if (need > 0) { tile::cp_async16(&sm.raw[buf][r][16 * chunk], aligned + 16 * chunk, need < 16 ? need : 16); }
            }
            return;
        }
        constexpr int kWords = (3 + kUnitBytes<F> + 3) / 4;
        for (int i = static_cast<int>(threadIdx.x); i < kRows * kWords; i += kDecodeThreads) {
            const int r = i / kWords, w = i % kWords;
            const std::uint8_t* src = row_of(r / tile::kTileRows, r % tile::kTileRows) + begin;
            const auto* aligned = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{3});
            const int offset    = static_cast<int>(src - aligned), shift = 4 * (r % 4);
            if (w == 0) { sm.raw_offset[buf][r] = static_cast<std::uint8_t>(shift + offset); }
            if (4 * w < offset + kUnitBytes<F>) { tile::cp_async4(&sm.raw[buf][r][shift + 4 * w], aligned + 4 * w); }
        }
    };
    const auto decode = [&](int st) {
        const int unit = st / kUnitStages, ubuf = unit & 1;
        if constexpr (kFold<F>) {
            // Thread (row, h16, dj): the 16 elements of half h16 of slice dj (one sub-scale); the lanes
            // of dj 0 and 1 read the same code bytes (other bit planes).
            const int i = static_cast<int>(threadIdx.x);
            const int row = i / 4, h16 = (i / 2) % 2, dj = i % 2;
            const int sb = (st - unit * kUnitStages) * kSlices + dj; // slice of the block
            const std::uint8_t* b = &sm.raw[ubuf][row][sm.raw_offset[ubuf][row]];
            int code[4], sc = 0, mn = 0;
            fold_half<F>(b, sb, h16, code, sc, mn);
            auto& image = sm.code[st & 1][row / tile::kTileRows];
            const int r = row % tile::kTileRows, grp = r / 16, g = r % 8, hr = (r % 16) / 8;
            const __nv_bfloat162 scale = __bfloat162bfloat162(__int2bfloat16_rn(sc));
            const auto scaled = [&](unsigned bytes) {
                union Bits {
                    __nv_bfloat162 pair;
                    unsigned bits;
                } v;
                v.bits = bf16_pair(bytes);
                v.pair = __hmul2(v.pair, scale);
                return v.bits;
            };
#pragma unroll
            for (int w = 0; w < 4; ++w) { // elements 4w .. 4w + 3 of the half: lane t = w
                *reinterpret_cast<uint2*>(&image.a[dj][grp][g][(2 * w + h16) ^ g][2 * hr]) =
                    make_uint2(scaled(static_cast<unsigned>(code[w])), scaled(static_cast<unsigned>(code[w]) >> 16));
            }
            if (dj == 0 && h16 == 0) { image.d[r] = f16_value(fold_d<F>(b)); }
            if constexpr (fast::kMinHalf<F>) {
                (&image.m[dj][r].x)[h16] = f16_value(u16_at(b + 82)) * static_cast<float>(mn);
            }
        } else {
#pragma unroll
        for (int i = 0; i < kDecodeChunks; ++i) {
            const int c = static_cast<int>(threadIdx.x) + kDecodeThreads * i;
            const int row = c / (4 * kSlices), dj = (c / 4) % kSlices, dp = c % 4;
            const int e = ((st - unit * kUnitStages) * kSlices + dj) * 32 + 8 * dp;
            const std::uint8_t* block = &sm.raw[ubuf][row][sm.raw_offset[ubuf][row]] + e / kBlockElems * kBlockBytes;
            const tile::TileChunk ch = tile::tile_chunk<F>(block, e % kBlockElems / 8);
            auto& image = sm.code[st & 1][row / tile::kTileRows];
            const int r = row % tile::kTileRows;
            {
                tile::store_chunk<F>(image, r, dj, dp, ch.w[0], ch.w[1], ch.sub, ch.d, ch.min_sub, ch.dmin);
            }
        }
        }
    };
    // The sum of a B register's two BF16 elements.
    const auto pair_sum = [](unsigned v) {
        return __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(v & 0xFFFFU))) +
               __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(v >> 16)));
    };

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4;
    // Warp (wr, wc): rows 32 wr .. 32 wr + 31 (m16 tiles m0, m0 + 1 of image wr / 2), columns 32 wc ..
    const int wr = warp / (kColumns / kWarpColumns), wc = warp % (kColumns / kWarpColumns);
    const int tile_h = wr / 2, m0 = 2 * (wr % 2);
    float block[kDecodeMTiles][kNTiles][4]; // kFold: the current block's sums of (code * sub) * x
#pragma unroll
    for (int mi = 0; mi < kDecodeMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                acc.r[mi][nt][q] = 0.0F;
                block[mi][nt][q] = 0.0F;
            }
        }
    }
    const auto mma_stage = [&](int st) {
        const int buf = st & 1;
        if constexpr (kFold<F>) {
            const auto& fs = sm.code[buf][tile_h];
#pragma unroll
            for (int j = 0; j < kSlices; ++j) {
                uint4 b[kNTiles];
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    b[nt] = *reinterpret_cast<const uint4*>(&sm.x[buf][j][wc * kWarpColumns + 8 * nt + g][8 * t]);
                }
                float xs[kNTiles][2][2] = {};
                if constexpr (kSums) {
#pragma unroll
                    for (int nt = 0; nt < kNTiles; ++nt) {
                        float h0 = pair_sum(b[nt].x) + pair_sum(b[nt].y), h1 = pair_sum(b[nt].z) + pair_sum(b[nt].w);
                        h0 += __shfl_xor_sync(0xFFFFFFFFU, h0, 1);
                        h1 += __shfl_xor_sync(0xFFFFFFFFU, h1, 1);
                        h0 += __shfl_xor_sync(0xFFFFFFFFU, h0, 2);
                        h1 += __shfl_xor_sync(0xFFFFFFFFU, h1, 2);
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            xs[nt][c][0] = __shfl_sync(0xFFFFFFFFU, h0, 4 * (2 * t + c));
                            xs[nt][c][1] = __shfl_sync(0xFFFFFFFFU, h1, 4 * (2 * t + c));
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < kDecodeMTiles; ++mi) {
                    const uint4 l4 = *reinterpret_cast<const uint4*>(fs.a[j][m0 + mi][g][(2 * t) ^ g]);
                    const uint4 h4 = *reinterpret_cast<const uint4*>(fs.a[j][m0 + mi][g][(2 * t + 1) ^ g]);
                    const unsigned lo[4] = {l4.x, l4.z, l4.y, l4.w}, hi[4] = {h4.x, h4.z, h4.y, h4.w};
#pragma unroll
                    for (int nt = 0; nt < kNTiles; ++nt) {
                        mma(block[mi][nt], lo, b[nt].x, b[nt].y);
                        mma(block[mi][nt], hi, b[nt].z, b[nt].w);
                    }
                    if constexpr (kSums) {
                        const int r = 16 * (m0 + mi) + g;
                        const float2 m0 = fs.m[j][r], m1 = fs.m[j][r + 8];
#pragma unroll
                        for (int nt = 0; nt < kNTiles; ++nt) {
                            float(&o)[4] = acc.r[mi][nt];
                            o[0] = fmaf(-m0.y, xs[nt][0][1], fmaf(-m0.x, xs[nt][0][0], o[0]));
                            o[1] = fmaf(-m0.y, xs[nt][1][1], fmaf(-m0.x, xs[nt][1][0], o[1]));
                            o[2] = fmaf(-m1.y, xs[nt][0][1], fmaf(-m1.x, xs[nt][0][0], o[2]));
                            o[3] = fmaf(-m1.y, xs[nt][1][1], fmaf(-m1.x, xs[nt][1][0], o[3]));
                        }
                    }
                }
            }
            if ((st + 1) % kUnitStages == 0) { // the block's last stage: its sums times d
#pragma unroll
                for (int mi = 0; mi < kDecodeMTiles; ++mi) {
                    const float d0 = fs.d[16 * (m0 + mi) + g], d1 = fs.d[16 * (m0 + mi) + g + 8];
#pragma unroll
                    for (int nt = 0; nt < kNTiles; ++nt) {
                        float(&o)[4] = acc.r[mi][nt];
                        float(&p)[4] = block[mi][nt];
                        o[0] = fmaf(d0, p[0], o[0]);
                        o[1] = fmaf(d0, p[1], o[1]);
                        o[2] = fmaf(d1, p[2], o[2]);
                        o[3] = fmaf(d1, p[3], o[3]);
                        p[0] = p[1] = p[2] = p[3] = 0.0F;
                    }
                }
            }
        } else {
            const tile::CodeStage& cs = sm.code[buf][tile_h];
    #pragma unroll
            for (int j = 0; j < kSlices; ++j) {
                uint4 b[kNTiles];
    #pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    b[nt] = *reinterpret_cast<const uint4*>(&sm.x[buf][j][wc * kWarpColumns + 8 * nt + g][8 * t]);
                }
                // xs[nt][c][h]: half h's X sum of output columns 2t + c of n-tile nt. Lane (g, t) holds column
                // g's unit t (half 0: b.x, b.y) and unit 4 + t (half 1: b.z, b.w); the 4 lanes of g add
                // them, then lane (g', t) reads columns 2t and 2t + 1 from lanes 4 (2t) and 4 (2t + 1).
                float xs[kNTiles][2][2] = {};
                if constexpr (kSums) {
    #pragma unroll
                    for (int nt = 0; nt < kNTiles; ++nt) {
                        float h0 = pair_sum(b[nt].x) + pair_sum(b[nt].y), h1 = pair_sum(b[nt].z) + pair_sum(b[nt].w);
                        h0 += __shfl_xor_sync(0xFFFFFFFFU, h0, 1);
                        h1 += __shfl_xor_sync(0xFFFFFFFFU, h1, 1);
                        h0 += __shfl_xor_sync(0xFFFFFFFFU, h0, 2);
                        h1 += __shfl_xor_sync(0xFFFFFFFFU, h1, 2);
    #pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            xs[nt][c][0] = __shfl_sync(0xFFFFFFFFU, h0, 4 * (2 * t + c));
                            xs[nt][c][1] = __shfl_sync(0xFFFFFFFFU, h1, 4 * (2 * t + c));
                        }
                    }
                }
    #pragma unroll
                for (int mi = 0; mi < kDecodeMTiles; ++mi) {
                    const int4 a = *reinterpret_cast<const int4*>(cs.a[j][m0 + mi][g][t]);
                    const unsigned lo[4] = {bf16_pair(static_cast<unsigned>(a.x)), bf16_pair(static_cast<unsigned>(a.y)),
                                            bf16_pair(static_cast<unsigned>(a.x) >> 16), bf16_pair(static_cast<unsigned>(a.y) >> 16)};
                    const unsigned hi[4] = {bf16_pair(static_cast<unsigned>(a.z)), bf16_pair(static_cast<unsigned>(a.w)),
                                            bf16_pair(static_cast<unsigned>(a.z) >> 16), bf16_pair(static_cast<unsigned>(a.w) >> 16)};
                    const int r = 16 * (m0 + mi) + g;
                    const float d0 = f16_value(cs.d[j][r]), d1 = f16_value(cs.d[j][r + 8]);
                    float s00 = d0, s01 = d0, s10 = d1, s11 = d1;
                    if constexpr (S != Scaling::kUnit) {
                        const int4 sc = *reinterpret_cast<const int4*>(cs.sub[j][m0 + mi][g]);
                        s00 = d0 * static_cast<float>(sc.x);
                        s01 = d0 * static_cast<float>(sc.y);
                        s10 = d1 * static_cast<float>(sc.z);
                        s11 = d1 * static_cast<float>(sc.w);
                    }
                    // The min terms of rows r and r + 8 (per half for kHalfMin: min0 | min1 << 8).
                    float m00 = 0.0F, m01 = 0.0F, m10 = 0.0F, m11 = 0.0F;
                    if constexpr (S == Scaling::kMin) {
                        m00 = m01 = f16_value(cs.dmin[j][r]) * static_cast<float>(cs.min[j][r]);
                        m10 = m11 = f16_value(cs.dmin[j][r + 8]) * static_cast<float>(cs.min[j][r + 8]);
                    } else if constexpr (S == Scaling::kHalfMin) {
                        const float dm0 = f16_value(cs.dmin[j][r]), dm1 = f16_value(cs.dmin[j][r + 8]);
                        m00 = dm0 * static_cast<float>(cs.min[j][r] & 0xFF);
                        m01 = dm0 * static_cast<float>(cs.min[j][r] >> 8);
                        m10 = dm1 * static_cast<float>(cs.min[j][r + 8] & 0xFF);
                        m11 = dm1 * static_cast<float>(cs.min[j][r + 8] >> 8);
                    }
    #pragma unroll
                    for (int nt = 0; nt < kNTiles; ++nt) {
                        float(&o)[4] = acc.r[mi][nt];
                        if constexpr (S == Scaling::kHalf || S == Scaling::kHalfMin) {
                            float p[4] = {0.0F, 0.0F, 0.0F, 0.0F}, q[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                            mma(p, lo, b[nt].x, b[nt].y);
                            mma(q, hi, b[nt].z, b[nt].w);
                            o[0] = fmaf(s01, q[0], fmaf(s00, p[0], o[0]));
                            o[1] = fmaf(s01, q[1], fmaf(s00, p[1], o[1]));
                            o[2] = fmaf(s11, q[2], fmaf(s10, p[2], o[2]));
                            o[3] = fmaf(s11, q[3], fmaf(s10, p[3], o[3]));
                        } else {
                            float p[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                            mma(p, lo, b[nt].x, b[nt].y);
                            mma(p, hi, b[nt].z, b[nt].w);
                            o[0] = fmaf(s00, p[0], o[0]);
                            o[1] = fmaf(s00, p[1], o[1]);
                            o[2] = fmaf(s10, p[2], o[2]);
                            o[3] = fmaf(s10, p[3], o[3]);
                        }
                        if constexpr (kSums) {
                            o[0] = fmaf(-m01, xs[nt][0][1], fmaf(-m00, xs[nt][0][0], o[0]));
                            o[1] = fmaf(-m01, xs[nt][1][1], fmaf(-m00, xs[nt][1][0], o[1]));
                            o[2] = fmaf(-m11, xs[nt][0][1], fmaf(-m10, xs[nt][0][0], o[2]));
                            o[3] = fmaf(-m11, xs[nt][1][1], fmaf(-m10, xs[nt][1][0], o[3]));
                        }
                    }
                }
            }

        }
    };
    // Every iteration commits two groups (x, then raw; either may be empty), so cp_wait<1> at its top
    // completes everything but the previous iteration's raw unit, which is first decoded at least one
    // iteration later.
    issue_raw(0, 0);
    ops::cp_commit();
    ops::cp_wait<0>();
    __syncthreads();
    issue_x(0, 0);
    ops::cp_commit();
    if (units > 1) { issue_raw(1, 1); }
    ops::cp_commit();
    decode(0);
    for (int st = 0; st < stages; ++st) {
        ops::cp_wait<1>();
        __syncthreads(); // stage st's image and x are in place; stage st - 1's buffers are free
        const int next = st + 1;
        if (next < stages) { issue_x(next, next & 1); }
        ops::cp_commit();
        if (next < stages && next % kUnitStages == 0 && next / kUnitStages + 1 < units) {
            issue_raw((next / kUnitStages + 1) & 1, next / kUnitStages + 1);
        }
        ops::cp_commit();
        if (next < stages) { decode(next); }
        mma_stage(st);
    }
    ops::cp_wait<0>();
    __syncthreads(); // the caller may reuse shared memory
}

// A dense call's tile: rows [kRows bx, + kRows) of a segment (or slab) whose plane holds `tiles` row
// tiles (the last image pair clamped to the last tile; those rows' outputs are not stored) against
// columns [kColumns by, + kColumns). out[t * n + out_offset + row] for row < rows, t < columns.
template <Scaling S, bool OutF32>
__global__ void __launch_bounds__(kThreads, 1)
    tile_kernel(const std::uint8_t* __restrict__ plane, int tiles, int rows, int slices, DenseColumns cols,
                void* __restrict__ out, int n, int out_offset, int kappa_log2) {
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm = *reinterpret_cast<Shared*>(bf16_shared);
    const int stages = slices / kSlices;
    const int tile0  = static_cast<int>(blockIdx.x) * 2;
    cols.col0        = static_cast<int>(blockIdx.y) * kColumns;
    constexpr std::size_t kStride = sizeof(tile::CodeStage);
    const std::uint8_t* const images[2] = {plane + static_cast<std::size_t>(min(tile0, tiles - 1)) * stages * kStride,
                                           plane + static_cast<std::size_t>(min(tile0 + 1, tiles - 1)) * stages * kStride};
    Accumulators acc;
    compute<S>(sm, images, stages, cols, acc);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = static_cast<int>(blockIdx.x) * kRows + out_row(warp, lane, mi, q);
                const int col = cols.col0 + out_column(warp, lane, nt, q);
                if (row < rows && col < cols.columns) {
                    const float v        = ldexpf(acc.r[mi][nt][q], kappa_log2);
                    const std::size_t at = static_cast<std::size_t>(col) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = v;
                    } else {
                        static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(v);
                    }
                }
            }
        }
    }
}

} // namespace infernix::ggml::bf16
