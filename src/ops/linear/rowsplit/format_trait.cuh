#pragma once

// Compile-time weight formats for the RowSplit kernel families (SIMT, MMA,
// sliced-K MMA). The shared bodies in this directory own grouping, staging and
// scheduling; a trait pins what a stored format changes: geometry, decode
// atoms, extra planes, and the staging predicates the generic body cannot
// express. A new stored format adds a trait, not a new kernel.

#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cstdint>

namespace infernix::ops::detail {

// High-plane bits a SIMT decode reads before the stage's scale (simt_preload_high); formats that
// read them inside the decode, or have none, return this tag.
struct RowSplitNoHighBits {};

// Q4 integers are exactly representable in BF16. Scaling each 64-element
// contraction in FP32 avoids a scaled-weight BF16 materialization.
__device__ __forceinline__ unsigned q4_sliced_k_integer_pair(std::uint8_t code) {
    const int q0 = (int(code & 15) ^ 8) - 8;
    const int q1 = (int(code >> 4) ^ 8) - 8;

    union {
        __nv_bfloat162 value;
        unsigned bits;
    } pair;

    pair.value = __floats2bfloat162_rn(float(q0), float(q1));
    return pair.bits;
}

__device__ __forceinline__ unsigned q5_sliced_k_integer_pair(std::uint8_t code, std::uint8_t high,
                                                             int pair) {
    const int shift = pair * 2;
    const int q0    = ((int(code & 15) | (((high >> shift) & 1) << 4)) ^ 16) - 16;
    const int q1    = ((int(code >> 4) | (((high >> (shift + 1)) & 1) << 4)) ^ 16) - 16;

    union {
        __nv_bfloat162 value;
        unsigned bits;
    } result;

    result.value = __floats2bfloat162_rn(float(q0), float(q1));
    return result.bits;
}

__device__ __forceinline__ unsigned q6_sliced_k_integer_pair(std::uint8_t code, std::uint8_t high,
                                                             int shift) {
    const int q0 = ((int(code & 15) | (((high >> shift) & 3) << 4)) ^ 32) - 32;
    const int q1 = ((int(code >> 4) | (((high >> (shift + 2)) & 3) << 4)) ^ 32) - 32;

    union {
        __nv_bfloat162 value;
        unsigned bits;
    } pair;

    pair.value = __floats2bfloat162_rn(float(q0), float(q1));
    return pair.bits;
}

// One packed nibble and one fp16 scale per 64-element group; no high-bit plane.
struct Q4RowSplitFormat {
    using Storage = Q4RowSplitStorage;

    static constexpr bool kHasHigh          = false;
    static constexpr int kHighBytesPerGroup = 0;

    // --- SIMT family ---

    static __device__ __forceinline__ int simt_vector_groups(int k) {
        return k / Storage::kGroupK;
    }

    static __device__ __forceinline__ RowSplitNoHighBits simt_preload_high(const uint4*, int, int) {
        return {};
    }

    static __device__ __forceinline__ void
    simt_decode_eight(const uint4* /*staged_high*/, RowSplitNoHighBits, std::uint32_t packed,
                      std::uint16_t scale_bits, int /*phase*/, int /*lane*/, float (&weights)[8]) {
        Q4SimtDecodeAtom::decode_eight(packed, scale_bits, weights);
    }

    // The fully reduced row routes through linear_finish_row, so apply_row
    // epilogues are available to this format's SIMT consumers.
    static constexpr bool kSimtApplyRow = true;
    static constexpr bool kSimtScalesBeforeHigh = false;
    static constexpr bool kSimtHighByPairs      = false;

    // Whole scale pairs only: no scalar remainder beyond the staged groups.
    static constexpr bool kScalarRemainder = false;

    // --- MMA family ---

    // Partial-K tiles stage the tile's groups without bounding the group index
    // by the logical group count.
    static constexpr bool kBoundGroups = false;

    template <class Staging>
    static __device__ __forceinline__ __nv_bfloat162
    mma_decode_pair(const Staging& staging, int stage, const std::uint8_t* scale_ptr,
                    std::int64_t staged_group, int lane) {
        return Q4MmaDecodeAtom::decode_pair(staging.Cr[stage], scale_ptr, staged_group, lane);
    }

    // The single-buffered quant prefetch runs only while the quant ring is
    // deeper than the activation ring.
    static __host__ __device__ constexpr bool mma_quant_prefetch(int activation_stages, int stages) {
        return activation_stages == 1 && activation_stages != stages;
    }

    // Epilogues may take over the tile finish (finish_tile).
    static constexpr bool kFinishTile = false;

    // --- sliced-K family ---

    static __device__ __forceinline__ int sliced_group_count(int logical_k) {
        return logical_k / Storage::kGroupK;
    }

    template <bool Full>
    static __device__ __forceinline__ bool sliced_x_group_valid(int first_group, int warp,
                                                                int groups, int /*logical_k*/,
                                                                int /*k8*/) {
        return Full || first_group + warp < groups;
    }

    template <class Staging>
    static __device__ __forceinline__ unsigned sliced_decode_pair(const Staging& staging, int slot,
                                                                  int row, int warp, int byte) {
        return q4_sliced_k_integer_pair(staging.codes[slot][row][warp][byte]);
    }

    // The store also bounds each token by the CTA's token capacity.
    static constexpr bool kSlicedCapacityGuard = true;

    // Streams its weights through the K loop: the kernel waits for the
    // producer first and lets dependents launch only once that loop is done.
    static constexpr bool kSlicedPdl = true;
    static constexpr bool kSlicedRowPolicyParam = true;

    static constexpr bool kSlicedScaleGuard  = false;
    static constexpr bool kSlicedSyncRestage = false;

    // Epilogues may take over the fragment finish (store_fragment).
    static constexpr bool kSlicedStoreFragment = true;
};

// A packed nibble plus a one-bit high plane and one fp16 scale per 64-element
// group; K only needs 8-element alignment, so partial scale pairs are handled.
struct Q5RowSplitFormat {
    using Storage = Q5RowSplitStorage;

    static constexpr bool kHasHigh          = true;
    static constexpr int kHighBytesPerGroup = Storage::kHighBytesPerGroup;

    // --- SIMT family ---

    // Only complete scale pairs run through the staged pipeline; the scalar
    // remainder below covers the tail.
    static __device__ __forceinline__ int simt_vector_groups(int k) {
        return k % 8 == 0 ? (k / 128) * 2 : 0;
    }

    // The high-plane byte is read inside the decode, after the scale.
    static __device__ __forceinline__ RowSplitNoHighBits simt_preload_high(const uint4*, int, int) {
        return {};
    }

    static __device__ __forceinline__ void
    simt_decode_eight(const uint4* staged_high, RowSplitNoHighBits, std::uint32_t packed,
                      std::uint16_t scale_bits, int phase, int lane, float (&weights)[8]) {
        Q5SimtDecodeAtom::decode_eight(
            packed, reinterpret_cast<const std::uint8_t*>(staged_high)[phase * 32 + lane],
            scale_bits, weights);
    }

    static constexpr bool kSimtApplyRow = true;
    // The SIMT stage addresses its scale pairs before its high plane and copies one high vector
    // per scale pair. Both only order address arithmetic: the generic order compiles this
    // format's SIMT kernels to different code with more registers.
    static constexpr bool kSimtScalesBeforeHigh = true;
    static constexpr bool kSimtHighByPairs      = true;

    // Warp zero decodes the K tail that no complete scale pair covers.
    static constexpr bool kScalarRemainder = true;

    static __device__ __forceinline__ void
    simt_remainder_pair(const std::uint8_t* codes, const std::uint8_t* high,
                        const std::uint8_t* scales, std::int64_t group_index, int lane, float& w0,
                        float& w1) {
        Q5ScalarDecodeAtom::load_pair(codes, high, scales, group_index, lane, w0, w1);
    }

    // --- MMA family ---

    // Partial-K tiles bound the staged group index by the logical group count.
    static constexpr bool kBoundGroups = true;

    // Q5 stages its high plane in group pairs, so one 16-byte zfill covers two
    // groups and an odd groups-per-tile count halves the transfer.
    template <class Schedule, bool Full, class Staging>
    static __device__ __forceinline__ void mma_stage_high(Staging& staging, int stage,
                                                          const std::uint8_t* high, int row0,
                                                          int rows, int k, int groups_per_row,
                                                          int group0, int tid) {
        constexpr int BM  = Schedule::kBlockRows;
        constexpr int GPB = Schedule::kGroupsPerK;

        constexpr int kHighGroups = GPB % 2 == 0 ? 2 : 1;
        for (int item = tid; item < BM * GPB / kHighGroups; item += Schedule::kThreads) {
            const int local_row = item / (GPB / kHighGroups);
            const int group     = (item % (GPB / kHighGroups)) * kHighGroups;
            const int row       = row0 + local_row;
            const bool valid    = Full || (row < rows && group0 + group < (k + 63) / 64);
            const std::int64_t index =
                valid ? std::int64_t(row) * groups_per_row + group0 + group : 0;
            cp_async_zfill<kHighGroups * 8, kHighGroups == 2 ? Schedule::kWeightCache : Cache::ca>(
                &staging.Hr[stage][item * kHighGroups * 8], high + index * 8,
                valid ? kHighGroups * 8 : 0);
        }
    }

    template <class Staging>
    static __device__ __forceinline__ __nv_bfloat162
    mma_decode_pair(const Staging& staging, int stage, const std::uint8_t* scale_ptr,
                    std::int64_t staged_group, int lane) {
        return Q5MmaDecodeAtom::decode_pair(staging.Cr[stage], staging.Hr[stage], scale_ptr,
                                            staged_group, lane);
    }

    static __host__ __device__ constexpr bool mma_quant_prefetch(int activation_stages, int stages) {
        return activation_stages == 1 && activation_stages != stages;
    }

    // Epilogues may take over the tile finish (finish_tile).
    static constexpr bool kFinishTile = true;

    // --- sliced-K family ---

    // Partial final groups run through the sliced path, bounded per element.
    static __device__ __forceinline__ int sliced_group_count(int logical_k) {
        return (logical_k + 63) / 64;
    }

    template <bool Full>
    static __device__ __forceinline__ bool sliced_x_group_valid(int first_group, int warp,
                                                                int /*groups*/, int logical_k,
                                                                int k8) {
        return Full || (first_group + warp) * 64 + k8 + 8 <= logical_k;
    }

    // Scale pairs share one 16-byte high staging vector.
    template <class Schedule, bool Full, class Staging, class RowPolicy>
    static __device__ __forceinline__ void
    sliced_stage_high(Staging& shared, int slot, const std::uint8_t* high, int row_begin, int rows,
                       int groups, int row_groups, int first_group, int tid,
                       const RowPolicy& row_policy) {
        constexpr int R = Schedule::kBlockRows;
        constexpr int W = Schedule::kKWarps;

        for (int item = tid; item < R * (W / 2); item += Schedule::kThreads) {
            const int row    = item / (W / 2);
            const int pair   = (item % (W / 2)) * 2;
            const bool valid = Full || (row_policy.weight_row(row_begin, row) < rows &&
                                        first_group + pair < groups);
            const auto index =
                valid ? std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups +
                            first_group + pair
                      : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(&shared.staging.high[slot][row][pair][0],
                                                       high + index * 8, valid ? 16 : 0);
        }
    }

    template <class Staging>
    static __device__ __forceinline__ unsigned sliced_decode_pair(const Staging& staging, int slot,
                                                                  int row, int warp, int byte) {
        return q5_sliced_k_integer_pair(staging.codes[slot][row][warp][byte],
                                        staging.high[slot][row][warp][byte / 4], byte % 4);
    }

    // The store also bounds each token by the CTA's token capacity.
    static constexpr bool kSlicedCapacityGuard = true;

    static constexpr bool kSlicedPdl          = false;
    static constexpr bool kSlicedRowPolicyParam = true;
    static constexpr bool kSlicedScaleGuard   = true;
    static constexpr bool kSlicedSyncRestage  = false;
    static constexpr bool kSlicedStoreFragment = true;
};

// A packed nibble plus a two-bit high plane and one fp16 scale per 64-element
// group; K stays a multiple of 128.
struct Q6RowSplitFormat {
    using Storage = Q6RowSplitStorage;

    static constexpr bool kHasHigh          = true;
    static constexpr int kHighBytesPerGroup = Storage::kHighBytesPerGroup;

    // --- SIMT family ---

    static __device__ __forceinline__ int simt_vector_groups(int k) {
        return k / Storage::kGroupK;
    }

    // The high-plane bits are read before the scale.
    static __device__ __forceinline__ std::uint16_t simt_preload_high(const uint4* staged_high,
                                                                      int phase, int lane) {
        return reinterpret_cast<const std::uint16_t*>(staged_high)[phase * 32 + lane];
    }

    static __device__ __forceinline__ void
    simt_decode_eight(const uint4* /*staged_high*/, std::uint16_t high_bits, std::uint32_t packed,
                      std::uint16_t scale_bits, int /*phase*/, int /*lane*/,
                      float (&weights)[8]) {
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }

    // The reduced row finishes per token; no apply_row epilogue routing.
    static constexpr bool kSimtApplyRow = false;
    static constexpr bool kSimtScalesBeforeHigh = false;
    static constexpr bool kSimtHighByPairs      = false;

    static constexpr bool kScalarRemainder = false;

    // --- MMA family ---

    static constexpr bool kBoundGroups = false;

    template <class Schedule, bool Full, class Staging>
    static __device__ __forceinline__ void mma_stage_high(Staging& staging, int stage,
                                                          const std::uint8_t* high, int row0,
                                                          int rows, int /*k*/, int groups_per_row,
                                                          int group0, int tid) {
        constexpr int BM  = Schedule::kBlockRows;
        constexpr int GPB = Schedule::kGroupsPerK;

        for (int row_group = tid; row_group < BM * GPB; row_group += Schedule::kThreads) {
            const int local_row = row_group / GPB;
            const int group     = row_group - local_row * GPB;
            const int row       = row0 + local_row;
            auto* dst           = &staging.Hr[stage][row_group * Storage::kHighBytesPerGroup];
            if constexpr (Full) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                cp_async<16, Schedule::kWeightCache>(
                    dst, &high[group_index * Storage::kHighBytesPerGroup]);
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                    cp_async<16, Schedule::kWeightCache>(
                        dst, &high[group_index * Storage::kHighBytesPerGroup]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    template <class Staging>
    static __device__ __forceinline__ __nv_bfloat162
    mma_decode_pair(const Staging& staging, int stage, const std::uint8_t* scale_ptr,
                    std::int64_t staged_group, int lane) {
        return Q6MmaDecodeAtom::decode_pair(staging.Cr[stage], staging.Hr[stage], scale_ptr,
                                            staged_group, lane);
    }

    // The quant prefetch runs whenever activations are single-buffered.
    static __host__ __device__ constexpr bool mma_quant_prefetch(int activation_stages, int /*stages*/) {
        return activation_stages == 1;
    }

    static constexpr bool kFinishTile = false;

    // --- sliced-K family ---

    static __device__ __forceinline__ int sliced_group_count(int logical_k) {
        return logical_k / Storage::kGroupK;
    }

    template <bool Full>
    static __device__ __forceinline__ bool sliced_x_group_valid(int first_group, int warp,
                                                                int groups, int /*logical_k*/,
                                                                int /*k8*/) {
        return Full || first_group + warp < groups;
    }

    // One 16-byte high vector per group.
    template <class Schedule, bool Full, class Staging, class RowPolicy>
    static __device__ __forceinline__ void
    sliced_stage_high(Staging& shared, int slot, const std::uint8_t* high, int row_begin, int rows,
                       int groups, int row_groups, int first_group, int tid,
                       const RowPolicy& /*row_policy*/) {
        constexpr int R = Schedule::kBlockRows;
        constexpr int W = Schedule::kKWarps;

        for (int item = tid; item < R * W; item += Schedule::kThreads) {
            const int group  = item % W;
            const int row    = item / W;
            const bool valid = Full || (row_begin + row < rows && first_group + group < groups);
            const std::int64_t index =
                valid ? std::int64_t(row_begin + row) * row_groups + first_group + group : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(&shared.staging.high[slot][row][group][0],
                                                       high + index * 16, valid ? 16 : 0);
        }
    }

    template <class Staging>
    static __device__ __forceinline__ unsigned sliced_decode_pair(const Staging& staging, int slot,
                                                                  int row, int warp, int byte) {
        return q6_sliced_k_integer_pair(staging.codes[slot][row][warp][byte],
                                        staging.high[slot][row][warp][byte / 2],
                                        (byte & 1) * 4);
    }

    // The token grid never splits a capacity, so the store only bounds row and
    // token by the runtime counts.
    static constexpr bool kSlicedCapacityGuard = false;

    static constexpr bool kSlicedPdl           = false;
    static constexpr bool kSlicedRowPolicyParam = false;
    static constexpr bool kSlicedScaleGuard    = false;
    // Every tile drains the CTA before restaging and always closes the group.
    static constexpr bool kSlicedSyncRestage   = true;
    static constexpr bool kSlicedStoreFragment = false;
};

} // namespace infernix::ops::detail
