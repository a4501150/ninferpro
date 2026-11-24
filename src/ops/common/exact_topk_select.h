#pragma once

// Exact top-k selection over per-column FP32 scores, shared by the pooled-key indexers
// (QSA block selection; the sparse MLA indexer's pool selection). A column's scores live at
// scores[column * stride + id] for ids [0, n) with n = (position + 1) / ratio; the Op that
// computes them owns their arithmetic. The selection itself is exact over the 64-bit
// (score, lower-id-first) order key of ops/common/score_id_order.cuh:
//
//   selected = the `top` highest scores of the column's n ids, equal scores taken in ascending
//              id order, written in ascending id order; count = top.
//   when n <= top the column selects every id: count = -1 and `selected` is not touched.
//
// A column's selection depends only on its own scores and position, never on the other columns
// of the call. Scores must be finite. The 2048 histogram bins order the scores' 64-bit keys
// through a monotone bit slice of the ordered bits (ops/common/score_id_order.cuh): bits [30:20]
// for non-negative scores (a sum of ReLUs times a positive scale is; the QSA caller), bits
// [31:21] for signed scores (a weighted sum of ReLUs is). The kernel is the multi-slice radix
// select formerly owned by the QSA select stage (an 8-bit radix select over the boundary bin's
// candidates with a re-walk fallback when the bin exceeds the candidate buffer); moved here so
// both indexers use one facility. The launch zeroes the state it is given.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

inline constexpr int kTopkSelectThreads     = 512;
inline constexpr int kTopkScoreBins         = 2048; // ordered score bits [30:20]
inline constexpr int kTopkBinShift          = 20;
inline constexpr int kTopkCandidateCapacity = 4096;
inline constexpr int kTopkMaxSlices         = 8;    // selection CTAs per column
inline constexpr int kTopkSliceBlocks       = 4096; // blocks per CTA at the largest context

/// Bytes of the selection state (one 2048-bin histogram plus one arrival counter per column).
[[nodiscard]] std::size_t topk_select_state_bytes(std::int32_t columns);

/// Selects the top `top` ids of every column of the group [group_begin, group_begin + columns).
/// scores: FP32 [columns, stride] (column-relative rows); state: `topk_select_state_bytes(columns)`
/// of scratch, zeroed by the call; selected: I32 with slot (t, i) at t * top + i for absolute
/// column t = group_begin + c (a [top, total_columns] plane in dimension-major order); counts:
/// I32 [total columns] indexed absolutely. max_bounds bounds every column's n (for shared memory).
void exact_topk_select(const std::int32_t* positions, int ratio, int top, int group_begin,
                       int columns, const float* scores, int stride, std::uint32_t* state,
                       std::int32_t* selected, std::int32_t* counts, int max_bounds,
                       bool non_negative_scores, cudaStream_t stream);

} // namespace infernix::ops::detail
