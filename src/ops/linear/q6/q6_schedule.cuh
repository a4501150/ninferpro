#pragma once

// Q6 schedule surface: aliases of the shared RowSplit schedules bound to the
// Q6 stored format. The sliced-K schedule keeps its original shape (no static
// K, no token capacity, no reduction choice).

#include "ops/common/memory.cuh"
#include "ops/linear/rowsplit/schedule.cuh"

namespace infernix::ops::detail {

using Q6MmaFragmentPipeline = RowSplitMmaFragmentPipeline;
using Q6ScaleLoad           = RowSplitScaleLoad;

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm>
using Q6A16SimtSchedule =
    RowSplitA16SimtSchedule<Q6RowSplitFormat, BlockRows, BlockTokens, WarpsPerRow,
                            GroupsPerWarpStage, Stages, CodeCache, MinBlocksPerSm>;

template <int BlockRows, int WarpsPerRow, int GroupsPerWarpStage, int Stages, Cache CodeCache,
          int MinBlocksPerSm>
using Q6A16GemvSchedule = Q6A16SimtSchedule<BlockRows, 1, WarpsPerRow, GroupsPerWarpStage, Stages,
                                            CodeCache, MinBlocksPerSm>;

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Q6MmaFragmentPipeline FragmentPipeline, Cache WeightCache,
          Cache ActivationCache, Q6ScaleLoad ScaleLoadMode, int ActivationStages = Stages>
using Q6A16MmaSchedule =
    RowSplitA16MmaSchedule<Q6RowSplitFormat, BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens,
                           Stages, MinBlocksPerSm, FragmentPipeline, WeightCache, ActivationCache,
                           ScaleLoadMode, ActivationStages>;

template <int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm>
using Q6A16SlicedKMmaSchedule =
    RowSplitA16SlicedKMmaSchedule<Q6RowSplitFormat, BlockRows, BlockTokens, KWarps, Stages,
                                  WeightCache, ActivationCache, MinBlocksPerSm, 0, BlockTokens,
                                  RowSplitSlicedKReduction::Linear>;

} // namespace infernix::ops::detail
