#pragma once

// Q5 schedule surface: aliases of the shared RowSplit schedules bound to the
// Q5 stored format, plus the Q5-specific direct-SIMT and GEMV schedules.

#include "ops/common/memory.cuh"
#include "ops/linear/rowsplit/schedule.cuh"

namespace infernix::ops::detail {

using Q5MmaFragmentPipeline = RowSplitMmaFragmentPipeline;
using Q5ScaleLoad           = RowSplitScaleLoad;
using Q5SlicedKReduction    = RowSplitSlicedKReduction;

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
using Q5A16SimtSchedule =
    RowSplitA16SimtSchedule<Q5RowSplitFormat, BlockRows, BlockTokens, WarpsPerRow,
                            GroupsPerWarpStage, Stages, CodeCache, MinBlocksPerSm, Predicated>;

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Q5MmaFragmentPipeline FragmentPipeline, Cache WeightCache,
          Cache ActivationCache, Q5ScaleLoad ScaleLoadMode, int ActivationStages = Stages>
using Q5A16MmaSchedule =
    RowSplitA16MmaSchedule<Q5RowSplitFormat, BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens,
                           Stages, MinBlocksPerSm, FragmentPipeline, WeightCache, ActivationCache,
                           ScaleLoadMode, ActivationStages>;

template <int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm, int StaticK = 0,
          int TokenCapacity = BlockTokens,
          Q5SlicedKReduction Reduction = Q5SlicedKReduction::Linear>
using Q5A16SlicedKMmaSchedule =
    RowSplitA16SlicedKMmaSchedule<Q5RowSplitFormat, BlockRows, BlockTokens, KWarps, Stages,
                                  WeightCache, ActivationCache, MinBlocksPerSm, StaticK,
                                  TokenCapacity, Reduction>;

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpTile,
          int MinBlocksPerSm, int StaticK = 0, bool ExactTokens = false>
struct Q5A16DirectSimtSchedule {
    static_assert(BlockRows > 0 && BlockTokens > 0 && BlockTokens <= 32);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpTile > 0 && GroupsPerWarpTile % 4 == 0);
    static_assert(MinBlocksPerSm > 0 && (StaticK == 0 || (StaticK > 0 && StaticK % 8 == 0)));
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile = GroupsPerWarpTile;
    static constexpr int kPhasesPerWarp     = GroupsPerWarpTile / 4;
    static constexpr int kBlockK            = WarpsPerRow * GroupsPerWarpTile * 64;
    static constexpr bool kExactTokens      = ExactTokens;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kWarps             = BlockRows * WarpsPerRow;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kSharedBytes       = kWarps * BlockTokens * sizeof(float);
    static_assert(kThreads <= 1024 && kSharedBytes <= 48 * 1024);
};

template <int BlockRows, int WarpsPerRow, int GroupsPerWarpTile, int Stages, Cache WeightCache,
          int MinBlocksPerSm, bool StageX = false, int StaticK = 0>
struct Q5A16GemvSchedule {
    static_assert(BlockRows > 0 && WarpsPerRow > 0 && BlockRows * WarpsPerRow <= 32);
    static_assert(GroupsPerWarpTile > 0 && GroupsPerWarpTile % 2 == 0);
    static_assert(Stages >= 1 && Stages <= 8 && MinBlocksPerSm > 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % 128 == 0));
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = 1;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile = GroupsPerWarpTile;
    static constexpr int kStages            = Stages;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr bool kStageX           = StageX;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kWarps             = BlockRows * WarpsPerRow;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kSharedBytes =
        (kWarps * Stages * GroupsPerWarpTile * 42 + kWarps * 4 + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024);
};

} // namespace infernix::ops::detail
