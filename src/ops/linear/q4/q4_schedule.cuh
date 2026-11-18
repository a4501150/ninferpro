#pragma once

// Q4 schedule surface: aliases of the shared RowSplit schedules bound to the
// Q4 stored format, plus the Q4-specific GEMV schedule.

#include "ops/common/memory.cuh"
#include "ops/linear/rowsplit/schedule.cuh"

namespace infernix::ops::detail {

using Q4MmaFragmentPipeline = RowSplitMmaFragmentPipeline;
using Q4ScaleLoad           = RowSplitScaleLoad;
using Q4SlicedKReduction    = RowSplitSlicedKReduction;

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
using Q4A16SimtSchedule =
    RowSplitA16SimtSchedule<Q4RowSplitFormat, BlockRows, BlockTokens, WarpsPerRow,
                            GroupsPerWarpStage, Stages, CodeCache, MinBlocksPerSm, Predicated>;

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Q4MmaFragmentPipeline FragmentPipeline, Cache WeightCache,
          Cache ActivationCache, Q4ScaleLoad ScaleLoadMode, int ActivationStages = Stages>
using Q4A16MmaSchedule =
    RowSplitA16MmaSchedule<Q4RowSplitFormat, BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens,
                           Stages, MinBlocksPerSm, FragmentPipeline, WeightCache, ActivationCache,
                           ScaleLoadMode, ActivationStages>;

template <int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm, int StaticK = 0,
          int TokenCapacity = BlockTokens,
          Q4SlicedKReduction Reduction = Q4SlicedKReduction::Linear>
using Q4A16SlicedKMmaSchedule =
    RowSplitA16SlicedKMmaSchedule<Q4RowSplitFormat, BlockRows, BlockTokens, KWarps, Stages,
                                  WeightCache, ActivationCache, MinBlocksPerSm, StaticK,
                                  TokenCapacity, Reduction>;

enum class Q4GemvActivationAccess {
    Direct,
    CtaSharedFullK,
};

enum class Q4GemvLaneMapping {
    PackedByte2,
    PackedWord8,
};

enum class Q4GemvDecodeMode {
    ScalarInteger,
    Fp16Mantissa,
};

enum class Q4GemvCodeTransfer {
    SyncVector16,
    AsyncVector16,
};

enum class Q4GemvScaleAccess {
    Scalar16Shuffle,
    SharedPair32,
};

template <int RowsPerCta, int WarpsPerRow, int GroupsPerWarpTile, int PipelineStages,
          Q4GemvActivationAccess ActivationAccess, Q4GemvLaneMapping LaneMapping,
          Q4GemvDecodeMode DecodeMode, Q4GemvCodeTransfer CodeTransfer,
          Q4GemvScaleAccess ScaleAccess, Cache CodeCache, int StaticGroupsPerRow,
          int LaunchBoundsMinBlocks>
struct Q4A16GemvSchedule {
    static_assert(RowsPerCta > 0, "Q4 GEMV requires at least one row per CTA");
    static_assert(WarpsPerRow > 0, "Q4 GEMV requires at least one warp per row");
    static_assert(GroupsPerWarpTile > 0 && (GroupsPerWarpTile % 2) == 0,
                  "Q4 GEMV group tiles must contain whole scale pairs");
    static_assert(GroupsPerWarpTile <= 32,
                  "Q4 GEMV scalar-shuffle scale access is limited to one warp");
    static_assert(PipelineStages >= 1 && PipelineStages <= 8,
                  "Q4 GEMV pipeline depth must fit cp.async wait-group immediates");
    static_assert(StaticGroupsPerRow == 0 ||
                      (StaticGroupsPerRow > 0 && (StaticGroupsPerRow % 2) == 0),
                  "Q4 GEMV static row ownership must contain whole scale pairs");
    static_assert(LaunchBoundsMinBlocks >= 1, "Q4 GEMV launch-bounds occupancy must be positive");

    static constexpr int kBlockRows          = RowsPerCta;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile  = GroupsPerWarpTile;
    static constexpr int kStages             = PipelineStages;
    static constexpr auto kActivationAccess  = ActivationAccess;
    static constexpr auto kLaneMapping       = LaneMapping;
    static constexpr auto kDecodeMode       = DecodeMode;
    static constexpr auto kCodeTransfer      = CodeTransfer;
    static constexpr auto kScaleAccess       = ScaleAccess;
    static constexpr auto kCodeCache         = CodeCache;
    static constexpr int kStaticGroupsPerRow = StaticGroupsPerRow;
    static constexpr int kMinBlocksPerSm     = LaunchBoundsMinBlocks;

    static constexpr int kWarps   = kBlockRows * kWarpsPerRow;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kCodeVectorsPerTile =
        kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup / sizeof(uint4);
    static constexpr int kScalePairsPerTile = kGroupsPerWarpTile / 2;

    static constexpr int kBlockTokens = 1;
    static constexpr int kStagingBytes =
        kWarps * kStages *
        (kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup +
         (kScaleAccess == Q4GemvScaleAccess::SharedPair32 ? kScalePairsPerTile * 4 : 0));
    static constexpr int kSharedBytes = (kStagingBytes + kWarps * 4 + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024);
    static_assert(kWarps <= 32, "Q4 GEMV cannot exceed the CUDA CTA warp limit");
    static_assert(kThreads <= 1024, "Q4 GEMV cannot exceed the CUDA CTA thread limit");
    static_assert((kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup) % sizeof(uint4) == 0,
                  "Q4 GEMV code tile must be representable as 16-byte vectors");

    static_assert((kLaneMapping == Q4GemvLaneMapping::PackedByte2 &&
                   kDecodeMode == Q4GemvDecodeMode::ScalarInteger) ||
                      (kLaneMapping == Q4GemvLaneMapping::PackedWord8 &&
                       kDecodeMode == Q4GemvDecodeMode::Fp16Mantissa),
                  "Q4 GEMV lane mapping and decode mode must describe the same packed ownership");
    static_assert((kCodeTransfer == Q4GemvCodeTransfer::SyncVector16 &&
                   kScaleAccess == Q4GemvScaleAccess::Scalar16Shuffle && kStages == 1 &&
                   kCodeCache == Cache::ca) ||
                      (kCodeTransfer == Q4GemvCodeTransfer::AsyncVector16 &&
                       kScaleAccess == Q4GemvScaleAccess::SharedPair32),
                  "Q4 GEMV transfer and scale schedules must select an implemented path");
};

} // namespace infernix::ops::detail
