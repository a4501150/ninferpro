#pragma once

// Shared schedules for the RowSplit kernel families, parameterized by the
// stored weight format (see format_trait.cuh). The per-format headers alias
// these with their existing names and template signatures; the member values
// are the ones the per-format schedules carried.

#include "ops/common/memory.cuh"
#include "ops/linear/rowsplit/format_trait.cuh"

#include <type_traits>

namespace infernix::ops::detail {

enum class RowSplitMmaFragmentPipeline {
    Serial,
    PingPong,
};

enum class RowSplitScaleLoad {
    Scalar16,
    Pair32,
};

enum class RowSplitSlicedKReduction {
    Linear,
    Pairwise,
};

template <class Format, int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage,
          int Stages, Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
struct RowSplitA16SimtSchedule {
    using RowSplitFormat = Format;
    static_assert(BlockRows > 0 && BlockRows <= 32);
    static_assert(BlockTokens > 0 && BlockTokens <= 8);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpStage > 0 && GroupsPerWarpStage % 2 == 0);
    static_assert(Stages >= 2 && Stages <= 8);
    static_assert(MinBlocksPerSm > 0);
    static constexpr bool kPredicated        = Predicated;
    static constexpr int kBlockRows          = BlockRows;
    static constexpr int kBlockTokens        = BlockTokens;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpStage = GroupsPerWarpStage;
    static constexpr int kStages             = Stages;
    static constexpr Cache kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm     = MinBlocksPerSm;
    static constexpr int kWarps              = BlockRows * WarpsPerRow;
    static constexpr int kThreads            = kWarps * 32;
    static constexpr int kStageK             = GroupsPerWarpStage * Format::Storage::kGroupK;
    static constexpr int kCodeVecsPerStage =
        GroupsPerWarpStage * Format::Storage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kHighVecsPerStage =
        GroupsPerWarpStage * Format::kHighBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kScalePairsPerStage = GroupsPerWarpStage / 2;
    static constexpr int kCodePhases =
        (GroupsPerWarpStage + Format::kSimtGroupsPerPhase - 1) / Format::kSimtGroupsPerPhase;
    static constexpr int kStagingBytes =
        kWarps * Stages * GroupsPerWarpStage *
        (Format::Storage::kCodeBytesPerGroup + Format::kHighBytesPerGroup +
         Format::Storage::kScaleBytesPerGroup);
    static constexpr int kPartialBytes = WarpsPerRow > 1 ? kWarps * BlockTokens * sizeof(float) : 0;
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024, "RowSplit SIMT shared memory exceeds 48 KiB");
};

template <class Format, int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int Stages, int MinBlocksPerSm, RowSplitMmaFragmentPipeline FragmentPipeline,
          Cache WeightCache, Cache ActivationCache, RowSplitScaleLoad ScaleLoadMode,
          int ActivationStages = Stages>
struct RowSplitA16MmaSchedule {
    using RowSplitFormat              = Format;
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static constexpr int kBlockK      = BlockK;
    static constexpr int kWarpRows    = WarpRows;
    static constexpr int kWarpTokens  = WarpTokens;

    static constexpr int kStages                                   = Stages;
    static constexpr int kMinBlocksPerSm                           = MinBlocksPerSm;
    static constexpr RowSplitMmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Cache kWeightCache                            = WeightCache;
    static constexpr Cache kActivationCache                        = ActivationCache;
    static constexpr RowSplitScaleLoad kScaleLoadMode              = ScaleLoadMode;
    static constexpr int kActivationStages                         = ActivationStages;

    static constexpr int kWarpGridRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpGridTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps          = kWarpGridRows * kWarpGridTokens;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaTokens      = kWarpTokens / 8;
    static constexpr int kMmaKSteps      = kBlockK / 16;
    static constexpr int kGroupsPerK     = kBlockK / Format::Storage::kGroupK;
    static constexpr int kScaleBytes =
        kScaleLoadMode == RowSplitScaleLoad::Pair32 ? 4 : Format::Storage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kStages * kBlockRows * kGroupsPerK *
            (Format::Storage::kCodeBytesPerGroup + Format::kHighBytesPerGroup + kScaleBytes);

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert(kBlockK % Format::Storage::kGroupK == 0,
                  "RowSplit MMA K tile must contain complete quant groups");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockTokens % kWarpTokens == 0,
                  "RowSplit MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpTokens % 8 == 0,
                  "RowSplit MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kStages >= 1 && kStages <= 8,
                  "RowSplit MMA cp.async pipeline depth must fit cp_wait");
    static_assert(
        kActivationStages == 1 || kActivationStages == kStages,
        "RowSplit MMA activation staging is single-buffered or follows the quant pipeline");
    static_assert(kActivationStages == kStages || kStages < 8,
                  "RowSplit single-buffered activations need a separate async group");
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "RowSplit MMA staged shared memory exceeds the static 48 KiB budget");
};

template <class Format, int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm, int StaticK = 0,
          int TokenCapacity                  = BlockTokens,
          RowSplitSlicedKReduction Reduction = RowSplitSlicedKReduction::Linear>
struct RowSplitA16SlicedKMmaSchedule {
    using RowSplitFormat = Format;
    static_assert(BlockRows == 16 || BlockRows == 32);
    static_assert(BlockTokens >= 8 && BlockTokens <= 64 && BlockTokens % 8 == 0);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % 128 == 0));
    static_assert(TokenCapacity > 0 && TokenCapacity <= BlockTokens);
    // Static K specializes a padding-free geometry, including physical row strides.
    static constexpr auto kReduction        = Reduction;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kTokenCapacity     = TokenCapacity;
    static constexpr int kKWarps            = KWarps;
    static constexpr int kWarps             = KWarps;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kWarpK             = 64;
    static constexpr int kBlockK            = KWarps * kWarpK;
    static constexpr int kStages            = Stages;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kMmaRows           = BlockRows / 16;
    static constexpr int kMmaTokens         = BlockTokens / 8;
    // Code/high/scale planes and BF16 activations. Partial results reuse this storage.
    static constexpr int kStagingBytes =
        Stages * (BlockRows * KWarps *
                      (Format::Storage::kCodeBytesPerGroup + Format::kHighBytesPerGroup +
                       Format::Storage::kScaleBytesPerGroup) +
                  KWarps * BlockTokens * 64 * 2);
    static constexpr int kPartialBytes = KWarps * BlockRows * BlockTokens * sizeof(float);
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024, "RowSplit sliced-K shared memory exceeds 48 KiB");
};

} // namespace infernix::ops::detail
