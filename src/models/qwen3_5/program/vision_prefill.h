#pragma once
#include "models/qwen3_5/program/internal.h"

#include "models/qwen_common/program/vision_control.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace infernix::models::qwen3_5::execution {
class VisionPrefillSession;
}

namespace infernix::models::qwen3_5::detail {

// One state per Program workspace. Only Vision encoding can replace the handoff;
// Text and decode use the disjoint general workspace and preserve its owner.
class VisionHandoffState {
    friend class execution::VisionPrefillSession;

    const execution::VisionPrefillSession* owner_ = nullptr;
    std::uint64_t generation_                     = 0;
};

struct VisionUseSpan {
    std::uint32_t begin               = 0;
    std::uint32_t end                 = 0;
    std::uint32_t prepared_item_index = 0;
    std::uint32_t control_index       = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen_common::VisionControlPlan> control_plan;
    std::shared_ptr<const qwen_common::VisionControl> control;
    std::vector<VisionUseSpan> uses;
    std::size_t max_merged_count = 0;
};

} // namespace infernix::models::qwen3_5::detail
