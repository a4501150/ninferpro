#pragma once

// The Qwen3.5-only remainder of the vision overlay layout: resolving the family's own weight
// binding into the shared overlay layout (qwen_common/load/vision_overlay.h).

#include "models/qwen3_5/weights.h"
#include "models/qwen_common/load/vision_overlay.h"

#include <vector>

namespace infernix::models::qwen3_5 {

namespace loading {
struct PendingWeight;
}

// Resolve each vision group's contiguous byte range in the pinned block. Throws when a
// group's objects are not all pinned or are not packed contiguously (gap beyond alignment).
[[nodiscard]] qwen_common::VisionOverlayLayout compute_vision_overlay_layout(
    const ModelWeights& weights, const std::vector<loading::PendingWeight>& pending,
    const artifact::MaterializationPlan& plan);

} // namespace infernix::models::qwen3_5
