#pragma once

// Where a vision tower kept in pinned host memory (vision offload) sits in the pinned weight
// block, grouped as its weight stream (models/common/vision/weight_stream.h) stages it: the
// prelude before the first layer, one group per layer, and the merger after the last.

#include "artifact/materializer.h"
#include "core/evictable_weight_pool.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::models::common {

// Staging regions and the objects inside them start on 256-byte boundaries.
[[nodiscard]] constexpr std::size_t vision_staging_align(std::size_t bytes) noexcept {
    return (bytes + 255) / 256 * 256;
}

// One contiguous run of pinned bytes staged into a group's device slot. The pinned offset is
// the object's position in the pinned block; the staging offset is its position in the group's
// device slot. A group's objects are not necessarily adjacent in the pinned block (the binder
// packs pinned objects in registration order, and a converter may store a packed parent apart
// from a layer's other weights), so a group is a list of segments rather than one range.
struct VisionOverlaySegment {
    std::size_t pinned_offset  = 0;
    std::size_t staging_offset = 0;
    std::size_t bytes          = 0;
};

struct VisionOverlayGroup {
    std::vector<VisionOverlaySegment> segments;
    std::size_t bytes = 0; // the staged extent: the segments, each on a 256-byte boundary
};

// Byte layout of the vision groups inside the pinned weight block. slot_bytes is the largest
// single layer group.
struct VisionOverlayLayout {
    VisionOverlayGroup prelude; // what runs before the first layer (the patch embedding)
    std::vector<VisionOverlayGroup> layers;
    VisionOverlayGroup merger; // what runs after the last layer
    std::size_t slot_bytes    = 0;
    std::size_t staging_bytes = 0;  // prelude + merger + two layer slots, aligned

    // The tower's pinned bytes: every group's.
    [[nodiscard]] std::size_t tower_bytes() const noexcept;
};

// Runtime assets the encode needs, published on the model when the engine runs with vision
// offload on. `pool` is set by a family that takes the staging from an evictable weight pool;
// it is non-const because the encode calls the pool's evict()/restore().
struct VisionOverlayAssets {
    EvictableWeightPool* pool               = nullptr;
    const std::byte* pinned_block           = nullptr;
    std::size_t pinned_bytes                = 0;
    std::size_t ladder_bytes                = 0;  // evictable tail a window may borrow
    VisionOverlayLayout layout;
};

// The overlay layout from each group's objects in staging order (prelude, one list per layer,
// merger), as placed by the materialization plan. Objects shared by several bindings (a packed
// parent) are staged once.
[[nodiscard]] VisionOverlayLayout make_vision_overlay_layout(
    const artifact::MaterializationPlan& plan, const std::vector<artifact::ObjectHandle>& prelude,
    const std::vector<std::vector<artifact::ObjectHandle>>& layers,
    const std::vector<artifact::ObjectHandle>& merger);

} // namespace infernix::models::common
