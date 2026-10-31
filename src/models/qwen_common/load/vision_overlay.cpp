#include "models/qwen_common/load/vision_overlay.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernix::models::qwen_common {
namespace {

// Pinned byte placement (offset, size) for each object, from the materialization plan.
struct PinnedRangeIndex {
    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> by_object;

    explicit PinnedRangeIndex(const artifact::MaterializationPlan& plan) {
        by_object.reserve(plan.pinned_objects.size());
        for (const artifact::PinnedPlacement& placement : plan.pinned_objects) {
            by_object.emplace(placement.object.index,
                              std::make_pair(static_cast<std::size_t>(placement.offset),
                                             static_cast<std::size_t>(placement.bytes)));
        }
    }

    std::pair<std::size_t, std::size_t> lookup(artifact::ObjectHandle handle) const {
        const auto it = by_object.find(handle.index);
        if (it == by_object.end()) {
            throw std::logic_error("vision overlay group tensor is not pinned");
        }
        return it->second;
    }
};

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

// Stage a group's distinct objects in the order listed, recording each object's pinned and
// staged offsets. Objects shared by several bindings (for example a fused qkv) are staged once.
VisionOverlayGroup build_group(const PinnedRangeIndex& index,
                               const std::vector<artifact::ObjectHandle>& handles) {
    VisionOverlayGroup group;
    std::vector<std::size_t> seen;
    seen.reserve(handles.size());
    std::size_t staging = 0;
    for (const artifact::ObjectHandle handle : handles) {
        if (std::find(seen.begin(), seen.end(), handle.index) != seen.end()) { continue; }
        seen.push_back(handle.index);
        const auto [pinned, bytes] = index.lookup(handle);
        group.segments.push_back(VisionOverlaySegment{pinned, staging, bytes});
        staging += bytes;
    }
    group.bytes = staging;
    return group;
}

} // namespace

VisionOverlayLayout make_vision_overlay_layout(const artifact::MaterializationPlan& plan,
                                               const std::vector<artifact::ObjectHandle>& prelude,
                                               const std::vector<std::vector<artifact::ObjectHandle>>& layers,
                                               const std::vector<artifact::ObjectHandle>& merger) {
    const PinnedRangeIndex index(plan);
    VisionOverlayLayout out;
    out.prelude = build_group(index, prelude);
    out.layers.reserve(layers.size());
    for (const auto& handles : layers) {
        VisionOverlayGroup group = build_group(index, handles);
        out.slot_bytes           = std::max(out.slot_bytes, group.bytes);
        out.layers.push_back(std::move(group));
    }
    out.merger        = build_group(index, merger);
    out.staging_bytes = staging_align(out.prelude.bytes) + staging_align(out.merger.bytes) +
                        2 * staging_align(out.slot_bytes);
    return out;
}

} // namespace infernix::models::qwen_common
