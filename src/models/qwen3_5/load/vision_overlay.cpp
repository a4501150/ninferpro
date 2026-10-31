#include "models/qwen3_5/load/vision_overlay.h"

#include "models/qwen3_5/load/bindings.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5 {
namespace {

std::vector<artifact::ObjectHandle>
objects_for(const std::vector<loading::PendingWeight>& pending, WeightId id) {
    std::vector<artifact::ObjectHandle> out;
    for (const auto& part : pending.at(id.index).reference.binding.parts) { out.push_back(part.object); }
    return out;
}

} // namespace

qwen_common::VisionOverlayLayout compute_vision_overlay_layout(const ModelWeights& weights,
                                                  const std::vector<loading::PendingWeight>& pending,
                                                  const artifact::MaterializationPlan& plan) {
    if (!weights.vision) { throw std::logic_error("vision overlay layout requires vision weights"); }
    const VisionWeights& vision = *weights.vision;
    const auto handles = [&](std::initializer_list<WeightId> ids) {
        std::vector<artifact::ObjectHandle> out;
        for (const WeightId id : ids) {
            auto part = objects_for(pending, id);
            out.insert(out.end(), part.begin(), part.end());
        }
        return out;
    };
    std::vector<std::vector<artifact::ObjectHandle>> layers;
    layers.reserve(vision.layers.size());
    for (const VisionBlockWeights& s : vision.layers) {
        layers.push_back(handles({s.norm1.weight, s.norm1.bias, s.norm2.weight, s.norm2.bias, s.query, s.key, s.value,
                                  s.query_bias, s.key_bias, s.value_bias, s.output, s.output_bias, s.fc1, s.fc1_bias,
                                  s.fc2, s.fc2_bias}));
    }
    return qwen_common::make_vision_overlay_layout(
        plan, handles({vision.patch_embedding, vision.patch_embedding_bias, vision.position_embedding}), layers,
        handles({vision.merger_norm.weight, vision.merger_norm.bias, vision.merger_fc1, vision.merger_fc1_bias,
                 vision.merger_fc2, vision.merger_fc2_bias}));
}

} // namespace infernix::models::qwen3_5
