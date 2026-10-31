#include "models/qwen_common/vision_config.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace infernix::models::qwen_common {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

VisionConfig vision_config(const Json& value, std::span<const std::string_view> model_types) {
    require_members(value,
                    {"model_type", "depth", "hidden_size", "intermediate_size", "num_heads",
                     "patch_size", "temporal_patch_size", "spatial_merge_size",
                     "num_position_embeddings"},
                    {}, "vision config");
    const auto& type = value.at("model_type");
    if (!type.is_string() || std::none_of(model_types.begin(), model_types.end(), [&](std::string_view t) {
            return type.get<std::string>() == t;
        })) {
        throw ArtifactError("unknown Vision model_type");
    }
    VisionConfig out;
    out.depth                   = dimension(value, "depth");
    out.hidden_size             = dimension(value, "hidden_size");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_heads               = dimension(value, "num_heads");
    out.patch_size              = dimension(value, "patch_size");
    out.temporal_patch_size     = dimension(value, "temporal_patch_size");
    out.spatial_merge_size      = dimension(value, "spatial_merge_size");
    out.num_position_embeddings = dimension(value, "num_position_embeddings");
    out.position_grid_side =
        static_cast<std::uint32_t>(std::sqrt(double(out.num_position_embeddings)));
    if (out.hidden_size % out.num_heads || (out.hidden_size / out.num_heads) % 4 ||
        std::uint64_t(out.position_grid_side) * out.position_grid_side !=
            out.num_position_embeddings) {
        throw ArtifactError("invalid Vision head or square position geometry");
    }
    (void)out.patch_width();
    (void)out.merger_width();
    return out;
}

} // namespace

std::uint64_t VisionConfig::patch_width() const {
    return artifact::checked_mul(artifact::checked_mul(3, temporal_patch_size, "patch channels"),
                                 std::uint64_t(patch_size) * patch_size, "patch width");
}

std::uint64_t VisionConfig::merger_width() const {
    return artifact::checked_mul(std::uint64_t(spatial_merge_size) * spatial_merge_size,
                                 hidden_size, "merger width");
}

VisionConfig parse_vision_config(const artifact::Json& value, std::span<const std::string_view> model_types) {
    return vision_config(value, model_types);
}

} // namespace infernix::models::qwen_common
