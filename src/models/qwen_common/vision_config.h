#pragma once

#include "artifact/schema.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace infernix::models::qwen_common {

// The Vision tower's config (a Qwen3.5-family tower), accepting the listed model_type values; also
// read by Qwen4Exp, whose tower differs only in its output width (bound on the merger).
struct VisionConfig {
    std::uint32_t depth                   = 0;
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_heads               = 0;
    std::uint32_t patch_size              = 0;
    std::uint32_t temporal_patch_size     = 0;
    std::uint32_t spatial_merge_size      = 0;
    std::uint32_t num_position_embeddings = 0;
    std::uint32_t position_grid_side      = 0;
    [[nodiscard]] std::uint64_t patch_width() const;
    [[nodiscard]] std::uint64_t merger_width() const;
};

[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& value,
                                               std::span<const std::string_view> model_types);

} // namespace infernix::models::qwen_common
