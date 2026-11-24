#pragma once

#include "infernix/types.h"

namespace infernix::models::glm5_next {

// The family's recommended sampling presets: the checkpoint's generation_config.json
// (temperature 1.0, top-p 0.95, no top-k) in both modes.
[[nodiscard]] inline ModelSamplingDefaults default_sampling() {
    const SamplingPreset preset{.temperature       = 1.0F,
                                .top_k             = 0,
                                .top_p             = 0.95F,
                                .min_p             = 0.0F,
                                .presence_penalty  = 0.0F,
                                .frequency_penalty = 0.0F};
    return {.thinking = preset, .non_thinking = preset};
}

} // namespace infernix::models::glm5_next
