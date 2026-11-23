#pragma once

#include "infernix/types.h"

namespace infernix::models::qwen4_exp {

// The family's recommended sampling presets.
[[nodiscard]] inline ModelSamplingDefaults default_sampling() {
    ModelSamplingDefaults sampling;
    sampling.thinking     = {.temperature       = 1.0F,
                             .top_k             = 20,
                             .top_p             = 0.95F,
                             .min_p             = 0.0F,
                             .presence_penalty  = 0.0F,
                             .frequency_penalty = 0.0F};
    sampling.non_thinking = {.temperature       = 0.7F,
                             .top_k             = 20,
                             .top_p             = 0.80F,
                             .min_p             = 0.0F,
                             .presence_penalty  = 1.5F,
                             .frequency_penalty = 0.0F};
    return sampling;
}

} // namespace infernix::models::qwen4_exp
