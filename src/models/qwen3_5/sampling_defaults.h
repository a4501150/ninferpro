#pragma once

#include "infernix/types.h"
#include "models/registry.h"

namespace infernix::models::qwen3_5 {

// The family's recommended sampling presets; the MoE checkpoint adds a thinking presence penalty.
[[nodiscard]] inline ModelSamplingDefaults default_sampling(Architecture architecture) {
    ModelSamplingDefaults sampling;
    sampling.thinking = {.temperature      = 1.0F,
                         .top_k            = 20,
                         .top_p            = 0.95F,
                         .min_p            = 0.0F,
                         .presence_penalty = architecture == Architecture::Qwen3_5Moe ? 1.5F : 0.0F,
                         .frequency_penalty = 0.0F};
    sampling.non_thinking = {.temperature       = 0.7F,
                             .top_k             = 20,
                             .top_p             = 0.80F,
                             .min_p             = 0.0F,
                             .presence_penalty  = 1.5F,
                             .frequency_penalty = 0.0F};
    return sampling;
}

} // namespace infernix::models::qwen3_5
