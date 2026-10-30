#pragma once

// The shared, artifact-free Engine option checks and defaults. Every family-dependent branch
// (the prefill chunk, the speculative policies, the prefix-cache tiers) lives with its family:
// models/qwen3_5/options.h and models/qwen4_exp/options.h, called here at their original points.

#include "infernix/types.h"
#include "models/registry.h"

#include <string>

namespace infernix::runtime {

// Validates the options and resolves every default; the hybrid prefix cache's Host tier, Device
// snapshot slots and tap budget depend on the model architecture.
EngineOptions normalize_engine_options(EngineOptions options,
                                        models::Architecture architecture = models::Architecture::Qwen3_5);

// The option checks that need no artifact (shared by every architecture); the Engine runs them
// before it opens the artifact.
void validate_engine_options(const EngineOptions& options, models::Architecture architecture);

// Everything the bytes of a persisted hybrid Host tier depend on besides its geometry (which the
// file records itself); a different value makes a saved file meaningless.
[[nodiscard]] std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature);

} // namespace infernix::runtime
