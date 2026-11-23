#pragma once

// The shared, artifact-free Engine option checks and defaults. Every family-dependent branch
// (the prefill chunk, the speculative policies, the prefix-cache tiers) lives with its family,
// which supplies FamilyOptionRules (models/<family>/options.h); the checks call its hooks at fixed
// points.

#include "infernix/types.h"

#include <cstdint>
#include <string>

namespace infernix::runtime {

// A model family's option defaults and refusals. Startup only; a null hook accepts.
struct FamilyOptionRules {
    std::uint32_t prefill_chunk = 0;
    // The chunk of a prompt prefilled while another request decodes; 0: prefill_chunk.
    std::uint32_t prefill_chunk_beside_decode = 0;
    // --draft-tokens auto's ceiling when none is given.
    std::uint32_t draft_tokens_auto = 0;
    // An explicit KV capacity.
    void (*validate_explicit_kv_capacity)(const EngineOptions&) = nullptr;
    void (*validate_logits_dtype)(const EngineOptions&)         = nullptr;
    void (*validate_experts)(const EngineOptions&)              = nullptr;
    void (*validate_expert_tail_skip)(const EngineOptions&)     = nullptr;
    // --draft-tokens auto.
    void (*validate_draft_backend)(const EngineOptions&) = nullptr;
    // A draft-confidence gate in [0,1].
    void (*validate_draft_min_p)(const EngineOptions&) = nullptr;
    // A prefix cache Disk tier directory.
    void (*validate_disk_tier)(const EngineOptions&) = nullptr;
    // The family's prefix-cache tier rules and defaults, with the cache enabled.
    void (*resolve_cache)(ContextCacheOptions& cache, const EngineOptions& options,
                          std::uint32_t concurrency, bool host_tier) = nullptr;
    // The chunk spacing the tap ladder keeps; null: the prefill chunk.
    std::uint32_t (*tap_ladder_chunk)(std::uint32_t prefill_chunk) noexcept = nullptr;
};

// Validates the options and resolves every default; the hybrid prefix cache's Host tier, Device
// snapshot slots and tap budget depend on the model family.
EngineOptions normalize_engine_options(EngineOptions options, const FamilyOptionRules& family);

// The option checks that need no artifact (shared by every family); the Engine runs them before
// it opens the artifact.
void validate_engine_options(const EngineOptions& options, const FamilyOptionRules& family);

// Everything the bytes of a persisted hybrid Host tier depend on besides its geometry (which the
// file records itself); a different value makes a saved file meaningless.
[[nodiscard]] std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature);

} // namespace infernix::runtime
