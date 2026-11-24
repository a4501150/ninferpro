#pragma once

// The Glm5Next (GLM-5.3-Flash) family's option defaults and refusals: its FamilyOptionRules,
// which the shared runtime::validate_engine_options / runtime::normalize_engine_options
// (src/runtime/engine/engine_options.cpp) call at fixed points.

#include "infernix/types.h"
#include "runtime/engine/engine_options.h"

#include <cstdint>

namespace infernix::models::glm5_next {

// The model's default prefill chunk, and its chunk while another request decodes (the engine runs
// one decode round per prefill call). Both are unmeasured starting points (design §17); the M7
// sweep sets them.
constexpr std::uint32_t kPrefillChunk             = 4096U;
constexpr std::uint32_t kPrefillChunkBesideDecode = 1024U;

// --vision-offload auto: on; the expert cache lends the encode's staging (design §21.1).
constexpr bool kVisionOffloadAuto = true;

// The DFlash2 drafter's block less its anchor (FamilyOptionRules::draft_tokens_auto; unused while
// --draft-tokens auto is refused).
constexpr std::uint32_t kDraftTokensAuto = 7U;

// The prefix cache Host tier's default (glm5_3-flash-design.md §18): about 26 snapshots beside
// the expert bank, each GiB costing about 78 RAM expert slots.
inline constexpr std::uint64_t kDefaultPrefixHostBytes = 2ULL << 30;

// Every round drafts --draft-tokens: --draft-tokens auto is refused.
void validate_draft_backend(const EngineOptions& options);

// Every draft is verified: --spec-min-p is refused.
void validate_draft_min_p(const EngineOptions& options);

// The family's prefix-cache tier rules: snapshots are born in the Host tier (its default is the
// design's 2 GiB); the Disk tier, a save file, Device snapshot slots, lossy Host state and
// tool-call holds are refused (the shared normalize calls this only with the cache enabled).
void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options, std::uint32_t concurrency,
                   bool host_tier);

[[nodiscard]] const runtime::FamilyOptionRules& option_rules();

} // namespace infernix::models::glm5_next
