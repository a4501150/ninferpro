#pragma once

// The Glm5Next (GLM-5.3-Flash) family's option defaults and refusals: its FamilyOptionRules,
// which the shared runtime::validate_engine_options / runtime::normalize_engine_options
// (src/runtime/engine/engine_options.cpp) call at fixed points.

#include "infernix/types.h"
#include "runtime/engine/engine_options.h"

namespace infernix::models::glm5_next {

// The model's default prefill chunk, and its chunk while another request decodes (the engine runs
// one decode round per prefill call). Both are unmeasured starting points (design §17); the M7
// sweep sets them.
constexpr std::uint32_t kPrefillChunk             = 4096U;
constexpr std::uint32_t kPrefillChunkBesideDecode = 1024U;

// The DFlash2 drafter's block less its anchor (FamilyOptionRules::draft_tokens_auto; unused while
// --draft-tokens auto is refused).
constexpr std::uint32_t kDraftTokensAuto = 7U;

// Every round drafts --draft-tokens: --draft-tokens auto is refused.
void validate_draft_backend(const EngineOptions& options);

// Every draft is verified: --spec-min-p is refused.
void validate_draft_min_p(const EngineOptions& options);

// It has no prefix cache yet: an enabled cache is refused (the shared normalize calls this only
// with the cache enabled).
void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options, std::uint32_t concurrency,
                   bool host_tier);

[[nodiscard]] const runtime::FamilyOptionRules& option_rules();

} // namespace infernix::models::glm5_next
