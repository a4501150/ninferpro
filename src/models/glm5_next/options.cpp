#include "models/glm5_next/options.h"

#include <stdexcept>

namespace infernix::models::glm5_next {

void validate_draft_backend(const EngineOptions&) {
    throw std::invalid_argument(
        "GLM-5.3-Flash drafts a fixed length: --spec dflash2 --draft-tokens N (1-7)");
}

void validate_draft_min_p(const EngineOptions&) {
    throw std::invalid_argument("GLM-5.3-Flash verifies every draft: --spec-min-p does not apply");
}

void resolve_cache(ContextCacheOptions&, const EngineOptions&, std::uint32_t, bool) {
    throw std::invalid_argument("GLM-5.3-Flash has no prefix cache yet: serve it with --no-prefix-reuse");
}

const runtime::FamilyOptionRules& option_rules() {
    static constexpr runtime::FamilyOptionRules rules{
        .prefill_chunk               = kPrefillChunk,
        .prefill_chunk_beside_decode = kPrefillChunkBesideDecode,
        .draft_tokens_auto           = kDraftTokensAuto,
        .validate_draft_backend      = validate_draft_backend,
        .validate_draft_min_p        = validate_draft_min_p,
        .resolve_cache               = resolve_cache,
    };
    return rules;
}

} // namespace infernix::models::glm5_next
