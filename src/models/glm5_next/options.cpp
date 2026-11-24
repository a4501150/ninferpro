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

void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options,
                   std::uint32_t concurrency, bool host_tier) {
    (void)options;
    (void)concurrency;
    // Snapshots are born in the Host tier: there is no Disk tier to write them through to.
    if (!cache.hybrid.disk_directory.empty() ||
        cache.hybrid.disk_capacity_bytes.value_or(0U) != 0U) {
        throw std::invalid_argument(
            "GLM-5.3-Flash's prefix cache keeps its snapshots in the Host tier only: drop "
            "--prefix-cache-disk and --prefix-cache-disk-gib");
    }
    if (!host_tier) {
        throw std::invalid_argument(
            "GLM-5.3-Flash's prefix cache keeps its snapshots in the Host tier: --host-context-mib "
            "must be positive");
    }
    // It keeps no cache across restarts.
    if (!cache.hybrid.persistent_file.empty() ||
        cache.hybrid.persistent_save_interval.count() != 0) {
        throw std::invalid_argument(
            "GLM-5.3-Flash's prefix cache has no persistence: drop --prefix-cache-file and "
            "--prefix-cache-save-mins");
    }
    if (cache.hybrid.device_snapshot_slots.value_or(0U) != 0U) {
        throw std::invalid_argument("GLM-5.3-Flash's prefix cache has no Device snapshot slots");
    }
    cache.hybrid.device_snapshot_slots = 0U;
    if (cache.hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
        throw std::invalid_argument(
            "GLM-5.3-Flash's prefix cache keeps snapshot state exact on the Host tier");
    }
    // Its captures publish asynchronously, after the lane that took them may be reused; holding
    // them needs a capture-to-snapshot handover it does not have.
    if (cache.hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0) {
        throw std::invalid_argument("GLM-5.3-Flash's prefix cache has no tool-call holds");
    }
    cache.hybrid.tool_call_hold = std::chrono::milliseconds{0};
}

const runtime::FamilyOptionRules& option_rules() {
    static const runtime::FamilyOptionRules rules{
        .prefill_chunk               = kPrefillChunk,
        .prefill_chunk_beside_decode = kPrefillChunkBesideDecode,
        .draft_tokens_auto           = kDraftTokensAuto,
        .host_cache_default          = kDefaultPrefixHostBytes,
        .validate_draft_backend      = validate_draft_backend,
        .validate_draft_min_p        = validate_draft_min_p,
        .resolve_cache               = resolve_cache,
    };
    return rules;
}

} // namespace infernix::models::glm5_next
