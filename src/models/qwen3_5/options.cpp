#include "models/qwen3_5/options.h"

#include <stdexcept>

namespace infernix::models::qwen3_5 {

void validate_headroom(const EngineOptions& options) {
    if (options.vram_headroom_bytes.has_value()) {
        throw std::invalid_argument(
            "Qwen3.5 applies --vram-headroom-mib only with --kv-capacity auto");
    }
}

void validate_expert_tail_skip(const EngineOptions& options) {
    if (options.expert_tail_skip != 0) {
        throw std::invalid_argument(
            "--expert-tail-skip applies to Qwen3.8-Flash-Next (its offloaded experts) only");
    }
}

void validate_draft_backend(const EngineOptions& options) {
    if (options.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument(
            "--draft-tokens auto requires --spec mtp (Qwen3.8-Flash-Next: also dflash2)");
    }
}

void validate_draft_min_p(const EngineOptions& options) {
    (void)options;
    throw std::invalid_argument(
        "--spec-min-p applies to Qwen3.8-Flash-Next with --spec mtp or dflash2 only");
}

void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options, std::uint32_t concurrency,
                   bool host_tier) {
    (void)options;
    cache.hybrid.device_snapshot_slots =
        cache.hybrid.device_snapshot_slots.value_or(concurrency + (host_tier ? 1U : 2U));
    cache.hybrid.tool_call_hold = cache.hybrid.tool_call_hold.value_or(kDefaultToolCallHold);
    // The shared range check keeps [1,64]; an explicitly disabled slot count is refused here.
    if (cache.hybrid.device_snapshot_slots.value_or(0U) == 0U) {
        throw std::invalid_argument("hybrid device snapshot slots must be in [1,64]");
    }
}

} // namespace infernix::models::qwen3_5
