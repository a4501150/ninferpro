#include "models/qwen4_exp/options.h"

#include <stdexcept>

namespace infernix::models::qwen4_exp {

void validate_draft_backend(const EngineOptions& options) {
    if (options.speculative.backend != SpeculativeBackend::Mtp &&
        options.speculative.backend != SpeculativeBackend::DFlash2) {
        throw std::invalid_argument(
            "--draft-tokens auto requires --spec mtp (Qwen3.8-Flash-Next: also dflash2)");
    }
}

void validate_draft_min_p(const EngineOptions& options) {
    if (options.speculative.backend != SpeculativeBackend::Mtp &&
        options.speculative.backend != SpeculativeBackend::DFlash2) {
        throw std::invalid_argument(
            "--spec-min-p applies to Qwen3.8-Flash-Next with --spec mtp or dflash2 only");
    }
}

void resolve_cache(ContextCacheOptions& cache, const EngineOptions& options, std::uint32_t concurrency,
                   bool host_tier) {
    (void)options;
    (void)concurrency;
    // The Disk tier's default capacity, only when a directory names one (the shared normalize
    // refused the family-less path and zeroed the field otherwise).
    if (!cache.hybrid.disk_directory.empty()) {
        cache.hybrid.disk_capacity_bytes =
            cache.hybrid.disk_capacity_bytes.value_or(kDefaultPrefixDiskCacheBytes);
        if (*cache.hybrid.disk_capacity_bytes == 0U) {
            throw std::invalid_argument("--prefix-cache-disk-gib must be positive");
        }
    }
    // Snapshots are born in the Host tier, or without one written through to the Disk tier.
    if (!host_tier && cache.hybrid.disk_directory.empty()) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next's prefix cache keeps its snapshots in the Host or Disk tier: "
            "--host-context-mib must be positive without --prefix-cache-disk");
    }
    // It keeps its prefix cache across restarts in the Disk tier, not a save file.
    if (!cache.hybrid.persistent_file.empty() || cache.hybrid.persistent_save_interval.count() != 0) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next keeps its prefix cache across restarts in the Disk tier: use "
            "--prefix-cache-disk DIR and --prefix-cache-disk-gib N instead of --prefix-cache-file and "
            "--prefix-cache-save-mins");
    }
    if (cache.hybrid.device_snapshot_slots.value_or(0U) != 0U) {
        throw std::invalid_argument("Qwen3.8-Flash-Next's prefix cache has no Device snapshot slots");
    }
    cache.hybrid.device_snapshot_slots = 0U;
    if (cache.hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next's prefix cache keeps snapshot state exact on the Host tier");
    }
    // Its endpoints publish asynchronously, after the lane that captured them may be reused;
    // holding them needs a capture-to-snapshot handover it does not have.
    if (cache.hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next's prefix cache has no tool-call holds");
    }
    cache.hybrid.tool_call_hold = std::chrono::milliseconds{0};
}

std::uint32_t tap_ladder_chunk(std::uint32_t prefill_chunk) noexcept {
    return prefill_chunk < 4096U ? prefill_chunk : 4096U;
}

} // namespace infernix::models::qwen4_exp
