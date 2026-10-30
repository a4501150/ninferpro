#include "runtime/engine/engine_options.h"

#include "models/qwen3_5/options.h"
#include "models/qwen4_exp/options.h"

#include <algorithm>
#include <stdexcept>
#include <system_error>

namespace infernix::runtime {

void validate_engine_options(const EngineOptions& options, models::Architecture architecture) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    // .infernix and .ninfer name the same v3 container (Infernix began as a fork of NInfer).
    if (const auto extension = options.artifact_path.extension();
        extension != ".infernix" && extension != ".ninfer") {
        throw std::invalid_argument("Infernix accepts only .infernix and .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        // Qwen3.5's headroom is what automatic KV capacity leaves; an explicit capacity has none
        // (Qwen3.8-Flash-Next's is what its expert cache leaves, with any KV capacity).
        if (architecture != models::Architecture::Qwen4Exp) {
            models::qwen3_5::validate_headroom(options);
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (const auto interval = options.context_cache.hybrid.persistent_save_interval; interval.count() != 0) {
        if (interval.count() < 0) { throw std::invalid_argument("prefix cache save interval must not be negative"); }
        if (options.context_cache.hybrid.persistent_file.empty()) {
            throw std::invalid_argument("a periodic prefix cache save needs a prefix cache file");
        }
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    // The prompt-kernel choice exists only for INT8 KV; accepting it elsewhere would select a
    // kernel that never runs.
    if (options.original_int8_prefill_kernel && options.kv_cache != KvCacheStorage::Int8Group64) {
        throw std::invalid_argument(
            "the original INT8 prefill kernel requires the INT8 KV cache (--kv-dtype int8)");
    }
    if (options.prefill_split_workspace_mib > kMaximumPrefillSplitWorkspaceMiB) {
        throw std::invalid_argument("Engine prefill_split_workspace_mib must be in [0,16384]");
    }
    if (options.original_nvfp4_prefill_kernel && options.kv_cache != KvCacheStorage::Nvfp4Group16) {
        throw std::invalid_argument(
            "the original NVFP4 prefill kernel requires the NVFP4 KV cache (--kv-dtype nvfp4)");
    }
    // 16-bit activations are the reference the fast paths are measured against.
    if (options.a16_activations && (options.fast_paths || !options.fast_path_candidates.empty())) {
        throw std::invalid_argument(
            "16-bit activations (the quality reference) run the exact route: disable fast paths");
    }
}

// The exact artifact, its execution signature, the KV and speculative formats, RoPE scaling, the
// prefill chunk and the product binary's build identity. Any difference makes the saved state
// meaningless, so the file is ignored.
std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature) {
    std::error_code error;
    const auto size = std::filesystem::file_size(options.artifact_path, error);
    const auto time = std::filesystem::last_write_time(options.artifact_path, error);
    std::string out = "artifact=" + std::filesystem::absolute(options.artifact_path).string();
    out += ";size=" + std::to_string(error ? 0U : size);
    out += ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    out += ";signature=" + signature;
    out += ";kv=" + std::to_string(static_cast<int>(options.kv_cache));
    out += ";speculative=" + std::to_string(static_cast<int>(options.speculative.backend));
    out += ";yarn=" + std::to_string(options.rope_yarn_factor);
    // Qwen3.5's exact-route prefill bits follow the chunk grid, so a state prefilled at another
    // chunk would not be the one this engine computes.
    out += ";chunk=" + std::to_string(options.prefill_chunk);
    out += ";host_state=" +
           std::to_string(static_cast<int>(options.context_cache.hybrid.host_state_precision));
    // Fast paths change the numbers cached KV and state hold.
    out += ";fast=" + std::to_string(options.fast_paths ? 1 : 0);
    for (const FastPath path : options.fast_path_candidates) { out += std::string(",") + fast_path_name(path); }
    out += ";build=" + options.context_cache.hybrid.persistent_identity;
    return out;
}

EngineOptions normalize_engine_options(EngineOptions options, models::Architecture architecture) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.kv_capacity         = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache = options.context_cache;
    if (options.speculative.ngram_archive_bytes != 0 &&
        options.speculative.ngram_draft_tokens == 0) {
        throw std::invalid_argument(
            "the cross-request ngram archive requires ngram drafting; enable --ngram-draft-tokens");
    }
    if (options.speculative.ngram_archive_bytes != 0 &&
        (options.speculative.ngram_session_bytes < (1ULL << 20) ||
         options.speculative.ngram_session_bytes > options.speculative.ngram_archive_bytes)) {
        throw std::invalid_argument("ngram session capacity must be between 1 MiB and the total "
                                    "archive capacity");
    }
    const std::uint32_t concurrency = options.max_concurrency;
    const bool qwen4_exp             = architecture == models::Architecture::Qwen4Exp;

    // The model's default prefill chunk (each family's tuning notes live with its constant).
    if (options.prefill_chunk == 0) {
        options.prefill_chunk = qwen4_exp ? models::qwen4_exp::kPrefillChunk
                                           : models::qwen3_5::kPrefillChunk;
    }
    if (options.prefill_chunk_beside_decode == 0) {
        options.prefill_chunk_beside_decode =
            qwen4_exp ? models::qwen4_exp::kPrefillChunkBesideDecode : options.prefill_chunk;
    }
    if (!qwen4_exp) { models::qwen3_5::validate_expert_tail_skip(options); }
    // An approximation whose output depends on the cache state: never on the exact route.
    if (options.expert_tail_skip != 0 && !options.fast_paths) {
        throw std::invalid_argument("--expert-tail-skip is approximate; the exact route (--disable-fast-paths) refuses it");
    }
    if (options.speculative.draft_tokens_auto) {
        if (qwen4_exp) {
            models::qwen4_exp::validate_draft_backend(options);
        } else {
            models::qwen3_5::validate_draft_backend(options);
        }
        if (options.speculative.draft_tokens == 0) {
            options.speculative.draft_tokens =
                qwen4_exp ? models::qwen4_exp::kDraftTokensAuto : models::qwen3_5::kDraftTokensAuto;
        }
    } else if (std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                         FastPath::AdaptiveMtp) != options.fast_path_candidates.end()) {
        throw std::invalid_argument("--fast-path adaptive-mtp is the policy of --draft-tokens auto; "
                                    "a fixed --draft-tokens N drafts N");
    }
    // The draft-confidence gate exists only on Qwen3.8-Flash-Next's one-row MTP rounds; another
    // model refuses it rather than ignoring a tuning value.
    if (const auto& p = options.speculative.draft_min_p) {
        if (!(*p >= 0.0F && *p <= 1.0F)) {
            throw std::invalid_argument("speculative draft_min_p (--spec-min-p) must be in [0,1]");
        }
        if (qwen4_exp) {
            models::qwen4_exp::validate_draft_min_p(options);
        } else {
            models::qwen3_5::validate_draft_min_p(options);
        }
    }
    if (cache.enabled) {
        // One pinned Host slab pool serves blocks and snapshots alike; its size is the only
        // capacity a deployment has to choose (docs/maintainer/hybrid-prefix-cache-spec.md §5.4).
        cache.host_capacity_bytes = cache.host_capacity_bytes.value_or(kDefaultHybridHostCacheBytes);
        const bool host_tier             = *cache.host_capacity_bytes != 0;
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        // Whether a turn resumes from lossy Host state depends on what eviction left, so the exact
        // route's output would depend on load (model-quality 6.1).
        if (!options.fast_paths && hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
            throw std::invalid_argument(
                "--cache-host-state fp24 and bf16 are lossy; --disable-fast-paths requires fp32");
        }
        if (!hybrid.disk_directory.empty()) {
#ifdef _WIN32
            throw std::invalid_argument("the prefix cache Disk tier (--prefix-cache-disk) needs cuFile, which has no "
                                        "Windows build");
#endif
            if (!qwen4_exp) {
                throw std::invalid_argument(
                    "the prefix cache Disk tier (--prefix-cache-disk) applies to Qwen3.8-Flash-Next only");
            }
        } else if (hybrid.disk_capacity_bytes.value_or(0U) != 0U) {
            throw std::invalid_argument("--prefix-cache-disk-gib needs --prefix-cache-disk");
        } else {
            hybrid.disk_capacity_bytes = 0U;
        }
        // The family's own tier rules and defaults.
        if (qwen4_exp) {
            models::qwen4_exp::resolve_cache(cache, options, concurrency, host_tier);
        } else {
            models::qwen3_5::resolve_cache(cache, options, concurrency, host_tier);
        }
        // Taps without a Host tier would evict other conversations' resident snapshots.
        hybrid.max_new_taps = hybrid.max_new_taps.value_or(host_tier ? 8U : 2U);
        // Ladder taps are realized on prefill chunk boundaries, so the ladder never refines below
        // the chunk: coarser ladders only waste snapshots on taps that share one boundary.
        const std::uint32_t chunk = std::max<std::uint32_t>(
            qwen4_exp ? models::qwen4_exp::tap_ladder_chunk(options.prefill_chunk)
                      : options.prefill_chunk,
            64U);
        hybrid.tap_ladder_tokens =
            hybrid.tap_ladder_tokens.value_or(std::max<std::uint32_t>(4096U, 2U * chunk));
        hybrid.tap_min_gap_tokens =
            hybrid.tap_min_gap_tokens.value_or(std::max<std::uint32_t>(1024U, chunk));
        if (*hybrid.device_snapshot_slots > 64) {
            throw std::invalid_argument("hybrid device snapshot slots must be in [1,64]");
        }
        if (hybrid.tool_call_hold->count() < 0) {
            throw std::invalid_argument("hybrid tool-call hold must not be negative");
        }
        if (*hybrid.max_new_taps > 64) {
            throw std::invalid_argument("hybrid taps per request must be at most 64");
        }
        if (*hybrid.tap_ladder_tokens < 64 || *hybrid.tap_min_gap_tokens < 64) {
            throw std::invalid_argument(
                "hybrid tap ladder and minimum gap must be at least 64 tokens");
        }
    } else {
        // Without a prefix cache every request starts from the root and nothing is retained:
        // the lanes' StateImages are the only Device state and there is no Host tier.
        if (cache.host_capacity_bytes.value_or(0U) != 0 ||
            cache.hybrid.device_snapshot_slots.value_or(0U) != 0 || !cache.hybrid.disk_directory.empty() ||
            cache.hybrid.disk_capacity_bytes.value_or(0U) != 0) {
            throw std::invalid_argument(
                "prefix cache capacities require the context cache to be enabled");
        }
        if (cache.hybrid.tool_call_hold.value_or(std::chrono::milliseconds{0}).count() != 0 ||
            cache.hybrid.host_state_precision != HybridHostStatePrecision::Fp32) {
            throw std::invalid_argument("prefix cache policies require the context cache to be enabled");
        }
        cache.host_capacity_bytes          = 0U;
        cache.hybrid.disk_capacity_bytes   = 0U;
        cache.hybrid.device_snapshot_slots = 0U;
        cache.hybrid.tool_call_hold        = std::chrono::milliseconds{0};
    }
    return options;
}

} // namespace infernix::runtime
