#pragma once

// The public Engine's one handle on the loaded model: a family's frontend, Program and common
// core (EngineCore or CausalScoreCore) behind one virtual interface. Each family's instance.cpp
// instantiates EngineBackendFor (engine_backend_for.h) over its own instance; the Engine makes one
// virtual call per public API call, never per token.

#include "infernix/types.h"
#include "runtime/contract/request.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::runtime {

// A prepared prompt in the form its backend consumes; only the backend that made it reads it.
class BackendPrompt {
public:
    virtual ~BackendPrompt() = default;
};

struct BackendPrepared {
    std::unique_ptr<BackendPrompt> prompt;
    PromptSummary summary;
    PromptPreparationStats preparation;
};

// A submitted generation request until its result is consumed.
class PendingGeneration {
public:
    virtual ~PendingGeneration()                                                          = default;
    virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
};

// What a family's shutdown save of its prefix cache wrote (a file, or the Disk tier).
struct PrefixCacheSave {
    bool ok = false;
    std::string message;
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
};

class EngineBackend {
public:
    virtual ~EngineBackend() = default;

    [[nodiscard]] virtual BackendPrepared prepare(PromptInput input,
                                                  const PreparationControl& control) const    = 0;
    [[nodiscard]] virtual BackendPrepared prepare_tokens(std::vector<TokenId> token_ids,
                                                         bool allow_prefix_identity) const    = 0;
    [[nodiscard]] virtual std::vector<TokenId> tokenize_text(std::string_view text) const     = 0;
    [[nodiscard]] virtual std::uint32_t count_tokens(PromptInput input,
                                                     const PreparationControl& control) const = 0;
    [[nodiscard]] virtual MediaCacheSummary media_cache_summary() const                       = 0;
    [[nodiscard]] virtual ModelSamplingDefaults sampling_defaults() const                     = 0;
    [[nodiscard]] virtual std::uint32_t capacity() const                                      = 0;

    // A CausalScoring backend scores and refuses submit; a Generation backend the reverse.
    [[nodiscard]] virtual ScoreResult score(std::unique_ptr<BackendPrompt> prompt,
                                            std::uint32_t first_target, ScoreOptions options) = 0;
    [[nodiscard]] virtual std::unique_ptr<PendingGeneration>
    submit(std::unique_ptr<BackendPrompt> prompt, PromptSummary prompt_summary,
           double prepare_seconds, ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
           GenerationObservationOptions observation,
           std::chrono::steady_clock::time_point pending_deadline) = 0;

    [[nodiscard]] virtual MemorySummary memory_summary() const = 0;
    [[nodiscard]] virtual RuntimeStats runtime_stats() const   = 0;
    [[nodiscard]] virtual bool is_available() const            = 0;
    virtual void reset_memory_peaks() noexcept                 = 0;

    [[nodiscard]] virtual ExpertPlacementReport expert_placement_status() const = 0;
    [[nodiscard]] virtual bool request_expert_placement(const ExpertPlacementSwitch& request,
                                                        std::string* error)     = 0;
    [[nodiscard]] virtual bool export_expert_placement(const std::filesystem::path& path,
                                                       std::string* error)      = 0;
    // Throws std::invalid_argument for a spec the live placement cannot take, or a model without
    // an expert cache.
    [[nodiscard]] virtual ExpertPlacementPreview
    preview_expert_placement(const std::string& experts_spec) const = 0;

    // Refuses new work; queued and active requests end within one unit of work.
    virtual void stop() noexcept = 0;
    // Joins the core's orderly stop (a Generation core saves its prefix cache there) and drops
    // it; the model stays for the save report.
    virtual void release_core() noexcept = 0;

    // The notice published when the stop begins, empty when these options persist no prefix
    // cache; and after release_core, what the save wrote (nullopt: the core did not stop cleanly).
    [[nodiscard]] virtual std::string
    prefix_cache_save_notice(const EngineOptions& options) const                            = 0;
    [[nodiscard]] virtual std::optional<PrefixCacheSave> prefix_cache_shutdown_save() const = 0;
};

struct ConstructedBackend {
    std::unique_ptr<EngineBackend> backend;
    LoadSummary load;
    ModelMetadata model_metadata;
};

} // namespace infernix::runtime
