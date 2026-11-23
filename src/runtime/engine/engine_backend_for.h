#pragma once

// EngineBackend over one family's instance and the common core for the Engine's purpose. The
// instance supplies `frontend`, `capacity` and `program`, and may supply the expert placement
// (`expert_placement_status`, `request_expert_placement`, `export_expert_placement`,
// `preview_expert_placement`) and the prefix cache's persistence (`prefix_cache_save_notice`,
// `prefix_cache_shutdown_save`); a missing one reports the model has none.

#include "core/device.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_backend.h"
#include "runtime/engine/engine_core.h"

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace infernix::runtime {

template <class Instance, template <class> class CoreFor>
class EngineBackendFor final : public EngineBackend {
    using Core                     = CoreFor<Instance>;
    using Prompt                   = typename Instance::ModelContract::PreparedPrompt;
    static constexpr bool kScoring = std::is_same_v<Core, CausalScoreCore<Instance>>;

    struct Payload final : BackendPrompt {
        explicit Payload(Prompt prepared) : value(std::move(prepared)) {}

        Prompt value;
    };

    static BackendPrepared wrap(Prompt prepared) {
        BackendPrepared out{.summary     = prepared.summary(),
                            .preparation = prepared.preparation_stats()};
        out.prompt = std::make_unique<Payload>(std::move(prepared));
        return out;
    }

    static Prompt unwrap(std::unique_ptr<BackendPrompt> prompt) {
        return std::move(static_cast<Payload&>(*prompt).value);
    }

    template <class Submission>
    class Pending final : public PendingGeneration {
    public:
        explicit Pending(Submission submission) : submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        Submission submission_;
    };

public:
    EngineBackendFor(std::unique_ptr<Instance> instance, DeviceContext& device,
                     const EngineOptions& options)
        : instance_(std::move(instance)) {
        if constexpr (kScoring) {
            core_ = std::make_unique<Core>(*instance_, device);
        } else {
            core_ = std::make_unique<Core>(*instance_, device, options);
        }
    }

    BackendPrepared prepare(PromptInput input, const PreparationControl& control) const override {
        return wrap(instance_->frontend.prepare(std::move(input), control));
    }

    BackendPrepared prepare_tokens(std::vector<TokenId> token_ids,
                                   bool allow_prefix_identity) const override {
        return wrap(
            instance_->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity));
    }

    std::vector<TokenId> tokenize_text(std::string_view text) const override {
        return instance_->frontend.tokenize_text(text);
    }

    std::uint32_t count_tokens(PromptInput input,
                               const PreparationControl& control) const override {
        return instance_->frontend.count_tokens(std::move(input), control);
    }

    MediaCacheSummary media_cache_summary() const override {
        return instance_->frontend.media_cache_summary();
    }

    ModelSamplingDefaults sampling_defaults() const override {
        return instance_->frontend.sampling_defaults();
    }

    std::uint32_t capacity() const override { return instance_->capacity; }

    ScoreResult score(std::unique_ptr<BackendPrompt> prompt, std::uint32_t first_target,
                      ScoreOptions options) override {
        if constexpr (kScoring) {
            return core().score(unwrap(std::move(prompt)), first_target, std::move(options));
        } else {
            throw std::logic_error("Engine scoring core is unavailable");
        }
    }

    std::unique_ptr<PendingGeneration>
    submit(std::unique_ptr<BackendPrompt> prompt, PromptSummary prompt_summary,
           double prepare_seconds, ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
           GenerationObservationOptions observation,
           std::chrono::steady_clock::time_point pending_deadline) override {
        if constexpr (kScoring) {
            throw std::logic_error("Engine generation core is unavailable");
        } else {
            auto submission = core().submit(unwrap(std::move(prompt)), std::move(prompt_summary),
                                            prepare_seconds, std::move(options), consumer_mode,
                                            std::move(observation), pending_deadline);
            return std::make_unique<Pending<decltype(submission)>>(std::move(submission));
        }
    }

    MemorySummary memory_summary() const override { return core().memory_summary(); }

    RuntimeStats runtime_stats() const override { return core().runtime_stats(); }

    bool is_available() const override { return core_ != nullptr && core_->is_available(); }

    void reset_memory_peaks() noexcept override {
        if (core_ != nullptr) { core_->reset_memory_peaks(); }
    }

    ExpertPlacementReport expert_placement_status() const override {
        return core_ != nullptr ? core_->expert_placement_status() : ExpertPlacementReport{};
    }

    bool request_expert_placement(const ExpertPlacementSwitch& request,
                                  std::string* error) override {
        if (core_ == nullptr) { return unavailable(error); }
        return core_->request_expert_placement(request, error);
    }

    bool export_expert_placement(const std::filesystem::path& path, std::string* error) override {
        if (core_ == nullptr) { return unavailable(error); }
        return core_->export_expert_placement(path, error);
    }

    ExpertPlacementPreview
    preview_expert_placement(const std::string& experts_spec) const override {
        if constexpr (requires(const ExpertPlacementReport& now) {
                          instance_->preview_expert_placement(experts_spec, now);
                      }) {
            return instance_->preview_expert_placement(experts_spec, expert_placement_status());
        } else {
            throw std::invalid_argument("this model runs without the expert cache");
        }
    }

    void stop() noexcept override {
        if (core_ != nullptr) { core_->stop(); }
    }

    void release_core() noexcept override { core_.reset(); }

    std::string prefix_cache_save_notice(const EngineOptions& options) const override {
        if constexpr (!kScoring && requires { instance_->prefix_cache_save_notice(options); }) {
            return instance_->prefix_cache_save_notice(options);
        } else {
            return {};
        }
    }

    std::optional<PrefixCacheSave> prefix_cache_shutdown_save() const override {
        if constexpr (requires { instance_->prefix_cache_shutdown_save(); }) {
            return instance_->prefix_cache_shutdown_save();
        } else {
            return std::nullopt;
        }
    }

private:
    [[nodiscard]] Core& core() const {
        if (core_ == nullptr) { throw std::logic_error("Engine core is unavailable"); }
        return *core_;
    }

    static bool unavailable(std::string* error) {
        if (error != nullptr) { *error = "Engine core is unavailable"; }
        return false;
    }

    std::unique_ptr<Instance> instance_;
    std::unique_ptr<Core> core_;
};

// The backend for the options' purpose; the caller's EngineFinalize phase covers the core's start.
template <class Instance>
[[nodiscard]] std::unique_ptr<EngineBackend> make_engine_backend(std::unique_ptr<Instance> instance,
                                                                 DeviceContext& device,
                                                                 const EngineOptions& options) {
    if (options.purpose == EnginePurpose::CausalScoring) {
        return std::make_unique<EngineBackendFor<Instance, CausalScoreCore>>(std::move(instance),
                                                                             device, options);
    }
    return std::make_unique<EngineBackendFor<Instance, EngineCore>>(std::move(instance), device,
                                                                    options);
}

} // namespace infernix::runtime
