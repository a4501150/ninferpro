#include "infernix/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/power_throttling.h"
#include "core/startup.h"
#include "models/registry.h"
#include "runtime/contract/request.h"
#include "runtime/contract/sampling.h"
#include "runtime/diagnostics.h"
#include "runtime/engine/engine_backend.h"
#include "runtime/engine/engine_options.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    // Every kernel is built for sm_120a alone: another GPU is refused by name here, before the
    // artifact is read, rather than at its first kernel launch after a long load (port of upstream
    // ea43e053, candidates IH-01).
    if (device.compute_capability() != 120) {
        throw std::runtime_error("Infernix needs an NVIDIA GPU of compute capability 12.0 (sm_120a: GeForce RTX 50 "
                                 "series, RTX PRO Blackwell); device " + std::to_string(options.device) + ", " +
                                 device.props.name + ", is compute capability " +
                                 std::to_string(device.props.major) + "." + std::to_string(device.props.minor));
    }
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    const std::vector<TokenId>& readout = options.execution.readout_tokens;
    if (readout.size() > kMaximumReadoutTokens) {
        throw std::invalid_argument("readout names more than kMaximumReadoutTokens tokens");
    }
    if (!readout.empty() && options.execution.requested_output_tokens == 0) {
        // The readout is taken where the first output token is chosen.
        throw std::invalid_argument("readout requires at least one requested output token");
    }
    std::vector<TokenId> distinct = readout;
    std::ranges::sort(distinct);
    if (std::ranges::adjacent_find(distinct) != distinct.end()) {
        throw std::invalid_argument("readout tokens must be distinct");
    }
    const TokenConstraint& constraint = options.execution.constraint;
    if (!constraint.empty()) {
        if (constraint.steps.size() > kMaximumConstraintSteps) {
            throw std::invalid_argument("constraint has more than kMaximumConstraintSteps steps");
        }
        if (options.execution.thinking.budget) {
            // The budget's injected close would occupy steps the constraint owns.
            throw std::invalid_argument("constraint cannot be combined with a thinking budget");
        }
        if (options.execution.thinking.loop == ReasoningLoopAction::Conclude) {
            // So would the loop guard's.
            throw std::invalid_argument("constraint cannot be combined with reasoning-loop conclude");
        }
        if (options.execution.requested_output_tokens > constraint.steps.size()) {
            throw std::invalid_argument("constraint has fewer steps than requested output tokens");
        }
        std::vector<std::vector<TokenId>> sets;
        for (const std::vector<TokenId>& step : constraint.steps) {
            if (step.empty() || step.size() > kMaximumConstraintChoices) {
                throw std::invalid_argument(
                    "constraint step must permit 1 to kMaximumConstraintChoices tokens");
            }
            std::vector<TokenId> sorted = step;
            std::ranges::sort(sorted);
            if (std::ranges::adjacent_find(sorted) != sorted.end()) {
                throw std::invalid_argument("constraint step tokens must be distinct");
            }
            if (step.size() > 1 && std::ranges::find(sets, step) == sets.end()) {
                sets.push_back(step);
            }
        }
        if (sets.size() > kMaximumConstraintSets) {
            throw std::invalid_argument(
                "constraint uses more than kMaximumConstraintSets distinct multi-token steps");
        }
        if (!constraint.observe.empty() && constraint.observe.size() != constraint.steps.size()) {
            throw std::invalid_argument("constraint observe needs one list per step");
        }
        for (const std::vector<TokenId>& observed : constraint.observe) {
            if (observed.size() > kMaximumObservedTokens) {
                throw std::invalid_argument(
                    "constraint step observes more than kMaximumObservedTokens tokens");
            }
            std::vector<TokenId> sorted = observed;
            std::ranges::sort(sorted);
            if (std::ranges::adjacent_find(sorted) != sorted.end()) {
                throw std::invalid_argument("constraint observed tokens must be distinct");
            }
        }
        // A token constraint chooses every output step itself, the end of the turn included.
        options.execution.thinking.end_turn_in_thinking = true;
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling, options.constraint.has_value());
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.execution.readout_tokens          = std::move(options.execution.readout_tokens);
    resolved.execution.constraint              = std::move(options.execution.constraint);
    resolved.stop                            = std::move(options.stop);
    resolved.output                            = options.output;
    resolved.constraint                        = options.constraint;
    resolved.tool_choice                       = std::move(options.tool_choice);
    resolved.ngram_session                     = std::move(options.ngram_session);
    return resolved;
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(runtime::BackendPrepared prepared, SamplingMode mode)
        : summary(std::move(prepared.summary)), prepare(std::move(prepared.preparation)),
          sampling_mode(mode), value(std::move(prepared.prompt)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    std::unique_ptr<runtime::BackendPrompt> value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    explicit Impl(EngineOptions engine_options)
        : options(normalize(std::move(engine_options))), device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        runtime::ConstructedBackend constructed =
            models::construct_engine_backend(architecture, options, device);
        backend             = std::move(constructed.backend);
        load                = std::move(constructed.load);
        model_metadata      = std::move(constructed.model_metadata);
        load.cuda_sync_mode = device.sync_mode();
        sampling_defaults   = backend->sampling_defaults();
    }

    // Options are checked before the artifact is opened; defaults that depend on the model (the
    // hybrid prefix cache's) need the artifact's family, read once here. An unreadable artifact
    // is reported by the default family's model construction, after option validation.
    EngineOptions normalize(EngineOptions engine_options) {
        try {
            architecture = models::detect_artifact_architecture(engine_options.artifact_path);
        } catch (const std::exception&) {}
        const runtime::FamilyOptionRules& family = models::family_option_rules(architecture);
        runtime::validate_engine_options(engine_options, family);
        return runtime::normalize_engine_options(std::move(engine_options), family);
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        const bool persists = !backend->prefix_cache_save_notice(options).empty();
        stop();
        // Joins the generation core's orderly stop, which saves the prefix cache before dropping
        // it.
        backend->release_core();
        try {
            device.synchronize();
        } catch (...) {}
        if (persists) { report_prefix_cache_save(); }
    }

    void stop() noexcept {
        if (stop_requested.exchange(true)) { return; }
        try {
            if (const std::string notice = backend->prefix_cache_save_notice(options);
                !notice.empty()) {
                runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                            "%s", notice.c_str());
            }
        } catch (...) {}
        backend->stop();
    }

    void report_prefix_cache_save() const noexcept {
        try {
            const std::optional<runtime::PrefixCacheSave> result =
                backend->prefix_cache_shutdown_save();
            if (!result) {
                runtime::publish_diagnostic(
                    options.diagnostic_observer, DiagnosticLevel::Warning,
                    "prefix cache not saved: the Engine did not stop cleanly");
                return;
            }
            const runtime::PrefixCacheSave& saved = *result;
            if (saved.ok) {
                runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                            "prefix cache saved: %llu blocks, %llu snapshots, "
                                            "%.1f MiB in %.1f s",
                                            static_cast<unsigned long long>(saved.blocks),
                                            static_cast<unsigned long long>(saved.snapshots),
                                            static_cast<double>(saved.bytes) / 1048576.0,
                                            saved.seconds);
            } else {
                runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                            "prefix cache not saved: %s", saved.message.c_str());
            }
        } catch (const std::exception& error) {
            runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                        "prefix cache not saved: %s", error.what());
        } catch (...) {}
    }

    models::Architecture architecture = models::kDefaultArchitecture; // set by normalize
    EngineOptions options;
    DeviceContext device;
    LoadSummary load;
    ModelMetadata model_metadata;
    ModelSamplingDefaults sampling_defaults;
    std::unique_ptr<runtime::EngineBackend> backend;
    std::atomic<bool> stop_requested{false};
};

Engine::Engine(EngineOptions options) {
    // Before any host thread starts: a throttled process runs every one of them slower.
    exempt_process_from_power_throttling();
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    runtime::BackendPrepared prepared = impl_->backend->prepare(std::move(input), control);
    const SamplingMode sampling_mode =
        prepared.summary.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (prepared.summary.prompt_tokens > impl_->backend->capacity()) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    return PreparedPrompt(
        std::make_unique<PreparedPrompt::Impl>(std::move(prepared), sampling_mode));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const std::uint32_t capacity = impl_->backend->capacity();
    if (token_ids.size() > capacity) {
        throw RequestError(
            ContextLengthDetails{.prompt_tokens = token_ids.size(), .max_context = capacity});
    }
    runtime::BackendPrepared prepared =
        impl_->backend->prepare_tokens(std::move(token_ids), allow_prefix_identity);
    if (prepared.summary.prompt_tokens > capacity) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    return PreparedPrompt(
        std::make_unique<PreparedPrompt::Impl>(std::move(prepared), SamplingMode::Thinking));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->tokenize_text(text);
}

ScoreResult Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target,
                                 ScoreOptions options) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    const std::size_t positions = tokens.size() - first_target;
    if (options.top_k > kMaximumScoreTopTokens ||
        options.candidates_per_position > kMaximumScoreTopTokens) {
        throw std::invalid_argument("score_tokens top_k and candidates_per_position must be in "
                                    "[0,64]");
    }
    if (options.candidates.size() != positions * options.candidates_per_position) {
        throw std::invalid_argument(
            "score_tokens needs candidates_per_position candidates for every scored position");
    }
    const std::size_t top_values       = positions * options.top_k;
    const std::size_t candidate_values = options.candidates.size();
    PreparedPrompt prompt              = prepare_tokens(std::move(tokens), false);
    ScoreResult result =
        impl_->backend->score(std::move(prompt.impl_->value), first_target, std::move(options));
    if (result.logprobs.size() != positions || result.top_ids.size() != top_values ||
        result.top_logprobs.size() != top_values ||
        result.candidate_logprobs.size() != candidate_values) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;
    const auto in_domain = [&](TokenId id) {
        return id >= 0 && static_cast<std::uint64_t>(id) < impl_->model_metadata.vocab_size;
    };
    for (const TokenId id : resolved_options.execution.readout_tokens) {
        if (!in_domain(id)) {
            throw std::invalid_argument("readout token is outside the tokenizer domain");
        }
    }
    for (const auto& step : resolved_options.execution.constraint.steps) {
        if (!std::ranges::all_of(step, in_domain)) {
            throw std::invalid_argument("constraint token is outside the tokenizer domain");
        }
    }

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(ContextLengthDetails{.prompt_tokens = prompt_summary.prompt_tokens,
                                                .max_context   = impl_->options.max_context});
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    struct BackendSubmission {
        std::unique_ptr<runtime::PendingGeneration> pending;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            return pending->wait(sink, cancellation);
        }
    } submission{impl_->backend->submit(std::move(prompt.impl_->value), prompt_summary,
                                        prepare_seconds, std::move(resolved_options), consumer_mode,
                                        std::move(observation), pending_deadline)};

    return GenerationHandle(
        std::make_unique<GenerationHandle::Impl>(impl_, std::move(submission), resolved_sampling));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

ModelMetadata Engine::model_metadata() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->model_metadata;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->memory_summary();
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->media_cache_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->runtime_stats();
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return impl_->backend->is_available();
}

ExpertPlacementReport Engine::expert_placement_status() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->expert_placement_status();
}

bool Engine::request_expert_placement(const ExpertPlacementSwitch& request, std::string* error) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->request_expert_placement(request, error);
}

bool Engine::export_expert_placement(const std::filesystem::path& path, std::string* error) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->export_expert_placement(path, error);
}

ExpertPlacementPreview Engine::preview_expert_placement(const std::string& experts_spec) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->backend->preview_expert_placement(experts_spec);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    impl_->backend->reset_memory_peaks();
}

void Engine::stop() noexcept {
    if (impl_ != nullptr) { impl_->stop(); }
}

} // namespace infernix
