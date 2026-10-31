#include "models/qwen3_5/instance.h"
#include "runtime/engine/engine_options.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/decode_graph.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/planning/fast_paths.h"
#include "runtime/diagnostics.h"

#include <algorithm>
#include <system_error>
#include <string>
#include <filesystem>
#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace infernix::runtime {
namespace {
using Clock = std::chrono::steady_clock;

// The extra VRAM headroom of the one startup retry after the CUDA Graphs found no device memory
// (NInfer 193675826's calibration retry, adapted): their allowance is measured at 2.2-4.1 MiB per
// executable, and 1 GiB covers the largest graph set (~274 MiB) several times.
constexpr std::size_t kGraphRetryHeadroomBytes = std::size_t{1} << 30;
} // namespace


namespace {

// The hybrid index ranks admission sources and values snapshots with the same calibrated
// prefill and Host-to-Device coefficients as the original cache's ResourceManager.
// Uncalibrated (zero) terms keep the index's generic defaults.
prefix_cache::CacheCostModel hybrid_cache_cost(const ContextMachineCostModel& model) {
    constexpr double kSecondsPerNs = 1.0e-9;
    constexpr double kQ32          = 4294967296.0;
    prefix_cache::CacheCostModel cost;
    if (model.prefill.chunk_ns != 0) {
        cost.chunk_seconds = static_cast<double>(model.prefill.chunk_ns) * kSecondsPerNs;
    }
    if (model.prefill.token_ns_q32 != 0) {
        cost.token_seconds = static_cast<double>(model.prefill.token_ns_q32) / kQ32 * kSecondsPerNs;
    }
    if (model.prefill.attention_pair_ns_q32 != 0) {
        cost.attention_pair_seconds =
            static_cast<double>(model.prefill.attention_pair_ns_q32) / kQ32 * kSecondsPerNs;
    }
    const ContextTransferCost& h2d =
        model.transfer[static_cast<std::size_t>(ContextTransferDirection::HostToDevice)];
    if (h2d.ns_per_byte_q32 != 0) {
        cost.h2d_bytes_per_second =
            1.0 / (static_cast<double>(h2d.ns_per_byte_q32) / kQ32 * kSecondsPerNs);
    }
    if (h2d.batch_ns != 0) {
        cost.transfer_batch_seconds = static_cast<double>(h2d.batch_ns) * kSecondsPerNs;
    }
    return cost;
}

} // namespace


namespace {

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

} // namespace


ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen_common::make_frontend(
          model->resources(),
          {.chat_template_path       = options.chat_template_path,
           .grammar_cache_bytes      = options.grammar_cache_bytes,
           .architecture             = model->config().text.architecture,
           .vision_enabled           = options.enable_vision,
           .max_context              = options.max_context,
           .media_cache_bytes        = options.media_cache_bytes,
           .media_live_bytes         = options.media_live_bytes,
           .media_preprocess_threads = options.media_preprocess_threads,
           .vision_max_merged_tokens = options.vision_max_merged_tokens,
           .ngram_sources_enabled    = options.speculative.ngram_draft_tokens != 0,
           .ngram_archive_enabled    = options.speculative.ngram_archive_bytes != 0})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options, models::Architecture::Qwen3_5);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options, models::Architecture::Qwen3_5));
    plan.set_host_reserve(options.ram_headroom_bytes);
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    std::optional<StartupPhaseScope> planning;
    planning.emplace(options.startup_observer, StartupPhase::TargetFinalize);
    // --draft-tokens auto where adaptive-mtp does not run: the configuration's fastest fixed length.
    if (const std::uint32_t fixed = models::qwen3_5::resolve_auto_draft_length(instance->parameters, options)) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "--draft-tokens auto: adaptive-mtp is not selected for this artifact (or fast paths "
                           "are disabled), so every round drafts %u, its fastest fixed length",
                           fixed);
    }
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    const std::size_t base_headroom = options.vram_headroom_bytes.value_or(kDefaultKvCapacityHeadroomBytes);
    std::size_t retry_headroom      = 0;
    std::uint32_t planned_prefill_chunk = 0, planned_idle_width = 0;
    std::optional<StartupPhaseScope> program;
    for (int attempt = 0;; ++attempt) {
        if (!planning) { planning.emplace(options.startup_observer, StartupPhase::TargetFinalize); }
        auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
        const std::size_t free_bytes = current_free_device_bytes();
        const std::size_t headroom   = base_headroom + retry_headroom;
        auto resolution =
            resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(), free_bytes, headroom);
        // idle-wide-prefill: plan the widest idle width (4096, 2048, then 1024 tokens, as the chunk is
        // rounded) whose larger prefill buffers and workspace keep the KV capacity: an explicit one
        // whole, an automatic one down to every lane's full context. None fitting keeps chunk-wide steps.
        {
            const std::uint64_t page_tokens = planner.capacity_curve().main_page_tokens;
            const std::uint64_t lane_page_groups =
                static_cast<std::uint64_t>(options.max_concurrency) *
                ((static_cast<std::uint64_t>(options.max_context) + page_tokens - 1U) / page_tokens);
            for (const std::uint32_t width :
                 models::qwen3_5::idle_prefill_widths(instance->parameters, options)) {
                auto wide = models::qwen3_5::make_sequence_planner(instance->parameters, device,
                                                                   options, width);
                if (const auto fitted =
                        resolve_kv_capacity_beside(options.kv_capacity, wide.capacity_curve(),
                                                   free_bytes, headroom, resolution, lane_page_groups)) {
                    planner    = std::move(wide);
                    resolution = *fitted;
                    break;
                }
            }
        }
        auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
        if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
            sequence.kv_capacity() != resolution.resolved_tokens) {
            throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
        }
        options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
        // Automatic tree widths on a target that cannot verify trees resolve to chain verification.
        options.speculative.draft_tree_auto = sequence.draft_tree_auto();
        instance->kv_capacity_resolution    = resolution;
        planned_prefill_chunk = sequence.prefill_chunk();
        planned_idle_width    = sequence.idle_prefill_width();
        planning->complete();
        planning.reset();
        program.emplace(options.startup_observer, StartupPhase::ProgramInitialize);
        try {
            instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                                device, options.startup_observer);
        } catch (const GraphMemoryExhausted& error) {
            // The CUDA Graphs needed more device memory than their allowance left (the driver's
            // per-executable cost varies): an automatic KV capacity retries once with
            // kGraphRetryHeadroomBytes more headroom, so fewer KV pages; an explicit one cannot shrink.
            if (attempt > 0 || options.kv_capacity.mode != KvCapacityMode::Automatic) { throw; }
            retry_headroom = kGraphRetryHeadroomBytes;
            program.reset();
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "%s at startup; retrying with %zu MiB more VRAM headroom (a smaller automatic KV "
                               "capacity)",
                               error.what(), kGraphRetryHeadroomBytes >> 20);
            continue;
        }
        break;
    }
    LoadSummary::PrefixCacheRestore restore;
    if (options.context_cache.enabled) {
        instance->program->set_hybrid_cost(hybrid_cache_cost(context_cost.model));
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait
        // is kept well inside its queue timeout.
        instance->program->set_hybrid_coalesce_wait_limit(
            static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0);
        const std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        if (!file.empty()) {
            const models::qwen3_5::HybridCachePersistence loaded =
                instance->program->attach_hybrid_cache_file(
                    file, hybrid_cache_fingerprint(options, signature), options.startup_observer);
            restore = LoadSummary::PrefixCacheRestore{
                .attempted           = true,
                .restored            = loaded.ok,
                .message             = loaded.message,
                .blocks              = loaded.blocks,
                .snapshots           = loaded.snapshots,
                .bytes               = loaded.bytes,
                .seconds             = loaded.seconds,
                .saved_blocks        = loaded.saved_blocks,
                .saved_snapshots     = loaded.saved_snapshots,
                .required_host_bytes = loaded.required_host_bytes,
                .host_bytes          = loaded.host_bytes,
            };
        }
    }
    device.synchronize();
    program->complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    {
        const models::qwen3_5::FastPathSelection fast = models::qwen3_5::select_fast_paths(
            *instance->model, options, instance->parameters.invariant_text.has_value());
        summary.fast_path_configuration = models::qwen3_5::describe(fast.configuration);
        for (const FastPath path : fast.paths) { summary.fast_paths.emplace_back(fast_path_name(path)); }
        summary.fast_paths_disabled = !options.fast_paths;
        summary.request_invariant   = fast.request_invariant();
        summary.request_invariance_capable = fast.invariant_capable;
        summary.prefill_chunk       = planned_prefill_chunk;
        if (fast.selected(FastPath::IdleWidePrefill)) { summary.idle_prefill_width = planned_idle_width; }
    }
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    summary.prefix_cache         = std::move(restore);

    // Model metadata for /v1/models. The architecture owns the dimension facts; the Engine adds the
    // registered identity and the artifact-measured element and payload totals. The effective
    // context ceiling is a property of this instance, so it is not part of the model facts.
    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = instance->model->config().text.vocab_size;
    metadata.embedding_size = instance->model->config().text.hidden_size;
    metadata.native_context = instance->model->config().text.max_position_embeddings;
    // Total logical elements and encoded payload bytes over the artifact's distinct weight parents.
    // Several logical weights (tied embeddings, packed projections) can share one encoded parent,
    // so count each parent once. ftype is the format covering the most weights (a single
    // llama.cpp-style name); the full format set stays in LoadSummary::weight_formats.
    std::set<const WeightParent*> counted;
    std::map<std::string, std::uint64_t> elements_by_format;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            if (part.parent == nullptr || !counted.insert(part.parent).second) { continue; }
            metadata.parameters += part.parent->geometry.elements;
            metadata.weight_bytes += part.parent->geometry.bytes;
            elements_by_format[std::string(artifact::format_name(part.parent->geometry.format))] +=
                part.parent->geometry.elements;
        }
    }
    std::uint64_t dominant_elements = 0;
    for (const auto& [name, elements] : elements_by_format) {
        if (elements > dominant_elements) {
            dominant_elements   = elements;
            metadata.weights_id = name;
        }
    }
    return {std::move(instance), std::move(summary), std::move(metadata)};
}

} // namespace infernix::runtime
