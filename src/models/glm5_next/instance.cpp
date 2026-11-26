#include "models/glm5_next/instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/glm5_next/load.h"
#include "models/glm5_next/options.h"
#include "models/glm5_next/sampling_defaults.h"
#include "models/load_options.h"
#include "models/offload/expert_placement_report.h"
#include "models/offload/expert_placement_startup.h"
#include "models/offload/expert_state.h"
#include "models/offload/memory_plan.h"
#include "models/registry.h"
#include "runtime/diagnostics.h"
#include "runtime/engine/engine_backend_for.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::runtime {
namespace {

namespace glm     = models::glm5_next;
namespace offload = models::offload;

// Planning allowance for the Program's pinned buffers (I/O staging, sampling, grammar masks).
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache (design §6).
constexpr std::uint32_t kFewFrames = 2048;

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

void report(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    runtime::publish_diagnostic(options.diagnostic_observer, level, "%s", text.c_str());
}

// The fixed device allocations, largest first in the order a user can act on them.
std::string device_contributors(std::uint64_t dense, const glm::ProgramDevicePlan& plan) {
    char text[320];
    std::snprintf(text, sizeof(text),
                  "dense weights %llu MiB, KV %llu, workspace %llu, expert staging %llu, state and "
                  "io %llu, drafter "
                  "%llu, expert tables %llu",
                  mib(dense), mib(plan.kv), mib(plan.workspace), mib(plan.staging),
                  mib(plan.state + plan.io), mib(plan.draft), mib(plan.residency));
    return text;
}

void validate(const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation && options.purpose != EnginePurpose::CausalScoring) {
        throw std::invalid_argument("GLM-5.3-Flash serves generation and causal scoring");
    }
    const auto& spec = options.speculative;
    if ((spec.backend != SpeculativeBackend::None && spec.backend != SpeculativeBackend::DFlash2) ||
        spec.ngram_draft_tokens != 0 || spec.ngram_archive_bytes != 0) {
        throw std::invalid_argument("GLM-5.3-Flash speculates with a DFlash2 drafter only (--spec "
                                    "dflash2 --draft-model FILE)");
    }
    if (spec.backend == SpeculativeBackend::DFlash2 &&
        (options.purpose != EnginePurpose::Generation || spec.draft_tokens == 0 ||
         spec.draft_tokens > 7)) {
        throw std::invalid_argument(
            "GLM-5.3-Flash DFlash2 drafts 1-7 tokens per round (--draft-tokens) while generating");
    }
    if (spec.draft_tree_auto ||
        std::any_of(spec.draft_tree_nodes.begin(), spec.draft_tree_nodes.end(),
                    [](std::uint32_t nodes) { return nodes != 0; })) {
        throw std::invalid_argument(
            "GLM-5.3-Flash verifies chains: --draft-tree-nodes is not supported");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Int8Group64) {
        throw std::invalid_argument(
            "GLM-5.3-Flash keeps its KV cache in BF16 or INT8 (--kv-dtype bf16|int8)");
    }
    if (options.a16_activations) {
        throw std::invalid_argument("GLM-5.3-Flash runs its experts W4A16 already: --a16-activations does not apply");
    }
    if (!options.fast_path_candidates.empty()) {
        throw std::invalid_argument("GLM-5.3-Flash has no fast paths to select");
    }
    if (options.expert_tail_skip != 0) {
        throw std::invalid_argument("GLM-5.3-Flash computes every routed expert: --expert-tail-skip does not apply");
    }
    if (!options.ngram_volume_path.empty()) {
        throw std::invalid_argument("GLM-5.3-Flash has no n-gram table: --ngram-volume does not apply");
    }
    if (!options.route_trace_path.empty()) {
        throw std::invalid_argument("GLM-5.3-Flash records no route traces yet: --route-trace does not apply");
    }
}

// The RAM ledger's demand before the placement resolution adjusts its expert cap.
offload::HostMemoryDemand host_demand(const EngineOptions& options, const glm::LoadPlan& plan,
                                      VramBudgetSource& vram) {
    offload::HostMemoryDemand demand;
    demand.reserve        = options.ram_headroom_bytes;
    demand.expert_banks   = plan.pinned_expert_bytes();
    demand.other_pinned   = plan.pinned_other_bytes();
    demand.later_pinned   = kProgramPinnedAllowance;
    demand.load_staging   = kLoadStagingBytes;
    demand.expert_cap     = options.expert_ram_bytes;
    demand.pin_room       = offload::pin_room(vram.query());
    demand.expert_pin_cap = options.expert_pin_bytes;
    // The prefix cache pins its whole Host tier at startup, beside the experts.
    if (options.context_cache.enabled) {
        demand.prefix_cache = options.context_cache.host_capacity_bytes.value_or(0U);
    }
    return demand;
}

} // namespace

Glm5NextInstance::Glm5NextInstance(std::unique_ptr<glm::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(glm::make_frontend(model->resources(),
                                  {.chat_template_path       = options.chat_template_path,
                                   .sampling                 = glm::default_sampling(),
                                   .vision_enabled           = options.enable_vision,
                                   .max_context              = options.max_context,
                                   .media_cache_bytes        = options.media_cache_bytes,
                                   .media_live_bytes         = options.media_live_bytes,
                                   .media_preprocess_threads = options.media_preprocess_threads,
                                   .vision_max_merged_tokens = options.vision_max_merged_tokens,
                                   .thinking_budget_message  = options.thinking_budget_message})),
      capacity(options.max_context) {}

Glm5NextInstance::~Glm5NextInstance() = default;

namespace {

struct ConstructedGlm5Next {
    std::unique_ptr<Glm5NextInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
};

ConstructedGlm5Next construct_glm5_next(const EngineOptions& options, DeviceContext& device) {
    validate(options);
    const auto start = std::chrono::steady_clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!glm::is_glm5_next(reader)) { throw std::invalid_argument("artifact is not Glm5NextForCausalLM"); }
    inspect.complete();
    models::LoadOptions load = models::load_options(options, glm::kVisionOffloadAuto);
    auto plan                = glm::plan_load(reader, load);
    auto vram                = open_vram_budget_source(device.device, options.vram_past_budget);
    offload::HostMemoryDemand demand = host_demand(options, plan, *vram);

    glm::ProgramOptions program_options;
    program_options.max_context                 = options.max_context;
    program_options.max_concurrency             = options.max_concurrency;
    program_options.causal_scoring              = options.purpose == EnginePurpose::CausalScoring;
    program_options.prefill_chunk               = options.prefill_chunk;
    program_options.prefill_chunk_beside_decode = options.prefill_chunk_beside_decode;
    program_options.prefill_stream              = options.prefill_expert_stream;
    program_options.bf16_logits                 = options.logits_dtype == LogitsDtype::BF16;
    program_options.kv_cache                    = options.kv_cache;
    program_options.decode_graphs               = options.use_cuda_graph;
    program_options.draft_tokens = options.speculative.backend == SpeculativeBackend::DFlash2
                                       ? options.speculative.draft_tokens
                                       : 0;
    program_options.auto_draft_length = options.speculative.draft_tokens_auto;
    program_options.draft_min_p       = options.speculative.draft_min_p.value_or(0.0F);
    program_options.diagnostics                 = options.diagnostic_observer;
    // The hybrid prefix cache (design §12). Snapshots are born in the pinned Host slab pool; the
    // cost constants rate a resume's restore against a re-prefill of the same tokens, at the
    // measured prefill and link rates of this family.
    if (options.context_cache.enabled) {
        const ContextCacheOptions& cache           = options.context_cache;
        program_options.prefix_cache               = true;
        program_options.prefix_host_bytes          = cache.host_capacity_bytes.value();
        program_options.prefix_taps.max_new_taps   = cache.hybrid.max_new_taps.value();
        program_options.prefix_taps.ladder_tokens  = cache.hybrid.tap_ladder_tokens.value();
        program_options.prefix_taps.min_gap_tokens = cache.hybrid.tap_min_gap_tokens.value();
        const std::uint32_t chunk                  = options.prefill_chunk;
        program_options.prefix_cost.chunk_seconds  = 0.1645 * (static_cast<double>(chunk) / 4096.0);
        program_options.prefix_cost.chunk_tokens   = chunk;
        program_options.prefix_cost.token_seconds  = 2.23e-5;
        program_options.prefix_cost.attention_pair_seconds = 1.06e-9;
        program_options.prefix_cost.call_route_fraction    = 10.0 / 512.0;
        program_options.prefix_cost.h2d_bytes_per_second =
            27.5e9; // the reference link, until measured
        program_options.prefix_cost.transfer_batch_seconds = 20.0e-6;
    }
    // A binding reserves its request's whole extent, so the automatic capacity holds every lane
    // at max_context and admission never waits for pages.
    program_options.kv_capacity_tokens = options.kv_capacity.mode == KvCapacityMode::Explicit
                                             ? options.kv_capacity.explicit_tokens
                                             : options.max_context * options.max_concurrency;
    offload::ExpertPlacementRuntime::Options& experts_options = program_options.placement;
    experts_options.vram_headroom = options.vram_headroom_bytes;
    experts_options.ram_headroom_bytes = options.ram_headroom_bytes;
    experts_options.expert_swap   = options.expert_swap;
    if (!options.expert_state_path.empty()) {
        experts_options.expert_state          = options.expert_state_path;
        experts_options.expert_state_identity = offload::expert_cache::artifact_identity(options.artifact_path);
    }

    // The device plan does not depend on the placement (the KV pool is fixed), so one plan serves
    // the placement's fit probe and the VRAM check below.
    const glm::ProgramDevicePlan device_plan =
        glm::Program::plan_device(program_options, plan.config(), plan.public_token_count());
    const std::uint64_t dense = plan.device_bytes();
    const auto frame_fit      = [&](const VramSnapshot& snapshot) {
        offload::VramDemand demand;
        demand.headroom    = options.vram_headroom_bytes;
        demand.graphs      = device_plan.graph_bound;
        demand.fixed       = dense + device_plan.fixed_bytes();
        demand.frame_bytes = device_plan.expert_frame_bytes; // an average frame of the per-size pools
        demand.max_frames  = device_plan.max_frames;
        return offload::size_expert_frames(snapshot, demand);
    };

    // Expert placement (--experts / --expert-placement): the shared resolution over the MoE
    // layers' keys.
    const glm::TextConfig& text = plan.config().text;
    const std::uint32_t layers  = text.moe_layers();
    const std::uint32_t experts = text.moe.experts;
    const std::uint64_t keys    = static_cast<std::uint64_t>(layers) * experts;
    const offload::VramSizing probe = frame_fit(vram->query());
    const offload::StartupPlacement placement = offload::resolve_startup_placement({
        .layers           = layers,
        .experts          = experts,
        .stride           = device_plan.expert_frame_bytes, // an average record (the sizing unit)
        .expert_banks     = demand.expert_banks,
        .experts_spec     = options.experts_spec,
        .expert_placement = options.expert_placement_path,
        .expert_ram_bytes = options.expert_ram_bytes,
        .lock_overhead    = demand.lock_overhead,
        // The ranking the spec's shares resolve by: the saved session's, else spread across the
        // layers (this family has no calibrated order).
        .load_ranking = [&](std::vector<std::uint32_t>& counts, std::vector<std::uint32_t>& ranked) {
            if (!experts_options.expert_state.empty()) {
                const auto saved = offload::expert_cache::load_expert_state(
                    experts_options.expert_state, experts_options.expert_state_identity, static_cast<std::uint32_t>(keys));
                if (saved.state) {
                    counts = saved.state->counts;
                    ranked = saved.state->ranked;
                }
            }
            return false;
        },
        // The resident and the placed probe plan the same device allocations.
        .probe = [&](bool) -> offload::StartupPlacementInputs::Fit {
            return {.frame_bytes = probe.frame_bytes, .frames = probe.frames, .max_frames = device_plan.max_frames,
                    .fixed_bytes = device_plan.fixed_bytes()};
        },
        .report = [&](DiagnosticLevel level, const std::string& line) { report(options, level, line); },
    });
    experts_options.resident_experts     = placement.resident;
    experts_options.expert_frame_target  = placement.frame_target;
    experts_options.expert_frame_ceiling = placement.frame_ceiling;
    experts_options.expert_pinned_keys   = placement.pinned_keys;
    experts_options.expert_tier_order    = placement.tier_order;
    if (placement.tier_expert_cap) { demand.expert_cap = *placement.tier_expert_cap; }
    if (placement.clear_lean_auto_cap) { demand.lean_auto_cap = false; }

    const auto ledger = offload::plan_host_memory(host_memory_snapshot(), demand);
    report(options, DiagnosticLevel::Info, ledger.summary());
    report(options, DiagnosticLevel::Debug, ledger.describe());
    if (ledger.cap_too_large) {
        throw std::runtime_error("--expert-ram-mib " + std::to_string(*options.expert_ram_bytes >> 20) +
                                 " asks for more Host RAM than is free or pinnable now (" + ledger.describe() +
                                 "); lower it, use auto, or free Host memory");
    }
    // The SSD tier: the banks stay in the artifact; the Program pins RAM slots for the experts the
    // ledger leaves room for (less their lock overhead) and reads the rest on use.
    const bool tier = ledger.placement == offload::ExpertPlacement::Tier;
    if (tier) {
        load.stream_experts = true;
        plan                = glm::plan_load(reader, load);
        experts_options.expert_ram_bytes =
            static_cast<std::uint64_t>(static_cast<double>(ledger.expert_pinned) / (1.0 + demand.lock_overhead));
        experts_options.expert_unpinned_bytes = ledger.expert_pageable;
        experts_options.later_pinned_bytes    = demand.later_pinned + offload::kPinRoomMarginBytes;
        experts_options.expert_stripe         = options.expert_stripe_path;
    } else if (!options.expert_stripe_path.empty()) {
        report(options, DiagnosticLevel::Info, "--expert-stripe is not used: every expert fits in RAM, so none is read from disk");
    }
    plan.set_host_reserve(options.ram_headroom_bytes, demand.later_pinned);

    // The VRAM check before the weights are read: the dense weights, the Program's fixed
    // allocations, the graph reserve and the headroom must leave room for the expert frames.
    const VramSnapshot before = vram->query();
    const auto check          = frame_fit(before);
    if (check.frame_bytes < 0) {
        throw std::runtime_error("GLM-5.3-Flash does not fit in device memory: " + std::to_string(mib(before.device_free)) +
                                 " MiB free; " + device_contributors(dense, device_plan) + "; reserve " +
                                 std::to_string(mib(check.reserve)) + "; headroom " + std::to_string(mib(check.headroom)) +
                                 " (" + check.describe() +
                                 "). Lower --max-context, --max-concurrency or --prefill-chunk, lower "
                                 "--vram-headroom-mib, or free VRAM");
    }
    if (check.frames < kFewFrames) {
        report(options, DiagnosticLevel::Warning,
               "decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                   std::to_string(kFewFrames) + " or more recommended); lower --max-context or --max-concurrency, "
                   "or free VRAM");
    }

    // The dense arena must land in device memory (WDDM can place it in system memory silently).
    SpillGuard guard(*vram);
    guard.begin();
    auto model = glm::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (const std::uint64_t spilled = guard.end(dense); spilled != 0) {
        throw std::runtime_error(std::to_string(mib(spilled)) +
                                 " MiB of the dense weights were placed in shared system memory by the driver's "
                                 "sysmem fallback: free VRAM and start again");
    }
    const std::uint64_t after_weights = vram->query().device_free;

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Glm5NextInstance>(std::move(model), options);
    frontend_phase.complete();

    StartupPhaseScope program_phase(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = std::make_unique<glm::Program>(instance->parameters, device, std::move(program_options));
    const offload::VramSizing& sizing = instance->program->vram_sizing();
    const std::uint64_t frames_bytes  = static_cast<std::uint64_t>(sizing.frames) * device_plan.expert_frame_bytes;
    const std::uint64_t after_startup = vram->query().device_free;
    {
        const double gib = static_cast<double>(1ULL << 30);
        char summary[256];
        std::snprintf(summary, sizeof(summary), "VRAM: model %.2f GiB, KV cache %.2f GiB, expert cache %u experts (%.1f GiB)",
                      static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib, sizing.frames,
                      static_cast<double>(frames_bytes) / gib);
        report(options, DiagnosticLevel::Info, summary);
        char line[512];
        std::snprintf(line, sizeof(line),
                      "VRAM ledger: %llu MiB card, %llu in use before loading; %s; reserve %llu (%u graphs); headroom "
                      "%llu (%s); expert frames %u; %llu free after startup",
                      mib(before.device_total), mib(before.device_total - before.device_free),
                      device_contributors(dense, device_plan).c_str(), mib(sizing.reserve), device_plan.graph_bound,
                      mib(sizing.headroom), options.vram_headroom_bytes ? "set" : "auto", sizing.frames,
                      mib(after_startup));
        report(options, DiagnosticLevel::Debug, line);
    }
    const MemorySummary memory               = instance->program->memory_summary();
    auto& resolution                         = instance->kv_capacity_resolution;
    resolution.mode                          = options.kv_capacity.mode;
    resolution.main_page_groups              = memory.kv_capacity_page_groups;
    resolution.maximum_main_page_groups      = memory.kv_capacity_max_page_groups;
    resolution.resolved_tokens               = memory.kv_capacity;
    resolution.runtime_reservation_bytes     = device_plan.fixed_bytes() + frames_bytes;
    resolution.available_after_weights_bytes = after_weights;
    resolution.available_after_startup_bytes = after_startup;
    resolution.automatic_headroom_bytes      = sizing.headroom;
    resolution.planned_slack_bytes =
        after_weights > resolution.runtime_reservation_bytes ? after_weights - resolution.runtime_reservation_bytes : 0;
    program_phase.complete();

    const auto& stats = instance->model->storage_stats();
    const auto& config = instance->model->config().text;
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Glm5Next));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"bf16", "fp32", "nvfp4_mul"};
    summary.fast_paths_disabled  = !options.fast_paths;

    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = config.vocab_size;
    metadata.embedding_size = config.hidden_size;
    metadata.native_context = config.max_position_embeddings;
    metadata.weights_id     = "nvfp4_mul";
    metadata.weight_bytes   = stats.device_capacity_bytes + stats.pinned_bytes;
    return {std::move(instance), std::move(summary), std::move(metadata)};
}

} // namespace

ExpertPlacementReport Glm5NextInstance::expert_placement_status() const {
    if (!program) { return {}; }
    return offload::to_expert_placement_report(program->expert_placement_status());
}

bool Glm5NextInstance::request_expert_placement(const ExpertPlacementSwitch& request, std::string* error) {
    if (!request.placement_file.empty()) {
        offload::ExpertPlacementRequest model_request;
        try {
            const auto& text = model->config().text;
            model_request    = offload::expert_placement_request_from_file(request.placement_file, text.moe_layers(),
                                                                           text.moe.experts);
        } catch (const std::exception& parse) {
            if (error != nullptr) { *error = parse.what(); }
            return false;
        }
        return program->request_expert_placement(model_request, error);
    }
    if (request.vram_experts == 0) {
        if (error != nullptr) { *error = "the switch needs {\"vram\": N, \"ram\": M} or {\"file\": PATH}"; }
        return false;
    }
    return program->request_expert_count(request.vram_experts, request.ram_experts, error);
}

bool Glm5NextInstance::export_expert_placement(const std::filesystem::path& path, std::string* error) {
    return program->export_expert_placement(path, error);
}

ExpertPlacementPreview Glm5NextInstance::preview_expert_placement(const std::string& experts_spec,
                                                                  const ExpertPlacementReport& now) const {
    if (!now.supported || now.total_experts == 0) {
        throw std::invalid_argument("this model runs without the expert cache");
    }
    return offload::preview_expert_placement(experts_spec, now);
}

ConstructedBackend construct_glm5_next_backend(EngineOptions& options, DeviceContext& device) {
    ConstructedGlm5Next constructed = construct_glm5_next(options, device);
    StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
    ConstructedBackend out{.load = std::move(constructed.load), .model_metadata = std::move(constructed.model_metadata)};
    out.backend = make_engine_backend(std::move(constructed.instance), device, options);
    finalize_phase.complete();
    return out;
}

std::unique_ptr<Glm5NextInstance> construct_glm5_next_instance(const EngineOptions& options,
                                                               DeviceContext& device) {
    return construct_glm5_next(options, device).instance;
}

} // namespace infernix::runtime
