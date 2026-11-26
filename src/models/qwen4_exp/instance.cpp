#include "models/qwen4_exp/instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/load_options.h"
#include "models/offload/expert_placement.h"
#include "models/offload/expert_placement_report.h"
#include "models/offload/expert_placement_startup.h"
#include "models/qwen4_exp/calibrated_order.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/options.h"
#include "models/offload/expert_state.h"
#include "models/qwen4_exp/program/fast_paths.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/sampling_defaults.h"
#include "models/offload/route_trace.h"
#include "models/offload/vram_monitor.h"
#include "models/registry.h"
#include "runtime/engine/engine_backend_for.h"
#include "runtime/engine/engine_options.h"
#include "runtime/engine/experts_spec.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::runtime {
namespace {

// The pinned Host tier of Qwen3.8-Flash-Next's prefix cache when --host-context-mib is not given.
// Planning allowance for the Program's pinned and mapped buffers (I/O staging, sampling, the CPU
// expert service's mapped records); the measured total is well below it.
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache and decode slows down
// markedly (design §9.7).
constexpr std::uint32_t kFewFrames = 2048;
// The RAM-lean default (see construct_qwen4_exp): frames of slack when probing whether the VRAM
// expert cache holds every routed expert (the probe sizes staging as if there were no tier, so it
// underestimates the frames), and the SSD tier's slot lists (ring, prefetch, demotion, stream
// bounce) plus an admission margin, in expert slots.
constexpr std::uint64_t kLeanFitMarginFrames = 512;
constexpr std::uint64_t kLeanTierListSlots   = 1024;

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

// The fixed device allocations, largest first in the order a user can act on them.
std::string device_contributors(std::uint64_t dense, const models::qwen4_exp::ProgramDevicePlan& plan) {
    char kv[64];
    if (plan.kv_max > plan.kv) {
        std::snprintf(kv, sizeof(kv), "%llu (grows to %llu from the expert cache)", mib(plan.kv), mib(plan.kv_max));
    } else {
        std::snprintf(kv, sizeof(kv), "%llu", mib(plan.kv));
    }
    char text[320];
    std::snprintf(text, sizeof(text),
                  "dense weights %llu MiB, KV %s, workspace %llu, expert staging %llu, state and io %llu, expert "
                  "tables %llu",
                  mib(dense), kv, mib(plan.workspace), mib(plan.staging), mib(plan.state + plan.io),
                  mib(plan.residency));
    return text;
}

void report(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    if (options.diagnostic_observer.callback) {
        try {
            options.diagnostic_observer.callback(Diagnostic{.level = level, .message = text});
        } catch (...) {}
    } else if (level != DiagnosticLevel::Debug) {
        std::fprintf(stderr, "[engine] %s\n", text.c_str());
    }
}

void validate(const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation && options.purpose != EnginePurpose::CausalScoring) {
        throw std::invalid_argument("Qwen3.8-Flash-Next serves generation and causal scoring");
    }
    if (options.speculative.backend == SpeculativeBackend::DFlash || options.speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next speculates with its MTP drafter, a DFlash2 drafter "
                                    "(--spec dflash2 --draft-model FILE) and n-gram copy proposals");
    }
    // Verification widths up to 8 keep its dense routes column-invariant (design 11.1); its rounds
    // verify chains.
    if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
        (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 7)) {
        throw std::invalid_argument("Qwen3.8-Flash-Next DFlash2 drafts 1-7 tokens per round (--draft-tokens)");
    }
    if (options.speculative.draft_tree_auto ||
        std::any_of(options.speculative.draft_tree_nodes.begin(), options.speculative.draft_tree_nodes.end(),
                    [](std::uint32_t nodes) { return nodes != 0; })) {
        throw std::invalid_argument("Qwen3.8-Flash-Next verifies chains: --draft-tree-nodes is not supported");
    }
    // 16-bit activations run every linear A16 and every expert bank W4A16 (the exact W4A16
    // arithmetic of design §17.3): the quality reference, so causal scoring only, on the exact
    // route.
    if (options.a16_activations && options.purpose != EnginePurpose::CausalScoring) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next runs 16-bit activations (the quality reference) for causal scoring only");
    }
    if (options.a16_activations && options.fast_paths) {
        throw std::invalid_argument(
            "16-bit activations (the quality reference) run the exact route: disable fast paths");
    }
    if (options.a16_activations && !options.fast_path_candidates.empty()) {
        throw std::invalid_argument("16-bit activations (the quality reference) run no fast path");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.expert_tail_skip > 9) {
        throw std::invalid_argument("--expert-tail-skip takes a top-10 rank 1-9 (0: off)");
    }
    if (options.expert_tail_skip != 0 && options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("--expert-tail-skip applies to generation (its decode rounds) only");
    }
}

} // namespace

Qwen4ExpInstance::Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> source,
                                   const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen_common::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .sampling                 = models::qwen4_exp::default_sampling(),
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads,
                               .vision_max_merged_tokens = options.vision_max_merged_tokens,
                               .thinking_budget_message  = options.thinking_budget_message})),
      capacity(options.max_context) {}

Qwen4ExpInstance::~Qwen4ExpInstance() = default;

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    validate(options);
    const auto start = std::chrono::steady_clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!models::qwen4_exp::is_qwen4_exp(reader)) {
        throw std::invalid_argument("artifact is not Qwen4ExpForCausalLM");
    }
    inspect.complete();
    auto plan = models::qwen4_exp::plan_load(
        reader, models::load_options(options, models::qwen4_exp::kVisionOffloadAuto));
    // The RAM ledger (design §9.8): the experts take what the reserve and every other planned
    // allocation leave, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = options.ram_headroom_bytes;
    demand.expert_banks = plan.pinned_expert_bytes();
    demand.other_pinned = plan.pinned_other_bytes();
    demand.later_pinned = kProgramPinnedAllowance;
    // The prefix cache pins its whole Host tier at startup, beside the experts.
    if (options.context_cache.enabled) {
        demand.prefix_cache = options.context_cache.host_capacity_bytes.value();
    }
    // The n-gram table its volume holds: the artifact's, or another volume's (any format) in its place.
    const std::filesystem::path volume = options.ngram_volume_path.empty()
                                             ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                             : options.ngram_volume_path;
    const models::qwen4_exp::NgramTableConfig table =
        models::qwen4_exp::resolve_ngram_volume(volume, plan.config().text.ple);
    if (const auto& own = plan.config().text.ple.table;
        table.format != own.format || table.volume_id != own.volume_id) {
        const auto name = [](models::qwen4_exp::NgramRowFormat format) {
            return format == models::qwen4_exp::NgramRowFormat::Bf16    ? std::string("BF16")
                   : format == models::qwen4_exp::NgramRowFormat::Iq4Nl ? std::string("IQ4_NL")
                                                                        : std::string("FP8");
        };
        report(options, DiagnosticLevel::Info,
               "n-gram table: " + name(table.format) + " rows from " + volume.string() + " in place of the artifact's " +
                   name(own.format) + " table");
    }
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(table);
    demand.load_staging = kLoadStagingBytes;
    demand.expert_cap   = options.expert_ram_bytes;
    // Windows page-locks memory for the GPU only up to the adapter's shared system memory; the
    // experts past that stay in RAM unpinned (the SSD tier's unpinned records).
    auto vram             = open_vram_budget_source(device.device, options.vram_past_budget);
    demand.pin_room       = models::qwen4_exp::pin_room(vram->query());
    demand.expert_pin_cap = options.expert_pin_bytes;

    // The ProgramOptions are assembled before the RAM ledger: the lean-tier decision below probes
    // the device plan, and the ledger only fills the SSD-tier fields afterwards.
    models::qwen4_exp::ProgramOptions program_options;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.causal_scoring  = options.purpose == EnginePurpose::CausalScoring;
    program_options.prefill_chunk   = options.prefill_chunk;
    program_options.prefill_chunk_beside_decode = options.prefill_chunk_beside_decode;
    program_options.kv_cache        = options.kv_cache;
    program_options.bf16_logits     = options.logits_dtype == LogitsDtype::BF16;
    program_options.diagnostics     = options.diagnostic_observer;
    program_options.kv_capacity_tokens =
        options.kv_capacity.mode == KvCapacityMode::Explicit ? options.kv_capacity.explicit_tokens : 0U;
    program_options.ngram_draft_tokens = options.speculative.ngram_draft_tokens;
    program_options.ngram_min_match    = options.speculative.ngram_min_match;
    program_options.mtp_draft_tokens   = options.speculative.backend == SpeculativeBackend::Mtp
                                             ? options.speculative.draft_tokens
                                             : 0U;
    program_options.dflash2_draft_tokens = options.speculative.backend == SpeculativeBackend::DFlash2
                                               ? options.speculative.draft_tokens
                                               : 0U;
    // A fixed draft length verifies every draft unless --spec-min-p asks for the cut.
    program_options.auto_draft_length = options.speculative.draft_tokens_auto;
    // MTP's automatic length cuts at 0.5 by default; DFlash2's calibrated length needs no cut.
    program_options.draft_min_p = options.speculative.draft_min_p.value_or(
        options.speculative.draft_tokens_auto && options.speculative.backend == SpeculativeBackend::Mtp ? 0.5F : 0.0F);
    if (!options.expert_state_path.empty()) {
        program_options.expert_state          = options.expert_state_path;
        program_options.expert_state_identity = models::offload::expert_cache::artifact_identity(options.artifact_path);
    }
    program_options.ngram_volume = volume;
    program_options.ngram_table  = table;
    // The hybrid prefix cache (design §15). Host-born snapshots live in the pinned slab pool;
    // the cost model ranks resume sources by the measured prefill coefficients.
    const ContextCacheOptions& cache = options.context_cache;
    if (cache.enabled) {
        const std::uint32_t chunk         = options.prefill_chunk;
        program_options.prefix_cache      = true;
        // normalize_engine_options resolved every value for this architecture.
        program_options.prefix_host_bytes          = cache.host_capacity_bytes.value();
        program_options.prefix_disk_bytes          = cache.hybrid.disk_capacity_bytes.value_or(0U);
        program_options.prefix_taps.max_new_taps   = cache.hybrid.max_new_taps.value();
        program_options.prefix_taps.ladder_tokens  = cache.hybrid.tap_ladder_tokens.value();
        program_options.prefix_taps.min_gap_tokens = cache.hybrid.tap_min_gap_tokens.value();
        // Fitted on the RTX 5090 at PCIe 5.0 x8, INT8 KV (the fit and its bounds are design §15.3).
        // The span is one pass of the experts over the link: the Program rescales
        // it, and the restore rate, to the link it measures at startup.
        program_options.prefix_cost.chunk_seconds          = 0.1645;
        program_options.prefix_cost.chunk_tokens           = chunk;
        program_options.prefix_cost.token_seconds          = 2.23e-5;
        program_options.prefix_cost.attention_pair_seconds = 1.06e-9;
        program_options.prefix_cost.call_route_fraction    = 10.0 / 512.0;
        program_options.prefix_span_seconds                = 1.67;
        program_options.prefix_cost.h2d_bytes_per_second   = 27.5e9; // the reference link, until measured
        program_options.prefix_cost.transfer_batch_seconds = 20.0e-6;
        program_options.prefix_save                        = cache.hybrid.persistent_save;
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait is
        // kept well inside its queue timeout.
        program_options.prefix_coalesce_wait_seconds = static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0;
    }
    program_options.route_trace   = options.route_trace_path.empty()
                                        ? models::offload::testing::route_trace()
                                        : options.route_trace_path;
    program_options.vram_headroom = options.vram_headroom_bytes;
    program_options.ram_headroom_bytes = options.ram_headroom_bytes;
    program_options.vram_past_budget = options.vram_past_budget;
    program_options.expert_tail_skip = options.expert_tail_skip;
    program_options.expert_swap      = options.expert_swap;
    if (!options.expert_swap) {
        report(options, DiagnosticLevel::Info,
               "--expert-swap off: the expert pools' content stays what the seeds and switches "
               "placed; misses still read and compute");
    }
    if (options.expert_tail_skip != 0) {
        report(options, DiagnosticLevel::Warning,
               "--expert-tail-skip " + std::to_string(options.expert_tail_skip) +
                   ": decode and verification rounds skip a routed expert missing from VRAM that every token "
                   "routes at top-10 rank " + std::to_string(options.expert_tail_skip) +
                   " or lower; output changes (lower quality) for faster decode");
    }

    // The n-gram table in RAM (design §12.5): pinned at Program start, so it is a later pin. Auto keeps it
    // only while every expert stays pinned in RAM with it; Ram keeps it whatever the experts' placement.
    const auto snapshot = host_memory_snapshot();
    // Expert placement (design §9.6; --experts / --expert-placement).
    // One of: Resident -- every routed expert in the frame pool, nothing lends or evicts, the KV
    // pool fully mapped, the RAM tier at its floor; Placed -- an explicit count or file naming the
    // VRAM set (pinned, the pool sized to it) and the RAM set; AutoHeadroom -- an explicit
    // --vram-headroom-mib the card cannot pay beside every expert, so the frames it evicts hold a
    // RAM hot set; Upstream -- a smaller card keeps the existing maximal RAM tier.
    const models::qwen4_exp::TextConfig& text_config = plan.config().text;
    const std::uint32_t placement_layers = static_cast<std::uint32_t>(text_config.num_hidden_layers);
    const std::uint32_t placement_experts = text_config.moe.experts;
    const std::uint64_t placement_keys = static_cast<std::uint64_t>(placement_layers) * placement_experts;
    const std::uint64_t stride = plan.pinned_expert_bytes() > 0 && placement_keys > 0
                                    ? plan.pinned_expert_bytes() / placement_keys
                                    : 0; // one expert record, the sizing unit of frames and slots
    // Expert placement (design §9.6; --experts / --expert-placement): the shared resolution
    // (models/offload/expert_placement_startup.h) turns a spec or file into the frame target and
    // ceiling, the pinned keys, the tier order and the ledger's expert cap -- or the resident
    // placement, when the family's device-plan probe holds every expert beside the fully-mapped
    // KV pool. One of: Resident -- every routed expert in the frame pool, nothing lends or
    // evicts, the KV pool fully mapped, the RAM tier at its floor; Placed -- an explicit count or
    // file naming the VRAM set (pinned, the pool sized to it) and the RAM set; AutoHeadroom -- an
    // explicit --vram-headroom-mib the card cannot pay beside every expert, so the frames it
    // evicts hold a RAM hot set; Upstream -- a smaller card keeps the existing maximal RAM tier.
    const models::offload::StartupPlacement placement = models::offload::resolve_startup_placement({
        .layers         = placement_layers,
        .experts        = placement_experts,
        .stride         = stride,
        .expert_banks   = demand.expert_banks,
        .experts_spec   = options.experts_spec,
        .expert_placement = options.expert_placement_path,
        .expert_ram_bytes = options.expert_ram_bytes,
        .lock_overhead  = demand.lock_overhead,
        // The ranking the spec's shares resolve by: the saved session's counts (the same state
        // the warm start reads), else the calibrated order, else spread across the layers.
        .load_ranking = [&](std::vector<std::uint32_t>& counts, std::vector<std::uint32_t>& ranked) {
            if (!program_options.expert_state.empty()) {
                const auto load = models::offload::expert_cache::load_expert_state(
                    program_options.expert_state, program_options.expert_state_identity,
                    static_cast<std::uint32_t>(placement_keys));
                if (load.state) {
                    counts = load.state->counts;
                    ranked = load.state->ranked;
                }
            }
            if (counts.empty() && ranked.empty() && text_config.hidden_size == 2560 &&
                placement_layers == models::qwen4_exp::kCalibratedLayers &&
                placement_experts == models::qwen4_exp::kCalibratedExperts) {
                const auto calibrated = models::qwen4_exp::calibrated_expert_order();
                counts = calibrated.counts;
                ranked = calibrated.ranked;
                return true;
            }
            return false;
        },
        // The family's device plan beside the fixed allocations; `resident` probes it with the
        // KV pool fully mapped (no elasticity, no frame target).
        .probe = [&](bool resident_probe) -> models::offload::StartupPlacementInputs::Fit {
            models::qwen4_exp::ProgramOptions probe_options = program_options;
            if (resident_probe) {
                probe_options.resident_experts    = true;
                probe_options.expert_frame_target = 0;
            }
            const auto probe = models::qwen4_exp::Program::plan_device(probe_options, plan.config(),
                                                                      plan.public_token_count());
            models::qwen4_exp::VramDemand probe_demand;
            probe_demand.headroom    = options.vram_headroom_bytes;
            probe_demand.graphs      = probe.graph_bound;
            probe_demand.fixed       = plan.device_bytes() + probe.fixed_bytes(); // includes the full KV pool
            probe_demand.frame_bytes = probe.expert_frame_bytes;
            probe_demand.max_frames  = probe.max_frames;
            const auto fit = models::qwen4_exp::size_expert_frames(vram->query(), probe_demand);
            return {.frame_bytes = fit.frame_bytes, .frames = fit.frames, .max_frames = probe.max_frames,
                    .fixed_bytes = probe.fixed_bytes()};
        },
        .report = [&](DiagnosticLevel level, const std::string& text) { report(options, level, text); },
    });
    program_options.expert_frame_target = placement.frame_target;
    if (!placement.pinned_keys.empty()) { program_options.expert_pinned_keys = placement.pinned_keys; }
    program_options.expert_tier_order    = placement.tier_order;
    program_options.expert_frame_ceiling = placement.frame_ceiling;
    if (placement.tier_expert_cap) { demand.expert_cap = *placement.tier_expert_cap; }
    if (placement.clear_lean_auto_cap) { demand.lean_auto_cap = false; }
    program_options.resident_experts = placement.resident;
    bool ngram_in_ram   = false;
    if (options.ngram_residency != NgramResidency::Volume) {
        auto with_table         = demand;
        with_table.later_pinned += models::qwen4_exp::NgramVolume::resident_bytes(table);
        with_table.pageable     = 0;
        const auto trial        = models::qwen4_exp::plan_host_memory(snapshot, with_table);
        if (options.ngram_residency == NgramResidency::Ram ||
            (trial.placement == models::qwen4_exp::ExpertPlacement::Full && !trial.cap_too_large)) {
            demand       = with_table;
            ngram_in_ram = true;
        }
    }
    program_options.ngram_in_ram = ngram_in_ram;
    const auto ledger   = models::qwen4_exp::plan_host_memory(snapshot, demand);
    report(options, DiagnosticLevel::Info,
           std::string("n-gram table: ") +
               (ngram_in_ram ? "in RAM (" + std::to_string(models::qwen4_exp::NgramVolume::resident_bytes(table) >> 20) +
                                   " MiB pinned)"
                             : options.ngram_residency == NgramResidency::Volume
                                   ? "read from its volume"
                                   : "read from its volume (the RAM ledger keeps the experts resident first)"));
    report(options, DiagnosticLevel::Info, ledger.summary());
    report(options, DiagnosticLevel::Debug, ledger.describe());
    if (ledger.cap_too_large) {
        throw std::runtime_error("--expert-ram-mib " + std::to_string(*options.expert_ram_bytes >> 20) +
                                 " asks for more Host RAM than is free or pinnable now (" + ledger.describe() +
                                 "); lower it, use auto, or free Host memory");
    }
    // The SSD tier (design §9.5): the banks stay in the artifact; the Program pins RAM slots for
    // the experts the ledger leaves room for (less their lock overhead) and reads the rest on use.
    const bool tier = ledger.placement == models::qwen4_exp::ExpertPlacement::Tier;
    if (tier) {
        auto streamed = models::load_options(options, models::qwen4_exp::kVisionOffloadAuto);
        streamed.stream_experts = true;
        plan                    = models::qwen4_exp::plan_load(reader, streamed);
    }
    // A small reserve trims other programs' working sets; keep Infernix's own pageable memory
    // resident so its heap and driver pages never hard-fault from the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report(options, DiagnosticLevel::Warning,
               "Windows would not guarantee Infernix " + std::to_string(demand.process_growth >> 20) +
                   " MiB of resident working memory (" + refused +
                   "); everything still works, but if RAM runs short Windows may move some of it to the "
                   "page file and responses slow down");
    }
    plan.set_host_reserve(options.ram_headroom_bytes, demand.later_pinned);

    if (!options.expert_stripe_path.empty()) {
        if (tier) {
            program_options.expert_stripe = options.expert_stripe_path;
        } else {
            report(options, DiagnosticLevel::Info,
                   "--expert-stripe is not used: every expert fits in RAM, so none is read from disk");
        }
    }
    if (tier) {
        program_options.expert_ram_bytes =
            static_cast<std::uint64_t>(static_cast<double>(ledger.expert_pinned) / (1.0 + demand.lock_overhead));
        program_options.expert_unpinned_bytes = ledger.expert_pageable;
        program_options.later_pinned_bytes    = demand.prefix_cache + demand.later_pinned + models::qwen4_exp::kPinRoomMarginBytes;
    }

    // The VRAM check (design §9.7), before the weights are read: the dense weights, the
    // Program's fixed allocations, the reserve for graph executables and the display headroom
    // must leave room for the expert cache. Fast paths are selected once the weights are bound, so
    // the check counts sampled-mtp-drafts' draft graphs whenever it could be selected.
    program_options.sampled_mtp_drafts =
        options.speculative.backend == SpeculativeBackend::Mtp &&
        (options.fast_paths || std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                                         FastPath::SampledMtpDrafts) != options.fast_path_candidates.end());
    const auto device_plan =
        models::qwen4_exp::Program::plan_device(program_options, plan.config(), plan.public_token_count());
    const std::uint64_t dense = plan.device_bytes();
    const VramSnapshot before = vram->query();
    models::qwen4_exp::VramDemand vram_demand;
    vram_demand.headroom    = options.vram_headroom_bytes;
    vram_demand.graphs      = device_plan.graph_bound;
    vram_demand.fixed       = dense + device_plan.fixed_bytes();
    vram_demand.frame_bytes = device_plan.expert_frame_bytes; // an average frame of the per-size pools
    vram_demand.max_frames  = device_plan.max_frames;
    const auto check        = models::qwen4_exp::size_expert_frames(before, vram_demand);
    if (check.frame_bytes < 0) {
        throw std::runtime_error(
            "Qwen3.8-Flash-Next does not fit in device memory: " + std::to_string(mib(before.device_free)) +
            " MiB free; " + device_contributors(dense, device_plan) + "; reserve " + std::to_string(mib(check.reserve)) +
            "; headroom " + std::to_string(mib(check.headroom)) + " (" + check.describe() +
            "). Lower --max-context, --max-concurrency or --prefill-chunk, use --kv-dtype int8, lower "
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
    auto model = models::qwen4_exp::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (const std::uint64_t spilled = guard.end(dense); spilled != 0) {
        throw std::runtime_error(std::to_string(mib(spilled)) +
                                 " MiB of the dense weights were placed in shared system memory by the driver's "
                                 "sysmem fallback: free VRAM and start again");
    }
    const std::uint64_t after_weights = vram->query().device_free;

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    // The fast paths qualified for this configuration (docs/maintainer/model-quality.md section 6.7).
    const models::qwen4_exp::FastPathSelection fast = models::qwen4_exp::select_fast_paths(instance->parameters, options);
    program_options.fast_paths        = fast.forward();
    program_options.mixed_calls       = program_options.fast_paths.column_invariant || !fast.request_invariant();
    program_options.pair_speculation  = fast.selected(FastPath::AdaptiveSpeculation);
    program_options.free_cache_resume = fast.selected(FastPath::FreeCacheResume) || fast.construction_exact();
    program_options.preemption        = fast.selected(FastPath::Preemption);
    program_options.sampled_mtp_drafts = fast.selected(FastPath::SampledMtpDrafts);

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        std::make_unique<models::qwen4_exp::Program>(instance->parameters, device, std::move(program_options));
    const auto& sizing = instance->program->vram_sizing();
    const std::uint64_t frames_bytes =
        static_cast<std::uint64_t>(sizing.frames) * device_plan.expert_frame_bytes;
    const std::uint64_t after_startup = vram->query().device_free;
    {
        const char* display = before.display == DisplayState::Headless   ? "no display"
                              : before.display == DisplayState::Attached ? "display attached"
                                                                         : "display unknown";
        // --vram-past-budget: the allowance already counted as free.
        const std::string past = before.budget_allowance == 0
                                     ? std::string()
                                     : " (" + std::to_string(mib(before.budget_allowance)) + " MiB past the OS budget)";
        char line[704];
        std::snprintf(line, sizeof(line),
                      "VRAM ledger: %llu MiB card, %s, %llu in use before loading%s; %s; reserve %llu (%u graphs); "
                      "headroom %llu (%s); expert frames %u (%.2f GiB); %llu free after startup",
                      mib(before.device_total), display, mib(before.device_total - before.device_free), past.c_str(),
                      device_contributors(dense, device_plan).c_str(), mib(sizing.reserve), device_plan.graph_bound,
                      mib(sizing.headroom), options.vram_headroom_bytes ? "set" : "auto", sizing.frames,
                      static_cast<double>(frames_bytes) / static_cast<double>(1ULL << 30), mib(after_startup));
        report(options, DiagnosticLevel::Debug, line);
        const double gib = static_cast<double>(1ULL << 30);
        char summary[320];
        if (device_plan.kv_max > device_plan.kv) {
            std::snprintf(summary, sizeof(summary),
                          "VRAM: model %.2f GiB, KV cache %.2f GiB (grows to %.2f GiB as contexts lengthen, taking room "
                          "from the expert cache), expert cache %u experts (%.1f GiB)",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib,
                          static_cast<double>(device_plan.kv_max) / gib, sizing.frames,
                          static_cast<double>(frames_bytes) / gib);
        } else {
            std::snprintf(summary, sizeof(summary), "VRAM: model %.2f GiB, KV cache %.2f GiB, expert cache %u experts (%.1f GiB)",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib, sizing.frames,
                          static_cast<double>(frames_bytes) / gib);
        }
        report(options, DiagnosticLevel::Info, summary);
    }
    const MemorySummary memory = instance->program->memory_summary();
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
    LoadSummary::PrefixCacheDisk disk;
    if (const std::filesystem::path& directory = options.context_cache.hybrid.disk_directory;
        options.context_cache.enabled && !directory.empty()) {
        // Stored states depend on the n-gram rows: another volume's differ from the artifact's table.
        std::string signature = "qwen4_exp;ngram=" + std::to_string(static_cast<int>(table.format)) + ":";
        for (const std::uint8_t byte : table.volume_id) {
            constexpr char kHex[] = "0123456789abcdef";
            signature += kHex[byte >> 4U];
            signature += kHex[byte & 15U];
        }
        if (table.fp8_scale) { signature += ";scale=" + std::to_string(*table.fp8_scale); }
        // A DFlash2 drafter's ring rides in every snapshot: it belongs to that companion artifact.
        if (const auto& draft = instance->model->info().draft_artifact_id) {
            signature += ";draft=";
            for (const std::byte byte : *draft) {
                constexpr char kHex[] = "0123456789abcdef";
                signature += kHex[std::to_integer<unsigned>(byte) >> 4U];
                signature += kHex[std::to_integer<unsigned>(byte) & 15U];
            }
        }
        const models::qwen4_exp::PrefixCacheDiskOpen opened =
            instance->program->attach_prefix_cache_disk(directory, hybrid_cache_fingerprint(options, signature));
        disk = LoadSummary::PrefixCacheDisk{
            .attached          = true,
            .mode              = opened.mode,
            .reset_reason      = opened.reset_reason,
            .damaged           = opened.damaged,
            .capacity_bytes    = options.context_cache.hybrid.disk_capacity_bytes.value_or(0U),
            .blocks            = opened.blocks,
            .snapshots         = opened.snapshots,
            .dropped_blocks    = opened.dropped_blocks,
            .dropped_snapshots = opened.dropped_snapshots,
            .bytes             = opened.bytes,
            .staging_bytes     = opened.staging_bytes,
            .seconds           = opened.seconds,
        };
    }
    program.complete();

    const auto& stats = instance->model->storage_stats();
    const auto& text  = instance->model->config().text;
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"bf16", "fp32", "nvfp4_mul"};
    summary.prefix_disk          = std::move(disk);
    summary.fast_path_configuration = models::qwen4_exp::describe(fast.configuration);
    for (const FastPath path : fast.paths) { summary.fast_paths.emplace_back(fast_path_name(path)); }
    summary.fast_paths_disabled        = !options.fast_paths;
    summary.request_invariance_capable = true;
    // Tail skip makes the output depend on which experts the cache holds.
    summary.request_invariant          = fast.request_invariant() && options.expert_tail_skip == 0;
    summary.column_invariant_kernels   = fast.forward().column_invariant;

    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = text.vocab_size;
    metadata.embedding_size = text.hidden_size;
    metadata.native_context = text.max_position_embeddings;
    metadata.weights_id     = "nvfp4_mul";
    metadata.weight_bytes   = stats.device_capacity_bytes + stats.pinned_bytes;

    EngineOptions resolved = options;
    return {std::move(instance), std::move(summary), std::move(metadata), std::move(resolved)};
}

const models::qwen4_exp::TextConfig& Qwen4ExpInstance::text_config_() const {
    return model->config().text;
}

ExpertPlacementReport Qwen4ExpInstance::expert_placement_status() const {
    if (!program) { return {}; }
    return models::offload::to_expert_placement_report(program->expert_placement_status());
}

bool Qwen4ExpInstance::request_expert_placement(const ExpertPlacementSwitch& request, std::string* error) {
    if (!program) {
        if (error != nullptr) { *error = "this engine runs without the expert cache"; }
        return false;
    }
    if (!request.placement_file.empty()) {
        models::offload::ExpertPlacementRequest model_request;
        try {
            const auto& text = text_config_();
            model_request = models::offload::expert_placement_request_from_file(request.placement_file,
                                                                                text.num_hidden_layers,
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

bool Qwen4ExpInstance::export_expert_placement(const std::filesystem::path& path, std::string* error) {
    if (!program) {
        if (error != nullptr) { *error = "this engine runs without the expert cache"; }
        return false;
    }
    return program->export_expert_placement(path, error);
}

ExpertPlacementPreview
Qwen4ExpInstance::preview_expert_placement(const std::string& experts_spec,
                                           const ExpertPlacementReport& now) const {
    if (!now.supported || now.total_experts == 0) {
        throw std::invalid_argument("this model runs without the expert cache");
    }
    return models::offload::preview_expert_placement(experts_spec, now);
}

std::string Qwen4ExpInstance::prefix_cache_save_notice(const EngineOptions& options) const {
    const HybridPrefixCacheOptions& hybrid = options.context_cache.hybrid;
    if (!options.context_cache.enabled || hybrid.disk_directory.empty()) { return {}; }
    return (options.context_cache.host_capacity_bytes.value_or(1U) == 0
                ? "finishing the prefix cache's writes to its Disk tier in "
                : "spilling the prefix cache to its Disk tier in ") +
           hybrid.disk_directory.string();
}

std::optional<PrefixCacheSave> Qwen4ExpInstance::prefix_cache_shutdown_save() const {
    const std::optional<models::qwen4_exp::PrefixCacheDiskSpill> spill =
        program->prefix_shutdown_spill();
    if (!spill) { return std::nullopt; }
    return PrefixCacheSave{.ok        = spill->ok,
                           .message   = spill->message,
                           .blocks    = spill->blocks,
                           .snapshots = spill->snapshots,
                           .bytes     = spill->bytes,
                           .seconds   = spill->seconds};
}

ConstructedBackend construct_qwen4_exp_backend(EngineOptions& options, DeviceContext& device) {
    ConstructedQwen4Exp constructed = construct_qwen4_exp(options, device);
    options                         = std::move(constructed.options);
    StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
    ConstructedBackend out{.load           = std::move(constructed.load),
                           .model_metadata = std::move(constructed.model_metadata)};
    out.backend = make_engine_backend(std::move(constructed.instance), device, options);
    finalize_phase.complete();
    return out;
}

} // namespace infernix::runtime
