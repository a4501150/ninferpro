// The routed experts' tiers through the shared expert placement runtime
// (models/offload/expert_placement_runtime.h): VRAM frames, with streamed banks the RAM tier's
// pinned slots and the artifact behind them, the warm start, the switches and the saved state.
// What stays here: the Forward's wiring, the round boundary and the device's error word the
// offloaded MoE reports through.

#include "models/glm5_next/program/program_impl.h"

#include <algorithm>
#include <stdexcept>
#include <system_error>

namespace infernix::models::glm5_next::detail {

namespace {

// The RAM tier's floor in pinned slots beside its lists: GLM's records are 8-13.5 MiB, so the
// floor is in hundreds of slots, not the thousands of a smaller expert.
constexpr std::uint32_t kMinTierSlots = 64;

} // namespace

void ProgramImpl::create_experts(execution::ForwardExperts& experts, std::uint64_t& allocated) {
    // A GGUF import's banks differ in record stride by layer (the residency's per-size pools);
    // NVFP4 banks share one.
    const auto& banks = parameters_.model.expert_banks();
    std::vector<const std::uint8_t*> records;
    std::vector<std::uint64_t> host_strides;
    std::uint64_t record_stride = 0;
    for (const auto& bank : banks) {
        records.push_back(reinterpret_cast<const std::uint8_t*>(bank.planes.device_records));
        host_strides.push_back(bank.planes.record_stride);
        record_stride = std::max(record_stride, bank.planes.record_stride);
    }
    if (record_stride != plan_.bytes.expert_record_stride) {
        throw std::logic_error("Glm5Next: the expert banks' largest record stride differs from the device plan's");
    }

    offload::ExpertPlacementRuntime::Options placement = options_.placement;
    placement.label                 = "GLM-5.3-Flash";
    placement.geometry              = bank_geometry(c_);
    placement.stripe_artifact_id    = parameters_.model.info().artifact_id;
    placement.staging_slots         = static_cast<std::uint32_t>(plan_.staging_slots);
    placement.chunk_columns         = static_cast<std::uint32_t>(chunk_);
    placement.min_tier_slots        = kMinTierSlots;
    placement.prefill_stream        = options_.prefill_stream;
    placement.stream_read_slots     = kStreamReadSlots;
    // The prefill stream's bounce per part: a chunk streams about a layer's unpinned share of its
    // experts (the stream copies two halves of a layer's unpinned records per call).
    if (options_.prefill_stream) {
        if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
            const std::uint64_t keys =
                static_cast<std::uint64_t>(store->layers()) * store->experts();
            const double share = std::min(
                1.0, static_cast<double>(options_.placement.expert_unpinned_bytes / record_stride) /
                         static_cast<double>(keys));
            placement.stream_bounce_per_part = std::clamp<std::uint32_t>(
                (static_cast<std::uint32_t>(std::ceil(share * store->experts())) + 15U) / 16U * 16U,
                kStreamBounceMin, kStreamBounceMax);
            placement.stream_bounce_parts = execution::ExpertStream::kBounceParts;
        }
    }
    offload::ExpertPlacementRuntime::Hooks hooks;
    hooks.diagnostic      = [this](const std::string& text, DiagnosticLevel level) { diagnostic(text, level); };
    hooks.on_tier_changed = [this](offload::HostExpertTier* tier, bool rebuilt) {
        tier_ = tier;
        if (cpu_service_ != nullptr) { cpu_service_->set_record_provider(tier); }
        // The stream's callbacks captured the tier a rebuild replaced: retarget them to this one.
        if (rebuilt && expert_stream_ != nullptr) {
            StreamBindings bindings = stream_bindings();
            expert_stream_->retarget(std::move(bindings.record_of), bindings.one_allocation,
                                     std::move(bindings.hold), std::move(bindings.ssd),
                                     std::move(bindings.staging));
        }
    };
    hooks.on_switch_settled = [this](const char* reason) {
        round_cost_.reset();
        diagnostic(std::string("DFlash2 round cost fit reset: ") + reason);
    };
    hooks.all_lanes_idle  = [this] {
        return std::all_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.phase == Phase::Free; });
    };
    hooks.kv_bytes = [this] { return static_cast<std::uint64_t>(plan_.kv); };
    vram_          = open_vram_budget_source(device_.device);
    placement_     = std::make_unique<offload::ExpertPlacementRuntime>(device_, *vram_, std::move(placement),
                                                                       std::move(hooks));

    allocate(staging_, plan_.staging, allocated);
    experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
    experts.staging_slots = plan_.staging_slots;
    residency_            = &placement_->create_residency(std::move(records), record_stride, columns_, std::move(host_strides));
    allocated += plan_.bytes.residency;

    CUDA_CHECK(cudaStreamCreateWithFlags(&overlap_stream_, cudaStreamNonBlocking));
    for (auto& event : overlap_events_) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    experts.overlap_stream = overlap_stream_;
    experts.overlap_events = overlap_events_;

    // Streamed banks: RAM slots hold the records the tier keeps and the rest are read from the
    // artifact when a round needs them.
    if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
        placement_->create_tier(*store);
        tier_          = placement_->tier();
        fetch_channel_ = placement_->fetch_channel();
    }

    // The prefill stream packs each layer's ring half at that layer's own record stride.
    if (options_.prefill_stream) {
        std::vector<std::uint64_t> stream_strides;
        stream_strides.reserve(banks.size());
        for (const auto& bank : banks) { stream_strides.push_back(bank.planes.record_stride); }
        StreamBindings bindings = stream_bindings();
        expert_stream_          = std::make_unique<execution::ExpertStream>(
            c_.moe_layers(), c_.moe.experts, std::move(stream_strides), std::move(bindings.record_of),
            bindings.one_allocation, std::move(bindings.hold), std::move(bindings.ssd));
        expert_stream_->set_staging(std::move(bindings.staging));
        experts.stream = expert_stream_.get();
    }

    // The frames take what the fixed allocations, the graph reserve and the headroom leave.
    offload::VramDemand demand;
    demand.headroom    = options_.placement.vram_headroom;
    demand.graphs      = plan_.bytes.graph_bound;
    demand.fixed       = plan_.bytes.fixed_bytes() > allocated ? plan_.bytes.fixed_bytes() - allocated : 0;
    demand.frame_bytes = residency_->frame_bytes();
    demand.max_frames  = residency_->frame_limit();
    sizing_            = placement_->start_expert_frames(demand).sizing;
    diagnostic("expert cache: " + std::to_string(sizing_.frames) + " of " + std::to_string(residency_->frame_limit()) +
               " experts in VRAM frames (" +
               std::to_string((static_cast<std::uint64_t>(sizing_.frames) * residency_->frame_bytes()) >> 20) + " MiB)");
    diagnostic("device address map: " + residency_->address_map(), DiagnosticLevel::Debug);

    // Decode and verification calls hand part of their host-memory misses to the CPU team, which
    // computes them from the records in place while the link stages the rest.
    const std::int32_t cpu_share = c_.ggml_experts ? options_.cpu_ggml_link_share256 : options_.cpu_link_share256;
    if (options_.cpu_expert_workers > 0 && options_.cpu_expert_jobs > 0) {
        std::vector<ops::offloaded_moe::CpuMissService::Layer> layers;
        for (const auto& bank : banks) {
            layers.push_back({.records = reinterpret_cast<const std::uint8_t*>(bank.planes.records),
                              .record_stride = bank.planes.record_stride,
                              .scales        = bank.scales.data(),
                              .ggml          = ggml_format(bank)});
        }
        const int decode_columns =
            std::max(ops::offloaded_moe::kMaxCpuColumns,
                     static_cast<int>(options_.max_concurrency * (options_.draft_tokens + 1)));
        cpu_service_ = std::make_unique<ops::offloaded_moe::CpuMissService>(
            std::move(layers),
            ops::offloaded_moe::CpuMissService::Options{
                .workers     = static_cast<int>(options_.cpu_expert_workers),
                .max_jobs    = static_cast<int>(std::min<std::uint32_t>(
                    options_.cpu_expert_jobs, ops::offloaded_moe::kMaxCpuJobs)),
                .max_columns = decode_columns,
                .pcie_share  = cpu_share,
                .activation  = ops::offloaded_moe::ExpertActivation::kA16,
                .geometry    = {.hidden       = static_cast<int>(c_.hidden_size),
                                .intermediate = static_cast<int>(c_.moe.intermediate),
                                .swiglu_limit = c_.swiglu_limit},
                .records     = tier_});
    }

    const offload::ExpertBankGeometry geometry = bank_geometry(c_);
    for (std::uint32_t l = 0; l < geometry.banks; ++l) {
        if (cpu_service_) { experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l))); }
        experts.frame_bases.push_back(residency_->frame_base(l));
        experts.frame_pitches.push_back(residency_->frame_pitch(l));
        experts.frames.push_back(residency_->table(l));
        if (tier_) {
            experts.host_tables.push_back(residency_->record_table(l));
            experts.fetch.push_back(fetch_channel_->channel(static_cast<int>(l)));
        }
    }
    experts.route_log     = residency_->route_log();
    experts.route_stride  = residency_->route_stride();
    experts.landing       = residency_->landing_table();
    experts.landed        = residency_->landed_log();
    experts.landing_slots = static_cast<std::int32_t>(offload::ExpertResidency::kLandingSlots);
    experts.error         = static_cast<std::uint32_t*>(device_error_.data());
    experts.resident           = residency_->resident();
    experts.ggml_bf16_prefill  = options_.ggml_bf16_prefill;
    experts.expert_a4_residual = options_.expert_a4_residual;

    // The saved state's experts (or, without one, the tier's spread fill) before the first request.
    placement_->warm_start_experts();
}

void ProgramImpl::before_call(bool decode_round) {
    // Beside an active layer walk nothing lands: the walk's stream planned against its start's
    // frames.
    residency_->before_round(device_.stream,
                             /*landing=*/decode_round && !residency_->resident() && !walk_.active,
                             /*walking=*/walk_.active);
    // The CPU lane's first request comes within this round.
    if (cpu_service_ && decode_round) { cpu_service_->wake(); }
    if (fetch_channel_ && tier_ && !tier_->round_open()) {
        throw std::logic_error("Glm5Next: a call reading SSD-only experts was enqueued outside the tier's round");
    }
}

void ProgramImpl::enqueue_routes(std::int32_t columns) { residency_->enqueue_route_download(device_.stream, columns); }

bool ProgramImpl::expert_route() const noexcept { return residency_->every_key_ready(); }

void ProgramImpl::set_expert_route() { forward_->set_experts_resident(expert_route()); }

void ProgramImpl::check_device_error() {
    auto* word                = static_cast<volatile std::uint32_t*>(device_error_.data());
    const std::uint32_t value = *word;
    if (value == 0) { return; }
    *word = 0;
    if (cpu_service_) {
        if (const std::string why = cpu_service_->failure(); !why.empty()) {
            throw std::runtime_error("GLM-5.3-Flash: the CPU expert service stopped (" + why + ")");
        }
    }
    if (const std::string why = placement_->tier_failure(); !why.empty()) {
        throw std::runtime_error("GLM-5.3-Flash: the SSD expert tier's read agent stopped (" + why + ")");
    }
    const std::uint32_t code = value & 0xFFFFU;
    if (code == execution::kErrorTokenId) {
        throw runtime::RecoverableExecutionError("GLM-5.3-Flash: a token id outside the vocabulary reached the token embedding");
    }
    const std::string what = code == ops::offloaded_moe::kErrorUnservedRecord ? "an SSD-only expert had no path to it"
                             : code == ops::offloaded_moe::kErrorHostSilent   ? "the CPU expert service stopped answering"
                             : code == ops::offloaded_moe::kErrorFetchSilent  ? "the SSD expert tier's read agent stopped answering"
                             : code == ops::offloaded_moe::kErrorRowsWaitTimeout
                                 ? std::string("the routed experts' down pass was not released in time")
                                 : "an expert record could not be read (" +
                                       std::generic_category().message(static_cast<int>(code)) + ")";
    throw runtime::RecoverableExecutionError("GLM-5.3-Flash MoE layer " + std::to_string(value >> 16U) + ": " + what);
}

std::optional<ExpertCacheStats> ProgramImpl::expert_cache_stats(const Lane& lane) const {
    const auto& s = residency_->stats();
    const auto& a = lane.cache_at_admission;
    return ExpertCacheStats{.routed        = s.routed - a.routed,
                            .hits          = s.hits - a.hits,
                            .decode_routed = s.decode_routed - a.decode_routed,
                            .decode_hits   = s.decode_hits - a.decode_hits};
}

ProgramImpl::StreamBindings ProgramImpl::stream_bindings() const {
    StreamBindings b;
    if (tier_ == nullptr) {
        // Pinned addresses (the stream batches its copies) and each bank's own stride.
        std::vector<std::pair<const std::uint8_t*, std::uint64_t>> banks;
        for (const auto& bank : parameters_.model.expert_banks()) {
            banks.emplace_back(reinterpret_cast<const std::uint8_t*>(bank.planes.device_records), bank.planes.record_stride);
        }
        b.record_of = [banks = std::move(banks)](std::uint32_t layer, std::uint32_t expert) {
            return banks[layer].first + static_cast<std::size_t>(expert) * banks[layer].second;
        };
        b.one_allocation = true;
        return b;
    }
    offload::HostExpertTier* tier = tier_;
    b.record_of                   = [tier](std::uint32_t layer, std::uint32_t expert) {
        return tier->record(tier->key(layer, expert));
    };
    b.hold = [tier](std::uint32_t layer, std::uint32_t expert) {
        tier->stream_pin(tier->key(layer, expert));
    };
    b.ssd.read = [tier](std::uint32_t layer, std::uint32_t expert) {
        return tier->stream_read(tier->key(layer, expert));
    };
    b.ssd.wait       = [tier](std::int32_t slot) { return tier->stream_wait(slot); };
    b.ssd.release    = [tier](std::int32_t slot) { tier->stream_release(slot); };
    b.ssd.slots      = [tier] { return tier->stream_slots(); };
    b.one_allocation = false;
    // Unpinned records stream through the tier's bounce slots, still two layers ahead; a rebuilt
    // tier holds none (its options clear the bounces), so its staging stays empty.
    if (tier->stream_bounce_slots() > 0) {
        b.staging = execution::ExpertStream::Staging{
            .unpinned =
                [tier](std::uint32_t layer, std::uint32_t expert) {
                    return tier->unpinned_record(tier->key(layer, expert));
                },
            .slot = [tier](std::uint32_t i) { return tier->stream_bounce_slot(i); },
            .copy =
                [tier](std::span<const std::pair<std::uint8_t*, const std::uint8_t*>> copies) {
                    tier->copy_records(copies);
                },
            .per_part = tier->stream_bounce_slots() / execution::ExpertStream::kBounceParts};
    }
    return b;
}

void ProgramImpl::maintain() {
    if (!walk_.active) { (void)placement_->apply_placement(true); }
    placement_->maintain_expert_state();
    // The prefix cache's copies publish while the engine is idle (its captures and writes land).
    if (prefix_) { prefix_->poll(); }
}

} // namespace infernix::models::glm5_next::detail
