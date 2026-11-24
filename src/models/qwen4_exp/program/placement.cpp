// The family's side of the expert placement: the hooks the shared placement runtime calls
// (models/offload/expert_placement_runtime.h), the prefill stream's bindings for the tier, the
// device-error check and the elastic KV control law's tail of apply_vram_target.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

void ProgramImpl::capture_sizing_inputs(const VramDemand& demand, const VramSnapshot& snapshot) {
    // The sizing law's inputs for grow_kv's spare allowance: the pool covers its own bytes
    // through pool_bytes, and the KV pool's later mappings move the usage baseline.
    spare_demand_       = demand;
    spare_demand_.elastic = true;
    spare_demand_.fixed   = 0;
    usage_at_sizing_    = snapshot.local_usage;
    pool_at_sizing_     = residency_->pool_bytes();
}

void ProgramImpl::on_tier_changed(offload::HostExpertTier* tier, bool rebuilt) {
    tier_ = tier;
    if (cpu_service_ != nullptr) { cpu_service_->set_record_provider(tier); }
    // The stream's callbacks captured the tier a rebuild replaced: retarget them to this one.
    if (rebuilt && expert_stream_ != nullptr) {
        StreamBindings bindings = stream_bindings();
        expert_stream_->retarget(std::move(bindings.record_of), bindings.one_allocation, std::move(bindings.hold),
                                 host_strides_, std::move(bindings.ssd), std::move(bindings.staging));
    }
}

bool ProgramImpl::has_calibrated_order() const noexcept {
    return c_.hidden_size == 2560 && c_.num_hidden_layers == qwen4_exp::kCalibratedLayers &&
           c_.moe.experts == qwen4_exp::kCalibratedExperts && c_.moe.top_k == 10 &&
           c_.moe.intermediate == 640 && c_.moe.shared_intermediate == 640;
}

ProgramImpl::StreamBindings ProgramImpl::stream_bindings() const {
    StreamBindings b;
    if (tier_ == nullptr) {
        std::vector<const std::uint8_t*> banks; // device addresses: the stream batches its copies
        for (const auto& layer : parameters_.layers) {
            banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.device_records));
        }
        b.record_of = [banks = std::move(banks), strides = host_strides_](std::uint32_t layer,
                                                                          std::uint32_t expert) {
            return banks[layer] + static_cast<std::size_t>(expert) * strides[layer];
        };
        b.one_allocation = true;
        return b;
    }
    offload::HostExpertTier* tier = tier_;
    b.record_of = [tier](std::uint32_t layer, std::uint32_t expert) { return tier->record(tier->key(layer, expert)); };
    b.hold      = [tier](std::uint32_t layer, std::uint32_t expert) { tier->stream_pin(tier->key(layer, expert)); };
    b.ssd.read    = [tier](std::uint32_t layer, std::uint32_t expert) {
        return tier->stream_read(tier->key(layer, expert));
    };
    b.ssd.wait    = [tier](std::int32_t slot) { return tier->stream_wait(slot); };
    b.ssd.release = [tier](std::int32_t slot) { tier->stream_release(slot); };
    b.ssd.slots   = [tier] { return tier->stream_slots(); };
    b.one_allocation = false;
    // Unpinned records stream through the tier's bounce slots, still two layers ahead; a
    // rebuilt tier holds none (its options clear the bounces), so its staging stays empty.
    if (tier->stream_bounce_slots() > 0) {
        b.staging = execution::ExpertStream::Staging{
            .unpinned = [tier](std::uint32_t layer, std::uint32_t expert) {
                return tier->unpinned_record(tier->key(layer, expert));
            },
            .slot     = [tier](std::uint32_t i) { return tier->stream_bounce_slot(i); },
            .copy     = [tier](std::span<const std::pair<std::uint8_t*, const std::uint8_t*>> copies) {
                tier->copy_records(copies);
            },
            .per_part = tier->stream_bounce_slots() / execution::ExpertStream::kBounceParts};
    }
    return b;
}

void ProgramImpl::check_device_error() {
    auto* word           = static_cast<volatile std::uint32_t*>(device_error_.data());
    std::uint32_t value  = *word;
    if (value == 0 && testing::take_expert_fault()) { value = EIO; } // layer 0
    if (value == 0) { return; }
    *word                    = 0;
    clear_prefix_on_release_ = true;
    // A stopped service or read agent fails every later call too: stop the Engine with its cause.
    if (cpu_service_) {
        if (const std::string why = cpu_service_->failure(); !why.empty()) {
            throw std::runtime_error("Qwen3.8-Flash-Next: the CPU expert service stopped (" + why + ")");
        }
    }
    if (tier_) {
        if (const std::string why = tier_->failure(); !why.empty()) {
            throw std::runtime_error("Qwen3.8-Flash-Next: the SSD expert tier's read agent stopped (" + why + ")");
        }
    }
    const std::uint32_t code = value & 0xFFFFU;
    if (code == execution::kErrorTokenId) {
        throw runtime::RecoverableExecutionError("Qwen3.8-Flash-Next: a token id outside the vocabulary reached the "
                                                 "token embedding");
    }
    if (code == execution::kErrorRowGateTimeout) {
        throw runtime::RecoverableExecutionError(
            "Qwen3.8-Flash-Next layer " + std::to_string(value >> 16U) + ": the n-gram rows were not published within " +
            std::to_string(kPinnedGateTimeoutNs / 1000000000ULL) + " s (the host thread reading them stalled)");
    }
    std::string what         = code == ops::offloaded_moe::kErrorUnservedRecord ? "an SSD-only expert had no path to it"
                               : code == ops::offloaded_moe::kErrorHostSilent   ? "the CPU expert service stopped answering"
                               : code == ops::offloaded_moe::kErrorFetchSilent  ? "the SSD expert tier's read agent stopped answering"
                               : code == ops::offloaded_moe::kErrorRowsWaitTimeout
                                 ? "the routed experts' down pass was not released within " +
                                   std::to_string(ops::offloaded_moe::kRowsWaitTimeoutNs / 1000000000ULL) +
                                   " s (an engine defect: the round failed rather than hang the GPU)"
                                                                                 : "an expert record could not be read (" +
                                                                                   std::generic_category().message(static_cast<int>(code)) + ")";
    throw runtime::RecoverableExecutionError("Qwen3.8-Flash-Next layer " + std::to_string(value >> 16U) + ": " + what);
}

void ProgramImpl::apply_vram_target(bool idle) {
    // The placement switches, the fills and the idle settle are the shared placement runtime's;
    // it returns whether the elastic KV control law may run (the pools hold every expert, or an
    // explicit placement names the pool's size and only its switches move it).
    if (!placement_runtime_->apply_placement(idle)) { return; }
    if (idle && kv_idle_due() &&
        std::all_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.phase == Phase::Free; })) {
        kv_busy(); // one attempt per idle period: blocks without host copies may still hold the top
        shrink_kv(true);
    }
}

} // namespace infernix::models::qwen4_exp::detail
