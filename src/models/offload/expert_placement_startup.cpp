// The startup expert-placement resolution (see the header), lifted from the first family's
// instance: the --experts / --expert-placement conflict, the file and spec forms, the resident
// probe and the placed form's fit probe, and the tier's RAM cap. The family's probes and
// ranking arrive as hooks.

#include "models/offload/expert_placement_startup.h"

#include "runtime/engine/experts_spec.h"

#include <stdexcept>
#include <utility>

namespace infernix::models::offload {

namespace {
// The RAM tier's fixed lists beside its resident slots (ring, prefetch, demotion, bounce; the
// Program sizes them from the same ledger fields).
constexpr std::uint64_t kTierListAllowance = 1536;
// The resident probe's staging is sized as if tierless, so it underestimates the frames; this
// much slack covers it.
constexpr std::uint32_t kFitMarginFrames = 512;
} // namespace

StartupPlacement resolve_startup_placement(const StartupPlacementInputs& inputs) {
    StartupPlacement out;
    const std::uint64_t keys = static_cast<std::uint64_t>(inputs.layers) * inputs.experts;
    if (inputs.expert_ram_bytes.has_value() &&
        (!inputs.experts_spec.empty() || !inputs.expert_placement.empty())) {
        throw std::runtime_error("--expert-ram-mib sizes the RAM tier by bytes; --experts and "
                                 "--expert-placement name it by experts. Choose one");
    }
    // The ranking the spec's shares resolve by: the saved session's counts (the same state the
    // warm start reads), else the calibrated order, else spread across the layers.
    if (!inputs.expert_placement.empty()) {
        out.spec = parse_expert_placement_file(inputs.expert_placement, inputs.layers, inputs.experts);
        out.placed = true;
        out.origin = inputs.expert_placement.string();
    } else if (!inputs.experts_spec.empty()) {
        // The exclusive three-tier spec: shares or counts by usage rank. vram=all (or 100%) is
        // the resident placement; the probe below holds it or fails the start.
        const runtime::ExpertsSpec experts = runtime::parse_experts_spec(inputs.experts_spec);
        const auto [vram_count, ram_count] = runtime::experts_spec_counts(experts, keys);
        if (vram_count >= keys) {
            out.all_in_vram = true;
            out.origin = inputs.experts_spec;
        } else {
            std::vector<std::uint32_t> counts, ranked;
            const bool calibrated = inputs.load_ranking(counts, ranked);
            out.spec = placement_from_counts(
                static_cast<std::uint32_t>(vram_count), static_cast<std::uint32_t>(ram_count), inputs.layers,
                inputs.experts, counts, ranked);
            out.placed = true;
            out.origin = inputs.experts_spec + " by " + (calibrated ? "the calibrated order" : "the ranking");
        }
    }

    const auto set_tier_cap = [&](std::uint64_t resident_slots) {
        const std::uint64_t slots = resident_slots + kTierListAllowance;
        std::uint64_t cap = slots * inputs.stride +
                            static_cast<std::uint64_t>(static_cast<double>(slots * inputs.stride) *
                                                       inputs.lock_overhead);
        if (inputs.expert_ram_bytes.has_value()) {
            cap = *inputs.expert_ram_bytes; // --expert-ram-mib keeps byte-level control
        }
        out.tier_expert_cap = cap;
    };
    if (out.placed) {
        // The VRAM set must fit the frame pool; the RAM set the ledger. A file names exact keys,
        // so its vram set pins; a counts/shares spec names sizes only, so its frames stay
        // adaptive within the count (the per-round policy swaps hot experts in and cold out).
        out.frame_target = static_cast<std::uint32_t>(out.spec.vram.size());
        if (!inputs.expert_placement.empty()) { out.pinned_keys = out.spec.vram; }
        out.tier_order    = out.spec.order;
        // The placement names the pool's size: the VRAM monitor never grows past it.
        out.frame_ceiling = static_cast<std::uint32_t>(out.spec.vram.size());
        set_tier_cap(out.spec.ram.size());
        if (inputs.report) {
            inputs.report(DiagnosticLevel::Info,
                          "expert placement: " + out.origin + " -- " + std::to_string(out.spec.vram.size()) +
                              " experts pinned in VRAM, " + std::to_string(out.spec.ram.size()) +
                              " in RAM, the rest from the artifact");
        }
    } else if (out.all_in_vram || (inputs.stride > 0 && inputs.expert_banks > 0 &&
                                   !inputs.expert_ram_bytes.has_value())) {
        // The resident probe: the device plan with the KV pool fully mapped (no elasticity) must
        // leave room for every expert beside it. vram=all demands it; the default falls back to
        // the upstream maximal tier when the card cannot.
        const StartupPlacementInputs::Fit fit = inputs.probe(true);
        const std::uint64_t experts = keys;
        const bool fits = fit.frames + kFitMarginFrames >= experts && fit.frame_bytes >= 0;
        if (!fits && out.all_in_vram) {
            throw std::runtime_error("--experts vram=all: every routed expert does not fit in VRAM beside "
                                     "the fully-mapped KV pool and the fixed allocations; lower the spec's "
                                     "vram share, or --max-context/--max-concurrency");
        }
        if (fits) {
            out.resident = true;
            out.frame_target = fit.max_frames;
            // Only the tier's floor: every expert lives in VRAM, so RAM holds its lists and the
            // cold fill's landing staging, nothing else.
            const std::uint64_t floor_slots = kTierListAllowance;
            out.tier_expert_cap = floor_slots * inputs.stride +
                                  static_cast<std::uint64_t>(static_cast<double>(floor_slots * inputs.stride) *
                                                             inputs.lock_overhead);
            out.clear_lean_auto_cap = true;
            const std::uint64_t kv_gib = fit.fixed_bytes >> 30;
            if (inputs.report) {
                inputs.report(DiagnosticLevel::Info,
                              "every expert fits in VRAM beside the fully-mapped KV pool (" + std::to_string(kv_gib) +
                                  " GiB fixed): holding all " + std::to_string(experts) +
                                  " resident -- nothing is lent or evicted; the fill's ~" +
                                  std::to_string((floor_slots * inputs.stride) >> 20) +
                                  " MiB RAM reader releases once the experts are loaded, leaving 0 host expert bytes. "
                                  "Pass --experts vram=N%,ram=M% for a smaller set");
            }
        }
        // Otherwise (a smaller card, or an explicit --expert-ram-mib): the existing maximal RAM
        // tier, unchanged. Headroom no longer sizes the expert placement: name the placement
        // with --experts and keep the headroom for the display and other programs.
    }
    if (out.placed && inputs.stride > 0) {
        // The VRAM set must fit beside the fixed allocations.
        const StartupPlacementInputs::Fit fit = inputs.probe(false);
        const std::uint64_t experts = keys;
        if (fit.frame_bytes < 0 || static_cast<std::uint64_t>(fit.frames) + kFitMarginFrames < out.spec.vram.size() ||
            out.spec.vram.size() > experts) {
            throw std::runtime_error(
                "the expert placement pins " + std::to_string(out.spec.vram.size()) +
                " experts in VRAM, but only " + std::to_string(fit.frames) +
                " frames fit beside the fixed allocations; lower the spec's vram share (or the file's vram "
                "rows), or --max-context/--max-concurrency");
        }
    }
    return out;
}

} // namespace infernix::models::offload
