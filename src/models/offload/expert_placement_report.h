#pragma once

// The instance-level conversions of an offloaded-MoE family's expert-placement endpoints
// (docs/maintainer/moe-offload-design.md §4.8, lifted from the first family's instance): the
// runtime status to the public report, a switch's placement file to the request, and the counts
// form's dry-run preview. Header-only: the families' instances call these beside their own glue
// (a missing program's error messages, the public Engine's types).

#include "infernix/types.h"
#include "models/offload/expert_placement.h"
#include "models/offload/expert_placement_runtime.h"
#include "runtime/engine/experts_spec.h"

#include <exception>
#include <filesystem>
#include <string>

namespace infernix::models::offload {

// The public report from the runtime's status.
[[nodiscard]] inline ExpertPlacementReport to_expert_placement_report(const ExpertPlacementStatus& status) {
    ExpertPlacementReport report;
    report.supported      = true;
    report.resident      = status.resident;
    report.switching     = status.target != 0;
    report.frames        = status.frames;
    report.frame_limit   = status.frame_limit;
    report.target        = status.target;
    report.pinned        = status.pinned;
    report.residents     = status.residents;
    report.tier_slots    = status.tier_slots;
    report.tier_residents = status.tier_residents;
    report.has_tier_target = status.has_tier_target;
    report.tier_target   = status.tier_target;
    report.fill_total    = status.fill_total;
    report.fill_seeded   = status.fill_seeded;
    report.total_experts = status.total_experts;
    report.swap_routed     = status.swap_routed;
    report.swap_hits      = status.swap_hits;
    report.swap_promotions = status.swap_promotions;
    report.swap_demotions  = status.swap_demotions;
    report.swap_promotion_bytes = status.swap_promotion_bytes;
    report.swap_demotion_bytes  = status.swap_demotion_bytes;
    report.swap_admitted   = status.swap_admitted;
    report.tier_shadows    = status.tier_shadows;
    report.kv_bytes        = status.kv_bytes;
    report.last_switch_ok      = status.last_switch_ok;
    report.last_switch_message = status.last_switch_message;
    report.expert_record_bytes = status.expert_record_bytes;
    report.frame_bytes   = status.frame_bytes;
    report.pool_bytes    = status.pool_bytes;
    report.ssd_experts   = status.ssd_experts;
    report.vram_free     = status.vram_free;
    report.vram_total    = status.vram_total;
    report.host_total    = status.host_total;
    report.host_available = status.host_available;
    report.landing_slots = status.landing_slots;
    return report;
}

// The switch's file form: the placement file resolved to the request (its vram set pinned, the
// pool sized to it, its order the refill's). Throws what parse_expert_placement_file throws.
[[nodiscard]] inline ExpertPlacementRequest expert_placement_request_from_file(const std::filesystem::path& path,
                                                                               std::uint32_t layers,
                                                                               std::uint32_t experts) {
    const ExpertPlacementSpec spec = parse_expert_placement_file(path, layers, experts);
    ExpertPlacementRequest request;
    request.frame_target = static_cast<std::uint32_t>(spec.vram.size());
    request.pinned       = spec.vram;
    request.order        = spec.order;
    request.has_ram_target = true;
    request.ram_target    = static_cast<std::uint32_t>(spec.ram.size());
    return request;
}

// A dry run of a counts switch: an --experts spec's tiers over the live placement and what
// applying it would take, computed from the placement's own numbers and applied to nothing.
[[nodiscard]] inline ExpertPlacementPreview preview_expert_placement(const std::string& experts_spec,
                                                                     const ExpertPlacementReport& now) {
    const runtime::ExpertsSpec spec  = runtime::parse_experts_spec(experts_spec);
    const auto [vram, ram]           = runtime::experts_spec_counts(spec, now.total_experts);
    const std::uint64_t frames_bytes = vram * now.frame_bytes;
    const std::uint64_t available    = now.vram_free + now.pool_bytes;
    return {.total_experts     = now.total_experts,
            .vram_experts      = vram,
            .ram_experts       = ram,
            .ssd_experts       = now.total_experts - vram - ram,
            .vram_expert_bytes = frames_bytes,
            .vram_free_after   = available > frames_bytes ? available - frames_bytes : 0,
            .ram_expert_bytes  = ram * now.expert_record_bytes,
            .ram_tier_slots    = now.tier_slots,
            .fits              = frames_bytes <= available,
            .ram_fits          = ram * now.expert_record_bytes <= now.host_available};
}

} // namespace infernix::models::offload
