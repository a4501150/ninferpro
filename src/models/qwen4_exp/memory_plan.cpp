#include "models/qwen4_exp/memory_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace infernix::models::qwen4_exp {
namespace {

std::uint64_t with_lock_overhead(std::uint64_t pinned, double fraction) {
    return pinned + static_cast<std::uint64_t>(std::ceil(static_cast<double>(pinned) * fraction));
}

std::uint64_t saturating_sub(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kLedgerGiB); }

} // namespace

HostMemoryLedger plan_host_memory(const HostMemorySnapshot& snapshot, const HostMemoryDemand& demand) {
    HostMemoryLedger out;
    out.snapshot = snapshot;
    out.demand   = demand;
    const std::uint64_t pinned_other =
        with_lock_overhead(demand.other_pinned + demand.later_pinned + demand.prefix_cache, demand.lock_overhead);
    out.planned   = pinned_other + demand.pageable + demand.load_staging + demand.process_growth;
    out.full_need = with_lock_overhead(demand.expert_banks, demand.lock_overhead) + demand.full_margin;
    // Physical memory: the reserve and everything planned come first.
    std::uint64_t room = saturating_sub(snapshot.available_physical, demand.reserve + out.planned);
    // Commit: every pin and the process's growth are committed too, and the reserve must survive.
    const std::uint64_t commit_room = saturating_sub(snapshot.available_commit, demand.reserve + out.planned);
    if (commit_room < room) {
        room               = commit_room;
        out.commit_limited = true;
    }
    out.expert_ram = std::min(room, out.full_need);
    if (demand.expert_cap) {
        out.cap_too_large = *demand.expert_cap > room && *demand.expert_cap < out.full_need;
        if (*demand.expert_cap < out.expert_ram) {
            out.expert_ram  = *demand.expert_cap;
            out.capped      = true;
            out.auto_capped = demand.lean_auto_cap;
        }
    }
    // Page-locking: the other pins come first; the experts pin what is left of the room, and the
    // rest of their RAM stays pageable (resident records the CPU serves).
    std::uint64_t pin_left =
        demand.pin_room ? saturating_sub(*demand.pin_room, pinned_other) : std::numeric_limits<std::uint64_t>::max();
    if (demand.expert_pin_cap && *demand.expert_pin_cap < pin_left) {
        pin_left       = *demand.expert_pin_cap;
        out.pin_capped = true;
    }
    const bool banks_pin = with_lock_overhead(demand.expert_banks, demand.lock_overhead) <= pin_left;
    out.placement = out.expert_ram >= out.full_need && banks_pin ? ExpertPlacement::Full : ExpertPlacement::Tier;
    out.expert_pinned   = std::min(out.expert_ram, pin_left);
    out.expert_pageable = out.expert_ram - out.expert_pinned;
    out.pin_limited     = out.placement == ExpertPlacement::Tier && out.expert_pageable > 0;
    out.pin_capped      = out.pin_capped && out.pin_limited;
    if (out.placement == ExpertPlacement::Full || out.capped) { out.commit_limited = false; }
    return out;
}

std::optional<std::uint64_t> pin_room(const VramSnapshot& snapshot) {
    if (snapshot.shared_system == 0) { return std::nullopt; }
    return saturating_sub(snapshot.shared_system, snapshot.nonlocal_usage + kPinRoomMarginBytes);
}

std::string HostMemoryLedger::describe() const {
    char pins[96] = "pin room unlimited";
    if (demand.pin_room) { std::snprintf(pins, sizeof(pins), "pin room %.2f", gib(*demand.pin_room)); }
    char line[768];
    std::snprintf(line, sizeof(line),
                  "RAM ledger: available %.2f GiB of %.2f (commit %.2f); %s; reserve %.2f; other pins %.2f; later "
                  "pins %.2f; prefix cache %.2f; caches %.2f; staging %.2f; growth %.2f -> experts %.2f GiB of %.2f "
                  "needed (banks %.2f + margin %.2f; pinned %.2f, resident %.2f)%s%s%s; %s",
                  gib(snapshot.available_physical), gib(snapshot.total_physical), gib(snapshot.available_commit), pins,
                  gib(demand.reserve), gib(demand.other_pinned), gib(demand.later_pinned), gib(demand.prefix_cache),
                  gib(demand.pageable), gib(demand.load_staging), gib(demand.process_growth), gib(expert_ram),
                  gib(full_need), gib(demand.expert_banks), gib(demand.full_margin), gib(expert_pinned),
                  gib(expert_pageable), commit_limited ? ", limited by the page file" : "",
                  capped ? (auto_capped ? ", capped by the RAM-lean default" : ", capped by --expert-ram-mib") : "",
                  pin_capped ? ", pinned share capped by --expert-pin-mib" : pin_limited ? ", pinned share limited by the pin room" : "",
                  placement == ExpertPlacement::Full ? "full mode (every expert in RAM)"
                                                     : "SSD tier (experts not in RAM are read from the artifact)");
    return line;
}

std::string HostMemoryLedger::summary() const {
    char line[480];
    if (placement == ExpertPlacement::Full) {
        std::snprintf(line, sizeof(line), "RAM: %.1f of %.1f GiB free; every expert fits in RAM (%.1f GiB)",
                      gib(snapshot.available_physical), gib(snapshot.total_physical), gib(demand.expert_banks));
        return line;
    }
    char split[200] = "";
    if (pin_limited) {
        std::snprintf(split, sizeof(split), "; %s, so %.1f GiB are page-locked for the GPU and %.1f GiB kept in RAM for the CPU",
                      pin_capped ? "--expert-pin-mib limits page-locking"
                                 : "Windows lets the GPU page-lock only part of RAM",
                      gib(expert_pinned), gib(expert_pageable));
    }
    if (expert_ram >= full_need) {
        std::snprintf(line, sizeof(line), "RAM: %.1f of %.1f GiB free; every expert fits in RAM (%.1f GiB)%s",
                      gib(snapshot.available_physical), gib(snapshot.total_physical), gib(demand.expert_banks), split);
    } else {
        std::snprintf(line, sizeof(line),
                      "RAM: %.1f of %.1f GiB free; %.1f of the experts' %.1f GiB fit in RAM%s%s, the rest are read "
                      "from the model files on demand (slower)",
                      gib(snapshot.available_physical), gib(snapshot.total_physical), gib(expert_ram),
                      gib(demand.expert_banks),
                      capped ? (auto_capped ? " (the RAM-lean default)" : " (--expert-ram-mib)") : "", split);
    }
    return line;
}

std::uint32_t graph_bound(std::uint32_t lanes, std::uint32_t max_width, std::uint32_t mtp_draft_tokens,
                          bool sampled_drafts, std::uint32_t dflash2_draft_tokens) {
    std::uint32_t graphs = lanes;
    if (max_width > 1) { graphs += lanes * max_width; }
    if (mtp_draft_tokens > 0) {
        graphs += lanes * max_width + lanes * mtp_draft_tokens * (sampled_drafts ? 2 : 1);
    }
    if (dflash2_draft_tokens > 0) { graphs += lanes + lanes * max_width; }
    return graphs;
}

std::uint64_t display_headroom(DisplayState display, bool elastic, std::optional<std::uint64_t> requested) {
    if (requested) { return *requested; }
    if (display == DisplayState::Headless) { return 256 * kLedgerMiB; }
    return (elastic ? 512 : 1024) * kLedgerMiB;
}

std::uint64_t internal_reserve(std::uint32_t graphs, std::uint64_t graph_bytes, bool elastic) {
    const std::uint64_t counted = elastic ? std::min<std::uint32_t>(graphs, 32) : graphs;
    return std::max(256 * kLedgerMiB, 128 * kLedgerMiB + graph_bytes * counted);
}

VramSizing size_expert_frames(const VramSnapshot& snapshot, const VramDemand& demand) {
    VramSizing out;
    out.snapshot = snapshot;
    out.demand   = demand;
    out.headroom = display_headroom(snapshot.display, demand.elastic, demand.headroom);
    out.reserve  = internal_reserve(demand.graphs, demand.graph_bytes, demand.elastic);
    const auto signed_bytes = [](std::uint64_t v) { return static_cast<std::int64_t>(v); };
    std::int64_t room = signed_bytes(snapshot.device_free) - signed_bytes(demand.fixed) - signed_bytes(out.headroom);
    if (snapshot.has_budget) {
        const std::int64_t budget = signed_bytes(snapshot.local_budget) - signed_bytes(snapshot.local_usage) -
                                    signed_bytes(demand.fixed) - signed_bytes(kVramBudgetMarginBytes);
        if (budget < room) {
            room               = budget;
            out.budget_limited = true;
        }
    }
    out.frame_bytes = room - signed_bytes(out.reserve);
    if (out.frame_bytes > 0 && demand.frame_bytes > 0) {
        const std::uint64_t frames = static_cast<std::uint64_t>(out.frame_bytes) / demand.frame_bytes;
        out.frames = static_cast<std::uint32_t>(std::min<std::uint64_t>(frames, demand.max_frames));
    }
    return out;
}

std::uint32_t frame_target(const VramDemand& demand, std::uint64_t chunk_bytes, const VramSnapshot& snapshot,
                           std::uint64_t pool_bytes, std::uint64_t usage_at_sizing, std::uint64_t pool_at_sizing) {
    const auto s = [](std::uint64_t v) { return static_cast<std::int64_t>(v); };
    const std::uint64_t total = internal_reserve(demand.graphs, demand.graph_bytes, true);
    // The unspent reserve: with an OS budget, this process's growth since sizing that is not the
    // pool's has spent it (the budget usage counts CUDA).
    std::uint64_t reserve_left = total;
    if (snapshot.has_budget) {
        const auto grown = s(snapshot.local_usage) - s(usage_at_sizing) - (s(pool_bytes) - s(pool_at_sizing));
        const std::uint64_t used = grown > 0 ? static_cast<std::uint64_t>(grown) : 0;
        reserve_left            = used >= total ? 0 : total - used;
    }
    std::int64_t room = s(snapshot.device_free) + s(pool_bytes) -
                       s(display_headroom(snapshot.display, true, demand.headroom));
    if (snapshot.has_budget) {
        room = std::min(room, s(snapshot.local_budget) - s(snapshot.local_usage) + s(pool_bytes) -
                                  s(kVramBudgetMarginBytes));
    }
    const std::int64_t bytes = room - s(reserve_left);
    if (bytes <= 0 || chunk_bytes == 0 || demand.frame_bytes == 0) { return 0; }
    // Whole chunks: the pool maps and releases in chunks.
    const std::uint64_t chunks = static_cast<std::uint64_t>(bytes) / chunk_bytes;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(chunks * chunk_bytes / demand.frame_bytes, demand.max_frames));
}

std::string VramSizing::describe() const {
    const auto mib = [](std::uint64_t bytes) { return static_cast<unsigned long long>(bytes / kLedgerMiB); };
    const char* display = snapshot.display == DisplayState::Headless   ? "no display"
                          : snapshot.display == DisplayState::Attached ? "display attached"
                                                                       : "display unknown (treated as attached)";
    char budget[96] = "";
    if (snapshot.has_budget) {
        std::snprintf(budget, sizeof(budget), "; OS budget %llu MiB, used %llu", mib(snapshot.local_budget),
                      mib(snapshot.local_usage));
    }
    char line[512];
    std::snprintf(line, sizeof(line),
                  "VRAM sizing: free %llu of %llu MiB; %s%s; to allocate %llu; reserve %llu (%u graphs); headroom "
                  "%llu (%s) -> expert frames %u (%.2f GiB)%s%s",
                  mib(snapshot.device_free), mib(snapshot.device_total), display, budget, mib(demand.fixed),
                  mib(reserve), demand.graphs, mib(headroom), demand.headroom ? "set" : "auto", frames,
                  gib(static_cast<std::uint64_t>(frames) * demand.frame_bytes),
                  budget_limited ? ", limited by the OS budget" : "",
                  frame_bytes < 0 ? "; the fixed allocations do not fit" : "");
    return line;
}

} // namespace infernix::models::qwen4_exp
