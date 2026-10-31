#pragma once

// The bank geometry every offload component is parameterized by (glm5_3-flash-design.md §8.2 and
// deepseek_v4_1-flash-design.md §8.2): a family composes it from its own config and hands it to
// the placement, cache, residency controller and tiers. No base class, no virtual interface —
// each Program owns its instances and calls them at its own round boundaries. The record format's
// stride stays a constructor argument (a bank's records carry it); the frame side entry and the
// H/I inner dimensions arrive with the families that need them.

#include <cstdint>

namespace infernix::models::offload {

struct ExpertBankGeometry {
    // The routed layers: one pinned bank of expert records each.
    std::uint32_t banks = 0;
    // Experts per bank.
    std::uint32_t experts_per_bank = 0;
    // Routed experts one token selects per layer (the route log's width).
    std::uint32_t top_k = 0;
};

} // namespace infernix::models::offload
