#pragma once

// The --experts grammar, independent of any model: serve and the Engine refuse a malformed spec
// before an artifact is opened, and a family with an expert cache resolves it to tier counts.

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace infernix::runtime {

// The --experts spec: the exclusive three-tier placement of the routed experts, as shares that
// add up to the whole or as counts (the mechanism; which numbers to pick is client policy):
//
//     --experts vram=60%,ram=25%,ssd=15%     shares over every routed expert
//     --experts vram=14745,ram=6144          counts; the ssd tier takes the rest
//     --experts vram=100%  /  vram=all       every expert in VRAM, none in host RAM
//
// The tiers may appear in any order. `ssd` may be omitted (it takes what the others leave);
// `ram` may be omitted (it is then 0). `vram` is required, must not be 0, and `all`/`100%`
// excludes the other tiers. Shares and counts cannot be mixed in one spec. Throws
// std::invalid_argument with the reason.
struct ExpertsSpec {
    bool all = false;                        // vram=all or vram=100%
    std::optional<std::uint64_t> vram_count; // set when vram=N
    std::optional<std::uint64_t> ram_count;
    std::optional<std::uint64_t> ssd_count;
    std::optional<std::uint64_t> vram_permille; // percent * 10, set when vram=N%
    std::optional<std::uint64_t> ram_permille;
    std::optional<std::uint64_t> ssd_permille;
};

[[nodiscard]] ExpertsSpec parse_experts_spec(const std::string& text);

// The spec's per-tier counts over `total` routed experts: shares round by largest remainder
// (ties to vram, then ram) so the three counts add up exactly to `total`; counts pass through
// (the ssd tier takes `total` - vram - ram). Throws std::invalid_argument when the spec is not
// consistent with `total` (vram beyond the whole, a negative remainder).
[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> experts_spec_counts(const ExpertsSpec& spec,
                                                                          std::uint64_t total);

} // namespace infernix::runtime
