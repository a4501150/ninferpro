#pragma once

// Explicit expert placement for Qwen3.8-Flash-Next (design §19.3.7): which routed experts the VRAM
// frame pool pins, which the RAM tier holds, and the placement file that assigns (layer, expert)
// pairs to tiers. The startup resolution (runtime/engine/qwen4_exp_instance.cpp) turns a
// placement into the frame-pool target, the RAM tier's slot count and the pinned key set; the
// Program's warm start and tier fill realize it, and the /admin/expert-placement endpoint can
// apply a new frame target at runtime.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::qwen4_exp {

// A resolved placement over the model's keys (layer * experts + expert).
struct ExpertPlacementSpec {
    // Pinned in the VRAM frame pool: seeded first by the warm start and never eviction victims,
    // so exactly this set stays resident (the pool is sized to hold it).
    std::vector<std::uint32_t> vram;
    // Held in RAM slots by the SSD tier's warm fill, best first.
    std::vector<std::uint32_t> ram;
    // Every key, the fill order the warm start and the tier follow: `vram` keys first, then
    // `ram`, then the rest in the given ranking (or spread across the layers without one).
    std::vector<std::uint32_t> order;
    // The origin of the ranking, for the startup log.
    bool from_saved_state = false;
};

// The counts form (--vram-experts N, --ram-experts M): the N most-used keys pinned in VRAM, the
// next M in RAM, ranked by `counts` (a saved session's use counts; empty ranks by `ranked`, the
// saved order; both empty spreads across the layers). `vram_count` 0 means auto (the sizing
// function's frame target); `ram_count` 0 means the RAM ledger's share.
[[nodiscard]] ExpertPlacementSpec placement_from_counts(std::uint32_t vram_count, std::uint32_t ram_count,
                                                         std::uint32_t layers, std::uint32_t experts,
                                                         std::span<const std::uint32_t> counts,
                                                         std::span<const std::uint32_t> ranked);

// The placement file (docs/qwen3_8-flash-next.md): CSV, one row per (layer, expert) pair,
//
//     # infernix expert placement v1
//     # tier is vram, ram or ssd; unlisted experts are ssd
//     0,5,vram
//     3,212,ram
//
// `vram` rows are the pinned set, `ram` rows the RAM tier's, and everything else (listed ssd or
// unlisted) is read from the artifact when used. Throws std::runtime_error with the file and line
// on a malformed row.
[[nodiscard]] ExpertPlacementSpec parse_expert_placement_file(const std::filesystem::path& path, std::uint32_t layers,
                                                               std::uint32_t experts);

// Writes the current placement in the load format: every (layer, expert) as
// `layer,expert,tier[,uses]`, tiers from the frame table (>= 0: vram), the RAM tier's host copies
// (`ram_keys`: resident or shadow slots) and the rest ssd, rows ordered by use count (the saved
// state's), best first, so loading the file reproduces the ranking too. `counts` may be empty
// (rows then keep the spread order). Returns false when the file cannot be written; `error` why.
[[nodiscard]] bool write_expert_placement_file(const std::filesystem::path& path, std::uint32_t layers,
                                               std::uint32_t experts, std::span<const std::int32_t> frame_table,
                                               std::span<const std::uint32_t> ram_keys,
                                               std::span<const std::uint32_t> counts, std::string* error = nullptr);

// The resident refill's read batch (the Program's placement grow, design §19.3.7): up to `batch`
// keys of `order` from `at` that are still without a frame (`frame_table`: < 0 not resident),
// advancing `at` past every key examined. Keys already resident are skipped, not re-read, so a
// refill that resumes after a partial fill reads only what the new frames still lack.
[[nodiscard]] std::vector<std::uint32_t> placement_refill_batch(std::span<const std::uint32_t> order,
                                                                std::size_t& at, std::uint32_t batch,
                                                                std::span<const std::int32_t> frame_table);

// The runtime switch (/admin/expert-placement): a new frame-pool target with its pinned set and
// the RAM tier's resident-slot target. `has_ram_target` names the tier: with it 0, the engine holds
// no expert bytes in host RAM (the tier is released once every expert is back in VRAM); without
// it, the tier keeps its current size.
struct ExpertPlacementRequest {
    std::uint32_t frame_target = 0; // frames the pool holds once applied
    std::vector<std::uint32_t> pinned; // keys never evicted (the placement's vram set)
    std::vector<std::uint32_t> order;  // the warm-start order the refill follows (may be empty)
    bool has_ram_target = false;      // the switch names the RAM tier's size
    std::uint32_t ram_target = 0;     // resident slots the RAM tier holds once applied
};

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

} // namespace infernix::models::qwen4_exp
