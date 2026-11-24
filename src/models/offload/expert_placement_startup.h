#pragma once

// The startup expert-placement resolution an offloaded-MoE family's instance runs before its
// Program is built (docs/maintainer/moe-offload-design.md §4.8, lifted from the first family's
// instance): the --experts spec or --expert-placement file resolves to the frame-pool target and
// ceiling, the pinned key set and the tier fill order, and the RAM ledger's expert cap for the
// named RAM set -- or the resident placement, when every routed expert fits in VRAM beside the
// fully-mapped KV pool. The family's own probes (its device plan) and ranking (its saved state
// and calibrated order) arrive as hooks; the family applies the result to its options and ledger.

#include "models/offload/expert_placement.h"
#include "runtime/diagnostics.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace infernix::models::offload {

struct StartupPlacementInputs {
    // The routed keys: layers * experts, and one expert record's bytes (the sizing unit of
    // frames and slots; 0: the model has no expert records).
    std::uint32_t layers = 0;
    std::uint32_t experts = 0;
    std::uint64_t stride = 0;
    // The experts' RAM bytes the ledger plans for (the resident probe skips when 0).
    std::uint64_t expert_banks = 0;
    // --experts SPEC (empty: none) and --expert-placement FILE (empty: none). Both empty
    // resolves to the resident probe or the family's default.
    std::string experts_spec;
    std::filesystem::path expert_placement;
    // --expert-ram-mib (bytes): the RAM tier is sized by bytes; it cannot be combined with a
    // spec or file (the resolver throws).
    std::optional<std::uint64_t> expert_ram_bytes;
    // The RAM ledger's lock-overhead fraction (the resident cap accounts for it).
    double lock_overhead = 0.0025;
    // The ranking the spec's counts resolve by: fills `counts` and `ranked` (the family's saved
    // session's state, else its calibrated order) and returns whether the calibrated order was
    // used (the origin line names it).
    std::function<bool(std::vector<std::uint32_t>& counts, std::vector<std::uint32_t>& ranked)> load_ranking;
    // The VRAM fit probe: the family's device plan beside the fixed allocations, with the
    // display headroom and the graph reserve. `resident`: probe with the KV pool fully mapped
    // (no elasticity, no frame target); else probe with the placement's target as set so far.
    struct Fit {
        std::int64_t frame_bytes = 0;   // what the frames may take (negative: the fixed do not fit)
        std::uint32_t frames = 0;      // the frames that fit
        std::uint32_t max_frames = 0;  // the residency's frame limit
        std::uint64_t fixed_bytes = 0; // the fixed allocations the probe planned
    };
    std::function<Fit(bool resident)> probe;
    // The diagnostic sink for the resolution's log lines.
    std::function<void(DiagnosticLevel level, const std::string& text)> report;
};

// The resolution's result; the family applies the fields to its ProgramOptions and RAM ledger
// demand (a field left at its default leaves the family's as-is).
struct StartupPlacement {
    // An explicit --experts spec or --expert-placement file was named.
    bool placed = false;
    // The resident placement: every routed expert in the frame pool, nothing lends or evicts.
    bool resident = false;
    // The spec named vram=all (or 100%): the resident probe holds it or the start fails.
    bool all_in_vram = false;
    // The resolved spec (a file's, or the counts form's ranking); empty when neither was named.
    ExpertPlacementSpec spec;
    // The origin for the startup log ("--experts ...", the file's path, ...).
    std::string origin;
    // The frame-pool target and the monitor's ceiling (an explicit placement names both).
    std::uint32_t frame_target = 0;
    std::uint32_t frame_ceiling = 0;
    // The placement's vram set (a file names exact keys: pinned; a counts spec names sizes
    // only: the pools stay adaptive within the count).
    std::vector<std::uint32_t> pinned_keys;
    // The placement's fill order (empty: the saved ranking's own).
    std::vector<std::uint32_t> tier_order;
    // The RAM ledger's expert_cap for the named RAM set (the resident placement's floor, or the
    // tier lists beside the named residents); empty: the family's cap stands.
    std::optional<std::uint64_t> tier_expert_cap;
    // The resident cap is not the RAM-lean default (the ledger's lean_auto_cap clears).
    bool clear_lean_auto_cap = false;
};

[[nodiscard]] StartupPlacement resolve_startup_placement(const StartupPlacementInputs& inputs);

} // namespace infernix::models::offload
