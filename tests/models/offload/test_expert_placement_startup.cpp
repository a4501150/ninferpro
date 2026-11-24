// The startup expert-placement resolution (models/offload/expert_placement_startup.h): the
// --experts / --expert-placement conflict, the file and counts forms' targets, the resident
// probe against the family's device-plan hook, and the RAM ledger's tier caps. Host only.

#include "models/offload/expert_placement_startup.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::models::offload;

namespace {

constexpr std::uint64_t kGiB = 1ULL << 30;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// A family of 48 layers of 512 experts, one 2.64 MiB record each, and a probe that fits
// `fit_frames` frames beside the fixed allocations.
struct Fixture {
    std::uint32_t fit_frames = 24'576;
    bool fits = true;
    std::uint32_t max_frames = 48 * 512 - 1;
    std::vector<std::uint32_t> counts, ranked;
    bool calibrated = false;
    std::vector<DiagnosticLevel> reported;

    StartupPlacement resolve(std::string spec = {}, const std::filesystem::path& file = {},
                             std::optional<std::uint64_t> ram_bytes = {}) {
        StartupPlacementInputs in;
        in.layers = 48;
        in.experts = 512;
        in.stride = 2'764'800;
        in.expert_banks = 24'576 * in.stride;
        in.experts_spec = spec;
        in.expert_placement = file;
        in.expert_ram_bytes = ram_bytes;
        in.lock_overhead = 0.0025;
        in.load_ranking = [this](std::vector<std::uint32_t>& counts, std::vector<std::uint32_t>& ranked) {
            counts = this->counts;
            ranked = this->ranked;
            return calibrated;
        };
        in.probe = [this](bool) -> StartupPlacementInputs::Fit {
            return {.frame_bytes = fits ? static_cast<std::int64_t>(8 * kGiB) : -1,
                    .frames = fit_frames, .max_frames = max_frames, .fixed_bytes = 5 * kGiB};
        };
        in.report = [this](DiagnosticLevel level, const std::string&) { reported.push_back(level); };
        return resolve_startup_placement(in);
    }
};

// The tier's cap for `resident_slots` RAM residents: them plus the 1536 list slots, with the
// lock overhead.
std::uint64_t tier_cap(std::uint64_t resident_slots) {
    const std::uint64_t slots = resident_slots + 1536;
    return slots * 2'764'800 + static_cast<std::uint64_t>(static_cast<double>(slots * 2'764'800) * 0.0025);
}

void test_conflict() {
    Fixture f;
    bool threw = false;
    try {
        f.resolve("vram=60%,ram=25%", {}, 32 * kGiB);
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("--expert-ram-mib sizes the RAM tier by bytes") != std::string::npos;
    }
    require(threw, "--expert-ram-mib with a spec or file is refused");
}

void test_counts_form() {
    Fixture f;
    f.counts.assign(48 * 512, 3);
    const auto out = f.resolve("vram=60%,ram=25%");
    require(out.placed && !out.resident && !out.all_in_vram, "a shares spec is placed");
    require(out.spec.vram.size() == 14'746 && out.spec.ram.size() == 6'144,
            "the shares round by largest remainder to the key count");
    require(out.frame_target == out.spec.vram.size() && out.frame_ceiling == out.frame_target,
            "the counts form names the target and the ceiling");
    require(out.pinned_keys.empty(), "the counts form pins nothing (the pools stay adaptive)");
    require(out.tier_order.size() == 48 * 512, "the counts form names the whole fill order");
    require(out.tier_expert_cap && *out.tier_expert_cap == tier_cap(out.spec.ram.size()),
            "the tier cap is the RAM set plus the lists with the lock overhead");
    require(out.origin.find("vram=60%,ram=25% by the ranking") == 0, "the origin names the spec and the ranking");
    require(f.reported.size() == 1, "one report line");
}

// The counts form ranks by the hook's ranking and names it in the origin.
void test_counts_ranking_origin() {
    Fixture f;
    f.calibrated = true;
    const auto out = f.resolve("vram=14745,ram=6144");
    require(out.origin.find("by the calibrated order") != std::string::npos,
            "a calibrated ranking is named in the origin");
}

// The resident probe: the family's device plan holds every expert beside the fully-mapped KV
// pool, so the placement is resident and the tier holds only its floor.
void test_resident_probe() {
    Fixture f;
    const auto out = f.resolve();
    require(out.resident && !out.placed, "a fitting probe is resident");
    require(out.frame_target == f.max_frames, "the resident target is the plan's frame limit");
    require(out.tier_expert_cap && *out.tier_expert_cap == tier_cap(0),
            "the resident placement caps the tier at its floor");
    require(out.clear_lean_auto_cap, "the resident cap is not the lean default");
    require(f.reported.size() == 1, "one report line");
}

// A probe that does not fit: vram=all fails the start; the default falls back to the upstream
// maximal tier and names nothing.
void test_resident_probe_refusals() {
    Fixture f;
    f.fits = false;
    bool threw = false;
    try {
        f.resolve("vram=all");
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("--experts vram=all: every routed expert does not fit") !=
                std::string::npos;
    }
    require(threw, "vram=all without room fails the start");
    const auto out = f.resolve();
    require(!out.resident && !out.placed && out.frame_target == 0 && !out.tier_expert_cap,
            "a default start without room keeps the upstream tier");
}

void test_file_form() {
    Fixture f;
    const std::filesystem::path file = "expert_placement_startup_test.csv";
    {
        std::ofstream out(file);
        out << "# infernix expert placement v1\n0,5,vram\n3,212,ram\n";
    }
    const auto out = f.resolve({}, file);
    std::error_code ec;
    std::filesystem::remove(file, ec);
    require(out.placed, "a file is placed");
    require(out.pinned_keys == out.spec.vram && out.pinned_keys.size() == 1, "a file's vram set pins");
    require(out.frame_target == 1 && out.frame_ceiling == 1, "a file names the target and the ceiling");
    require(out.tier_expert_cap && *out.tier_expert_cap == tier_cap(1), "the file's RAM set caps the tier");
    // The placed form probes the fit of its vram set: a target beyond the fit fails the start.
    Fixture tight;
    tight.fit_frames = 24'062; // the fit's margin leaves 24'574, one short of the 24'575 asked
    tight.counts.assign(48 * 512, 1);
    bool threw = false;
    try {
        tight.resolve("vram=24575,ram=0");
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("frames fit beside the fixed allocations") != std::string::npos;
    }
    require(threw, "a vram set beyond the fit fails the start");
}

} // namespace

int main() {
    try {
        test_conflict();
        test_counts_form();
        test_counts_ranking_origin();
        test_resident_probe();
        test_resident_probe_refusals();
        test_file_form();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("offload expert placement startup checks passed\n");
    return 0;
}
