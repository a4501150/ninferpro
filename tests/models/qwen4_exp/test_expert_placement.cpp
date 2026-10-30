// Explicit expert placement (design §19.3.7; --vram-experts / --ram-experts / --expert-placement):
// the placement file's parse and errors, the counts ranking, the export round trip, and the LFRU
// pinning that keeps a placed set resident (a pinned key is never a victim; admissions stop while
// only they remain; a shrink they block is refused).

#include "models/offload/expert_placement.h"
#include "models/offload/expert_cache.h"
#include "models/offload/host_tier.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::models::offload;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

constexpr std::uint32_t kLayers = 4, kExperts = 8, kKeys = kLayers * kExperts;

std::filesystem::path temp_file(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("infernix-placement-" + name + ".csv");
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::trunc);
    out << text;
}

void test_parse() {
    const auto path = temp_file("parse");
    write_text(path, "# infernix expert placement v1\n"
                     "# tier is vram, ram or ssd\n"
                     "0,1,vram\n"
                     "1,2,ram\n"
                     "2,3,ssd\n"
                     "\n");
    const ExpertPlacementSpec spec = parse_expert_placement_file(path, kLayers, kExperts);
    require(spec.vram == std::vector<std::uint32_t>{0 * kExperts + 1}, "the vram row parses");
    require(spec.ram == std::vector<std::uint32_t>{1 * kExperts + 2}, "the ram row parses");
    require(spec.order.size() == kKeys, "the order covers every key");
    require(spec.order.front() == spec.vram.front(), "the vram set leads the order");
    require(spec.order[1] == spec.ram.front(), "the ram set follows it");
}

void test_parse_errors() {
    struct Case {
        std::string text;
        std::string message;
    };
    const std::vector<Case> cases = {
        {"0,1,disk\n", "the tier"},
        {"4,0,vram\n", "the layer"},  // kLayers == 4
        {"0,8,vram\n", "the expert"}, // kExperts == 8
        {"0,1,vram\n0,1,ram\n", "twice"},
        {"0,1\n", "not `layer,expert,tier`"},
        {"0,1,vram,7,extra\n", "more than four columns"},
    };
    for (const Case& test : cases) {
        const auto path = temp_file("error");
        write_text(path, test.text);
        bool threw = false;
        try {
            (void)parse_expert_placement_file(path, kLayers, kExperts);
        } catch (const std::runtime_error&) { threw = true; }
        require(threw, "a malformed placement file (" + test.text + ") throws");
    }
    const auto empty = temp_file("empty");
    write_text(empty, "# only comments\n");
    bool threw = false;
    try {
        (void)parse_expert_placement_file(empty, kLayers, kExperts);
    } catch (const std::runtime_error&) { threw = true; }
    require(threw, "a file assigning no expert throws");
}

void test_counts_ranking() {
    // The saved ranking leads, then the counts (most used first, ties by spread), then the rest.
    std::vector<std::uint32_t> ranked{5, 9};
    std::vector<std::uint32_t> counts(kKeys, 0);
    counts[3] = 7;
    counts[2] = 3;
    const ExpertPlacementSpec spec = placement_from_counts(2, 1, kLayers, kExperts, counts, ranked);
    require(spec.vram == std::vector<std::uint32_t>({5, 9}), "the top 2 of the ranking pin in VRAM");
    require(spec.ram == std::vector<std::uint32_t>({3}), "the next key by count holds RAM");
    require(spec.order[3] == 2, "a used key precedes the never-used spread tail");
    require(spec.order.size() == kKeys, "the order covers every key");
    // Counts alone rank: the calibrated order case.
    const ExpertPlacementSpec spread = placement_from_counts(1, 1, kLayers, kExperts, {}, {});
    require(spread.vram.size() == 1 && spread.ram.size() == 1, "counts without a ranking still split");
    require(spread.order.size() == kKeys, "the spread order covers every key");
}

void test_round_trip() {
    // The export writes the frame table and the tier's host copies; parsing it back reproduces the
    // tiers and the ranking order.
    std::vector<std::int32_t> table(kKeys, -1);
    table[0] = 0;
    table[1] = 1;
    std::vector<std::uint32_t> ram{2, 3};
    std::vector<std::uint32_t> counts(kKeys, 0);
    counts[3] = 10;
    counts[1] = 5;
    const auto path = temp_file("roundtrip");
    std::string error;
    require(write_expert_placement_file(path, kLayers, kExperts, table, ram, counts, &error),
            "the export writes: " + error);
    const ExpertPlacementSpec spec = parse_expert_placement_file(path, kLayers, kExperts);
    require(spec.vram.size() == 2, "both frame-table residents parse back as vram");
    require(spec.ram.size() == 2, "both host copies parse back as ram");
    require(spec.vram.front() == 1 && spec.vram[1] == 0, "the export orders rows best first");
    require(spec.order.front() == spec.vram.front(), "the order keeps the ranking");
}

void test_lfru_pinning() {
    // Pinned keys are never victims: a full policy with every resident pinned refuses admissions
    // (the landing is served, not adopted), and a shrink they block is refused rather than broken.
    expert_cache::LfruPolicy policy(kKeys, 4, 0);
    std::vector<std::uint32_t> resident{0, 1, 2, 3};
    std::vector<std::uint32_t> counts(kKeys, 1);
    policy.seed(resident, counts);
    policy.pin_keys(std::span<const std::uint32_t>(resident.data(), 2)); // keys 0 and 1

    // A miss that needs a victim evicts an unpinned resident, never a pinned one.
    expert_cache::LfruPolicy::Step step;
    std::vector<std::uint32_t> group{4};
    policy.step(group, step);
    require(step.admitted.size() == 1, "a miss with an unpinned victim is admitted");
    require(step.victims.size() == 1 && step.victims[0] != 0 && step.victims[0] != 1,
            "the victim is unpinned");

    // With only pinned residents left to evict, admission stops: the miss stays a miss.
    // Evict the remaining unpinned residents by churning misses through them.
    for (std::uint32_t key = 5; key < kKeys; ++key) {
        std::vector<std::uint32_t> miss{key};
        policy.step(miss, step);
    }
    // Now residents are the two pinned keys plus whatever the churn admitted; force the pinned-only
    // state by shrinking the capacity to the pinned count, which must refuse.
    std::vector<std::uint32_t> victims;
    bool refused = false;
    try {
        policy.set_capacity(1, victims); // cannot hold both pinned keys
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a shrink below the pinned count is refused");
    require(policy.resident(0) && policy.resident(1), "the pinned keys survived the refusal");

    // A promote into a fully-pinned pool is refused, not thrown.
    expert_cache::LfruPolicy full(kKeys, 2, 0);
    const std::vector<std::uint32_t> full_resident{0, 1};
    full.seed(full_resident, counts);
    full.pin_keys(std::span<const std::uint32_t>(resident.data(), 2));
    std::optional<std::uint32_t> victim;
    require(!full.promote(7, {}, victim), "a promote into a pinned-full pool is refused");
    require(!victim.has_value(), "a refused promote names no victim");

    // Pinning replaces the set: the old keys evict again.
    policy.pin_keys(std::span<const std::uint32_t>(resident.data() + 2, 2)); // keys 2 and 3
    require(!policy.pinned(0) && policy.pinned(2), "pin_keys replaces the pinned set");
}

void test_placement_fits_shape() {
    // The spec's own invariants: every key appears once in the order; the sets are disjoint.
    std::vector<std::uint32_t> counts(kKeys, 0);
    counts[7] = 4;
    const ExpertPlacementSpec spec = placement_from_counts(3, 2, kLayers, kExperts, counts, {});
    std::vector<std::uint8_t> seen(kKeys, 0);
    for (const std::uint32_t key : spec.order) {
        require(key < kKeys && seen[key] == 0, "the order lists every key once");
        seen[key] = 1;
    }
    for (const std::uint32_t key : spec.vram) {
        require(std::find(spec.ram.begin(), spec.ram.end(), key) == spec.ram.end(), "the sets are disjoint");
    }
}


// The resident warm fill's window walk (program_impl.h warm_frames; design §19.3.7): the tier's
// slots hold a window of the fill order, the frames seed it, the shadows free, and the next
// window reads the following keys from the artifact. The cursor rule is the regression: it
// starts past prefill_tier's fill and advances by what each prefill really landed. A fixed span
// from the order's start re-reads keys an earlier window seeded (they have no host copy once
// released), ranks keys already in frames, seeds nothing new and stalls the walk after one
// window -- the walk that booted with 1,104 of 24,576 experts resident. Mirrored against a real
// HostTier, with a set standing in for the frames; warm_start skips keys already seeded.
void test_resident_fill_walk() {
    expert_cache::HostTier::Config config;
    config.num_keys = 1500;
    config.slots    = 97;  // fewer than any window span: every window is slot-bound
    config.ring = 32; config.prefetch = 4; config.demotion = 8; config.decode_demotions = 8;
    constexpr std::uint32_t window_span = 400;

    const auto walk = [&](bool fixed_cursor, std::uint32_t& seeded_out, std::uint32_t& reads_out,
                          std::uint32_t& windows_out) {
        expert_cache::HostTier tier(config);
        const std::uint32_t keys = config.num_keys;
        std::vector<std::uint32_t> order(keys);
        for (std::uint32_t k = 0; k < keys; ++k) { order[k] = k; }
        std::vector<std::uint8_t> in_frames(keys, 0);
        std::uint32_t reads = 0;
        // prefill_tier: the order's top, as many as the slots hold.
        for (std::uint32_t k = 0; k < keys && tier.has_free_resident(); ++k) {
            std::int32_t slot = -1;
            if (tier.prefill(order[k], slot)) { ++reads; }
        }
        std::size_t at = fixed_cursor ? static_cast<std::size_t>(reads) : 0;
        std::uint32_t seeded = 0, windows = 0;
        for (;;) {
            std::vector<std::uint32_t> ranked;
            for (const std::uint32_t key : order) {
                if (tier.host_copy(key)) { ranked.push_back(key); }
            }
            if (ranked.empty()) { break; }
            std::uint32_t loaded = 0;
            for (const std::uint32_t key : ranked) {
                if (in_frames[key] != 0) { continue; } // the policy skips what a frame already holds
                in_frames[key] = 1;
                tier.promotion_completed(key, true);  // the frame published: the slot is its shadow
                ++loaded;
            }
            ++windows;
            seeded += loaded;
            tier.release_shadows();
            if (loaded == 0 || at >= order.size() || !tier.has_free_resident()) { break; }
            const std::size_t end = std::min(order.size(), at + window_span);
            std::uint32_t landed = 0;
            for (std::size_t i = at; i < end && tier.has_free_resident(); ++i) {
                std::int32_t slot = -1;
                if (tier.prefill(order[i], slot)) { ++landed; ++reads; }
            }
            at = fixed_cursor ? at + landed : end; // the cursor rule under test
            if (landed == 0) { break; }
        }
        seeded_out = seeded; reads_out = reads; windows_out = windows;
    };

    std::uint32_t seeded = 0, reads = 0, windows = 0;
    walk(true, seeded, reads, windows);
    require(seeded == config.num_keys, "the fixed cursor seeds every expert");
    require(reads == config.num_keys, "every expert record is read exactly once");
    std::uint32_t stale_seeded = 0, stale_reads = 0, stale_windows = 0;
    walk(false, stale_seeded, stale_reads, stale_windows);
    require(stale_seeded < config.num_keys,
            "the pre-fix cursor (a fixed span from the order's start) stalls the walk");
    std::printf("  resident fill walk: %u experts over %u windows (%u reads); "
                "pre-fix rule stalled at %u experts\n",
                seeded, windows, reads, stale_seeded);
}

// The resident refill's read batch (placement_refill_batch; the placement grow's window walk):
// only keys still without a frame are read, the cursor advances past every key examined (skipped
// keys are not re-read on a later tick), the batch respects its cap, and a resumed walk (a
// resident prefix of the order) reads only what the new frames lack.
void test_refill_batch() {
    // 8 keys; 0, 2 and 5 already hold a frame.
    const std::vector<std::int32_t> table = {0, -1, 3, -1, -1, 7, -1, -1};
    const std::vector<std::uint32_t> order = {0, 1, 2, 3, 4, 5, 6, 7};
    std::size_t at = 0;
    std::vector<std::uint32_t> want = placement_refill_batch(order, at, 4, table);
    require((want == std::vector<std::uint32_t>{1, 3, 4, 6}),
            "the batch reads the non-resident keys, skipping the resident ones, up to its cap");
    require(at == 7, "the cursor advances past every key examined, resident or not");
    want = placement_refill_batch(order, at, 4, table);
    require((want == std::vector<std::uint32_t>{7}), "the last tick reads what remains");
    require(at == order.size(), "the cursor stops at the order's end");
    want = placement_refill_batch(order, at, 4, table);
    require(want.empty(), "an exhausted order reads nothing");
    // A resumed refill: the first three keys of the order became resident after a partial fill.
    const std::vector<std::int32_t> grown = {2, 4, 6, -1, -1, -1, -1, -1};
    std::size_t resume = 0;
    want = placement_refill_batch(order, resume, 8, grown);
    require((want == std::vector<std::uint32_t>{3, 4, 5, 6, 7}),
            "a resumed walk reads only the keys the new frames still lack");
}


// ---- the --experts spec ----

void expect_spec_error(const std::string& text, const char* why) {
    try {
        parse_experts_spec(text);
    } catch (const std::invalid_argument&) { return; }
    require(false, ("the spec should have been rejected: " + std::string(why) + " (" + text + ")").c_str());
}

void test_experts_spec() {
    // Grammar and defaults.
    require(parse_experts_spec("vram=all").all, "vram=all");
    require(parse_experts_spec("vram=100%").all, "vram=100% reads as all");
    const ExpertsSpec shares = parse_experts_spec("vram=60%,ram=25%,ssd=15%");
    require(shares.vram_permille == 600 && shares.ram_permille == 250 && shares.ssd_permille == 150,
            "the shares parse as permille");
    const ExpertsSpec counts = parse_experts_spec("vram=14745,ram=6144");
    require(counts.vram_count == 14745 && counts.ram_count == 6144 && !counts.ssd_count.has_value(),
            "counts parse; ssd omitted");
    const ExpertsSpec ramless = parse_experts_spec("vram=50%,ssd=50%");
    require(!ramless.ram_permille.has_value(), "ram may be omitted");

    // Rejections.
    expect_spec_error("", "empty");
    expect_spec_error("ram=50%", "vram required");
    expect_spec_error("vram=0", "vram=0");
    expect_spec_error("vram=0%", "vram=0%");
    expect_spec_error("vram=all,ram=100", "all excludes the others");
    expect_spec_error("vram=50,ram=25%", "mixed forms");
    expect_spec_error("vram=60%,ram=25%,ssd=20%", "shares over 100");
    expect_spec_error("vram=60%,ram=25%,ssd=10%", "named shares must add to 100");
    expect_spec_error("vram=10,vram=20", "duplicate tier");
    expect_spec_error("vram=1,disk=2", "unknown tier");
    expect_spec_error("vram", "no value");

    // Counts over a total: pass-through; the ssd tier takes the rest.
    require(experts_spec_counts(parse_experts_spec("vram=100,ram=50"), 1000) == std::pair{std::uint64_t{100}, std::uint64_t{50}},
            "counts pass through");
    try {
        experts_spec_counts(parse_experts_spec("vram=600,ram=500"), 1000);
        require(false, "counts beyond the total are rejected");
    } catch (const std::invalid_argument&) {}

    // Shares round by largest remainder and add up exactly.
    const auto [v6, r6] = experts_spec_counts(parse_experts_spec("vram=60%,ram=25%,ssd=15%"), 24576);
    require(v6 + r6 + (24576 - v6 - r6) == 24576 && v6 == 14746 && r6 == 6144,
            "60/25/15 of 24576 rounds to 14746/6144/3686");
    const auto [vh, rh] = experts_spec_counts(parse_experts_spec("vram=50%,ram=50%"), 24575);
    require(vh + rh == 24575 && vh == 12288 && rh == 12287, "50/50 splits the odd key to vram");
    const auto [vt, rt] = experts_spec_counts(parse_experts_spec("vram=all"), 24576);
    require(vt == 24576 && rt == 0, "all is every expert in vram, none in ram");
    const auto [v3, r3] = experts_spec_counts(parse_experts_spec("vram=33%"), 300);
    require(v3 == 99 && r3 == 0, "a vram-only spec puts the rest on ssd");
}
} // namespace

int main() {
    try {
        test_parse();
        test_parse_errors();
        test_counts_ranking();
        test_round_trip();
        test_lfru_pinning();
        test_placement_fits_shape();
        test_resident_fill_walk();
        test_refill_batch();
        test_experts_spec();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp expert placement checks passed\n");
    return 0;
}
