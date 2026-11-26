// The model-independent orchestration of an offloaded-MoE family's expert placement, lifted from
// the first family's Program (see the header): the VRAM monitor's start and snapshots, the warm
// start and the saved state, the RAM tier's lifecycle and fills, the placement switches and their
// refill, and the status, export, drift and settle. The family's hooks carry what stays with the
// family (diagnostics, its fallback seed order, the tier's consumers, the round-cost reset, the
// elastic KV interplay's queries).

#include "models/offload/expert_placement_runtime.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <span>
#include <system_error>
#include <utility>

namespace infernix::models::offload {

ExpertPlacementRuntime::ExpertPlacementRuntime(DeviceContext& device, VramBudgetSource& vram, Options options,
                                               Hooks hooks)
    : device_(device), vram_(vram), options_(std::move(options)), hooks_(std::move(hooks)) {}

ExpertPlacementRuntime::~ExpertPlacementRuntime() = default;

void ExpertPlacementRuntime::diagnostic(const std::string& text, DiagnosticLevel level) const noexcept {
    if (hooks_.diagnostic) {
        try {
            hooks_.diagnostic(text, level);
        } catch (...) {}
    }
}

ExpertResidency& ExpertPlacementRuntime::create_residency(std::vector<const std::uint8_t*> banks,
                                                          std::uint64_t record_stride, std::int32_t max_columns,
                                                          std::vector<std::uint64_t> host_strides) {
    record_stride_ = record_stride;
    residency_ = std::make_unique<ExpertResidency>(options_.geometry, std::move(banks), record_stride, max_columns,
                                                   device_.device, std::move(host_strides));
    // An explicit placement pins its vram set; a resident placement also stops every frame
    // lender (the wide workspace falls back to a temporary allocation).
    if (!options_.expert_pinned_keys.empty()) { residency_->pin_keys(options_.expert_pinned_keys); }
    if (options_.resident_experts) { residency_->mark_resident(); }
    residency_->set_swap_enabled(options_.expert_swap);
    return *residency_;
}

void ExpertPlacementRuntime::create_tier(const ExpertStore& store) {
    // Slots of the largest record (a GGUF import's layers differ); a VRAM victim's demotion
    // copies a whole frame (that pitch) into one.
    // A call's CPU jobs hold their ring slots until the call's answer, and a pass's fetched
    // records until the pass has copied them: the ring holds both with a pass to spare.
    const std::uint32_t cpu_cap =
        options_.cpu_workers > 0 && options_.cpu_jobs > 0
            ? std::min<std::uint32_t>(std::max(options_.cpu_jobs, options_.cpu_assist_jobs),
                                      static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs))
            : 0U;
    HostExpertTier::Options tier;
    tier.ring = std::max<std::uint32_t>(128, cpu_cap + 2U * options_.staging_slots);
    const std::uint32_t lists = tier.ring + tier.prefetch + tier.demotion;
    const std::uint64_t keys  = static_cast<std::uint64_t>(store.layers()) * store.experts();
    // Pinned slots up to every key besides the lists; unpinned records for the keys they leave.
    // With unpinned records, tier.bounce of the pinned slots stage their promotions and
    // tier.stream_bounce the prefill stream's copies of them.
    if (options_.prefill_stream && options_.expert_cache && options_.stream_bounce_per_part > 0) {
        tier.stream_bounce = options_.stream_bounce_parts * options_.stream_bounce_per_part;
    }
    tier.slots    = static_cast<std::uint32_t>(std::min<std::uint64_t>(options_.expert_ram_bytes / record_stride_, keys + lists));
    // The prefill stream's reads of SSD-only experts (one per layer walk span instead of one
    // per call) borrow RAM slots for the span (the lowest-ranked residents leave RAM).
    if (options_.prefill_stream && options_.expert_cache) { tier.stream = std::min(options_.stream_read_slots, tier.slots / 8); }
    const std::uint32_t reserved =
        options_.expert_unpinned_bytes >= record_stride_ ? lists + tier.bounce + tier.stream_bounce : lists;
    tier.unpinned = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        options_.expert_unpinned_bytes / record_stride_, keys - std::min<std::uint64_t>(keys, tier.slots - std::min(tier.slots, reserved))));
    tier.min_slots    = std::max<std::uint32_t>(options_.min_tier_slots, lists + options_.min_tier_slots / 2);
    tier.pin_reserve  = options_.later_pinned_bytes;
    const std::uint32_t needed = tier.min_slots + (tier.unpinned > 0 ? tier.bounce + tier.stream_bounce : 0U);
    if (tier.slots < needed) {
        throw std::runtime_error(
            (options_.label.empty() ? std::string() : options_.label + ": ") +
            "the SSD expert tier needs RAM for at least " + std::to_string(needed) +
            " pinned expert slots (" + std::to_string(tier.slots) +
            " fit in the RAM ledger's page-locked expert share; on Windows page-locked memory is limited to "
            "the GPU's shared memory, so lower --host-context-mib or close programs that use it)");
    }
    fetch_channel_ = std::make_unique<ops::offloaded_moe::FetchChannel>();
    tier.fetch     = fetch_channel_.get();
    // A layer walk's step calls ceil(layers / n) layers once per chunk of its n-chunk span (one
    // tier round per step): at most layers + n - 1 layer calls.
    tier.layer_calls =
        store.layers() + options_.layer_walk_tokens / std::max<std::uint32_t>(options_.chunk_columns, 1U) + 1U;
    // An expert stripe on a second drive: each record's tail is read from it while its head
    // is read from the artifact, at the same depth per drive.
    const ExpertStore* records = &store;
    if (!options_.expert_stripe.empty()) {
        striped_store_ = std::make_unique<ExpertStore>(
            with_stripe(store, read_expert_stripe(options_.expert_stripe), options_.stripe_artifact_id));
        records = &*striped_store_;
        tier.io.demand_depth *= 2;
        tier.io.total_depth = tier.io.demand_depth + tier.io.prefetch_depth;
        diagnostic("SSD tier: expert record tails from " + options_.expert_stripe.string(), DiagnosticLevel::Info);
    }
    tier_ = std::make_unique<HostExpertTier>(*records, std::move(tier));
    tier_capacity_.store(tier_->resident_capacity(), std::memory_order_relaxed);
    residency_->attach_tier(tier_.get());
    // A placement switch rebuilds the tier from the boot options (the rebuild's clone):
    // the store (with its stripe), the queue depths, the ring sizing.
    tier_store_  = records;
    tier_options_ = tier_->options();
    if (tier_->refused_slots() > 0) {
        diagnostic("SSD expert tier: Windows refused to page-lock " + std::to_string(tier_->refused_slots()) +
                       " of the planned expert slots (its limit is the GPU's shared memory); they are kept as "
                       "unpinned records the CPU serves",
                   DiagnosticLevel::Warning);
    }
    if (!tier_->unpinned_lock_failure().empty()) {
        diagnostic("SSD expert tier: the OS would not lock the " + std::to_string(tier_->unpinned_records()) +
                       " unpinned expert records in RAM (" + tier_->unpinned_lock_failure() +
                       "); they still work, but under memory pressure the OS may page them out and responses slow down",
                   DiagnosticLevel::Warning);
    }
}

ExpertPlacementRuntime::FrameStart ExpertPlacementRuntime::start_expert_frames(const VramDemand& demand) {
    FrameStart out;
    out.sizing = size_expert_frames(vram_.query(), demand);
    // Each pool maps whole chunks: one chunk of slack per pool beyond the first.
    const auto slack = static_cast<std::uint32_t>((residency_->pools() - 1) * ExpertResidency::kChunkBytes /
                                                  residency_->frame_bytes());
    out.sizing.frames   = out.sizing.frames > slack ? out.sizing.frames - slack : 0U;
    // An explicit placement names the pool's exact size; the startup resolution validated
    // it against the same sizing, so a clamp here only guards a stray direct use.
    if (options_.expert_frame_target != 0) {
        if (options_.expert_frame_target > out.sizing.frames) {
            diagnostic("expert placement: the frame target " + std::to_string(options_.expert_frame_target) +
                           " exceeds what fits (" + std::to_string(out.sizing.frames) +
                           "); starting at the fit", DiagnosticLevel::Warning);
        }
        out.sizing.frames = std::min(options_.expert_frame_target, out.sizing.frames);
    }
    out.resized = residency_->resize(out.sizing.frames, device_.stream, vram_);
    if (out.resized.spilled != 0) {
        diagnostic(std::to_string(out.resized.spilled >> 20) + " MiB of the expert frames were placed in system memory "
               "and given back; the expert cache starts with " + std::to_string(out.resized.frames) + " of " +
               std::to_string(out.sizing.frames) + " frames", DiagnosticLevel::Warning);
    } else if (out.resized.refused) {
        diagnostic("the GPU refused expert-cache memory; the cache starts with " +
                                             std::to_string(out.resized.frames) + " of " +
                                             std::to_string(out.sizing.frames) + " frames", DiagnosticLevel::Warning);
    }
    out.sizing.frames = out.resized.frames;
    // An explicit placement names the pool's size: the VRAM monitor's pressure-driven
    // resizes are policy and stay off; the placement's own switches move the pool.
    placement_ceiling_ = options_.expert_frame_ceiling;
    start_vram_monitor(demand);
    return out;
}

void ExpertPlacementRuntime::start_vram_monitor(const VramDemand& demand) {
    if (!options_.vram_monitor || options_.expert_frame_ceiling != 0) { return; }
    const VramSnapshot now = vram_.query();
    if (residency_->elastic()) {
        // The sizing law's inputs for the family's grow_kv spare allowance: handed to the
        // family, whose elastic KV pool's later mappings move the usage baseline.
        if (hooks_.capture_sizing) { hooks_.capture_sizing(demand, now); }
    }
    placement_ceiling_ = options_.expert_frame_ceiling;
    monitor_ = std::make_unique<VramMonitor>(device_.device, vram_,
                                             [this](const VramSnapshot& snapshot) { on_vram_snapshot(snapshot); });
}

void ExpertPlacementRuntime::on_vram_snapshot(const VramSnapshot& snapshot) {
    const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
    if (hooks_.kv_idle_due && hooks_.kv_idle_due() && waker_) { waker_(); }
    const std::uint64_t headroom = display_headroom(snapshot.display, false, options_.vram_headroom);
    const bool pressure = snapshot.device_free < headroom / 2 ||
                          (snapshot.has_budget && snapshot.local_usage > snapshot.local_budget);
    if (pressure && !pressure_warned_) {
        const std::uint64_t short_by = headroom > snapshot.device_free ? headroom - snapshot.device_free : 0;
        diagnostic("free VRAM is " + std::to_string(snapshot.device_free >> 20) + " MiB, below the " +
                       std::to_string(headroom >> 20) + " MiB headroom by " + std::to_string(short_by >> 20) +
                       " MiB; the engine does not resize the expert cache itself -- name a smaller placement "
                       "through --experts or POST /admin/expert-placement to give the memory back",
                   DiagnosticLevel::Warning);
    }
    pressure_warned_ = pressure;
}

void ExpertPlacementRuntime::set_maintenance_waker(std::function<void()> waker) {
    const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
    waker_ = std::move(waker);
}

void ExpertPlacementRuntime::wake_maintenance() {
    const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
    if (waker_) { waker_(); }
}

void ExpertPlacementRuntime::warm_start_experts() {
    const auto total = keys();
    expert_cache::ExpertStateLoad load;
    SeedSource source = SeedSource::None;
    if (!options_.expert_state.empty() && options_.expert_cache) {
        load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, total);
        if (load.state) {
            source = SeedSource::Saved;
        } else if (auto fallback = hooks_.fallback_seed_order ? hooks_.fallback_seed_order() : std::nullopt) {
            diagnostic("expert cache: no saved state used: " + load.message, DiagnosticLevel::Debug);
            load.state = std::move(fallback);
            source     = SeedSource::Calibrated;
        }
    }
    // An explicit placement's order (its vram set first, then its ram set) overrides the
    // saved ranking for the seed and the tier fill; the counts stay the session's.
    if (load.state && !options_.expert_tier_order.empty()) {
        load.state = expert_cache::SavedState{load.state->counts, options_.expert_tier_order};
    }
    const expert_cache::SavedState* state = load.state ? &*load.state : nullptr;
    TierFill fill;
    if (tier_) { fill = prefill_tier(state, source); }
    if (options_.resident_experts && tier_ && residency_->frames() > 0) {
        // A resident placement holds only the tier's floor in RAM, so the one-pass warm start
        // would seed little: walk the fill order through the tier's free slots in windows
        // instead -- each window reads from the artifact, seeds into frames, and its slots
        // (then shadows) free for the next. One pass of the artifact at startup, every expert
        // resident before the first request.
        const auto begin    = std::chrono::steady_clock::now();
        const std::uint32_t tier_keys = tier_->keys();
        expert_cache::SavedState window;
        if (state != nullptr && state->counts.size() == tier_keys) { window.counts = state->counts; }
        std::uint32_t seeded = 0;
        // The cursor's first un-read key: prefill_tier already read the order's top
        // fill.filled keys into the tier's slots. Advancing past what each window really
        // landed (not a fixed span) keeps a window from re-reading keys an earlier window
        // seeded -- a re-read key would rank again, seed nothing (it is already in a
        // frame) and stall the walk after one window.
        std::size_t at = fill.filled;
        for (;;) {
            // Seed every key the tier's slots hold (the first window is prefill_tier's, the
            // later ones its leftovers -- a pool's share can fall short -- plus the keys the
            // last prefill landed), then free their slots and read the next window of the
            // order from the artifact. The seed ranks only the keys with a host copy: the
            // policy seeds what it is given, so ranking the whole order would mark every
            // key resident and leave no room.
            window.ranked.clear();
            for (const std::uint32_t key : fill.order) {
                if (tier_->record(key) != nullptr) { window.ranked.push_back(key); }
            }
            if (window.ranked.empty()) { break; }
            const std::uint32_t loaded = residency_->warm_start(
                window, kSeedCountCap, static_cast<std::uint32_t>(window.ranked.size()), device_.stream);
            device_.synchronize();
            seeded += loaded;
            const std::uint32_t released = tier_->controller().release_shadows();
            diagnostic("resident warm fill: seeded " + std::to_string(loaded) + ", released " +
                           std::to_string(released) + " slots, free resident slots " +
                           (tier_->controller().has_free_resident() ? "yes" : "no"),
                       DiagnosticLevel::Debug);
            if (loaded == 0 || at >= fill.order.size() || !tier_->controller().has_free_resident()) {
                break;
            }
            const std::size_t end = std::min(fill.order.size(), at + 4096);
            const std::uint32_t landed =
                tier_->prefill(std::span<const std::uint32_t>(fill.order.data() + at, end - at));
            at += landed;
            if (landed == 0) { break; }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        diagnostic("expert cache: resident warm fill loaded " + std::to_string(seeded) + " of " +
                       std::to_string(total) + " experts from the artifact in " + std::to_string(seconds) +
                           " s (" + std::to_string(residency_->frames()) + " frames)");
        // The resident placement holds every expert in VRAM: the tier only read the fill.
        // Nothing can be evicted, so no demotion or fetch can arrive; its pinned slots
        // release and the process's host expert bytes drop to zero. A placement switch
        // that names a RAM tier rebuilds it on demand.
        shrink_tier_to_reader("the resident placement holds every expert; the fill's reader releases");
        return;
    }
    const std::uint32_t seeded = warm_start_frames(load, source);
    if (tier_) { finish_tier_fill(fill, seeded); }
}

std::vector<std::uint32_t> ExpertPlacementRuntime::tier_fill_order(const expert_cache::SavedState* state) const {
    const std::uint32_t layers = options_.geometry.banks, experts = options_.geometry.experts_per_bank,
                       total = keys();
    if (tier_->keys() != total) {
        throw std::logic_error((options_.label.empty() ? std::string() : options_.label + ": ") +
                               "the SSD tier has another key count");
    }
    // An explicit placement names the fill order itself (its vram set first, then its ram set).
    if (!options_.expert_tier_order.empty()) {
        std::vector<std::uint32_t> order = options_.expert_tier_order;
        std::vector<std::uint8_t> listed(total, 0);
        for (const std::uint32_t key : order) { listed[key] = 1; }
        for (std::uint32_t i = 0; i < total; ++i) {
            const std::uint32_t key = (i % layers) * experts + i / layers; // spread(key) == i
            if (listed[key] == 0) { order.push_back(key); }
        }
        return order;
    }
    // A key's place in the layer-interleaved order: expert e of every layer before expert e + 1.
    const auto spread = [&](std::uint32_t key) { return (key % experts) * layers + key / experts; };
    std::vector<std::uint32_t> order;
    order.reserve(total);
    std::vector<std::uint8_t> listed(total, 0);
    if (state != nullptr) {
        for (const std::uint32_t key : state->ranked) {
            if (key < total && listed[key] == 0) {
                order.push_back(key);
                listed[key] = 1;
            }
        }
        if (state->counts.size() == total) {
            std::vector<std::uint32_t> used;
            for (std::uint32_t key = 0; key < total; ++key) {
                if (listed[key] == 0 && state->counts[key] != 0) { used.push_back(key); }
            }
            std::sort(used.begin(), used.end(), [&](std::uint32_t a, std::uint32_t b) {
                const std::uint32_t ca = state->counts[a], cb = state->counts[b];
                return ca != cb ? ca > cb : spread(a) < spread(b);
            });
            for (const std::uint32_t key : used) {
                order.push_back(key);
                listed[key] = 1;
            }
        }
    }
    for (std::uint32_t i = 0; i < total; ++i) {
        const std::uint32_t key = (i % layers) * experts + i / layers; // spread(key) == i
        if (listed[key] == 0) { order.push_back(key); }
    }
    return order;
}

ExpertPlacementRuntime::TierFill ExpertPlacementRuntime::prefill_tier(const expert_cache::SavedState* state,
                                                                      SeedSource source) {
    const auto start = std::chrono::steady_clock::now();
    TierFill fill;
    fill.source = source;
    fill.order = tier_fill_order(state);
    if (state != nullptr && state->counts.size() == tier_->keys()) {
        tier_->controller().seed_uses(state->counts, kSeedCountCap);
    }
    fill.filled  = tier_->prefill(fill.order);
    fill.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return fill;
}

void ExpertPlacementRuntime::finish_tier_fill(const TierFill& fill, std::uint32_t seeded) {
    const auto start        = std::chrono::steady_clock::now();
    std::uint32_t released  = 0, refilled = 0;
    if (seeded > 0) {
        released = tier_->controller().release_shadows();
        if (released > 0) {
            refilled = tier_->prefill(std::span<const std::uint32_t>(fill.order).subspan(fill.filled));
        }
    }
    // The unpinned records take the next keys of the order that neither the slots nor the frames
    // hold: they keep them for good, so a copy of a VRAM seed would be wasted.
    const std::int32_t* in_vram = residency_->host_table();
    const std::uint32_t unpinned =
        tier_->fill_unpinned(fill.order, [in_vram](std::uint32_t key) { return in_vram != nullptr && in_vram[key] >= 0; });
    const double seconds =
        fill.seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::uint32_t total = tier_->keys(), in_ram = fill.filled - released + refilled;
    const std::uint32_t slots = tier_->pinned_slots();
    const double gib          = static_cast<double>(tier_->record_bytes()) / (1ULL << 30); // a slot: the largest record
    char split[160] = "";
    if (tier_->unpinned_records() > 0) {
        std::snprintf(split, sizeof(split), " and %u unpinned records (%.1f GiB, served by the CPU) hold %u more",
                      tier_->unpinned_records(), gib * tier_->unpinned_records(), unpinned);
    }
    char line[560];
    std::snprintf(line, sizeof(line),
                  "SSD expert tier: %u RAM slots (%.1f GiB) hold %u experts (%s)%s%s, read in %.1f s; %u of %u "
                  "experts are read from the artifact when used",
                  slots, gib * slots, in_ram,
                  fill.source == SeedSource::Saved        ? "the saved ranking, then the last session's use counts"
                  : fill.source == SeedSource::Calibrated ? "the calibrated expert order"
                                                          : "spread across the layers",
                  released > 0 ? (", none of them among the " + std::to_string(released) + " in VRAM").c_str() : "",
                  split, seconds, total - in_ram - released - unpinned, total);
    diagnostic(line);
}

std::uint32_t ExpertPlacementRuntime::warm_start_frames(const expert_cache::ExpertStateLoad& load, SeedSource source) {
    if (options_.expert_state.empty() || !options_.expert_cache) { return 0; }
    if (source == SeedSource::None) {
        diagnostic("expert cache: no saved state yet, so it starts empty and fills as requests arrive");
        diagnostic("expert cache starts empty: " + load.message, DiagnosticLevel::Debug);
        expert_state_saved_ = std::chrono::steady_clock::now();
        return 0;
    }
    const auto start           = std::chrono::steady_clock::now();
    // An explicit placement seeds the whole pool: its vram set is the seed order, and the
    // target holds exactly it. Auto keeps half the frames free for the fill phase's landing.
    const auto max_keys = static_cast<std::uint32_t>(
        options_.expert_tier_order.empty() ? kSeedShare * residency_->frames() : residency_->frames());
    const std::uint32_t loaded = residency_->warm_start(*load.state, kSeedCountCap, max_keys, device_.stream);
    device_.synchronize();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double loaded_gib =
        static_cast<double>(loaded) * static_cast<double>(residency_->frame_bytes()) / (1ULL << 30);
    char line[256];
    if (source == SeedSource::Saved) {
        std::snprintf(line, sizeof(line),
                      "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                      loaded, loaded_gib, seconds);
    } else {
        std::snprintf(line, sizeof(line),
                      "expert cache: no saved state yet, so it starts with the %u highest-ranked experts of "
                      "the calibrated order (%.1f GiB in %.2f s)",
                      loaded, loaded_gib, seconds);
    }
    diagnostic(line);
    std::snprintf(line, sizeof(line), "expert cache warm start: %u of %u frames from %s", loaded,
                  residency_->frames(),
                  source == SeedSource::Saved ? options_.expert_state.string().c_str() : "the calibrated order");
    diagnostic(line, DiagnosticLevel::Debug);
    expert_state_saved_  = std::chrono::steady_clock::now();
    expert_state_routed_ = residency_->stats().routed;
    return loaded;
}

void ExpertPlacementRuntime::save_expert_state() noexcept {
    if (options_.expert_state.empty() || !residency_ || residency_->frames() == 0) { return; }
    try {
        const std::string error =
            expert_cache::save_expert_state(options_.expert_state, options_.expert_state_identity, residency_->saved_state());
        if (!error.empty()) { diagnostic("expert state not saved: " + error, DiagnosticLevel::Warning); }
        expert_state_saved_  = std::chrono::steady_clock::now();
        expert_state_routed_ = residency_->stats().routed;
    } catch (const std::exception& error) {
        diagnostic(std::string("expert state not saved: ") + error.what(), DiagnosticLevel::Warning);
    } catch (...) {}
}

void ExpertPlacementRuntime::maintain_expert_state() noexcept {
    if (options_.expert_state.empty() || !residency_ || residency_->stats().routed == expert_state_routed_) { return; }
    if (std::chrono::steady_clock::now() - expert_state_saved_ < kExpertStateInterval) { return; }
    save_expert_state();
}

void ExpertPlacementRuntime::record_switch(bool ok, std::string message) {
    const std::lock_guard<std::mutex> lock(placement_mutex_);
    placement_last_ok_      = ok;
    placement_last_message_ = std::move(message);
}

void ExpertPlacementRuntime::trim_host_heap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

void ExpertPlacementRuntime::ensure_tier(std::uint32_t residents) {
    if (tier_store_ == nullptr) {
        diagnostic("expert placement: this engine runs without the SSD expert tier; the RAM target is ignored",
                   DiagnosticLevel::Warning);
        return;
    }
    if (tier_ != nullptr && tier_->resident_capacity() == residents) { return; }
    // The old tier's records seed the new one: no demotion may still be landing in its slots.
    residency_->quiesce_tier(device_.stream);
    // A new tier that does not fit beside the old one (a tier holding most of the host's RAM)
    // replaces it empty: the old releases first and the fill reads the records again.
    if (tier_ != nullptr && tier_bytes(residents) > host_memory_snapshot().available_physical) {
        release_tier("its rebuild does not fit beside it");
    }
    HostExpertTier::Options options = tier_options_;
    const std::uint32_t lists       = options.ring + options.prefetch + options.demotion;
    // ram=0 below full VRAM still stages SSD reads through the tier's lists; the tier needs
    // one resident slot beyond them.
    residents                        = std::max<std::uint32_t>(residents, 1);
    options.slots                    = residents + lists;
    options.unpinned                 = 0; // a rebuild holds no pageable records
    options.bounce                   = 0; // nothing to stage without them
    options.stream_bounce            = 0;
    options.min_slots                = lists + 1;
    auto next                        = std::make_unique<HostExpertTier>(*tier_store_, std::move(options));
    std::size_t kept                 = 0;
    if (tier_ != nullptr) {
        // The keys the old tier holds, best first: its own decayed usage, else identity.
        const std::uint32_t total = keys();
        std::vector<std::uint32_t> order(total);
        for (std::uint32_t k = 0; k < total; ++k) { order[k] = k; }
        const auto& lfu = tier_->controller().lfu();
        std::sort(order.begin(), order.end(),
                  [&](std::uint32_t a, std::uint32_t b) { return lfu.value(a) > lfu.value(b); });
        std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> copies;
        for (const std::uint32_t key : order) {
            if (copies.size() >= residents) { break; }
            const std::uint8_t* record = tier_->host_record(key);
            if (record == nullptr) { continue; }
            std::int32_t slot = -1;
            if (next->controller().prefill(key, slot)) {
                copies.emplace_back(const_cast<std::uint8_t*>(next->slot_bytes(static_cast<std::uint32_t>(slot))),
                                    record);
            }
        }
        next->copy_records(copies);
        kept = copies.size();
        residency_->retier(next.get(), device_.stream);
        device_.synchronize();
        tier_ = std::move(next); // the old tier's pinned chunks release
        tier_capacity_.store(tier_->resident_capacity(), std::memory_order_relaxed);
        trim_host_heap();       // and its pageable records go back to the OS
        diagnostic("expert placement: RAM tier rebuilt for " + std::to_string(residents) +
                       " resident slots (" + std::to_string(kept) + " records kept)",
                   DiagnosticLevel::Info);
        // The stream's callbacks captured the tier a rebuild replaced: the family's hook
        // retargets them to this one.
        if (hooks_.on_tier_changed) { hooks_.on_tier_changed(tier_.get(), true); }
    } else {
        tier_ = std::move(next);
        tier_capacity_.store(tier_->resident_capacity(), std::memory_order_relaxed);
        residency_->retier(tier_.get(), device_.stream);
        device_.synchronize();
        diagnostic("expert placement: RAM tier built for " + std::to_string(residents) + " resident slots",
                   DiagnosticLevel::Info);
        if (hooks_.on_tier_changed) { hooks_.on_tier_changed(tier_.get(), false); }
    }
    start_tier_fill();
}

void ExpertPlacementRuntime::shrink_tier_to_reader(const char* why) {
    if (tier_ == nullptr) { return; }
    tier_backfill_.active = false;
    if (residency_->resident_count() >= static_cast<std::size_t>(options_.geometry.banks) *
                                            options_.geometry.experts_per_bank) {
        release_tier(why); // every key holds a frame: nothing is read through RAM
        return;
    }
    ensure_tier(0);
    diagnostic(std::string("expert placement: RAM tier reduced to its artifact reader (") + why + ")",
               DiagnosticLevel::Info);
}

void ExpertPlacementRuntime::release_tier(const char* why) {
    if (tier_ == nullptr) { return; }
    tier_backfill_.active = false;
    residency_->retier(nullptr, device_.stream);
    device_.synchronize();
    tier_.reset();
    tier_capacity_.store(0, std::memory_order_relaxed);
    if (hooks_.on_tier_changed) { hooks_.on_tier_changed(nullptr, false); }
    trim_host_heap();
    diagnostic(std::string("expert placement: RAM tier released (") + why + ")", DiagnosticLevel::Info);
}

void ExpertPlacementRuntime::start_tier_fill() {
    if (tier_ == nullptr || residency_ == nullptr || tier_->resident_capacity() == 0) { return; }
    const expert_cache::SavedState state = residency_->saved_state();
    tier_backfill_          = TierBackfill{};
    // Every key by usage: the residents (skipped, they hold frames), then the rest by count.
    tier_backfill_.order  = tier_fill_order(&state);
    tier_backfill_.active = true;
    wake_maintenance();
}

void ExpertPlacementRuntime::advance_tier_fill() {
    if (!tier_backfill_.active || tier_ == nullptr || residency_ == nullptr) { return; }
    const std::uint32_t total = keys();
    const std::int32_t* table = residency_->host_table();
    std::vector<std::uint32_t> want;
    for (std::size_t i = tier_backfill_.at; i < tier_backfill_.order.size() && want.size() < kRefillBatch; ++i) {
        const std::uint32_t key = tier_backfill_.order[i];
        tier_backfill_.at            = i + 1;
        if (key < total && table[key] < 0 && tier_->record(key) == nullptr) { want.push_back(key); }
        if (want.size() >= kRefillBatch) { break; }
    }
    const std::uint32_t landed = want.empty() ? 0 : tier_->prefill(want);
    if (!tier_->controller().has_free_resident() || tier_backfill_.at >= tier_backfill_.order.size()) {
        tier_backfill_.active = false;
        diagnostic("expert placement: RAM tier fill read " + std::to_string(landed) + " more records; "
                   "the tier is full or the order is exhausted", DiagnosticLevel::Info);
        return;
    }
    if (landed > 0) {
        tier_backfill_.stall = 0;
    } else if (++tier_backfill_.stall >= 2) {
        tier_backfill_.active = false;
        return;
    }
    wake_maintenance();
}

bool ExpertPlacementRuntime::request_placement(const ExpertPlacementRequest& request, std::string* error) {
    std::unique_lock<std::mutex> lock(placement_mutex_);
    if (!residency_ || residency_->frames() == 0 && request.frame_target == 0) {
        if (error != nullptr) { *error = "this engine runs without the expert cache"; }
        return false;
    }
    const std::uint32_t target =
        request.frame_target != 0 ? request.frame_target : residency_->frames();
    if (target > residency_->frame_limit()) {
        if (error != nullptr) {
            *error = "the frame target " + std::to_string(target) + " exceeds the pool limit " +
                     std::to_string(residency_->frame_limit());
        }
        return false;
    }
    if (!residency_->placement_fits(target)) {
        if (error != nullptr) {
            *error = "the frame target " + std::to_string(target) + " (limit " +
                     std::to_string(residency_->frame_limit()) + ", split[0] " +
                     std::to_string(residency_->placement_split(0)) + ", pinned " +
                     std::to_string(residency_->pinned_keys_total()) +
                     ") cannot hold every pool's pinned keys; raise it or change --expert-placement";
        }
        return false;
    }
    if (request.has_ram_target && request.ram_target > 0 &&
        tier_capacity_.load(std::memory_order_relaxed) != request.ram_target) {
        // The records the rebuilt RAM tier would pin, plus its lists: they must fit the host's
        // free memory and what the current tier gives back (the rebuild copies from it beside the
        // new tier when both fit, else releases it first; ensure_tier). The mechanism's numbers;
        // whether to spend them is client policy. A tier that already has the named capacity is
        // kept as it is.
        const HostMemorySnapshot host = host_memory_snapshot();
        const std::uint64_t bytes     = tier_bytes(request.ram_target);
        const std::uint32_t current   = tier_capacity_.load(std::memory_order_relaxed);
        const std::uint64_t room =
            host.available_physical + (current != 0 ? tier_bytes(current) : 0);
        if (bytes > room) {
            if (error != nullptr) {
                *error = "the RAM tier's " + std::to_string(request.ram_target) + " records need " +
                         std::to_string(bytes >> 20) + " MiB of host memory; " +
                         std::to_string(host.available_physical >> 20) +
                         " MiB is free beside the current tier's " +
                         std::to_string((room - host.available_physical) >> 20);
            }
            return false;
        }
    }
    placement_pending_    = request;
    placement_pending_set_ = true;
    lock.unlock();
    // An idle engine runs no round boundary that would apply the switch: wake the worker so
    // it applies now instead of at the next request.
    wake_maintenance();
    return true;
}

bool ExpertPlacementRuntime::request_count_placement(std::uint32_t vram_experts, std::uint32_t ram_experts,
                                                     std::string* error) {
    if (residency_ == nullptr) {
        if (error != nullptr) { *error = "this engine runs without the expert cache"; }
        return false;
    }
    ExpertPlacementRequest request;
    request.frame_target  = std::min(vram_experts, residency_->frame_limit());
    request.has_ram_target = true;
    request.ram_target    = ram_experts;
    // No order: the fills rank every key by usage (saved_state().ranked holds only residents).
    return request_placement(request, error);
}

void ExpertPlacementRuntime::apply_pending_placement() {
    ExpertPlacementRequest request;
    {
        const std::lock_guard<std::mutex> lock(placement_mutex_);
        if (!placement_pending_set_) { return; }
        request              = std::move(placement_pending_);
        placement_pending_set_ = false;
    }
    if (!residency_) { return; }
    const std::uint32_t target = request.frame_target != 0 ? request.frame_target : residency_->frames();
    if (!residency_->placement_fits(target)) {
        diagnostic("expert placement switch to " + std::to_string(target) +
                       " frames refused: a pool's pinned keys do not fit; the placement is unchanged",
                   DiagnosticLevel::Warning);
        return;
    }
    try {
        record_switch(true, "applied: " + std::to_string(target) + " VRAM frames, RAM tier target " +
                                std::to_string(request.has_ram_target ? request.ram_target : tier_resident_slots()));
        const bool full = target >= residency_->frame_limit();
        // The RAM tier first: a shrink demotes into its slots, so it must exist with the
        // named capacity before the pool resizes.
        if (request.has_ram_target) {
            if (full && request.ram_target == 0) {
                placement_wants_no_tier_ = true;
            } else {
                placement_wants_no_tier_ = false;
                if (tier_ == nullptr || tier_resident_slots() != request.ram_target) {
                    ensure_tier(request.ram_target);
                } else {
                    start_tier_fill(); // the capacity is right; top up what a drop left free
                }
            }
        }
        // A switch that names no pins (the counts form, the resident default) releases the
        // previous profile's: the gaming shape's pins would otherwise survive into the full
        // one, and the refill below could not seat keys where a stale pin sits.
        residency_->pin_keys(request.pinned);
        const std::uint32_t before = residency_->frames();
        if (target != before) {
            const auto resized = residency_->resize(target, device_.stream, vram_);
            // Several pools back whole per-layer shares: the target rounds down to them.
            const std::uint32_t reachable = residency_->reachable_frames(target);
            if (resized.frames != reachable) {
                record_switch(false, "the expert cache reached " + std::to_string(resized.frames) + " of " +
                                         std::to_string(reachable) + " frames: VRAM ran out" +
                                         (resized.refused ? "" : " or a lease held frames"));
            } else {
                record_switch(true, "expert cache " + std::to_string(before) + " -> " +
                                        std::to_string(resized.frames) + " frames, RAM tier " +
                                        std::to_string(tier_resident_slots()) + " slots");
            }
            {
                const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
                // The switch names the pool's size from here on.
                placement_ceiling_ = resized.frames;
            }
            // A grow backs new frames without filling them; the resident refill walks the
            // usage-rank order into them between rounds (a shrink needs none: its survivors
            // demoted through the tier on the way down, the rest backfilled by the tier fill).
            if (resized.frames > before) {
                // The refill reads through the tier: a reader-sized tier stages one key a
                // tick, so give the walk a batch of slots and shrink back when it finishes.
                const std::uint32_t named = request.has_ram_target ? request.ram_target : tier_resident_slots();
                const bool staging        = tier_ != nullptr && tier_resident_slots() < kRefillBatch;
                if (staging) { ensure_tier(kRefillBatch); }
                start_placement_fill(request.order);
                placement_fill_.after_switch = true;
                placement_fill_.restore_tier = staging;
                placement_fill_.tier_after   = named;
            }
            else if (tier_ != nullptr) {
                // Land the shrink's demotions now, so the tier's counts are final when GET
                // stops reporting the switch, and the fill sees which slots stay free.
                residency_->quiesce_tier(device_.stream);
                // Exclusive tiers: survivors that kept a RAM copy give the slot back, so the
                // fill can bring the named RAM set up to its count.
                tier_->controller().release_shadows();
                start_tier_fill();
            }
            diagnostic("expert placement switched: expert cache " + std::to_string(before) + " -> " +
                       std::to_string(resized.frames) + " frames (" +
                       std::to_string(residency_->pinned_keys_total()) + " keys pinned, RAM tier " +
                       std::to_string(tier_resident_slots()) + " slots)" +
                       (placement_fill_.active ? ", resident refill started" : "") +
                       (tier_backfill_.active ? ", RAM tier fill started" : ""));
        } else if (!request.pinned.empty()) {
            const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
            placement_ceiling_ = before;
            diagnostic("expert placement switched: " + std::to_string(residency_->pinned_keys_total()) +
                       " keys pinned over " + std::to_string(before) + " frames, RAM tier " +
                       std::to_string(tier_resident_slots()) + " slots");
        } else {
            const std::lock_guard<std::recursive_mutex> lock(vram_mutex_);
            placement_ceiling_ = before;
            diagnostic("expert placement switched: " + std::to_string(residency_->pinned_keys_total()) +
                       " keys pinned released over " + std::to_string(before) + " frames, RAM tier " +
                       std::to_string(tier_resident_slots()) + " slots");
            if (placement_wants_no_tier_ && !placement_fill_.active) {
                placement_wants_no_tier_ = false;
                shrink_tier_to_reader("every expert is resident in VRAM and the switch named no RAM tier");
            }
        }
    } catch (const std::exception& error) {
        diagnostic("expert placement switch failed: " + std::string(error.what()), DiagnosticLevel::Error);
        record_switch(false, std::string("failed: ") + error.what() + "; the placement keeps what it reached");
        // Land whatever copies the failed step left in flight, so the next round starts settled.
        try {
            if (residency_) { residency_->quiesce_tier(device_.stream); }
        } catch (const std::exception& settle) {
            diagnostic("expert placement: settling after the failed switch failed: " + std::string(settle.what()),
                       DiagnosticLevel::Error);
        }
    }
    if (hooks_.on_switch_settled) {
        hooks_.on_switch_settled("the expert placement switched"); // a failed switch may still have moved experts
    }
}

void ExpertPlacementRuntime::start_placement_fill(const std::vector<std::uint32_t>& request_order) {
    if (tier_ == nullptr || residency_ == nullptr || residency_->frames() == 0) { return; }
    const expert_cache::SavedState state = residency_->saved_state();
    placement_fill_                      = PlacementFill{};
    const std::uint32_t total = keys();
    if (state.counts.size() == total) { placement_fill_.window.counts = state.counts; }
    // A file placement names every key; anything shorter (saved_state().ranked holds only the
    // residents) would leave the grown frames without the keys to fill them.
    placement_fill_.order = request_order.size() >= total
                                ? request_order
                                : tier_fill_order(state.counts.empty() && state.ranked.empty()
                                                      ? nullptr
                                                      : &state);
    placement_fill_.total  = static_cast<std::uint32_t>(placement_fill_.order.size());
    placement_fill_.begin  = std::chrono::steady_clock::now();
    placement_fill_.active = true;
    wake_maintenance();
}

void ExpertPlacementRuntime::advance_placement_fill() {
    if (!placement_fill_.active) { return; }
    if (residency_ == nullptr || tier_ == nullptr || residency_->frames() == 0) {
        placement_fill_.active = false;
        return;
    }
    // Seed what the tier holds (the last tick's reads, its leftovers, the switch's survivors),
    // then free their slots for this tick's read -- the boot warm fill's window walk.
    placement_fill_.window.ranked.clear();
    for (const std::uint32_t key : placement_fill_.order) {
        if (tier_->record(key) != nullptr) { placement_fill_.window.ranked.push_back(key); }
    }
    std::uint32_t seeded = 0;
    if (!placement_fill_.window.ranked.empty()) {
        seeded = residency_->warm_start(placement_fill_.window, kSeedCountCap,
                                        static_cast<std::uint32_t>(placement_fill_.window.ranked.size()),
                                        device_.stream);
        placement_fill_.seeded += seeded;
        device_.synchronize();
    }
    tier_->controller().release_shadows();
    // Read the next batch of the order: only keys still without a frame.
    const std::int32_t* table = residency_->host_table();
    const std::uint32_t total  = keys();
    const std::uint32_t landed =
        tier_->prefill(placement_refill_batch(placement_fill_.order, placement_fill_.at, kRefillBatch,
                                              std::span<const std::int32_t>(table, total)));
    const bool filled = residency_->resident_count() >= residency_->frames();
    if (seeded > 0 || landed > 0) {
        placement_fill_.stall = 0;
    } else if (++placement_fill_.stall >= 2 || placement_fill_.at >= placement_fill_.order.size()) {
        // Keys still mid-copy on the first walk were skipped: walk the order again while
        // frames stay empty, a bounded number of times.
        if (!filled && placement_fill_.passes < 2) {
            ++placement_fill_.passes;
            placement_fill_.at    = 0;
            placement_fill_.stall = 0;
            wake_maintenance();
            return;
        }
        finish_placement_fill(); // nothing more to seed or read: every frame that can hold a key does
        return;
    }
    if (filled) {
        finish_placement_fill();
        return;
    }
    wake_maintenance(); // not done: keep the cadence going between rounds and while idle
}

void ExpertPlacementRuntime::finish_placement_fill() {
    placement_fill_.active = false;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - placement_fill_.begin).count();
    diagnostic("expert placement: resident refill seeded " + std::to_string(placement_fill_.seeded) +
               " experts into " + std::to_string(residency_->frames()) + " frames in " +
               std::to_string(seconds) + " s (" + std::to_string(residency_->resident_count()) +
               " resident)");
    if (placement_fill_.after_switch) {
        if (hooks_.on_switch_settled) { hooks_.on_switch_settled("the placement's resident refill completed"); }
    } else {
        wake_maintenance(); // a lease's refill: the RAM tier settles next (settle_placement)
    }
    // The switch this fill served named no RAM tier and every key is now in VRAM: the tier
    // served as the fill's reader, and with nothing left to demote or fetch it releases.
    if (placement_wants_no_tier_ && residency_->resident_count() + 1 >= residency_->frames() &&
        residency_->frames() >= residency_->frame_limit()) {
        placement_wants_no_tier_     = false;
        placement_fill_.restore_tier = false;
        shrink_tier_to_reader("the grow placed every expert in VRAM and the switch named no RAM tier");
    }
    if (placement_fill_.restore_tier) {
        placement_fill_.restore_tier = false;
        ensure_tier(placement_fill_.tier_after); // the staging tier's slots return
    } else if (tier_ != nullptr && tier_resident_slots() > 1) {
        // Exclusive tiers: the keys the refill seeded into VRAM give their RAM slots back, and
        // the tier fill gives them to the next keys without a frame.
        tier_->controller().release_shadows();
        start_tier_fill();
    }
}

bool ExpertPlacementRuntime::apply_placement(bool idle) {
    apply_pending_placement();
    advance_placement_fill(); // the placement grow's refill: advances at every boundary and idle
    advance_tier_fill();      // the RAM tier's fill: the same cadence
    if (idle && hooks_.all_lanes_idle && hooks_.all_lanes_idle()) {
        settle_placement();
    }
    if (options_.resident_experts) { return false; } // the pools hold every expert: nothing resizes them
    if (placement_ceiling_ != 0) { return false; } // an explicit placement: only its switches move the pool
    return true; // the caller's elastic KV control law may run
}

ExpertPlacementStatus ExpertPlacementRuntime::placement_status() const {
    ExpertPlacementStatus status;
    if (residency_ == nullptr) { return status; }
    status.resident   = residency_->resident();
    status.frames     = residency_->frames();
    status.frame_limit = residency_->frame_limit();
    {
        const std::lock_guard<std::mutex> lock(placement_mutex_);
        status.target = placement_pending_set_ ? placement_pending_.frame_target : 0;
        status.has_tier_target = placement_pending_set_ && placement_pending_.has_ram_target;
        status.tier_target = status.has_tier_target ? placement_pending_.ram_target : 0;
    }
    status.pinned     = residency_->pinned_keys_total();
    status.residents  = residency_->resident_count();
    status.fill_total  = placement_fill_.active ? placement_fill_.total : 0;
    status.fill_seeded = placement_fill_.active ? placement_fill_.seeded : 0;
    { // The swap counters: the residency's per-round policy and the tier's admissions.
        const auto& stats = residency_->stats();
        status.swap_routed     = stats.routed;
        status.swap_hits      = stats.hits;
        status.swap_promotions = stats.promotions;
        status.swap_demotions  = stats.demotions;
        status.swap_promotion_bytes = stats.promotion_bytes;
        status.swap_demotion_bytes  = stats.demotion_bytes;
        if (tier_ != nullptr) { status.swap_admitted = tier_->stats().admitted; }
    }
    status.total_experts = keys();
    status.frame_bytes    = residency_->frame_bytes();
    status.pool_bytes     = residency_->pool_bytes();
    status.landing_slots  = status.resident ? 0 : ExpertResidency::kLandingSlots * options_.geometry.banks;
    {
        const VramSnapshot vram = vram_.query();
        status.vram_free  = vram.device_free;
        status.vram_total = vram.device_total;
        const HostMemorySnapshot host = host_memory_snapshot();
        status.host_total     = host.total_physical;
        status.host_available = host.available_physical;
    }
    if (tier_ != nullptr) {
        status.tier_slots     = tier_->pinned_slots();
        status.tier_residents = 0;
        const std::uint32_t total = keys();
        for (std::uint32_t key = 0; key < total; ++key) {
            const bool host_copy = tier_->record(key) != nullptr;
            const bool framed    = residency_->has_frame(key);
            if (host_copy && !framed) { ++status.tier_residents; }
            if (host_copy && framed) { ++status.tier_shadows; }
            if (!host_copy && !framed) { ++status.ssd_experts; }
        }
    } else {
        // Without a tier every unframed key is served from the artifact.
        const std::uint32_t total = keys();
        for (std::uint32_t key = 0; key < total; ++key) {
            if (!residency_->has_frame(key)) { ++status.ssd_experts; }
        }
    }
    status.expert_record_bytes = residency_->frame_bytes(); // one expert record's bytes
    status.kv_bytes            = hooks_.kv_bytes ? hooks_.kv_bytes() : 0;
    {
        const std::lock_guard<std::mutex> lock(placement_mutex_);
        status.last_switch_ok      = placement_last_ok_;
        status.last_switch_message = placement_last_message_;
    }
    return status;
}

bool ExpertPlacementRuntime::export_placement(const std::filesystem::path& path, std::string* error) {
    if (residency_ == nullptr) {
        if (error != nullptr) { *error = "this engine runs without the expert cache"; }
        return false;
    }
    const expert_cache::SavedState state = residency_->saved_state();
    std::vector<std::uint32_t> ram;
    const std::uint32_t total = keys();
    if (tier_ != nullptr) {
        for (std::uint32_t key = 0; key < total; ++key) {
            if (tier_->record(key) != nullptr) { ram.push_back(key); }
        }
    }
    return write_expert_placement_file(
        path, options_.geometry.banks, options_.geometry.experts_per_bank,
        std::span<const std::int32_t>(residency_->host_table(), total), ram, state.counts, error);
}

bool ExpertPlacementRuntime::placement_drifted() const {
    if (tier_ == nullptr || residency_ == nullptr || tier_backfill_.active || placement_fill_.active) { return false; }
    if (residency_->stats().lent_frames == 0 && residency_->resident_count() + 1 < residency_->frames()) { return true; }
    auto& tier = tier_->controller();
    return tier.shadow_count() > 0 || tier.has_free_resident();
}

void ExpertPlacementRuntime::settle_placement() {
    if (!placement_drifted()) { return; }
    const std::size_t resident = residency_->resident_count();
    if (residency_->stats().lent_frames == 0 && resident + 1 < residency_->frames()) {
        diagnostic("expert placement: refilling " + std::to_string(residency_->frames() - resident) +
                       " empty VRAM frames (a lease evicted their experts)",
                   DiagnosticLevel::Info);
        start_placement_fill({}); // the tier settles at the next idle tick after the refill
        return;
    }
    const std::uint32_t total  = keys();
    const std::int32_t* table = residency_->host_table();
    bool ssd_only = false;
    for (std::uint32_t key = 0; key < total && !ssd_only; ++key) {
        ssd_only = table[key] < 0 && tier_->record(key) == nullptr;
    }
    if (!ssd_only) { return; }
    // Promotions pin their RAM source until they are published, and an idle engine runs no
    // round boundary: land and publish the finished ones here. A promotion still queued keeps
    // its shadow pinned until a later settle (at most one boundary's batch, 64 at 50/50).
    residency_->quiesce_tier(device_.stream);
    const std::uint32_t released = tier_->controller().release_shadows();
    if (released == 0 && !tier_->controller().has_free_resident()) { return; } // the shadows are pinned
    diagnostic("expert placement: RAM tier settling (" + std::to_string(released) +
                   " shadow slots released for SSD-only experts)",
               DiagnosticLevel::Info);
    start_tier_fill();
}

} // namespace infernix::models::offload
