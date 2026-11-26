#pragma once

// The model-independent orchestration of an offloaded-MoE family's expert placement
// (docs/maintainer/moe-offload-design.md §4.8-§4.9, lifted from the first family's Program):
// the VRAM residency (ExpertResidency: the frame pools and their LFRU policy), the optional RAM
// tier over the artifact store (HostExpertTier, with its fetch channel and expert stripe), the
// VRAM monitor, the warm start (saved state, fallback seed order, the tier's fill order, the
// resident warm fill's window walk), the expert state's save and maintenance, the RAM tier's
// lifecycle (build, rebuild, shrink to the artifact reader, release, the tier fill), the
// placement switches (request validation, apply, the grow's resident refill), the placement
// status, export, drift and settle.
//
// Everything the runtime does runs at startup or between rounds: zero hot-path cost. The
// per-round residency calls (before_round, enqueue_route_download, after_round, tier_step,
// seed_prompt, the leases) stay direct calls on the residency the runtime exposes.
//
// The family composes the runtime from its own config and options; what stays with the family
// reaches it through the hooks: diagnostics, a fallback seed order (a calibrated expert order is
// family data), the tier's consumers (the CPU miss service and the prefill expert stream
// retarget to a rebuilt tier), the round-cost reset after a switch, the all-lanes-idle and
// KV-idle queries of the elastic KV interplay, the KV bytes of the status, and the sizing
// baseline the family's grow_kv spare allowance reads. The family's error codes stay with the
// family (check_device_error); the runtime exposes the tier's failure() for it.

#include "core/device.h"
#include "core/vram_budget.h"
#include "models/offload/expert_cache.h"
#include "models/offload/expert_placement.h"
#include "models/offload/expert_residency.h"
#include "models/offload/expert_state.h"
#include "models/offload/expert_store.h"
#include "models/offload/host_expert_tier.h"
#include "models/offload/memory_plan.h"
#include "models/offload/vram_monitor.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "runtime/diagnostics.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace infernix::models::offload {

// The placement the Program serves now (the families alias it into their Program types; the
// public report converts it). kv_bytes comes from the family (the runtime's hook).
struct ExpertPlacementStatus {
    bool resident = false;      // every routed expert held, nothing takes a frame
    std::uint32_t frames        = 0; // average frames backed now
    std::uint32_t frame_limit    = 0; // the pools' limit
    std::uint32_t target         = 0; // a pending switch's frame target (0: none pending)
    std::size_t pinned           = 0; // keys pinned by the placement
    std::size_t residents        = 0; // keys resident in frames
    std::uint32_t tier_slots     = 0; // the RAM tier's pinned slots
    std::size_t tier_residents   = 0; // keys held only in the tier (no frame): the RAM tier's experts
    std::size_t tier_shadows     = 0; // keys with both a frame and a tier copy (released by fills)
    std::uint64_t kv_bytes       = 0; // VRAM the KV cache maps now
    bool last_switch_ok = true;            // the last applied switch reached its target
    std::string last_switch_message;       // what it did, or why it fell short or failed
    bool has_tier_target = false;    // a pending switch names the RAM tier's size
    std::uint32_t tier_target = 0;   // its resident-slot target
    // The mechanism a client computes a placement from (the policy is client code):
    std::uint32_t total_experts = 0;      // routed (layer, expert) keys
    std::uint64_t expert_record_bytes = 0; // one expert's record
    std::uint64_t frame_bytes = 0;         // one VRAM frame (the pools' average)
    std::uint64_t pool_bytes  = 0;          // VRAM the frames occupy now
    std::uint32_t ssd_experts = 0;          // keys with neither a frame nor a host copy
    std::uint64_t vram_free = 0, vram_total = 0;       // cudaMemGetInfo, device-wide
    std::uint64_t host_total = 0, host_available = 0; // the host's RAM
    std::uint32_t landing_slots = 0; // transient landing frames a decode round reserves
    std::uint32_t fill_total     = 0; // keys the placement grow's resident refill walks (0: none)
    std::uint32_t fill_seeded    = 0; // keys the refill seeded so far
    // The swap counters (expert moves within the current counts; 0 with the swap policy off):
    std::uint64_t swap_routed = 0, swap_hits = 0;
    std::uint64_t swap_promotions = 0, swap_demotions = 0, swap_admitted = 0;
    std::uint64_t swap_promotion_bytes = 0, swap_demotion_bytes = 0;
};

class ExpertPlacementRuntime {
public:
    // ---- the family's inputs (all plain values; nothing names a family) ----
    struct Options {
        // The family's name in the moved error strings (empty: a plain message). The shared
        // code never names a family; a family hands its own label in.
        std::string label;
        // The bank geometry (geometry.h): the routed layers, experts per layer, top_k.
        ExpertBankGeometry geometry;
        // The warm start's saved state (read at startup, written at stop and between requests)
        // and the artifact it belongs to. Empty: no warm start.
        std::filesystem::path expert_state;
        std::string expert_state_identity;
        bool expert_cache = true;
        // The placement: resident (every routed expert in the frame pool, nothing takes a
        // frame), or an explicit frame target with its pinned keys and fill order (0: the
        // sizing function's target), and the ceiling the monitor's control law never grows past
        // (0: automatic). The swap policy is the residency's per-round promotions and evictions.
        bool resident_experts = false;
        std::uint32_t expert_frame_target = 0;
        std::uint32_t expert_frame_ceiling = 0;
        std::vector<std::uint32_t> expert_pinned_keys;
        std::vector<std::uint32_t> expert_tier_order;
        bool expert_swap = true;
        // The VRAM monitor thread (§4.9) and the headroom it reports pressure below; empty
        // selects the headroom from the display state.
        bool vram_monitor = true;
        std::optional<std::uint64_t> vram_headroom;
        // The SSD tier's sizing inputs: the RAM ledger's page-locked expert share, its pageable
        // share (unpinned records), the pinned allocations made after the tier, and an expert
        // stripe's path (its tails on another drive) with the artifact identity it must match.
        std::uint64_t expert_ram_bytes      = 0;
        std::uint64_t expert_unpinned_bytes = 0;
        std::uint64_t later_pinned_bytes    = 0;
        // Host memory a RAM tier rebuilt beside the current one must leave free (--ram-headroom-mib).
        std::uint64_t ram_headroom_bytes = 0;
        std::filesystem::path expert_stripe;
        artifact::ArtifactId stripe_artifact_id; // the stripe must be of this artifact
        // Tier ring sizing inputs that come from the family:
        bool prefill_stream = false;        // the prefill expert stream exists (bounce slots, read slots)
        std::uint32_t staging_slots = 0;    // the round's expert staging slots (the ring holds them)
        // The CPU miss service's job caps (0: no CPU service; the ring holds its jobs).
        std::uint32_t cpu_workers     = 0;
        std::uint32_t cpu_jobs        = 0;
        std::uint32_t cpu_assist_jobs = 0;
        // The prefill stream's bounce per part and the parts per layer (0: none); the family
        // sizes it from its stream's bounce halves.
        std::uint32_t stream_bounce_per_part = 0;
        std::uint32_t stream_bounce_parts    = 0;
        std::uint32_t stream_bounce_min      = 0;
        std::uint32_t stream_bounce_max      = 0;
        // The layer walk's span (tokens) and the prefill chunk (columns): a walk's step calls
        // ceil(layers / n) layers once per chunk of its n-chunk span.
        std::uint32_t layer_walk_tokens = 0;
        std::uint32_t chunk_columns     = 0;
        // The prefill stream's read slots (0: disable).
        std::uint32_t stream_read_slots = 0;
        // The tier's floor in pinned slots.
        std::uint32_t min_tier_slots = 1024;
    };

    // ---- the family's hooks (startup or between rounds only, never per token) ----
    struct Hooks {
        // The diagnostic sink (every moved log line goes here).
        std::function<void(const std::string& text, DiagnosticLevel level)> diagnostic;
        // A fallback seed order when the saved state is missing or of another artifact (a
        // family's calibrated expert order); empty: none (the fill spreads across the layers).
        std::function<std::optional<expert_cache::SavedState>()> fallback_seed_order;
        // The RAM tier was built, rebuilt or released: `tier` is the new one (null on release).
        // `rebuilt`: an existing tier was replaced -- whatever captured the old tier's records
        // (the prefill expert stream's callbacks) must retarget to the new one.
        std::function<void(HostExpertTier* tier, bool rebuilt)> on_tier_changed;
        // A placement switch (or its resident refill) settled: the family drops state that
        // depended on the previous placement (a failed switch may still have moved experts).
        std::function<void(const char* reason)> on_switch_settled;
        // Whether every execution lane is free (a placement settle runs on an idle engine only).
        std::function<bool()> all_lanes_idle;
        // The elastic KV pool's idle state (the VRAM monitor's snapshot wakes an idle engine).
        std::function<bool()> kv_idle_due;
        // VRAM the KV cache maps now (the placement status reports it).
        std::function<std::uint64_t()> kv_bytes;
        // The monitor's start hands the sizing baseline to the family (the spare-demand inputs
        // its grow_kv reads, moved by its elastic KV mappings afterwards): the demand the pool
        // was sized from and the snapshot at that moment. Called when the pool is elastic.
        std::function<void(const VramDemand& demand, const VramSnapshot& snapshot)> capture_sizing;
    };

    // `device` is the Program's device context (the runtime synchronizes the compute stream
    // between rounds); `vram` is the budget source the frames resize against (the family owns
    // it; it must outlive the runtime).
    ExpertPlacementRuntime(DeviceContext& device, VramBudgetSource& vram, Options options, Hooks hooks);
    ~ExpertPlacementRuntime();
    ExpertPlacementRuntime(const ExpertPlacementRuntime&)            = delete;
    ExpertPlacementRuntime& operator=(const ExpertPlacementRuntime&) = delete;

    // ---- construction (the family's allocation sequence calls these in order) ----
    // The VRAM residency over the banks' device records: `banks[l]` layer l's pinned records
    // (null entries with a tier), `record_stride` the largest record stride, `host_strides` each
    // bank's own (empty: every layer's is record_stride), `max_columns` the most columns one
    // round routes. Pins the placement's keys, marks the resident placement and applies the swap
    // policy's option. Returns the residency (the family wires its tables into its Forward and
    // keeps the pointer; it stays valid for the runtime's lifetime).
    [[nodiscard]] ExpertResidency& create_residency(std::vector<const std::uint8_t*> banks,
                                                    std::uint64_t record_stride, std::int32_t max_columns,
                                                    std::vector<std::uint64_t> host_strides);
    // The boot RAM tier over the artifact's expert store, sized from the options (ring from the
    // staging slots and CPU jobs, pinned slots from the RAM bytes, unpinned records, bounce and
    // stream slots, the layer-walk ring, the expert stripe), with its fetch channel. Attaches the
    // tier to the residency and remembers the store and options a placement switch rebuilds from.
    void create_tier(const ExpertStore& store);
    [[nodiscard]] ops::offloaded_moe::FetchChannel* fetch_channel() const noexcept { return fetch_channel_.get(); }

    // ---- the frames (the family's sizing call, before the warm start) ----
    // Sizes the frame pool from `demand` (the sizing function), resizes the residency to it (an
    // explicit frame target clamps), reports spills and refusals, names the ceiling and starts
    // the VRAM monitor. The family reads `sizing` into its own state and uses `resized` for what
    // the workspace lends from the pools.
    struct FrameStart {
        VramSizing sizing;
        ExpertResidency::Resize resized;
    };
    [[nodiscard]] FrameStart start_expert_frames(const VramDemand& demand);

    // ---- warm start and the saved state (§4.2) ----
    // Fills the frames from the saved state, before the first request; without one, from the
    // fallback seed order's ranking when a hook provides it. With the tier, RAM is filled first
    // (the frames load from it), then the seeds' RAM copies are released and their slots refilled
    // with the next keys of the fill order, so the two levels hold different experts. A resident
    // placement walks the fill order through the tier's free slots in windows, one pass of the
    // artifact, every expert resident before the first request.
    void warm_start_experts();
    // Writes the saved state (at stop, and from maintain_expert_state). Never throws.
    void save_expert_state() noexcept;
    // Between requests: saves when rounds ran since the last save and 10 minutes have passed.
    void maintain_expert_state() noexcept;

    // ---- the placement switches (§4.8) ----
    // A switch requested from any thread, applied by the engine thread at the next maintenance
    // boundary. Validated now: the target against the pools' limit and their pinned keys, the
    // RAM tier's bytes against the host's free memory.
    [[nodiscard]] bool request_placement(const ExpertPlacementRequest& request, std::string* error);
    // The counts form: the current ranking's top N hold VRAM frames and the next M the RAM
    // tier's slots -- adaptive within the counts, never pinned.
    [[nodiscard]] bool request_count_placement(std::uint32_t vram_experts, std::uint32_t ram_experts,
                                               std::string* error);

    // Applies a pending switch, advances the placement grow's resident refill and the RAM tier's
    // fill (the same between-rounds cadence), and settles the placement when the engine is idle
    // (the hooks' all_lanes_idle). Returns whether the caller's elastic KV control law may run
    // after it (false: the pools hold every expert, or an explicit placement names the pool's
    // size and only its switches move it).
    bool apply_placement(bool idle);

    // The placement the Program serves now.
    [[nodiscard]] ExpertPlacementStatus placement_status() const;
    // Writes the current placement in the load format. Returns false when it cannot; `error` why.
    [[nodiscard]] bool export_placement(const std::filesystem::path& path, std::string* error);
    // True when the placement has drifted from its named counts (empty frames, tier shadows or
    // free slots) and the engine idles.
    [[nodiscard]] bool placement_drifted() const;
    // Returns the placement to its named counts: empty frames refilled, then the tier's shadows
    // released and its fill reading the ranking's best SSD-only keys.
    void settle_placement();

    // ---- accessors the family and its hooks use ----
    [[nodiscard]] ExpertResidency* residency() const noexcept { return residency_.get(); }
    [[nodiscard]] HostExpertTier* tier() const noexcept { return tier_.get(); }
    [[nodiscard]] std::uint32_t tier_resident_slots() const noexcept {
        return tier_ != nullptr ? tier_->resident_capacity() : 0;
    }
    // The explicit placement's ceiling (0: automatic); only a switch moves the pool past it.
    [[nodiscard]] std::uint32_t placement_ceiling() const noexcept { return placement_ceiling_; }
    // The monitor's waker: the engine thread's maintenance tick also advances the fills while
    // the engine idles, when no round would apply the placement.
    void set_maintenance_waker(std::function<void()> waker);
    void wake_maintenance();
    // The control path's mutex: the family's elastic KV control resizes the residency with it
    // held, as the runtime's own placement path does. Recursive.
    [[nodiscard]] std::recursive_mutex& vram_mutex() const noexcept { return vram_mutex_; }
    // Why the RAM tier's read agent stopped (empty while it runs); the family's device-error
    // check reports it.
    [[nodiscard]] std::string tier_failure() const { return tier_ != nullptr ? tier_->failure() : std::string(); }
    [[nodiscard]] const Options& options() const noexcept { return options_; }

private:
    // Where the warm start's ranking and counts came from.
    enum class SeedSource { None, Saved, Calibrated };

    // The SSD tier's startup fill: its key order, how many keys the first pass read, and its time.
    struct TierFill {
        std::vector<std::uint32_t> order;
        std::uint32_t filled = 0;
        double seconds       = 0;
        SeedSource source    = SeedSource::None;
    };

    // The RAM tier's fill order (§4.8): the saved state's ranking (the last session's VRAM
    // residents, best first), then every other key that session used, most used first, then the
    // keys it never used one expert of every layer in turn, so that no layer is left wholly on
    // the SSD. An explicit placement's order (its vram set first, then its ram set) overrides it.
    [[nodiscard]] std::vector<std::uint32_t> tier_fill_order(const expert_cache::SavedState* state) const;

    // The tier's first RAM pass, before the frames' warm start reads from it.
    TierFill prefill_tier(const expert_cache::SavedState* state, SeedSource source);

    // After the frames' warm start: the seeds are in VRAM, so their RAM copies (shadows) are
    // released and their slots take the next keys of the fill order. A seed VRAM later evicts is
    // demoted into RAM when it outranks the coldest resident, as any key without a host copy.
    void finish_tier_fill(const TierFill& fill, std::uint32_t seeded);

    // The frames' warm start from the saved state or the fallback seed order; returns how many
    // experts it loaded.
    std::uint32_t warm_start_frames(const expert_cache::ExpertStateLoad& load, SeedSource source);

    // A seeded count is capped, so an expert used heavily in an old session yields to new uses.
    static constexpr std::uint32_t kSeedCountCap = 16;
    // The share of the frames a warm start fills (the best-ranked experts); the rest stay free
    // for the fill phase's landing, so a workload unlike the saved state still fills quickly.
    static constexpr double kSeedShare = 0.5;
    static constexpr std::chrono::minutes kExpertStateInterval{10};
    std::chrono::steady_clock::time_point expert_state_saved_{};
    std::uint64_t expert_state_routed_ = 0;

    // The VRAM monitor thread: wakes an idle engine so it returns idle KV pages to the frames.
    // Free VRAM below the headroom is only reported -- the engine does not resize the expert
    // cache under another program's pressure; the client owns the placement.
    void start_vram_monitor(const VramDemand& demand);
    void on_vram_snapshot(const VramSnapshot& snapshot);

    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const
        noexcept;
    [[nodiscard]] std::uint32_t keys() const noexcept { return options_.geometry.banks * options_.geometry.experts_per_bank; }

    void record_switch(bool ok, std::string message);

    // Engine thread, at a maintenance boundary: resize to the target and re-pin, with the RAM
    // tier the switch names in place first -- a shrink's demotions land in its slots. A shrink
    // evicts each pool's lowest-score residents (pinned keys never), demoting them through the
    // RAM tier when it has room and dropping the rest to the artifact (the tier fill backfills
    // them); a grow admits from use and refills between rounds. Freed VMM chunks release their
    // VRAM. A switch that names no RAM tier while holding every key in VRAM releases it once
    // the grow's refill completes.
    void apply_pending_placement();

    // Freed pageable tier memory stays in the allocator's arenas unless handed back.
    static void trim_host_heap();

    // Builds or rebuilds the tier to hold `residents` resident records beside its lists (the
    // read ring, prefetch and demotion -- the boot derivation's). Engine thread, at a maintenance
    // boundary. An existing tier's records seed the new one best-first when the new one fits in
    // the host's free memory beside it, else the old releases first; the rest fill by the tier
    // fill and demand below. The old tier's pinned chunks release with it.
    // With `background`, the rebuild runs as a TierRebuild instead and ensure_tier returns at once.
    void ensure_tier(std::uint32_t residents, bool background = false);

    // A RAM tier rebuild off the engine thread. Pinning tens of GiB, copying the kept records and
    // releasing the old tier's chunks took 17 s or more on the engine thread (a 86 -> 60 GiB
    // shrink), past the Engine's queue timeout. Two forms:
    // - Seeded, when the new tier fits beside the old one with the RAM headroom left free: a
    //   worker pins the new tier (kPinning); at the next boundary the engine thread picks the kept
    //   keys best-first and rebuild-pins their slots in the old tier, which keeps serving
    //   (HostTier::rebuild_pin); the worker copies them (kCopying); at the boundary after that the
    //   engine thread quiesces the old tier and retiers to the new one (milliseconds), and the old
    //   tier is destroyed on tier_reaper_.
    // - Fresh, otherwise: the engine thread swaps in a reader-sized tier (its lists and one slot,
    //   which SSD reads stage through) and hands the old tier to the worker, which destroys it
    //   before it pins the new one, so the two never coexist; the next boundary installs the new
    //   tier empty and the tier fill reads its records.
    struct TierRebuild {
        enum class Phase : std::uint8_t { kIdle, kPinning, kPinned, kCopying, kCopied, kFailed };
        std::atomic<Phase> phase{Phase::kIdle};
        std::uint32_t residents = 0;
        bool fresh = false;
        std::unique_ptr<HostExpertTier> old; // fresh: the replaced tier, destroyed by the worker
        std::unique_ptr<HostExpertTier> next;
        std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> copies;
        std::string error;
        std::thread worker;
        std::chrono::steady_clock::time_point begin{};
    } tier_rebuild_;
    // The pending rebuild's resident target for the status (-1: none).
    std::atomic<std::int64_t> tier_rebuild_target_{-1};
    std::thread tier_reaper_;

    // The rebuilt tier's options for `residents` resident slots (no pageable records).
    [[nodiscard]] HostExpertTier::Options rebuild_options(std::uint32_t residents) const;
    // The old tier's keys best-first that `next` takes, prefilled into its controller: (slot in
    // next, record in the old tier) for each.
    [[nodiscard]] std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> kept_records(HostExpertTier& next,
                                                                                           std::uint32_t residents);
    // Retiers to `next` in place of the current tier (engine thread, the old tier quiesced) and
    // returns the old one.
    [[nodiscard]] std::unique_ptr<HostExpertTier> swap_tier(std::unique_ptr<HostExpertTier> next);
    // swap_tier, then retires the old tier on the reaper.
    void install_tier(std::unique_ptr<HostExpertTier> next, std::size_t kept);
    // Advances a background rebuild at a maintenance boundary.
    void advance_tier_rebuild();
    // Abandons a background rebuild (waiting for its worker) and releases its pins.
    void cancel_tier_rebuild();
    // Destroys a replaced tier off the engine thread (one at a time).
    void retire_tier(std::unique_ptr<HostExpertTier> old);

    // The host memory a tier of `residents` resident records and its lists takes (an estimate
    // in frame sizes).
    [[nodiscard]] std::uint64_t tier_bytes(std::uint32_t residents) const noexcept {
        return (static_cast<std::uint64_t>(residents) + 1536) * residency_->frame_bytes();
    }

    // The resident placement still leaves one key without a frame (the pools hold one key fewer
    // than the set), and artifact reads stage through the tier's lists: keep only those lists and
    // one slot, caching no experts in RAM.
    void shrink_tier_to_reader(const char* why);

    // Releases the tier: every key is resident in a frame (or read from the artifact), so no host
    // expert bytes remain. The record table republishes empty before the pinned chunks release.
    void release_tier(const char* why);

    // The tier fill: a rebuild or a demotion-dropping shrink leaves free resident slots; the
    // ranking's keys that are neither in VRAM nor held read into them in bounded batches between
    // rounds (the placement refill's cadence), so the tier converges on its named capacity.
    struct TierBackfill {
        std::vector<std::uint32_t> order; // keys in read order (the ranking, VRAM keys skipped)
        std::size_t at  = 0;              // cursor into `order`
        std::uint32_t stall = 0;          // consecutive ticks without a read
        bool active = false;
    } tier_backfill_;

    void start_tier_fill();

    // One bounded slice of the tier fill, at a maintenance boundary: read the next keys of the
    // order that hold neither a frame nor a host copy. Ends when the tier is full or two ticks
    // pass without a read.
    void advance_tier_fill();

    // A placement grow's resident refill: the switch backs new frames but does not fill them --
    // boot's warm start does -- so without a refill the keys arrive one miss at a time from the
    // artifact. The refill walks the usage-rank order in bounded batches between rounds: seed
    // every key the tier holds, free its slots, read the next batch (the boot fill's window
    // walk, sliced so no round waits on it). `stall` counts ticks without progress: two end it.
    struct PlacementFill {
        std::vector<std::uint32_t> order;   // keys in fill order (usage rank)
        std::size_t at = 0;                 // cursor into `order`
        expert_cache::SavedState window{};  // counts for seeding; `ranked` holds the tier's keys
        std::uint32_t seeded = 0;           // keys seeded so far
        std::uint32_t total  = 0;           // keys the walk covers (progress reporting)
        std::uint32_t stall  = 0;           // consecutive ticks without a seed or a read
        std::uint32_t passes = 0;           // walks of the order so far (keys mid-copy are retried)
        bool active = false;
        std::chrono::steady_clock::time_point begin{};
        // A grow from a tier smaller than one read batch stages through a temporary one; the
        // RAM target the switch named is restored when the fill finishes.
        bool restore_tier = false;
        std::uint32_t tier_after = 0;
        bool after_switch = false; // a placement switch's grow (else frames a lease left empty)
    } placement_fill_;
    static constexpr std::uint32_t kRefillBatch = 512; // keys read per tick (~0.1 s of artifacts)
    // The last applied switch's outcome for the status (placement_mutex_).
    bool placement_last_ok_ = true;
    std::string placement_last_message_;

    // The RAM tier's lifecycle state (the placement switches own it). The boot tier's store
    // (with its stripe) and options: a rebuild clones them with a new resident-slot count.
    // `placement_wants_no_tier_`: a pending switch grows to every key resident in VRAM naming no
    // RAM tier; the tier releases once that grow's refill completes.
    const ExpertStore* tier_store_           = nullptr;
    HostExpertTier::Options tier_options_{};
    bool placement_wants_no_tier_     = false;

    // A switch requested from any thread, applied by the engine thread at the next maintenance
    // boundary, where no round's kernels are in flight.
    mutable std::recursive_mutex vram_mutex_;
    mutable std::mutex placement_mutex_;
    ExpertPlacementRequest placement_pending_;
    bool placement_pending_set_ = false;

    // Starts the resident refill after a placement grow: the usage-rank order to walk (the
    // request's, else the session's saved ranking and the tier fill's spread), the seeding window
    // the tier's keys fill, and the maintenance cadence that advances it between rounds and while
    // the engine idles.
    void start_placement_fill(const std::vector<std::uint32_t>& request_order);

    // One bounded slice of the resident refill, at a maintenance boundary (no round's kernels in
    // flight): seed every key the tier holds and free its slots, then read the next batch of the
    // order -- only keys still without a frame. Ends when every frame is backed (the pools hold
    // one key fewer than the set at the limit), or after two ticks without progress.
    void advance_placement_fill();

    void finish_placement_fill();

    // The maintenance waker: a tick that is not done re-arms it, so a refill also advances while
    // the engine idles, when no round would apply the placement.
    std::function<void()> waker_;
    // The explicit placement's frame ceiling, updated by each applied placement switch.
    std::uint32_t placement_ceiling_ = 0;
    bool pressure_warned_     = false;

    DeviceContext& device_;
    VramBudgetSource& vram_;
    Options options_;
    Hooks hooks_;
    std::unique_ptr<VramMonitor> monitor_;
    std::unique_ptr<ExpertResidency> residency_;
    // The striped store and the fetch channel outlive the tier: its agent reads both until joined.
    std::unique_ptr<ExpertStore> striped_store_; // the boot tier's records when a stripe is named
    std::unique_ptr<ops::offloaded_moe::FetchChannel> fetch_channel_;
    std::unique_ptr<HostExpertTier> tier_;
    // tier_'s resident capacity (0: none), for the switch requests the server thread checks.
    std::atomic<std::uint32_t> tier_capacity_{0};
    std::uint64_t record_stride_ = 0;            // the largest record stride (create_residency's)
};

} // namespace infernix::models::offload
