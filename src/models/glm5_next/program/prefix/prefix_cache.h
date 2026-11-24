#pragma once

// Physical binding of the generic prefix index (runtime/prefix_cache) to Glm5Next's KV page pool,
// lane state and pinned Host slab pool (docs/maintainer/glm5_3-flash-design.md §12). The index
// decides; this class holds what its opaque ids name and moves the bytes its decisions imply.
//
// - A block id owns one Device page lease: a cached full block, or a snapshot's Device tail. Lanes
//   map cached pages into their block tables without a lease of their own; the index's path pins
//   keep a mapped page's node (and so its lease) alive while any lane maps it.
// - Snapshots are Host-born: an image (and a tail page, when the frontier is not a block boundary)
//   is copied straight from a lane to reserved Host slabs, then published. The index has no Device
//   snapshot slots and no Disk tier (the Host tier is this family's only snapshot residency).
// - Host writes run on the transfer stream after a producer event; restores run on the restore
//   stream in forward order, one event per decoder layer and one for the drafter group, so a lane's
//   first call waits for each layer's state just before that layer reads it.

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "infernix/types.h"
#include "models/glm5_next/program/prefix/prefix_index.h"
#include "models/glm5_next/program/prefix/state_image.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernix::models::glm5_next::prefix {

struct PrefixCacheCounters {
    std::uint64_t admissions          = 0;
    std::uint64_t reused_tokens       = 0;
    std::uint64_t blocks_inserted     = 0;
    std::uint64_t blocks_reattached   = 0;
    std::uint64_t blocks_duplicate    = 0;
    std::uint64_t taps_created        = 0;
    std::uint64_t taps_skipped        = 0;
    std::uint64_t prompt_end_captures = 0; // the zero-token resume's prompt-end snapshots
    // Prompt ends not captured: gated by design (within one tap separation of the deepest
    // published frontier), or failed for want of a staging slot, Host slabs or a free page.
    std::uint64_t prompt_end_gated    = 0;
    std::uint64_t prompt_end_failed   = 0;
    std::uint64_t host_image_writes   = 0;
    std::uint64_t host_block_writes   = 0;
    std::uint64_t host_image_restores = 0;
    std::uint64_t host_block_restores = 0; // prefetched blocks included
    std::uint64_t prefetched_blocks   = 0;
    std::uint64_t host_tail_restores  = 0;
    std::uint64_t host_write_bytes    = 0;
    std::uint64_t host_restore_bytes  = 0;
};

// Geometry of the Host tier: one slab holds one page record (the MLA latent and pooled planes of
// all layers, 4 KiB aligned); an image spans image_slabs slabs and a snapshot tail one more.
struct PrefixHostLayout {
    HostKVPageLayout page;
    std::size_t slab_bytes        = 0;
    std::size_t image_bytes       = 0;
    std::uint32_t image_slabs     = 0;
    std::uint32_t slabs           = 0; // 0: no Host tier
    std::uint32_t slabs_per_chunk = 0;
};

[[nodiscard]] PrefixHostLayout plan_prefix_host_layout(const KvPageGeometry& pages, const StateImageLayout& image,
                                                       std::uint64_t host_bytes);

struct PrefixCacheConfig {
    prefix::cache_index::PrefixIndexConfig index; // capacities filled by the cache from the layouts
    runtime::prefix_cache::TapPlannerConfig taps;
    std::uint32_t kv_pages = 0; // Device page groups of the pool
};

// One admission's restore: Host pages into new pages, an image into a lane, then the events a
// lane's first call waits on (one per decoder layer, then the drafter group).
struct RestoreTicket {
    std::uint64_t id = 0;
};

class PrefixCache final : public prefix::cache_index::PrefixIndexBackend {
public:
    // `host_bytes` sizes the Host tier (it is needed: snapshots are born there).
    PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const KvPageGeometry& pages,
                const LaneStateImage& lanes, std::uint64_t host_bytes);
    ~PrefixCache() override;

    PrefixCache(const PrefixCache&)            = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] prefix::cache_index::PrefixCacheIndex& index() noexcept { return *index_; }
    [[nodiscard]] const prefix::cache_index::PrefixCacheIndex& index() const noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::TapPlannerConfig& taps() const noexcept { return taps_; }
    [[nodiscard]] PrefixCacheCounters& counters() noexcept { return counters_; }
    [[nodiscard]] const PrefixCacheCounters& counters() const noexcept { return counters_; }
    [[nodiscard]] const PrefixHostLayout& host_layout() const noexcept { return host_; }
    [[nodiscard]] bool host_tier() const noexcept { return host_.slabs != 0; }
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept;
    // The compute stream, which every capture, write and restore names as its producer.
    void set_compute_stream(cudaStream_t stream) noexcept { producer_ = stream; }

    // ---- blocks ------------------------------------------------------------------------------
    // Inserts a committed full block whose page is `lease`. When the index keeps it (a new node,
    // or a Host-only node adopting it), the lease moves into the cache and the result reports
    // device_attached; otherwise `lease` is left with the caller.
    [[nodiscard]] prefix::cache_index::InsertResult insert_block(prefix::cache_index::NodeRef parent,
                                                                 std::uint64_t lookup_hash,
                                                                 std::span<const TokenId> tokens,
                                                                 std::uint64_t extra, DeviceKVPageLease& lease,
                                                                 bool attach);
    // The Device page of a Device-resident block or snapshot tail.
    [[nodiscard]] DeviceKVPageHandle page(std::uint32_t device_id) const;
    // Adopts a lane's partial page as a snapshot's Device tail; returns its device id.
    [[nodiscard]] std::uint32_t adopt_tail(DeviceKVPageLease&& lease);

    // ---- snapshot meta -----------------------------------------------------------------------
    [[nodiscard]] const StateImageHeader& meta(prefix::cache_index::SnapshotRef snapshot) const;

    // ---- Host-born captures ------------------------------------------------------------------
    // Copies lane `lane`'s state image (and, with a tail, the partial page `tail_page`) to reserved
    // Host slabs after `producer` reaches the current point, in forward order, and publishes the
    // snapshot when the copy lands (poll). The lane's next writes to a layer's state, and to that
    // layer's planes of `tail_page`, wait on that layer's event (capture_events).
    struct Capture {
        std::uint64_t id = 0; // the caller's name for capture_result
        prefix::cache_index::NodeRef anchor;
        std::uint32_t frontier = 0;
        std::vector<TokenId> tail;
        std::optional<DeviceKVPageHandle> tail_page;
        std::optional<DeviceKVPageLease> tail_lease; // handed over as the Device tail when present
        prefix::cache_index::SnapshotKind kind = prefix::cache_index::SnapshotKind::Tap;
        double claim = 0.0; // the new snapshot's estimate_priority
        StateImageHeader meta;
        std::optional<prefix::cache_index::SnapshotRef> supersedes;
        // When `supersedes` is the snapshot the lane resumed from: the lane's tokens from its frontier
        // on, so the snapshots the lineage went past are superseded too (PrefixCacheIndex::supersede).
        std::vector<TokenId> continuation;
    };
    // Returns false when no slabs could be reserved for its claim (the capture is skipped).
    [[nodiscard]] bool capture(std::uint32_t lane, Capture&& capture, cudaStream_t producer);
    // The zero-token resume's prompt-end capture: a snapshot at the prompt's own frontier whose
    // image carries the head's saved residual, so an exact repeat regenerates its first token
    // without prefill. The lane's next round is a graph-replayed decode that cannot wait on
    // capture events, so the image stages through a fixed Device slot on `producer` (stream-
    // ordered before that round) and the Host copy reads the slot, never the live state.
    // `capture.tail_page` must be the caller's private copy of the lane's tail page (its lease
    // moves to the snapshot). Returns false when no staging slot or Host slabs could be had.
    [[nodiscard]] bool capture_prompt_end(std::uint32_t lane, Capture&& capture, cudaStream_t producer);
    // Drops a capture that will never publish: waits for its copy and returns its slabs, its
    // staging slot and its tail page.
    void abandon_capture(std::uint64_t id) noexcept;
    // A capture staged inside a prefill layer walk (prefill_walk.cpp): its Host slabs are reserved
    // at open (absent: none for `claim`), each layer group is copied once the walk has enqueued
    // that layer of the tap's chunk (`producer` reaches that point), and commit publishes it as
    // capture() would once the last group lands. Block-aligned frontiers only (no tail page).
    [[nodiscard]] std::optional<std::uint64_t> open_staged(double claim);
    void capture_staged_group(std::uint64_t staged, std::uint32_t group, std::uint32_t lane,
                              cudaStream_t producer);
    // Group `group`'s copy-out event once captured (the lane's next writes of it wait), else null.
    [[nodiscard]] cudaEvent_t staged_event(std::uint64_t staged,
                                           std::uint32_t group) const noexcept;
    // Every group must have been captured.
    void commit_staged(std::uint64_t staged, Capture&& capture);
    void abandon_staged(std::uint64_t staged) noexcept;
    [[nodiscard]] std::uint32_t image_groups() const noexcept;
    // The fixed Device staging slots and their per-slot pinned headers (the Program's
    // preallocated packed images, one per concurrent prompt-end capture).
    void set_prompt_end_staging(std::vector<std::byte*> slots);
    // The capture's copy-out events in layer order (decoder layers, then drafter): each is recorded
    // once that layer's state (and its planes of the tail page) has been read. Empty once landed.
    [[nodiscard]] std::span<const cudaEvent_t> capture_events(std::uint64_t id) const noexcept;
    // Orders the compute stream (every later writer of a pool page runs on it or after it) after
    // the pending captures that read one of `pages` as their tail, before those pages return to the
    // pool: a tap reads the lane's page on the transfer stream, and a page reused under that copy
    // would put another request's KV into the snapshot. Free once the copies have landed.
    void order_page_reuse(std::span<const DeviceKVPageLease> pages) const;
    // Copies every Device-only node of `nodes` to a Host slab, after `producer`'s queued work.
    void write_blocks(std::span<const prefix::cache_index::NodeRef> nodes, cudaStream_t producer);
    // Publishes landed captures and writes and retires landed restores. Non-blocking.
    void poll();
    // Waits for every capture copy, Host write and restore in flight and publishes them.
    void drain();
    [[nodiscard]] bool transfers_pending() const noexcept {
        return !captures_.empty() || !writes_.empty() || !landing_.empty();
    }
    // The publication of a landed capture (absent while it is in flight); forget it when done.
    [[nodiscard]] std::optional<prefix::cache_index::PublishResult> capture_result(std::uint64_t id) const;
    void forget_capture(std::uint64_t id) noexcept;

    // ---- restores (one admission staged at a time) --------------------------------------------
    void open_restore(cudaStream_t producer);
    // A Host-only block into a new cache page (the lease moves into the cache; node Filling).
    void restore_block(prefix::cache_index::NodeRef node, DeviceKVPageLease&& destination);
    // A block's Host page into a lane's private page (a copy-on-write anchor), the block pinned
    // until the batch lands.
    void restore_node_page(prefix::cache_index::NodeRef node, DeviceKVPageHandle destination);
    // A snapshot's Host tail page into a lane's private page.
    void restore_tail_page(prefix::cache_index::SnapshotRef snapshot, DeviceKVPageHandle destination);
    // A snapshot's Host image into a lane.
    void restore_image(prefix::cache_index::SnapshotRef snapshot, std::uint32_t lane);
    // Enqueues the batch per decoder layer (that layer's page planes or state parts), then the
    // drafter group, recording one event after each. Returns the ticket the lane's calls wait with.
    [[nodiscard]] RestoreTicket submit_restore();
    [[nodiscard]] bool restore_open() const noexcept { return restore_.open; }
    // The batch's events in layer order (decoder layers, then drafter); empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> restore_events(RestoreTicket ticket) const noexcept;
    // Orders `consumer` after every copy of a landing batch.
    void order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const;
    // Waits on the host for a landing batch (and every batch before it) and retires what landed.
    void await_restore(RestoreTicket ticket);
    void abort_restore() noexcept;

    // Releases every reference the cache holds; the index is rebuilt empty.
    void clear() noexcept;

    // PrefixIndexBackend
    void release_device_block(std::uint32_t device_id) noexcept override;
    void release_snapshot(prefix::cache_index::SnapshotRef snapshot) noexcept override;
    void drop_snapshot_device_image(prefix::cache_index::SnapshotRef snapshot) noexcept override;
    void release_disk_record(std::span<const std::uint32_t> slots) noexcept override;
    void disk_record_replaced(prefix::cache_index::NodeRef node) noexcept override;

private:
    struct ImageTarget {
        prefix::cache_index::HostImageReservation host;
    };
    struct PendingCapture {
        Capture capture;
        ImageTarget image;
        std::vector<cudaEvent_t> events; // decoder layers, then drafter
        // The prompt-end staging slot this capture reads; a slot is free once no pending capture
        // names it. Null for the whole-image captures, and once the copy has landed.
        std::byte* staging = nullptr;
    };
    // The image's segments, as HostImage takes them, and its tail record and that record's group.
    [[nodiscard]] std::vector<std::byte*> image_segments(const ImageTarget& image) const;
    [[nodiscard]] std::byte* tail_record(const ImageTarget& image) const;
    [[nodiscard]] std::uint32_t tail_group(const ImageTarget& image) const;
    // Reserves where a capture lands (Host slabs); absent when none.
    [[nodiscard]] std::optional<ImageTarget> reserve_image(bool tail, double claim);
    void release_image(ImageTarget&& image) noexcept;
    // Writes the header on the Host directly.
    void write_header(const ImageTarget& image, const StateImageHeader& meta, cudaStream_t stream);
    // Group g's copy-out (its MLA layer's planes of a tail page first) on the transfer stream.
    void copy_tail_page_to_record(DeviceKVPageHandle page, std::byte* record, std::uint32_t group,
                                  cudaStream_t stream) const;
    void copy_capture_group(std::uint32_t group, std::uint32_t lane, const HostImage& destination,
                            std::byte* tail_record, std::uint32_t tail_group,
                            std::optional<DeviceKVPageHandle> tail_page);
    // Copies a prompt-end capture's slot into its image on the transfer stream, after `after`.
    void copy_prompt_end(PendingCapture& c, cudaEvent_t after);
    struct PendingWrite {
        std::vector<prefix::cache_index::NodeRef> nodes;
        cudaEvent_t done = nullptr;
    };
    struct PageCopy {
        DeviceKVPageHandle page;
        std::byte* record = nullptr;
        std::uint32_t group = 0;
    };
    struct RestoreBatch {
        bool open = false;
        RestoreTicket ticket;
        std::vector<prefix::cache_index::NodeRef> nodes; // Filling until the batch lands
        std::vector<prefix::cache_index::NodeRef> pins;
        std::vector<prefix::cache_index::SnapshotRef> snapshot_pins;
        std::vector<PageCopy> pages;
        std::optional<prefix::cache_index::SnapshotRef> image;
        std::uint32_t image_lane = 0;
        std::vector<cudaEvent_t> events; // decoder layers, then drafter
    };

    [[nodiscard]] std::uint32_t allocate_block_id(DeviceKVPageLease&& lease);
    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept;
    [[nodiscard]] std::uint32_t slab_group(std::uint32_t id) const noexcept { return id / host_.slabs_per_chunk; }
    [[nodiscard]] cudaEvent_t take_event();
    void recycle(cudaEvent_t event) noexcept;
    void finish_restore(RestoreBatch& batch) noexcept;
    void copy_pages_from_host(std::span<const PageCopy> copies, std::size_t plane_begin, std::size_t plane_end,
                              cudaStream_t stream) const;
    void publish_capture(PendingCapture& c);

    runtime::prefix_cache::TapPlannerConfig taps_;
    DeviceKVPagePool* pool_ = nullptr;
    KvPageGeometry geometry_;
    const LaneStateImage* lanes_ = nullptr;
    PrefixHostLayout host_;
    std::vector<PinnedHostBuffer> chunks_;
    cudaStream_t transfer_ = nullptr;
    // The compute stream (the producer of every capture, write and restore).
    cudaStream_t producer_ = nullptr;
    cudaStream_t restore_stream_ = nullptr;
    std::vector<std::optional<DeviceKVPageLease>> blocks_;
    std::vector<std::uint32_t> free_blocks_;
    std::vector<StateImageHeader> snapshot_meta_; // by snapshot index
    std::deque<PendingCapture> captures_;

    struct StagedCapture {
        std::uint64_t id = 0;
        ImageTarget image;
        std::vector<cudaEvent_t> events; // per group, null until captured
    };

    std::vector<StagedCapture> staged_;
    std::uint64_t next_staged_ = 0;
    std::deque<PendingWrite> writes_;
    std::unordered_map<std::uint64_t, prefix::cache_index::PublishResult> published_;
    RestoreBatch restore_;
    std::deque<RestoreBatch> landing_;
    std::uint64_t next_ticket_ = 1;
    std::vector<cudaEvent_t> spare_events_;
    std::vector<std::byte*> prompt_end_slots_;         // the fixed staging images (set_prompt_end_staging)
    std::vector<PinnedHostBuffer> prompt_end_headers_; // one pinned header per slot, its capture's own
    PrefixCacheCounters counters_;

    std::optional<prefix::cache_index::PrefixCacheIndex> index_;
};

} // namespace infernix::models::glm5_next::prefix
