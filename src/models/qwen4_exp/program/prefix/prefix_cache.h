#pragma once

// Physical binding of the generic prefix index (runtime/prefix_cache) to Qwen4Exp's KV page pool,
// lane state, pinned Host slab pool and Disk tier (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.1). The index decides; this class holds what its opaque ids name and moves the bytes its
// decisions imply.
//
// - A block id owns one Device page lease: a cached full block, or a snapshot's Device tail. Lanes
//   map cached pages into their block tables without a lease of their own; the index's path pins
//   keep a mapped page's node (and so its lease) alive while any lane maps it.
// - Snapshots are Host-born: an image (and a tail page, when the frontier is not a block boundary)
//   is copied straight from a lane to reserved Host slabs, then published. Optional packed Device
//   slots hold images on the Device when the index has slots.
// - Host writes run on the transfer stream after a producer event; restores run on the restore
//   stream in forward order, one event per decoder layer and one for the MTP group, so a lane's
//   first call waits for each layer's state just before that layer reads it.
// - The Disk tier (disk_tier.cpp, --prefix-cache-disk) holds slots of one Host slab each in a
//   DiskStore and backs the memory tiers (hybrid-prefix-cache-spec §17.1): with a Host tier, a
//   writer copies every retained snapshot and its path to it, held and most valuable first, so the
//   Host tier can drop those copies for nothing; without one, every capture lands in a VRAM staging
//   region and is written through, and the writer follows with its path. A Disk-only source is
//   fetched before its admission (fetch): into Host fills, or without a Host tier blocks into new
//   cache pages and the image and tail into a staging region the restore reads. All Disk I/O is
//   cuFile batch I/O polled from poll(), which the Engine also runs while idle.

#include "core/arena.h"
#include "core/cufile_io.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "models/qwen4_exp/program/prefix/disk_store.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "infernix/types.h"
#include "models/qwen4_exp/program/prefix/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infernix::models::qwen4_exp::prefix {

struct PrefixCacheCounters {
    std::uint64_t admissions                  = 0;
    std::uint64_t reused_tokens               = 0;
    std::uint64_t blocks_inserted             = 0;
    std::uint64_t blocks_reattached           = 0;
    std::uint64_t blocks_duplicate            = 0;
    std::uint64_t taps_created                = 0;
    std::uint64_t taps_skipped                = 0;
    std::uint64_t prompt_end_captures         = 0; // the zero-token resume's prompt-end snapshots
    // Prompt ends not captured: gated by design (within one tap separation of the deepest
    // published frontier, an excluded frontier, drafter cells not written, unpublished blocks), or
    // failed for want of a staging slot, Host slabs or a free page.
    std::uint64_t prompt_end_gated            = 0;
    std::uint64_t prompt_end_failed           = 0;
    std::uint64_t endpoints_created           = 0;
    std::uint64_t endpoint_skipped_mtp        = 0;
    std::uint64_t endpoint_mismatch_fallbacks = 0;
    std::uint64_t mtp_continuation_mismatches = 0;
    std::uint64_t mtp_branch_mismatches       = 0;
    std::uint64_t host_image_writes           = 0;
    std::uint64_t host_block_writes           = 0;
    std::uint64_t host_image_restores         = 0;
    std::uint64_t host_block_restores         = 0; // prefetched blocks included
    std::uint64_t prefetched_blocks           = 0;
    std::uint64_t host_tail_restores          = 0;
    std::uint64_t host_write_bytes            = 0;
    std::uint64_t host_restore_bytes          = 0;
    std::uint64_t disk_image_writes           = 0; // snapshots spilled or written through
    std::uint64_t disk_block_writes           = 0;
    std::uint64_t disk_write_bytes            = 0;
    std::uint64_t disk_fetches                = 0; // sources read from the Disk tier
    std::uint64_t disk_image_reads            = 0;
    std::uint64_t disk_block_reads            = 0;
    std::uint64_t disk_read_bytes             = 0;
    std::uint64_t disk_failures               = 0; // failed transfers, and fetched images whose header differs
    std::uint64_t disk_capture_skips          = 0; // no staging region or Disk slots for a capture (no Host tier)
    std::uint64_t disk_spill_skips            = 0; // writer jobs ended for want of Disk slots or a source copy
};

// The Disk tier at startup (attach_disk).
struct DiskAttachResult {
    std::string reset_reason; // why it started empty; empty when its journal was trusted
    bool damaged = false;     // replay stopped at a damaged record
    std::uint64_t blocks            = 0; // rebuilt into the index
    std::uint64_t snapshots         = 0;
    std::uint64_t dropped_blocks    = 0; // replayed but not rebuilt
    std::uint64_t dropped_snapshots = 0;
    std::uint64_t bytes             = 0; // the slots the rebuilt records hold
    double seconds                  = 0.0;
    CuFileQueue::Mode mode          = CuFileQueue::Mode::Compatibility;
};

// The shutdown spill (spill_for_shutdown).
struct DiskSpillResult {
    bool ok = false;
    std::string message; // why it stopped early
    std::uint64_t snapshots = 0;
    std::uint64_t blocks    = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
};

// Geometry of the Host tier: one slab holds one page record (931,840 bytes with INT8 KV, 4 KiB
// aligned); an image spans image_slabs slabs and a snapshot tail one more.
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
    std::uint32_t kv_pages             = 0; // Device page groups of the pool
    std::uint32_t device_snapshot_slots = 0;
};

// One admission's restore: Host pages into new pages, an image into a lane, then the events a
// lane's first call waits on (one per decoder layer, then the MTP group).
struct RestoreTicket {
    std::uint64_t id = 0;
};

class PrefixCache final : public prefix::cache_index::PrefixIndexBackend {
public:
    // `host_bytes` sizes the Host tier and `disk_bytes` the Disk tier (0: none); one of them is
    // needed. The Disk tier is used once attached (attach_disk).
    PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const KvPageGeometry& pages,
                const LaneStateImage& lanes, std::uint64_t host_bytes, std::uint64_t disk_bytes = 0);
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
    [[nodiscard]] std::uint32_t disk_slots() const noexcept { return disk_slots_; }
    // The compute stream, which every capture, write and restore names as its producer.
    void set_compute_stream(cudaStream_t stream) noexcept { producer_ = stream; }
    // The Device bytes of the Disk tier's staging regions and block records.
    [[nodiscard]] std::uint64_t disk_staging_bytes() const noexcept;

    // ---- blocks ------------------------------------------------------------------------------
    // Inserts a committed full block whose page is `lease`. When the index keeps it (a new node,
    // or a Host-only node adopting it), the lease moves into the cache and the result reports
    // device_attached; otherwise `lease` is left with the caller.
    [[nodiscard]] prefix::cache_index::InsertResult insert_block(prefix::cache_index::NodeRef parent,
                                                                   std::uint64_t lookup_hash,
                                                                   std::span<const TokenId> tokens,
                                                                   std::uint64_t extra, DeviceKVPageLease& lease,
                                                                   bool attach, std::int32_t mtp_next);
    // The token a node's last MTP cell encodes (history[64(b+1)] of its publisher); -1 when unknown.
    [[nodiscard]] std::int32_t mtp_next(prefix::cache_index::NodeRef node) const;
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
        // A prompt-end capture on a block boundary under MTP: the anchor block holds the pending
        // drafter cell, so the lane publishes it only after the first decode round has written that
        // cell. The copy runs now; publication waits for resolve_capture_anchor.
        bool anchor_pending    = false;
        std::uint32_t frontier = 0;
        std::vector<TokenId> tail;
        std::optional<DeviceKVPageHandle> tail_page;
        std::optional<DeviceKVPageLease> tail_lease; // handed over as the Device tail when present
        prefix::cache_index::SnapshotKind kind = prefix::cache_index::SnapshotKind::Tap;
        double claim = 0.0; // the new snapshot's estimate_priority
        StateImageHeader meta;
        std::optional<prefix::cache_index::SnapshotRef> supersedes;
        // When `supersedes` is the snapshot the lane resumed from: the lane's tokens from its frontier
        // on, so the endpoints the lineage went past are superseded too (PrefixCacheIndex::supersede).
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
    // Names the anchor of a capture taken with `anchor_pending` (the block the capturing lane has
    // now published); the capture publishes once it has landed.
    void resolve_capture_anchor(std::uint64_t id, prefix::cache_index::NodeRef anchor);
    // Drops a capture whose anchor will never be published (the lane ended before the block):
    // waits for its copy and returns its slabs, its staging slot and its tail page.
    void abandon_capture(std::uint64_t id) noexcept;
    // The fixed Device staging slots and their per-slot pinned headers (the Program's
    // preallocated packed images, one per concurrent prompt-end capture).
    void set_prompt_end_staging(std::vector<std::byte*> slots);
    // A capture taken group by group, for a layer walk that passes the frontier inside a span
    // (prefill_walk.cpp): the walk reaches the frontier in each layer at another point of its queue,
    // so each group (with that layer's planes of the tail page) is copied once the walk has enqueued
    // that layer of the frontier's chunk, and the capture is committed after the span. Opening
    // reserves the slabs (nothing when none could be reserved: the capture is skipped) and the
    // events the lane's next writes to each group wait on (staged_events, layer order).
    [[nodiscard]] std::optional<std::uint64_t> open_staged_capture(bool tail, double claim);
    void capture_group(std::uint64_t staged, std::uint32_t group, std::uint32_t lane,
                       std::optional<DeviceKVPageHandle> tail_page, cudaStream_t producer);
    [[nodiscard]] std::span<const cudaEvent_t> staged_events(std::uint64_t staged) const;
    // Every group copied; `capture` as `capture` takes it (claim and tail as opened). Published when
    // the copies land (poll), in commit order with every other capture.
    void commit_staged_capture(std::uint64_t staged, Capture&& capture);
    // A staged capture that will not be committed: its slabs return once its copies have landed.
    void abandon_staged_capture(std::uint64_t staged) noexcept;
    // The capture's copy-out events in layer order (decoder layers, then MTP): each is recorded once
    // that layer's state (and its planes of the tail page) has been read. Empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> capture_events(std::uint64_t id) const noexcept;
    // Orders the compute stream (every later writer of a pool page runs on it or after it) after
    // the pending captures that read one of `pages` as their tail, before those pages return to the
    // pool: a tap reads the lane's page on the transfer stream, and a page reused under that copy
    // would put another request's KV into the snapshot. Free once the copies have landed.
    void order_page_reuse(std::span<const DeviceKVPageLease> pages) const;
    // Copies every Device-only node of `nodes` to a Host slab, after `producer`'s queued work.
    // Without a Host tier nothing is written: blocks reach the Disk with the snapshots below them.
    void write_blocks(std::span<const prefix::cache_index::NodeRef> nodes, cudaStream_t producer);
    // Publishes landed captures and writes and retires landed restores. Non-blocking.
    void poll();
    // Waits for every capture copy, Host write and restore in flight and publishes them. Disk writes
    // are not waited for: they stay polled (a capture written through publishes once they land).
    void drain();
    [[nodiscard]] bool transfers_pending() const noexcept {
        return !captures_.empty() || !writes_.empty() || !landing_.empty();
    }
    // Waits for an endpoint copy-out a prompt extends (its anchor is on the prompt's matched `path`
    // and its tail tokens continue the prompt), and publishes it. Only such a prompt waits; the
    // copy is first in the transfer stream, at most an image's copy time. Returns whether it waited.
    [[nodiscard]] bool await_extended_endpoint(std::span<const prefix::cache_index::NodeRef> path,
                                               std::span<const TokenId> tokens);
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
    // Enqueues the batch per decoder layer (that layer's page planes or state parts), then the MTP
    // group, recording one event after each. Returns the ticket the lane's calls wait with.
    [[nodiscard]] RestoreTicket submit_restore();
    [[nodiscard]] bool restore_open() const noexcept { return restore_.open; }
    // The batch's events in layer order (decoder layers, then MTP); empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> restore_events(RestoreTicket ticket) const noexcept;
    // Orders `consumer` after every copy of a landing batch.
    void order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const;
    // Waits on the host for a landing batch (and every batch before it) and retires what landed.
    void await_restore(RestoreTicket ticket);
    void abort_restore() noexcept;

    // Releases every reference the cache holds; the index is rebuilt empty.
    void clear() noexcept;

    // ---- Disk tier (disk_tier.cpp) ---------------------------------------------------------------
    // Opens the Disk tier in `directory` and rebuilds its records into this empty cache as
    // Disk-only entries; no data is read. `fingerprint` names everything the stored bytes depend
    // on besides their geometry; a mismatch clears the tier. Throws when the directory is in use
    // or cuFile cannot open it.
    [[nodiscard]] DiskAttachResult attach_disk(const std::filesystem::path& directory, std::string_view fingerprint);
    [[nodiscard]] bool disk_attached() const noexcept { return store_ != nullptr; }
    [[nodiscard]] CuFileQueue::Mode disk_mode() const;
    // Reads what `snapshot` and its matched `path` hold only on the Disk: blocks into new cache
    // pages (Device Filling until they land), the image and tail into a staging region. False when
    // no fetch can start now (one is running, no region, Disk slots unreadable, or too few pages).
    [[nodiscard]] bool start_fetch(prefix::cache_index::SnapshotRef snapshot,
                                   std::span<const prefix::cache_index::NodeRef> path);
    [[nodiscard]] bool fetch_running() const noexcept { return fetch_ && !fetch_->ready; }
    // Disk tier work poll() would advance with no request running: transfers in flight, records
    // to journal, or snapshots published since the writer last found nothing to write.
    [[nodiscard]] bool background_pending() const noexcept;
    // `--debug disk` lines go to `observer`.
    void set_diagnostics(const DiagnosticObserver& observer) { diagnostics_ = observer; }
    // A capture being written through to the Disk tier (no Host tier) that `tokens` extends: its
    // anchor is on the prompt's matched `path` and its tail continues the prompt. It publishes
    // once its writes complete.
    [[nodiscard]] bool disk_capture_pending(std::span<const prefix::cache_index::NodeRef> path,
                                            std::span<const TokenId> tokens) const;
    [[nodiscard]] bool fetch_holds(prefix::cache_index::SnapshotRef snapshot) const noexcept {
        return fetch_ && fetch_->ready && fetch_->snapshot == snapshot;
    }
    // The Disk bytes of `candidate` a landed fetch holds in its region (its image and tail).
    [[nodiscard]] std::uint64_t fetched_bytes(const prefix::cache_index::MatchCandidate& candidate) const noexcept;
    // Drops a landed fetch: its pins and, once no restore reads it, its region.
    void release_fetch() noexcept;
    // A snapshot whose fetch failed: not fetched again.
    [[nodiscard]] bool disk_unreadable(prefix::cache_index::SnapshotRef snapshot) const noexcept;
    // At shutdown, after every lane is released: spills every retained Host snapshot without a
    // Disk copy, highest value first, records every snapshot's hits and makes the journal durable.
    // `abandoned` stops it between snapshots.
    [[nodiscard]] DiskSpillResult spill_for_shutdown(const CancellationView& abandoned);

    // PrefixIndexBackend
    void release_device_block(std::uint32_t device_id) noexcept override;
    void release_snapshot(prefix::cache_index::SnapshotRef snapshot) noexcept override;
    void drop_snapshot_device_image(prefix::cache_index::SnapshotRef snapshot) noexcept override;
    void release_disk_record(std::span<const std::uint32_t> slots) noexcept override;
    void disk_record_replaced(prefix::cache_index::NodeRef node) noexcept override;

private:
    static constexpr std::uint32_t kNoRegion = UINT32_MAX;
    // Where a capture's image lands: Host slabs, or (no Host tier) a staging region whose slots are
    // then written to reserved Disk slots.
    struct ImageTarget {
        prefix::cache_index::HostImageReservation host;
        std::optional<prefix::cache_index::DiskImageReservation> disk;
        std::uint32_t region = kNoRegion;
    };
    struct PendingCapture {
        Capture capture;
        ImageTarget image;
        std::vector<cudaEvent_t> events; // decoder layers, then MTP
        // The prompt-end staging slot this capture reads; a slot is free once no pending capture
        // names it. Null for the group-by-group captures, and once the copy has landed.
        std::byte* staging = nullptr;
        // A region's header copy, after the groups' (a committed staged capture): the image has
        // landed once it has.
        cudaEvent_t header_done = nullptr;
        // Disk writes of the region (no Host tier): issued once the copy lands.
        bool writes_issued       = false;
        std::uint32_t writes     = 0; // in flight
        bool write_failed        = false;
        // A prompt-end capture waiting for a region (no Host tier): the producer's copy into its
        // slot. Its image is reserved and copied out once a region is free; until then it has no
        // events.
        cudaEvent_t deferred = nullptr;
    };
    struct StagedCapture {
        ImageTarget image;
        bool tail = false;
        std::vector<cudaEvent_t> events; // decoder layers, then MTP; each recorded once its group is copied
        std::uint32_t copied = 0;        // groups copied
    };
    // The image's segments, as HostImage takes them, and its tail record and that record's group.
    [[nodiscard]] std::vector<std::byte*> image_segments(const ImageTarget& image) const;
    // Copies a prompt-end capture's slot into its image on the transfer stream, after `after`.
    void copy_prompt_end(PendingCapture& c, cudaEvent_t after);
    // A deferred prompt-end capture takes a free region: false while none is; one whose image
    // cannot be reserved is dropped (`dropped`).
    bool start_deferred(PendingCapture& c, bool& dropped);
    [[nodiscard]] std::byte* tail_record(const ImageTarget& image) const;
    [[nodiscard]] std::uint32_t tail_group(const ImageTarget& image) const;
    // Reserves where a capture lands (Host slabs, or a region and Disk slots); absent when none.
    [[nodiscard]] std::optional<ImageTarget> reserve_image(bool tail, double claim);
    void release_image(ImageTarget&& image) noexcept;
    // Writes the header: on the Host directly, into a region on `stream`.
    void write_header(const ImageTarget& image, const StateImageHeader& meta, cudaStream_t stream);
    // Group g's copy-out (its layer's planes of a tail page first) on the transfer stream.
    // A prompt-end capture's private tail page, every KV layer's planes to its Host record.
    void copy_tail_page_to_record(DeviceKVPageHandle page, std::byte* record, std::uint32_t group,
                                  cudaStream_t stream) const;
    void copy_capture_group(std::uint32_t group, std::uint32_t lane, const HostImage& destination,
                            std::byte* tail_record, std::uint32_t tail_group, std::optional<DeviceKVPageHandle> tail_page);
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
        std::uint32_t image_region = kNoRegion; // a fetched image, read from its region
        std::vector<std::uint32_t> regions;      // regions read until the batch lands
        std::vector<cudaEvent_t> events; // decoder layers, then MTP
    };

    // ---- Disk tier state (disk_tier.cpp) ----
    struct Region {
        DeviceBuffer image;         // image_slabs + 1 slots: the packed image, then the tail record
        PinnedHostBuffer header;    // a capture's header, or a fetch's read of the first slot
        std::uint32_t users = 0;
    };
    struct DiskIo {
        enum class Kind : std::uint8_t { Capture, Block, Image, FetchImage, FetchBlock } kind = Kind::Capture;
        std::uint64_t owner = 0;
    };
    // One block written to its Disk slot: from its Host slab, or through a staging record from its
    // Device page. The node is pinned and its Disk copy Filling until done.
    struct BlockWrite {
        enum class Stage : std::uint8_t { Queued, Copying, Writing, Done } stage = Stage::Queued;
        prefix::cache_index::NodeRef node;
        std::uint32_t slot = 0;
        const std::byte* host = nullptr;
        DeviceKVPageHandle page;
        std::uint32_t staging = kNoRegion;
        cudaEvent_t copied    = nullptr;
        bool ok               = true;
    };
    // The Disk writer's job (§17.1 rules 3-4): a snapshot's path blocks without a Disk copy, root
    // first, with at most kDiskWriteWindow bytes of block writes in flight, then its image. It pins
    // nothing between writes: an entry evicted before its turn ends the job.
    struct Writer {
        prefix::cache_index::SnapshotRef snapshot;
        std::vector<prefix::cache_index::NodeRef> path;
        std::size_t next     = 0;
        double claim         = 0.0; // the snapshot's GDSF value, claimed for every Disk slot
        bool image           = true; // false: the image was written through (no Host tier)
        bool image_started   = false; // the snapshot is pinned and its Disk copy Filling
        bool image_issued    = false;
        std::uint32_t writes = 0;
        bool ok              = true;
    };
    struct Fetch {
        struct Block {
            enum class Stage : std::uint8_t { Queued, Reading, Copying, Done } stage = Stage::Queued;
            prefix::cache_index::NodeRef node;
            DeviceKVPageHandle page;
            std::uint32_t slot    = 0;
            std::uint32_t staging = kNoRegion; // its staging record (no Host tier)
            std::uint32_t slab    = kNoRegion; // its Host slab (a Host tier)
            cudaEvent_t copied    = nullptr;
            bool ok               = true;
        };
        prefix::cache_index::SnapshotRef snapshot;
        std::vector<prefix::cache_index::NodeRef> path; // acquired
        // With a Host tier, the image and blocks are read into Host fills (`image_fill`; a tail
        // without a Disk copy is copied from its Device page, `tail_copy`) and restored from the
        // Host as usual. Without one, blocks go through staging records into new Device pages and
        // the image into a region.
        bool host              = false;
        bool image_fill        = false;
        cudaEvent_t tail_copy  = nullptr;
        std::uint32_t region = kNoRegion;
        bool tail            = false; // the tail record is in the region
        std::vector<Block> blocks;
        std::uint32_t reads = 0; // image and tail reads in flight
        cudaEvent_t header  = nullptr; // the image header's copy into pinned memory
        bool header_copied  = false;
        bool failed         = false;
        bool ready          = false;
        std::chrono::steady_clock::time_point expires;
    };
    // A record appended once the record it links to (parent block, anchor) has been.
    struct JournalWait {
        prefix::cache_index::NodeRef node;
        prefix::cache_index::SnapshotRef snapshot;
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

    // Disk tier (disk_tier.cpp).
    [[nodiscard]] std::byte* region_base(std::uint32_t region) const noexcept;
    [[nodiscard]] std::optional<std::uint32_t> take_region() noexcept;
    void release_region(std::uint32_t region) noexcept;
    [[nodiscard]] std::size_t staging_offset(std::uint32_t staging) const noexcept;
    [[nodiscard]] std::byte* staging_record(std::uint32_t staging) const noexcept;
    [[nodiscard]] bool disk_writable() const noexcept;
    // `buffer` is a registered base (a region, the block staging) or Host memory, `buffer_offset`
    // the transfer's bytes within it: cuFile finds registered memory by its base.
    void submit_disk(CuFileQueue::Op op, std::uint32_t slot, std::size_t bytes, void* buffer,
                     std::size_t buffer_offset, DiskIo io);
    void reap_disk();
    void issue_capture_writes(PendingCapture& c);
    // Queues `node`'s Disk write (nothing when it has a Disk copy); false when it has no source copy
    // or no Disk slot could be freed for `claim`.
    bool queue_block_write(prefix::cache_index::NodeRef node, double claim);
    void service_block_writes();
    // Starts the writer on `snapshot` (`image`: its image too); false when it is busy, the snapshot
    // needs nothing, or a path block has no copy to write from.
    [[nodiscard]] bool start_writer(prefix::cache_index::SnapshotRef snapshot, bool image);
    void service_writer();
    void service_fetch();
    void finish_fetch();
    void service_journal();
    bool start_host_fetch(prefix::cache_index::SnapshotRef snapshot, std::span<const prefix::cache_index::NodeRef> path,
                          bool image, std::span<const prefix::cache_index::NodeRef> blocks);
    void finish_host_fetch();
    [[nodiscard]] bool journalable(prefix::cache_index::NodeRef node) const;
    void service_replaced();
    void poll_disk();
    // Waits for every Disk write in flight (captures, blocks, the writer's job and the queued path
    // jobs) and journals them; starts no new writer job meanwhile.
    void drain_disk();
    [[nodiscard]] bool disk_writes_pending() const noexcept;
    void disk_trace(const char* format, ...) const noexcept __attribute__((format(printf, 2, 3)));

    runtime::prefix_cache::TapPlannerConfig taps_;
    DeviceKVPagePool* pool_ = nullptr;
    KvPageGeometry geometry_;
    const LaneStateImage* lanes_ = nullptr;
    PrefixHostLayout host_;
    std::vector<PinnedHostBuffer> chunks_;
    cudaStream_t transfer_ = nullptr;
    // The compute stream (the producer of every capture, write and restore): Disk tier copies into
    // and out of cache pages are ordered after its work.
    cudaStream_t producer_ = nullptr;
    cudaStream_t restore_stream_ = nullptr;
    std::vector<std::optional<DeviceKVPageLease>> blocks_;
    std::vector<std::uint32_t> free_blocks_;
    std::vector<std::int32_t> node_mtp_next_;      // by node index
    std::vector<StateImageHeader> snapshot_meta_;  // by snapshot index
    std::deque<PendingCapture> captures_;
    std::unordered_map<std::uint64_t, StagedCapture> staged_;
    std::uint64_t next_staged_ = 1;
    std::deque<PendingWrite> writes_;
    std::unordered_map<std::uint64_t, prefix::cache_index::PublishResult> published_;
    RestoreBatch restore_;
    std::deque<RestoreBatch> landing_;
    std::uint64_t next_ticket_ = 1;
    std::vector<cudaEvent_t> spare_events_;
    std::vector<std::byte*> prompt_end_slots_;      // the fixed staging images (set_prompt_end_staging)
    std::vector<PinnedHostBuffer> prompt_end_headers_; // one pinned header per slot, its capture's own
    PrefixCacheCounters counters_;

    std::uint32_t disk_slots_ = 0;
    std::unique_ptr<DiskStore> store_;
    std::vector<Region> regions_;
    // kStagingRecords block records: Device memory without a Host tier, pinned memory with one (no
    // VRAM, so the prefill workspace keeps its room).
    DeviceBuffer block_staging_device_;
    std::optional<PinnedHostBuffer> block_staging_host_;
    std::byte* block_staging_ = nullptr;
    std::vector<std::uint32_t> free_staging_;
    std::unordered_map<std::uint64_t, DiskIo> disk_io_; // by transfer tag
    std::uint64_t next_disk_tag_ = 1;
    std::vector<CuFileQueue::Completion> completions_;
    std::map<std::uint64_t, BlockWrite> block_writes_; // in queue order
    std::uint64_t next_block_write_ = 1;
    DiagnosticObserver diagnostics_;
    std::optional<Writer> writer_;
    // Written-through snapshots (no Host tier) whose paths the writer follows with.
    std::deque<prefix::cache_index::SnapshotRef> path_jobs_;
    // Snapshots the writer could not finish, skipped until the index's snapshots change.
    std::vector<prefix::cache_index::SnapshotRef> writer_refused_;
    std::uint64_t writer_refused_epoch_ = 0;
    bool writer_scans_                  = true;
    // The index's snapshot epoch at the last scan that found nothing to start (UINT64_MAX: scan).
    std::uint64_t scanned_epoch_ = UINT64_MAX;
    std::chrono::steady_clock::time_point next_writer_scan_{};
    std::optional<Fetch> fetch_;
    std::vector<prefix::cache_index::SnapshotRef> unreadable_;
    std::vector<JournalWait> journal_waits_;
    // Blocks whose Disk record a sequence's page replaced: written again, and the records below
    // them journaled again after them.
    std::vector<prefix::cache_index::NodeRef> replaced_;
    std::vector<bool> slot_journaled_;                                          // by a record's first slot
    std::unordered_map<std::uint32_t, prefix::cache_index::SnapshotRef> journaled_snapshots_; // by first slot
    bool disk_broken_ = false; // a removal could not be journaled: no slot is rewritten
    std::optional<prefix::cache_index::PrefixCacheIndex> index_;
};

} // namespace infernix::models::qwen4_exp::prefix
