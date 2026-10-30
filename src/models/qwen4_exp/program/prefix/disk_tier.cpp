// The Disk tier of Qwen4Exp's prefix cache (prefix_cache.h): spills, write-through, fetches,
// the journal and the startup rebuild. Every residency change
// runs on the worker thread from poll(); cuFile transfers are polled, never waited on, except at
// shutdown, clear and destruction.

#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

using Clock = std::chrono::steady_clock;

// A landed fetch keeps its pins and region this long for the admission that asked for it.
constexpr auto kFetchHold = std::chrono::seconds(2);
// How often the Host eviction order is scanned for a snapshot to spill.
constexpr auto kSpillScan = std::chrono::milliseconds(100);
// The copy group of a block staging record (it lies in no Host chunk).
constexpr std::uint32_t kStagingGroup = UINT32_MAX;

bool same_meta(const StateImageHeader& a, const StateImageHeader& b) {
    return a.frontier == b.frontier && a.mtp_written == b.mtp_written && a.mtp_next == b.mtp_next &&
           a.lineage_echo == b.lineage_echo && a.opener == b.opener && a.head_residual == b.head_residual &&
           a.mtp_accept == b.mtp_accept && a.draft_frontier == b.draft_frontier;
}

} // namespace

// ---- staging -------------------------------------------------------------------------------------

std::uint64_t PrefixCache::disk_staging_bytes() const noexcept {
    std::uint64_t bytes = block_staging_device_.bytes;
    for (const Region& region : regions_) { bytes += region.image.bytes; }
    return bytes;
}

std::byte* PrefixCache::region_base(std::uint32_t region) const noexcept {
    return static_cast<std::byte*>(regions_[region].image.p);
}

std::optional<std::uint32_t> PrefixCache::take_region() noexcept {
    for (std::uint32_t r = 0; r < regions_.size(); ++r) {
        if (regions_[r].users == 0) {
            regions_[r].users = 1;
            return r;
        }
    }
    return std::nullopt;
}

void PrefixCache::release_region(std::uint32_t region) noexcept {
    if (region < regions_.size() && regions_[region].users != 0) { --regions_[region].users; }
}

std::size_t PrefixCache::staging_offset(std::uint32_t staging) const noexcept {
    return static_cast<std::size_t>(staging) * host_.slab_bytes;
}

std::byte* PrefixCache::staging_record(std::uint32_t staging) const noexcept {
    return block_staging_ + staging_offset(staging);
}

bool PrefixCache::disk_writable() const noexcept { return store_ && !disk_broken_ && !store_->failed(); }

CuFileQueue::Mode PrefixCache::disk_mode() const {
    if (!store_) { throw std::logic_error("Qwen4Exp prefix cache: the Disk tier is not attached"); }
    return store_->io().mode();
}

// ---- transfers -----------------------------------------------------------------------------------

void PrefixCache::submit_disk(CuFileQueue::Op op, std::uint32_t slot, std::size_t bytes, void* buffer,
                              std::size_t buffer_offset, DiskIo io) {
    const std::uint64_t tag = next_disk_tag_++;
    store_->io().submit(CuFileQueue::Transfer{.op            = op,
                                              .offset        = store_->slot_offset(slot),
                                              .bytes         = bytes,
                                              .buffer        = buffer,
                                              .buffer_offset = buffer_offset,
                                              .tag           = tag,
                                              // A staging region and an admission wait on these;
                                              // block writes and spills do not.
                                              .urgent = io.kind == DiskIo::Kind::Capture ||
                                                        io.kind == DiskIo::Kind::FetchImage ||
                                                        io.kind == DiskIo::Kind::FetchBlock});
    disk_io_.emplace(tag, io);
}

void PrefixCache::reap_disk() {
    completions_.clear();
    store_->io().poll(completions_);
    for (const CuFileQueue::Completion& done : completions_) {
        const auto found = disk_io_.find(done.tag);
        if (found == disk_io_.end()) { continue; }
        const DiskIo io = found->second;
        disk_io_.erase(found);
        if (!done.ok) { ++counters_.disk_failures; }
        switch (io.kind) {
        case DiskIo::Kind::Capture:
            for (PendingCapture& c : captures_) {
                if (c.capture.id != io.owner) { continue; }
                --c.writes;
                c.write_failed = c.write_failed || !done.ok;
                break;
            }
            break;
        case DiskIo::Kind::Block:
            if (const auto write = block_writes_.find(io.owner); write != block_writes_.end()) {
                write->second.ok    = done.ok;
                write->second.stage = BlockWrite::Stage::Done;
            }
            break;
        case DiskIo::Kind::Spill:
            if (spill_) {
                --spill_->writes;
                spill_->ok = spill_->ok && done.ok;
            }
            break;
        case DiskIo::Kind::FetchImage:
            if (fetch_) {
                --fetch_->reads;
                fetch_->failed = fetch_->failed || !done.ok;
            }
            break;
        case DiskIo::Kind::FetchBlock:
            if (fetch_ && io.owner < fetch_->blocks.size()) {
                Fetch::Block& block = fetch_->blocks[io.owner];
                if (done.ok && fetch_->host) {
                    block.stage = Fetch::Block::Stage::Done; // read into its Host slab
                } else if (done.ok) {
                    block.stage = Fetch::Block::Stage::Copying; // the copy into its page is issued next
                } else if (fetch_->host) {
                    block.ok       = false;
                    block.stage    = Fetch::Block::Stage::Done;
                    fetch_->failed = true;
                } else {
                    block.ok       = false;
                    block.stage    = Fetch::Block::Stage::Done;
                    fetch_->failed = true;
                    free_staging_.push_back(block.staging);
                    block.staging = kNoRegion;
                }
            }
            break;
        }
    }
}

// ---- write-through of captures (no Host tier) ----------------------------------------------------

void PrefixCache::issue_capture_writes(PendingCapture& c) {
    if (!disk_writable()) {
        c.write_failed  = true;
        c.writes_issued = true;
        return;
    }
    const std::vector<std::uint32_t>& slots = c.image.disk->slots;
    // A slot freed by a removal is rewritten only once that removal is durable.
    if (!store_->writable(slots)) { return; }
    for (std::uint32_t i = 0; i < slots.size(); ++i) {
        submit_disk(CuFileQueue::Op::Write, slots[i], host_.slab_bytes, region_base(c.image.region),
                    static_cast<std::size_t>(i) * host_.slab_bytes, DiskIo{DiskIo::Kind::Capture, c.capture.id});
    }
    c.writes        = static_cast<std::uint32_t>(slots.size());
    c.writes_issued = true;
    ++counters_.disk_image_writes;
    counters_.disk_write_bytes += slots.size() * host_.slab_bytes;
}

// ---- block writes --------------------------------------------------------------------------------

bool PrefixCache::queue_block_write(pc::NodeRef node, double claim) {
    const pc::NodeView view = index_->node(node);
    if (view.disk != pc::CopyState::Absent) { return true; }
    const bool host   = view.host == pc::CopyState::Resident;
    const bool device = view.device == pc::CopyState::Resident;
    if (!host && !device) { return false; }
    index_->pin_node(node);
    const std::optional<std::uint32_t> slot = index_->begin_node_disk_fill(node, claim);
    if (!slot) {
        index_->unpin_node(node);
        return false;
    }
    BlockWrite write;
    write.node = node;
    write.slot = *slot;
    if (host) {
        write.host = slab(view.host_slab);
    } else {
        write.page = page(view.device_id);
    }
    block_writes_.emplace(next_block_write_++, write);
    return true;
}

bool PrefixCache::queue_path_writes(pc::NodeRef anchor, double claim) {
    std::vector<pc::NodeRef> path;
    for (pc::NodeRef node = anchor; node.valid(); node = index_->node(node).parent) { path.push_back(node); }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        if (!queue_block_write(*it, claim)) { return false; }
    }
    return true;
}

void PrefixCache::service_block_writes() {
    bool ordered = false; // the transfer stream waits for the compute stream's queued work
    for (auto it = block_writes_.begin(); it != block_writes_.end();) {
        BlockWrite& w = it->second;
        const std::uint32_t slots[1] = {w.slot};
        if (w.stage == BlockWrite::Stage::Queued) {
            if (!disk_writable()) {
                w.ok    = false;
                w.stage = BlockWrite::Stage::Done;
            } else if (w.host != nullptr) {
                if (store_->writable(slots)) {
                    submit_disk(CuFileQueue::Op::Write, w.slot, host_.slab_bytes, const_cast<std::byte*>(w.host), 0,
                                DiskIo{DiskIo::Kind::Block, it->first});
                    w.stage = BlockWrite::Stage::Writing;
                }
            } else if (!free_staging_.empty()) {
                // A published page is final, but the compute work that wrote it may still be queued.
                if (!ordered) {
                    cudaEvent_t start = take_event();
                    CUDA_CHECK(cudaEventRecord(start, producer_));
                    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
                    recycle(start);
                    ordered = true;
                }
                w.staging = free_staging_.back();
                free_staging_.pop_back();
                const DeviceKVPageHandle pages[1] = {w.page};
                std::byte* const records[1]       = {staging_record(w.staging)};
                const std::uint32_t groups[1]      = {kStagingGroup};
                pool_->copy_to_host_records(pages, records, groups, host_.page, transfer_);
                w.copied = take_event();
                CUDA_CHECK(cudaEventRecord(w.copied, transfer_));
                w.stage = BlockWrite::Stage::Copying;
            }
        }
        if (w.stage == BlockWrite::Stage::Copying) {
            const cudaError_t status = cudaEventQuery(w.copied);
            if (status != cudaErrorNotReady) {
                CUDA_CHECK(status);
                if (!disk_writable()) {
                    w.ok    = false;
                    w.stage = BlockWrite::Stage::Done;
                } else if (store_->writable(slots)) {
                    submit_disk(CuFileQueue::Op::Write, w.slot, host_.slab_bytes, block_staging_,
                                staging_offset(w.staging), DiskIo{DiskIo::Kind::Block, it->first});
                    w.stage = BlockWrite::Stage::Writing;
                }
            }
        }
        if (w.stage != BlockWrite::Stage::Done) {
            ++it;
            continue;
        }
        if (w.staging != kNoRegion) { free_staging_.push_back(w.staging); }
        recycle(w.copied);
        if (w.ok) {
            index_->complete_node_disk_fill(w.node);
            journal_waits_.push_back(JournalWait{w.node, {}});
            ++counters_.disk_block_writes;
            counters_.disk_write_bytes += host_.slab_bytes;
        } else {
            index_->abort_node_disk_fill(w.node);
        }
        index_->unpin_node(w.node);
        it = block_writes_.erase(it);
    }
}

// ---- spills (Host tier) --------------------------------------------------------------------------

bool PrefixCache::start_spill(pc::SnapshotRef snapshot) {
    if (spill_ || !disk_writable() || !index_->valid(snapshot)) { return false; }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident || view.disk != pc::CopyState::Absent || view.pins != 0) { return false; }
    // Every path block without a Disk copy needs a copy to write it from.
    for (pc::NodeRef node = view.anchor; node.valid();) {
        const pc::NodeView n = index_->node(node);
        if (n.disk == pc::CopyState::Absent && n.host != pc::CopyState::Resident && n.device != pc::CopyState::Resident) {
            return false;
        }
        node = n.parent;
    }
    index_->pin_snapshot(snapshot);
    if (!index_->begin_snapshot_disk_fill(snapshot)) {
        index_->unpin_snapshot(snapshot);
        ++counters_.disk_spill_skips;
        return false;
    }
    if (view.anchor.valid() && !queue_path_writes(view.anchor, view.priority)) {
        // The blocks queued so far are written anyway; the snapshot is not.
        index_->abort_snapshot_disk_fill(snapshot);
        index_->unpin_snapshot(snapshot);
        ++counters_.disk_spill_skips;
        return false;
    }
    spill_ = Spill{.snapshot = snapshot};
    return true;
}

void PrefixCache::service_spill() {
    if (!spill_) { return; }
    Spill& s = *spill_;
    if (!s.issued) {
        if (!disk_writable()) {
            s.ok     = false;
            s.issued = true;
        } else {
            const pc::SnapshotView view = index_->snapshot(s.snapshot);
            if (!store_->writable(view.disk_slots)) { return; }
            for (std::size_t i = 0; i < view.disk_slots.size(); ++i) {
                submit_disk(CuFileQueue::Op::Write, view.disk_slots[i], host_.slab_bytes, slab(view.host_slabs[i]), 0,
                            DiskIo{DiskIo::Kind::Spill, 0});
            }
            s.writes = static_cast<std::uint32_t>(view.disk_slots.size());
            s.issued = true;
            ++counters_.disk_image_writes;
            counters_.disk_write_bytes += view.disk_slots.size() * host_.slab_bytes;
        }
    }
    if (s.writes != 0) { return; }
    if (s.ok) {
        index_->complete_snapshot_disk_fill(s.snapshot);
        journal_waits_.push_back(JournalWait{{}, s.snapshot});
    } else {
        index_->abort_snapshot_disk_fill(s.snapshot);
    }
    index_->unpin_snapshot(s.snapshot);
    spill_.reset();
}

// ---- fetches -------------------------------------------------------------------------------------

bool PrefixCache::disk_capture_pending(std::span<const pc::NodeRef> path, std::span<const TokenId> tokens) const {
    return std::any_of(captures_.begin(), captures_.end(), [&](const PendingCapture& c) {
        if ((!c.image.disk && c.deferred == nullptr) || c.capture.anchor_pending || c.capture.frontier > tokens.size()) {
            return false;
        }
        const std::uint32_t k = c.capture.frontier / pc::kBlockTokens;
        if (k > 0 && (path.size() < k || path[k - 1] != c.capture.anchor)) { return false; }
        return std::equal(c.capture.tail.begin(), c.capture.tail.end(),
                          tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens);
    });
}

bool PrefixCache::disk_unreadable(pc::SnapshotRef snapshot) const noexcept {
    return std::find(unreadable_.begin(), unreadable_.end(), snapshot) != unreadable_.end();
}

std::uint64_t PrefixCache::fetched_bytes(const pc::MatchCandidate& candidate) const noexcept {
    if (!fetch_ || !fetch_->ready || fetch_->failed || fetch_->snapshot != candidate.snapshot) { return 0; }
    return (candidate.image_on_disk_only && fetch_->region != kNoRegion ? host_.image_bytes : 0U) +
           (candidate.tail_on_disk_only && fetch_->tail ? host_.page.page_stride : 0U);
}

bool PrefixCache::start_fetch(pc::SnapshotRef snapshot, std::span<const pc::NodeRef> path) {
    if (!store_ || fetch_running() || disk_unreadable(snapshot) || !index_->valid(snapshot)) { return false; }
    release_fetch();
    const pc::SnapshotView view = index_->snapshot(snapshot);
    const bool image = view.host != pc::CopyState::Resident && view.disk == pc::CopyState::Resident;
    const bool tail  = image && view.tail_len != 0 && view.tail_device_copy != pc::CopyState::Resident;
    std::vector<pc::NodeRef> blocks;
    for (const pc::NodeRef node : path) {
        const pc::NodeView n = index_->node(node);
        if (n.device == pc::CopyState::Absent && n.host != pc::CopyState::Resident && n.disk == pc::CopyState::Resident) {
            blocks.push_back(node);
        }
    }
    if (!image && blocks.empty()) { return false; }
    if (host_tier()) { return start_host_fetch(snapshot, path, image, blocks); }
    std::uint32_t region = kNoRegion;
    if (image) {
        const std::optional<std::uint32_t> taken = take_region();
        if (!taken) { return false; }
        region = *taken;
    }
    if (!path.empty()) { index_->acquire_path(path); }
    index_->pin_snapshot(snapshot);
    const auto count = static_cast<std::uint32_t>(blocks.size());
    while (pool_->available_pages() < count && index_->evict_backed_device_blocks(1) != 0) {}
    std::optional<DeviceKVPageReservation> reservation;
    if (count != 0 && pool_->available_pages() >= count) { reservation = pool_->reserve(count); }
    if (count != 0 && !reservation) {
        index_->unpin_snapshot(snapshot);
        if (!path.empty()) { index_->release_path(path); }
        if (region != kNoRegion) { release_region(region); }
        return false;
    }
    Fetch& f   = fetch_.emplace();
    f.snapshot = snapshot;
    f.path.assign(path.begin(), path.end());
    f.region = region;
    f.tail   = tail;
    if (count != 0) {
        std::vector<DeviceKVPageLease> pages;
        pages.reserve(count);
        pool_->materialize(*reservation, count, pages);
        for (std::uint32_t i = 0; i < count; ++i) {
            Fetch::Block block;
            block.node = blocks[i];
            block.page = pages[i].handle();
            block.slot = index_->node(blocks[i]).disk_slot;
            index_->begin_device_fill(blocks[i], allocate_block_id(std::move(pages[i])));
            f.blocks.push_back(block);
        }
        counters_.disk_block_reads += count;
        counters_.disk_read_bytes += static_cast<std::uint64_t>(count) * host_.slab_bytes;
    }
    if (image) {
        for (std::uint32_t i = 0; i < host_.image_slabs + (tail ? 1U : 0U); ++i) {
            const std::uint32_t slot = i < host_.image_slabs ? view.disk_slots[i] : view.disk_slots.back();
            submit_disk(CuFileQueue::Op::Read, slot, host_.slab_bytes, region_base(region),
                        static_cast<std::size_t>(i) * host_.slab_bytes, DiskIo{DiskIo::Kind::FetchImage, 0});
            ++f.reads;
        }
        ++counters_.disk_image_reads;
        counters_.disk_read_bytes += static_cast<std::uint64_t>(f.reads) * host_.slab_bytes;
    }
    ++counters_.disk_fetches;
    service_fetch();
    return true;
}

bool PrefixCache::start_host_fetch(pc::SnapshotRef snapshot, std::span<const pc::NodeRef> path, bool image,
                                   std::span<const pc::NodeRef> blocks) {
    if (!path.empty()) { index_->acquire_path(path); }
    index_->pin_snapshot(snapshot);
    Fetch& f   = fetch_.emplace();
    f.snapshot = snapshot;
    f.path.assign(path.begin(), path.end());
    f.host = true;
    // Every Host fill first, so a fetch that cannot hold all of them starts nothing: the request
    // then resumes from a shallower source instead of waiting for one that never lands.
    bool reserved = true;
    for (const pc::NodeRef node : blocks) {
        const std::optional<std::uint32_t> slab_id = index_->begin_host_fill(node);
        if (!slab_id) {
            reserved = false;
            break;
        }
        Fetch::Block block;
        block.node    = node;
        block.slot    = index_->node(node).disk_slot;
        block.slab    = *slab_id;
        f.blocks.push_back(block);
    }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    const bool tail_on_disk     = view.disk_slots.size() > host_.image_slabs;
    if (reserved && image && view.tail_len != 0 && !tail_on_disk && view.tail_device_copy != pc::CopyState::Resident) {
        reserved = false;
    }
    // The read serves a request waiting now: it displaces the least valuable Host entries whatever
    // their value, as a block's write-through does.
    if (reserved && image) {
        reserved = f.image_fill =
            index_->begin_snapshot_host_fill(snapshot, std::numeric_limits<double>::infinity());
    }
    if (!reserved) {
        for (const Fetch::Block& b : f.blocks) { index_->abort_host_fill(b.node); }
        fetch_.reset();
        index_->unpin_snapshot(snapshot);
        if (!path.empty()) { index_->release_path(path); }
        return false;
    }
    for (std::uint32_t i = 0; i < f.blocks.size(); ++i) {
        Fetch::Block& b = f.blocks[i];
        submit_disk(CuFileQueue::Op::Read, b.slot, host_.slab_bytes, slab(b.slab), 0, DiskIo{DiskIo::Kind::FetchBlock, i});
        b.stage = Fetch::Block::Stage::Reading;
    }
    counters_.disk_block_reads += f.blocks.size();
    counters_.disk_read_bytes += f.blocks.size() * host_.slab_bytes;
    if (f.image_fill) {
        const pc::SnapshotView filling = index_->snapshot(snapshot);
        for (std::uint32_t i = 0; i < host_.image_slabs; ++i) {
            submit_disk(CuFileQueue::Op::Read, filling.disk_slots[i], host_.slab_bytes, slab(filling.host_slabs[i]), 0,
                        DiskIo{DiskIo::Kind::FetchImage, 0});
            ++f.reads;
        }
        if (filling.tail_len != 0) {
            const std::uint32_t tail_slab = filling.host_slabs.back();
            if (tail_on_disk) {
                submit_disk(CuFileQueue::Op::Read, filling.disk_slots.back(), host_.slab_bytes, slab(tail_slab), 0,
                            DiskIo{DiskIo::Kind::FetchImage, 0});
                ++f.reads;
            } else {
                // The tail's Device page was written on the compute stream.
                cudaEvent_t start = take_event();
                CUDA_CHECK(cudaEventRecord(start, producer_));
                CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
                recycle(start);
                copy_tail_page_to_record(page(filling.tail_device), slab(tail_slab), slab_group(tail_slab), transfer_);
                f.tail_copy = take_event();
                CUDA_CHECK(cudaEventRecord(f.tail_copy, transfer_));
            }
        }
        ++counters_.disk_image_reads;
        counters_.disk_read_bytes += static_cast<std::uint64_t>(f.reads) * host_.slab_bytes;
    }
    ++counters_.disk_fetches;
    return true;
}

void PrefixCache::finish_host_fetch() {
    Fetch& f = *fetch_;
    if (f.image_fill && !f.failed) {
        const pc::SnapshotView view = index_->snapshot(f.snapshot);
        const auto header           = read_state_image_header(lanes_->layout(), slab(view.host_slabs.front()));
        if (!header || !same_meta(*header, snapshot_meta_[f.snapshot.index])) {
            ++counters_.disk_failures;
            f.failed = true;
            unreadable_.push_back(f.snapshot);
        }
    }
    if (f.image_fill) {
        if (f.failed) {
            index_->abort_snapshot_host_fill(f.snapshot);
        } else {
            index_->complete_snapshot_host_fill(f.snapshot);
        }
    }
    for (const Fetch::Block& b : f.blocks) {
        if (b.ok) {
            index_->complete_host_fill(b.node);
        } else {
            index_->abort_host_fill(b.node);
        }
    }
    f.ready   = true;
    f.expires = Clock::now() + kFetchHold;
    if (f.failed) { release_fetch(); }
}

void PrefixCache::service_fetch() {
    if (!fetch_) { return; }
    Fetch& f = *fetch_;
    if (f.ready) {
        if (Clock::now() >= f.expires) { release_fetch(); }
        return;
    }
    if (f.host) {
        if (f.reads != 0 ||
            std::any_of(f.blocks.begin(), f.blocks.end(), [](const Fetch::Block& b) { return b.stage != Fetch::Block::Stage::Done; })) {
            return;
        }
        if (f.tail_copy != nullptr) {
            const cudaError_t status = cudaEventQuery(f.tail_copy);
            if (status == cudaErrorNotReady) { return; }
            CUDA_CHECK(status);
            recycle(f.tail_copy);
            f.tail_copy = nullptr;
        }
        finish_host_fetch();
        return;
    }
    bool ordered = false;
    bool landed  = f.reads == 0;
    // The header is checked against the journal's meta in pinned memory, copied out of the landed
    // image.
    if (landed && f.region != kNoRegion && !f.failed && !f.header_copied) {
        if (f.header == nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(regions_[f.region].header.data(), region_base(f.region), CuFileQueue::kAlignment,
                                       cudaMemcpyDeviceToHost, transfer_));
            f.header = take_event();
            CUDA_CHECK(cudaEventRecord(f.header, transfer_));
        }
        const cudaError_t status = cudaEventQuery(f.header);
        if (status == cudaErrorNotReady) {
            landed = false;
        } else {
            CUDA_CHECK(status);
            recycle(f.header);
            f.header        = nullptr;
            f.header_copied = true;
        }
    }
    for (std::uint32_t i = 0; i < f.blocks.size(); ++i) {
        Fetch::Block& b = f.blocks[i];
        if (b.stage == Fetch::Block::Stage::Queued && !free_staging_.empty()) {
            b.staging = free_staging_.back();
            free_staging_.pop_back();
            submit_disk(CuFileQueue::Op::Read, b.slot, host_.slab_bytes, block_staging_, staging_offset(b.staging),
                        DiskIo{DiskIo::Kind::FetchBlock, i});
            b.stage = Fetch::Block::Stage::Reading;
        }
        if (b.stage == Fetch::Block::Stage::Copying && b.copied == nullptr) {
            // The page's previous users ran on the compute stream.
            if (!ordered) {
                cudaEvent_t start = take_event();
                CUDA_CHECK(cudaEventRecord(start, producer_));
                CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
                recycle(start);
                ordered = true;
            }
            const std::byte* const records[1]  = {staging_record(b.staging)};
            const std::uint32_t groups[1]       = {kStagingGroup};
            const DeviceKVPageHandle pages[1]   = {b.page};
            pool_->copy_from_host_records(records, groups, pages, host_.page, transfer_);
            b.copied = take_event();
            CUDA_CHECK(cudaEventRecord(b.copied, transfer_));
        }
        if (b.stage == Fetch::Block::Stage::Copying) {
            const cudaError_t status = cudaEventQuery(b.copied);
            if (status != cudaErrorNotReady) {
                CUDA_CHECK(status);
                index_->complete_device_fill(b.node);
                recycle(b.copied);
                b.copied = nullptr;
                free_staging_.push_back(b.staging);
                b.staging = kNoRegion;
                b.stage   = Fetch::Block::Stage::Done;
            }
        }
        landed = landed && b.stage == Fetch::Block::Stage::Done;
    }
    if (landed) { finish_fetch(); }
}

void PrefixCache::finish_fetch() {
    Fetch& f = *fetch_;
    for (const Fetch::Block& b : f.blocks) {
        if (!b.ok) { index_->abort_device_fill(b.node); }
    }
    if (!f.failed && f.region != kNoRegion) {
        const auto header = read_state_image_header(lanes_->layout(), static_cast<const std::byte*>(regions_[f.region].header.data()));
        if (!header || !same_meta(*header, snapshot_meta_[f.snapshot.index])) {
            ++counters_.disk_failures;
            f.failed = true;
        }
    }
    f.ready   = true;
    f.expires = Clock::now() + kFetchHold;
    if (f.failed) {
        unreadable_.push_back(f.snapshot);
        release_fetch();
    }
}

void PrefixCache::release_fetch() noexcept {
    if (!fetch_ || !fetch_->ready) { return; }
    try {
        if (index_->valid(fetch_->snapshot)) { index_->unpin_snapshot(fetch_->snapshot); }
        if (!fetch_->path.empty()) { index_->release_path(fetch_->path); }
    } catch (...) {}
    if (fetch_->region != kNoRegion) { release_region(fetch_->region); }
    fetch_.reset();
}

// ---- journal -------------------------------------------------------------------------------------

bool PrefixCache::journalable(pc::NodeRef node) const {
    for (; node.valid(); node = index_->node(node).parent) {
        const pc::NodeView view = index_->node(node);
        if (view.disk == pc::CopyState::Absent) { return false; }
        // A write in flight queues its record when it lands.
        if (view.disk == pc::CopyState::Filling || slot_journaled_[view.disk_slot]) { return true; }
    }
    return true;
}

void PrefixCache::service_journal() {
    bool progress = !journal_waits_.empty();
    while (progress) {
        progress = false;
        for (auto it = journal_waits_.begin(); it != journal_waits_.end();) {
            const JournalWait wait = *it;
            // The slot of the record this one links to, once appended; nothing while it is not. A
            // link whose record can no longer be appended (its path lost a Disk copy) ends the wait.
            bool drop          = true;
            const auto linked = [&](pc::NodeRef node) -> std::optional<std::uint32_t> {
                if (!node.valid()) { return pc::kNoId; }
                const pc::NodeView view = index_->node(node);
                if (view.disk == pc::CopyState::Resident && slot_journaled_[view.disk_slot]) { return view.disk_slot; }
                drop = !journalable(node);
                return std::nullopt;
            };
            if (wait.node.valid()) {
                if (index_->valid(wait.node)) {
                    const pc::NodeView view = index_->node(wait.node);
                    if (view.disk == pc::CopyState::Resident && !slot_journaled_[view.disk_slot]) {
                        if (const std::optional<std::uint32_t> parent = linked(view.parent)) {
                            const pc::BlockIdentity identity = index_->block_identity(wait.node);
                            DiskBlockRecord record;
                            record.slot        = view.disk_slot;
                            record.parent_slot = *parent;
                            record.lookup_hash = identity.lookup_hash;
                            record.extra       = identity.extra;
                            record.mtp_next    = node_mtp_next_[wait.node.index];
                            std::copy(identity.tokens.begin(), identity.tokens.end(), record.tokens.begin());
                            store_->append_block(record);
                            slot_journaled_[view.disk_slot] = true;
                            progress                        = true;
                        }
                    }
                }
            } else if (index_->valid(wait.snapshot)) {
                const pc::SnapshotView view = index_->snapshot(wait.snapshot);
                if (view.disk == pc::CopyState::Resident && !slot_journaled_[view.disk_slots.front()]) {
                    if (const std::optional<std::uint32_t> anchor = linked(view.anchor)) {
                        DiskSnapshotRecord record;
                        record.slots.assign(view.disk_slots.begin(), view.disk_slots.end());
                        record.anchor_slot = *anchor;
                        record.frontier    = view.frontier;
                        record.tail.assign(view.tail.begin(), view.tail.end());
                        record.kind = view.kind;
                        record.hits = view.hits;
                        record.meta = snapshot_meta_[wait.snapshot.index];
                        store_->append_snapshot(record);
                        slot_journaled_[view.disk_slots.front()]      = true;
                        journaled_snapshots_[view.disk_slots.front()] = wait.snapshot;
                        progress                                      = true;
                    }
                }
            }
            it = drop ? journal_waits_.erase(it) : std::next(it);
        }
    }
}

void PrefixCache::release_disk_record(std::span<const std::uint32_t> slots) noexcept {
    if (!store_ || slots.empty() || slots.front() >= slot_journaled_.size() || !slot_journaled_[slots.front()]) {
        // Never recorded: nothing names the slots, which may be rewritten at once.
        return;
    }
    slot_journaled_[slots.front()] = false;
    journaled_snapshots_.erase(slots.front());
    try {
        store_->append_removal(slots.front(), slots);
    } catch (...) {
        disk_broken_ = true; // the slots would be rewritten under a live record
    }
}

void PrefixCache::disk_record_replaced(pc::NodeRef node) noexcept {
    try {
        replaced_.push_back(node);
    } catch (...) {
        disk_broken_ = true;
    }
}

void PrefixCache::service_replaced() {
    for (const pc::NodeRef node : replaced_) {
        if (!index_->valid(node)) { continue; }
        // The journal links each record below to the released one, so a restart would drop them:
        // they are removed from the journal and appended again once the block's new record is.
        std::vector<pc::NodeRef> blocks;
        std::vector<pc::SnapshotRef> snapshots;
        index_->disk_records_below(node, blocks, snapshots);
        if (blocks.empty() && snapshots.empty()) { continue; }
        double claim = 0.0;
        for (const pc::NodeRef block : blocks) {
            const std::uint32_t slot = index_->node(block).disk_slot;
            release_disk_record(std::span<const std::uint32_t>(&slot, 1));
            journal_waits_.push_back(JournalWait{block, {}});
        }
        for (const pc::SnapshotRef snapshot : snapshots) {
            const pc::SnapshotView view = index_->snapshot(snapshot);
            release_disk_record(view.disk_slots);
            journal_waits_.push_back(JournalWait{{}, snapshot});
            claim = std::max(claim, view.priority);
        }
        // Without a slot, the records below stay out of the journal: a restart does not rebuild them.
        (void)queue_block_write(node, claim);
    }
    replaced_.clear();
}

// ---- service -------------------------------------------------------------------------------------

void PrefixCache::poll_disk() {
    service_replaced();
    service_block_writes();
    service_spill();
    service_fetch();
    service_journal();
    // The next spill waits for the previous one's path: each queued block write pins its block's
    // Device page or Host slab, and spills queued faster than the Disk absorbs them would pin the
    // whole cache.
    if (host_tier() && !spill_ && block_writes_.empty() && disk_writable() && Clock::now() >= next_spill_scan_) {
        next_spill_scan_ = Clock::now() + kSpillScan;
        for (const pc::SnapshotRef snapshot : index_->disk_spill_candidates(host_.slabs / 4U)) {
            if (start_spill(snapshot)) {
                service_spill();
                break;
            }
        }
    }
}

bool PrefixCache::disk_writes_pending() const noexcept {
    return spill_.has_value() || !block_writes_.empty() ||
           std::any_of(captures_.begin(), captures_.end(),
                       [](const PendingCapture& c) {
                           return c.deferred != nullptr || (c.image.disk && (!c.writes_issued || c.writes != 0));
                       });
}

void PrefixCache::drain_disk() {
    while (disk_writes_pending()) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        poll();
    }
}

// ---- startup and shutdown ------------------------------------------------------------------------

DiskAttachResult PrefixCache::attach_disk(const std::filesystem::path& directory, std::string_view fingerprint) {
    if (disk_slots_ == 0) { throw std::logic_error("Qwen4Exp prefix cache: no Disk tier was sized"); }
    if (store_) { throw std::logic_error("Qwen4Exp prefix cache: the Disk tier is already attached"); }
    if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
        throw std::logic_error("Qwen4Exp prefix cache: the Disk tier is attached to a cache in use");
    }
    const auto start = Clock::now();
    const DiskGeometry geometry{.slot_bytes        = host_.slab_bytes,
                                .slots             = disk_slots_,
                                .image_slots       = host_.image_slabs,
                                .page_stride       = host_.page.page_stride,
                                .image_fingerprint = lanes_->layout().fingerprint};
    store_ = std::make_unique<DiskStore>(directory, fingerprint, geometry);
    // Registered staging moves without cuFile's bounce buffers where GPUDirect Storage serves it.
    for (Region& region : regions_) { (void)store_->io().register_buffer(region.image.p, region.image.bytes); }
    if (block_staging_device_.p != nullptr) {
        (void)store_->io().register_buffer(block_staging_device_.p, block_staging_device_.bytes);
    }

    DiskAttachResult out;
    out.reset_reason = store_->reset_reason();
    out.damaged      = store_->damaged();
    out.mode         = store_->io().mode();
    std::vector<DiskBlockRecord> kept_blocks;
    std::vector<DiskSnapshotRecord> kept_snapshots;
    std::unordered_map<std::uint32_t, pc::NodeRef> nodes; // by slot
    index_->begin_disk_rebuild();
    for (const DiskBlockRecord& record : store_->replayed_blocks()) {
        pc::NodeRef parent;
        if (record.parent_slot != pc::kNoId) {
            const auto found = nodes.find(record.parent_slot);
            if (found == nodes.end()) {
                ++out.dropped_blocks;
                continue;
            }
            parent = found->second;
        }
        const std::optional<pc::NodeRef> node =
            index_->restore_disk_block(parent, record.lookup_hash, record.tokens, record.extra, record.slot);
        if (!node) {
            ++out.dropped_blocks;
            continue;
        }
        nodes.emplace(record.slot, *node);
        node_mtp_next_[node->index] = record.mtp_next;
        slot_journaled_[record.slot] = true;
        kept_blocks.push_back(record);
    }
    for (const DiskSnapshotRecord& record : store_->replayed_snapshots()) {
        pc::NodeRef anchor;
        if (record.anchor_slot != pc::kNoId) {
            const auto found = nodes.find(record.anchor_slot);
            if (found == nodes.end()) {
                ++out.dropped_snapshots;
                continue;
            }
            anchor = found->second;
        }
        const std::optional<pc::SnapshotRef> snapshot = index_->restore_disk_snapshot(
            anchor, record.frontier, record.tail, record.kind, record.hits, record.meta.head_residual, record.slots);
        if (!snapshot) {
            ++out.dropped_snapshots;
            continue;
        }
        snapshot_meta_[snapshot->index]              = record.meta;
        slot_journaled_[record.slots.front()]        = true;
        journaled_snapshots_[record.slots.front()]   = *snapshot;
        kept_snapshots.push_back(record);
    }
    index_->end_disk_rebuild();
    store_->compact(kept_blocks, kept_snapshots);
    out.blocks    = kept_blocks.size();
    out.snapshots = kept_snapshots.size();
    std::uint64_t slots = kept_blocks.size();
    for (const DiskSnapshotRecord& record : kept_snapshots) { slots += record.slots.size(); }
    out.bytes   = slots * host_.slab_bytes;
    out.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

DiskSpillResult PrefixCache::spill_for_shutdown(const CancellationView& abandoned) {
    DiskSpillResult out;
    if (!store_) {
        out.message = "the Disk tier is not attached";
        return out;
    }
    const auto start                   = Clock::now();
    const PrefixCacheCounters before   = counters_;
    drain();
    if (host_tier()) {
        std::vector<pc::SnapshotRef> order = index_->disk_spill_candidates(UINT64_MAX);
        std::reverse(order.begin(), order.end()); // the most valuable first
        for (const pc::SnapshotRef snapshot : order) {
            if (abandoned.requested()) {
                out.message = "abandoned";
                break;
            }
            if (!disk_writable()) { break; }
            if (!start_spill(snapshot)) { continue; }
            drain_disk();
        }
    }
    drain_disk();
    service_journal();
    for (const auto& [first, snapshot] : journaled_snapshots_) {
        if (!index_->valid(snapshot)) { continue; }
        const pc::SnapshotView view = index_->snapshot(snapshot);
        if (view.disk == pc::CopyState::Resident && view.disk_slots.front() == first) { store_->append_hits(first, view.hits); }
    }
    try {
        store_->flush();
        if (disk_broken_) { out.message = "a removal could not be recorded"; }
    } catch (const std::exception& error) { out.message = error.what(); }
    out.ok        = out.message.empty();
    out.snapshots = counters_.disk_image_writes - before.disk_image_writes;
    out.blocks    = counters_.disk_block_writes - before.disk_block_writes;
    out.bytes     = counters_.disk_write_bytes - before.disk_write_bytes;
    out.seconds   = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

} // namespace infernix::models::qwen4_exp::prefix
