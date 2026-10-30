#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

// Largest single pinned allocation of the slab pool: one very large pinned allocation can fail or
// stall under WDDM where several smaller ones succeed. A slab never crosses chunks.
constexpr std::size_t kHostChunkBytes = std::size_t{4} << 30U;
constexpr std::size_t kSlabAlignment  = 4096;
// The Disk tier's VRAM staging: whole-image regions and block records. A record's copy group (it
// lies in no Host chunk) is one id per buffer.
constexpr std::uint32_t kDiskRegions     = 2;
constexpr std::uint32_t kStagingRecords  = 64;

std::size_t align_slab(std::size_t bytes) { return (bytes + kSlabAlignment - 1U) / kSlabAlignment * kSlabAlignment; }

} // namespace

PrefixHostLayout plan_prefix_host_layout(const KvPageGeometry& pages, const StateImageLayout& image,
                                         std::uint64_t host_bytes) {
    PrefixHostLayout out;
    out.page        = plan_host_kv_page_layout(pages.geometry);
    out.slab_bytes  = align_slab(out.page.page_stride);
    out.image_bytes = image.image_bytes;
    out.image_slabs = static_cast<std::uint32_t>((image.image_bytes + out.slab_bytes - 1U) / out.slab_bytes);
    const std::uint64_t slabs = host_bytes / out.slab_bytes;
    if (slabs > pc::kNoId / 4U) { throw std::invalid_argument("Qwen4Exp prefix cache: the Host tier is too large"); }
    out.slabs           = static_cast<std::uint32_t>(slabs);
    out.slabs_per_chunk = static_cast<std::uint32_t>(std::max<std::size_t>(1U, kHostChunkBytes / out.slab_bytes));
    return out;
}

PrefixCache::PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const KvPageGeometry& pages,
                         const LaneStateImage& lanes, std::uint64_t host_bytes, std::uint64_t disk_bytes)
    : taps_(config.taps), pool_(&pool), geometry_(pages), lanes_(&lanes),
      host_(plan_prefix_host_layout(pages, lanes.layout(), host_bytes)) {
    if (pool.plane_count() != pages.geometry.planes.size()) {
        throw std::invalid_argument("Qwen4Exp prefix cache: the page geometry does not match the pool");
    }
    const std::uint64_t disk_slots = disk_bytes / host_.slab_bytes;
    if (disk_slots > pc::kNoId / 4U) { throw std::invalid_argument("Qwen4Exp prefix cache: the Disk tier is too large"); }
    disk_slots_ = static_cast<std::uint32_t>(disk_slots);
    if (host_.slabs == 0 && disk_slots_ == 0) {
        // Snapshots are born in the Host tier or written through to the Disk tier: without either
        // there is no cache.
        throw std::invalid_argument(
            "Qwen4Exp prefix cache needs a Host tier (--host-context-mib) or a Disk tier (--prefix-cache-disk)");
    }
    const std::uint64_t snapshot_mib =
        (static_cast<std::uint64_t>(host_.image_slabs + 2U) * host_.slab_bytes + (1U << 20U) - 1U) >> 20U;
    if (host_.slabs != 0 && host_.slabs < host_.image_slabs + 2U) {
        throw std::invalid_argument("--host-context-mib is too small for the Qwen4Exp prefix cache: one snapshot needs " +
                                    std::to_string(snapshot_mib) + " MiB");
    }
    if (disk_slots_ != 0 && disk_slots_ < host_.image_slabs + 2U) {
        throw std::invalid_argument("--prefix-cache-disk-gib is too small for the Qwen4Exp prefix cache: one snapshot needs " +
                                    std::to_string(snapshot_mib) + " MiB");
    }
    pc::PrefixIndexConfig index = config.index;
    index.max_nodes             = config.kv_pages + host_.slabs + disk_slots_ + 1U;
    index.max_snapshots = config.device_snapshot_slots + host_.slabs / host_.image_slabs + disk_slots_ / host_.image_slabs + 1U;
    index.host_slabs            = host_.slabs;
    index.disk_slots            = disk_slots_;
    index.image_slabs           = host_.image_slabs;
    index.device_snapshot_slots = config.device_snapshot_slots;
    index.block_bytes           = host_.page.page_stride;
    index.image_bytes           = host_.image_bytes;
    index.host_blocks           = true;
    if (config.device_snapshot_slots != 0) {
        throw std::invalid_argument("Qwen4Exp prefix cache: Device snapshot slots are not supported");
    }
    const std::uint64_t total = static_cast<std::uint64_t>(host_.slabs) * host_.slab_bytes;
    std::uint64_t pinned      = 0;
    for (std::uint32_t first = 0; first < host_.slabs; first += host_.slabs_per_chunk) {
        const std::uint32_t slabs = std::min(host_.slabs_per_chunk, host_.slabs - first);
        const std::size_t bytes   = static_cast<std::size_t>(slabs) * host_.slab_bytes;
        try {
            chunks_.emplace_back(bytes);
        } catch (const std::exception& error) {
            throw std::runtime_error("pinning the Qwen4Exp prefix cache Host tier failed after " +
                                     std::to_string(pinned >> 20U) + " of " + std::to_string(total >> 20U) + " MiB (" +
                                     error.what() + "); lower --host-context-mib");
        }
        pinned += bytes;
    }
    if (disk_slots_ != 0) {
        const std::size_t staging_bytes = static_cast<std::size_t>(kStagingRecords) * host_.slab_bytes;
        if (host_.slabs == 0) {
            // Without a Host tier, two VRAM regions: a capture written through and a fetch, or a
            // fetch read by its restore while the next one lands. With one, fetches read into Host
            // slabs and nothing stages in VRAM.
            for (std::uint32_t r = 0; r < kDiskRegions; ++r) {
                regions_.push_back(Region{DeviceBuffer(static_cast<std::size_t>(host_.image_slabs + 1U) * host_.slab_bytes),
                                          PinnedHostBuffer(CuFileQueue::kAlignment), 0});
            }
            block_staging_device_ = DeviceBuffer(staging_bytes);
            block_staging_        = static_cast<std::byte*>(block_staging_device_.p);
        } else {
            block_staging_host_.emplace(staging_bytes);
            block_staging_ = static_cast<std::byte*>(block_staging_host_->data());
        }
        for (std::uint32_t i = kStagingRecords; i-- > 0;) { free_staging_.push_back(i); }
        slot_journaled_.assign(disk_slots_, false);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&restore_stream_, cudaStreamNonBlocking));
    node_mtp_next_.assign(index.max_nodes, -1);
    snapshot_meta_.assign(index.max_snapshots, StateImageHeader{});
    blocks_.reserve(index.max_nodes);
    index_.emplace(index, *this);
}

PrefixCache::~PrefixCache() {
    if (store_) {
        // Transfers in flight still move bytes of the staging buffers and Host slabs.
        try {
            store_->io().wait(completions_);
        } catch (...) {}
        store_.reset();
    }
    // Only CUDA objects are released here; leases go back to the pool with blocks_.
    for (cudaStream_t stream : {transfer_, restore_stream_}) {
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }
    }
    for (PendingCapture& c : captures_) {
        for (cudaEvent_t e : c.events) { (void)cudaEventDestroy(e); }
        if (c.header_done != nullptr) { (void)cudaEventDestroy(c.header_done); }
        if (c.deferred != nullptr) { (void)cudaEventDestroy(c.deferred); }
    }
    for (auto& [id, w] : block_writes_) {
        if (w.copied != nullptr) { (void)cudaEventDestroy(w.copied); }
    }
    if (fetch_) {
        for (Fetch::Block& b : fetch_->blocks) {
            if (b.copied != nullptr) { (void)cudaEventDestroy(b.copied); }
        }
    }
    for (auto& [id, s] : staged_) {
        for (cudaEvent_t e : s.events) { (void)cudaEventDestroy(e); }
    }
    for (PendingWrite& w : writes_) {
        if (w.done != nullptr) { (void)cudaEventDestroy(w.done); }
    }
    for (cudaEvent_t e : restore_.events) { (void)cudaEventDestroy(e); }
    for (RestoreBatch& batch : landing_) {
        for (cudaEvent_t e : batch.events) { (void)cudaEventDestroy(e); }
    }
    for (cudaEvent_t e : spare_events_) { (void)cudaEventDestroy(e); }
    index_.reset();
}

std::uint64_t PrefixCache::pinned_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const PinnedHostBuffer& chunk : chunks_) { bytes += chunk.size(); }
    return bytes;
}

// ---- ids, events, slabs ----------------------------------------------------------------------------

std::uint32_t PrefixCache::allocate_block_id(DeviceKVPageLease&& lease) {
    if (!free_blocks_.empty()) {
        const std::uint32_t id = free_blocks_.back();
        free_blocks_.pop_back();
        blocks_[id].emplace(std::move(lease));
        return id;
    }
    if (blocks_.size() >= pc::kNoId - 1U) { throw std::overflow_error("Qwen4Exp prefix cache block ids exhausted"); }
    blocks_.emplace_back(std::move(lease));
    return static_cast<std::uint32_t>(blocks_.size() - 1U);
}

DeviceKVPageHandle PrefixCache::page(std::uint32_t device_id) const {
    if (device_id >= blocks_.size() || !blocks_[device_id]) {
        throw std::logic_error("Qwen4Exp prefix cache block id is not registered");
    }
    return blocks_[device_id]->handle();
}

std::byte* PrefixCache::slab(std::uint32_t id) const noexcept {
    const std::uint32_t chunk = id / host_.slabs_per_chunk;
    return static_cast<std::byte*>(chunks_[chunk].data()) +
           static_cast<std::size_t>(id % host_.slabs_per_chunk) * host_.slab_bytes;
}

cudaEvent_t PrefixCache::take_event() {
    if (!spare_events_.empty()) {
        cudaEvent_t event = spare_events_.back();
        spare_events_.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void PrefixCache::recycle(cudaEvent_t event) noexcept {
    if (event != nullptr) { spare_events_.push_back(event); }
}

// ---- blocks --------------------------------------------------------------------------------------

pc::InsertResult PrefixCache::insert_block(pc::NodeRef parent, std::uint64_t lookup_hash,
                                           std::span<const TokenId> tokens, std::uint64_t extra,
                                           DeviceKVPageLease& lease, bool attach, std::int32_t mtp_next) {
    if (!lease.valid()) { throw std::invalid_argument("Qwen4Exp prefix cache: a block needs its page"); }
    const DeviceKVPageHandle handle = lease.handle();
    const std::uint32_t id          = allocate_block_id(std::move(lease));
    pc::InsertResult result;
    try {
        result = index_->insert_block(parent, lookup_hash, tokens, extra, id, attach);
    } catch (...) {
        lease = std::move(*blocks_[id]);
        blocks_[id].reset();
        free_blocks_.push_back(id);
        throw;
    }
    if (!result.device_attached) {
        // The index kept an existing copy: the page stays the caller's private page.
        lease = std::move(*blocks_[id]);
        blocks_[id].reset();
        free_blocks_.push_back(id);
        ++counters_.blocks_duplicate;
    } else {
        // The replaced Device copy is unmapped and unread (the index replaces only unpinned ones),
        // so its page returns to the pool as an eviction's would.
        if (result.replaced_device_id != pc::kNoId) { release_device_block(result.replaced_device_id); }
        node_mtp_next_[result.node.index] = mtp_next;
        ++(result.inserted ? counters_.blocks_inserted : counters_.blocks_reattached);
    }
    (void)handle;
    return result;
}

std::int32_t PrefixCache::mtp_next(pc::NodeRef node) const {
    if (!index_->valid(node)) { throw std::logic_error("Qwen4Exp prefix cache: stale node"); }
    return node_mtp_next_[node.index];
}

std::uint32_t PrefixCache::adopt_tail(DeviceKVPageLease&& lease) { return allocate_block_id(std::move(lease)); }

const StateImageHeader& PrefixCache::meta(pc::SnapshotRef snapshot) const {
    if (!index_->valid(snapshot)) { throw std::logic_error("Qwen4Exp prefix cache: stale snapshot"); }
    return snapshot_meta_[snapshot.index];
}

// ---- captures and write-through ------------------------------------------------------------------

bool PrefixCache::capture(std::uint32_t lane, Capture&& capture, cudaStream_t producer) {
    const bool tail = !capture.tail.empty();
    if (tail != capture.tail_page.has_value() || capture.tail.size() >= pc::kBlockTokens) {
        throw std::invalid_argument("Qwen4Exp prefix cache: a capture's tail does not match its page");
    }
    auto image = reserve_image(tail, capture.claim);
    if (!image) {
        ++counters_.taps_skipped;
        return false;
    }
    // The anchor must outlive the copy: the capturing lane may release its path before it lands.
    if (capture.anchor.valid()) { index_->pin_node(capture.anchor); }
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
    recycle(start);
    const std::vector<std::byte*> segments = image_segments(*image);
    write_header(*image, capture.meta, transfer_);
    const HostImage destination{segments, host_.slab_bytes};
    std::byte* const record          = tail ? tail_record(*image) : nullptr;
    const std::uint32_t group        = tail ? tail_group(*image) : 0U;
    const std::optional<DeviceKVPageHandle> tail_page = capture.tail_page;
    const bool host = !image->disk;
    PendingCapture pending{std::move(capture), std::move(*image), {}};
    for (std::uint32_t g = 0; g < lanes_->layout().groups(); ++g) {
        copy_capture_group(g, lane, destination, record, group, tail_page);
        cudaEvent_t event = take_event();
        CUDA_CHECK(cudaEventRecord(event, transfer_));
        pending.events.push_back(event);
    }
    if (host) {
        counters_.host_write_bytes += host_.image_bytes + (tail ? host_.page.page_stride : 0U);
        ++counters_.host_image_writes;
    }
    captures_.push_back(std::move(pending));
    return true;
}

void PrefixCache::set_prompt_end_staging(std::vector<std::byte*> slots) {
    prompt_end_headers_.clear();
    prompt_end_headers_.reserve(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        prompt_end_headers_.emplace_back(kStateImageHeaderBytes);
    }
    prompt_end_slots_ = std::move(slots);
}

bool PrefixCache::capture_prompt_end(std::uint32_t lane, Capture&& capture, cudaStream_t producer) {
    const bool tail = !capture.tail.empty();
    if (tail != capture.tail_page.has_value() || tail != capture.tail_lease.has_value() ||
        capture.tail.size() >= pc::kBlockTokens) {
        throw std::invalid_argument("Qwen4Exp prefix cache: a prompt-end capture's tail does not match its page");
    }
    if (capture.anchor_pending && (tail || capture.anchor.valid() || capture.frontier == 0)) {
        throw std::invalid_argument("Qwen4Exp prefix cache: only a block-aligned prompt end awaits its anchor");
    }
    // A free slot: one no pending capture reads. A landed-but-unpublished capture still holds it
    // until the next poll, which every admission runs.
    std::size_t slot = prompt_end_slots_.size();
    for (std::size_t i = 0; i < prompt_end_slots_.size() && slot == prompt_end_slots_.size(); ++i) {
        if (std::none_of(captures_.begin(), captures_.end(),
                         [&](const PendingCapture& c) { return c.staging == prompt_end_slots_[i]; })) {
            slot = i;
        }
    }
    if (slot == prompt_end_slots_.size()) {
        ++counters_.prompt_end_failed;
        return false;
    }
    std::byte* const staging = prompt_end_slots_[slot];
    // Without a Host tier, both regions may be held (an open tap, a tap being written): the image
    // waits in its slot for one rather than being dropped, since the prompt end serves repeats.
    const bool defer = !host_tier() && disk_writable() &&
                       std::none_of(regions_.begin(), regions_.end(), [](const Region& r) { return r.users == 0; });
    std::optional<ImageTarget> image;
    if (!defer) {
        image = reserve_image(tail, capture.claim);
        if (!image) {
            ++counters_.prompt_end_failed;
            return false;
        }
    }
    // The anchor must outlive the copy: the capturing lane may release its path before it lands.
    if (capture.anchor.valid()) { index_->pin_node(capture.anchor); }
    // The header and every part into the slot, on the producer: the compute stream runs the lane's
    // next (graph-replayed) round only after the copy, so the slot never sees the state move.
    write_state_image_header(lanes_->layout(), capture.meta,
                             static_cast<std::byte*>(prompt_end_headers_[slot].data()));
    CUDA_CHECK(cudaMemcpyAsync(staging, prompt_end_headers_[slot].data(), kStateImageHeaderBytes,
                               cudaMemcpyHostToDevice, producer));
    lanes_->copy_lane_to_packed(lane, staging, producer);
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    ++counters_.prompt_end_captures;
    PendingCapture pending{std::move(capture), {}, {}, staging};
    if (defer) {
        pending.deferred = start;
        captures_.push_back(std::move(pending));
        return true;
    }
    pending.image = std::move(*image);
    copy_prompt_end(pending, start);
    recycle(start);
    captures_.push_back(std::move(pending));
    return true;
}

void PrefixCache::copy_prompt_end(PendingCapture& c, cudaEvent_t after) {
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, after, 0));
    const bool tail                        = !c.capture.tail.empty();
    const std::vector<std::byte*> segments = image_segments(c.image);
    lanes_->copy_packed_to_host(c.staging, HostImage{segments, host_.slab_bytes}, transfer_);
    if (tail) { copy_tail_page_to_record(*c.capture.tail_page, tail_record(c.image), tail_group(c.image), transfer_); }
    // One event per layer group, as the group-by-group captures take: the whole image is copied at
    // once, so every layer's wait is the copy's end (a lane's next call waits the full set).
    for (std::uint32_t g = 0; g < lanes_->layout().groups(); ++g) {
        cudaEvent_t event = take_event();
        CUDA_CHECK(cudaEventRecord(event, transfer_));
        c.events.push_back(event);
    }
    if (!c.image.disk) {
        counters_.host_write_bytes += host_.image_bytes + (tail ? host_.page.page_stride : 0U);
        ++counters_.host_image_writes;
    }
}

bool PrefixCache::start_deferred(PendingCapture& c, bool& dropped) {
    dropped = false;
    if (std::none_of(regions_.begin(), regions_.end(), [](const Region& r) { return r.users == 0; })) { return false; }
    std::optional<ImageTarget> image = reserve_image(!c.capture.tail.empty(), c.capture.claim);
    if (!image) {
        dropped = true;
        ++counters_.prompt_end_failed;
        return false;
    }
    c.image = std::move(*image);
    copy_prompt_end(c, c.deferred);
    recycle(c.deferred);
    c.deferred = nullptr;
    return true;
}

void PrefixCache::copy_tail_page_to_record(DeviceKVPageHandle page, std::byte* record, std::uint32_t group,
                                           cudaStream_t stream) const {
    for (std::uint32_t k = 0; k < geometry_.kv_layers; ++k) {
        const PlaneRange planes = kv_layer_planes(geometry_, k);
        const DeviceKVPageHandle pages[1] = {page};
        std::byte* const records[1]       = {record};
        const std::uint32_t groups[1]      = {group};
        pool_->copy_to_host_records(std::span(pages, 1), std::span(records, 1), std::span(groups, 1), host_.page,
                                    planes.begin, planes.end, stream);
    }
}

std::vector<std::byte*> PrefixCache::image_segments(const ImageTarget& image) const {
    std::vector<std::byte*> segments;
    segments.reserve(host_.image_slabs);
    for (std::uint32_t i = 0; i < host_.image_slabs; ++i) {
        segments.push_back(image.disk ? region_base(image.region) + static_cast<std::size_t>(i) * host_.slab_bytes
                                      : slab(image.host.slabs[i]));
    }
    return segments;
}

std::byte* PrefixCache::tail_record(const ImageTarget& image) const {
    return image.disk ? region_base(image.region) + static_cast<std::size_t>(host_.image_slabs) * host_.slab_bytes
                      : slab(image.host.slabs.back());
}

std::uint32_t PrefixCache::tail_group(const ImageTarget& image) const {
    return image.disk ? kNoRegion - 1U - image.region : slab_group(image.host.slabs.back());
}

std::optional<PrefixCache::ImageTarget> PrefixCache::reserve_image(bool tail, double claim) {
    if (host_tier()) {
        auto host = index_->reserve_host_image(tail, claim);
        if (!host) { return std::nullopt; }
        return ImageTarget{std::move(*host), std::nullopt, kNoRegion};
    }
    if (!disk_writable()) { return std::nullopt; }
    const std::optional<std::uint32_t> region = take_region();
    if (!region) {
        ++counters_.disk_capture_skips;
        return std::nullopt;
    }
    auto disk = index_->reserve_disk_image(tail, claim);
    if (!disk) {
        release_region(*region);
        ++counters_.disk_capture_skips;
        return std::nullopt;
    }
    return ImageTarget{{}, std::move(*disk), *region};
}

void PrefixCache::release_image(ImageTarget&& image) noexcept {
    if (image.disk) {
        index_->release_disk_image(std::move(*image.disk));
        image.disk.reset();
        release_region(image.region);
        image.region = kNoRegion;
    } else {
        index_->release_host_image(std::move(image.host));
    }
}

void PrefixCache::write_header(const ImageTarget& image, const StateImageHeader& meta, cudaStream_t stream) {
    if (!image.disk) {
        write_state_image_header(lanes_->layout(), meta, slab(image.host.slabs.front()));
        return;
    }
    // The region's pinned header is its own until the region is released.
    auto* header = static_cast<std::byte*>(regions_[image.region].header.data());
    write_state_image_header(lanes_->layout(), meta, header);
    CUDA_CHECK(cudaMemcpyAsync(region_base(image.region), header, kStateImageHeaderBytes, cudaMemcpyHostToDevice, stream));
}

void PrefixCache::copy_capture_group(std::uint32_t group, std::uint32_t lane, const HostImage& destination,
                                     std::byte* tail_record, std::uint32_t tail_group,
                                     std::optional<DeviceKVPageHandle> tail_page) {
    const StateImageLayout& layout = lanes_->layout();
    // KV layer of the group: attention layers in order, then MTP's (when the model keeps one).
    std::optional<std::uint32_t> kv_layer;
    std::uint32_t attention = 0;
    for (std::uint32_t g = 0; g < group && g < layout.spec.layer_types.size(); ++g) {
        if (layout.spec.layer_types[g] == MixerKind::Attention) { ++attention; }
    }
    if (group == layout.drafter_group()) {
        if (geometry_.kv_layers > attention) { kv_layer = attention; }
    } else if (layout.spec.layer_types[group] == MixerKind::Attention) {
        kv_layer = attention;
    }
    if (tail_page && kv_layer) {
        const PlaneRange planes            = kv_layer_planes(geometry_, *kv_layer);
        std::byte* const records[1]        = {tail_record};
        const std::uint32_t groups[1]      = {tail_group};
        const DeviceKVPageHandle pages[1]  = {*tail_page};
        pool_->copy_to_host_records(pages, records, groups, host_.page, planes.begin, planes.end, transfer_);
    }
    lanes_->copy_group_to_host(group, lane, destination, transfer_);
}

std::optional<std::uint64_t> PrefixCache::open_staged_capture(bool tail, double claim) {
    auto image = reserve_image(tail, claim);
    if (!image) {
        ++counters_.taps_skipped;
        return std::nullopt;
    }
    StagedCapture staged{std::move(*image), tail, {}, 0};
    for (std::uint32_t g = 0; g < lanes_->layout().groups(); ++g) { staged.events.push_back(take_event()); }
    const std::uint64_t id = next_staged_++;
    staged_.emplace(id, std::move(staged));
    return id;
}

void PrefixCache::capture_group(std::uint64_t staged, std::uint32_t group, std::uint32_t lane,
                                std::optional<DeviceKVPageHandle> tail_page, cudaStream_t producer) {
    StagedCapture& s = staged_.at(staged);
    if (group != s.copied || tail_page.has_value() != s.tail) {
        throw std::logic_error("Qwen4Exp prefix cache: a staged capture's groups are copied out of order");
    }
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
    recycle(start);
    const std::vector<std::byte*> segments = image_segments(s.image);
    copy_capture_group(group, lane, HostImage{segments, host_.slab_bytes}, s.tail ? tail_record(s.image) : nullptr,
                       s.tail ? tail_group(s.image) : 0U, tail_page);
    CUDA_CHECK(cudaEventRecord(s.events[group], transfer_));
    ++s.copied;
}

std::span<const cudaEvent_t> PrefixCache::staged_events(std::uint64_t staged) const {
    return staged_.at(staged).events;
}

void PrefixCache::commit_staged_capture(std::uint64_t staged, Capture&& capture) {
    const auto found = staged_.find(staged);
    if (found == staged_.end() || found->second.copied != lanes_->layout().groups() ||
        found->second.tail != !capture.tail.empty() || capture.tail.size() >= pc::kBlockTokens) {
        throw std::logic_error("Qwen4Exp prefix cache: a staged capture is committed incomplete or with another tail");
    }
    StagedCapture s = std::move(found->second);
    staged_.erase(found);
    if (capture.anchor.valid()) { index_->pin_node(capture.anchor); }
    write_header(s.image, capture.meta, transfer_);
    PendingCapture pending{std::move(capture), std::move(s.image), std::move(s.events)};
    if (pending.image.disk) {
        // The header's copy follows the groups' on the transfer stream: the region has landed after it.
        pending.header_done = take_event();
        CUDA_CHECK(cudaEventRecord(pending.header_done, transfer_));
    } else {
        counters_.host_write_bytes += host_.image_bytes + (s.tail ? host_.page.page_stride : 0U);
        ++counters_.host_image_writes;
    }
    captures_.push_back(std::move(pending));
}

void PrefixCache::abandon_staged_capture(std::uint64_t staged) noexcept {
    const auto found = staged_.find(staged);
    if (found == staged_.end()) { return; }
    // The slabs may still be written: an unrecorded event's synchronize returns at once.
    for (cudaEvent_t event : found->second.events) {
        (void)cudaEventSynchronize(event);
        recycle(event);
    }
    release_image(std::move(found->second.image));
    staged_.erase(found);
}

std::span<const cudaEvent_t> PrefixCache::capture_events(std::uint64_t id) const noexcept {
    for (const PendingCapture& c : captures_) {
        if (c.capture.id == id) { return c.events; }
    }
    return {};
}

void PrefixCache::order_page_reuse(std::span<const DeviceKVPageLease> pages) const {
    for (const PendingCapture& c : captures_) {
        // An endpoint owns its tail page and a prompt-end capture copies its own; only a tap
        // reads a page someone else can free.
        if (c.staging != nullptr || !c.capture.tail_page || c.capture.tail_lease || c.events.empty()) {
            continue;
        }
        const bool reads = std::any_of(pages.begin(), pages.end(), [&](const DeviceKVPageLease& page) {
            return page.valid() && page.handle() == *c.capture.tail_page;
        });
        if (reads) { CUDA_CHECK(cudaStreamWaitEvent(producer_, c.events.back(), 0)); }
    }
}

void PrefixCache::write_blocks(std::span<const pc::NodeRef> nodes, cudaStream_t producer) {
    if (!host_tier()) { return; }
    std::vector<pc::NodeRef> written;
    std::vector<DeviceKVPageHandle> pages;
    std::vector<std::byte*> records;
    std::vector<std::uint32_t> groups;
    for (const pc::NodeRef node : nodes) {
        if (!index_->valid(node)) { continue; }
        const pc::NodeView view = index_->node(node);
        if (view.device != pc::CopyState::Resident || view.host != pc::CopyState::Absent) { continue; }
        index_->pin_node(node);
        const auto slab_id = index_->begin_host_fill(node);
        if (!slab_id) {
            index_->unpin_node(node);
            break; // the Host tier cannot free a slab for anything worth less
        }
        written.push_back(node);
        pages.push_back(page(view.device_id));
        records.push_back(slab(*slab_id));
        groups.push_back(slab_group(*slab_id));
    }
    if (written.empty()) { return; }
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
    recycle(start);
    pool_->copy_to_host_records(pages, records, groups, host_.page, transfer_);
    cudaEvent_t done = take_event();
    CUDA_CHECK(cudaEventRecord(done, transfer_));
    counters_.host_block_writes += written.size();
    counters_.host_write_bytes += written.size() * host_.page.page_stride;
    writes_.push_back(PendingWrite{std::move(written), done});
}

void PrefixCache::resolve_capture_anchor(std::uint64_t id, pc::NodeRef anchor) {
    for (PendingCapture& c : captures_) {
        if (c.capture.id != id) { continue; }
        if (!c.capture.anchor_pending || !anchor.valid()) {
            throw std::logic_error("Qwen4Exp prefix cache: a capture's anchor is resolved twice or to no block");
        }
        index_->pin_node(anchor);
        c.capture.anchor         = anchor;
        c.capture.anchor_pending = false;
        return;
    }
    throw std::logic_error("Qwen4Exp prefix cache: an anchor resolved for no pending capture");
}

void PrefixCache::abandon_capture(std::uint64_t id) noexcept {
    const auto found = std::find_if(captures_.begin(), captures_.end(),
                                    [&](const PendingCapture& c) { return c.capture.id == id; });
    if (found == captures_.end()) { return; }
    if (found->deferred != nullptr) {
        // Nothing reserved yet: the slot is free once the producer's copy into it has ended.
        (void)cudaEventSynchronize(found->deferred);
        recycle(found->deferred);
        if (found->capture.anchor.valid()) { index_->unpin_node(found->capture.anchor); }
        captures_.erase(found);
        return;
    }
    // The slabs and the staging slot are still read until the copy lands, a region until its
    // Disk writes complete.
    (void)cudaEventSynchronize(found->events.back());
    if (found->header_done != nullptr) { (void)cudaEventSynchronize(found->header_done); }
    try {
        while (found->writes != 0) {
            reap_disk();
            if (found->writes != 0) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
        }
    } catch (...) {
        std::terminate(); // the region would be reused under a transfer in flight
    }
    for (cudaEvent_t event : found->events) { recycle(event); }
    recycle(found->header_done);
    if (found->capture.anchor.valid()) { index_->unpin_node(found->capture.anchor); }
    release_image(std::move(found->image));
    captures_.erase(found);
}

void PrefixCache::publish_capture(PendingCapture& c) {
    pc::PublishResult result;
    if (c.image.disk) {
        release_region(c.image.region);
        c.image.region = kNoRegion;
    }
    if (c.image.disk && c.write_failed) {
        // Nothing was recorded: the slots return, and the tail page with the dropped lease.
        index_->release_disk_image(std::move(*c.image.disk));
        c.capture.tail_lease.reset();
    } else {
        std::optional<std::uint32_t> tail_device;
        if (c.capture.tail_lease) { tail_device = adopt_tail(std::move(*c.capture.tail_lease)); }
        result = c.image.disk ? index_->publish_disk_snapshot(c.capture.anchor, c.capture.frontier, c.capture.tail,
                                                              tail_device, std::move(*c.image.disk), c.capture.kind,
                                                              c.capture.meta.head_residual)
                              : index_->publish_host_snapshot(c.capture.anchor, c.capture.frontier, c.capture.tail,
                                                              tail_device, std::move(c.image.host), c.capture.kind,
                                                              c.capture.meta.head_residual);
    }
    if (result.created) {
        snapshot_meta_[result.snapshot.index] = c.capture.meta;
        ++(c.capture.kind == pc::SnapshotKind::Endpoint ? counters_.endpoints_created : counters_.taps_created);
        if (c.image.disk) {
            // Written through: its path follows it to the Disk, and its record follows theirs.
            journal_waits_.push_back(JournalWait{{}, result.snapshot});
            if (c.capture.anchor.valid() &&
                !queue_path_writes(c.capture.anchor, index_->snapshot(result.snapshot).priority)) {
                ++counters_.disk_capture_skips;
            }
        }
    }
    if (c.capture.supersedes && index_->valid(*c.capture.supersedes) &&
        (!result.snapshot.valid() || *c.capture.supersedes != result.snapshot)) {
        index_->supersede(*c.capture.supersedes, c.capture.continuation);
    }
    if (c.capture.anchor.valid()) { index_->unpin_node(c.capture.anchor); }
    published_[c.capture.id] = result;
    for (cudaEvent_t event : c.events) { recycle(event); }
    recycle(c.header_done);
    c.header_done = nullptr;
}

void PrefixCache::poll() {
    if (store_) { reap_disk(); }
    // Captures land in transfer-stream order; one still waiting for its anchor block, or for its
    // Disk writes, keeps its place while the ones behind it publish.
    for (auto it = captures_.begin(); it != captures_.end();) {
        PendingCapture& c = *it;
        if (c.deferred != nullptr) {
            bool dropped = false;
            if (!start_deferred(c, dropped)) {
                if (dropped) {
                    (void)cudaEventSynchronize(c.deferred);
                    recycle(c.deferred);
                    if (c.capture.anchor.valid()) { index_->unpin_node(c.capture.anchor); }
                    it = captures_.erase(it);
                } else {
                    ++it;
                }
                continue;
            }
        }
        const cudaError_t status = cudaEventQuery(c.header_done != nullptr ? c.header_done : c.events.back());
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        c.staging = nullptr; // the copy has read the prompt-end slot
        if (c.image.disk) {
            if (!c.writes_issued) { issue_capture_writes(c); }
            if (!c.writes_issued || c.writes != 0) {
                ++it;
                continue;
            }
        }
        if (c.capture.anchor_pending) {
            ++it;
            continue;
        }
        publish_capture(c);
        it = captures_.erase(it);
    }
    while (!writes_.empty()) {
        PendingWrite& w = writes_.front();
        const cudaError_t status = cudaEventQuery(w.done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        for (const pc::NodeRef node : w.nodes) {
            index_->complete_host_fill(node);
            index_->unpin_node(node);
        }
        recycle(w.done);
        writes_.pop_front();
    }
    while (!landing_.empty()) {
        RestoreBatch& batch       = landing_.front();
        const cudaError_t status = cudaEventQuery(batch.events.back());
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        finish_restore(batch);
        landing_.pop_front();
    }
    if (store_) { poll_disk(); }
}

void PrefixCache::drain() {
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    CUDA_CHECK(cudaStreamSynchronize(restore_stream_));
    poll();
}

bool PrefixCache::await_extended_endpoint(std::span<const pc::NodeRef> path, std::span<const TokenId> tokens) {
    for (const PendingCapture& c : captures_) {
        if (c.capture.kind != pc::SnapshotKind::Endpoint || tokens.size() <= c.capture.frontier) { continue; }
        const std::uint32_t k = c.capture.frontier / pc::kBlockTokens;
        if (k > 0 && (path.size() < k || path[k - 1] != c.capture.anchor)) { continue; }
        if (!std::equal(c.capture.tail.begin(), c.capture.tail.end(),
                        tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens)) {
            continue;
        }
        const std::uint64_t id = c.capture.id;
        CUDA_CHECK(cudaEventSynchronize(c.events.back()));
        poll();
        // Written through to the Disk tier, it publishes once its writes complete.
        while (std::any_of(captures_.begin(), captures_.end(), [&](const PendingCapture& p) { return p.capture.id == id; })) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            poll();
        }
        return true;
    }
    return false;
}

std::optional<pc::PublishResult> PrefixCache::capture_result(std::uint64_t id) const {
    const auto found = published_.find(id);
    if (found == published_.end()) { return std::nullopt; }
    return found->second;
}

void PrefixCache::forget_capture(std::uint64_t id) noexcept { published_.erase(id); }

// ---- restores --------------------------------------------------------------------------------------

void PrefixCache::open_restore(cudaStream_t producer) {
    if (restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: a restore is already open"); }
    restore_          = RestoreBatch{};
    restore_.open     = true;
    restore_.ticket   = RestoreTicket{next_ticket_++};
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(restore_stream_, start, 0));
    recycle(start);
}

void PrefixCache::restore_block(pc::NodeRef node, DeviceKVPageLease&& destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident || view.device != pc::CopyState::Absent) {
        throw std::logic_error("Qwen4Exp prefix cache: restored block is not Host-only");
    }
    const DeviceKVPageHandle handle = destination.handle();
    const std::uint32_t id          = allocate_block_id(std::move(destination));
    index_->begin_device_fill(node, id);
    index_->pin_node(node); // until the batch lands
    restore_.pins.push_back(node);
    restore_.nodes.push_back(node);
    restore_.pages.push_back(PageCopy{handle, slab(view.host_slab), slab_group(view.host_slab)});
    ++counters_.host_block_restores;
}

void PrefixCache::restore_node_page(pc::NodeRef node, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Qwen4Exp prefix cache: copied block has no Host copy"); }
    index_->pin_node(node);
    restore_.pins.push_back(node);
    restore_.pages.push_back(PageCopy{destination, slab(view.host_slab), slab_group(view.host_slab)});
}

void PrefixCache::restore_tail_page(pc::SnapshotRef snapshot, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    const bool fetched          = fetch_ && fetch_->ready && !fetch_->failed && fetch_->snapshot == snapshot && fetch_->tail;
    if (view.tail_len == 0 || (view.host != pc::CopyState::Resident && !fetched)) {
        throw std::logic_error("Qwen4Exp prefix cache: restored tail has no Host copy");
    }
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    if (view.host == pc::CopyState::Resident) {
        const std::uint32_t tail_slab = view.host_slabs.back();
        restore_.pages.push_back(PageCopy{destination, slab(tail_slab), slab_group(tail_slab)});
        ++counters_.host_tail_restores;
        return;
    }
    ++regions_[fetch_->region].users;
    restore_.regions.push_back(fetch_->region);
    restore_.pages.push_back(PageCopy{destination,
                                      region_base(fetch_->region) + static_cast<std::size_t>(host_.image_slabs) * host_.slab_bytes,
                                      kNoRegion - 1U - fetch_->region});
}

void PrefixCache::restore_image(pc::SnapshotRef snapshot, std::uint32_t lane) {
    if (!restore_.open || restore_.image) { throw std::logic_error("Qwen4Exp prefix cache: image restore is not open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    const bool fetched = fetch_ && fetch_->ready && !fetch_->failed && fetch_->snapshot == snapshot &&
                         fetch_->region != kNoRegion;
    if (view.host != pc::CopyState::Resident && !fetched) {
        throw std::logic_error("Qwen4Exp prefix cache: restored image is not on the Host");
    }
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    restore_.image      = snapshot;
    restore_.image_lane = lane;
    if (view.host == pc::CopyState::Resident) {
        ++counters_.host_image_restores;
        return;
    }
    // The fetched image, read from its region until the batch lands.
    restore_.image_region = fetch_->region;
    ++regions_[fetch_->region].users;
    restore_.regions.push_back(fetch_->region);
}

void PrefixCache::copy_pages_from_host(std::span<const PageCopy> copies, std::size_t plane_begin, std::size_t plane_end,
                                       cudaStream_t stream) const {
    if (copies.empty()) { return; }
    std::vector<const std::byte*> records;
    std::vector<std::uint32_t> groups;
    std::vector<DeviceKVPageHandle> pages;
    records.reserve(copies.size());
    groups.reserve(copies.size());
    pages.reserve(copies.size());
    for (const PageCopy& c : copies) {
        records.push_back(c.record);
        groups.push_back(c.group);
        pages.push_back(c.page);
    }
    pool_->copy_from_host_records(records, groups, pages, host_.page, plane_begin, plane_end, stream);
}

RestoreTicket PrefixCache::submit_restore() {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const StateImageLayout& layout = lanes_->layout();
    std::vector<const std::byte*> segments;
    if (restore_.image && restore_.image_region == kNoRegion) {
        const pc::SnapshotView view = index_->snapshot(*restore_.image);
        for (std::uint32_t i = 0; i < host_.image_slabs; ++i) { segments.push_back(slab(view.host_slabs[i])); }
    }
    const HostImageConst image{segments, host_.slab_bytes};
    std::uint32_t attention = 0;
    for (std::uint32_t g = 0; g < layout.groups(); ++g) {
        if (g == layout.drafter_group()) {
            if (geometry_.kv_layers > attention) {
                const PlaneRange planes = kv_layer_planes(geometry_, attention);
                copy_pages_from_host(restore_.pages, planes.begin, planes.end, restore_stream_);
            }
        } else if (layout.spec.layer_types[g] == MixerKind::Attention) {
            const PlaneRange planes = kv_layer_planes(geometry_, attention++);
            copy_pages_from_host(restore_.pages, planes.begin, planes.end, restore_stream_);
        }
        if (restore_.image_region != kNoRegion) {
            lanes_->copy_group_from_packed(g, region_base(restore_.image_region), restore_.image_lane, restore_stream_);
        } else if (restore_.image) {
            lanes_->copy_group_from_host(g, image, restore_.image_lane, restore_stream_);
        }
        cudaEvent_t event = take_event();
        CUDA_CHECK(cudaEventRecord(event, restore_stream_));
        restore_.events.push_back(event);
    }
    counters_.host_restore_bytes += restore_.pages.size() * host_.page.page_stride +
                                    (restore_.image && restore_.image_region == kNoRegion ? host_.image_bytes : 0U);
    restore_.open             = false;
    const RestoreTicket ticket = restore_.ticket;
    landing_.push_back(std::move(restore_));
    restore_ = RestoreBatch{};
    return ticket;
}

std::span<const cudaEvent_t> PrefixCache::restore_events(RestoreTicket ticket) const noexcept {
    for (const RestoreBatch& batch : landing_) {
        if (batch.ticket.id == ticket.id) { return batch.events; }
    }
    return {};
}

void PrefixCache::order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const {
    const auto events = restore_events(ticket);
    if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(consumer, events.back(), 0)); }
}

void PrefixCache::await_restore(RestoreTicket ticket) {
    // One restore stream completes batches in submission order.
    const auto events = restore_events(ticket);
    if (!events.empty()) { CUDA_CHECK(cudaEventSynchronize(events.back())); }
    poll();
}

void PrefixCache::finish_restore(RestoreBatch& batch) noexcept {
    for (const pc::NodeRef node : batch.nodes) { index_->complete_device_fill(node); }
    for (const pc::NodeRef node : batch.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : batch.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    for (const std::uint32_t region : batch.regions) { release_region(region); }
    batch.regions.clear();
    for (cudaEvent_t event : batch.events) { recycle(event); }
    batch.events.clear();
}

void PrefixCache::abort_restore() noexcept {
    if (!restore_.open) { return; }
    // Nothing was enqueued yet: the destinations go back and the pins are dropped.
    for (const pc::NodeRef node : restore_.nodes) { index_->abort_device_fill(node); }
    for (const pc::NodeRef node : restore_.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : restore_.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    for (const std::uint32_t region : restore_.regions) { release_region(region); }
    restore_ = RestoreBatch{};
}

void PrefixCache::clear() noexcept {
    try {
        drain();
        // The Disk writes in flight still read the slabs, pages and staging records dropped below.
        if (store_) { drain_disk(); }
        // A fetch in flight still reads into its pages and region.
        while (fetch_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            poll();
        }
        CUDA_CHECK(cudaStreamSynchronize(transfer_));
        poll();
    } catch (...) {}
    release_fetch();
    for (RestoreBatch& batch : landing_) {
        for (cudaEvent_t event : batch.events) { recycle(event); }
    }
    landing_.clear();
    while (!staged_.empty()) { abandon_staged_capture(staged_.begin()->first); }
    // Landed captures still waiting for their anchor (drain published every other one).
    while (!captures_.empty()) { abandon_capture(captures_.front().capture.id); }
    writes_.clear();
    published_.clear();
    for (auto& [id, w] : block_writes_) { recycle(w.copied); }
    block_writes_.clear();
    spill_.reset();
    journal_waits_.clear();
    unreadable_.clear();
    journaled_snapshots_.clear();
    std::fill(slot_journaled_.begin(), slot_journaled_.end(), false);
    for (Region& region : regions_) { region.users = 0; }
    free_staging_.clear();
    for (std::uint32_t i = kStagingRecords; block_staging_ != nullptr && i-- > 0;) { free_staging_.push_back(i); }
    if (store_) {
        // The emptied index names no record: neither does the journal.
        try {
            store_->compact({}, {});
        } catch (...) { disk_broken_ = true; }
    }
    pc::PrefixIndexConfig config = index_->config();
    index_.reset();
    blocks_.clear();
    free_blocks_.clear();
    std::fill(node_mtp_next_.begin(), node_mtp_next_.end(), -1);
    std::fill(snapshot_meta_.begin(), snapshot_meta_.end(), StateImageHeader{});
    index_.emplace(config, *this);
}

// ---- PrefixIndexBackend ------------------------------------------------------------------------------

void PrefixCache::release_device_block(std::uint32_t device_id) noexcept {
    if (device_id >= blocks_.size() || !blocks_[device_id]) { std::terminate(); }
    // A published tail page can still be read by the tap that captured it.
    try {
        order_page_reuse(std::span<const DeviceKVPageLease>(&*blocks_[device_id], 1));
    } catch (...) { std::terminate(); } // the page would be reused under a copy in flight
    blocks_[device_id].reset(); // the lease returns its page to the pool
    free_blocks_.push_back(device_id);
}

void PrefixCache::release_snapshot(pc::SnapshotRef snapshot) noexcept {
    if (snapshot.index < snapshot_meta_.size()) { snapshot_meta_[snapshot.index] = StateImageHeader{}; }
}

void PrefixCache::drop_snapshot_device_image(pc::SnapshotRef) noexcept {
    // Snapshots are Host-born; there are no Device images to drop.
}

} // namespace infernix::models::qwen4_exp::prefix
