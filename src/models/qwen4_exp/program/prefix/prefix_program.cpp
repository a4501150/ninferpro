// The Program side of Qwen4Exp's prefix cache (docs/maintainer/qwen3_8-flash-next-design.md §19.3.1):
// admission against the index, activation with per-layer restores, block publication, Host-born taps
// and endpoints, and release. The binding (prefix_cache.h) moves the bytes.

#include "models/qwen4_exp/program/program_impl.h"
#include "models/qwen3_5/program/prefix/block_keys.h"
#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace infernix::models::qwen4_exp::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::uint32_t kBlock = pc::kBlockTokens;
constexpr std::size_t kFlushWords = 10; // an MR3 flush call's staging words
// Blocks one prefetch batch copies at most (one page each; 256 pages are ~240 MB, ~9 ms of H2D).
constexpr std::uint32_t kPrefetchBatchBlocks = 256;

} // namespace

// ---- construction ------------------------------------------------------------------------------------

void ProgramImpl::create_prefix_cache() {
    if (!options_.prefix_cache) { return; }
    const auto lanes = static_cast<std::uint32_t>(options_.max_concurrency);
    // A DFlash2 drafter's context ring: per layer, one lane's K and V (the lane is outermost).
    prefix::DraftRingGeometry ring;
    std::vector<std::byte*> ring_k, ring_v;
    if (dflash2_) {
        ring.layers = dflash_ring_->layer_count();
        for (std::uint32_t l = 0; l < ring.layers; ++l) {
            const CyclicKVCacheLayerView view = dflash_ring_->layer_view(l);
            ring.k_bytes                      = view.k.bytes() / lanes;
            ring.v_bytes                      = view.v.bytes() / lanes;
            ring_k.push_back(static_cast<std::byte*>(view.k.data));
            ring_v.push_back(static_cast<std::byte*>(view.v.data));
        }
    }
    image_layout_    = prefix::plan_state_image(prefix::state_image_spec(c_, mtp_, options_.kv_cache, ring));
    lane_image_      = std::make_unique<prefix::LaneStateImage>(
        image_layout_,
        prefix::LaneStateBuffers{.gdn       = gdn_.get(),
                                 .lanes     = lanes,
                                 .ple       = static_cast<std::byte*>(ple_backing_.p),
                                 .tails     = static_cast<std::byte*>(tails_backing_.p),
                                 .mtp_saved = mtp_ ? static_cast<std::byte*>(mtp_saved_.p) : nullptr,
                                 .head_saved = mtp_ ? nullptr : static_cast<std::byte*>(head_saved_.p),
                                 .window    = static_cast<std::byte*>(window_backing_.p),
                                 .window_geometry = window_geometry_,
                                 .ring_k    = std::move(ring_k),
                                 .ring_v    = std::move(ring_v)});
    prefix::PrefixCacheConfig config;
    config.index.cost = options_.prefix_cost;
    config.taps       = options_.prefix_taps;
    config.kv_pages   = kv_pages_;
    prefix_           = std::make_unique<prefix::PrefixCache>(config, *pool_, kv_geometry_, *lane_image_,
                                                            options_.prefix_host_bytes, options_.prefix_disk_bytes);
    prefix_->set_compute_stream(device_.stream);
    if (options_.free_cache_resume) {
        // The zero-token resume's prompt-end captures stage their image through fixed Device slots
        // (a lane's replayed decode graphs cannot wait on capture events): two cover a burst of
        // concurrent prompt ends, a third skips and counts. Their bytes come out of the expert
        // frames, which are sized after this from the memory the fixed allocations leave.
        const std::uint32_t slots = std::min(lanes, 2U);
        std::vector<std::byte*> bytes;
        prompt_end_staging_.reserve(slots);
        bytes.reserve(slots);
        for (std::uint32_t s = 0; s < slots; ++s) {
            prompt_end_staging_.emplace_back(image_layout_.image_bytes);
            bytes.push_back(static_cast<std::byte*>(prompt_end_staging_.back().p));
        }
        prefix_->set_prompt_end_staging(std::move(bytes));
    }
    if (mtp_) {
        // The MR3 flush cell's inputs: token, cell, slot and table row.
        // id, cell, slot, row, then the cell's RoPE position and its pooled-block start (3 each).
        flush_host_   = PinnedHostBuffer(kFlushWords * sizeof(std::int32_t));
        flush_device_ = DeviceBuffer(kFlushWords * sizeof(std::int32_t));
        CUDA_CHECK(cudaEventCreateWithFlags(&flush_uploaded_, cudaEventDisableTiming));
    }
}

PrefixCacheDiskOpen ProgramImpl::attach_prefix_cache_disk(const std::filesystem::path& directory,
                                                          std::string_view fingerprint) {
    if (!prefix_ || prefix_->disk_slots() == 0) { throw std::logic_error("Qwen4Exp: no prefix cache Disk tier was sized"); }
    const prefix::DiskAttachResult opened = prefix_->attach_disk(directory, fingerprint);
    return PrefixCacheDiskOpen{.mode              = cufile_mode_name(opened.mode),
                               .reset_reason      = opened.reset_reason,
                               .damaged           = opened.damaged,
                               .blocks            = opened.blocks,
                               .snapshots         = opened.snapshots,
                               .dropped_blocks    = opened.dropped_blocks,
                               .dropped_snapshots = opened.dropped_snapshots,
                               .bytes             = opened.bytes,
                               .staging_bytes     = prefix_->disk_staging_bytes(),
                               .seconds           = opened.seconds};
}

// After every lane is released: captures and writes land, then the Host tier's snapshots without a
// Disk copy are spilled, the most valuable first.
void ProgramImpl::spill_prefix_cache_for_shutdown() noexcept {
    if (!prefix_ || !prefix_->disk_attached()) { return; }
    try {
        device_.synchronize();
        prefix_->drain();
        // The product may abandon the spill (a Ctrl+C during the stop). It counts as running only
        // from here, so an exit never waits for the Device work above.
        const PrefixCacheSaveControl& control = options_.prefix_save;
        if (!control.begin()) {
            prefix_shutdown_spill_ = PrefixCacheDiskSpill{.message = "abandoned before it began"};
            return;
        }
        struct Ended {
            const PrefixCacheSaveControl& control;
            bool saved = false;
            ~Ended() { control.end(saved); }
        } ended{control};
        const prefix::DiskSpillResult spilled =
            prefix_->spill_for_shutdown(CancellationView([&control] { return control.abandoned(); }));
        prefix_shutdown_spill_ = PrefixCacheDiskSpill{.ok        = spilled.ok,
                                                      .message   = spilled.message,
                                                      .blocks    = spilled.blocks,
                                                      .snapshots = spilled.snapshots,
                                                      .bytes     = spilled.bytes,
                                                      .seconds   = spilled.seconds};
        ended.saved = spilled.ok;
    } catch (const std::exception& error) {
        prefix_shutdown_spill_ = PrefixCacheDiskSpill{.message = error.what()};
    } catch (...) { prefix_shutdown_spill_ = PrefixCacheDiskSpill{.message = "unknown error"}; }
}

void ProgramImpl::pc_make_room(std::uint32_t pages) {
    pc::PrefixCacheIndex& cache = prefix_->index();
    while (pool_->available_pages() < pages) {
        if (cache.evict_device_blocks(pages - pool_->available_pages()) == 0) { return; }
    }
}

// MR3: one kv-only MTP cell for the pending position (cell state_tokens - 1 from the saved residual
// and token history[state_tokens]), the cell the next round's step 0 or catch-up would write.
void ProgramImpl::mtp_flush_cell(Lane& lane, std::uint32_t index) {
    const cudaStream_t stream = device_.stream;
    CUDA_CHECK(cudaEventSynchronize(flush_uploaded_)); // the previous flush has read the staging words
    auto* words      = static_cast<std::int32_t*>(flush_host_.data());
    const auto cell  = lane.state_tokens - 1U;
    words[0]         = lane.history[lane.state_tokens];
    words[1]         = static_cast<std::int32_t>(cell);
    words[2]         = static_cast<std::int32_t>(index);
    words[3]         = static_cast<std::int32_t>(index);
    stage_rope_value(rope_of(lane.rope, cell), std::span<std::int32_t>(words + 4, 3), 1, 0);
    stage_rope_value(block_start_rope(lane.rope, cell, static_cast<std::uint32_t>(r_)), std::span<std::int32_t>(words + 7, 3),
                     1, 0);
    CUDA_CHECK(cudaMemcpyAsync(flush_device_.p, words, kFlushWords * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(flush_uploaded_, stream));
    auto* device = static_cast<std::int32_t*>(flush_device_.p);
    execution::MtpCall call{.residuals        = saved_column(index).view({width_, 1}),
                            .ids              = Tensor(device, DType::I32, {1}),
                            .positions        = Tensor(device + 1, DType::I32, {1}),
                            .rope_positions   = Tensor(device + 4, DType::I32, {1, 3}),
                            .block_start_rope = Tensor(device + 7, DType::I32, {1, 3}),
                            .slots            = Tensor(device + 2, DType::I32, {1}),
                            .table_rows       = Tensor(device + 3, DType::I32, {1}),
                            .batch            = 1,
                            .width            = 1,
                            .kv_only          = true};
    forward_->run_mtp(call);
    lane.mtp_cells = lane.state_tokens;
}

// ---- admission ---------------------------------------------------------------------------------------

// Chooses the resume source for `prompt` in lane `lane`: the deepest affordable snapshot, else the
// root, or nothing when even the root does not fit now (design §19.3.1, plan §7.1).
std::optional<PrefixSelection> ProgramImpl::prefix_select(const qwen3_5::PreparedPromptData& prompt,
                                                                       const BasePlanImpl& base,
                                                                       std::uint32_t lane) {
    pc::PrefixCacheIndex& index = prefix_->index();
    const auto n                = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::uint32_t E       = binding_pages(base, n);
    const auto fits = [&](PrefixSelection& s) {
        // Entries this admission pins stop being evictable: its unpinned Device path blocks and a
        // Device copy-on-write source.
        std::uint32_t evictable_on_path = 0, host_only = 0;
        const std::uint32_t k = s.frontier / kBlock;
        for (std::uint32_t b = 0; b < k; ++b) {
            const pc::NodeView view = index.node(s.path[b]);
            if (view.pins == 0 && view.device == pc::CopyState::Resident) { ++evictable_on_path; }
            if (b < s.shared && view.device != pc::CopyState::Resident) { ++host_only; }
        }
        if (s.cow == PrefixCow::Tail) {
            const pc::SnapshotView view = index.snapshot(*s.snapshot);
            if (view.tail_device_copy == pc::CopyState::Resident && view.pins == 0) { ++evictable_on_path; }
        }
        s.need                     = (E - s.shared) + host_only;
        const std::uint64_t room   = static_cast<std::uint64_t>(pool_->available_pages()) + index.device_evictable_blocks() +
                                   kv_grow_pages();
        return static_cast<std::uint64_t>(s.need) + evictable_on_path <= room;
    };
    std::uint32_t cached_tokens = 0; // the prompt prefix held as cached blocks
    const auto planned = [&](PrefixSelection s) {
        // Where the prompt leaves a cached path well past its resume frontier (a client re-sending a
        // long transcript with new content inserted before its end, Strata PR #2843): a boundary
        // snapshot there lets the next such prompt resume at the branch instead of re-prefilling
        // the shared tokens after the frontier again.
        const std::uint32_t branch =
            cached_tokens < n && cached_tokens >= s.frontier + kBranchTapMinTokens ? cached_tokens : 0;
        s.plan          = plan_prefill(prompt, s.frontier, s.existing, base.reuse, branch);
        s.cached_tokens = cached_tokens;
        return s;
    };
    PrefixSelection root;
    root.need = E;
    if (base.reuse && n > 1) {
        prefix_->poll();
        pc::MatchResult match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        if (prefix_->await_extended_endpoint(match.path, prompt.token_ids)) {
            // The endpoint this prompt continues has just been published.
            match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        }
        if (prefetch_landing() && std::any_of(match.candidates.begin(), match.candidates.end(),
                                              [](const pc::MatchCandidate& c) { return c.filling_blocks != 0; })) {
            // A prefetch is copying blocks this prompt resumes over: they land within a batch's copy
            // time, and skipping them would pick a shallower source.
            prefix_->await_restore(*prefetch_ticket_);
            match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        }
        cached_tokens = static_cast<std::uint32_t>(match.path.size()) * kBlock;
        // A frontier strictly inside a Vision item would resume half an image (design §19.3.1).
        const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
        // Without free cache resume a request resumes only from prefill-computed state on the
        // 16-token grid: a tap, never an endpoint (its state is decode's).
        const bool grid = !options_.free_cache_resume;
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return c.filling_blocks != 0 || prefix::inside_exclusion(c.frontier, spans) ||
                   (grid && (c.frontier % kResumeGrid != 0 ||
                             index.snapshot(c.snapshot).kind == pc::SnapshotKind::Endpoint));
        });
        // A lane-resident snapshot needs no image restore: its state is still in the lane.
        const std::optional<pc::SnapshotRef> resident =
            lane < lanes_.size() ? prefix_resident(lanes_[lane]) : std::nullopt;
        prefix_credit(match.candidates, resident);
        const pc::AdmissionChoice choice = index.choose(match, n);
        std::vector<std::uint32_t> frontiers;
        for (const pc::MatchCandidate& c : match.candidates) { frontiers.push_back(c.frontier); }
        std::vector<std::size_t> order;
        if (choice.candidate) { order.push_back(*choice.candidate); }
        for (std::size_t i = 0; i < match.candidates.size(); ++i) {
            if (!choice.candidate || i != *choice.candidate) { order.push_back(i); }
        }
        const auto select = [&]() -> std::optional<PrefixSelection> {
        for (const std::size_t i : order) {
            const pc::MatchCandidate& c = match.candidates[i];
            PrefixSelection s;
            s.snapshot = c.snapshot;
            s.frontier = c.frontier;
            s.path.assign(match.path.begin(), match.path.begin() + c.path_blocks);
            const std::uint32_t k = c.frontier / kBlock;
            s.shared              = k;
            if (c.frontier % kBlock != 0) {
                s.cow = PrefixCow::Tail;
            } else if (mtp_ && k > 0 &&
                       !(c.frontier < n && prefix_->mtp_next(s.path[k - 1]) == prompt.token_ids[c.frontier])) {
                // MR6: the anchor's last MTP cell (F - 1) must encode history[F]; otherwise the lane
                // rewrites it in a private copy of the anchor page. At the prompt's own end
                // (F == n, the zero-token resume) history[F] is the token the resume regenerates,
                // unknown here, so the anchor is always copied: its cell holds the capturing run's
                // first token, and the lane's first decode round writes its own.
                s.cow    = PrefixCow::Anchor;
                s.shared = k - 1;
            }
            s.image_restore = !(resident && *resident == c.snapshot);
            s.existing      = frontiers;
            if (c.frontier == n) {
                // The zero-token resume: the snapshot at the prompt's own end carries the whole
                // state and its head's saved residual, so the prefill plan is empty (plan_calls
                // refuses an empty one by design) and the first token regenerates from the residual
                // at the prefill step (resume_from_saved_head). Match offers the frontier only
                // with its residual.
                if (c.head_residual && fits(s)) {
                    s.cached_tokens = cached_tokens;
                    return s;
                }
                continue;
            }
            if (fits(s)) { return planned(std::move(s)); }
        }
        return std::nullopt;
        };
        if (std::optional<PrefixSelection> s = select()) { return s; }
        if (!match.candidates.empty() && prefix_->transfers_pending()) {
            // Blocks pinned only by in-flight Host writes become evictable once those land: a cached
            // source is not given up for the root while a finished lane's writes land.
            prefix_->drain();
            if (std::optional<PrefixSelection> s = select()) { return s; }
        }
    }
    if (fits(root)) { return planned(std::move(root)); }
    if (prefix_->transfers_pending()) {
        // Blocks pinned only by in-flight copies become evictable once those land.
        prefix_->drain();
        if (fits(root)) { return planned(std::move(root)); }
    }
    return std::nullopt;
}

// The blocked FIFO head's Host-only blocks are copied into spare Device cache while it waits
// (hybrid-prefix-cache-spec §6.6): its matched path is pinned while room is made from free pages and
// Host-backed cached blocks, least recently used first, so a prefetch never waits for a transfer,
// never grows the pool into the expert frames, and never drops a block's last copy.
std::optional<std::uint32_t> ProgramImpl::hybrid_prefetch(const RequestBasePlan& base) {
    if (!prefix_ || !prefix_->host_tier() || base.impl_ == nullptr || !base.impl_->reuse || !base.impl_->prompt) {
        return 0U;
    }
    if (transaction_lane_ || prefix_->restore_open()) { return std::nullopt; }
    prefix_->poll();
    if (prefetch_landing()) { return std::nullopt; }
    prefetch_ticket_.reset();
    const auto& prompt = qwen3_5::PreparedPromptAccess::view(*base.impl_->prompt);
    const auto n       = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n <= 1 || prompt.block_hashes.size() != n / kBlock) { return 0U; }
    // The source the admission would choose now, as in prefix_select.
    pc::PrefixCacheIndex& index = prefix_->index();
    pc::MatchResult match       = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
    std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
        return c.filling_blocks != 0 || prefix::inside_exclusion(c.frontier, spans);
    });
    const pc::AdmissionChoice choice = index.choose(match, n);
    if (!choice.candidate || match.candidates[*choice.candidate].host_only_blocks == 0) { return 0U; }
    const std::span<const pc::NodeRef> path(match.path.data(), match.candidates[*choice.candidate].path_blocks);
    index.acquire_path(path);
    std::uint32_t started = 0;
    try {
        std::uint32_t wanted = std::min(match.candidates[*choice.candidate].host_only_blocks, kPrefetchBatchBlocks);
        while (pool_->available_pages() < wanted && index.evict_backed_device_blocks(1) != 0) {}
        wanted = std::min(wanted, pool_->available_pages());
        std::optional<DeviceKVPageReservation> reservation;
        if (wanted != 0) { reservation = pool_->reserve(wanted); }
        if (reservation) {
            std::vector<DeviceKVPageLease> pages;
            pages.reserve(wanted);
            pool_->materialize(*reservation, wanted, pages);
            prefix_->open_restore(device_.stream);
            for (const pc::NodeRef node : path) {
                if (started == wanted) { break; }
                const pc::NodeView view = index.node(node);
                if (view.device != pc::CopyState::Absent || view.host != pc::CopyState::Resident) { continue; }
                prefix_->restore_block(node, std::move(pages[started]));
                ++started;
            }
            if (started != 0) {
                prefetch_ticket_ = prefix_->submit_restore();
                prefix_->counters().prefetched_blocks += started;
            } else {
                prefix_->abort_restore();
            }
        }
    } catch (...) {
        prefix_->abort_restore();
        index.release_path(path);
        throw;
    }
    index.release_path(path);
    return started;
}

// Credits what needs no transfer, then leaves only sources restorable now: a lane-resident
// snapshot's image is still in the lane, a landed fetch holds its source's Disk bytes in VRAM, and
// any other Disk byte must be fetched first (prefix_disk_fetch).
void ProgramImpl::prefix_credit(std::vector<pc::MatchCandidate>& candidates,
                                std::optional<pc::SnapshotRef> resident) const {
    for (pc::MatchCandidate& c : candidates) {
        c.disk_bytes -= std::min(c.disk_bytes, prefix_->fetched_bytes(c));
        if (resident && *resident == c.snapshot) {
            std::uint64_t& bytes = c.image_on_disk_only ? c.disk_bytes : c.restore_bytes;
            bytes -= std::min<std::uint64_t>(bytes, image_layout_.image_bytes);
        }
    }
    std::erase_if(candidates, [](const pc::MatchCandidate& c) { return c.disk_bytes != 0; });
}

// The Disk tier (design §19.3.1): the source a fresh request would choose with every candidate
// restorable is read before its admission when it lies (partly) only on the Disk, and a snapshot
// of the prompt still being written through to the Disk is awaited. While other requests run, the
// request is deferred (the Engine asks again), so their rounds never wait on the Disk; an idle
// Engine has nothing else to run and waits here, at most kDiskWaitSeconds.
bool ProgramImpl::prefix_disk_fetch(const qwen3_5::PreparedPromptData& prompt) {
    if (!prefix_->disk_attached()) { return false; }
    constexpr double kDiskWaitSeconds = 5.0;
    const auto start                  = std::chrono::steady_clock::now();
    const bool idle = std::all_of(lanes_.begin(), lanes_.end(), [](const Lane& lane) { return lane.phase == Phase::Free; });
    while (prefix_disk_pending(prompt)) {
        if (!idle) { return true; }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > kDiskWaitSeconds) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}

// Whether the prompt's admission must wait for the Disk tier: a fetch is running (the head's choice
// may need it), a snapshot on the prompt's path is still being written through, or a fetch of the
// source it would choose has just started.
bool ProgramImpl::prefix_disk_pending(const qwen3_5::PreparedPromptData& prompt) {
    prefix_->poll();
    if (prefix_->fetch_running()) { return true; }
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n <= 1 || prompt.block_hashes.size() != n / kBlock) { return false; }
    pc::PrefixCacheIndex& index = prefix_->index();
    pc::MatchResult match       = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    if (prefix_->disk_capture_pending(match.path, prompt.token_ids)) { return true; }
    const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
    const bool grid                           = !options_.free_cache_resume;
    std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
        return c.filling_blocks != 0 || prefix::inside_exclusion(c.frontier, spans) ||
               prefix_->disk_unreadable(c.snapshot) ||
               (grid && (c.frontier % kResumeGrid != 0 || index.snapshot(c.snapshot).kind == pc::SnapshotKind::Endpoint)) ||
               (c.frontier == n && !c.head_residual);
    });
    for (pc::MatchCandidate& c : match.candidates) { c.disk_bytes -= std::min(c.disk_bytes, prefix_->fetched_bytes(c)); }
    const pc::AdmissionChoice choice = index.choose(match, n);
    if (!choice.candidate) { return false; }
    const pc::MatchCandidate& c = match.candidates[*choice.candidate];
    if (c.disk_bytes == 0) { return false; }
    const std::span<const pc::NodeRef> path(match.path.data(), c.path_blocks);
    return prefix_->start_fetch(c.snapshot, path);
}

std::uint32_t ProgramImpl::hybrid_prefetch_room() const noexcept {
    if (!prefix_ || !prefix_->host_tier()) { return 0U; }
    return pool_->available_pages() + prefix_->index().device_backed_evictable_blocks();
}

// The snapshot each queued request would resume from now, as prefix_select would choose it, is
// held in admission order (hybrid-prefix-cache-spec §9.6). A block landing for another admission
// does not change what the request needs, so filling candidates stay eligible.
void ProgramImpl::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) {
    if (!prefix_) { return; }
    pc::PrefixCacheIndex& index = prefix_->index();
    std::vector<pc::SnapshotRef> holds;
    for (const RequestBasePlan* base : queue) {
        if (holds.size() == pc::kMaxQueueHolds) { break; }
        if (base == nullptr || base->impl_ == nullptr || !base->impl_->reuse || !base->impl_->prompt) { continue; }
        const auto& prompt = qwen3_5::PreparedPromptAccess::view(*base->impl_->prompt);
        const auto n       = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (n <= 1 || prompt.block_hashes.size() != n / kBlock) { continue; }
        pc::MatchResult match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
        std::erase_if(match.candidates,
                      [&](const pc::MatchCandidate& c) { return prefix::inside_exclusion(c.frontier, spans); });
        const pc::AdmissionChoice choice = index.choose(match, n);
        if (choice.candidate) { holds.push_back(match.candidates[*choice.candidate].snapshot); }
    }
    index.set_queue_holds(holds);
}

std::uint64_t ProgramImpl::hybrid_cache_epoch() const noexcept {
    return prefix_ ? prefix_->index().snapshot_epoch() : 0U;
}

// A fresh request whose prompt shares a prefix with a lane still prefilling waits for that lane's
// snapshot at the divergence instead of prefilling the shared part again (the coalescing of the
// Qwen3.5 hybrid cache, hybrid-prefix-cache-spec): a /v1/decide fan-out's questions over one state,
// or concurrent requests over one image, prefill the shared part once.
bool ProgramImpl::prefix_await_sibling(const qwen3_5::PreparedPromptData& prompt, std::uint32_t reuse) {
    // Coalescing waits on another request and inserts its tap into this one's prefill.
    if (!options_.free_cache_resume) { return false; }
    if (options_.prefix_coalesce_wait_seconds <= 0.0) { return false; }
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n < 2) { return false; }
    const pc::CacheCostModel& cost = options_.prefix_cost;
    struct Wait {
        std::uint32_t lane   = 0;
        std::uint32_t target = 0;
        bool plan_tap        = false;
    };
    std::optional<Wait> best;
    const auto consider = [&](const Wait& wait) {
        if (!best || wait.target > best->target) { best = wait; }
    };
    for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
        const Lane& lane = lanes_[i];
        // A replaying lane plans no taps (its blocks are already cached up to its frontier).
        if (lane.phase == Phase::Free || !lane.prefix.reuse || !lane.prompt || lane.replay) { continue; }
        const auto& sibling = qwen3_5::PreparedPromptAccess::view(*lane.prompt);
        // The waiting request keeps at least one prompt token to prefill.
        const std::size_t limit = std::min<std::size_t>(n - 1U, lane.prompt_tokens);
        const auto equal        = static_cast<std::uint32_t>(
            std::mismatch(prompt.token_ids.begin(), prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(limit),
                          lane.history.begin())
                .first -
            prompt.token_ids.begin());
        // A sibling encoding the same image saves this request the encode as well as the prefill.
        const std::uint32_t shared = qwen3_5::detail::media_agreed_prefix(prompt, sibling, equal);
        if (shared <= reuse) { continue; }
        // A capture of the lane's inside the shared prefix that has not landed publishes within a
        // copy time.
        if (lane.prefix.capture != 0 && !prefix_->capture_result(lane.prefix.capture) &&
            lane.prefix.deepest > reuse && lane.prefix.deepest <= shared) {
            consider(Wait{.lane = i, .target = lane.prefix.deepest, .plan_tap = false});
        }
        if (lane.phase != Phase::Prefill) { continue; }
        // Taps after the lane's frontier can still be realized, except inside a running layer walk's
        // span: its calls are enqueued, and its span end is realized when the span completes.
        const bool walking            = walk_.active && walk_.lane == i;
        const std::uint32_t walk_end  = walking ? lane.calls[walk_.end_call - 1] : 0U;
        const auto realizable = [&](std::uint32_t position) {
            return position > lane.state_tokens && (!walking || position >= walk_end);
        };
        // A new tap goes on the block boundary below the divergence (its block publishes with it);
        // inside an image both prompts carry it moves to the image's end, so the waiting request
        // resumes past the whole image and encodes nothing for it.
        std::uint32_t aligned = shared / kBlock * kBlock;
        for (const pc::TapExclusion& span : lane.prefix.exclusions) {
            if (span.begin < aligned && aligned < span.end && span.end <= shared) { aligned = span.end; }
        }
        std::optional<Wait> wait;
        if (realizable(aligned) && aligned > reuse && aligned < lane.prompt_tokens) {
            wait = Wait{.lane = i, .target = aligned, .plan_tap = true};
        }
        // An exact tap the lane already plans near the divergence serves instead, unless prefilling
        // the tokens between it and the new tap costs more than the split.
        for (const pc::PlannedTap& tap : lane.prefix.taps) {
            if (tap.position > shared) { break; }
            if (tap.placement != pc::TapPlacement::Exact || !realizable(tap.position) || tap.position <= reuse) { continue; }
            if (tap.position >= aligned || cost.prefill_seconds(tap.position, aligned - tap.position) <= cost.chunk_seconds) {
                wait = Wait{.lane = i, .target = tap.position, .plan_tap = false};
            }
        }
        if (!wait || prefix::inside_exclusion(wait->target, lane.prefix.exclusions) ||
            wait->target < lane.prefix.deepest + kBlock) {
            continue;
        }
        // Waiting saves this request's prefill of the shared tokens; a new tap costs the sibling what
        // its cut adds: a call unless the tap is a call end already, and a span restart when the cut
        // leaves a call too narrow to walk (a tap between walk-eligible calls is captured inside the
        // span; nothing at a running walk's span end). The sibling's remaining prefill to the snapshot
        // is the wait, doubled because other lanes' decode rounds interleave with it.
        const double saved     = cost.prefill_seconds(reuse, wait->target - reuse);
        const double predicted = 2.0 * cost.prefill_seconds(lane.state_tokens, wait->target - lane.state_tokens);
        double split           = 0.0;
        if (wait->plan_tap && !(walking && wait->target == walk_end)) {
            // The pending call ends with the tap inserted; the calls meeting at it are [left, target)
            // and [target, right).
            std::vector<std::uint32_t> ends(lane.calls.begin() + static_cast<std::ptrdiff_t>(lane.next_call), lane.calls.end());
            const auto at = std::lower_bound(ends.begin(), ends.end(), wait->target);
            const bool cut = at == ends.end() || *at != wait->target;
            const auto tap = cut ? ends.insert(at, wait->target) : at;
            const std::uint32_t left  = tap == ends.begin() ? lane.state_tokens : *std::prev(tap);
            const std::uint32_t right = *std::next(tap); // the tap is before the prompt's end
            const auto streamed       = static_cast<std::uint32_t>(kStreamMinColumns);
            if (cut) { split += cost.chunk_seconds; }
            if (cut && right - left >= streamed && (wait->target - left < streamed || right - wait->target < streamed)) {
                split += options_.prefix_span_seconds;
            }
        }
        if (saved > split && predicted <= options_.prefix_coalesce_wait_seconds) {
            consider(*wait);
        }
    }
    if (!best) { return false; }
    // The snapshot the waiting request resumes from is where two conversations diverge: it is
    // published as a boundary, so neither lineage supersedes it.
    Lane& lane = lanes_[best->lane];
    // An in-flight capture (a decoding sibling's, or one at or before a prefilling sibling's state,
    // whose call ended there and is enqueued): nothing to plan. Splitting the remaining calls at such
    // a target would plan a call that ends where the lane already is.
    if (lane.phase != Phase::Prefill || best->target <= lane.state_tokens) { return true; }
    auto& taps = lane.prefix.taps;
    const auto at = std::lower_bound(taps.begin(), taps.end(), best->target,
                                     [](const pc::PlannedTap& tap, std::uint32_t position) { return tap.position < position; });
    if (at != taps.end() && at->position == best->target) {
        at->boundary = true;
        if (at->placement == pc::TapPlacement::Exact) { return true; }
        at->placement = pc::TapPlacement::Exact;
    } else if (best->plan_tap) {
        taps.insert(at, pc::PlannedTap{.position = best->target, .placement = pc::TapPlacement::Exact, .boundary = true});
    }
    // An exact tap ends a call: split the call that spans it (it is not enqueued yet).
    auto& calls = lane.calls;
    const auto end = std::lower_bound(calls.begin() + static_cast<std::ptrdiff_t>(lane.next_call), calls.end(), best->target);
    if (end != calls.end() && *end != best->target) { calls.insert(end, best->target); }
    return true;
}

// Pins what a selection reads (its matched path up to the anchor and the snapshot) before room is
// made for it, so making room cannot evict them.
void ProgramImpl::prefix_pin(const PrefixSelection& s) {
    if (!s.snapshot) { return; }
    const std::span<const pc::NodeRef> path(s.path.data(), s.frontier / kBlock);
    if (!path.empty()) { prefix_->index().acquire_path(path); }
    prefix_->index().pin_snapshot(*s.snapshot);
}

void ProgramImpl::prefix_unpin(const PrefixSelection& s) noexcept {
    if (!s.snapshot) { return; }
    try {
        const std::span<const pc::NodeRef> path(s.path.data(), s.frontier / kBlock);
        if (!path.empty()) { prefix_->index().release_path(path); }
        prefix_->index().unpin_snapshot(*s.snapshot);
    } catch (...) {}
}

// Activates a pinned selection in a lane whose pages are reserved: maps shared pages, stages
// restores and copy-on-write sources, and sets the lane's frontier and MTP state (plan §7.2).
void ProgramImpl::prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& s,
                                  DeviceKVPageReservation& reservation, std::uint32_t pages) {
    pc::PrefixCacheIndex& cache = prefix_->index();
    const cudaStream_t stream   = device_.stream;
    // A copy-out of this lane's previous state must land before anything overwrites the slot.
    for (const cudaEvent_t event : prefix_->capture_events(lane.prefix.capture)) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
    }
    const std::uint32_t k = s.frontier / kBlock;
    // The lane keeps the pins of the blocks it maps (prefix_pin took one on each up to the anchor).
    lane.prefix.path.assign(s.path.begin(), s.path.begin() + s.shared);
    lane.pages.clear();
    lane.pages.reserve(pages - s.shared);
    pool_->materialize(reservation, pages - s.shared, lane.pages);
    lane.prefix.page_base = s.shared;
    const bool restores   = s.snapshot && (s.image_restore || s.cow != PrefixCow::None ||
                                         std::any_of(lane.prefix.path.begin(), lane.prefix.path.end(), [&](pc::NodeRef n) {
                                             return cache.node(n).device != pc::CopyState::Resident;
                                         }));
    if (s.snapshot) {
        if (restores) { prefix_->open_restore(stream); }
        for (const pc::NodeRef node : lane.prefix.path) {
            if (cache.node(node).device != pc::CopyState::Resident) {
                prefix_->restore_block(node, pool_->materialize_one(reservation));
            }
        }
        const DeviceKVPageHandle first_private = lane.pages.empty() ? DeviceKVPageHandle{} : lane.pages.front().handle();
        if (s.cow == PrefixCow::Tail) {
            const pc::SnapshotView view = cache.snapshot(*s.snapshot);
            if (view.tail_device_copy == pc::CopyState::Resident) {
                pool_->copy_page(prefix_->page(view.tail_device), first_private, stream);
            } else {
                prefix_->restore_tail_page(*s.snapshot, first_private);
            }
        } else if (s.cow == PrefixCow::Anchor) {
            const pc::NodeRef anchor = s.path[k - 1];
            const pc::NodeView view  = cache.node(anchor);
            if (view.device == pc::CopyState::Resident) {
                pool_->copy_page(prefix_->page(view.device_id), first_private, stream);
            } else {
                prefix_->restore_node_page(anchor, first_private);
            }
        }
        if (s.image_restore) { prefix_->restore_image(*s.snapshot, index); }
        if (restores) { lane.prefix.restore = prefix_->submit_restore(); }
        if (s.cow == PrefixCow::Anchor) {
            // The anchor's copy is stream-ordered (or pinned by the restore batch): drop its pin.
            const std::span<const pc::NodeRef> to_anchor(s.path.data(), k);
            cache.release_path(to_anchor);
            if (k > 1) { cache.acquire_path(to_anchor.first(k - 1)); }
        }
        cache.note_hit(*s.snapshot);
        cache.unpin_snapshot(*s.snapshot); // the restore batch holds its own pin while it lands
        // The fetch this admission waited for is consumed: the batch reads its region until it lands.
        if (prefix_->fetch_holds(*s.snapshot)) { prefix_->release_fetch(); }
        lane.prefix.resume  = s.snapshot;
        lane.prefix.deepest = s.frontier;
        ++prefix_->counters().admissions;
        prefix_->counters().reused_tokens += s.frontier;
    }
    // The slot no longer holds the endpoint image once anything is admitted into it.
    prefix_->forget_capture(lane.prefix.resident_capture);
    lane.prefix.resident_capture = 0;
    lane.prefix.captured         = false;
    if (!s.snapshot) { reset_slot(index); }
    // The block table: shared cache pages, then the private leases.
    std::vector<DeviceKVPageHandle> handles;
    handles.reserve(pages);
    for (const pc::NodeRef node : lane.prefix.path) { handles.push_back(prefix_->page(cache.node(node).device_id)); }
    for (const DeviceKVPageLease& lease : lane.pages) { handles.push_back(lease.handle()); }
    tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageHandle>(handles), stream);
    lane.state_tokens = s.frontier;
    if (s.snapshot) {
        // MR6: whether the drafter's cell F - 1 holds this prompt's continuation.
        const prefix::StateImageHeader& meta = prefix_->meta(*s.snapshot);
        // The opener of an echo lineage serves only when the endpoint after it did not: the client
        // did not echo the generated turn exactly (design §19.3.1, "Generation opener").
        if (meta.opener && meta.lineage_echo) { ++prefix_->counters().endpoint_mismatch_fallbacks; }
        const bool continues = static_cast<std::size_t>(s.frontier) < lane.history.size() &&
                               meta.mtp_next == lane.history[s.frontier];
        if (s.cow == PrefixCow::Tail) {
            lane.mtp_cells = meta.mtp_written && continues ? s.frontier : s.frontier - 1U;
        } else if (s.cow == PrefixCow::Anchor) {
            lane.mtp_cells = s.frontier - 1U;
        } else {
            lane.mtp_cells = s.frontier;
        }
        lane.mtp_live = mtp_;
        // The zero-token resume's lane flag, read at the prefill step (the snapshot itself may be
        // evicted once its restore has landed; the lane keeps the state it restored).
        lane.prefix.head_resume = meta.head_residual && s.frontier == lane.prompt_tokens;
        // A zero-token resume's pending cell is unwritten by construction (the capture precedes the
        // first decode round), not a continuation that differs.
        if (mtp_ && lane.mtp_cells != s.frontier && !lane.prefix.head_resume) {
            ++prefix_->counters().mtp_continuation_mismatches;
        }
        lane.dflash_frontier = std::min(meta.draft_frontier, s.frontier);
        // Penalty counts restart with the request; the lane's state came from the snapshot.
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + index * static_cast<std::size_t>(token_domain_),
                                   0, 4ULL * token_domain_, stream));
    }
    // Taps for this prompt beyond the resume frontier, realized at the plan's call boundaries.
    lane.prefix.taps = s.plan.taps;
}

// The prompt's prefill calls from `frontier` (design §19.3.1, "Prefill integration"): with `taps`, the
// prefix taps beyond the frontier (no tap inside a Vision item), exact ones admitted by their cost.
prefix::CallPlan ProgramImpl::plan_prefill(const qwen3_5::PreparedPromptData& prompt, std::uint32_t frontier,
                                           std::span<const std::uint32_t> existing, bool taps,
                                           std::uint32_t branch) const {
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    std::vector<pc::PlannedTap> planned;
    if (taps && prefix_) {
        const std::vector<pc::TapExclusion> spans = prefix_exclusions(prompt);
        planned = pc::plan_taps(n, frontier, prompt.tap_hints.hints, existing, spans, prefix_->taps());
        if (branch > frontier && branch < n && !prefix::inside_exclusion(branch, spans)) {
            const auto at = std::lower_bound(planned.begin(), planned.end(), branch,
                                             [](const pc::PlannedTap& tap, std::uint32_t p) { return tap.position < p; });
            if (at != planned.end() && at->position == branch) {
                at->placement = pc::TapPlacement::Exact;
                at->boundary  = true;
            } else {
                planned.insert(at, pc::PlannedTap{.position = branch, .placement = pc::TapPlacement::Exact,
                                                  .boundary = true});
            }
        }
        if (!options_.free_cache_resume) {
            // Each tap moves to the grid position at or before it, so it cuts a call on the grid.
            for (pc::PlannedTap& tap : planned) { tap.position = tap.position / kResumeGrid * kResumeGrid; }
            std::erase_if(planned, [&](const pc::PlannedTap& tap) {
                return tap.position <= frontier || prefix::inside_exclusion(tap.position, spans);
            });
            planned.erase(std::unique(planned.begin(), planned.end(),
                                      [](const pc::PlannedTap& a, const pc::PlannedTap& b) {
                                          return a.position == b.position;
                                      }),
                          planned.end());
        }
        // The single-chunk tail tap, in both resume modes. A flexible tap (the protocol's
        // generation opener, demoted by its cost) realizes at the start of its final call, so a
        // span of two or more calls publishes a snapshot at its last inner boundary — at most
        // one chunk short of the end — that repeats, forks and continuations resume from; with
        // free cache resume that is the opener's realization, and a tail tap beside it would
        // only absorb it (the opener would realize at the tail split instead, stranding sharers
        // that diverge before it). A span that fits one call has no inner boundary for the
        // opener to realize at, and the finish endpoint (free cache resume only) covers the
        // whole generated turn, so a continuation takes it but a repeat of the prompt alone
        // cannot (the endpoint sits past the prompt's own length): such a prompt publishes
        // nothing a repeat or fork could resume from and starts from the root. Without free
        // cache resume the opener is grid-floored and demoted, and never realizes at all, so
        // every span needs the tail tap; with it, only a fresh one-call span does (a resumed
        // request's tail was published by its original prefill, and splitting the resumed span
        // would change the arithmetic path its preemption and replay equivalences are checked
        // on). An exact boundary tap at the grid frontier at or before the prompt's last token
        // splits a final call of at most kResumeGrid tokens off the plan: its snapshot is the
        // prompt's own tail, which
        // any later prompt sharing the prefix resumes from at the cost of one short tail call.
        // Taps it covers (they realize at the boundary at or past their position, never deeper)
        // are dropped; an exact boundary tap within one separation below it already cuts there
        // (an explicit client breakpoint).
        const std::uint32_t tail = (n - 1U) / kResumeGrid * kResumeGrid;
        const bool single_call   = n - frontier <= static_cast<std::uint32_t>(chunk_);
        const auto tail_cut = [&](const pc::PlannedTap& tap) {
            return tap.placement == pc::TapPlacement::Exact && tap.boundary &&
                   tap.position + pc::kMinimumTapSeparation > tail;
        };
        if ((!options_.free_cache_resume || (single_call && frontier == 0U)) && tail > frontier &&
            tail >= frontier + kBlock && !prefix::inside_exclusion(tail, spans) &&
            !std::any_of(planned.begin(), planned.end(), tail_cut) &&
            !std::any_of(existing.begin(), existing.end(),
                         [&](std::uint32_t f) { return f + pc::kMinimumTapSeparation > tail; })) {
            std::erase_if(planned, [&](const pc::PlannedTap& tap) {
                return tap.position + pc::kMinimumTapSeparation > tail &&
                       !(tap.placement == pc::TapPlacement::Exact && tap.boundary);
            });
            planned.push_back(pc::PlannedTap{.position  = tail,
                                             .placement = pc::TapPlacement::Exact,
                                             .boundary  = true});
        }
    }
    return prefix::plan_calls(frontier, n, static_cast<std::uint32_t>(chunk_), planned, call_cost());
}

HybridPrefixCacheStats ProgramImpl::prefix_stats() const noexcept {
    HybridPrefixCacheStats out;
    if (!prefix_) { return out; }
    const pc::PrefixIndexStats index         = prefix_->index().stats();
    const prefix::PrefixCacheCounters& count = prefix_->counters();
    out.nodes                   = index.nodes;
    out.snapshots               = index.snapshots;
    out.device_resident_blocks  = index.device_resident_blocks;
    out.device_evictable_blocks = index.device_evictable_blocks;
    out.host_slabs              = prefix_->host_layout().slabs;
    out.host_free_slabs         = index.host_free_slabs;
    out.host_slab_bytes         = prefix_->host_layout().slab_bytes;
    out.snapshot_hits           = index.snapshot_hits;
    out.reused_tokens           = count.reused_tokens;
    out.blocks_inserted         = count.blocks_inserted;
    out.blocks_reattached       = count.blocks_reattached;
    out.blocks_duplicate        = count.blocks_duplicate;
    out.taps_created            = count.taps_created;
    out.taps_skipped            = count.taps_skipped;
    out.prompt_end_captures     = count.prompt_end_captures;
    out.prompt_end_skipped      = count.prompt_end_skipped;
    out.endpoints_created       = count.endpoints_created;
    out.host_image_writes       = count.host_image_writes;
    out.host_block_writes       = count.host_block_writes;
    out.host_image_restores     = count.host_image_restores;
    out.host_block_restores     = count.host_block_restores;
    out.prefetched_blocks       = count.prefetched_blocks;
    out.host_write_bytes        = count.host_write_bytes;
    out.host_restore_bytes      = count.host_restore_bytes;
    out.evicted_blocks          = index.device_block_evictions;
    out.host_snapshot_evictions = index.host_snapshot_evictions;
    out.host_dead_reclaims      = index.host_dead_reclaims;
    out.unbacked_node_losses    = index.unbacked_node_losses;
    out.held_snapshots          = index.held_snapshots;
    out.held_device_evictions   = index.held_device_evictions;
    out.held_snapshot_losses    = index.held_snapshot_losses;
    out.held_host_refusals      = index.held_host_refusals;
    out.disk_slots              = prefix_->disk_slots();
    out.disk_free_slots         = index.disk_free_slots;
    out.disk_image_writes       = count.disk_image_writes;
    out.disk_block_writes       = count.disk_block_writes;
    out.disk_write_bytes        = count.disk_write_bytes;
    out.disk_fetches            = count.disk_fetches;
    out.disk_image_reads        = count.disk_image_reads;
    out.disk_block_reads        = count.disk_block_reads;
    out.disk_read_bytes         = count.disk_read_bytes;
    out.disk_failures           = count.disk_failures;
    out.disk_capture_skips      = count.disk_capture_skips;
    out.disk_spill_skips        = count.disk_spill_skips;
    out.disk_snapshot_evictions = index.disk_snapshot_evictions;
    out.disk_dead_reclaims      = index.disk_dead_reclaims;
    return out;
}

std::vector<pc::TapExclusion> ProgramImpl::prefix_exclusions(const qwen3_5::PreparedPromptData& prompt) {
    std::vector<pc::TapExclusion> out;
    if (prompt.vision_items.empty()) { return out; }
    for (const auto& range : qwen3_5::detail::vision_ranges(prompt)) { out.push_back({range.begin, range.end}); }
    return out;
}

// ---- pause and replay (design §19.3.13) --------------------------------------------------------------

// The resumed request's ledger as a prompt: its tokens through the next input, their RoPE positions,
// and its lookup keys (the prompt's, chained over the output blocks as prefix_publish chains them);
// no tap hints (a replay publishes blocks, not taps).
qwen3_5::PreparedPromptData ProgramImpl::ledger_prompt(const qwen3_5::PreparedPromptData& prompt,
                                                   const ResumeStateImpl& r) const {
    qwen3_5::PreparedPromptData out = prompt;
    const std::size_t n = r.history.size(), p = prompt.token_ids.size();
    if (r.prompt_tokens != p || n <= p || !std::equal(prompt.token_ids.begin(), prompt.token_ids.end(), r.history.begin())) {
        throw std::logic_error("Qwen4Exp: a paused ledger does not extend its prompt");
    }
    out.token_ids.assign(r.history.begin(), r.history.end());
    out.token_types.resize(n, 0);
    if (prompt.positions.size() != 3 * p) { throw std::logic_error("Qwen4Exp: a prompt carries no RoPE positions"); }
    out.positions.resize(3 * n);
    for (std::size_t axis = 3; axis-- > 0;) {
        // Axis-major [3, n]: move the prompt's rows to their new stride, then continue each.
        std::copy_backward(prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * p),
                           prompt.positions.begin() + static_cast<std::ptrdiff_t>((axis + 1) * p),
                           out.positions.begin() + static_cast<std::ptrdiff_t>(axis * n + p));
        for (std::size_t token = p; token < n; ++token) {
            out.positions[axis * n + token] = static_cast<std::int32_t>(static_cast<std::int64_t>(token) + prompt.rope_delta);
        }
    }
    const std::uint64_t trailing = prefix_trailing_extra(prompt);
    const bool extras            = !prompt.block_extras.empty();
    for (std::size_t b = out.block_hashes.size(); b < n / kBlock; ++b) {
        const std::uint64_t previous = b == 0 ? pc::kRootLookupHash : out.block_hashes[b - 1];
        const std::span<const TokenId> tokens(out.token_ids.data() + b * kBlock, kBlock);
        out.block_hashes.push_back(pc::block_lookup_hash(previous, tokens, trailing));
        if (extras) { out.block_extras.push_back(trailing); }
    }
    out.tap_hints.hints.clear();
    return out;
}


// ---- publication -------------------------------------------------------------------------------------

// The frontier below which no write of the lane remains (plan §7.4).
std::uint32_t ProgramImpl::prefix_frontier(const Lane& lane, bool finishing) const noexcept {
    if (!lane.mtp_live) { return lane.state_tokens; }
    if (finishing || lane.phase == Phase::Prefill) { return lane.mtp_cells; }
    return std::min(lane.mtp_cells, lane.state_tokens - 1U);
}

void ProgramImpl::prefix_publish(Lane& lane, std::uint32_t frontier) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    // Without free cache resume only blocks prefill computed are published: decode's KV differs.
    if (!options_.free_cache_resume) { frontier = std::min(frontier, lane.prompt_tokens); }
    pc::PrefixCacheIndex& cache = prefix_->index();
    // Blocks past the prompt (generated output) chain their lookup hash from the previous block's
    // and carry the prompt's trailing Vision key, as Qwen3.5's publication does.
    const std::uint32_t blocks =
        std::min<std::uint32_t>(frontier, static_cast<std::uint32_t>(lane.history.size())) / kBlock;
    while (lane.prefix.path.size() < blocks) {
        const auto b = static_cast<std::uint32_t>(lane.prefix.path.size());
        const pc::NodeRef parent = b == 0 ? pc::NodeRef{} : lane.prefix.path.back();
        const std::span<const TokenId> tokens(lane.history.data() + static_cast<std::size_t>(b) * kBlock, kBlock);
        const std::uint64_t extra =
            b < lane.prefix.extras.size() ? lane.prefix.extras[b] : lane.prefix.trailing_extra;
        if (b >= lane.prefix.hashes.size()) {
            const std::uint64_t previous = b == 0 ? pc::kRootLookupHash : lane.prefix.hashes[b - 1];
            lane.prefix.hashes.push_back(pc::block_lookup_hash(previous, tokens, extra));
        }
        const std::int32_t next      = lane.history.size() > static_cast<std::size_t>(kBlock) * (b + 1U)
                                           ? lane.history[static_cast<std::size_t>(kBlock) * (b + 1U)]
                                           : -1;
        // MR5: adopt a Host-only twin only when its last MTP cell encodes this lane's next token.
        bool attach = true;
        if (const auto existing = cache.find_child(parent, lane.prefix.hashes[b], tokens, extra); existing && mtp_) {
            attach = prefix_->mtp_next(*existing) == next;
            if (!attach) { ++prefix_->counters().mtp_branch_mismatches; }
        }
        DeviceKVPageLease& lease = lane.pages.at(b - lane.prefix.page_base);
        const pc::InsertResult result =
            prefix_->insert_block(parent, lane.prefix.hashes[b], tokens, extra, lease, attach, next);
        lane.prefix.path.push_back(result.node);
    }
    // The prompt-end capture's anchor is the prompt's last block, published once the first decode
    // round has written its pending cell.
    if (lane.prefix.pending_anchor != 0 && lane.prefix.path.size() >= lane.prompt_tokens / kBlock) {
        prefix_->resolve_capture_anchor(lane.prefix.pending_anchor, lane.prefix.path[lane.prompt_tokens / kBlock - 1]);
        lane.prefix.pending_anchor = 0;
    }
}

// A tap or endpoint snapshot of the lane's state at its frontier F = state_tokens (Host-born).
void ProgramImpl::prefix_capture(Lane& lane, std::uint32_t index, pc::SnapshotKind kind, bool hand_over_tail) {
    const std::uint32_t F = lane.state_tokens;
    const std::uint32_t k = F / kBlock;
    if (F == 0 || lane.prefix.path.size() < k) { return; }
    prefix::PrefixCache::Capture c =
        prefix_capture_at(lane, F, kind, hand_over_tail, lane.mtp_live && lane.mtp_cells == F);
    const std::uint64_t id = c.id;
    if (prefix_->capture(index, std::move(c), device_.stream)) { prefix_captured(lane, id, F); }
}

void ProgramImpl::prefix_captured(Lane& lane, std::uint64_t id, std::uint32_t F) {
    prefix_->forget_capture(lane.prefix.capture);
    lane.prefix.capture  = id;
    lane.prefix.captured = true;
    lane.prefix.deepest  = F;
}

prefix::PrefixCache::Capture ProgramImpl::prefix_capture_at(Lane& lane, std::uint32_t F, pc::SnapshotKind kind,
                                                            bool hand_over_tail, bool mtp_written,
                                                            DeviceKVPageLease* fresh_tail) {
    const std::uint32_t k = F / kBlock;
    prefix::PrefixCache::Capture c;
    c.id       = ++next_capture_;
    // A block-aligned MTP prompt end's anchor block is not published yet (anchor_pending).
    c.anchor   = k == 0 || lane.prefix.path.size() < k ? pc::NodeRef{} : lane.prefix.path[k - 1];
    c.frontier = F;
    c.tail.assign(lane.history.begin() + static_cast<std::ptrdiff_t>(k) * kBlock, lane.history.begin() + F);
    if (!c.tail.empty()) {
        if (fresh_tail != nullptr) {
            // The prompt-end capture: the snapshot takes a private copy of the lane's page, which
            // decode keeps writing (the lane's own stays with it, its table entry unchanged).
            c.tail_page = fresh_tail->handle();
            c.tail_lease.emplace(std::move(*fresh_tail));
        } else {
            DeviceKVPageLease& page = lane.pages.at(k - lane.prefix.page_base);
            c.tail_page             = page.handle();
            if (hand_over_tail) { c.tail_lease.emplace(std::move(page)); }
        }
    }
    c.kind             = kind;
    c.claim            = prefix_->index().estimate_priority(lane.prefix.deepest, F, !c.tail.empty());
    c.meta.frontier    = F;
    c.meta.mtp_written = mtp_written;
    c.meta.mtp_next    = lane.history.size() > F ? lane.history[F] : -1;
    c.meta.lineage_echo = lane.prefix.resume && prefix_->index().valid(*lane.prefix.resume) &&
                          prefix_->index().snapshot(*lane.prefix.resume).kind == pc::SnapshotKind::Endpoint;
    c.meta.opener = kind != pc::SnapshotKind::Endpoint &&
                    std::any_of(lane.prefix.hints.begin(), lane.prefix.hints.end(), [&](const pc::TapHint& hint) {
                        return hint.kind == pc::TapHintKind::GenerationOpener && hint.position == F;
                    });
    c.meta.mtp_accept = lane.mtp_accept;
    c.meta.draft_frontier = std::min(lane.dflash_frontier, F);
    // The request's first capture supersedes the snapshot it resumed from, which then serves only
    // requests diverging before this one. Its own earlier captures stay retained: its ladder taps and
    // its system-block boundary are where edits and side calls over the history diverge (§19.3.1).
    if (!lane.prefix.captured && lane.prefix.resume) {
        c.supersedes = *lane.prefix.resume;
        if (prefix_->index().valid(*lane.prefix.resume)) {
            const std::uint32_t from = prefix_->index().snapshot(*lane.prefix.resume).frontier;
            if (from < F) {
                c.continuation.assign(lane.history.begin() + from, lane.history.begin() + F);
            }
        }
    }
    return c;
}

bool ProgramImpl::prefix_take_taps(Lane& lane, std::uint32_t B, std::size_t next_call, bool& boundary) {
    if (lane.prefix.taps.empty()) { return false; }
    // A boundary inside a Vision item cannot be resumed from: its flexible taps wait for the next.
    if (prefix::inside_exclusion(B, lane.prefix.exclusions)) { return false; }
    // An exact tap ends a call; a flexible one is realized at the first boundary at or past it, or
    // at the start of the prompt's final call.
    const bool final_next = next_call + 1U == lane.calls.size();
    bool due              = false;
    std::erase_if(lane.prefix.taps, [&](const pc::PlannedTap& tap) {
        if (tap.position > B && !(final_next && tap.placement == pc::TapPlacement::Flexible)) { return false; }
        due      = true;
        boundary = boundary || tap.boundary;
        return true;
    });
    return due;
}

// After a prefill call ending at the lane's frontier: publish its blocks and realize the taps due
// at this call boundary.
void ProgramImpl::prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    prefix_publish(lane, prefix_frontier(lane, false));
    if (last) {
        prefix_prompt_end_capture(lane, index);
        return;
    }
    const std::uint32_t B = lane.state_tokens;
    bool boundary         = false;
    if (prefix_take_taps(lane, B, lane.next_call, boundary) && B >= lane.prefix.deepest + kBlock) {
        prefix_capture(lane, index, boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap, false);
    }
}

// The zero-token resume's prompt-end capture: a snapshot at the prompt's own frontier n, with the
// head's saved residual in its image, that an exact repeat resumes from with no prefill at all.
// The lane's next round is a graph-replayed decode, so the copy must not read the lane's state
// after it: the image stages through a fixed slot on the compute stream (capture_prompt_end), and
// the tail is a private copy of the lane's page. Under MTP the snapshot is the transition itself
// (MR1): the drafter's pending cell n - 1 is unwritten (its token is the first one the resume
// regenerates) and the resumed lane's first decode round writes it, into its private copy of the
// tail page or, for a block-aligned n, of the anchor block (MR6 at F == n). That anchor block is
// published only after the capturing lane's own first round has written the cell, so its capture
// copies now and publishes once the lane names the block (anchor_pending). Skips (and counts) when
// a drafter whose cells are not maintained, a frontier inside a Vision item, unpublished blocks,
// or no staging slot, Host slabs or free page rule it out. A snapshot below n suppresses it only
// within one tap separation: what the image saves a
// repeat is the re-fill from the path's deepest published frontier (the opener's realization, or
// the tail tap of a one-call span) up to n, while the image is a fixed cost beside the decode
// rounds that follow — measured on the artifact, capturing at every transition cost a concurrent
// six-request decode ~4 % tok/s and filled the Host tier, for repeats a tail call of at most the
// separation re-fills anyway. A span whose opener demoted to its last inner boundary (the
// multi-call re-fill loss, up to chunk - 1 tokens) is past the separation and still captures.
void ProgramImpl::prefix_prompt_end_capture(Lane& lane, std::uint32_t index) {
    if (!options_.free_cache_resume) { return; }
    const std::uint32_t n = lane.state_tokens, k = n / kBlock;
    // MR1 at the transition: the cells end at n - 1, or at n after a replay whose ledger knows the
    // token at n (either way the resume rewrites cell n - 1 privately).
    const bool drafter = !mtp_ || (lane.mtp_live && lane.mtp_cells + 1U >= n);
    // Under MTP the published blocks end at the cells: a block-aligned n's last block waits.
    const bool anchor_pending = mtp_ && k > 0 && lane.prefix.path.size() + 1U == k && lane.mtp_cells < n;
    if (n == 0 || !drafter || (lane.prefix.path.size() < k && !anchor_pending) ||
        prefix::inside_exclusion(n, lane.prefix.exclusions) ||
        n - lane.prefix.deepest < pc::kMinimumTapSeparation) {
        ++prefix_->counters().prompt_end_skipped;
        return;
    }
    // The fresh tail page: the snapshot's own copy of the lane's last partial block, taken before
    // the capture (decode writes the lane's page without waiting). Every later device writer of a
    // pool page runs on the compute stream or after it (the restore stream waits on its event), so
    // returning the page on a refusal cannot race the copy still ordered there.
    std::optional<DeviceKVPageReservation> reservation;
    std::optional<DeviceKVPageLease> fresh;
    if (n % kBlock != 0) {
        reservation = pool_->reserve(1);
        if (!reservation) {
            ++prefix_->counters().prompt_end_skipped;
            return;
        }
        fresh = pool_->materialize_one(*reservation);
        pool_->copy_page(lane.pages.at(k - lane.prefix.page_base).handle(), fresh->handle(), device_.stream);
    }
    // The prompt's end is the divergence point of repeats and forks: a Boundary. The estimate
    // values it against the lineage's deepest snapshot (the gate above already held the distance
    // to it to at least the separation; the tap spacing rule does not apply beyond that).
    prefix::PrefixCache::Capture c =
        prefix_capture_at(lane, n, pc::SnapshotKind::Boundary, false, false, fresh ? &*fresh : nullptr);
    c.meta.head_residual = true;
    c.anchor_pending     = anchor_pending;
    const std::uint64_t id = c.id;
    if (prefix_->capture_prompt_end(index, std::move(c), device_.stream)) {
        if (anchor_pending) { lane.prefix.pending_anchor = id; }
        // Not `deepest`: the finish endpoint above must still be captured (its guard reads
        // deepest + kBlock). A continuation resumes there, past the blocks the last turn
        // published with decode KV; a resume inside such a block deduplicates to the published
        // copy — whose bits differ from the prefill's — and a capture over that path pairs the
        // lane's prefill state with a path its own KV contradicts.
        const std::uint32_t deepest = lane.prefix.deepest;
        prefix_captured(lane, id, n);
        lane.prefix.deepest = deepest;
    } else {
        fresh.reset();       // the capture was refused: no staging slot or Host slabs (counted)
        reservation.reset();
    }
}

// Finish or a consistent abort: the M3 flush, the remaining blocks, the endpoint and write-through.
void ProgramImpl::prefix_finish(Lane& lane, std::uint32_t index) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    // MR3: write the pending MTP cell when its token is known, so the endpoint is final.
    if (lane.mtp_live && lane.mtp_cells + 1U == lane.state_tokens && lane.history.size() > lane.state_tokens) {
        mtp_flush_cell(lane, index);
    }
    prefix_publish(lane, prefix_frontier(lane, true));
    if (lane.prefix.pending_anchor != 0) {
        // The lane ends before its first token was decided (no flush could write the pending cell):
        // the prompt-end capture has no anchor block to publish under.
        prefix_->abandon_capture(lane.prefix.pending_anchor);
        lane.prefix.pending_anchor = 0;
    }
    const std::uint32_t F = lane.state_tokens;
    const bool full_blocks = !lane.mtp_live || lane.mtp_cells >= F / kBlock * kBlock;
    // An endpoint holds decode-computed state: published only with free cache resume.
    if (!options_.free_cache_resume) {
        prefix_->write_blocks(lane.prefix.path, device_.stream);
        lane.prefix.resident_capture = lane.prefix.deepest == F ? lane.prefix.capture : 0;
        return;
    }
    if (F >= lane.prefix.deepest + kBlock && full_blocks && lane.prefix.path.size() >= F / kBlock) {
        prefix_capture(lane, index, pc::SnapshotKind::Endpoint, true);
    } else if (!full_blocks) {
        ++prefix_->counters().endpoint_skipped_mtp;
    }
    // Blocks reach the Host after the endpoint (one transfer stream, FIFO).
    prefix_->write_blocks(lane.prefix.path, device_.stream);
    // The lane's slot holds the image of its latest capture until the lane is reused, but only when
    // that capture is at the lane's frontier: when the endpoint was not captured (too close to the
    // deepest snapshot, unfinished MTP blocks), the latest capture is an earlier tap whose state the
    // slot no longer holds, and a resume from it must restore its image.
    lane.prefix.resident_capture = lane.prefix.deepest == F ? lane.prefix.capture : 0;
}

// The endpoint this lane published at its last finish, while its slot still holds that image: only
// a snapshot the capture created (a duplicate's image came from another lineage).
std::uint64_t ProgramImpl::prefix_trailing_extra(const qwen3_5::PreparedPromptData& prompt) {
    std::uint64_t cumulative = 0;
    for (const auto& range : qwen3_5::detail::vision_ranges(prompt)) {
        cumulative = qwen3_5::detail::accumulate_vision(cumulative, range.key);
    }
    return cumulative;
}

std::optional<pc::SnapshotRef> ProgramImpl::prefix_resident(const Lane& lane) const {
    if (lane.prefix.resident_capture == 0) { return std::nullopt; }
    const auto result = prefix_->capture_result(lane.prefix.resident_capture);
    if (!result || !result->created || !prefix_->index().valid(result->snapshot)) { return std::nullopt; }
    return result->snapshot;
}

void ProgramImpl::prefix_release(Lane& lane) noexcept {
    if (!prefix_) { return; }
    try {
        // A lane released before its first call (cancelled, failed or aborted after binding) may still
        // have its restore landing in its private page and state slot: whatever reuses them next runs
        // on the compute stream, so that stream waits for the restore (free once it has landed).
        if (lane.prefix.restore) { prefix_->order_after_restore(*lane.prefix.restore, device_.stream); }
        // A tap taken just before the release (cancel, failure, pause) may still be copying the
        // lane's partial tail page; release() returns the lane's pages to the pool next.
        prefix_->order_page_reuse(lane.pages);
        if (lane.prefix.pending_anchor != 0) { prefix_->abandon_capture(lane.prefix.pending_anchor); }
        if (!lane.prefix.path.empty()) { prefix_->index().release_path(lane.prefix.path); }
        prefix_->poll();
    } catch (...) {}
    lane.prefix.path.clear();
    lane.prefix.taps.clear();
    lane.prefix.exclusions.clear();
    lane.prefix.restore.reset();
    lane.prefix.resume.reset();
    lane.prefix.head_resume = false;
    lane.prefix.pending_anchor = 0;
    lane.prefix.page_base = 0;
    lane.prefix.deepest   = 0;
    lane.prefix.reuse     = false;
}

std::array<std::span<const cudaEvent_t>, 2> ProgramImpl::prefix_waits(const Lane& lane) const noexcept {
    if (!prefix_) { return {}; }
    return {lane.prefix.restore ? prefix_->restore_events(*lane.prefix.restore) : std::span<const cudaEvent_t>{},
            prefix_->capture_events(lane.prefix.capture)};
}

} // namespace infernix::models::qwen4_exp::detail
