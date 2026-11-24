// The Program side of Glm5Next's prefix cache (docs/maintainer/glm5_3-flash-design.md §18):
// admission against the index, activation with per-layer restores, block publication, Host-born
// taps and the prompt-end captures of the zero-token resume, and release. The binding
// (prefix/prefix_cache.h) moves the bytes.
//
// Exactness (the phase-aligned chunking): GLM's BF16 projection routes are chosen by call width,
// so a resumed request's bits equal the publishing run's only when its calls from the resume
// frontier repeat the publishing run's call boundaries. Every tap lands on the chunk grid — a
// multiple of the prefill chunk — which is exactly the publishing run's call grid, so a resume
// from any tap replays the same call widths over the same restored state, and a snapshot's bits
// are the publishing run's at every position it covers. The prompt-end capture holds the prefill
// transition's own state (and the head's saved residual), so an exact repeat regenerates its
// first token with no prefill and continues on the decode grid the cold run used. There are no
// endpoint snapshots: decode-computed state at a position differs from prefill-computed state
// there (route by width), so a resume from one would not equal a cold run of the transcript.

#include "models/glm5_next/program/program_impl.h"
#include "models/glm5_next/program/prefix/block_keys.h"
#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>

namespace infernix::models::glm5_next::detail {

namespace pc = infernix::models::glm5_next::prefix::cache_index;

namespace {

constexpr std::uint32_t kBlock = pc::kBlockTokens;
// Blocks one prefetch batch copies at most (one page each).
constexpr std::uint32_t kPrefetchBatchBlocks = 256;

} // namespace

// A branch tap (prefix_select) needs the prompt to share at least this many cached tokens past
// its resume frontier.
inline constexpr std::uint32_t kBranchTapMinTokens = 1024;

// ---- construction ------------------------------------------------------------------------------------

void ProgramImpl::create_prefix_cache(std::uint64_t& allocated) {
    if (!options_.prefix_cache) { return; }
    const auto lanes = static_cast<std::uint32_t>(options_.max_concurrency);
    // A DFlash2 drafter's context ring: per layer, one lane's K and V (the lane is outermost).
    prefix::DraftRingGeometry ring;
    std::vector<std::byte*> ring_k, ring_v;
    if (drafter_) {
        ring.layers = ring_->layer_count();
        for (std::uint32_t l = 0; l < ring.layers; ++l) {
            const CyclicKVCacheLayerView view = ring_->layer_view(l);
            ring.k_bytes                      = view.k.bytes() / lanes;
            ring.v_bytes                      = view.v.bytes() / lanes;
            ring_k.push_back(static_cast<std::byte*>(view.k.data));
            ring_v.push_back(static_cast<std::byte*>(view.v.data));
        }
    }
    // The indexer's ring: per MLA block the raw keys and gates of the incomplete pool, two planes
    // whose lane is the innermost axis.
    const std::size_t index_bytes = static_cast<std::size_t>(c_.indexer.head_dim) * 64U * 2U;
    const std::size_t index_plane_bytes = index_bytes * lanes;
    image_layout_ = std::make_unique<prefix::StateImageLayout>(
        prefix::plan_state_image(prefix::state_image_spec(c_, ring, dim(c_.residual_width()) * 2U)));
    lane_image_    = std::make_unique<prefix::LaneStateImage>(
        *image_layout_,
        prefix::LaneStateBuffers{.kda               = kda_.get(),
                                 .lanes             = lanes,
                                 .index_ring        = static_cast<std::byte*>(index_ring_.p),
                                 .index_plane_bytes = index_plane_bytes,
                                 .head_saved        = static_cast<std::byte*>(head_saved_.p),
                                 .ring_k            = std::move(ring_k),
                                 .ring_v            = std::move(ring_v)});
    prefix::PrefixCacheConfig config;
    config.index.cost = options_.prefix_cost;
    config.taps       = options_.prefix_taps;
    config.kv_pages   = plan_.kv_pages;
    prefix_           = std::make_unique<prefix::PrefixCache>(config, *pool_, kv_geometry_, *lane_image_,
                                                              options_.prefix_host_bytes);
    prefix_->set_compute_stream(device_.stream);
    // The zero-token resume's prompt-end captures stage their image through fixed Device slots (a
    // lane's replayed decode graphs cannot wait on capture events): two cover a burst of
    // concurrent prompt ends, a third skips and counts.
    const std::uint32_t slots = std::min(lanes, 2U);
    std::vector<std::byte*> bytes;
    prompt_end_staging_.reserve(slots);
    bytes.reserve(slots);
    for (std::uint32_t s = 0; s < slots; ++s) {
        prompt_end_staging_.emplace_back();
        allocate(prompt_end_staging_.back(), image_layout_->image_bytes, allocated);
        bytes.push_back(static_cast<std::byte*>(prompt_end_staging_.back().p));
    }
    prefix_->set_prompt_end_staging(std::move(bytes));
}

void ProgramImpl::pc_make_room(std::uint32_t pages) {
    pc::PrefixCacheIndex& cache = prefix_->index();
    while (pool_->available_pages() < pages) {
        if (cache.evict_device_blocks(pages - pool_->available_pages()) == 0) { return; }
    }
}

// ---- admission ---------------------------------------------------------------------------------------

// Chooses the resume source for `prompt` in lane `lane`: the deepest affordable snapshot, else the
// root, or nothing when even the root does not fit now (design §18).
std::optional<PrefixSelection> ProgramImpl::prefix_select(const PreparedPromptData& prompt,
                                                          const BasePlanImpl& base, std::uint32_t lane) {
    pc::PrefixCacheIndex& index = prefix_->index();
    const auto n                = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::uint32_t E       = base.pages;
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
        if (s.cow_tail) {
            const pc::SnapshotView view = index.snapshot(*s.snapshot);
            if (view.tail_device_copy == pc::CopyState::Resident && view.pins == 0) { ++evictable_on_path; }
        }
        s.need                     = (E - s.shared) + host_only;
        const std::uint64_t room   = static_cast<std::uint64_t>(pool_->available_pages()) +
                                   index.device_evictable_blocks();
        return static_cast<std::uint64_t>(s.need) + evictable_on_path <= room;
    };
    std::uint32_t cached_tokens = 0; // the prompt prefix held as cached blocks
    const auto planned = [&](PrefixSelection s) {
        // Where the prompt leaves a cached path well past its resume frontier (a client re-sending
        // a long transcript with new content inserted before its end): a boundary snapshot there
        // lets the next such prompt resume at the branch instead of re-prefilling the shared
        // tokens after the frontier again.
        const std::uint32_t branch =
            cached_tokens < n && cached_tokens >= s.frontier + kBranchTapMinTokens ? cached_tokens : 0;
        s.plan          = plan_prefill(prompt, s.frontier, s.existing, true, branch);
        s.cached_tokens = cached_tokens;
        return s;
    };
    PrefixSelection root;
    root.need = E;
    if (n > 1) {
        prefix_->poll();
        pc::MatchResult match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        if (prefetch_ticket_ && !prefix_->restore_events(*prefetch_ticket_).empty() &&
            std::any_of(match.candidates.begin(), match.candidates.end(),
                        [](const pc::MatchCandidate& c) { return c.filling_blocks != 0; })) {
            // A prefetch is copying blocks this prompt resumes over: they land within a batch's
            // copy time, and skipping them would pick a shallower source.
            prefix_->await_restore(*prefetch_ticket_);
            prefetch_ticket_.reset();
            match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        }
        cached_tokens = static_cast<std::uint32_t>(match.path.size()) * kBlock;
        // Every resume keeps the publishing run's call grid: a candidate's frontier is a multiple
        // of the chunk — except the prompt's own end with its head residual, the zero-token resume
        // (no prefill at all). Snapshots inside the grid (their tails) serve only that.
        const std::uint32_t chunk = static_cast<std::uint32_t>(chunk_);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return c.filling_blocks != 0 ||
                   (c.frontier != n && c.frontier % chunk != 0) ||
                   (c.frontier == n && !c.head_residual);
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
        for (const std::size_t i : order) {
            const pc::MatchCandidate& c = match.candidates[i];
            PrefixSelection s;
            s.snapshot = c.snapshot;
            s.frontier = c.frontier;
            s.path.assign(match.path.begin(), match.path.begin() + c.path_blocks);
            const std::uint32_t k = c.frontier / kBlock;
            s.shared              = k;
            if (c.frontier % kBlock != 0) { s.cow_tail = true; }
            s.image_restore = !(resident && *resident == c.snapshot);
            s.existing      = frontiers;
            if (c.frontier == n) {
                // The zero-token resume: the snapshot at the prompt's own end carries the whole
                // state and its head's saved residual, so the prefill plan is empty and the first
                // token regenerates from the residual. Match offers the frontier only with its
                // residual.
                if (fits(s)) {
                    s.cached_tokens = cached_tokens;
                    return s;
                }
                continue;
            }
            if (fits(s)) { return planned(std::move(s)); }
        }
        if (!match.candidates.empty() && prefix_->transfers_pending()) {
            // Blocks pinned only by in-flight Host writes become evictable once those land: a
            // cached source is not given up for the root while a finished lane's writes land.
            prefix_->drain();
            for (std::size_t i : order) {
                const pc::MatchCandidate& c = match.candidates[i];
                PrefixSelection s;
                s.snapshot = c.snapshot;
                s.frontier = c.frontier;
                s.path.assign(match.path.begin(), match.path.begin() + c.path_blocks);
                const std::uint32_t k = c.frontier / kBlock;
                s.shared              = k;
                if (c.frontier % kBlock != 0) { s.cow_tail = true; }
                s.image_restore = !(resident && *resident == c.snapshot);
                s.existing      = frontiers;
                if (c.frontier == n) {
                    if (fits(s)) {
                        s.cached_tokens = cached_tokens;
                        return s;
                    }
                    continue;
                }
                if (fits(s)) { return planned(std::move(s)); }
            }
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

// Credits what needs no transfer, then leaves only sources restorable now (a lane-resident
// snapshot's image is still in the lane).
void ProgramImpl::prefix_credit(std::vector<pc::MatchCandidate>& candidates,
                                std::optional<pc::SnapshotRef> resident) const {
    for (pc::MatchCandidate& c : candidates) {
        if (resident && *resident == c.snapshot) {
            c.restore_bytes -= std::min<std::uint64_t>(c.restore_bytes, image_layout_->image_bytes);
        }
    }
}

std::uint32_t ProgramImpl::hybrid_prefetch_room() const noexcept {
    if (!prefix_ || !prefix_->host_tier()) { return 0U; }
    return pool_->available_pages() + prefix_->index().device_backed_evictable_blocks();
}

// The blocked FIFO head's Host-only blocks are copied into spare Device cache while it waits
// (hybrid-prefix-cache-spec §6.6): its matched path is pinned while room is made from free pages
// and Host-backed cached blocks, least recently used first, so a prefetch never waits for a
// transfer, never grows the pool into the expert frames, and never drops a block's last copy.
std::optional<std::uint32_t> ProgramImpl::hybrid_prefetch(const RequestBasePlan& base) {
    if (!prefix_ || !prefix_->host_tier() || base.impl_ == nullptr || !base.impl_->reuse || !base.impl_->prompt) {
        return 0U;
    }
    if (transaction_lane_ || prefix_->restore_open()) { return std::nullopt; }
    prefix_->poll();
    if (prefetch_ticket_ && !prefix_->restore_events(*prefetch_ticket_).empty()) { return std::nullopt; }
    prefetch_ticket_.reset();
    const auto& prompt = PreparedPromptAccess::view(*base.impl_->prompt);
    const auto n       = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n <= 1 || prompt.block_hashes.size() != n / kBlock) { return 0U; }
    // The source the admission would choose now, as in prefix_select.
    pc::PrefixCacheIndex& index = prefix_->index();
    pc::MatchResult match       = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
    const std::uint32_t chunk   = static_cast<std::uint32_t>(chunk_);
    std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
        return c.filling_blocks != 0 ||
               (c.frontier != n && c.frontier % chunk != 0) || (c.frontier == n && !c.head_residual);
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
        const auto& prompt = PreparedPromptAccess::view(*base->impl_->prompt);
        const auto n       = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (n <= 1 || prompt.block_hashes.size() != n / kBlock) { continue; }
        pc::MatchResult match = index.match(prompt.token_ids, prompt.block_hashes, prompt.block_extras, n);
        const std::uint32_t chunk = static_cast<std::uint32_t>(chunk_);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return c.filling_blocks != 0 ||
                   (c.frontier != n && c.frontier % chunk != 0) || (c.frontier == n && !c.head_residual);
        });
        const pc::AdmissionChoice choice = index.choose(match, n);
        if (choice.candidate) { holds.push_back(match.candidates[*choice.candidate].snapshot); }
    }
    index.set_queue_holds(holds);
}

std::uint64_t ProgramImpl::hybrid_cache_epoch() const noexcept {
    return prefix_ ? prefix_->index().snapshot_epoch() : 0U;
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
// restores and copy-on-write sources, and sets the lane's frontier (design §18).
void ProgramImpl::prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& s,
                                  DeviceKVPageReservation& reservation, std::uint32_t pages) {
    pc::PrefixCacheIndex& cache = prefix_->index();
    const cudaStream_t stream   = device_.stream;
    // A copy-out of this lane's previous state must land before anything overwrites the slot.
    if (lane.prefix.capture != 0) {
        const auto events = prefix_->capture_events(lane.prefix.capture);
        if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(stream, events.back(), 0)); }
    }
    // The lane keeps the pins of the blocks it maps (prefix_pin took one on each up to the anchor).
    lane.prefix.path.assign(s.path.begin(), s.path.begin() + s.shared);
    lane.pages.clear();
    lane.pages.reserve(pages - s.shared);
    pool_->materialize(reservation, pages - s.shared, lane.pages);
    lane.prefix.page_base = s.shared;
    const bool restores   = s.snapshot && (s.image_restore || s.cow_tail ||
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
        if (s.cow_tail) {
            const pc::SnapshotView view = cache.snapshot(*s.snapshot);
            if (view.tail_device_copy == pc::CopyState::Resident) {
                pool_->copy_page(prefix_->page(view.tail_device), first_private, stream);
            } else {
                prefix_->restore_tail_page(*s.snapshot, first_private);
            }
        }
        if (s.image_restore) { prefix_->restore_image(*s.snapshot, index); }
        if (restores) { lane.prefix.restore = prefix_->submit_restore(); }
        cache.note_hit(*s.snapshot);
        cache.unpin_snapshot(*s.snapshot); // the restore batch holds its own pin while it lands
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
        const prefix::StateImageHeader& meta = prefix_->meta(*s.snapshot);
        // The zero-token resume's lane flag, read at the prefill step (the snapshot itself may be
        // evicted once its restore has landed; the lane keeps the state it restored).
        lane.prefix.head_resume = meta.head_residual && s.frontier == lane.prompt_tokens;
        // Penalty counts restart with the request; the lane's state came from the snapshot.
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + index * static_cast<std::size_t>(token_domain_),
                                   0, 4ULL * token_domain_, stream));
    }
    // Taps for this prompt beyond the resume frontier, realized at the plan's call boundaries.
    lane.prefix.taps = s.plan.taps;
}

// The prompt's prefill calls from `frontier`: without taps the grid from the frontier; with them,
// a tap lands on the chunk grid at or before its planned position, so each is a call end of the
// grid the publishing run used and splits nothing (the phase-aligned chunking). A branch tap (a
// cached path the prompt leaves well past its resume frontier) is exact there — it becomes the
// next chunk-grid boundary at or past it.
prefix::CallPlan ProgramImpl::plan_prefill(const PreparedPromptData& prompt, std::uint32_t frontier,
                                           std::span<const std::uint32_t> existing, bool taps,
                                           std::uint32_t branch) const {
    const auto n = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::uint32_t chunk = static_cast<std::uint32_t>(chunk_);
    std::vector<pc::PlannedTap> planned;
    if (taps && prefix_) {
        planned = pc::plan_taps(n, frontier, prompt.tap_hints.hints, existing, {}, prefix_->taps());
        if (branch > frontier && branch < n) {
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
        // Every tap to the chunk-grid position at or before it: it cuts no call and a resume from
        // it keeps the publishing run's call grid.
        for (pc::PlannedTap& tap : planned) { tap.position = tap.position / chunk * chunk; }
        std::erase_if(planned, [&](const pc::PlannedTap& tap) { return tap.position <= frontier; });
        planned.erase(std::unique(planned.begin(), planned.end(),
                                  [](const pc::PlannedTap& a, const pc::PlannedTap& b) {
                                      return a.position == b.position;
                                  }),
                      planned.end());
        // The prompt's last chunk-grid boundary: a snapshot there covers every fork and repeat
        // that shares the prompt's body but changes its tail, which the prompt-end capture (only
        // an exact repeat resumes from) leaves stranded. The boundary needs a block below it and
        // the ladder's separation from the frontier.
        const std::uint32_t tail = (n - 1U) / chunk * chunk;
        if (tail > frontier && tail >= frontier + kBlock && tail % kBlock == 0 &&
            !std::any_of(planned.begin(), planned.end(),
                         [&](const pc::PlannedTap& tap) { return tap.position == tail; }) &&
            !std::any_of(existing.begin(), existing.end(),
                         [&](std::uint32_t f) { return f + pc::kMinimumTapSeparation > tail; })) {
            planned.push_back(pc::PlannedTap{.position  = tail,
                                             .placement = pc::TapPlacement::Exact,
                                             .boundary  = true});
        }
        planned.shrink_to_fit();
    }
    return prefix::plan_calls(frontier, n, chunk, planned, call_cost());
}

prefix::CallCost ProgramImpl::call_cost() const {
    prefix::CallCost cost;
    cost.model               = options_.prefix_cost;
    cost.served_columns      = 8;      // calls this narrow promote at the decode rate (execution.cpp)
    cost.served_call_seconds = 0.035;  // their fixed cost, estimated
    cost.split_budget        = 0.1;    // the most an automatic exact tap may add
    return cost;
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
    out.prompt_end_gated        = count.prompt_end_gated;
    out.prompt_end_failed       = count.prompt_end_failed;
    out.host_image_writes       = count.host_image_writes;
    out.host_block_writes       = count.host_block_writes;
    out.host_image_restores     = count.host_image_restores;
    out.host_block_restores     = count.host_block_restores;
    out.prefetched_blocks       = count.prefetched_blocks;
    out.host_write_bytes        = count.host_write_bytes;
    out.host_restore_bytes      = count.host_restore_bytes;
    out.evicted_blocks          = index.device_block_evictions;
    out.host_snapshot_evictions = index.host_snapshot_evictions;
    out.host_backed_evictions   = index.host_backed_evictions;
    out.host_dead_reclaims      = index.host_dead_reclaims;
    out.unbacked_node_losses    = index.unbacked_node_losses;
    out.held_snapshots          = index.held_snapshots;
    out.held_device_evictions   = index.held_device_evictions;
    out.held_snapshot_losses    = index.held_snapshot_losses;
    out.held_host_refusals      = index.held_host_refusals;
    return out;
}

// ---- publication -------------------------------------------------------------------------------------

void ProgramImpl::prefix_publish(Lane& lane, std::uint32_t frontier) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    // Only blocks prefill computed are published: decode's KV differs (route by width).
    frontier = std::min(frontier, lane.prompt_tokens);
    pc::PrefixCacheIndex& cache = prefix_->index();
    const std::uint32_t blocks  = frontier / kBlock;
    while (lane.prefix.path.size() < blocks) {
        const auto b = static_cast<std::uint32_t>(lane.prefix.path.size());
        const pc::NodeRef parent = b == 0 ? pc::NodeRef{} : lane.prefix.path.back();
        const std::span<const TokenId> tokens(lane.history.data() + static_cast<std::size_t>(b) * kBlock, kBlock);
        if (b >= lane.prefix.hashes.size()) {
            const std::uint64_t previous = b == 0 ? pc::kRootLookupHash : lane.prefix.hashes[b - 1];
            lane.prefix.hashes.push_back(pc::block_lookup_hash(previous, tokens, lane.prefix.trailing_extra));
        }
        DeviceKVPageLease& lease = lane.pages.at(b - lane.prefix.page_base);
        const pc::InsertResult result =
            prefix_->insert_block(parent, lane.prefix.hashes[b], tokens, lane.prefix.trailing_extra, lease, true);
        lane.prefix.path.push_back(result.node);
    }
}

// A tap snapshot of the lane's state at its frontier F = state_tokens (Host-born).
void ProgramImpl::prefix_capture(Lane& lane, std::uint32_t index, pc::SnapshotKind kind) {
    const std::uint32_t F = lane.state_tokens;
    const std::uint32_t k = F / kBlock;
    if (F == 0 || lane.prefix.path.size() < k) { return; }
    prefix::PrefixCache::Capture c = prefix_capture_at(lane, F, kind, false);
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
                                                            bool hand_over_tail, DeviceKVPageLease* fresh_tail) {
    const std::uint32_t k = F / kBlock;
    prefix::PrefixCache::Capture c;
    c.id       = ++next_capture_;
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
    // The generation-opener tap (a protocol hint): it marks the split before the assistant turn.
    c.meta.opener = std::any_of(lane.prefix.hints.begin(), lane.prefix.hints.end(),
                                [&](const runtime::prefix_cache::TapHint& hint) {
                                    return hint.kind == runtime::prefix_cache::TapHintKind::GenerationOpener &&
                                           hint.position == F;
                                });
    c.meta.draft_frontier = F;
    // The request's first capture supersedes the snapshot it resumed from, which then serves only
    // requests diverging before this one. Its own earlier captures stay retained: its taps are
    // where edits and side calls over the history diverge (§17.1).
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
    // An exact tap ends a call; it is realized at the boundary at or past its position.
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
    prefix_publish(lane, lane.state_tokens);
    if (last) {
        prefix_prompt_end_capture(lane, index);
        return;
    }
    const std::uint32_t B = lane.state_tokens;
    bool boundary         = false;
    if (prefix_take_taps(lane, B, lane.next_call, boundary) && B >= lane.prefix.deepest + kBlock) {
        prefix_capture(lane, index, boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap);
    }
}

// The zero-token resume's prompt-end capture: a snapshot at the prompt's own frontier n, with the
// head's saved residual in its image, that an exact repeat resumes from with no prefill at all.
// The lane's next round is a graph-replayed decode, so the copy must not read the lane's state
// after it: the image stages through a fixed slot on the compute stream (capture_prompt_end), and
// the tail is a private copy of the lane's page. Skips (and counts) when the frontier is within
// one tap separation of the path's deepest published frontier (what the image saves a repeat is
// the re-fill from that frontier up to n, and the image is a fixed cost beside the decode rounds
// that follow), or when unpublished blocks or no staging slot, Host slabs or free page rule it
// out. The capture does not raise the lane's `deepest`: in this family there is no finish endpoint
// above it to protect, but the value gate against the lineage's taps still reads it.
void ProgramImpl::prefix_prompt_end_capture(Lane& lane, std::uint32_t index) {
    const std::uint32_t n = lane.state_tokens, k = n / kBlock;
    if (n == 0 || lane.prefix.path.size() < k || n - lane.prefix.deepest < pc::kMinimumTapSeparation) {
        ++prefix_->counters().prompt_end_gated;
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
            ++prefix_->counters().prompt_end_failed;
            return;
        }
        fresh = pool_->materialize_one(*reservation);
        pool_->copy_page(lane.pages.at(k - lane.prefix.page_base).handle(), fresh->handle(), device_.stream);
    }
    // The prompt's end is the divergence point of repeats and forks: a Boundary. The estimate
    // values it against the lineage's deepest snapshot (the gate above already held the distance
    // to it to at least the separation; the tap spacing rule does not apply beyond that).
    prefix::PrefixCache::Capture c =
        prefix_capture_at(lane, n, pc::SnapshotKind::Boundary, false, fresh ? &*fresh : nullptr);
    c.meta.head_residual = true;
    const std::uint64_t id = c.id;
    if (prefix_->capture_prompt_end(index, std::move(c), device_.stream)) {
        prefix_captured(lane, id, n);
        // Not `deepest`: the finish's block publication still runs (its guard reads deepest +
        // kBlock); the capture's own value is what a later lineage's estimate sees.
        lane.prefix.deepest = n;
    } else {
        fresh.reset();       // the capture was refused: no staging slot or Host slabs (counted)
        reservation.reset();
    }
}

// Finish or a consistent abort: the remaining blocks and write-through. No endpoint: this family
// publishes only prefill-computed state.
void ProgramImpl::prefix_finish(Lane& lane, std::uint32_t index) {
    if (!prefix_ || !lane.prefix.reuse) { return; }
    prefix_publish(lane, lane.state_tokens);
    prefix_->write_blocks(lane.prefix.path, device_.stream);
    // The lane's slot holds the image of its latest capture until the lane is reused, but only
    // when that capture is at the lane's frontier: when the last capture is an earlier tap, its
    // state the slot no longer holds, and a resume from it must restore its image.
    lane.prefix.resident_capture = lane.prefix.deepest == lane.state_tokens ? lane.prefix.capture : 0;
}

void ProgramImpl::prefix_release(Lane& lane) noexcept {
    if (!prefix_) { return; }
    try {
        // A lane released before its first call (cancelled, failed or aborted after binding) may
        // still have its restore landing in its private page and state slot: whatever reuses them
        // next runs on the compute stream, so that stream waits for the restore (free once it has
        // landed).
        if (lane.prefix.restore) { prefix_->order_after_restore(*lane.prefix.restore, device_.stream); }
        // A tap taken just before the release (cancel, failure) may still be copying the lane's
        // partial tail page; release() returns the lane's pages to the pool next.
        prefix_->order_page_reuse(lane.pages);
        if (!lane.prefix.path.empty()) { prefix_->index().release_path(lane.prefix.path); }
        prefix_->poll();
    } catch (...) {}
    lane.prefix.path.clear();
    lane.prefix.taps.clear();
    lane.prefix.hints.clear();
    lane.prefix.restore.reset();
    lane.prefix.resume.reset();
    lane.prefix.head_resume = false;
    lane.prefix.page_base = 0;
    lane.prefix.deepest   = 0;
    lane.prefix.reuse     = false;
    lane.prefix.captured  = false;
    lane.prefix.capture   = 0;
    lane.prefix.resident_capture = 0;
    lane.prefix.cached_tokens    = 0;
    lane.prefix.restored_bytes   = 0;
}

std::span<const cudaEvent_t> ProgramImpl::prefix_capture_wait(const Lane& lane) const noexcept {
    if (!prefix_ || lane.prefix.capture == 0) { return {}; }
    return prefix_->capture_events(lane.prefix.capture);
}

std::span<const cudaEvent_t> ProgramImpl::prefix_restore_wait(const Lane& lane) const noexcept {
    if (!prefix_ || !lane.prefix.restore) { return {}; }
    return prefix_->restore_events(*lane.prefix.restore);
}

std::optional<pc::SnapshotRef> ProgramImpl::prefix_resident(const Lane& lane) const {
    if (lane.prefix.resident_capture == 0) { return std::nullopt; }
    const auto result = prefix_->capture_result(lane.prefix.resident_capture);
    if (!result || !result->created || !prefix_->index().valid(result->snapshot)) { return std::nullopt; }
    return result->snapshot;
}

PrefixSelection ProgramImpl::prefix_root(const PreparedPromptData& prompt, const BasePlanImpl& base) {
    PrefixSelection root;
    root.need = base.pages;
    root.plan = plan_prefill(prompt, 0, {}, base.reuse);
    return root;
}

} // namespace infernix::models::glm5_next::detail
