// The layer walk of Glm5Next's prefill (docs/maintainer/glm5_3-flash-design.md §11.1): a span of
// consecutive streamed calls of one prompt runs layer-major, so each MoE layer's non-resident
// experts cross PCIe once per span instead of once per call. A chunk-major prompt of 4,096-token
// calls re-streams nearly every non-resident expert for every call (about 100 GB at vram=42%),
// more than the call's compute.
//
// The span keeps the prompt's call grid: chunk c of layer l runs exactly what the chunk-major call
// c runs at layer l (its own columns at its own width, and the layer's KDA state and MLA caches
// after chunks 0..c-1), so every output bit equals the chunk-major prefill's. Only where an expert
// is read from (a frame, the stream's ring or staging) differs, which the MoE's arithmetic does not
// see. A span ends at a call whose end a prefix tap is due at; the tap is captured after the span,
// where the lane's state is the chunk-major run's state there.
//
// Each advance_prefill call enqueues about one call's work (a few layers of every chunk) and
// synchronizes, so cancellation and other lanes' rounds interleave as between chunk-major calls.
// While a walk is active, other lanes' rounds land and promote no experts and the placement does
// not switch: the span's stream planned its copies against the frames of its start.

#include "models/glm5_next/program/program_impl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace infernix::models::glm5_next::detail {

namespace {

constexpr std::size_t kWalkAlign = 256;

std::size_t walk_align(std::size_t bytes) {
    return (bytes + kWalkAlign - 1) / kWalkAlign * kWalkAlign;
}

// The chunk words: {slot, table row, logit column, pad}.
constexpr std::size_t kMetaWords = 4;

} // namespace

bool ProgramImpl::tap_due(const Lane& lane, std::uint32_t B, std::size_t next_call) const noexcept {
    if (!prefix_ || !lane.prefix.reuse) { return false; }
    const bool final_next = next_call + 1U == lane.calls.size();
    return std::any_of(lane.prefix.taps.begin(), lane.prefix.taps.end(),
                       [&](const runtime::prefix_cache::PlannedTap& tap) {
                           return tap.position <= B ||
                                  (final_next &&
                                   tap.placement == runtime::prefix_cache::TapPlacement::Flexible);
                       });
}

ProgramImpl::LayerWalk ProgramImpl::walk_layout(const Lane& lane, std::size_t end) const {
    LayerWalk w;
    w.first_call       = lane.next_call;
    w.end_call         = end;
    std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens);
    for (std::size_t c = lane.next_call; c < end; ++c) {
        const auto width = static_cast<std::int32_t>(lane.calls[c]) - begin;
        w.begins.push_back(begin);
        w.widths.push_back(width);
        w.offsets.push_back(w.columns);
        begin += width;
        w.columns += width;
    }
    const std::size_t n = w.widths.size(), N = static_cast<std::size_t>(w.columns);
    const std::size_t H = c_.hidden_size, S = c_.hc.streams;
    std::size_t at  = 0;
    const auto take = [&](std::size_t bytes) {
        const std::size_t offset = at;
        at += walk_align(bytes);
        return offset;
    };
    w.residual  = take(2 * S * H * N);
    w.x         = take(2 * H * N);
    w.post      = take(4 * S * N);
    w.comb      = take(4 * S * S * N);
    w.ids       = take(4 * N);
    w.positions = take(4 * N);
    w.vision    = take(4 * N);
    w.meta      = take(4 * kMetaWords * n);
    if (drafter_) {
        const std::size_t rows = parameters_.model.config().dflash2->feature_width(c_.hidden_size);
        const std::size_t window = parameters_.model.config().dflash2->window;
        w.feature_stride         = walk_align(2 * rows * window);
        w.features               = take(w.feature_stride * n);
    }
    w.bytes = at;
    return w;
}

std::size_t ProgramImpl::walk_span_end(Lane& lane) {
    const std::size_t first = lane.next_call;
    if (!options_.prefill_walk || !expert_stream_ || residency_->frames() == 0 || walk_.active ||
        scoring_) {
        return first;
    }
    std::size_t end    = first;
    std::uint32_t from = lane.state_tokens, total = 0;
    while (end < lane.calls.size()) {
        const std::uint32_t width = lane.calls[end] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) ||
            total + width > kWalkMaxTokens) {
            break;
        }
        total += width;
        from = lane.calls[end];
        ++end;
        if (end < lane.calls.size() && from % prefix::cache_index::kBlockTokens != 0 &&
            tap_due(lane, from, end)) {
            break;
        }
    }
    if (end - first < 2) { return first; }
    if (!stream_chunk(static_cast<std::int32_t>(lane.calls[first] - lane.state_tokens))) {
        return first;
    }
    // The area comes out of the frames as well; a span too large for what may be lent shrinks.
    while (end - first >= 2 && !residency_->can_lend(walk_layout(lane, end).bytes)) { --end; }
    return end - first >= 2 ? end : first;
}

void ProgramImpl::walk_begin(Lane& lane, std::uint32_t index, std::size_t end) {
    const cudaStream_t s = device_.stream;
    walk_                = walk_layout(lane, end);
    walk_lease_          = residency_->lend(walk_.bytes, s, {});
    if (!walk_lease_.valid() || walk_lease_.memory.bytes < walk_.bytes) {
        if (walk_lease_.valid()) { residency_->give_back(walk_lease_, s); }
        walk_lease_ = {};
        walk_       = LayerWalk{};
        throw std::runtime_error("Glm5Next: the layer walk's area could not be lent");
    }
    walk_.lane      = index;
    walk_.host_slot = static_cast<std::int32_t>(index);
    walk_.active = true; // from here a failure abandons the span (walk_abandon gives the area back)
    try {
        walk_plan_taps(lane);
        walk_stage(lane, index);
    } catch (...) {
        walk_abandon();
        throw;
    }
    diagnostic("prefill layer walk: " + std::to_string(walk_.widths.size()) + " calls, " +
                   std::to_string(walk_.columns) + " tokens from position " +
                   std::to_string(walk_.begins[0]),
               DiagnosticLevel::Debug);
}

void ProgramImpl::walk_stage(Lane& lane, std::uint32_t index) {
    const cudaStream_t s = device_.stream;

    // The span's ids, positions and chunk words, staged once.
    const std::size_t n    = walk_.widths.size();
    const std::size_t host = walk_.meta + 4 * kMetaWords * n - walk_.ids;
    if (walk_host_.size() < host) { walk_host_ = PinnedHostBuffer(host); }
    auto* base      = static_cast<std::byte*>(walk_host_.data());
    auto* ids       = reinterpret_cast<std::int32_t*>(base);
    auto* positions = reinterpret_cast<std::int32_t*>(base + (walk_.positions - walk_.ids));
    auto* meta      = reinterpret_cast<std::int32_t*>(base + (walk_.meta - walk_.ids));
    for (std::int32_t i = 0; i < walk_.columns; ++i) {
        const std::int32_t p = walk_.begins[0] + i;
        ids[i]               = lane.history[static_cast<std::size_t>(p)];
        positions[i]         = p;
    }
    auto* vision = reinterpret_cast<std::int32_t*>(base + (walk_.vision - walk_.ids));
    walk_.vision_first.assign(n, 0);
    walk_.vision_count.assign(n, 0);
    for (std::size_t c = 0; c < n; ++c) {
        std::tie(walk_.vision_first[c], walk_.vision_count[c]) =
            vision_columns(lane, static_cast<std::uint32_t>(walk_.begins[c]),
                           static_cast<std::uint32_t>(walk_.widths[c]), vision + walk_.offsets[c]);
    }
    for (std::size_t c = 0; c < n; ++c) {
        meta[kMetaWords * c + 0] = static_cast<std::int32_t>(index);
        meta[kMetaWords * c + 1] = static_cast<std::int32_t>(index);
        meta[kMetaWords * c + 2] = walk_.widths[c] - 1;
        meta[kMetaWords * c + 3] = 0;
    }
    CUDA_CHECK(cudaMemcpyAsync(walk_area() + walk_.ids, base, host, cudaMemcpyHostToDevice, s));

    // The lane's pending copies (its restore, and the taps' capture of its state) land before the
    // span reads or writes that state.
    const auto restore_events = prefix_restore_wait(lane);
    if (!restore_events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(s, restore_events.back(), 0)); }
    const auto capture_events = prefix_capture_wait(lane);
    if (!capture_events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(s, capture_events.back(), 0)); }

    // One residency round for the span; its stream reads SSD-only experts once for every chunk.
    residency_->before_round(s, /*landing=*/false, /*walking=*/false, kStreamReadSlots);
    if (fetch_channel_ && tier_ && !tier_->round_open()) {
        throw std::logic_error("Glm5Next: a layer walk began outside the tier's round");
    }
    expert_stream_->begin(ring_span(), residency_->host_table(), s, /*ssd=*/true);
    set_expert_route();
}

void ProgramImpl::walk_plan_taps(Lane& lane) {
    const std::size_t n = walk_.widths.size();
    walk_.taps.assign(n, {});
    if (!prefix_ || !lane.prefix.reuse) { return; }
    // As prefix_after_prefill_call after each inner call of a chunk-major prefill.
    constexpr std::uint32_t kBlock = prefix::cache_index::kBlockTokens;
    std::uint32_t deepest          = lane.prefix.deepest;
    for (std::size_t c = 0; c + 1 < n; ++c) {
        const std::uint32_t B = lane.calls[walk_.first_call + c];
        bool boundary         = false;
        if (!prefix_take_taps(lane, B, walk_.first_call + c + 1, boundary) ||
            B < deepest + kBlock) {
            continue;
        }
        const auto staged =
            prefix_->open_staged(prefix_->index().estimate_priority(deepest, B, false));
        if (!staged) { continue; }
        walk_.taps[c] = {.frontier = B,
                         .kind     = boundary ? prefix::cache_index::SnapshotKind::Boundary
                                              : prefix::cache_index::SnapshotKind::Tap,
                         .staged   = *staged};
        deepest       = B;
    }
}

void ProgramImpl::walk_commit_taps(Lane& lane) {
    for (LayerWalk::Tap& tap : walk_.taps) {
        if (tap.staged == 0) { continue; }
        const std::uint64_t staged = std::exchange(tap.staged, 0);
        // The blocks up to the tap first, as after its chunk-major call (its anchor is the last).
        prefix_publish(lane, tap.frontier);
        if (lane.prefix.path.size() < tap.frontier / prefix::cache_index::kBlockTokens) {
            prefix_->abandon_staged(staged);
            continue;
        }
        prefix::PrefixCache::Capture capture =
            prefix_capture_at(lane, tap.frontier, tap.kind, false);
        const std::uint64_t id = capture.id;
        prefix_->commit_staged(staged, std::move(capture));
        prefix_captured(lane, id, tap.frontier);
    }
}

execution::ForwardBatch ProgramImpl::walk_batch(std::size_t c) const {
    std::byte* area      = walk_area();
    const std::int32_t w = walk_.widths[c];
    const auto offset    = static_cast<std::size_t>(walk_.offsets[c]);
    std::byte* meta      = area + walk_.meta + 4 * kMetaWords * c;
    execution::ForwardBatch fb;
    fb.ids           = Tensor(area + walk_.ids + 4 * offset, DType::I32, {w});
    fb.positions     = Tensor(area + walk_.positions + 4 * offset, DType::I32, {w});
    fb.slots         = Tensor(meta, DType::I32, {1});
    fb.table_rows    = Tensor(meta + 4, DType::I32, {1});
    fb.logit_columns = Tensor(meta + 8, DType::I32, {1});
    fb.host_slots    = std::span<const std::int32_t>(&walk_.host_slot, 1);
    fb.batch         = 1;
    fb.width         = w;
    vision_batch(lanes_[walk_.lane], walk_.vision_first[c], walk_.vision_count[c],
                 area + walk_.vision + 4 * offset, fb);
    if (drafter_) {
        const std::int32_t rows =
            dim(parameters_.model.config().dflash2->feature_width(c_.hidden_size));
        const std::int32_t window = dim(parameters_.model.config().dflash2->window);
        fb.draft_features = Tensor(area + walk_.features + c * walk_.feature_stride, DType::BF16,
                                   {rows, std::min(w, window)});
    }
    // The prompt's last call keeps its final residual (the zero-token resume's head input).
    if (c + 1 == walk_.widths.size() && walk_.end_call == lanes_[walk_.lane].calls.size()) {
        fb.head_residual_out = Tensor(static_cast<std::byte*>(head_saved_.p) +
                                          static_cast<std::size_t>(walk_.lane) * plan_.head_saved,
                                      DType::BF16, {width_, 1});
    }
    return fb;
}

execution::Forward::CallColumns ProgramImpl::walk_columns(std::size_t c) const {
    std::byte* area      = walk_area();
    const std::int32_t w = walk_.widths[c];
    const auto offset    = static_cast<std::size_t>(walk_.offsets[c]);
    const std::size_t H = c_.hidden_size, S = c_.hc.streams;
    return {.residual =
                Tensor(area + walk_.residual + 2 * S * H * offset, DType::BF16, {dim(S * H), w}),
            .x    = Tensor(area + walk_.x + 2 * H * offset, DType::BF16, {dim(H), w}),
            .post = Tensor(area + walk_.post + 4 * S * offset, DType::FP32, {dim(S), w}),
            .comb =
                Tensor(area + walk_.comb + 4 * S * S * offset, DType::FP32, {dim(S), dim(S), w})};
}

PrefillProgress ProgramImpl::walk_step(Lane& lane, std::uint32_t index, Clock::time_point start) {
    const cudaStream_t s  = device_.stream;
    const std::uint32_t L = c_.num_hidden_layers;
    const std::size_t n   = walk_.widths.size();
    if (walk_.steps > 0) {
        // A step boundary with the device idle: the tier closes its round (or another lane's round
        // already closed it) and opens the next, as the fetch channel needs an open round.
        device_.synchronize();
        residency_->tier_step(s);
    }
    // About one call's work per step; beside another lane's decode, about the call width a
    // prompt is split to there.
    bool decoding = false;
    for (std::uint32_t other = 0; other < options_.max_concurrency; ++other) {
        decoding = decoding || (other != index && lanes_[other].phase == Phase::Decode);
    }
    const std::uint32_t step_tokens = decoding && beside_decode_chunk_ > 0
                                          ? std::min<std::uint32_t>(beside_decode_chunk_, chunk_)
                                          : chunk_;
    const auto per_step             = static_cast<std::uint32_t>(std::max<std::uint64_t>(
        1, (static_cast<std::uint64_t>(L) * step_tokens + walk_.columns - 1) / walk_.columns));
    const std::uint32_t to          = std::min(L, walk_.next_layer + per_step);
    const bool last                 = walk_.end_call == lane.calls.size();
    try {
        for (std::uint32_t l = walk_.next_layer; l < to; ++l) {
            for (std::size_t c = 0; c < n; ++c) {
                const execution::ForwardBatch fb        = walk_batch(c);
                execution::Forward::CallColumns columns = walk_columns(c);
                forward_->begin_call(fb);
                if (l == 0) { forward_->embed(fb, columns); }
                // The previous chunk's tap copies this layer's state before this chunk writes it.
                if (c > 0 && walk_.taps[c - 1].staged != 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(
                        s, prefix_->staged_event(walk_.taps[c - 1].staged, l), 0));
                }
                forward_->step(l, fb, columns, {.first = c == 0, .last = c + 1 == n});
                if (walk_.taps[c].staged != 0) {
                    prefix_->capture_staged_group(walk_.taps[c].staged, l, index, s);
                }
                if (l + 1 < L) { continue; }
                const bool logits_here = last && c + 1 == n;
                Tensor logits(logits32_.p, DType::FP32, {vocab_, 1});
                forward_->finish(fb, columns, logits_here ? &logits : nullptr);
                if (logits_here) { round_logits(1); }
                if (drafter_) {
                    // The chunk's features, as its chunk-major call leaves them for the drafter
                    // (the context's positions are read from pinned memory in stream order: the
                    // previous chunk's must have been read before they are rewritten).
                    if (c > 0) { device_.synchronize(); }
                    if (c > 0 && walk_.taps[c - 1].staged != 0) {
                        CUDA_CHECK(cudaStreamWaitEvent(
                            s, prefix_->staged_event(walk_.taps[c - 1].staged, L), 0));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(features_.p, fb.draft_features.data,
                                               fb.draft_features.bytes(), cudaMemcpyDeviceToDevice,
                                               s));
                    call_context(index, walk_.begins[c], walk_.widths[c]);
                }
                // The tap's drafter group: the context ring once this chunk's context is in it.
                if (walk_.taps[c].staged != 0) {
                    prefix_->capture_staged_group(walk_.taps[c].staged, L, index, s);
                }
            }
        }
    } catch (...) {
        walk_abandon();
        throw;
    }
    walk_.next_layer = to;
    ++walk_.steps;
    if (to < L) {
        device_.synchronize();
        check_device_error();
        PrefillProgress out;
        out.summary                 = lane.begin;
        out.processed_prompt_tokens = 0;
        out.complete                = false;
        out.timing.submit_host_ns   = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        return out;
    }
    // The span is enqueued: the lane's state is at its end in every layer. The route log holds the
    // span's last chunk (every chunk routes nearly every expert).
    const std::int32_t begin = walk_.begins.back(), width = walk_.widths.back();
    const auto tokens = static_cast<std::uint32_t>(walk_.columns);
    enqueue_routes(width);
    expert_stream_->end();
    residency_->give_back(walk_lease_, s);
    walk_lease_       = {};
    lane.state_tokens = lane.calls[walk_.end_call - 1];
    lane.next_call    = walk_.end_call;
    walk_commit_taps(lane);
    walk_ = LayerWalk{};
    if (last && !prefilling_besides(index)) { return_stream_lease(); }
    prefix_after_prefill_call(lane, index, last);
    return after_prefill(lane, index, begin, width, tokens, last, 0, start);
}

void ProgramImpl::walk_abandon() noexcept {
    if (!walk_.active) { return; }
    try {
        device_.synchronize();
    } catch (...) {}
    expert_stream_->end();
    for (const LayerWalk::Tap& tap : walk_.taps) {
        if (tap.staged != 0) { prefix_->abandon_staged(tap.staged); }
    }
    if (walk_lease_.valid()) {
        try {
            residency_->give_back(walk_lease_, device_.stream);
        } catch (...) {}
    }
    walk_lease_ = {};
    walk_       = LayerWalk{};
}

} // namespace infernix::models::glm5_next::detail
