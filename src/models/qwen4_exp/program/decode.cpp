// Prefill, decode rounds (plain and mixed), commit, finish and abort, the round settlement and
// the decode budget.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks) {
    const GrammarScope grammar(*this, masks);
    const auto start  = Clock::now();
    const auto index  = lane_of(sequence);
    Lane& lane        = lanes_[index];
    if (lane.phase != Phase::Prefill) { throw std::logic_error("Qwen4Exp: prefill on a lane that is not prefilling"); }
    if (walk_.active) {
        if (walk_.lane != index) { throw std::logic_error("Qwen4Exp: another lane's prefill holds the layer walk"); }
        return walk_step(lane, index, start);
    }
    // The zero-token resume: the plan is empty because the prompt's own end snapshot holds the
    // lane's whole state; the first token regenerates from its saved head residual, with no
    // prefill call at all.
    if (lane.calls.empty()) { return resume_from_saved_head(lane, index, start); }
    const std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens);
    if (lane.next_call >= lane.calls.size() || lane.calls[lane.next_call] <= lane.state_tokens) {
        throw std::logic_error(
            "Qwen4Exp: the lane's prefill plan does not continue its state (lane " + std::to_string(index) +
            ", call " + std::to_string(lane.next_call) + " of " + std::to_string(lane.calls.size()) +
            (lane.next_call < lane.calls.size() ? ", ending at " + std::to_string(lane.calls[lane.next_call]) : "") +
            ", state " + std::to_string(lane.state_tokens) + ", prompt end " + std::to_string(lane.prefill_end) + ")");
    }
    if (const std::size_t end = walk_span_end(lane, index); end > lane.next_call) {
        walk_begin(lane, index, end);
        return walk_step(lane, index, start);
    }
    split_beside_decode(lane, index);
    const std::int32_t width = static_cast<std::int32_t>(lane.calls[lane.next_call++] - lane.state_tokens);
    const bool last          = begin + width == static_cast<std::int32_t>(lane.prefill_end);
    // A call the CPU serves (an opener tail, a tiny suffix) promotes at the decode rate (design
    // §13.7). Wider prefill calls promote nothing: the answer's staged misses land in free
    // frames and pick its experts, while the prompt's would displace them (16 per layer per
    // call measured slower than none in every regime: serve C = 4 fill -4.3 %, warm -5.6 %;
    // design §13.7). Their routes still count toward the cache's scores.
    const std::size_t promotions =
        width <= kServedCallColumns ? decode_budget(static_cast<std::uint32_t>(width)) : 0;
    // Host phases of a chunk for nsys attribution (the GPU side is in the CUDA trace).
    const nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                        static_cast<std::uint64_t>(width));
    // A call wide enough to stream reads its n-gram rows behind the gate once it is enqueued
    // (run_rows_behind); the others, and a call the stream turns down, read them first.
    // Upstream measured this on FP4-Local from a volume (RTX 5090, 2026-10-30, three interleaved
    // runs): pp16413 6,829 -> 6,999 tok/s, pp4096 2,727 -> 2,802.
    const bool rows_behind = static_cast<bool>(expert_stream_) && width >= kStreamMinColumns;
    {
        const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(rows_behind ? 0 : width));
        stage_sequence(index, begin, width, lane.history, !rows_behind);
    }
    const auto chunk = stage_mtp_chunk(lane, index, begin, width);
    // A resumed lane's first call waits per layer for its restore; any call waits per layer for
    // a copy-out of this lane's state still in flight (design §15.1).
    next_waits_ = prefix_waits(lane);
    const bool streamed = !walk_.active && stream_chunk(width);
    score_column_       = begin;
    const execution::VisionInput* vision = stage_vision(lane, static_cast<std::uint32_t>(begin), width);
    if (rows_behind && streamed) {
        run_rows_behind(lane, begin, width, chunk ? &*chunk : nullptr, vision);
    } else {
        if (rows_behind) {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(width));
            stage_ngram(lane, lane.history, begin, width, 0);
        }
        run(1, width, 1, chunk ? &*chunk : nullptr, vision, streamed);
    }
    if (last && !prefilling_besides(index)) {
        return_stream_lease();
        return_wide_work();
    }
    lane.state_tokens += static_cast<std::uint32_t>(width);
    if (dflash2_) { lane.dflash_frontier = lane.state_tokens; }
    prefix_after_prefill_call(lane, index, last);
    // The handoff is read only by the prompt's calls: it returns once the last is enqueued.
    if (last && lane.vision) { vision_release(lane); }
    if (!last) {
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            device_.synchronize();
        }
        check_device_error();
        const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                static_cast<std::uint64_t>(width));
        trace_round(offload::RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                    static_cast<std::uint32_t>(begin));
        note_prompt_routes(lane, width);
        residency_->after_round(device_.stream, width, promotions);
        apply_vram_target(false);
    }

    return prefill_progress(lane, index, begin, width, static_cast<std::uint32_t>(width), last, promotions, start);
}

void ProgramImpl::split_beside_decode(Lane& lane, std::uint32_t index) {
    const std::uint32_t cap = std::min<std::uint32_t>(beside_decode_chunk_, static_cast<std::uint32_t>(chunk_));
    if (cap == 0 || lane.calls[lane.next_call] - lane.state_tokens <= cap) { return; }
    bool decoding = false;
    for (std::uint32_t other = 0; other < options_.max_concurrency; ++other) {
        decoding = decoding || (other != index && lanes_[other].phase == Phase::Decode);
    }
    if (!decoding) { return; }
    const std::uint32_t split = (lane.state_tokens + cap) / kResumeGrid * kResumeGrid;
    if (split <= lane.state_tokens) { return; }
    lane.calls.insert(lane.calls.begin() + static_cast<std::ptrdiff_t>(lane.next_call), split);
}

std::int32_t ProgramImpl::mixable_width(Lane& lane, std::uint32_t index) {
    if (!options_.mixed_calls || mtp_ || scoring_ || walk_.active || lane.phase != Phase::Prefill || lane.replay ||
        lane.vision || lane.calls.empty() || lane.next_call >= lane.calls.size() ||
        lane.calls[lane.next_call] <= lane.state_tokens) {
        return 0;
    }
    if (walk_span_end(lane, index) > lane.next_call) { return 0; }
    const std::int32_t room = chunk_ - static_cast<std::int32_t>(options_.max_concurrency) * max_width_;
    const std::uint32_t cap = static_cast<std::uint32_t>(std::max<std::int32_t>(
        0, std::min<std::int32_t>(beside_decode_chunk_ != 0 ? static_cast<std::int32_t>(beside_decode_chunk_) : chunk_,
                                  room)));
    std::uint32_t end = lane.calls[lane.next_call];
    if (end - lane.state_tokens > cap) { end = (lane.state_tokens + cap) / kResumeGrid * kResumeGrid; }
    if (end <= lane.state_tokens || end >= lane.prefill_end) { return 0; }
    return static_cast<std::int32_t>(end - lane.state_tokens);
}

bool ProgramImpl::can_mix(SequenceHandle prefill) {
    const auto index = lane_of(prefill);
    return mixable_width(lanes_[index], index) > 0;
}

MixedProgress ProgramImpl::mixed(SequenceHandle prefill, std::span<const SequenceHandle> sequences,
                                 std::span<const runtime::RoundBudget> budgets, runtime::TokenMaskProvider* masks) {
    const auto start  = Clock::now();
    const auto index  = lane_of(prefill);
    Lane& lane        = lanes_[index];
    const std::int32_t width = mixable_width(lane, index);
    if (width == 0) { throw std::logic_error("Qwen4Exp: a mixed round's prompt call cannot ride in it"); }
    for (const SequenceHandle& sequence : sequences) {
        if (lane_of(sequence) == index) { throw std::logic_error("Qwen4Exp: a mixed round decodes its prefilling lane"); }
    }
    const auto begin = lane.state_tokens;
    if (lane.calls[lane.next_call] != begin + static_cast<std::uint32_t>(width)) {
        lane.calls.insert(lane.calls.begin() + static_cast<std::ptrdiff_t>(lane.next_call),
                          begin + static_cast<std::uint32_t>(width));
    }
    ++lane.next_call;
    const nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                        static_cast<std::uint64_t>(width));
    // The ring is lent before the round's before_round publishes the tables its frames left.
    mixed_ = {.active   = true,
              .lane     = index,
              .slot     = static_cast<std::int32_t>(index),
              .begin    = static_cast<std::int32_t>(begin),
              .width    = width,
              .streamed = stream_chunk(width)};
    struct Close {
        MixedPrefill& m;
        ~Close() { m.active = false; }
    } close{mixed_};
    PendingBatch pending = decode(sequences, budgets, masks); // synchronized: its acceptance or draw
    check_device_error();
    lane.state_tokens += static_cast<std::uint32_t>(width);
    if (dflash2_) { lane.dflash_frontier = lane.state_tokens; }
    prefix_after_prefill_call(lane, index, false);
    // The round's settle credits the call's routes with the decode rows' (one residency round);
    // the prompt's seeding counts its columns, which follow the decode rows'.
    note_prompt_routes(lane, width, deferred_.columns);
    deferred_.prefill_columns = width;
    // The call's time is no decode round's cost sample.
    last_round_steps_ = -1;
    last_round_extra_ = -1;
    return MixedProgress{
        .prefill = prefill_progress(lane, index, static_cast<std::int32_t>(begin), width,
                                    static_cast<std::uint32_t>(width), false, 0, start),
        .pending = std::move(pending)};
}

void ProgramImpl::stage_mixed_prefill(std::int32_t rows, std::int32_t decode_columns) {
    Lane& lane                  = lanes_[mixed_.lane];
    const std::int32_t D        = decode_columns, P = mixed_.width;
    auto* ids                   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
    auto* positions             = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
    for (std::int32_t i = 0; i < P; ++i) {
        ids[D + i]       = lane.history[static_cast<std::size_t>(mixed_.begin + i)];
        positions[D + i] = mixed_.begin + i;
    }
    const auto first = static_cast<std::uint32_t>(mixed_.begin);
    stage_rope(lane.rope, first, P, io_words(io_layout_.rope + 12ULL * D, 3ULL * P), P, 0);
    stage_rope_value(block_start_rope(lane.rope, first, static_cast<std::uint32_t>(r_)),
                     io_words(io_layout_.block_rope + 12ULL * rows, 3), 1, 0);
    reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots)[rows] = mixed_.slot;
    reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows)[rows]  = mixed_.slot;
    stage_ngram(lane, lane.history, mixed_.begin, P, static_cast<std::size_t>(D));
}

void ProgramImpl::run_mixed(std::int32_t batch, std::int32_t width, const execution::ForwardVerify* verify) {
    const cudaStream_t s = device_.stream;
    Lane& lane           = lanes_[mixed_.lane];
    const std::int32_t D = batch * width, P = mixed_.width, T = D + P;
    const bool streamed  = mixed_.streamed;
    split_call_          = streamed && split_policy_ && T <= split_columns();
    if (streamed) { expert_stream_->begin(ring_span(), residency_->host_table(), s, split_call_); }
    try {
        const WideWork wide(*this, T);
        require_tier_round();
        set_expert_route();
        auto* base                  = static_cast<std::byte*>(io_device_.p);
        execution::ForwardBatch fb  = io_batch(base, batch, width, D);
        fb.ids                      = Tensor(base + io_layout_.ids, DType::I32, {T});
        fb.positions                = Tensor(base + io_layout_.positions, DType::I32, {T});
        fb.ngram_rows               = Tensor(base + io_layout_.ngram, DType::U8,
                                             {dim(ngram_table().row_bytes), dim(hash_.heads()), T});
        fb.verify                   = verify;
        fb.layer_waits              = prefix_waits(lane);
        fb.stream                   = streamed;
        fb.split                    = split_call_;
        execution::ForwardPrefillPart part{
            .width            = P,
            .rope_positions   = Tensor(base + io_layout_.rope + 12ULL * D, DType::I32, {P, 3}),
            .block_start_rope = Tensor(base + io_layout_.block_rope + 12ULL * batch, DType::I32, {1, 3}),
            .slots            = Tensor(base + io_layout_.slots + 4ULL * batch, DType::I32, {1}),
            .table_rows       = Tensor(base + io_layout_.rows + 4ULL * batch, DType::I32, {1}),
            .host_slots       = std::span<const std::int32_t>(&mixed_.slot, 1)};
        if (dflash2_) {
            fb.draft_features      = dflash_feature_columns(0, D);
            const auto [column, n] = dflash_call_region(P);
            part.draft_features    = dflash_feature_columns(column, n);
        }
        if (head_saved_.p != nullptr) { fb.head_residual_out = head_saved_column(mixed_.lane); }
        fb.prefill_part = &part;
        Tensor logits = head_logits(D);
        forward_->run(fb, logits);
        round_logits(D, device_.stream);
        if (dflash2_) {
            // A restore into the lane's ring or a copy-out of it lands first (dflash2_call_context).
            dflash2_wait_copies(lane);
            const auto [column, n] = dflash_call_region(P);
            dflash2_context(column, n, mixed_.begin + P - n, mixed_.slot);
        }
    } catch (...) {
        if (streamed) { expert_stream_->end(); }
        split_call_ = false;
        throw;
    }
    if (streamed) { expert_stream_->end(); }
    split_call_ = false;
    residency_->enqueue_route_download(s, T);
}

void ProgramImpl::note_prompt_routes(Lane& lane, std::int32_t width, std::int32_t first) {
    if (width <= 0) { return; }
    const std::uint32_t L = c_.num_hidden_layers, E = c_.moe.experts, K = c_.moe.top_k;
    if (lane.prompt_uses.size() != static_cast<std::size_t>(L) * E) {
        lane.prompt_uses.assign(static_cast<std::size_t>(L) * E, 0);
        lane.prompt_counted = 0;
    }
    const std::int32_t* routes = residency_->route_host();
    const std::size_t stride   = residency_->route_stride();
    for (std::uint32_t l = 0; l < L; ++l) {
        const std::int32_t* ids = routes + static_cast<std::size_t>(l) * stride + static_cast<std::size_t>(first) * K;
        std::uint32_t* uses     = lane.prompt_uses.data() + static_cast<std::size_t>(l) * E;
        for (std::size_t i = 0; i < static_cast<std::size_t>(width) * K; ++i) {
            const std::int32_t expert = ids[i];
            if (expert >= 0 && static_cast<std::uint32_t>(expert) < E) { ++uses[expert]; }
        }
    }
    lane.prompt_counted += static_cast<std::uint32_t>(width);
}

void ProgramImpl::seed_prompt_experts(Lane& lane) {
    if (lane.prompt_counted == 0) { return; }
    residency_->seed_prompt(lane.prompt_uses, kPromptSeedCredit / lane.prompt_counted);
    clear_prompt_uses(lane);
}

void ProgramImpl::clear_prompt_uses(Lane& lane) {
    std::fill(lane.prompt_uses.begin(), lane.prompt_uses.end(), 0U);
    lane.prompt_counted = 0;
}

PrefillProgress ProgramImpl::resume_from_saved_head(Lane& lane, std::uint32_t index, Clock::time_point start) {
    if (lane.phase != Phase::Prefill || !lane.calls.empty() ||
        lane.state_tokens != lane.prefill_end || lane.state_tokens != lane.prompt_tokens ||
        lane.state_tokens == 0 || !lane.prefix.head_resume) {
        throw std::logic_error(
            "Qwen4Exp: a zero-token resume on a lane that is not at its prompt's end snapshot");
    }
    if (lane.prefix.restore) { prefix_->order_after_restore(*lane.prefix.restore, device_.stream); }
    // Under MTP the drafter's saved column is the same final residual (finish copies column
    // T - 1 into both), so the image keeps it once.
    Tensor residual = mtp_ ? saved_column(index) : head_saved_column(index);
    Tensor logits   = head_logits(1);
    forward_->head_logits(residual, logits);
    round_logits(1, device_.stream);
    const std::uint32_t rows[] = {index};
    const std::int32_t positions[] = {static_cast<std::int32_t>(lane.prompt_tokens) - 1};
    // The readout reads the logits before the sampler's constraints mask them.
    std::optional<PromptReadout> readout;
    if (!lane.readout.empty()) { readout = read_prompt_frontier(lane); }
    sample(rows, positions);
    // No call reads the encode window (the snapshot's KV holds the image columns); the bind
    // completed only once the encode did, so the handoff returns here, as after a last call.
    if (lane.vision) { vision_release(lane); }
    PrefillProgress out;
    out.summary                 = prefill_summary(lane);
    out.processed_prompt_tokens = 0;
    out.complete                = true;
    if (readout) { out.readout = std::move(*readout); }
    const SequenceHandle handles[] = {handle(index)};
    out.timing.submit_host_ns = elapsed_ns(start);
    lane.prefill_ns += out.timing.submit_host_ns;
    out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, handles,
                                                     {pending_tokens_.data(), 1}, out.timing));
    mark_row_failures(*out.pending, {});
    plain_round_ = false;
    pending_transaction_ = next_transaction_;
    return out;
}

PrefillProgress ProgramImpl::prefill_progress(Lane& lane, std::uint32_t index, std::int32_t begin, std::int32_t width,
                                              std::uint32_t tokens, bool last, std::size_t promotions,
                                              Clock::time_point start) {
    PrefillProgress out;
    out.summary                 = prefill_summary(lane);
    out.processed_prompt_tokens = tokens;
    out.complete                = last;
    out.capacity_released       = !last && leases_returnable();
    if (last && lane.replay) {
        // A replay ends at its paused frontier, whose next input the ledger already holds: the
        // lane decodes from there (design §15.7).
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            device_.synchronize();
        }
        trace_round(offload::RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                    static_cast<std::uint32_t>(begin));
        note_prompt_routes(lane, width);
        residency_->after_round(device_.stream, width, promotions);
        seed_prompt_experts(lane);
        apply_vram_target(false);
        lane.replay = false;
        lane.phase  = Phase::Decode;
        out.timing.submit_host_ns = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        return out;
    }
    if (last) {
        const std::uint32_t lanes[] = {index};
        const std::int32_t positions[] = {begin + width - 1};
        // The readout reads the logits before the sampler's constraints mask them.
        if (!lane.readout.empty()) { out.readout = read_prompt_frontier(lane); }
        sample(lanes, positions);
        {
            const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                    static_cast<std::uint64_t>(width));
            trace_round(offload::RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                        static_cast<std::uint32_t>(begin));
            note_prompt_routes(lane, width);
            residency_->after_round(device_.stream, width, promotions);
            seed_prompt_experts(lane);
            apply_vram_target(false);
        }
        const SequenceHandle rows[] = {handle(index)};
        out.timing.submit_host_ns = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, rows,
                                                         {pending_tokens_.data(), 1}, out.timing));
        mark_row_failures(*out.pending, {});
        plain_round_ = false;
        pending_transaction_ = next_transaction_;
    } else {
        out.timing.submit_host_ns = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
    }
    return out;
}

PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> sequences,
                                 std::span<const runtime::RoundBudget> budgets, runtime::TokenMaskProvider* masks) {
    const GrammarScope grammar(*this, masks);
    const auto start = Clock::now();
    const auto batch = static_cast<std::int32_t>(sequences.size());
    if (batch <= 0 || batch > static_cast<std::int32_t>(options_.max_concurrency)) {
        throw std::logic_error("Qwen4Exp: decode batch is empty or too large");
    }
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::array<std::int32_t, kMaximumConcurrency> positions{};
    for (std::int32_t b = 0; b < batch; ++b) {
        lanes[b]   = lane_of(sequences[b]);
        Lane& lane = lanes_[lanes[b]];
        if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
            throw std::logic_error("Qwen4Exp: decode row is not at a committed frontier");
        }
        positions[b] = static_cast<std::int32_t>(lane.state_tokens);
    }
    // Proposals: a round verifies 1 + the longest row's drafts; a row may draft at most one token
    // fewer than it may still emit, so every verified position lies in its reservation. A longer
    // n-gram copy proposal replaces a row's MTP drafts. By round size (design 11.4, 11.5):
    // - one row: its own draft length, maximizing its expected tokens per round time;
    // - two MTP rows, with the adaptive-speculation fast path: one draft length for both
    //   (choose_pair_draft_length), so no row is padded to the other's; its at most 2 x 3 = 6
    //   columns stay on the column-invariant dense routes, so greedy output equals C = 1, and
    //   below the 7-column fork and CPU-job limits; the shared length couples the rows, and a
    //   pair round's output differed from the rows alone in 1-4 of 96 segments (2026-10-14);
    // - DFlash2 rows of any size, with the fast path: one shared length for the round
    //   (choose_batch_dflash_length). The drafter proposes every row's block in one pass, so
    //   the proposal does not grow with the round, and the policy reads the size's measured
    //   round times (batch_cost_), so it stays off by itself where wide rounds lose -- the
    //   offloaded setups the priors were measured on. A block past 8 columns takes the MMA
    //   dense routes, so a greedy row may flip at a near-tie against decoding alone, as a pair
    //   round's can (the exact route disables the fast path and decodes plain);
    // - otherwise (MTP at three or more rows) rounds decode plain (one draft each measured
    //   slower at four rows).
    // A plain round's W = 1 catch-up keeps each drafter current.
    const std::int32_t pair_k =
        mtp_ && batch == 2 && options_.pair_speculation
            ? choose_pair_draft_length(lanes_[lanes[0]], lanes_[lanes[1]])
            : 0;
    const std::int32_t batch_k =
        dflash2_ && batch > 1 && options_.pair_speculation
            ? choose_batch_dflash_length(std::span<const std::uint32_t>(lanes.data(), static_cast<std::size_t>(batch)))
            : 0;
    // The round's shared multi-row draft length (an MTP pair's or a DFlash2 batch's).
    const std::int32_t shared_k = mtp_ ? pair_k : batch_k;
    const bool speculate = batch == 1 || shared_k > 0;
    std::array<std::int32_t, kMaximumConcurrency> wanted{};
    std::int32_t steps = 0;
    if (mtp_ || dflash2_) {
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            wanted[b]  = speculate && drafter_live(lane)
                         ? (batch == 1 ? choose_draft_length(lane) : shared_k)
                         : 0;
            steps      = std::max(steps, wanted[b]);
        }
    }
    observe_round(start, batch);
    // Each row's draft count and source are fixed before the drafter runs (an MTP row's count
    // is its draft length; a longer n-gram copy proposal replaces its drafts), so the round's
    // width, and with it every row's n-gram landing columns, is known while it drafts.
    std::array<std::int32_t, kMaximumConcurrency> extents{}, limits{};
    std::int32_t width = 1;
    for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
        Lane& lane = lanes_[lanes[b]];
        drafts_[b].clear();
        from_ngram_[b] = false;
        const std::uint32_t remaining =
            static_cast<std::size_t>(b) < budgets.size() ? budgets[b].generated_tokens_remaining : 0U;
        if (remaining < 2) { continue; }
        auto limit = std::min<std::uint32_t>(static_cast<std::uint32_t>(max_width_ - 1), remaining - 1U);
        if (batch > 1) { limit = std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(shared_k)); }
        limits[b] = static_cast<std::int32_t>(limit);
        if (drafter_live(lane) && wanted[b] > 0) {
            extents[b] = static_cast<std::int32_t>(std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(wanted[b])));
        }
        ngram_bucket_[b] = -1;
        if (speculate && lane.proposer) {
            auto match = lane.proposer->propose(lane.history, limit, options_.ngram_min_match);
            const int bucket = ngram_bucket(match.matched);
            // A longer copy proposal replaces the row's MTP drafts; a one-row round takes it only when
            // its expected tokens per round time (its bucket's learned acceptance per position) beat the
            // MTP drafts' (Strata's DraftPolicy).
            bool take = match.tokens.size() > static_cast<std::size_t>(extents[b]);
            if (take && batch == 1) {
                const double cost = width_cost();
                double reach = 1.0, mtp_tokens = 1.0;
                for (std::int32_t k = 0; k < extents[b]; ++k) {
                    reach *= lane.mtp_accept[static_cast<std::size_t>(k)];
                    mtp_tokens += reach;
                }
                const double p = lane.ngram_accept[static_cast<std::size_t>(bucket)];
                double copy_reach = 1.0, copy_tokens = 1.0;
                for (std::size_t k = 0; k < match.tokens.size(); ++k) {
                    copy_reach *= p;
                    copy_tokens += copy_reach;
                }
                take = copy_tokens / (1.0 + cost * static_cast<double>(match.tokens.size())) >
                       mtp_tokens / (1.0 + cost * extents[b]);
            }
            if (take) {
                drafts_[b]       = std::move(match.tokens);
                from_ngram_[b]   = true;
                ngram_bucket_[b] = bucket;
                extents[b]       = static_cast<std::int32_t>(drafts_[b].size());
            }
        }
        width = std::max(width, 1 + extents[b]);
    }
    if (batch == 1 && mtp_ && !from_ngram_[0]) { last_round_steps_ = steps; }
    // Columns of each verification row whose n-gram rows are already staged: the drafter's
    // wait reads every committed-token column (the anchor's) and all of a copy row's.
    std::array<std::int32_t, kMaximumConcurrency> staged{};
    if (mtp_) {
        mtp_draft(std::span<const std::uint32_t>(lanes.data(), batch), steps, [&] {
            if (width == 1) { return; }
            for (std::int32_t b = 0; b < batch; ++b) {
                staged[b] = from_ngram_[b] ? width : 1;
                queue_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], width, 0, staged[b]);
            }
            read_queued_ngram();
        });
        for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
            if (!from_ngram_[b] && extents[b] > 0) {
                drafts_[b].assign(mtp_drafts_[b].begin(), mtp_drafts_[b].begin() + extents[b]);
            }
        }
        // A one-row round verifies only its confident drafts: those before the first whose
        // draft-head probability is below --spec-min-p (draft_min_logprob_). Its width may
        // still shrink, as the drafter staged only column 0's n-gram rows; a pair round's
        // columns are staged at their offsets, so it keeps its length.
        if (batch == 1 && !from_ngram_[0] && extents[0] > 0) {
            extents[0] = std::min(extents[0], mtp_confident_[0]);
            drafts_[0].resize(static_cast<std::size_t>(extents[0]));
            width = 1 + extents[0];
        }
    } else if (dflash2_ && width > 1) {
        // One proposal for every row that takes the drafter's tokens (none when every speculating
        // row copies an n-gram), while the host stages the verification's n-gram rows.
        bool neural = false;
        for (std::int32_t b = 0; b < batch; ++b) { neural = neural || (!from_ngram_[b] && extents[b] > 0); }
        // While the proposal drafts, the host settles the previous round's cache update (DeferredRound), then stages
        // the verification's n-gram rows.
        const auto stage_rows = [&] {
            settle_round();
            for (std::int32_t b = 0; b < batch; ++b) {
                staged[b] = from_ngram_[b] ? width : 1;
                queue_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], width, 0, staged[b]);
            }
            read_queued_ngram();
        };
        if (!neural) { settle_round(); }
        if (neural) {
            dflash2_propose(std::span<const std::uint32_t>(lanes.data(), batch), stage_rows);
            for (std::int32_t b = 0; b < batch; ++b) {
                if (!from_ngram_[b] && extents[b] > 0) {
                    drafts_[b].assign(dflash_drafts_[b].begin(), dflash_drafts_[b].begin() + extents[b]);
                }
            }
            // A one-row round verifies its drafts up to the first whose selector probability is
            // below --spec-min-p, and with --draft-tokens auto only the calibrated length of them
            // (draft_length.h). The proposal staged only column 0's n-gram rows.
            if (batch == 1 && !from_ngram_[0] && extents[0] > 0) {
                extents[0] = std::min(extents[0], dflash_confident_[0]);
                if (options_.auto_draft_length) { extents[0] = dflash2_auto_length(lanes_[lanes[0]], extents[0]); }
                drafts_[0].resize(static_cast<std::size_t>(extents[0]));
                width = 1 + extents[0];
            }
        }
        dflash_round_ = neural && width > 1;
    }
    if (width > 1) {
        if (dflash2_) { last_round_extra_ = batch * (width - 1); } // the batched cost fit's sample
        return verify(sequences, std::span<const std::uint32_t>(lanes.data(), batch),
                      std::span<const std::int32_t>(positions.data(), batch), width,
                      std::span<const std::int32_t>(staged.data(), batch), start);
    }
    for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].speculative.fallback_steps; }
    if (dflash2_) { last_round_extra_ = 0; } // a plain round: the size's zero-column sample
    round_width_ = 1;
    plain_round_ = true;
    const std::span<const std::uint32_t> round_lanes(lanes.data(), static_cast<std::size_t>(batch));
    const std::span<const std::int32_t> round_positions(positions.data(), static_cast<std::size_t>(batch));
    // A replayed round reads its rows after the launch, while the GPU embeds and runs layer 0,
    // and releases them to the gate before ple_embed (design §12.4, n-gram S4b); a round that
    // warms or captures its graph reads them first.
    // A round whose rows are all in the host cache reads nothing, so it stages them first.
    // A mixed round reads every row first: its eager Forward uploads them with the io.
    const bool gated = !mixed_.active && graphs_[static_cast<std::size_t>(batch - 1)].executable.ready() &&
                       !decode_rows_cached(round_lanes, round_positions);
    stage_decode(round_lanes, round_positions);
    if (mixed_.active) { stage_mixed_prefill(batch, batch); }
    const std::uint32_t word = next_gate_word();
    if (!gated) {
        stage_decode_rows(round_lanes, round_positions);
        publish_pinned_word(gate_ready(), word);
    }
    if (mixed_.active) {
        if (one_column_divisor() != 0) {
            *reinterpret_cast<std::int32_t*>(host_io() + io_layout_.divisor) = plain_divisor(round_positions[0]);
        }
        upload_pinned(io_device_.p, io_host_.data(), io_prefix(batch + mixed_.width), device_.stream);
        if (cpu_service_) { cpu_service_->wake(); }
        settle_round(); // a deferred update precedes the next forward
        residency_->before_round(device_.stream, /*landing=*/true, /*walking=*/false);
        run_mixed(batch, 1, nullptr);
    } else {
        run_decode(batch, round_positions[0]);
        // Nothing that can wait for the device sits between the gated launch and its publish
        // (the gate would deadlock): the rows are read and released before anything else queues.
        if (gated) {
            read_behind_gate(lanes_[lanes[0]], word, io_prefix(batch) - io_layout_.ngram,
                             [&] { stage_decode_rows(round_lanes, round_positions); });
        }
        residency_->enqueue_route_download(device_.stream, batch);
    }
    for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].state_tokens; }
    sample(round_lanes, round_positions);
    defer_round(batch, 1, false, 1, round_lanes);
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    for (std::int32_t b = 0; b < batch; ++b) {
        lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
        lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
    }
    pending_transaction_  = ++next_transaction_;
    auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                {pending_tokens_.data(), static_cast<std::size_t>(batch)}, timing);
    mark_row_failures(pending, {});
    return pending;
}

runtime::ExecutionTiming ProgramImpl::append_forced(std::span<const SequenceHandle> sequences,
                                                    std::span<const TokenId> tokens, std::uint32_t stride) {
    const auto start = Clock::now();
    if (stride == 0 || tokens.size() != static_cast<std::size_t>(stride) * sequences.size()) {
        throw std::invalid_argument("Qwen4Exp: forced tokens do not match their rows");
    }
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        const auto index = lane_of(sequences[row]);
        Lane& lane       = lanes_[index];
        if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
            throw std::logic_error("Qwen4Exp: forced tokens on a lane that is not at a committed frontier");
        }
        const auto forced = tokens.subspan(row * stride, stride);
        // The model reads the pending input token and every forced token but the last, which
        // becomes the next decode input.
        lane.history.insert(lane.history.end(), forced.begin(), forced.end());
        if (lane.proposer) {
            for (const auto token : forced) { lane.proposer->append(token); }
        }
        const auto begin = static_cast<std::int32_t>(lane.state_tokens);
        stage_sequence(index, begin, static_cast<std::int32_t>(stride), lane.history);
        const auto chunk = stage_mtp_chunk(lane, index, begin, static_cast<std::int32_t>(stride));
        run(1, static_cast<std::int32_t>(stride), 1, chunk ? &*chunk : nullptr);
        device_.synchronize();
        check_device_error();
        trace_round(offload::RouteTraceKind::ForcedTokens, 1, static_cast<std::int32_t>(stride), stride,
                    kDecodePromotionsPerLayer, static_cast<std::uint32_t>(begin));
        residency_->after_round(device_.stream, static_cast<std::int32_t>(stride), kDecodePromotionsPerLayer);
        apply_vram_target(false);
        lane.state_tokens += stride;
        if (dflash2_) { lane.dflash_frontier = lane.state_tokens; }
    }
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    return timing;
}

CommitRowResult::SpeculativeCounters ProgramImpl::speculative_counters(const SpeculativeStats& stats) const noexcept {
    if (max_width_ <= 1) { return {}; }
    return {.rounds          = stats.rounds,
            .drafted_tokens  = stats.drafted_tokens,
            .accepted_tokens = stats.accepted_tokens,
            .fallback_steps  = stats.fallback_steps};
}

CommitResult ProgramImpl::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
    if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_ ||
        pending_transaction_ == 0) {
        throw std::logic_error("Qwen4Exp: commit of a foreign or stale pending batch");
    }
    const auto rows = ContractAccess::rows(pending);
    if (decisions.size() != rows.size()) { throw std::logic_error("Qwen4Exp: commit decisions are not row aligned"); }
    // A failed row (its PendingRowFailure) commits nothing and is released like a cancelled one.
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto& d = decisions[row];
        if ((d.cancelled && d.failed) ||
            (!d.cancelled && d.failed != (pending.row_failure(row) != runtime::PendingRowFailure::None)) ||
            (d.failed && (d.accepted_tokens != 0 || !d.terminal))) {
            throw std::logic_error("Qwen4Exp: a commit decision is invalid");
        }
    }
    if (round_width_ > 1) { return commit_verified(std::move(pending), decisions); }
    CommitResult out;
    out.row_count = rows.size();
    // After a plain decode round the drafter writes the round's cell (its pending cell moves
    // one position); after the prefill's first token the pending cell stays the prompt's last.
    const bool catch_up = (mtp_ || dflash2_) && plain_round_;
    std::int32_t* commit = catch_up ? spec_host(spec_layout_.commit) : nullptr;
    bool any_cell        = false;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto index = lane_of(rows[row]);
        Lane& lane       = lanes_[index];
        const auto& d    = decisions[row];
        if (commit != nullptr) { commit[row] = 0; }
        out.rows[row].speculative_counters = speculative_counters(lane.speculative);
        if (d.cancelled || d.failed) {
            if (catch_up && lane.mtp_live) {
                // MR2: decode() advanced the state past the round's position at submission and
                // a cancelled row skips the catch-up, so that position's cell becomes the
                // pending one: its residual, which the round exported, becomes the saved one.
                CUDA_CHECK(cudaMemcpyAsync(saved_column(index).data,
                                           static_cast<const std::byte*>(mtp_residuals_.p) + 2ULL * width_ * row,
                                           2ULL * width_, cudaMemcpyDeviceToDevice, device_.stream));
                lane.mtp_cells = lane.state_tokens - 1;
            }
            lane.constraint_round.clear();
            out.rows[row].timings     = timings(lane);
            out.rows[row].speculative = lane.speculative;
            out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                 : runtime::CommitDisposition::CancelledReleased;
            continue;
        }
        if (d.accepted_tokens != 1) { throw std::logic_error("Qwen4Exp: a row commits exactly one token"); }
        commit_constraint_draws(lane, 1);
        lane.history.push_back(pending_tokens_[row]);
        if (commit != nullptr && (lane.mtp_live || dflash2_)) {
            commit[row] = 1;
            any_cell    = true;
        }
        if (lane.proposer) { lane.proposer->append(pending_tokens_[row]); }
        if (lane.phase == Phase::Prefill) { lane.phase = Phase::Decode; }
        if (d.terminal) {
            lane.phase                = Phase::Finishable;
            out.rows[row].disposition = runtime::CommitDisposition::Finishable;
        } else {
            out.rows[row].disposition = runtime::CommitDisposition::Active;
        }
    }
    if (any_cell) {
        upload_pinned(spec_device(spec_layout_.commit), commit, 4ULL * rows.size(), device_.stream);
        if (mtp_) {
            mtp_catch_up(rows, 1, commit);
        } else {
            dflash2_catch_up(rows, 1, commit);
        }
    }
    for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!decisions[row].cancelled && !decisions[row].failed) {
            Lane& lane = lanes_[ContractAccess::lane(rows[row])];
            prefix_publish(lane, prefix_frontier(lane, false));
        }
    }
    settle_round();
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (decisions[row].cancelled || decisions[row].failed) { release(ContractAccess::lane(rows[row])); }
    }
    plain_round_ = false;
    ContractAccess::consume(pending);
    pending_transaction_ = 0;
    return out;
}

DiscardResult ProgramImpl::discard(PendingBatch&& pending) noexcept {
    DiscardResult out;
    if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_) {
        return out;
    }
    const auto rows = ContractAccess::rows(pending);
    for (const auto& row : rows) {
        const auto index = ContractAccess::lane(row);
        if (index < options_.max_concurrency && lanes_[index].phase != Phase::Free) { release(index); }
    }
    out.status    = runtime::ConsumeStatus::Consumed;
    out.row_count = rows.size();
    ContractAccess::consume(pending);
    pending_transaction_ = 0;
    round_width_         = 1;
    deferred_.pending    = false;
    return out;
}

FinishResult ProgramImpl::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    try {
        const auto index = lane_of(sequence);
        if (lanes_[index].phase != Phase::Finishable) {
            diagnostic("Qwen4Exp: finish of a lane that is not finishable", DiagnosticLevel::Error);
            return out;
        }
        if (!mtp_settled(lanes_[index])) {
            diagnostic("Qwen4Exp: a finished request's MTP cells are out of step with its state", DiagnosticLevel::Warning);
        }
        out.timings           = timings(lanes_[index]);
        out.speculative       = lanes_[index].speculative;
        out.constrained_draws = std::move(lanes_[index].constraint_trace);
        if (residency_) {
            const auto& s = residency_->stats();
            const auto& a = lanes_[index].cache_at_admission;
            out.expert_cache = ExpertCacheStats{.routed        = s.routed - a.routed,
                                                .hits          = s.hits - a.hits,
                                                .decode_routed = s.decode_routed - a.decode_routed,
                                                .decode_hits   = s.decode_hits - a.decode_hits};
        }
        report_cache(lanes_[index]);
        report_ngram(lanes_[index]);
        report_verify_rounds(lanes_[index]);
        prefix_finish(lanes_[index], index);
        release(index);
        if (placement_drifted()) { wake_maintenance(); } // settled when the engine idles (settle_placement)
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (const std::exception& error) {
        // The Engine fails a terminal sequence that does not finish; say why.
        diagnostic(std::string("Qwen4Exp: finish failed: ") + error.what(), DiagnosticLevel::Error);
    } catch (...) {
        diagnostic("Qwen4Exp: finish failed", DiagnosticLevel::Error);
    }
    return out;
}

AbortResult ProgramImpl::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    try {
        const auto index = lane_of(sequence);
        out.timings      = timings(lanes_[index]);
        out.speculative  = lanes_[index].speculative;
        // A consistent abort (no unit in flight) leaves an endpoint a resend resumes from; inside
        // a layer walk the layers are at different positions, so it leaves none.
        if (pending_transaction_ == 0 && lanes_[index].phase != Phase::Free && !(walk_.active && walk_.lane == index)) {
            prefix_finish(lanes_[index], index);
        }
        release(index);
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (...) {}
    return out;
}

bool ProgramImpl::recover_after_failure() noexcept {
    try {
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { release(i); }
        }
        transaction_lane_.reset();
        pending_transaction_ = 0;
        round_width_         = 1;
        deferred_.pending    = false; // the failed round's cache update
        device_.synchronize();
        if (prefix_) {
            prefix_->drain();
            if (clear_prefix_on_release_) { prefix_->clear(); }
        }
        clear_prefix_on_release_ = false;
        return true;
    } catch (...) {
        return false;
    }
}

void ProgramImpl::release_all() noexcept {
    for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
        if (lanes_[i].phase != Phase::Free) { release(i); }
    }
    transaction_lane_.reset();
    pending_transaction_ = 0;
    round_width_         = 1; // an abandoned verification round leaves no width behind
    deferred_.pending    = false; // after a failure the last round's cache update is dropped (shutdown settles it first)
    try { device_.synchronize(); } catch (...) {}
    if (clear_prefix_on_release_) { // a failed round may have published state it never computed
        if (prefix_) { prefix_->clear(); }
        clear_prefix_on_release_ = false;
    }
}

PhysicalUsageSnapshot ProgramImpl::usage() const noexcept {
    PhysicalUsageSnapshot out;
    for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
        if (lanes_[i].phase != Phase::Free) { ++out.occupied.state_slots; }
    }
    out.capacity.state_slots   = options_.max_concurrency;
    out.occupied.main_kv_pages = pool_->allocated_pages();
    out.capacity.main_kv_pages = pool_->capacity_pages();
    return out;
}

MemorySummary ProgramImpl::memory() const noexcept {
    MemorySummary out;
    out.max_context             = options_.max_context;
    out.kv_capacity             = kv_pages_ * static_cast<std::uint32_t>(kPagedKVPageSize);
    out.kv_capacity_page_groups = kv_pages_;
    out.kv_capacity_max_page_groups = kv_pages_;
    out.kv_cache                = options_.kv_cache;
    out.kv_payload_bytes        = kv_vmm_ ? kv_vmm_->mapped_bytes() : kv_backing_.bytes;
    out.workspace_logical_peak_bytes = work_capacity_;
    return out;
}

std::size_t ProgramImpl::decode_budget(std::uint32_t tokens) {
    // The fill phase: until this session promoted or landed as many experts as there are frames.
    // Seeded frames (a warm start) do not end it: promotions only follow misses, so a workload
    // the saved state matches promotes little anyway, while one it does not match replaces the
    // seeded experts at the fill rate instead of the steady-state rate.
    const offload::ExpertResidency::Stats& stats = residency_->stats();
    if (stats.promotions + stats.landed < residency_->frames()) {
        return kDecodePromotionsPerLayer * tokens;
    }
    const std::uint64_t interval = promotion_interval();
    budget_tokens_ += tokens;
    const std::uint64_t due = budget_tokens_ / interval;
    budget_tokens_ %= interval;
    return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
}

std::uint64_t ProgramImpl::promotion_interval() const noexcept {
    const double scaled = static_cast<double>(kDecodePromotionInterval) * kReferenceLinkBytesPerSecond / link_bytes_per_second_;
    return static_cast<std::uint64_t>(std::clamp(std::lround(scaled), 1L, 8L));
}

void ProgramImpl::defer_round(std::int32_t columns, std::uint32_t tokens, bool live, std::int32_t width,
                              std::span<const std::uint32_t> lanes, bool to_proposal) {
    settle_round();
    deferred_ = {.columns     = columns,
                 .tokens      = tokens,
                 .live        = live,
                 .pending     = true,
                 .width       = width,
                 .to_proposal = to_proposal};
    for (std::size_t b = 0; b < lanes.size() && b < deferred_.lanes.size(); ++b) {
        deferred_.lanes[b] = static_cast<std::int32_t>(lanes[b]);
    }
}

void ProgramImpl::settle_round() {
    if (!deferred_.pending) { return; }
    deferred_.pending = false;
    const std::size_t budget = decode_budget(deferred_.tokens);
    if (trace_) {
        const std::int32_t width = deferred_.live ? deferred_.width : 1;
        trace_round(deferred_.live ? offload::RouteTraceKind::Verify : offload::RouteTraceKind::Decode, deferred_.columns / width,
                    width, deferred_.tokens, budget, offload::RouteTrace::kUnknownPosition,
                    deferred_.live ? std::span<const std::uint8_t>(live_.data(),
                                                                   static_cast<std::size_t>(deferred_.columns))
                                   : std::span<const std::uint8_t>{},
                    std::span<const std::int32_t>(deferred_.lanes.data(),
                                                  static_cast<std::size_t>(deferred_.columns / width)));
    }
    // A mixed round's prompt call counts with its rows (every one of its columns live).
    const auto columns = static_cast<std::size_t>(deferred_.columns + deferred_.prefill_columns);
    if (deferred_.live && deferred_.prefill_columns > 0) {
        if (live_.size() < columns) { live_.resize(columns); }
        std::fill(live_.begin() + deferred_.columns, live_.begin() + static_cast<std::ptrdiff_t>(columns), 1);
    }
    residency_->after_round(device_.stream, static_cast<std::int32_t>(columns), budget,
                            deferred_.live ? std::span<const std::uint8_t>(live_.data(), columns)
                                           : std::span<const std::uint8_t>{});
    apply_vram_target(false);
}

void ProgramImpl::settle_deferred() { settle_round(); }  void ProgramImpl::settle_deferred_or_drop() noexcept {
    try {
        settle_round();
    } catch (...) {
        deferred_.pending = false;
    }
}

} // namespace infernix::models::qwen4_exp::detail
