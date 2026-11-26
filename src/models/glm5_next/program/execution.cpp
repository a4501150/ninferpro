// Prefill, decode rounds, forced tokens, commit, finish and abort, and the helpers they share: io
// staging, the Forward call and its decode graphs, and the sampler.

#include "models/glm5_next/program/program_impl.h"

#include "infernix/ops/cast.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace infernix::models::glm5_next::detail {

namespace {

// A prompt's routed experts gain this much LFRU count in total when its last call completes, spread
// over the prompt's tokens (moe-offload-design.md §4.3).
constexpr double kPromptSeedCredit = 64.0;
// Calls at most this wide promote at the decode rate; wider prefill calls promote nothing.
constexpr std::int32_t kServedCallColumns = 8;

} // namespace

std::uint32_t ProgramImpl::lane_of(const SequenceHandle& sequence) const {
    const auto lane = ContractAccess::lane(sequence);
    if (ContractAccess::owner(sequence) != this || lane >= options_.max_concurrency ||
        lanes_[lane].phase == Phase::Free || lanes_[lane].epoch != ContractAccess::epoch(sequence)) {
        throw std::logic_error("Glm5Next: stale or foreign sequence handle");
    }
    return lane;
}

GenerationTimings ProgramImpl::timings(const Lane& lane) const {
    GenerationTimings out;
    out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
    out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
    out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
    return out;
}

void ProgramImpl::release(std::uint32_t index) noexcept {
    Lane& lane = lanes_[index];
    if (walk_.active && walk_.lane == index) { walk_abandon(); }
    if (stream_lease_.valid() && !prefilling_besides(index)) {
        try {
            return_stream_lease();
        } catch (...) {}
    }
    vision_release(lane);
    prefix_release(lane);
    lane.pages.clear();
    lane.reservation.reset();
    lane.row = KVExecutionRowLease{};
    lane.history.clear();
    lane.history.shrink_to_fit();
    lane.calls.clear();
    lane.next_call = 0;
    lane.readout.clear();
    lane.constraint.reset();
    lane.prompt.reset();
    lane.constraint_round.clear();
    lane.constraint_trace.clear();
    lane.phase        = Phase::Free;
    lane.state_tokens = 0;
    lane.extent       = 0;
}

void ProgramImpl::split_beside_decode(Lane& lane, std::uint32_t index) {
    const std::uint32_t cap = std::min<std::uint32_t>(beside_decode_chunk_, static_cast<std::uint32_t>(chunk_));
    if (cap == 0 || lane.calls[lane.next_call] - lane.state_tokens <= cap) { return; }
    // A lane that publishes keeps the cold call grid: this family's exactness contract ties a
    // snapshot's bits to the publishing run's call boundaries (projection routes are chosen by
    // call width), so its prefill is not split beside another lane's decode.
    if (lane.prefix.reuse) { return; }
    bool decoding = false;
    for (std::uint32_t other = 0; other < options_.max_concurrency; ++other) {
        decoding = decoding || (other != index && lanes_[other].phase == Phase::Decode);
    }
    if (!decoding) { return; }
    lane.calls.insert(lane.calls.begin() + static_cast<std::ptrdiff_t>(lane.next_call), lane.state_tokens + cap);
}

void ProgramImpl::stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width) {
    auto* ids       = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.ids);
    auto* positions = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.positions);
    auto* columns   = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.columns);
    const auto& history = lanes_[lane].history;
    for (std::int32_t i = 0; i < width; ++i) {
        ids[i]       = history[static_cast<std::size_t>(begin + i)];
        positions[i] = begin + i;
    }
    columns[0] = width - 1;
    reinterpret_cast<std::int32_t*>(host_io() + plan_.io.slots)[0] = static_cast<std::int32_t>(lane);
    reinterpret_cast<std::int32_t*>(host_io() + plan_.io.rows)[0]  = static_cast<std::int32_t>(lane);
    host_lanes_[0] = static_cast<std::int32_t>(lane);
}

void ProgramImpl::stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
    auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.ids);
    auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.positions);
    auto* columns = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.columns);
    auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.slots);
    auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.rows);
    for (std::size_t b = 0; b < lanes.size(); ++b) {
        const Lane& lane = lanes_[lanes[b]];
        ids[b]           = lane.history[static_cast<std::size_t>(positions[b])];
        pos[b]           = positions[b];
        columns[b]       = static_cast<std::int32_t>(b);
        slots[b]         = static_cast<std::int32_t>(lanes[b]);
        rows[b]          = static_cast<std::int32_t>(lanes[b]);
        host_lanes_[b]   = static_cast<std::int32_t>(lanes[b]);
    }
}

execution::ForwardBatch ProgramImpl::io_batch(std::int32_t batch, std::int32_t width, std::int32_t logit_columns) const {
    auto* base              = static_cast<std::byte*>(io_device_.p);
    const std::int32_t cols = batch * width;
    execution::ForwardBatch fb;
    fb.ids           = Tensor(base + plan_.io.ids, DType::I32, {cols});
    fb.positions     = Tensor(base + plan_.io.positions, DType::I32, {cols});
    fb.slots         = Tensor(base + plan_.io.slots, DType::I32, {batch});
    fb.table_rows    = Tensor(base + plan_.io.rows, DType::I32, {batch});
    fb.logit_columns = Tensor(base + plan_.io.columns, DType::I32, {logit_columns});
    fb.host_slots    = std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(batch));
    fb.batch         = batch;
    fb.width         = width;
    return fb;
}

void ProgramImpl::forward_call(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                               bool verify) {
    set_expert_route();
    execution::ForwardBatch fb = io_batch(batch, width, logit_columns);
    if (verify) { fb.verify = &records_; }
    // The drafter's features: every column of a round, a call's last window.
    if (drafter_) {
        const std::int32_t window = dim(parameters_.model.config().dflash2->window);
        fb.draft_features =
            feature_columns(verify || width == 1 ? batch * width : std::min(width, window));
    }
    // A scored prompt's calls keep their final residuals for the LM head (causal_score).
    if (scoring_) { fb.residual_out = score_residuals(score_column_, batch * width); }
    if (scoring_ && capture_rows_ > 0) {
        fb.draft_features = Tensor(static_cast<std::byte*>(capture_features_.p) +
                                       2ULL * static_cast<std::size_t>(capture_rows_) *
                                           static_cast<std::size_t>(score_column_),
                                   DType::BF16, {capture_rows_, batch * width});
    }
    // A prompt's last call keeps its final residual (the zero-token resume's head input).
    if (head_saved_lane_ >= 0) {
        fb.head_residual_out =
            Tensor(static_cast<std::byte*>(head_saved_.p) +
                       static_cast<std::size_t>(head_saved_lane_) * plan_.head_saved,
                   DType::BF16, {width_, 1});
        head_saved_lane_ = -1;
    }
    if (vision_call_lane_ >= 0) {
        vision_batch(lanes_[static_cast<std::size_t>(vision_call_lane_)], vision_call_.first,
                     vision_call_.second, static_cast<std::byte*>(vision_io_device_.p), fb);
    }
    Tensor logits(logits32_.p, DType::FP32, {vocab_, logit_columns});
    forward_->run(fb, logits);
    round_logits(logit_columns);
}

bool ProgramImpl::stream_chunk(std::int32_t width) {
    if (!expert_stream_ || width < kStreamMinColumns || residency_->frames() == 0) { return false; }
    if (!stream_lease_.valid()) {
        const std::int32_t* table = residency_->host_table();
        const std::uint32_t E     = c_.moe.experts;
        const std::uint32_t L     = c_.moe_layers();
        std::uint32_t most        = 0;
        for (std::uint32_t l = 0; l < L; ++l) {
            std::uint32_t absent = 0;
            for (std::uint32_t e = 0; e < E; ++e) {
                absent += table[static_cast<std::size_t>(l) * E + e] < 0 ? 1U : 0U;
            }
            most = std::max(most, absent);
        }
        // The ring: two halves of the layer with the most non-resident experts, in frames of the
        // pool with the most room, plus one frame for the slot tables. The frames it takes evict
        // the pool's layers' cheapest experts; their share per layer stays within the half the
        // absent count leaves spare, and a layer beyond the half's capacity stages the rest.
        const std::uint64_t slot     = residency_->lend_pitch();
        const cudaStream_t writers[] = {expert_stream_->stream()};
        const auto room_slots = static_cast<std::uint32_t>(residency_->lendable_bytes() / slot);
        const std::uint32_t half =
            std::min(std::min(most, E), room_slots > 1 ? (room_slots - 1) / 2 : 0U);
        if (half < kStreamHalfMin) { return false; }
        stream_lease_      = residency_->lend((2ULL * half + 1) * slot, device_.stream, writers);
        stream_ring_bytes_ = stream_lease_.memory.bytes;
        stream_at_lease_   = expert_stream_->stats();
    }
    return true;
}

void ProgramImpl::return_stream_lease() {
    if (!stream_lease_.valid()) { return; }
    const auto frames = stream_lease_.count;
    residency_->give_back(stream_lease_, device_.stream);
    stream_lease_       = {};
    stream_ring_bytes_  = 0;
    const auto& now     = expert_stream_->stats();
    const auto streamed = now.streamed - stream_at_lease_.streamed;
    const auto copies   = now.copies - stream_at_lease_.copies;
    char line[256];
    std::snprintf(
        line, sizeof(line),
        "prefill expert stream: %llu experts (%.2f GiB) in %llu copies through %u lent frames",
        static_cast<unsigned long long>(streamed),
        static_cast<double>(now.bytes - stream_at_lease_.bytes) / (1ULL << 30),
        static_cast<unsigned long long>(copies), frames);
    diagnostic(line, DiagnosticLevel::Debug);
}

bool ProgramImpl::prefilling_besides(std::uint32_t index) const noexcept {
    for (std::uint32_t other = 0; other < options_.max_concurrency; ++other) {
        if (other != index && lanes_[other].phase == Phase::Prefill) { return true; }
    }
    return false;
}

void ProgramImpl::replay(DecodeGraph& graph, const std::function<void()>& body) {
    const cudaStream_t s = device_.stream;
    if (graph.executable.ready() && graph.resident != expert_route()) {
        graph.executable.reset();
        graph.definition.reset();
    }
    if (graph.executable.ready()) {
        graph.executable.launch(s);
    } else if (!graph.warmed || !options_.decode_graphs) {
        body();
        graph.warmed = true;
    } else {
        graph.resident = expert_route();
        graph.definition.capture(s, body);
        graph.executable.instantiate(graph.definition);
        graph.executable.launch(s);
    }
}

void ProgramImpl::run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns, bool decode_round) {
    const cudaStream_t s = device_.stream;
    CUDA_CHECK(cudaMemcpyAsync(io_device_.p, io_host_.data(), plan_.io.bytes, cudaMemcpyHostToDevice, s));
    before_call(decode_round);
    // A prefill call wide enough streams its non-resident experts by copy-engine DMA two layers
    // ahead of compute (glm5_3-flash-design.md §11.2); decode and verification rounds never do.
    const bool streamed = !decode_round && !walk_.active && stream_chunk(width);
    if (streamed) { expert_stream_->begin(ring_span(), residency_->host_table(), s); }
    try {
        if (decode_round) {
            replay(graphs_[static_cast<std::size_t>(batch - 1)], [&] {
                forward_call(batch, 1, batch);
                if (drafter_) { round_context(batch); }
            });
        } else {
            forward_call(batch, width, logit_columns);
            if (drafter_) {
                const auto* positions =
                    reinterpret_cast<const std::int32_t*>(host_io() + plan_.io.positions);
                call_context(static_cast<std::uint32_t>(host_lanes_[0]), positions[0], width);
            }
        }
    } catch (...) {
        // A failed call (an unreadable record, say) still ends the stream: left active, the next
        // streamed chunk could never begin.
        if (streamed) { expert_stream_->end(); }
        throw;
    }
    if (streamed) { expert_stream_->end(); }
    enqueue_routes(batch * width);
}

Tensor ProgramImpl::read_logits(std::int32_t columns) {
    if (options_.bf16_logits) { return Tensor(logits16_.p, DType::BF16, {vocab_, columns}); }
    return Tensor(logits32_.p, DType::FP32, {vocab_, columns});
}

void ProgramImpl::round_logits(std::int32_t columns) {
    if (!options_.bf16_logits || columns == 0) { return; }
    const Tensor wide(logits32_.p, DType::FP32, {vocab_, columns});
    Tensor narrow(logits16_.p, DType::BF16, {vocab_, columns});
    ops::cast_fp32_to_bf16(wide, narrow, device_.stream);
}

void ProgramImpl::sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
    const cudaStream_t s = device_.stream;
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    // The sampler reads the FP32 logits, or their BF16 rounding; a constraint masks them in place
    // (the prompt readout has read them first).
    Tensor logits = read_logits(batch);
    auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
    auto* logical = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.positions);
    for (std::int32_t b = 0; b < batch; ++b) {
        configs[b]      = lanes_[lanes[b]].sampling;
        configs[b].mask = grammar_mask(static_cast<std::size_t>(b));
        reasoning_ban(static_cast<std::size_t>(b), configs[b]);
        logical[b] = positions[b];
    }
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    upload_pinned(sample_pos_.p, logical, 4ULL * batch, s);
    const bool constrained = stage_constraints(lanes);
    if (constrained) { constrain(logits); }
    Tensor out(sampled_.p, DType::I32, {batch});
    const Tensor logical_positions(sample_pos_.p, DType::I32, {batch});
    {
        auto scope = work_->scope();
        ops::sample(logits, out, token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p), logical_positions,
                    ops::kSamplePurposeDecode, *work_, s);
    }
    CUDA_CHECK(cudaMemcpyAsync(host_sampled_.data(), sampled_.p, 4ULL * batch, cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    check_device_error();
    const auto* tokens = static_cast<const std::int32_t*>(host_sampled_.data());
    for (std::int32_t b = 0; b < batch; ++b) { pending_tokens_[static_cast<std::size_t>(b)] = tokens[b]; }
    if (constrained) {
        for (std::int32_t b = 0; b < batch; ++b) {
            take_constraint_draws(lanes_[lanes[b]], static_cast<std::size_t>(b), pending_tokens_[static_cast<std::size_t>(b)]);
        }
    }
}

void ProgramImpl::note_prompt_routes(Lane& lane, std::int32_t width) {
    if (width <= 0) { return; }
    const std::uint32_t L = c_.moe_layers(), E = c_.moe.experts, K = c_.moe.top_k;
    if (lane.prompt_uses.size() != static_cast<std::size_t>(L) * E) {
        lane.prompt_uses.assign(static_cast<std::size_t>(L) * E, 0);
        lane.prompt_counted = 0;
    }
    const std::int32_t* routes = residency_->route_host();
    const std::size_t stride   = residency_->route_stride();
    for (std::uint32_t l = 0; l < L; ++l) {
        const std::int32_t* ids = routes + static_cast<std::size_t>(l) * stride;
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
    std::fill(lane.prompt_uses.begin(), lane.prompt_uses.end(), 0U);
    lane.prompt_counted = 0;
}

void ProgramImpl::after_call(std::int32_t columns, std::size_t promotions) {
    // An active layer walk's stream planned its copies against the frames of its start.
    residency_->after_round(device_.stream, columns, walk_.active ? 0 : promotions);
    if (!walk_.active) { (void)placement_->apply_placement(false); }
}

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks) {
    const GrammarScope grammar(*this, masks);
    const auto start = Clock::now();
    const auto index = lane_of(sequence);
    Lane& lane       = lanes_[index];
    if (lane.phase != Phase::Prefill) { throw std::logic_error("Glm5Next: prefill on a lane that is not prefilling"); }
    if (walk_.active) {
        if (walk_.lane != index) {
            throw std::logic_error("Glm5Next: another lane's prefill holds the layer walk");
        }
        return walk_step(lane, index, start);
    }
    // The zero-token resume: the lane's state is the prompt's own end snapshot and its head's
    // saved residual, so the first token regenerates from the residual through the head at the
    // same width the cold run's last prefill call used (no prefill call at all).
    if (lane.prefix.head_resume) {
        lane.prefix.head_resume = false;
        Tensor logits(logits32_.p, DType::FP32, {vocab_, 1});
        const Tensor residual(static_cast<std::byte*>(head_saved_.p) +
                                  static_cast<std::size_t>(index) * plan_.head_saved,
                              DType::BF16, {width_, 1});
        forward_->head_logits(residual, logits);
        round_logits(1);
        const std::uint32_t lanes[]    = {index};
        const std::int32_t positions[] = {static_cast<std::int32_t>(lane.state_tokens - 1U)};
        PrefillProgress out;
        out.summary                 = lane.begin;
        out.processed_prompt_tokens = 0;
        out.complete                = true;
        if (!lane.readout.empty()) { out.readout = read_prompt_frontier(lane); }
        sample(lanes, positions); // synchronizes
        const SequenceHandle rows[] = {handle(index)};
        out.timing.submit_host_ns   = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        pending_transaction_ = ++next_transaction_;
        out.pending.emplace(ContractAccess::make_pending(this, pending_transaction_, rows,
                                                         {pending_tokens_.data(), 1}, out.timing));
        mark_row_failures(*out.pending);
        return out;
    }
    if (lane.next_call >= lane.calls.size() || lane.calls[lane.next_call] <= lane.state_tokens) {
        throw std::logic_error("Glm5Next: the lane's prefill plan does not continue its state");
    }
    if (lane.vision && !lane.vision->encoded) { vision_encode(lane); }
    if (const std::size_t end = walk_span_end(lane); end > lane.next_call) {
        walk_begin(lane, index, end);
        return walk_step(lane, index, start);
    }
    split_beside_decode(lane, index);
    const auto begin         = static_cast<std::int32_t>(lane.state_tokens);
    const auto width         = static_cast<std::int32_t>(lane.calls[lane.next_call++] - lane.state_tokens);
    const bool last          = lane.state_tokens + static_cast<std::uint32_t>(width) == lane.prompt_tokens;
    // A call the size of a decode round promotes at the decode rate; wider prefill calls promote
    // nothing (their routes still count toward the cache's scores).
    const std::size_t promotions = width <= kServedCallColumns ? decode_budget(static_cast<std::uint32_t>(width)) : 0;
    // The lane's pending copies (its restore, and the taps' capture of its state) land before the
    // call reads or writes that state.
    const auto restore_events = prefix_restore_wait(lane);
    if (!restore_events.empty()) {
        CUDA_CHECK(cudaStreamWaitEvent(device_.stream, restore_events.back(), 0));
    }
    const auto capture_events = prefix_capture_wait(lane);
    if (!capture_events.empty()) {
        CUDA_CHECK(cudaStreamWaitEvent(device_.stream, capture_events.back(), 0));
    }
    stage_sequence(index, begin, width);
    vision_call_ =
        vision_columns(lane, static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(width),
                       static_cast<std::int32_t*>(vision_io_host_.data()));
    if (vision_call_.second != 0) {
        vision_call_lane_ = static_cast<std::int32_t>(index);
        CUDA_CHECK(cudaMemcpyAsync(vision_io_device_.p, vision_io_host_.data(),
                                   4ULL * vision_call_.second, cudaMemcpyHostToDevice,
                                   device_.stream));
    }
    score_column_ = begin;
    if (last) { head_saved_lane_ = static_cast<std::int32_t>(index); }
    run(1, width, 1, false);
    vision_call_lane_ = -1;
    // The stream ring stays lent across the prompt's calls; it returns once the prompt's last is
    // enqueued (its compute has waited for every copy) and no lane is prefilling.
    if (last && !prefilling_besides(index)) { return_stream_lease(); }
    lane.state_tokens += static_cast<std::uint32_t>(width);
    prefix_after_prefill_call(lane, index, last);
    return after_prefill(lane, index, begin, width, static_cast<std::uint32_t>(width), last,
                         promotions, start);
}

PrefillProgress ProgramImpl::after_prefill(Lane& lane, std::uint32_t index, std::int32_t begin,
                                           std::int32_t width, std::uint32_t tokens, bool last,
                                           std::size_t promotions, Clock::time_point start) {
    PrefillProgress out;
    out.summary                 = lane.begin;
    out.processed_prompt_tokens = tokens;
    out.complete                = last;
    if (last) {
        const std::uint32_t lanes[]    = {index};
        const std::int32_t positions[] = {begin + width - 1};
        // The readout reads the logits before the sampler's constraints mask them.
        if (!lane.readout.empty()) { out.readout = read_prompt_frontier(lane); }
        sample(lanes, positions); // synchronizes
        vision_release(lane);     // the prompt's calls have read the handoff
        note_prompt_routes(lane, width);
        after_call(width, promotions);
        seed_prompt_experts(lane);
        const SequenceHandle rows[] = {handle(index)};
        out.timing.submit_host_ns   = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        pending_transaction_ = ++next_transaction_;
        out.pending.emplace(ContractAccess::make_pending(this, pending_transaction_, rows,
                                                         {pending_tokens_.data(), 1}, out.timing));
        mark_row_failures(*out.pending);
    } else {
        device_.synchronize();
        check_device_error();
        note_prompt_routes(lane, width);
        after_call(width, promotions);
        out.timing.submit_host_ns = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
    }
    return out;
}

PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> sequences,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::TokenMaskProvider* masks) {
    const GrammarScope grammar(*this, masks);
    const auto start = Clock::now();
    const auto batch = static_cast<std::int32_t>(sequences.size());
    if (batch <= 0 || batch > static_cast<std::int32_t>(options_.max_concurrency)) {
        throw std::logic_error("Glm5Next: decode batch is empty or too large");
    }
    if (pending_transaction_ != 0) { throw std::logic_error("Glm5Next: a round awaits its commit"); }
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::array<std::int32_t, kMaximumConcurrency> positions{};
    for (std::int32_t b = 0; b < batch; ++b) {
        lanes[static_cast<std::size_t>(b)] = lane_of(sequences[static_cast<std::size_t>(b)]);
        const Lane& lane                   = lanes_[lanes[static_cast<std::size_t>(b)]];
        if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
            throw std::logic_error("Glm5Next: decode row is not at a committed frontier");
        }
        if (lane.state_tokens >= lane.extent) { throw std::logic_error("Glm5Next: decode past the request's extent"); }
        positions[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(lane.state_tokens);
    }
    const std::span<const std::uint32_t> round_lanes(lanes.data(), static_cast<std::size_t>(batch));
    const std::span<const std::int32_t> round_positions(positions.data(), static_cast<std::size_t>(batch));
    if (drafter_) {
        // A one-row round proposes what its row has room for and verifies the drafts
        // verify_length keeps (none: it decodes plainly). A round of several rows, or of a
        // token-constrained row, decodes plainly: with experts beyond VRAM a verify column's routes
        // cost more than its accepted tokens save (glm5_3-flash-design.md §10.3).
        std::int32_t room = plan_.draft;
        bool plain        = batch > 1 || budgets.size() != sequences.size();
        for (std::int32_t b = 0; b < batch && !plain; ++b) {
            const Lane& lane = lanes_[lanes[static_cast<std::size_t>(b)]];
            plain            = lane.constraint != nullptr;
            room = std::min(room, draft_room(lane, budgets[static_cast<std::size_t>(b)]));
        }
        if (!plain && room > 0) {
            propose(round_lanes, room + 1);
            const std::int32_t k = verify_length(lanes_[lanes[0]], room);
            if (k > 0) { return speculate(sequences, round_lanes, round_positions, k + 1, start); }
        }
        for (std::int32_t b = 0; b < batch; ++b) {
            ++lanes_[lanes[static_cast<std::size_t>(b)]].speculative.fallback_steps;
        }
    }
    stage_decode(round_lanes, round_positions);
    run(batch, 1, batch, true);
    for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[static_cast<std::size_t>(b)]].state_tokens; }
    sample(round_lanes, round_positions); // synchronizes
    after_call(batch, decode_budget(1));
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane = lanes_[lanes[static_cast<std::size_t>(b)]];
        lane.decode_ns += timing.submit_host_ns;
        lane.decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
    }
    pending_transaction_ = ++next_transaction_;
    auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                {pending_tokens_.data(), static_cast<std::size_t>(batch)}, timing);
    mark_row_failures(pending);
    return pending;
}

runtime::ExecutionTiming ProgramImpl::append_forced(std::span<const SequenceHandle> sequences,
                                                    std::span<const TokenId> tokens, std::uint32_t stride) {
    const auto start = Clock::now();
    if (stride == 0 || tokens.size() != static_cast<std::size_t>(stride) * sequences.size()) {
        throw std::invalid_argument("Glm5Next: forced tokens do not match their rows");
    }
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        const auto index = lane_of(sequences[row]);
        Lane& lane       = lanes_[index];
        if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
            throw std::logic_error("Glm5Next: forced tokens on a lane that is not at a committed frontier");
        }
        if (lane.state_tokens + stride > lane.extent) {
            throw std::logic_error("Glm5Next: forced tokens run past the request's extent");
        }
        // The model reads the pending input token and every forced token but the last, which
        // becomes the next decode input.
        const auto forced = tokens.subspan(row * stride, stride);
        lane.history.insert(lane.history.end(), forced.begin(), forced.end());
        const auto begin = static_cast<std::int32_t>(lane.state_tokens);
        stage_sequence(index, begin, static_cast<std::int32_t>(stride));
        run(1, static_cast<std::int32_t>(stride), 1, false);
        // A forced run wide enough to stream does not continue a prompt's prefill: the ring
        // returns when no lane is prefilling.
        if (stream_lease_.valid() && !prefilling_besides(index)) { return_stream_lease(); }
        device_.synchronize();
        check_device_error();
        after_call(static_cast<std::int32_t>(stride), decode_budget(stride));
        lane.state_tokens += stride;
    }
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    return timing;
}

CommitResult ProgramImpl::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
    if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_ ||
        pending_transaction_ == 0) {
        throw std::logic_error("Glm5Next: commit of a foreign or stale pending batch");
    }
    const auto rows = ContractAccess::rows(pending);
    if (decisions.size() != rows.size()) { throw std::logic_error("Glm5Next: commit decisions are not row aligned"); }
    // A failed row (its PendingRowFailure) commits nothing and is released like a cancelled one.
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto& d = decisions[row];
        if ((d.cancelled && d.failed) ||
            (!d.cancelled && d.failed != (pending.row_failure(row) != runtime::PendingRowFailure::None)) ||
            (d.failed && (d.accepted_tokens != 0 || !d.terminal))) {
            throw std::logic_error("Glm5Next: a commit decision is invalid");
        }
    }
    if (round_width_ > 1) { return commit_verified(std::move(pending), decisions); }
    CommitResult out;
    out.row_count = rows.size();
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto index = ContractAccess::lane(rows[row]);
        Lane& lane       = lanes_[index];
        const auto& d    = decisions[row];
        out.rows[row].speculative_counters = speculative_counters(lane.speculative);
        if (d.cancelled || d.failed) {
            lane.constraint_round.clear();
            out.rows[row].speculative = lane.speculative;
            out.rows[row].timings     = timings(lane);
            out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                 : runtime::CommitDisposition::CancelledReleased;
            continue;
        }
        if (d.accepted_tokens != 1) { throw std::logic_error("Glm5Next: a row commits exactly one token"); }
        commit_constraint_draws(lane);
        lane.history.push_back(pending_tokens_[row]);
        if (lane.phase == Phase::Prefill) { lane.phase = Phase::Decode; }
        if (d.terminal) {
            lane.phase                = Phase::Finishable;
            out.rows[row].disposition = runtime::CommitDisposition::Finishable;
        } else {
            out.rows[row].disposition = runtime::CommitDisposition::Active;
        }
    }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (decisions[row].cancelled || decisions[row].failed) { release(ContractAccess::lane(rows[row])); }
    }
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
    return out;
}

FinishResult ProgramImpl::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    try {
        const auto index = lane_of(sequence);
        if (lanes_[index].phase != Phase::Finishable) {
            diagnostic("Glm5Next: finish of a lane that is not finishable", DiagnosticLevel::Error);
            return out;
        }
        out.timings           = timings(lanes_[index]);
        out.speculative       = lanes_[index].speculative;
        out.constrained_draws = std::move(lanes_[index].constraint_trace);
        out.expert_cache      = expert_cache_stats(lanes_[index]);
        prefix_finish(lanes_[index], index);
        release(index);
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (const std::exception& error) {
        // The Engine fails a terminal sequence that does not finish; say why.
        diagnostic(std::string("Glm5Next: finish failed: ") + error.what(), DiagnosticLevel::Error);
    } catch (...) {
        diagnostic("Glm5Next: finish failed", DiagnosticLevel::Error);
    }
    return out;
}

AbortResult ProgramImpl::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    try {
        const auto index = lane_of(sequence);
        out.timings      = timings(lanes_[index]);
        out.speculative  = lanes_[index].speculative;
        // A consistent abort publishes the blocks the prompt prefilled (its prefill-computed
        // prefix stays reusable).
        prefix_finish(lanes_[index], index);
        release(index);
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (...) {}
    return out;
}

bool ProgramImpl::recover_after_failure() noexcept {
    try {
        release_all();
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
    round_width_         = 1;
    try {
        device_.synchronize();
    } catch (...) {}
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
    out.max_context                  = options_.max_context;
    out.kv_capacity                  = plan_.kv_pages * static_cast<std::uint32_t>(kPagedKVPageSize);
    out.kv_capacity_page_groups      = plan_.kv_pages;
    out.kv_capacity_max_page_groups  = plan_.kv_pages;
    out.kv_cache                     = options_.kv_cache;
    out.kv_payload_bytes             = kv_backing_.bytes;
    out.workspace_logical_peak_bytes = plan_.work;
    return out;
}

} // namespace infernix::models::glm5_next::detail
