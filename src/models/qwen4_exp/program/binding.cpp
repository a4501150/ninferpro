// Request admission and binding, and pause and replay (design §15.7): a request's plan, its
// prefix-cache sources and context pages, and the transactions that pause and resume it.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

RequestBasePlan ProgramImpl::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    const auto& data = qwen_common::PreparedPromptAccess::view(prompt);
    std::shared_ptr<const VisionAdmission> vision;
    if (data.has_media()) { vision = plan_vision(data); }
    const auto n = static_cast<std::uint32_t>(data.token_ids.size());
    if (n == 0 || n > options_.max_context) {
        throw std::invalid_argument("Qwen4Exp: the prompt is empty or exceeds max_context");
    }
    for (const TokenId id : options.readout_tokens) {
        if (id < 0 || id >= token_domain_) {
            throw std::invalid_argument("readout token is outside the public token domain");
        }
    }
    auto base                       = std::make_unique<BasePlanImpl>();
    base->readout_tokens            = options.readout_tokens;
    base->constraint                = compile_constraint(options.constraint);
    base->summary.prompt_tokens     = n;
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output   = options_.max_context - n + 1U;
    base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                ? FinishReason::OutputLimit
                                                : FinishReason::ContextCapacity;
    base->reuse                           = prefix_ != nullptr && options.allow_prefix_reuse && data.identity.reusable;
    base->summary.publish_continuation    = base->reuse;
    base->vision   = std::move(vision);
    base->sampling = translate(options.sampling);
    const std::uint32_t positions = std::min(options_.max_context, n + base->summary.effective_output_tokens);
    base->positions = positions;
    base->pages     = (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
    base->prompt = std::make_shared<const PreparedPrompt>(std::move(prompt));
    return RequestBasePlan(std::move(base));
}

bool ProgramImpl::feasible(const RequestBasePlan& base) const noexcept {
    // A window lends at most what serving never gives up (a quarter of the frames stays).
    const bool frames = !base.impl_ || !base.impl_->vision ||
                        base.impl_->vision->window.lease_bytes <=
                            residency_->lend_capacity_bytes();
    return base.impl_ != nullptr && frames && base.impl_->pages <= pool_->capacity_pages() &&
           base.impl_->pages <= static_cast<std::uint32_t>(pages_per_row_);
}

std::uint32_t ProgramImpl::binding_pages(const BasePlanImpl& base, std::uint32_t end) const noexcept {
    std::uint32_t positions = std::min(base.positions, end + round_positions());
    // Without preemption the request holds its whole extent from admission (less a page of the
    // pool), so no unit of it grows and nothing pauses it.
    if (!options_.preemption) {
        const std::uint32_t pool = pool_->capacity_pages() > 1 ? pool_->capacity_pages() - 1U : 1U;
        positions = std::max(positions, std::min(base.positions, pool * static_cast<std::uint32_t>(kPagedKVPageSize)));
    }
    return (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
}

PrefixReusePath ProgramImpl::reuse_path_for(prefix::cache_index::SnapshotKind kind) noexcept {
    return kind == prefix::cache_index::SnapshotKind::Endpoint ? PrefixReusePath::HybridEndpoint
                                                                 : PrefixReusePath::HybridSnapshot;
}

SourceCandidate ProgramImpl::make_source(const BasePlanImpl& base, const qwen_common::PreparedPromptData& data,
                                         std::optional<PrefixSelection> selection) const {
    auto impl      = std::make_shared<QuoteImpl>();
    impl->summary  = base.summary;
    impl->sampling = base.sampling;
    impl->pages    = binding_pages(base, base.summary.prompt_tokens);
    impl->vision   = base.vision;
    SourceCandidate out;
    if (selection) {
        if (selection->snapshot) {
            // Items inside the reused prefix are not encoded again (design §16.4).
            if (impl->vision && selection->frontier > 0) { impl->vision = plan_vision(data, selection->frontier); }
            out.reused_tokens = selection->frontier;
            out.reuse_path    = reuse_path_for(prefix_->index().snapshot(*selection->snapshot).kind);
            out.from_disk     = prefix_->fetch_holds(*selection->snapshot);
        }
        impl->prefix = std::move(selection);
    }
    out.hybrid = std::move(impl);
    return out;
}

std::vector<SourceCandidate> ProgramImpl::hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
    const BasePlanImpl& b = *base.impl_;
    const auto& data      = qwen_common::PreparedPromptAccess::view(*b.prompt);
    std::vector<SourceCandidate> out;
    if (prefix_) {
        // A source held only on the Disk tier is read first and the request waits for it, a
        // paused request's restore too (spec §17.1 rule 7).
        if (b.reuse && prefix_disk_fetch(data, maximum_frontier)) { return {}; }
        std::optional<PrefixSelection> chosen = prefix_select(data, b, kNoLane);
        const std::uint32_t reuse = chosen && chosen->snapshot ? chosen->frontier : 0U;
        // Coalescing only defers a fresh request (an empty list: the Engine asks again).
        if (b.reuse && maximum_frontier == UINT32_MAX && prefix_await_sibling(data, reuse)) { return {}; }
        if (chosen && chosen->snapshot && chosen->frontier > 0 && chosen->frontier <= maximum_frontier) {
            out.push_back(make_source(b, data, std::move(chosen)));
        }
        out.push_back(make_source(b, data, prefix_root(data, b)));
    } else {
        out.push_back(make_source(b, data, std::nullopt));
    }
    return out;
}

runtime::ResourceReservation ProgramImpl::start_binding(const RequestBasePlan& base, runtime::LaneId destination,
                                                        const SourceCandidate& source, ResumeState* resume) {
    if (source.hybrid == nullptr) { throw std::logic_error("Qwen4Exp: a binding source carries no quote"); }
    if (resume != nullptr && resume->impl_ == nullptr) { throw std::logic_error("Qwen4Exp: a resume carries no state"); }
    const QuoteImpl& q = *source.hybrid;
    const std::uint32_t index = destination.value;
    if (index >= options_.max_concurrency || lanes_[index].phase != Phase::Free || in_transaction()) {
        throw std::logic_error("Qwen4Exp: the binding destination is not free");
    }
    // A request paused after its first token binds its ledger; one paused inside its prompt binds
    // the prompt again, from the deepest cached state (its pause published the frontier).
    if (resume != nullptr && resume->impl_->generated) { return bind_replay(*base.impl_, index, *resume->impl_); }
    Lane& lane = lanes_[index];
    kv_busy();
    const BasePlanImpl& b = *base.impl_;
    const auto& data      = qwen_common::PreparedPromptAccess::view(*b.prompt);
    std::optional<PrefixSelection> selection;
    if (prefix_) {
        if (q.prefix && q.prefix->snapshot) {
            selection = prefix_select(data, b, index);
            if (!selection || !selection->snapshot || selection->frontier != source.reused_tokens) {
                return {}; // the quoted snapshot changed: no shortage, the Engine moves to the root source
            }
        } else {
            selection = prefix_root(data, b);
        }
        prefix_pin(*selection);
    }
    lane.history.assign(data.token_ids.begin(), data.token_ids.end());
    lane.history.reserve(lane.history.size() + q.summary.effective_output_tokens + 1);
    lane.prompt_tokens  = static_cast<std::uint32_t>(lane.history.size());
    // A prompt with media keeps its M-RoPE positions; every other token is its index + delta.
    lane.rope.prompt_tokens = lane.prompt_tokens;
    lane.rope.delta         = data.rope_delta;
    if (data.has_media()) {
        lane.rope.prompt.assign(data.positions.begin(), data.positions.end());
    } else {
        lane.rope.prompt.clear();
    }
    // Reset before activation: a resume's activation sets the lane's frontier and state.
    lane.state_tokens   = 0;
    lane.vision_seconds = 0;
    lane.vision_timer.reset();
    lane.speculative    = {};
    lane.mtp_cells      = 0;
    lane.mtp_live       = mtp_;
    lane.dflash_frontier = 0;
    lane.row = tables_->acquire(static_cast<std::int32_t>(index));
    const auto shortage_of = [&](std::uint32_t need) {
        runtime::ResourceReservation out;
        out.shortage.main_kv_pages = need > pool_->available_pages() ? need - pool_->available_pages() : 0;
        return out;
    };
    if (selection) {
        lane.prefix.reuse  = q.summary.publish_continuation;
        lane.prefix.hashes = data.block_hashes;
        lane.prefix.extras = data.block_extras;
        lane.prefix.trailing_extra = prefix_trailing_extra(data);
        lane.prefix.hints  = data.tap_hints.hints;
        lane.prefix.exclusions = prefix_exclusions(data);
        // Make room before reserving: an elastic pool grows only for what eviction cannot free, so
        // idle cache blocks never hold memory the expert frames could use (design §15.6).
        pc_make_room(selection->need);
        if (pool_->available_pages() < selection->need) { (void)grow_kv(selection->need); }
        auto reservation = pool_->reserve(selection->need);
        if (!reservation) {
            const auto out = shortage_of(selection->need);
            prefix_unpin(*selection);
            lane.row = KVExecutionRowLease{};
            return out;
        }
        if (q.vision) { vision_reserve(lane, q.vision, data); }
        const std::uint64_t restored = prefix_->counters().host_restore_bytes;
        prefix_activate(lane, index, *selection, *reservation, q.pages);
        lane.prefix.cached_tokens  = selection->cached_tokens;
        lane.prefix.restored_bytes = prefix_->counters().host_restore_bytes - restored;
        lane.reservation    = std::move(*reservation);
        lane.prefix.reused  = selection->frontier;
        lane.calls          = selection->plan.ends;
    } else {
        if (pool_->available_pages() < q.pages) { (void)grow_kv(q.pages); }
        auto reservation = pool_->reserve(q.pages);
        if (!reservation) {
            const auto out = shortage_of(q.pages);
            lane.row = KVExecutionRowLease{};
            return out;
        }
        if (q.vision) { vision_reserve(lane, q.vision, data); }
        lane.pages.clear();
        lane.pages.reserve(q.pages);
        pool_->materialize(*reservation, q.pages, lane.pages);
        lane.reservation = std::move(*reservation);
        tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
        reset_slot(index);
        lane.calls = plan_prefill(data, 0, {}, false).ends;
    }
    lane.next_call = 0;
    clear_prompt_uses(lane);
    lane.mtp_accept.fill(kAcceptancePrior);
    lane.mtp_policy_rounds = 0;
    lane.dflash_calibration.reset();
    lane.ngram_accept      = kNgramAcceptancePriors;
    if (max_width_ > 1) {
        if (options_.ngram_draft_tokens > 0) {
            lane.proposer =
                std::make_unique<qwen_common::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
            for (const auto token : lane.history) { lane.proposer->append(token); }
        }
        lane.speculative.enabled      = true;
        lane.speculative.backend      = mtp_       ? SpeculativeBackend::Mtp
                                        : dflash2_ ? SpeculativeBackend::DFlash2
                                                   : SpeculativeBackend::None;
        lane.speculative.draft_window = static_cast<std::uint32_t>(max_width_ - 1);
        lane.speculative.accepted_per_position.assign(static_cast<std::size_t>(max_width_ - 1), 0);
    }
    lane.sampling       = q.sampling;
    lane.readout        = b.readout_tokens;
    lane.prompt         = b.prompt;
    lane.constraint     = b.constraint;
    lane.constraint_round.clear();
    lane.constraint_trace.clear();
    lane.sampling.token_counts =
        static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
    *nonfinite_flag(index)  = 0; // the lane's earlier rounds have completed
    lane.sampling.nonfinite = nonfinite_flag(index);
    lane.epoch           = ++next_epoch_;
    lane.phase           = Phase::Prefill;
    lane.prefill_ns      = 0;
    lane.decode_ns       = 0;
    lane.decode_share_ns = 0;
    lane.cache_at_admission = residency_->stats();
    lane.cpu_served_at_admission = cpu_service_ ? cpu_service_->served_experts() : 0;
    if (tier_) { lane.tier_at_admission = tier_->stats(); }
    lane.verify_at_admission = {verify_rounds_, verify_columns_, verify_padded_};
    lane.ngram                   = {};
    lane.gate                    = {};
    lane.begin = runtime::BeginSummary{.prompt_tokens        = lane.prompt_tokens,
                                       .reused_prompt_tokens = source.reused_tokens,
                                       .prefix_reuse_path =
                                           source.reused_tokens ? source.reuse_path : PrefixReusePath::Root,
                                       .prefix_from_disk = source.reused_tokens != 0 && source.from_disk};
    CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(gate_stats_.p) + kGateStatsBytes * index, 0, kGateStatsBytes,
                               device_.stream));
    lane.prefill_end = lane.prompt_tokens;
    lane.replay      = false;
    lane.extent      = b.positions;
    // A request paused inside its prompt keeps what it had measured and decided.
    if (resume != nullptr) { restore_request(lane, *resume->impl_); }
    transaction_lane_ = index;
    replaying_        = false;
    ++revision_;
    runtime::ResourceReservation out;
    out.reserved = true;
    return out;
}

void ProgramImpl::save_request(const Lane& lane, ResumeStateImpl& r) const {
    r.begin                   = lane.begin;
    r.prefill_ns              = lane.prefill_ns;
    r.decode_ns               = lane.decode_ns;
    r.decode_share_ns         = lane.decode_share_ns;
    r.vision_seconds          = vision_seconds_of(lane);
    r.speculative             = lane.speculative;
    r.mtp_accept              = lane.mtp_accept;
    r.mtp_policy_rounds       = lane.mtp_policy_rounds;
    r.dflash_calibration      = lane.dflash_calibration;
    r.ngram_accept            = lane.ngram_accept;
    r.constraint_trace        = lane.constraint_trace;
    r.cache_at_admission      = lane.cache_at_admission;
    r.cpu_served_at_admission = lane.cpu_served_at_admission;
    r.ngram                   = lane.ngram;
    r.gate                    = lane.gate;
}

void ProgramImpl::restore_request(Lane& lane, const ResumeStateImpl& r) {
    lane.begin                   = r.begin;
    lane.prefill_ns              = r.prefill_ns;
    lane.decode_ns               = r.decode_ns;
    lane.decode_share_ns         = r.decode_share_ns;
    lane.vision_seconds          = r.vision_seconds;
    lane.speculative             = r.speculative;
    lane.mtp_accept              = r.mtp_accept;
    lane.mtp_policy_rounds       = r.mtp_policy_rounds;
    lane.dflash_calibration      = r.dflash_calibration;
    lane.ngram_accept            = r.ngram_accept;
    lane.constraint_trace        = r.constraint_trace;
    lane.cache_at_admission      = r.cache_at_admission;
    lane.cpu_served_at_admission = r.cpu_served_at_admission;
    lane.ngram                   = r.ngram;
    lane.gate                    = r.gate;
}

runtime::ResourceReservation ProgramImpl::bind_replay(const BasePlanImpl& b, std::uint32_t index, ResumeStateImpl& r) {
    Lane& lane = lanes_[index];
    kv_busy();
    const auto& prompt = qwen_common::PreparedPromptAccess::view(*b.prompt);
    const qwen_common::PreparedPromptData ledger = ledger_prompt(prompt, r);
    const std::uint32_t pages = binding_pages(b, r.target + 1U);
    std::optional<PrefixSelection> selection;
    if (prefix_) {
        // The pause's capture lands first, so the selection can see it.
        prefix_->drain();
        selection = b.reuse ? prefix_select(ledger, b, index) : prefix_root(ledger, b);
        if (!selection) {
            runtime::ResourceReservation out;
            out.shortage.main_kv_pages = pages > pool_->available_pages() ? pages - pool_->available_pages() : 1U;
            return out;
        }
        // A state cached at the paused frontier itself leaves nothing to replay.
        selection->plan = selection->frontier < r.target
                              ? prefix::plan_calls(selection->frontier, r.target, static_cast<std::uint32_t>(chunk_), {},
                                                   call_cost())
                              : prefix::CallPlan{};
        prefix_pin(*selection);
    }
    lane.history       = r.history;
    lane.prompt_tokens = r.prompt_tokens;
    lane.rope.prompt_tokens = r.prompt_tokens;
    lane.rope.delta         = prompt.rope_delta;
    if (prompt.has_media()) {
        lane.rope.prompt.assign(prompt.positions.begin(), prompt.positions.end());
    } else {
        lane.rope.prompt.clear();
    }
    lane.state_tokens   = 0;
    lane.vision_seconds = 0;
    lane.vision_timer.reset();
    lane.speculative    = {};
    lane.mtp_cells      = 0;
    lane.mtp_live       = mtp_;
    lane.dflash_frontier = 0;
    lane.row            = tables_->acquire(static_cast<std::int32_t>(index));
    if (selection) {
        lane.prefix.reuse          = b.reuse;
        lane.prefix.hashes         = ledger.block_hashes;
        lane.prefix.extras         = prompt.block_extras;
        lane.prefix.trailing_extra = prefix_trailing_extra(prompt);
        lane.prefix.hints.clear();
        lane.prefix.exclusions = prefix_exclusions(prompt);
        pc_make_room(selection->need);
        if (pool_->available_pages() < selection->need) { (void)grow_kv(selection->need); }
        auto reservation = pool_->reserve(selection->need);
        if (!reservation) {
            runtime::ResourceReservation out;
            out.shortage.main_kv_pages = selection->need - std::min(selection->need, pool_->available_pages());
            prefix_unpin(*selection);
            lane.row = KVExecutionRowLease{};
            return out;
        }
        if (b.vision) { vision_reserve(lane, plan_vision(ledger, selection->frontier), ledger); }
        const std::uint64_t restored = prefix_->counters().host_restore_bytes;
        prefix_activate(lane, index, *selection, *reservation, pages);
        lane.prefix.cached_tokens  = selection->cached_tokens;
        lane.prefix.restored_bytes = prefix_->counters().host_restore_bytes - restored;
        lane.prefix.taps.clear();
        lane.reservation   = std::move(*reservation);
        lane.prefix.reused = selection->frontier;
        lane.calls         = selection->plan.ends;
    } else {
        if (pool_->available_pages() < pages) { (void)grow_kv(pages); }
        auto reservation = pool_->reserve(pages);
        if (!reservation) {
            runtime::ResourceReservation out;
            out.shortage.main_kv_pages = pages - std::min(pages, pool_->available_pages());
            lane.row = KVExecutionRowLease{};
            return out;
        }
        if (b.vision) { vision_reserve(lane, plan_vision(ledger, 0), ledger); }
        lane.pages.clear();
        lane.pages.reserve(pages);
        pool_->materialize(*reservation, pages, lane.pages);
        lane.reservation = std::move(*reservation);
        tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
        reset_slot(index);
        lane.calls = prefix::plan_calls(0, r.target, static_cast<std::uint32_t>(chunk_), {}, call_cost()).ends;
    }
    lane.next_call = 0;
    clear_prompt_uses(lane);
    if (max_width_ > 1 && options_.ngram_draft_tokens > 0) {
        lane.proposer = std::make_unique<qwen_common::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
        for (const auto token : lane.history) { lane.proposer->append(token); }
    }
    lane.sampling = b.sampling;
    lane.sampling.token_counts =
        static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
    *nonfinite_flag(index)  = 0;
    lane.sampling.nonfinite = nonfinite_flag(index);
    rebuild_token_counts(lane);
    lane.readout.clear(); // the prompt's readout was delivered before the pause
    lane.prompt     = b.prompt;
    lane.constraint = b.constraint;
    lane.constraint_round.clear();
    lane.epoch = ++next_epoch_;
    restore_request(lane, r);
    CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(gate_stats_.p) + kGateStatsBytes * index, 0, kGateStatsBytes,
                               device_.stream));
    lane.extent      = b.positions;
    lane.prefill_end = r.target;
    // The cached state may reach the paused frontier itself: then nothing replays.
    lane.replay = lane.state_tokens < r.target;
    lane.phase  = lane.replay ? Phase::Prefill : Phase::Decode;
    if (!lane.replay && lane.vision) { vision_release(lane); }
    transaction_lane_ = index;
    replaying_        = lane.replay;
    ++revision_;
    runtime::ResourceReservation out;
    out.reserved = true;
    return out;
}

void ProgramImpl::rebuild_token_counts(const Lane& lane) {
    const cudaStream_t s = device_.stream;
    auto* counts         = lane.sampling.token_counts;
    if (lane.sampling.presence_penalty == 0.0F && lane.sampling.frequency_penalty == 0.0F) {
        CUDA_CHECK(cudaMemsetAsync(counts, 0, 4ULL * token_domain_, s));
        return;
    }
    std::vector<std::int32_t> host(static_cast<std::size_t>(token_domain_), 0);
    for (std::size_t i = lane.prompt_tokens; i < lane.history.size(); ++i) {
        const std::int32_t token = lane.history[i];
        if (token >= 0 && token < token_domain_) { ++host[static_cast<std::size_t>(token)]; }
    }
    CUDA_CHECK(cudaMemcpyAsync(counts, host.data(), 4ULL * token_domain_, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s)); // the host table is read before it goes out of scope
}

bool ProgramImpl::start_pause(SequenceHandle sequence) {
    if (in_transaction() || pending_transaction_ != 0) {
        diagnostic(std::string("Qwen4Exp: pause refused: ") +
                       (in_transaction() ? "a context transaction is open" : "a round awaits its commit"),
                   DiagnosticLevel::Warning);
        return false;
    }
    const std::uint32_t index = lane_of(sequence);
    Lane& lane                = lanes_[index];
    if (lane.phase != Phase::Prefill && lane.phase != Phase::Decode) {
        diagnostic("Qwen4Exp: pause refused: lane " + std::to_string(index) + " is in phase " +
                       std::to_string(static_cast<int>(lane.phase)),
                   DiagnosticLevel::Warning);
        return false;
    }
    // Inside a layer walk's span the layers are at different positions: the pause publishes
    // nothing past the span's start (its taps did), and the walk is abandoned with the lane.
    const bool walking = walk_.active && walk_.lane == index;
    device_.synchronize();
    auto saved           = std::make_unique<ResumeStateImpl>();
    saved->generated     = lane.replay || lane.phase == Phase::Decode;
    saved->target        = lane.replay ? lane.prefill_end : lane.state_tokens;
    saved->history       = lane.history;
    saved->prompt_tokens = lane.prompt_tokens;
    save_request(lane, *saved);
    // A replaying lane short of its frontier publishes what it has rebuilt so far.
    if (!walking) { prefix_finish(lane, index); }
    saved->capture = lane.prefix.capture;
    release(index);
    pause_.emplace(ResumeState(std::move(saved)));
    ++revision_;
    return true;
}

ReplayProgress ProgramImpl::advance_replay(SequenceHandle sequence) {
    const std::uint32_t index = lane_of(sequence);
    if (!lanes_[index].replay || lanes_[index].phase != Phase::Prefill) {
        throw std::logic_error("Qwen4Exp: replay on a lane that is not replaying");
    }
    const PrefillProgress progress = advance_prefill(sequence, nullptr); // replay samples nothing
    ReplayProgress out;
    out.processed_tokens = progress.processed_prompt_tokens;
    out.complete         = progress.complete;
    out.timing           = progress.timing;
    return out;
}

runtime::ResourceReservation ProgramImpl::reserve_units(std::span<const ExecutionUnit> units) {
    runtime::ResourceReservation out;
    for (const ExecutionUnit& unit : units) {
        const std::uint32_t index = lane_of(unit.sequence);
        Lane& lane                = lanes_[index];
        std::uint32_t end         = lane.state_tokens + round_positions();
        if (unit.kind == ExecutionUnitKind::Control) { end += unit.tokens; }
        if (unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) {
            end = lane.prefill_end + round_positions();
        }
        const std::uint32_t needed = (std::min(end, lane.extent) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::uint32_t have   = lane.prefix.page_base + static_cast<std::uint32_t>(lane.pages.size());
        if (needed <= have) { continue; }
        const std::uint32_t missing = needed - have;
        settle_round(); // growing the KV pool may take expert frames: a deferred cache update lands first
        kv_busy();
        if (prefix_) { pc_make_room(missing); }
        if (pool_->available_pages() < missing) { (void)grow_kv(missing); }
        if (pool_->available_pages() < missing && prefix_ && prefix_->transfers_pending()) {
            // Blocks pinned only by their in-flight Host writes (a lane just paused or finished)
            // become evictable once those land: wait for them before reporting a shortage.
            prefix_->drain();
            pc_make_room(missing);
        }
        auto reservation = pool_->reserve(missing);
        if (!reservation) {
            out.shortage.main_kv_pages += std::max(1U, missing - std::min(missing, pool_->available_pages()));
            return out;
        }
        std::vector<DeviceKVPageLease> grown;
        grown.reserve(missing);
        pool_->materialize(*reservation, missing, grown);
        tables_->publish(lane.row.handle(), static_cast<std::int32_t>(have), std::span<const DeviceKVPageLease>(grown),
                         device_.stream);
        for (DeviceKVPageLease& page : grown) { lane.pages.push_back(std::move(page)); }
        ++revision_;
    }
    out.reserved = true;
    return out;
}

ContextProgress ProgramImpl::poll_context(runtime::CancellationFlagView cancellation) {
    if (pause_) {
        ContextProgress out{.kind = ContextOperationKind::Pause};
        out.advanced  = true;
        out.complete  = true;
        out.published = true;
        out.paused.emplace(std::move(*pause_));
        pause_.reset();
        return out;
    }
    if (!transaction_lane_) { throw std::logic_error("Qwen4Exp: there is no context operation"); }
    ContextProgress out{.kind = ContextOperationKind::Bind};
    Lane& admitted = lanes_[*transaction_lane_];
    if (cancellation.requested()) {
        release(*transaction_lane_);
        transaction_lane_.reset();
        out.complete = true;
        return out;
    }
    if (admitted.vision && !vision_encoded(admitted)) {
        try {
            vision_step(admitted);
        } catch (...) {
            release(*transaction_lane_);
            transaction_lane_.reset();
            throw;
        }
        out.advanced = true;
        if (!vision_encoded(admitted)) { return out; }
    }
    out.advanced  = true;
    out.complete  = true;
    out.published = true;
    out.replaying = replaying_;
    out.sequence  = handle(*transaction_lane_);
    transaction_lane_.reset();
    return out;
}

bool ProgramImpl::context_blocks(SequenceHandle sequence) const noexcept {
    const std::uint32_t index = ContractAccess::lane(sequence);
    if (transaction_lane_ && *transaction_lane_ == index) { return true; }
    return walk_.active && walk_.lane != index && index < lanes_.size() && lanes_[index].phase == Phase::Prefill;
}

bool ProgramImpl::hybrid_reclaim(runtime::ContextResourceUsage shortage) {
    if (shortage.main_kv_pages == 0) { return false; }
    const std::uint32_t before = pool_->available_pages();
    const std::uint32_t target = before + shortage.main_kv_pages;
    if (prefix_) {
        pc_make_room(target);
        if (pool_->available_pages() == before && prefix_->transfers_pending()) {
            // Blocks pinned only by in-flight Host writes become evictable once those land.
            prefix_->drain();
            pc_make_room(target);
        }
    }
    // A long prompt's stream ring and wide arena stay lent until its last call, and growth is
    // refused while frames are lent, so a request queued behind it waited out its prefill (issue
    // #24). Between calls they can go back: the prompt's next streamed call lends again from
    // the smaller cache, or stages its experts (the same bytes either way, design §13.3).
    // Only when the pool's capacity leaves room for the growth: a return that cannot grow the pool
    // only makes the prompt's next walk span lend again (the measured no-growth return is design
    // §9.4).
    const bool room = pool_->capacity_pages() - pool_->backed_pages() >=
                      target - std::min(target, pool_->available_pages());
    if (pool_->available_pages() < target && room && leases_returnable()) {
        return_stream_lease();
        return_wide_work();
        (void)grow_kv(target);
    }
    return pool_->available_pages() > before;
}

bool ProgramImpl::leases_returnable() const noexcept {
    if (!kv_vmm_ || !residency_ || walk_.active || wide_active_) { return false; }
    const std::uint64_t lent = residency_->stats().lent_frames;
    const std::uint64_t returnable = (stream_lease_.valid() ? stream_lease_.count : 0U) +
                                     (work_lease_.valid() ? work_lease_.count : 0U);
    return lent != 0 && lent == returnable;
}

} // namespace infernix::models::qwen4_exp::detail
