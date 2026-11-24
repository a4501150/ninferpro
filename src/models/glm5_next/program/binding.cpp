// Request admission and binding: a request's plan, its root source and its KV pages. A binding
// reserves the request's whole extent (prompt plus effective output), so no unit of it grows,
// nothing pauses it and there is nothing to replay.

#include "models/glm5_next/program/program_impl.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::glm5_next::detail {

namespace {

ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
    if (!(s.temperature >= 0.0F) || !(s.top_p > 0.0F && s.top_p <= 1.0F) || s.min_p < 0.0F || s.min_p > 1.0F) {
        throw std::invalid_argument("Glm5Next: sampling parameters are out of range");
    }
    ops::SamplingConfig out;
    out.temperature       = s.temperature;
    out.top_k             = s.top_k;
    out.top_p             = s.top_p;
    out.min_p             = s.min_p;
    out.presence_penalty  = s.presence_penalty;
    out.frequency_penalty = s.frequency_penalty;
    out.seed              = s.seed;
    return out;
}

} // namespace

RequestBasePlan ProgramImpl::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    const auto& data = PreparedPromptAccess::view(prompt);
    if (data.has_media()) { throw std::invalid_argument("Glm5Next: image and video prompts are not supported yet"); }
    const auto n = static_cast<std::uint32_t>(data.token_ids.size());
    if (n == 0 || n > options_.max_context) {
        throw std::invalid_argument("Glm5Next: the prompt is empty or exceeds max_context");
    }
    for (const TokenId id : options.readout_tokens) {
        if (id < 0 || id >= token_domain_) {
            throw std::invalid_argument("readout token is outside the public token domain");
        }
    }
    auto base                             = std::make_unique<BasePlanImpl>();
    base->readout_tokens                  = options.readout_tokens;
    base->constraint                      = compile_constraint(options.constraint);
    base->summary.prompt_tokens           = n;
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output   = options_.max_context - n + 1U;
    base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                ? FinishReason::OutputLimit
                                                : FinishReason::ContextCapacity;
    base->summary.publish_continuation = false; // no prefix cache to publish to
    base->sampling                     = translate(options.sampling);
    base->positions = std::min(options_.max_context, n + base->summary.effective_output_tokens);
    base->pages     = (base->positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
    base->prompt    = std::make_shared<const PreparedPrompt>(std::move(prompt));
    return RequestBasePlan(std::move(base));
}

bool ProgramImpl::feasible(const RequestBasePlan& base) const noexcept {
    return base.impl_ != nullptr && base.impl_->pages <= pool_->capacity_pages() &&
           base.impl_->pages <= static_cast<std::uint32_t>(plan_.pages_per_row);
}

std::vector<SourceCandidate> ProgramImpl::hybrid_sources(const RequestBasePlan& base) {
    auto quote     = std::make_shared<QuoteImpl>();
    quote->pages   = base.impl_->pages;
    SourceCandidate out;
    out.hybrid = std::move(quote);
    return {std::move(out)};
}

std::vector<std::uint32_t> ProgramImpl::plan_calls(std::uint32_t begin, std::uint32_t end) const {
    std::vector<std::uint32_t> ends;
    for (std::uint32_t at = begin; at < end;) {
        at = std::min(end, at + static_cast<std::uint32_t>(chunk_));
        ends.push_back(at);
    }
    return ends;
}

void ProgramImpl::reset_slot(std::uint32_t slot) {
    const cudaStream_t s = device_.stream;
    for (std::uint32_t l = 0; l < c_.kda_layers; ++l) {
        Tensor conv      = kda_->conv_slot(l, static_cast<std::int32_t>(slot));
        Tensor recurrent = kda_->recurrent_slot(l, static_cast<std::int32_t>(slot));
        CUDA_CHECK(cudaMemsetAsync(conv.data, 0, conv.bytes(), s));
        CUDA_CHECK(cudaMemsetAsync(recurrent.data, 0, recurrent.bytes(), s));
    }
    CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + slot * static_cast<std::size_t>(token_domain_),
                               0, 4ULL * token_domain_, s));
}

runtime::ResourceReservation ProgramImpl::start_binding(const RequestBasePlan& base, runtime::LaneId destination,
                                                        const SourceCandidate& source, ResumeState* resume) {
    if (source.hybrid == nullptr) { throw std::logic_error("Glm5Next: a binding source carries no quote"); }
    if (resume != nullptr) { throw std::logic_error("Glm5Next: this Program never pauses a request"); }
    const std::uint32_t index = destination.value;
    if (index >= options_.max_concurrency || lanes_[index].phase != Phase::Free || in_transaction()) {
        throw std::logic_error("Glm5Next: the binding destination is not free");
    }
    const BasePlanImpl& b = *base.impl_;
    const QuoteImpl& q    = *source.hybrid;
    Lane& lane            = lanes_[index];
    auto reservation      = pool_->reserve(q.pages);
    if (!reservation) {
        runtime::ResourceReservation out;
        out.shortage.main_kv_pages = q.pages - std::min(q.pages, pool_->available_pages());
        return out;
    }
    const auto& data = PreparedPromptAccess::view(*b.prompt);
    lane.history.assign(data.token_ids.begin(), data.token_ids.end());
    lane.history.reserve(lane.history.size() + b.summary.effective_output_tokens + 1);
    lane.prompt_tokens = static_cast<std::uint32_t>(lane.history.size());
    lane.state_tokens  = 0;
    lane.extent        = b.positions;
    lane.row           = tables_->acquire(static_cast<std::int32_t>(index));
    lane.pages.clear();
    lane.pages.reserve(q.pages);
    pool_->materialize(*reservation, q.pages, lane.pages);
    lane.reservation = std::move(*reservation);
    tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
    reset_slot(index);
    lane.calls     = plan_calls(0, lane.prompt_tokens);
    lane.next_call = 0;
    std::fill(lane.prompt_uses.begin(), lane.prompt_uses.end(), 0U);
    lane.prompt_counted = 0;
    lane.sampling       = b.sampling;
    lane.sampling.token_counts =
        static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
    *nonfinite_flag(index)  = 0; // the lane's earlier rounds have completed
    lane.sampling.nonfinite = nonfinite_flag(index);
    lane.readout            = b.readout_tokens;
    lane.prompt             = b.prompt;
    lane.constraint         = b.constraint;
    lane.constraint_round.clear();
    lane.constraint_trace.clear();
    lane.epoch           = ++next_epoch_;
    lane.phase           = Phase::Prefill;
    lane.prefill_ns      = 0;
    lane.decode_ns       = 0;
    lane.decode_share_ns = 0;
    lane.speculative     = {};
    if (drafter_) {
        lane.speculative.enabled      = true;
        lane.speculative.backend      = SpeculativeBackend::DFlash2;
        lane.speculative.draft_window = static_cast<std::uint32_t>(plan_.draft);
        lane.speculative.accepted_per_position.assign(static_cast<std::size_t>(plan_.draft), 0);
    }
    lane.cache_at_admission = residency_->stats();
    lane.begin        = runtime::BeginSummary{.prompt_tokens = lane.prompt_tokens};
    transaction_lane_ = index;
    runtime::ResourceReservation out;
    out.reserved = true;
    return out;
}

ContextProgress ProgramImpl::poll_context(runtime::CancellationFlagView cancellation) {
    if (!transaction_lane_) { throw std::logic_error("Glm5Next: there is no context operation"); }
    ContextProgress out{.kind = ContextOperationKind::Bind};
    if (cancellation.requested()) {
        release(*transaction_lane_);
        transaction_lane_.reset();
        out.complete = true;
        return out;
    }
    out.advanced  = true;
    out.complete  = true;
    out.published = true;
    out.sequence  = handle(*transaction_lane_);
    transaction_lane_.reset();
    return out;
}

bool ProgramImpl::context_blocks(SequenceHandle sequence) const noexcept {
    return transaction_lane_ && *transaction_lane_ == ContractAccess::lane(sequence);
}

runtime::ResourceReservation ProgramImpl::reserve_units(std::span<const ExecutionUnit> units) {
    // Every lane holds its whole extent from admission; a unit past it is a contract violation.
    for (const ExecutionUnit& unit : units) {
        const Lane& lane         = lanes_[lane_of(unit.sequence)];
        const std::uint32_t need = unit.kind == ExecutionUnitKind::Control ? lane.state_tokens + 1U + unit.tokens
                                                                           : lane.state_tokens + 1U;
        if (unit.kind != ExecutionUnitKind::Prefill && std::min(need, lane.extent) >
                                                           static_cast<std::uint32_t>(lane.pages.size()) * kPagedKVPageSize) {
            throw std::logic_error("Glm5Next: a unit runs past its request's reserved extent");
        }
    }
    runtime::ResourceReservation out;
    out.reserved = true;
    return out;
}

bool ProgramImpl::start_pause(SequenceHandle) {
    diagnostic("Glm5Next: pause refused: requests hold their whole extent", DiagnosticLevel::Warning);
    return false;
}

ReplayProgress ProgramImpl::advance_replay(SequenceHandle) {
    throw std::logic_error("Glm5Next: replay on a Program that never pauses");
}

} // namespace infernix::models::glm5_next::detail
