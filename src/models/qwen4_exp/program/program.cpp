#include "models/qwen4_exp/program/program_impl.h"

#include <atomic>

namespace infernix::models::qwen4_exp {

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept : impl_(std::move(impl)) {}
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;
std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->target : 0U; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}

Program::~Program() noexcept = default;

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const Config& config,
                                       std::uint32_t public_tokens) {
    return detail::ProgramImpl::plan_device(options, config, static_cast<std::int32_t>(public_tokens)).bytes;
}

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_layout().bytes; }

const VramSizing& Program::vram_sizing() const noexcept { return impl_->vram_sizing(); }

void Program::set_maintenance_waker(std::function<void()> waker) { impl_->set_maintenance_waker(std::move(waker)); }

void Program::maintain() {
    impl_->apply_vram_target(true);
    impl_->maintain_expert_state();
}

Program::ExpertPlacementStatus Program::expert_placement_status() const {
    return impl_->placement_status();
}

bool Program::request_expert_placement(const ExpertPlacementRequest& request, std::string* error) {
    return impl_->request_placement(request, error);
}

bool Program::request_expert_count(std::uint32_t vram_experts, std::uint32_t ram_experts, std::string* error) {
    return impl_->request_count_placement(vram_experts, ram_experts, error);
}

bool Program::export_expert_placement(const std::filesystem::path& path, std::string* error) {
    return impl_->export_placement(path, error);
}

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept { return impl_->feasible(base); }

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
    return impl_->hybrid_sources(base, maximum_frontier);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind, std::uint32_t) {
    const runtime::ResourceReservation reserved = impl_->start_binding(base, lane, source, resume);
    BindingReservation out;
    out.reserved = static_cast<bool>(reserved);
    out.shortage = reserved.shortage;
    return out;
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    return impl_->poll_context(cancellation);
}

bool Program::has_context_transaction() const noexcept { return impl_->in_transaction(); }

bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

bool Program::hybrid_reclaim(runtime::ContextResourceUsage shortage) { return impl_->hybrid_reclaim(shortage); }
std::optional<std::uint32_t> Program::hybrid_prefetch(const RequestBasePlan& base) { return impl_->hybrid_prefetch(base); }
std::uint32_t Program::hybrid_prefetch_room() const noexcept { return impl_->hybrid_prefetch_room(); }
void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) { impl_->hybrid_hold_queue(queue); }
std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->hybrid_cache_epoch(); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Program::start_pause(SequenceHandle sequence, runtime::ExecutionTiming*) {
    return impl_->start_pause(sequence);
}

ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming*) {
    return impl_->advance_replay(sequence);
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, masks);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming*) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

void Program::fail_all_cleanup() noexcept { impl_->release_all(); }

bool Program::recover_after_failure() noexcept { return impl_->recover_after_failure(); }

void Program::shutdown_cleanup() noexcept {
    impl_->report_cache_layers();
    impl_->release_all();
    impl_->save_expert_state();
    impl_->spill_prefix_cache_for_shutdown();
}

PrefixCacheDiskOpen Program::attach_prefix_cache_disk(const std::filesystem::path& directory,
                                                      std::string_view fingerprint) {
    return impl_->attach_prefix_cache_disk(directory, fingerprint);
}

std::optional<PrefixCacheDiskSpill> Program::prefix_shutdown_spill() const { return impl_->prefix_shutdown_spill(); }

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    return impl_->causal_score(std::move(prompt), first_target, options);
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

namespace testing {

namespace {
std::atomic<std::uint32_t> g_expert_fault{0};
} // namespace

void set_expert_fault(std::uint32_t checks) noexcept { g_expert_fault.store(checks); }

bool take_expert_fault() noexcept {
    std::uint32_t left = g_expert_fault.load(std::memory_order_relaxed);
    while (left != 0) {
        if (g_expert_fault.compare_exchange_weak(left, left - 1)) { return left == 1; }
    }
    return false;
}

} // namespace testing

} // namespace infernix::models::qwen4_exp
