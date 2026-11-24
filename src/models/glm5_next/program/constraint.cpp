// /v1/decide for Glm5Next: constrained generation (TokenConstraint), grammar masks and prompt
// readouts. A round's constrained rows mask their logits with ops::constrain_logits before the
// sampler reads them; the probability record of the token a row produced becomes the request's
// ConstrainedDraw as it commits. A readout reads the prompt's next-token distribution from the
// logits its first token is sampled from, before any mask. Every round samples one column per row.

#include "models/glm5_next/program/program_impl.h"

#include "infernix/ops/target_logprobs.h"
#include "infernix/ops/top_logprobs.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace infernix::models::glm5_next::detail {

namespace {

constexpr std::size_t kSets     = kMaximumConstraintSets;
constexpr std::size_t kChoices  = ops::kTokenConstraintChoices;
constexpr std::size_t kRecord   = ops::kTokenConstraintRecord;
constexpr std::size_t kObserve  = ops::kTokenObserveTokens;
constexpr std::size_t kObserved = ops::kTokenObserveRecord;

} // namespace

// The constraint_ buffer: choices [kChoices, lanes x kSets], counts [lanes x kSets], descriptors
// [lanes], records [kRecord, lanes], observe [kObserve, lanes], observed [kObserved, lanes] (all
// 4-byte words; one column per lane).
std::shared_ptr<const ConstraintPlan> ProgramImpl::compile_constraint(const TokenConstraint& constraint) {
    if (constraint.empty()) { return nullptr; }
    auto plan = std::make_shared<ConstraintPlan>();
    for (const std::vector<TokenId>& step : constraint.steps) {
        if (step.empty() || step.size() > kMaximumConstraintChoices) {
            throw std::invalid_argument("constraint step has an invalid token count");
        }
        for (const TokenId id : step) {
            if (id < 0 || id >= token_domain_) {
                throw std::invalid_argument("constraint token is outside the public token domain");
            }
        }
        if (step.size() == 1) {
            plan->steps.push_back(-2 - step.front());
            continue;
        }
        auto set = std::find(plan->sets.begin(), plan->sets.end(), step);
        if (set == plan->sets.end()) {
            if (plan->sets.size() == kSets) { throw std::invalid_argument("constraint uses too many distinct token sets"); }
            set = plan->sets.insert(plan->sets.end(), step);
        }
        plan->steps.push_back(static_cast<std::int32_t>(set - plan->sets.begin()));
    }
    for (const std::vector<TokenId>& observed : constraint.observe) {
        if (observed.size() > kObserve) { throw std::invalid_argument("constraint step observes too many tokens"); }
        for (const TokenId id : observed) {
            if (id < 0 || id >= token_domain_) {
                throw std::invalid_argument("observed token is outside the public token domain");
            }
        }
    }
    plan->observe = constraint.observe;
    plan->serial  = next_constraint_serial_++;
    return plan;
}

bool ProgramImpl::stage_constraints(std::span<const std::uint32_t> lanes) {
    constraint_columns_ = 0;
    if (std::none_of(lanes.begin(), lanes.end(), [&](std::uint32_t l) { return lanes_[l].constraint != nullptr; })) {
        return false;
    }
    const cudaStream_t s = device_.stream;
    const std::size_t columns = lanes.size();
    if (columns > constraint_host_.size()) { throw std::logic_error("Glm5Next: a constrained round is wider than its table"); }
    const std::size_t sets_total = static_cast<std::size_t>(options_.max_concurrency) * kSets;
    auto* const words            = static_cast<std::int32_t*>(constraint_.p);
    std::fill(constraint_host_.begin(), constraint_host_.begin() + static_cast<std::ptrdiff_t>(columns), -1);
    for (std::size_t row = 0; row < columns; ++row) {
        const std::uint32_t index = lanes[row];
        const Lane& lane          = lanes_[index];
        if (!lane.constraint) { continue; }
        const ConstraintPlan& plan = *lane.constraint;
        if (constraint_serials_[index] != plan.serial) {
            // The lane's set columns: written once per request.
            std::vector<std::int32_t> choices(kSets * kChoices, 0), counts(kSets, 0);
            for (std::size_t set = 0; set < plan.sets.size(); ++set) {
                std::copy(plan.sets[set].begin(), plan.sets[set].end(),
                          choices.begin() + static_cast<std::ptrdiff_t>(set * kChoices));
                counts[set] = static_cast<std::int32_t>(plan.sets[set].size());
            }
            CUDA_CHECK(cudaMemcpyAsync(words + index * kSets * kChoices, choices.data(), 4ULL * choices.size(),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(words + sets_total * kChoices + index * kSets, counts.data(), 4ULL * counts.size(),
                                       cudaMemcpyHostToDevice, s));
            constraint_serials_[index] = plan.serial;
        }
        // The row samples output step (generated tokens).
        const std::size_t step = lane.history.size() - lane.prompt_tokens;
        if (step < plan.steps.size()) {
            const std::int32_t d = plan.steps[step];
            constraint_host_[row] = d >= 0 ? static_cast<std::int32_t>(index * kSets) + d : d;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(words + sets_total * (kChoices + 1), constraint_host_.data(), 4ULL * columns,
                               cudaMemcpyHostToDevice, s));
    // Every constrained round stages its observe lists: -1 where the step observes nothing.
    std::vector<std::int32_t> observe(columns * kObserve, -1);
    for (std::size_t row = 0; row < columns; ++row) {
        const Lane& lane = lanes_[lanes[row]];
        if (!lane.constraint || lane.constraint->observe.empty()) { continue; }
        const std::size_t step = lane.history.size() - lane.prompt_tokens;
        if (step < lane.constraint->observe.size()) {
            const std::vector<TokenId>& tokens = lane.constraint->observe[step];
            std::copy(tokens.begin(), tokens.end(), observe.begin() + static_cast<std::ptrdiff_t>(row * kObserve));
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(words + observe_offset(), observe.data(), 4ULL * observe.size(), cudaMemcpyHostToDevice,
                               s));
    constraint_columns_ = static_cast<std::int32_t>(columns);
    return true;
}

std::size_t ProgramImpl::observe_offset() const noexcept {
    const std::size_t sets_total = static_cast<std::size_t>(options_.max_concurrency) * kSets;
    return sets_total * (kChoices + 1) + constraint_host_.size() * (1 + kRecord);
}

void ProgramImpl::constrain(Tensor& logits) {
    const std::int32_t columns = constraint_columns_;
    if (columns == 0 || logits.ne[1] != columns) {
        throw std::logic_error("Glm5Next: constrained logits do not match the staged round");
    }
    const auto sets_total = static_cast<std::int32_t>(options_.max_concurrency * kSets);
    auto* const words     = static_cast<std::int32_t*>(constraint_.p);
    const Tensor choices(words, DType::I32, {static_cast<std::int32_t>(kChoices), sets_total});
    const Tensor counts(words + sets_total * kChoices, DType::I32, {sets_total});
    const Tensor descriptors(words + sets_total * (kChoices + 1), DType::I32, {columns});
    Tensor records(words + sets_total * (kChoices + 1) + constraint_host_.size(), DType::FP32,
                   {static_cast<std::int32_t>(kRecord), columns});
    const Tensor observe(words + observe_offset(), DType::I32, {static_cast<std::int32_t>(kObserve), columns});
    Tensor observed(words + observe_offset() + constraint_host_.size() * kObserve, DType::FP32,
                    {static_cast<std::int32_t>(kObserved), columns});
    ops::constrain_logits(logits, nullptr, descriptors, choices, counts, token_domain_, records, &observe, &observed,
                          device_.stream);
}

void ProgramImpl::take_constraint_draws(Lane& lane, std::size_t column, TokenId token) {
    if (!lane.constraint) { return; }
    const ConstraintPlan& plan = *lane.constraint;
    if (column >= static_cast<std::size_t>(constraint_columns_)) {
        throw std::logic_error("Glm5Next: constraint records are outside the round");
    }
    const std::size_t sets_total = static_cast<std::size_t>(options_.max_concurrency) * kSets;
    const auto* records = reinterpret_cast<const float*>(static_cast<const std::int32_t*>(constraint_.p) +
                                                         sets_total * (kChoices + 1) + constraint_host_.size()) +
                          column * kRecord;
    std::vector<float> host(kRecord);
    CUDA_CHECK(cudaMemcpy(host.data(), records, 4ULL * host.size(), cudaMemcpyDeviceToHost));
    std::vector<float> observed;
    if (!plan.observe.empty()) {
        observed.resize(kObserved);
        const auto* values = reinterpret_cast<const float*>(static_cast<const std::int32_t*>(constraint_.p) +
                                                            observe_offset() + constraint_host_.size() * kObserve) +
                             column * kObserved;
        CUDA_CHECK(cudaMemcpy(observed.data(), values, 4ULL * observed.size(), cudaMemcpyDeviceToHost));
    }
    const std::size_t step = lane.history.size() - lane.prompt_tokens;
    lane.constraint_round.clear();
    if (step >= plan.steps.size()) {
        throw std::logic_error("Glm5Next: a constrained request produced a token past its last step");
    }
    const std::int32_t d    = plan.steps[step];
    const std::size_t count = d >= 0 ? plan.sets[static_cast<std::size_t>(d)].size() : 1U;
    ConstrainedDraw draw{.token         = token,
                         .probabilities = std::vector<float>(host.data(), host.data() + count),
                         .mass          = host[kChoices]};
    if (!plan.observe.empty() && step < plan.observe.size()) {
        draw.observed_logprobs.assign(observed.data(), observed.data() + plan.observe[step].size());
        draw.top_logprob = observed[kObserve];
    }
    lane.constraint_round.push_back(std::move(draw));
}

void ProgramImpl::commit_constraint_draws(Lane& lane) {
    if (lane.constraint_round.empty()) { return; }
    lane.constraint_trace.insert(lane.constraint_trace.end(), std::make_move_iterator(lane.constraint_round.begin()),
                                 std::make_move_iterator(lane.constraint_round.end()));
    lane.constraint_round.clear();
}

PromptReadout ProgramImpl::read_prompt_frontier(const Lane& lane) {
    const std::size_t n = lane.readout.size();
    if (n == 0 || n > kMaximumReadoutTokens) { throw std::invalid_argument("prompt readout names an invalid token count"); }
    const cudaStream_t s = device_.stream;
    // The logits the sampler reads next (before its constraints mask them).
    Tensor logits      = read_logits(1);
    auto scope         = work_->scope();
    const auto count   = static_cast<std::int32_t>(n);
    Tensor ids         = work_->alloc(DType::I32, {count, 1});
    Tensor logprobs    = work_->alloc(DType::FP32, {count, 1});
    Tensor top_id      = work_->alloc(DType::I32, {1, 1});
    Tensor top_logprob = work_->alloc(DType::FP32, {1, 1});
    CUDA_CHECK(cudaMemcpyAsync(ids.data, lane.readout.data(), 4ULL * n, cudaMemcpyHostToDevice, s));
    ops::target_logprobs(logits, ids, token_domain_, logprobs, s);
    ops::top_logprobs(logits, token_domain_, top_id, top_logprob, s);
    PromptReadout readout;
    readout.logprobs.resize(n);
    CUDA_CHECK(cudaMemcpyAsync(readout.logprobs.data(), logprobs.data, 4ULL * n, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&readout.top_token, top_id.data, sizeof(TokenId), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&readout.top_logprob, top_logprob.data, sizeof(float), cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    return readout;
}

std::size_t ProgramImpl::grammar_mask_bytes(std::int32_t token_domain, std::int32_t lanes,
                                            std::int32_t width) noexcept {
    return 4ULL * ((static_cast<std::size_t>(token_domain) + 31) / 32) *
           static_cast<std::size_t>(lanes) * static_cast<std::size_t>(width);
}

ops::SamplingMask ProgramImpl::grammar_mask(std::size_t row, std::span<const TokenId> drafts) {
    grammar_dead_[row] = 0;
    if (grammar_ == nullptr || !grammar_->constrained(row)) { return {}; }
    if (drafts.size() + 1 > static_cast<std::size_t>(plan_.width)) {
        throw std::logic_error(
            "Glm5Next: a constrained row verifies more positions than its mask holds");
    }
    const std::size_t words  = (static_cast<std::size_t>(token_domain_) + 31) / 32;
    const std::size_t offset = row * static_cast<std::size_t>(plan_.width) * words;
    std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_host_.data()) + offset,
                                  (drafts.size() + 1) * words);
    grammar_dead_[row] = grammar_->fill(row, drafts, host);
    auto* device_words = static_cast<std::uint32_t*>(grammar_masks_.p) + offset;
    // Read through the host mapping: a copy-engine copy would wait behind expert promotions.
    upload_pinned(device_words, host.data(), host.size_bytes(), device_.stream);
    grammar_->uploaded(row, host.size_bytes());
    return {device_words, static_cast<std::int32_t>(words)};
}

void ProgramImpl::reasoning_ban(std::size_t row, ops::SamplingConfig& config) const noexcept {
    const runtime::ReasoningBan ban = grammar_ != nullptr ? grammar_->reasoning_ban(row) : runtime::ReasoningBan{};
    config.reasoning_close   = ban.close;
    config.reasoning_ends[0] = ban.ends[0];
    config.reasoning_ends[1] = ban.ends[1];
}

void ProgramImpl::mark_row_failures(PendingBatch& pending,
                                    std::span<const std::int32_t> counts) noexcept {
    const auto rows = ContractAccess::rows(pending);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        std::int32_t* const nonfinite = nonfinite_flag(ContractAccess::lane(rows[row]));
        // A dead end fails the row only at a position it licensed.
        const std::int32_t count = counts.empty() ? 1 : counts[row];
        const std::uint64_t reached =
            count >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << count) - 1U;
        ContractAccess::set_row_failure(
            pending, row,
            std::exchange(*nonfinite, 0) != 0     ? runtime::PendingRowFailure::NonFiniteLogits
            : (grammar_dead_[row] & reached) != 0 ? runtime::PendingRowFailure::ConstraintDeadEnd
                                                  : runtime::PendingRowFailure::None);
    }
}

std::size_t ProgramImpl::constraint_bytes(std::int32_t lanes) noexcept {
    const auto sets    = static_cast<std::size_t>(lanes) * kMaximumConstraintSets;
    const auto columns = static_cast<std::size_t>(lanes);
    return 4ULL * (sets * ops::kTokenConstraintChoices + sets +
                   columns * (1 + ops::kTokenConstraintRecord + ops::kTokenObserveTokens + ops::kTokenObserveRecord));
}

} // namespace infernix::models::glm5_next::detail
