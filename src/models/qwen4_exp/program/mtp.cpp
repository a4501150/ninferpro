// The MTP drafter and the draft-length policies (design 11.2).

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

void ProgramImpl::allocate_mtp(std::int32_t lanes, std::uint64_t& allocated) {
    const std::size_t column = plan_.mtp_column_bytes;
    mtp_columns_ = plan_.mtp_columns;
    allocate(mtp_residuals_, column * lanes * max_width_, allocated);
    allocate(mtp_saved_, column * lanes, allocated);
    allocate(mtp_chain_, column * lanes, allocated);
    allocate(mtp_records_, 2ULL * di_ * max_width_ * lanes, allocated);
    mtp_saved_.fill(0);
    std::vector<float> ones(static_cast<std::size_t>(c_.hc.streams) * mtp_columns_, 1.0F);
    allocate(mtp_ones_, plan_.mtp_ones, allocated);
    mtp_ones_.copy_from_host(ones.data(), ones.size() * sizeof(float));
    mtp_io_ = plan_.mtp_io;
    allocate(mtp_device_, 4ULL * mtp_io_.words, allocated);
    mtp_host_ = PinnedHostBuffer(4ULL * mtp_io_.words);
    for (auto& d : mtp_drafts_) { d.assign(static_cast<std::size_t>(mtp_k_), 0); }
    mtp_catch_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
    mtp_draft_graphs_.resize(static_cast<std::size_t>(lanes) * mtp_k_);
    mtp_gated_graphs_.resize(static_cast<std::size_t>(mtp_k_));
    if (sampled_drafts_) { mtp_sampled_graphs_.resize(static_cast<std::size_t>(lanes) * mtp_k_); }
}

bool ProgramImpl::mtp_settled(const Lane& lane) const noexcept {
    return !mtp_ || !lane.mtp_live || lane.mtp_cells == lane.state_tokens ||
           lane.mtp_cells + 1U == lane.state_tokens;
}

void ProgramImpl::require_mtp_settled(Lane& lane) const noexcept {
    if (mtp_settled(lane)) { return; }
    lane.mtp_live = false;
    diagnostic("Qwen4Exp: the MTP drafter's cells are out of step with the model state; drafting is "
             "off for the rest of this request", DiagnosticLevel::Warning);
}

std::optional<execution::MtpChunk> ProgramImpl::stage_mtp_chunk(Lane& lane, std::uint32_t index, std::int32_t begin,
                                                                std::int32_t width) {
    if (!mtp_) { return std::nullopt; }
    const bool prepend = begin > 0 && lane.mtp_cells < static_cast<std::uint32_t>(begin);
    const bool known   = static_cast<std::int32_t>(lane.history.size()) > begin + width;
    const std::int32_t columns = known ? width : width - 1;
    const std::int32_t first   = prepend ? begin - 1 : begin;
    const std::int32_t cells   = (prepend ? 1 : 0) + columns;
    auto* ids   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_ids);
    auto* cells_host = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_cells);
    for (std::int32_t i = 0; i < cells; ++i) {
        cells_host[i] = first + i;
        ids[i]        = lane.history[static_cast<std::size_t>(first + i + 1)];
    }
    // Cell c takes the target's RoPE position of c (design §16.4), per sub-chunk.
    const std::int32_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
    stage_mtp_chunk_rope(lane.rope, static_cast<std::uint32_t>(first), cells, execution::kMtpChunkColumns,
                         static_cast<std::uint32_t>(r_), io_words(io_layout_.mtp_rope, 3ULL * cells),
                         io_words(io_layout_.mtp_block_rope, 3ULL * subchunks));
    lane.mtp_cells = static_cast<std::uint32_t>(begin + columns);
    lane.mtp_live  = true;
    auto* base = static_cast<std::byte*>(io_device_.p);
    execution::MtpChunk chunk;
    chunk.saved     = saved_column(index);
    chunk.prepend   = prepend;
    chunk.columns   = columns;
    // A one-token prompt has no cell yet (its only position is the pending cell).
    if (cells > 0) {
        chunk.ids              = Tensor(base + io_layout_.mtp_ids, DType::I32, {cells});
        chunk.positions        = Tensor(base + io_layout_.mtp_cells, DType::I32, {cells});
        chunk.rope_positions   = reinterpret_cast<std::int32_t*>(base + io_layout_.mtp_rope);
        chunk.block_start_rope = reinterpret_cast<std::int32_t*>(base + io_layout_.mtp_block_rope);
        chunk.vision           = stage_mtp_vision(lane, static_cast<std::uint32_t>(first), cells, subchunks);
    }
    return chunk;
}

std::int32_t ProgramImpl::choose_draft_length(Lane& lane) const {
    const std::int32_t most = neural_k();
    // DFlash2 proposes its whole block; its length is chosen after the proposal (dflash2_auto_length).
    if (!options_.auto_draft_length || dflash2_) { return most; }
    std::int32_t best = 0;
    double best_score = 1.0;
    double reach = 1.0, tokens = 1.0;
    const double cost = width_cost();
    for (std::int32_t k = 1; k <= most; ++k) {
        reach *= lane.mtp_accept[static_cast<std::size_t>(k - 1)];
        tokens += reach;
        const double score = tokens / (1.0 + cost * k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    if (++lane.mtp_policy_rounds % kProbeInterval == 0) { best = std::min(best + 1, most); }
    return best;
}

void ProgramImpl::reset_round_costs(const char* why) {
    round_cost_       = {};
    batch_cost_       = {};
    last_round_steps_ = -1;
    last_round_batch_ = 0;
    last_round_extra_ = -1;
    diagnostic(std::string("speculation cost fits reset: ") + why);
}

bool ProgramImpl::batch_cost_model(std::int32_t batch, double& a_p, double& fixed, double& b) const noexcept {
    const BatchCostFit& fit = batch_cost_[static_cast<std::size_t>(batch)];
    if (fit.plain_n < kPlainMinRounds) { return false; }
    const RoundCostFit& f = fit.draft;
    const double spread  = f.n * f.kk - f.k * f.k;
    if (f.n < kCostMinRounds || spread < kCostMinSpread * f.n * f.n) { return false; }
    b = (f.n * f.kt - f.k * f.t) / spread;
    const double a_d = (f.t - b * f.k) / f.n;
    a_p   = fit.plain_t / fit.plain_n;
    fixed = std::max(0.0, a_d - a_p);
    return a_p > 0.0 && a_d > 0.0 && b > 0.0;
}

bool ProgramImpl::round_cost_fit(double& a, double& b) const noexcept {
    const RoundCostFit& f = round_cost_;
    const double spread   = f.n * f.kk - f.k * f.k;
    if (f.n < kCostMinRounds || spread < kCostMinSpread * f.n * f.n) { return false; }
    b = (f.n * f.kt - f.k * f.t) / spread;
    a = (f.t - b * f.k) / f.n;
    return a > 0.0 && b > 0.0;
}

double ProgramImpl::width_cost() const noexcept {
    double a = 0.0, b = 0.0;
    return round_cost_fit(a, b) ? std::min(b / a, kCostMax) : kWidthCost;
}

void ProgramImpl::observe_round(Clock::time_point start, std::int32_t batch) {
    if (last_round_steps_ >= 0 && batch == 1) {
        const double seconds = std::chrono::duration<double>(start - last_round_start_).count();
        RoundCostFit& f      = round_cost_;
        const double k       = last_round_steps_;
        double a = 0.0, b = 0.0;
        // Before the fit, a cycle past kCostGapRatio mean cycles is a gap.
        const double predict = round_cost_fit(a, b) ? a + b * k : (f.n > 0.0 ? f.t / f.n : seconds);
        if (seconds < kCostGapRatio * predict) {
            f.n  = kCostDecay * f.n + 1.0;
            f.k  = kCostDecay * f.k + k;
            f.kk = kCostDecay * f.kk + k * k;
            f.t  = kCostDecay * f.t + seconds;
            f.kt = kCostDecay * f.kt + k * seconds;
        }
    }
    // The batched DFlash2 policy: consecutive decode rounds of the same size time that size's
    // plain and drafting rounds (the interval holds drafting, verification, commit and host work,
    // as round_cost_'s does; a different size or an idle gap in between is not a sample). The
    // zero-column rounds feed the plain EMA, the drafting rounds the columns regression.
    if (dflash2_ && last_round_extra_ >= 0 && last_round_batch_ == batch && batch >= 2) {
        const double seconds = std::chrono::duration<double>(start - last_round_start_).count();
        BatchCostFit& fit    = batch_cost_[static_cast<std::size_t>(batch)];
        const double columns = last_round_extra_;
        if (columns == 0.0) {
            const double predict = fit.plain_n > 0.0 ? fit.plain_t / fit.plain_n : seconds;
            if (seconds < kCostGapRatio * predict) {
                fit.plain_n = kCostDecay * fit.plain_n + 1.0;
                fit.plain_t = kCostDecay * fit.plain_t + seconds;
            }
        } else {
            RoundCostFit& f = fit.draft;
            double a_p = 0.0, fixed = 0.0, b = 0.0;
            const double predict = batch_cost_model(batch, a_p, fixed, b)
                                       ? a_p + fixed + b * columns
                                       : (f.n > 0.0 ? f.t / f.n : seconds);
            if (seconds < kCostGapRatio * predict) {
                f.n  = kCostDecay * f.n + 1.0;
                f.k  = kCostDecay * f.k + columns;
                f.kk = kCostDecay * f.kk + columns * columns;
                f.t  = kCostDecay * f.t + seconds;
                f.kt = kCostDecay * f.kt + columns * seconds;
            }
        }
    }
    last_round_start_ = start;
    last_round_batch_ = batch;
    last_round_steps_ = -1; // set by the round once its drafts are known
    last_round_extra_ = -1;  // set by the round once its width is known
}

std::int32_t ProgramImpl::choose_pair_draft_length(const Lane& a, const Lane& b) {
    if (!drafter_live(a) || !drafter_live(b)) { return 0; }
    const std::int32_t most = std::min(kPairMaxDrafts, neural_k());
    if (!options_.auto_draft_length) { return most; }
    std::int32_t best = 0;
    double best_score = 2.0;
    double reach_a = 1.0, reach_b = 1.0, tokens = 2.0;
    for (std::int32_t k = 1; k <= most; ++k) {
        reach_a *= a.mtp_accept[static_cast<std::size_t>(k - 1)];
        reach_b *= b.mtp_accept[static_cast<std::size_t>(k - 1)];
        tokens += reach_a + reach_b;
        const double score = tokens / (1.0 + kPairWidthCost * k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    if (++pair_policy_rounds_ % kProbeInterval == 0) { best = std::min(best + 1, most); }
    return best;
}

std::int32_t ProgramImpl::choose_batch_dflash_length(std::span<const std::uint32_t> lanes) {
    const auto batch = static_cast<std::int32_t>(lanes.size());
    if (batch < 2) { return 0; }
    const std::int32_t most = std::min(dflash_k_, max_width_ - 1);
    if (most <= 0) { return 0; }
    std::array<std::span<const double>, kMaximumConcurrency> rows{};
    std::int32_t drafting = 0;
    for (const std::uint32_t lane : lanes) {
        const Lane& l = lanes_[lane];
        if (drafter_live(l)) {
            rows[static_cast<std::size_t>(drafting++)] = std::span<const double>(l.mtp_accept.data(), 8);
        }
    }
    if (drafting == 0) { return 0; }
    if (!options_.auto_draft_length) { return most; }
    // The cost of drafting k columns per row, in plain rounds of this size: 1 + fixed +
    // per_draft * k, the fixed overhead (the drafter pass, selector, sampling) plus the
    // marginal column, until the size's own plain and drafting rounds have been timed. The
    // warm-up prior drafts at two rows (the one-row prior measured drafting pays there on every
    // card tested) and stays plain from three (wide rounds paid only where columns were much
    // cheaper than rows -- the offloaded 5090 -- which is what the fit is there to learn);
    // the probes keep both sides measured either way.
    double per_draft = batch == 2 ? kDFlashDraftCost : kCostMax;
    double fixed_cost = 0.0;
    double a_p = 0.0, f_secs = 0.0, b = 0.0;
    const bool fitted = batch_cost_model(batch, a_p, f_secs, b);
    if (fitted) {
        per_draft  = std::min(kCostMax, b * static_cast<double>(batch) / a_p);
        fixed_cost = std::min(kCostMax, f_secs / a_p);
    }
    if ((batch_policy_rounds_ & 63U) == 0U) {
        const BatchCostFit& fit = batch_cost_[static_cast<std::size_t>(batch)];
        char line[224];
        std::snprintf(line, sizeof(line),
                      "DFlash2 batch policy: %d rows, plain n=%.0f a=%.1f ms, draft n=%.0f "
                      "fixed=%.1f ms b=%.2f ms/column -> per draft %.2f + fixed %.2f plain rounds, most %d",
                      batch, fit.plain_n, fitted ? a_p * 1e3 : 0.0, fit.draft.n,
                      fitted ? f_secs * 1e3 : 0.0, fitted ? b * 1e3 : 0.0, per_draft, fixed_cost, most);
        diagnostic(line);
    }
    const std::int32_t best = shared_draft_length(
        std::span<const std::span<const double>>(rows.data(), static_cast<std::size_t>(drafting)),
        batch - drafting, most, per_draft, fixed_cost, kBatchDraftMargin);
    // The probe keeps the fit honest: while drafting wins, every second interval one draft more
    // than the choice (the higher lengths' acceptances and costs stay measured) and every other
    // interval a plain round (the plain EMA's sample -- without it the plain round time goes
    // stale and the drafting rounds' fixed overhead is priced into the columns it does not
    // belong to). While plain wins, that often would give away a round costing twice a plain
    // one every 2 * kProbeInterval rounds (design §11.5 holds the measured cost), so once the size
    // is fitted the probe thins to every kProbeLossFactor'th cycle.
    // Until then it probes every cycle, so a size's fit forms within a few hundred rounds instead
    // of thousands. The probes alternate one and two drafts: the columns fit needs their spread.
    const std::uint64_t round = ++batch_policy_rounds_;
    if (best == 0) {
        const std::uint64_t period = 2 * kProbeInterval * (fitted ? kProbeLossFactor : 1);
        if (round % period == 0) { return static_cast<std::int32_t>(std::min<std::uint64_t>(round / period % 2U + 1, most)); }
        return 0;
    }
    const std::uint64_t cycle = round % (2 * kProbeInterval);
    if (cycle == 0) { return std::min(best + 1, most); }
    if (cycle == kProbeInterval) { return 0; }
    return best;
}

void ProgramImpl::mtp_draft(std::span<const std::uint32_t> lanes, std::int32_t steps,
                            const std::function<void()>& while_drafting) {
    const cudaStream_t s = device_.stream;
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    const std::size_t column = 2ULL * width_;
    mtp_sampled_round_       = false;
    bool any = false, unwritten = false;
    for (std::int32_t b = 0; b < batch; ++b) {
        any |= lanes_[lanes[b]].mtp_live;
        unwritten |= lanes_[lanes[b]].mtp_live && lanes_[lanes[b]].mtp_cells < lanes_[lanes[b]].state_tokens;
    }
    // Without drafts (a gated or zero-length round) only unwritten pending cells need the drafter.
    if (!any || (steps == 0 && !unwritten)) { return; }
    auto* slots = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
    auto* rows  = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
    for (std::int32_t b = 0; b < batch; ++b) {
        const Lane& lane = lanes_[lanes[b]];
        const std::int32_t cell  = static_cast<std::int32_t>(lane.state_tokens) - 1;
        // The lane's last mapped position: its shared prefix pages, then its private ones.
        const std::int32_t limit =
            static_cast<std::int32_t>(lane.prefix.page_base + lane.pages.size()) * kPagedKVPageSize - 1;
        mtp_host(mtp_io_.ids)[b] = lane.history.back();
        for (std::int32_t j = 0; j < mtp_k_; ++j) {
            const std::int32_t at = std::min(cell + j, limit);
            mtp_host(mtp_io_.cells)[j * batch + b] = at;
            const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
            stage_rope_value(rope_of(lane.rope, static_cast<std::uint32_t>(at)),
                             std::span<std::int32_t>(mtp_host(mtp_io_.rope_cells + step), 3ULL * batch), batch, b);
            stage_rope_value(block_start_rope(lane.rope, static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(r_)),
                             std::span<std::int32_t>(mtp_host(mtp_io_.block_rope_cells + step), 3ULL * batch), batch, b);
        }
        slots[b] = static_cast<std::int32_t>(lanes[b]);
        rows[b]  = static_cast<std::int32_t>(lanes[b]);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mtp_chain_.p) + column * b, saved_column(lanes[b]).data,
                                   column, cudaMemcpyDeviceToDevice, s));
    }
    auto* io = static_cast<std::byte*>(io_device_.p);
    upload_pinned(io + io_layout_.slots, slots, 4ULL * batch, s);
    upload_pinned(io + io_layout_.rows, rows, 4ULL * batch, s);
    upload_pinned(mtp_device_.p, mtp_host_.data(), 4ULL * mtp_io_.drafts, s);
    const Tensor slot_tensor(io + io_layout_.slots, DType::I32, {batch});
    const Tensor row_tensor(io + io_layout_.rows, DType::I32, {batch});
    Tensor chain(mtp_chain_.p, DType::BF16, {width_, batch});
    Tensor anchors(mtp_device(mtp_io_.ids), DType::I32, {batch});
    if (unwritten) {
        // The pending cells' K/V (rewriting a written cell is idempotent).
        set_expert_route();
        forward_->run_mtp({.residuals        = chain,
                           .ids              = anchors,
                           .positions        = Tensor(mtp_device(mtp_io_.cells), DType::I32, {batch}),
                           .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells), DType::I32, {batch, 3}),
                           .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells), DType::I32, {batch, 3}),
                           .slots            = slot_tensor,
                           .table_rows       = row_tensor,
                           .batch            = batch,
                           .width            = 1,
                           .kv_only          = true});
    }
    if (steps == 0) {
        // No synchronize here: the uploads above may still be queued when the caller stages the
        // verification round into the same pinned slots and rows, which is safe only because it
        // stages the same lanes in the same order (and mtp_host_ is next written after that
        // round's synchronize). A caller that stages other lanes there must synchronize first.
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            if (lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
        }
        return;
    }
    // Sampled drafts when a row samples (greedy rows draft the argmax either way).
    const bool sampled = sampled_drafts_ && std::any_of(lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
                             return lanes_[lane].sampling.temperature > 0.0F;
                         });
    mtp_sampled_round_ = sampled;
    if (sampled) {
        // The rows' samplers for the draws inside the draft graph (verify uploads them again,
        // with grammar masks, after this round's synchronize).
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        for (std::int32_t b = 0; b < batch; ++b) {
            configs[b]      = lanes_[lanes[b]].sampling;
            configs[b].mask = {};
            configs[b].nonfinite = nullptr; // a draft's selection never fails the request
        }
        upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    }
    const auto draft_step = [&](std::int32_t j) {
        Tensor drafts(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j) * batch), DType::I32, {batch});
        const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
        // Sampled drafts: step j's top-rank scratch and proposal planes.
        const auto plane = [&](std::size_t offset, std::int32_t rows, DType type) {
            return sampled
                       ? Tensor(mtp_device(offset + static_cast<std::size_t>(rows) * batch * j), type, {rows, batch})
                       : Tensor{};
        };
        constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
        set_expert_route();
        forward_->run_mtp({.residuals    = chain,
                           .ids          = j == 0 ? anchors
                                                  : Tensor(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j - 1) * batch),
                                                           DType::I32, {batch}),
                           .positions    = Tensor(mtp_device(mtp_io_.cells + static_cast<std::size_t>(j) * batch),
                                                  DType::I32, {batch}),
                           .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells + step), DType::I32, {batch, 3}),
                           .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells + step), DType::I32, {batch, 3}),
                           .slots        = slot_tensor,
                           .table_rows   = row_tensor,
                           .batch        = batch,
                           .width        = 1,
                           .kv_only      = false,
                           .residual_out = chain,
                           .drafts       = drafts,
                           .draft_logprobs =
                               Tensor(mtp_device(mtp_io_.logprobs + static_cast<std::size_t>(j) * batch), DType::FP32,
                                      {1, batch}),
                           .sample_configs =
                               sampled ? static_cast<const ops::SamplingConfig*>(configs_.p) : nullptr,
                           .draft_step       = static_cast<std::uint32_t>(j),
                           .sample_rows      = plane(mtp_io_.sample_rows, kSampledDraftRanks, DType::I32),
                           .sample_logprobs  = plane(mtp_io_.sample_logprobs, kSampledDraftRanks, DType::FP32),
                           .draft_candidates = plane(mtp_io_.draft_candidates, C, DType::I32),
                           .draft_q          = plane(mtp_io_.draft_q, C, DType::FP32)});
    };
    // A one-row argmax round verifies its drafts only up to the first below --spec-min-p, so a
    // draft step runs only while every earlier draft cleared it: the device applies the host's
    // cut (the same binary64 comparison) and skips the steps whose drafts it would discard. A
    // skipped step leaves its draft and confidence words stale, beyond the cut that reads them.
    // (The measured share of skipped steps and the decode gain are design §11.2.)
    if (!sampled && batch == 1 && steps > 1 && std::isfinite(draft_min_logprob_)) {
        replay_gated(mtp_gated_graphs_[static_cast<std::size_t>(steps - 1)], steps, draft_step,
                     [&](std::int32_t j, cudaGraphConditionalHandle next) {
                         const auto* confidence =
                             reinterpret_cast<const float*>(mtp_device(mtp_io_.logprobs + static_cast<std::size_t>(j)));
                         launch_graph_gate(confidence, draft_min_logprob_, next, s);
                     });
    } else {
        auto& draft_graphs = sampled ? mtp_sampled_graphs_ : mtp_draft_graphs_;
        replay(draft_graphs[static_cast<std::size_t>(batch - 1) * mtp_k_ + (steps - 1)], [&] {
            for (std::int32_t j = 0; j < steps; ++j) { draft_step(j); }
        });
    }
    CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.drafts), mtp_device(mtp_io_.drafts), 4ULL * steps * batch,
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.logprobs), mtp_device(mtp_io_.logprobs), 4ULL * steps * batch,
                               cudaMemcpyDeviceToHost, s));
    if (while_drafting) {
        device_.flush();
        while_drafting();
    }
    device_.synchronize();
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane = lanes_[lanes[b]];
        if (unwritten && lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
        for (std::int32_t j = 0; j < steps; ++j) {
            mtp_drafts_[b][static_cast<std::size_t>(j)] = mtp_host(mtp_io_.drafts)[j * batch + b];
        }
        // The row's confident drafts: those before the first below draft_min_logprob_ (every
        // sampled draft with sampled-mtp-drafts: a sampled proposal's acceptance does not follow
        // the head's top probability, and its rounds were measured faster without the cut).
        mtp_confident_[b] = steps;
        if (sampled && lane.sampling.temperature > 0.0F) { continue; }
        const auto* logprobs = reinterpret_cast<const float*>(mtp_host(mtp_io_.logprobs));
        for (std::int32_t j = 0; j < steps; ++j) {
            if (logprobs[j * batch + b] < draft_min_logprob_) {
                mtp_confident_[b] = j;
                break;
            }
        }
    }
}

void ProgramImpl::mtp_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit) {
    const cudaStream_t s = device_.stream;
    const auto batch     = static_cast<std::int32_t>(rows.size());
    const std::size_t column = 2ULL * width_;
    auto* io = static_cast<std::byte*>(io_device_.p);
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int32_t k = commit[b];
        for (std::int32_t j = 0; j < W; ++j) {
            mtp_host(mtp_io_.up_ids)[b * W + j] =
                j < k ? pending_tokens_[static_cast<std::size_t>(b * W + j)] : pending_tokens_[static_cast<std::size_t>(b * W)];
        }
    }
    upload_pinned(mtp_device(mtp_io_.up_ids), mtp_host(mtp_io_.up_ids), 4ULL * batch * W, s);
    const Tensor positions(io + io_layout_.positions, DType::I32, {W * batch});
    const Tensor rope(io + io_layout_.rope, DType::I32, {W * batch, 3});
    const Tensor block_rope(io + io_layout_.block_rope, DType::I32, {batch, 3});
    const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
    replay(mtp_catch_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)], [&] {
    set_expert_route();
    forward_->run_mtp({.residuals   = Tensor(mtp_residuals_.p, DType::BF16, {width_, W * batch}),
                       .ids         = Tensor(mtp_device(mtp_io_.up_ids), DType::I32, {W * batch}),
                       .positions   = positions,
                       .rope_positions   = rope,
                       .block_start_rope = block_rope,
                       .slots       = slots,
                       .table_rows  = Tensor(io + io_layout_.rows, DType::I32, {batch}),
                       .batch       = batch,
                       .width       = W,
                       .kv_only     = true,
                       .key_records = Tensor(mtp_records_.p, DType::BF16, {di_, W, batch})});
    Tensor tails(static_cast<std::byte*>(tails_backing_.p) +
                     static_cast<std::size_t>(c_.attention_layers) * di_ * (r_ - 1) * options_.max_concurrency * 2,
                 DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), 1});
    ops::qsa_commit_tails(Tensor(mtp_records_.p, DType::BF16, {di_, W, batch, 1}), positions.view({W, batch}),
                          Tensor(spec_device(spec_layout_.commit), DType::I32, {batch}), tails, slots,
                          execution::qsa_geometry(c_), s);
    });
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int32_t k = commit[b];
        const auto lane      = ContractAccess::lane(rows[static_cast<std::size_t>(b)]);
        if (k < 1) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(saved_column(lane).data,
                                   static_cast<const std::byte*>(mtp_residuals_.p) + column * (b * W + k - 1),
                                   column, cudaMemcpyDeviceToDevice, s));
        lanes_[lane].mtp_cells = lanes_[lane].state_tokens;
    }
}

void ProgramImpl::sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
    const cudaStream_t s = device_.stream;
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    // The sampler reads the FP32 logits, or their BF16 rounding (design §17.7); a constraint
    // masks them in place (the prompt readout has read them first).
    Tensor narrow = read_logits(batch);
    auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
    auto* sample_positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
    for (std::int32_t b = 0; b < batch; ++b) {
        configs[b]          = lanes_[lanes[b]].sampling;
        configs[b].mask     = grammar_mask(static_cast<std::size_t>(b), {});
        reasoning_ban(static_cast<std::size_t>(b), configs[b]);
        sample_positions[b] = positions[b];
    }
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    upload_pinned(sample_pos_.p, sample_positions, 4ULL * batch, s);
    const bool constrained = stage_constraints(lanes, 1);
    if (constrained) { constrain(narrow, nullptr); }
    Tensor out(sampled_.p, DType::I32, {batch});
    Tensor logical(sample_pos_.p, DType::I32, {batch});
    {
        auto scope = work_->scope();
        ops::sample(narrow, out, token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p), logical,
                    ops::kSamplePurposeDecode, *work_, s);
    }
    CUDA_CHECK(cudaMemcpyAsync(host_sampled_.data(), sampled_.p, 4ULL * batch, cudaMemcpyDeviceToHost, s));
    {
        const nvtx::ScopedRange wait(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control);
        device_.synchronize();
    }
    check_device_error();
    const auto* tokens = static_cast<const std::int32_t*>(host_sampled_.data());
    for (std::int32_t b = 0; b < batch; ++b) { pending_tokens_[b] = tokens[b]; }
    if (constrained) {
        for (std::int32_t b = 0; b < batch; ++b) {
            take_constraint_draws(lanes_[lanes[b]], static_cast<std::size_t>(b),
                                  std::span<const std::int32_t>(&pending_tokens_[b], 1));
        }
    }
}

} // namespace infernix::models::qwen4_exp::detail
