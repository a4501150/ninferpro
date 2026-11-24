// The DFlash2 drafter in Glm5Next's Program (docs/maintainer/dflash.md): every lane keeps a context
// ring of the drafter's K/V for its last `window` positions, projected from the target's features
// (the stream mean of the tapped layers' completed residuals). A speculating round proposes once
// (one graph per batch size and width), verifies the drafts in one Forward call whose KDA blocks
// read their lanes' states without writing them, and accepts them on the device (greedy, or by
// sparse rejection sampling against the drafter's laws when a row samples). The commit folds each
// row's committed columns into its KDA states and convolution windows and writes their context
// into the ring; the MLA latent and the indexer are position-addressed, so a rejected column's
// entries are rewritten before any query reads them. Prefill calls and forced tokens write their
// last window's context eagerly, plain decode rounds inside their graph. The ring only steers
// drafts, which verification accepts or rejects, so it never changes output.

#include "models/glm5_next/program/program_impl.h"

#include "infernix/ops/argmax.h"
#include "infernix/ops/speculative_round.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::glm5_next::detail {

void ProgramImpl::allocate_dflash2(std::uint64_t& allocated) {
    const auto& io = plan_.dflash_io;
    allocate(features_, plan_.features, allocated);
    // A zero ring: every slot a proposal may read before its context lands holds finite values.
    allocate(ring_backing_, plan_.ring, allocated);
    ring_backing_.fill(0);
    ring_ = std::make_unique<CyclicKVCache>(DeviceSpan{ring_backing_.p, ring_backing_.bytes},
                                            plan_.ring_layout);
    allocate(records_backing_, plan_.records, allocated);
    const std::int32_t W = dim(c_.kda.width()), R = plan_.lanes * plan_.width;
    auto* at        = static_cast<std::byte*>(records_backing_.p);
    const auto take = [&](std::int32_t rows) {
        Tensor out(at, DType::BF16, {rows, R});
        at += align(out.bytes());
        return out;
    };
    for (std::uint32_t l = 0; l < c_.kda_layers; ++l) {
        execution::KdaVerifyRecord record;
        record.qkv    = take(3 * W);
        record.key    = take(W);
        record.value  = take(W);
        record.forget = take(W);
        record.beta   = take(dim(c_.kda.heads));
        records_.push_back(record);
    }
    allocate(dflash_device_, 4ULL * io.words, allocated);
    dflash_device_.fill(0);
    dflash_host_ = PinnedHostBuffer(4ULL * io.words);
    std::fill_n(dflash_host(io.ones), plan_.lanes, 1);
    upload_pinned(dflash_device(io.ones), dflash_host(io.ones), 4ULL * plan_.lanes, device_.stream);
    const auto graphs =
        static_cast<std::size_t>(plan_.lanes) * static_cast<std::size_t>(plan_.draft);
    propose_graphs_.resize(graphs);
    verify_graphs_.resize(graphs);
    context_graphs_.resize(graphs);
}

Tensor ProgramImpl::feature_columns(std::int32_t n) const {
    const std::int32_t rows =
        dim(parameters_.model.config().dflash2->feature_width(c_.hidden_size));
    return Tensor(features_.p, DType::BF16, {rows, n});
}

std::int32_t ProgramImpl::draft_room(const Lane& lane,
                                     const runtime::RoundBudget& budget) const noexcept {
    const std::int64_t room = std::min<std::int64_t>(
        {plan_.draft, static_cast<std::int64_t>(budget.generated_tokens_remaining) - 1,
         static_cast<std::int64_t>(lane.extent) - lane.state_tokens - 1});
    return static_cast<std::int32_t>(std::max<std::int64_t>(room, 0));
}

void ProgramImpl::call_context(std::uint32_t lane, std::int32_t begin, std::int32_t width) {
    const auto& io            = plan_.dflash_io;
    const std::int32_t window = dim(parameters_.model.config().dflash2->window);
    const std::int32_t n      = std::min(width, window);
    std::int32_t* positions   = dflash_host(io.context_positions);
    for (std::int32_t i = 0; i < n; ++i) { positions[i] = begin + width - n + i; }
    *dflash_host(io.context_count) = n;
    *dflash_host(io.context_slot)  = static_cast<std::int32_t>(lane);
    upload_pinned(dflash_device(io.context_positions), positions,
                  4ULL * (io.context_slot + 1 - io.context_positions), device_.stream);
    drafter_->materialize(
        {.features  = feature_columns(n).view({feature_columns(n).ne[0], n, 1}),
         .positions = Tensor(dflash_device(io.context_positions), DType::I32, {n, 1}),
         .counts    = Tensor(dflash_device(io.context_count), DType::I32, {1}),
         .slots     = Tensor(dflash_device(io.context_slot), DType::I32, {1}),
         .min_count = static_cast<std::uint32_t>(n),
         .max_count = static_cast<std::uint32_t>(n)});
}

void ProgramImpl::round_context(std::int32_t batch) {
    auto* io_base         = static_cast<std::byte*>(io_device_.p);
    const Tensor features = feature_columns(batch);
    drafter_->materialize(
        {.features  = features.view({features.ne[0], 1, batch}),
         .positions = Tensor(io_base + plan_.io.positions, DType::I32, {1, batch}),
         .counts    = Tensor(dflash_device(plan_.dflash_io.ones), DType::I32, {batch}),
         .slots     = Tensor(io_base + plan_.io.slots, DType::I32, {batch}),
         .min_count = 1,
         .max_count = 1});
}

void ProgramImpl::propose(std::span<const std::uint32_t> lanes, std::int32_t width) {
    const cudaStream_t s     = device_.stream;
    const auto& io           = plan_.dflash_io;
    const auto batch         = static_cast<std::int32_t>(lanes.size());
    const std::int32_t K     = width - 1;
    constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
    // Each row's block starts at its unprocessed anchor; the ring holds every earlier position. The
    // selector draws a sampled row's drafts with its seed (verification uploads the configs again,
    // with grammar masks).
    auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
    for (std::int32_t b = 0; b < batch; ++b) {
        const Lane& lane             = lanes_[lanes[b]];
        dflash_host(io.anchors)[b]   = lane.history.back();
        dflash_host(io.frontiers)[b] = static_cast<std::int32_t>(lane.state_tokens);
        dflash_host(io.valid)[b]     = width;
        dflash_host(io.slots)[b]     = static_cast<std::int32_t>(lanes[b]);
        configs[b]                   = lane.sampling;
        configs[b].mask              = {};
        configs[b].nonfinite         = nullptr; // a draft's selection never fails the request
    }
    upload_pinned(dflash_device(io.anchors), dflash_host(io.anchors), 4ULL * (io.ids - io.anchors),
                  s);
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    const execution::DFlash2ProposalCall call{
        .anchors       = Tensor(dflash_device(io.anchors), DType::I32, {batch}),
        .frontiers     = Tensor(dflash_device(io.frontiers), DType::I32, {batch}),
        .valid_columns = Tensor(dflash_device(io.valid), DType::I32, {batch}),
        .slots         = Tensor(dflash_device(io.slots), DType::I32, {batch}),
        .configs       = static_cast<const ops::SamplingConfig*>(configs_.p),
        .batch         = batch,
        .width         = width,
        .ids           = Tensor(dflash_device(io.ids), DType::I32, {width, batch}),
        .positions     = Tensor(dflash_device(io.positions), DType::I32, {width, batch}),
        .drafts        = Tensor(dflash_device(io.drafts), DType::I32, {K, batch}),
        .candidates    = Tensor(dflash_device(io.candidates), DType::I32, {C, K, batch}),
        .proposal_q    = Tensor(dflash_device(io.q), DType::FP32, {C, K, batch})};
    replay(graph(propose_graphs_, batch, width), [&] { drafter_->propose(call); });
    CUDA_CHECK(cudaMemcpyAsync(dflash_host(io.drafts), dflash_device(io.drafts), 4ULL * K * batch,
                               cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t j = 0; j < K; ++j) {
            drafts_[b][static_cast<std::size_t>(j)] = dflash_host(io.drafts)[b * K + j];
        }
    }
}

PendingBatch ProgramImpl::speculate(std::span<const SequenceHandle> sequences,
                                    std::span<const std::uint32_t> lanes,
                                    std::span<const std::int32_t> positions, std::int32_t width,
                                    Clock::time_point start) {
    const cudaStream_t s = device_.stream;
    const auto& io       = plan_.dflash_io;
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    const std::int32_t W = width, K = width - 1;
    propose(lanes, W);

    auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.ids);
    auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.positions);
    auto* columns = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.columns);
    auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.slots);
    auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + plan_.io.rows);
    for (std::int32_t b = 0; b < batch; ++b) {
        const Lane& lane     = lanes_[lanes[b]];
        const std::int32_t p = positions[b];
        for (std::int32_t j = 0; j < W; ++j) {
            ids[b * W + j] =
                j == 0 ? lane.history.back() : drafts_[b][static_cast<std::size_t>(j - 1)];
            pos[b * W + j]     = p + j;
            columns[b * W + j] = b * W + j;
        }
        slots[b]                         = static_cast<std::int32_t>(lanes[b]);
        rows[b]                          = static_cast<std::int32_t>(lanes[b]);
        host_lanes_[b]                   = static_cast<std::int32_t>(lanes[b]);
        dflash_host(io.extents)[b]       = K;
        dflash_host(io.lengths)[b]       = p;
        dflash_host(io.round_anchors)[b] = lane.history.back();
    }
    upload_pinned(io_device_.p, io_host_.data(), plan_.io.bytes, s);
    upload_pinned(dflash_device(io.extents), dflash_host(io.extents),
                  4ULL * (io.licensed - io.extents), s);
    before_call(true);
    replay(graph(verify_graphs_, batch, W),
           [&] { forward_call(batch, W, batch * W, /*verify=*/true); });
    enqueue_routes(batch * W);

    // Acceptance on the device, from the logits plain decode samples (FP32, or their BF16
    // rounding).
    Tensor logits = read_logits(batch * W);
    Tensor flat_target(dflash_device(io.target), DType::I32, {W * batch});
    ops::argmax(logits, flat_target, token_domain_, s);
    const Tensor target = flat_target.view({W, batch});
    auto* configs       = static_cast<ops::SamplingConfig*>(host_configs_.data());
    bool sampled        = false;
    for (std::int32_t b = 0; b < batch; ++b) {
        configs[b] = lanes_[lanes[b]].sampling;
        configs[b].mask =
            grammar_mask(static_cast<std::size_t>(b),
                         std::span<const TokenId>(drafts_[b].data(), static_cast<std::size_t>(K)));
        reasoning_ban(static_cast<std::size_t>(b), configs[b]);
        sampled = sampled || configs[b].temperature > 0.0F;
    }
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    const auto* device_configs = static_cast<const ops::SamplingConfig*>(configs_.p);
    Tensor drafts(dflash_device(io.drafts), DType::I32, {K, batch});
    Tensor extents(dflash_device(io.extents), DType::I32, {batch});
    Tensor lengths(dflash_device(io.lengths), DType::I32, {batch});
    Tensor anchors(dflash_device(io.round_anchors), DType::I32, {batch});
    Tensor licensed(dflash_device(io.licensed), DType::I32, {W, batch});
    Tensor counts(dflash_device(io.counts), DType::I32, {batch});
    Tensor accepted(dflash_device(io.accepted), DType::I32, {batch});
    {
        auto scope        = work_->scope();
        const Tensor view = logits.view({vocab_, W, batch});
        if (sampled) {
            // A sampled row's drafts were drawn from the selector's laws (a greedy row's are
            // one-hot): sparse rejection sampling against them.
            constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
            Tensor candidates(dflash_device(io.candidates), DType::I32, {C, K, batch});
            Tensor q(dflash_device(io.q), DType::FP32, {C, K, batch});
            ops::speculative_accept_sparse_drafts(
                target, view, drafts, candidates, q, extents, lengths, anchors, licensed, counts,
                accepted, token_domain_, device_configs, ops::SpeculativeAcceptExecutionEnvelope{},
                *work_, s);
            // The sparse acceptance leaves the token counts to the caller.
            ops::increment_licensed_token_counts(licensed, counts, device_configs, s);
        } else {
            ops::speculative_accept_greedy_drafts(target, view, drafts, extents, lengths, anchors,
                                                  licensed, counts, accepted, token_domain_,
                                                  device_configs, *work_, s);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(dflash_host(io.licensed), licensed.data,
                               4ULL * (io.accepted + plan_.lanes - io.licensed),
                               cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    check_device_error();

    const std::int32_t* licensed_host = dflash_host(io.licensed);
    std::int32_t advanced             = 1;
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane           = lanes_[lanes[b]];
        const std::int32_t L = dflash_host(io.counts)[b], A = dflash_host(io.accepted)[b];
        if (L < 1 || L > W || A != L - 1) {
            throw std::logic_error("Glm5Next: acceptance returned an invalid extent");
        }
        for (std::int32_t j = 0; j < W; ++j) {
            pending_tokens_[static_cast<std::size_t>(b * W + j)] =
                j < L ? licensed_host[b * W + j] : 0;
        }
        pending_counts_[static_cast<std::size_t>(b)] = L;
        advanced                                     = std::max(advanced, L);
        auto& stats                                  = lane.speculative;
        ++stats.rounds;
        stats.drafted_tokens += static_cast<std::uint64_t>(K);
        stats.accepted_tokens += static_cast<std::uint64_t>(A);
        for (std::int32_t j = 0; j < A; ++j) {
            ++stats.accepted_per_position[static_cast<std::size_t>(j)];
        }
    }
    after_call(batch * W, decode_budget(static_cast<std::uint32_t>(advanced)));
    round_width_ = W;
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane = lanes_[lanes[b]];
        lane.decode_ns += timing.submit_host_ns;
        lane.decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
    }
    pending_transaction_ = ++next_transaction_;
    const std::span<const std::int32_t> row_counts(pending_counts_.data(),
                                                   static_cast<std::size_t>(batch));
    auto pending =
        ContractAccess::make_pending(this, pending_transaction_, sequences,
                                     {pending_tokens_.data(), static_cast<std::size_t>(batch * W)},
                                     timing, row_counts, static_cast<std::uint32_t>(W));
    mark_row_failures(pending, row_counts);
    return pending;
}

CommitResult ProgramImpl::commit_verified(PendingBatch&& pending,
                                          std::span<const runtime::CommitDecision> decisions) {
    const cudaStream_t s = device_.stream;
    const auto& io       = plan_.dflash_io;
    const auto rows      = ContractAccess::rows(pending);
    const auto batch     = static_cast<std::int32_t>(rows.size());
    const std::int32_t W = round_width_;
    CommitResult out;
    out.row_count = rows.size();
    std::array<std::int32_t, kMaximumConcurrency> slots{};
    std::int32_t* commit = dflash_host(io.commit);
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto index = ContractAccess::lane(rows[static_cast<std::size_t>(row)]);
        Lane& lane       = lanes_.at(index);
        const auto& d    = decisions[static_cast<std::size_t>(row)];
        slots[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(index);
        commit[row]                          = 0;
        auto& result                         = out.rows[static_cast<std::size_t>(row)];
        result.speculative_counters          = speculative_counters(lane.speculative);
        if (d.cancelled || d.failed) {
            result.timings     = timings(lane);
            result.speculative = lane.speculative;
            result.disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                          : runtime::CommitDisposition::CancelledReleased;
            continue;
        }
        const auto k = static_cast<std::int32_t>(d.accepted_tokens);
        if (k < 1 || k > pending_counts_[static_cast<std::size_t>(row)]) {
            throw std::logic_error("Glm5Next: commit exceeds the licensed tokens");
        }
        for (std::int32_t j = 0; j < k; ++j) {
            lane.history.push_back(pending_tokens_[static_cast<std::size_t>(row * W + j)]);
        }
        commit[row] = k;
        lane.state_tokens += static_cast<std::uint32_t>(k);
        if (d.terminal) {
            lane.phase         = Phase::Finishable;
            result.disposition = runtime::CommitDisposition::Finishable;
        } else {
            result.disposition = runtime::CommitDisposition::Active;
        }
    }
    // The KDA blocks fold each row's committed columns; the drafter's ring takes their context.
    forward_->commit_verified(
        records_, batch, W,
        std::span<const std::int32_t>(slots.data(), static_cast<std::size_t>(batch)),
        std::span<const std::int32_t>(commit, static_cast<std::size_t>(batch)));
    upload_pinned(dflash_device(io.commit), commit, 4ULL * batch, s);
    auto* io_base         = static_cast<std::byte*>(io_device_.p);
    const Tensor features = feature_columns(batch * W);
    const execution::DFlash2ContextCall call{
        .features  = features.view({features.ne[0], W, batch}),
        .positions = Tensor(io_base + plan_.io.positions, DType::I32, {W, batch}),
        .counts    = Tensor(dflash_device(io.commit), DType::I32, {batch}),
        .slots     = Tensor(io_base + plan_.io.slots, DType::I32, {batch}),
        .min_count = 0,
        .max_count = static_cast<std::uint32_t>(W)};
    replay(graph(context_graphs_, batch, W), [&] { drafter_->materialize(call); });
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto& d = decisions[static_cast<std::size_t>(row)];
        if (d.cancelled || d.failed) {
            release(ContractAccess::lane(rows[static_cast<std::size_t>(row)]));
        }
    }
    ContractAccess::consume(pending);
    pending_transaction_ = 0;
    round_width_         = 1;
    return out;
}

CommitRowResult::SpeculativeCounters
ProgramImpl::speculative_counters(const SpeculativeStats& stats) const noexcept {
    if (!drafter_) { return {}; }
    return {.rounds          = stats.rounds,
            .drafted_tokens  = stats.drafted_tokens,
            .accepted_tokens = stats.accepted_tokens,
            .fallback_steps  = stats.fallback_steps};
}

} // namespace infernix::models::glm5_next::detail
