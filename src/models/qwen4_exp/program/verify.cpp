// Speculative verification: the verify windows and the commit of their accepted columns.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

void ProgramImpl::allocate_verification(std::int32_t lanes, std::uint64_t& allocated) {
    allocate(records_backing_, plan_.records_bytes, allocated);
    records_ = GdnReplayRecords(DeviceSpan{records_backing_.p, records_backing_.bytes}, plan_.records);
    allocate(ple_records_, plan_.ple_records, allocated);
    allocate(qsa_records_, plan_.qsa_records, allocated);
    folds_.resize(static_cast<std::size_t>(max_width_) + 1);
    verify_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
    const std::size_t cells = static_cast<std::size_t>(lanes) * max_width_;
    spec_layout_ = plan_.spec;
    allocate(spec_device_, 4ULL * spec_layout_.words, allocated);
    spec_host_ = PinnedHostBuffer(4ULL * spec_layout_.words);
    pending_tokens_.assign(cells, 0);
    live_.assign(cells, 0);
    proposer_tokens_ = std::bit_ceil(std::max<std::size_t>(64, options_.max_context + 64ULL));
}

execution::ForwardVerify ProgramImpl::verify_view(std::int32_t batch, std::int32_t width) const {
    execution::ForwardVerify v;
    v.gdn        = records_.narrowed(width);
    v.ple_inputs = Tensor(ple_records_.p, DType::BF16, {width_, width, batch});
    v.qsa_keys   = Tensor(qsa_records_.p, DType::BF16, {di_, width, batch, dim(c_.attention_layers)});
    return v;
}

const ops::GdnReplayFoldPlan& ProgramImpl::fold(std::int32_t width) {
    auto& plan = folds_.at(static_cast<std::size_t>(width));
    if (!plan) { plan.emplace(records_.narrowed(width), gdn_->all_layers_view()); }
    return *plan;
}

PendingBatch ProgramImpl::verify(std::span<const SequenceHandle> sequences, std::span<const std::uint32_t> lanes,
                                 std::span<const std::int32_t> positions, std::int32_t width,
                                 std::span<const std::int32_t> staged, Clock::time_point start) {
    const cudaStream_t s = device_.stream;
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    const std::int32_t W = width, K = width - 1;
    auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
    auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
    auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
    auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
    auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane = lanes_[lanes[b]];
        const auto n = static_cast<std::int32_t>(drafts_[b].size());
        const std::int32_t p = positions[b];
        for (std::int32_t j = 0; j < W; ++j) {
            ids[b * W + j]     = verify_token(lane, static_cast<std::size_t>(b), p, p + j);
            pos[b * W + j]     = p + j;
            columns[b * W + j] = b * W + j;
        }
        for (std::int32_t j = 0; j < K; ++j) { spec_host(spec_layout_.drafts)[b * K + j] = ids[b * W + j + 1]; }
        slots[b]       = static_cast<std::int32_t>(lanes[b]);
        rows[b]        = static_cast<std::int32_t>(lanes[b]);
        host_lanes_[b] = static_cast<std::int32_t>(lanes[b]);
        stage_call_rope(lane, static_cast<std::uint32_t>(p), W, batch * W, b * W, batch, b);
        spec_host(spec_layout_.extents)[b] = n;
        spec_host(spec_layout_.lengths)[b] = p;
        spec_host(spec_layout_.anchors)[b] = lane.history.back();
    }
    // The round is rectangular at its widest row, so a row with fewer drafts than W - 1
    // carries columns past its drafts: they route through every layer's MoE today, and a
    // column mask could keep them off that path. Counted here, host-side, per width.
    std::int64_t padded = 0;
    for (std::int32_t b = 0; b < batch; ++b) {
        padded += std::max<std::int64_t>(0, W - 1 - static_cast<std::int32_t>(drafts_[b].size()));
    }
    ++verify_rounds_;
    verify_columns_ += static_cast<std::uint64_t>(batch) * static_cast<std::uint64_t>(W);
    verify_padded_ += static_cast<std::uint64_t>(padded);
    verify_by_width_.resize(std::max(verify_by_width_.size(), static_cast<std::size_t>(W) + 1));
    auto& by_width = verify_by_width_[static_cast<std::size_t>(W)];
    ++by_width[0];
    by_width[1] += static_cast<std::uint64_t>(batch) * static_cast<std::uint64_t>(W);
    by_width[2] += static_cast<std::uint64_t>(padded);
    const auto read_rows = [&] {
        for (std::int32_t b = 0; b < batch; ++b) {
            queue_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], W, staged[b], W);
        }
        read_queued_ngram();
    };
    const std::uint32_t word = next_gate_word();
    DecodeGraph& graph = verify_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)];
    // A mixed round reads every row first: its eager Forward uploads them with the io.
    const bool gated   = !mixed_.active && graph.executable.ready();
    if (!gated) {
        read_rows();
        publish_pinned_word(gate_ready(), word);
    }
    if (mixed_.active) { stage_mixed_prefill(batch, batch * W); }
    upload_pinned(io_device_.p, io_host_.data(), mixed_.active ? io_prefix(batch * W + mixed_.width) : io_layout_.ngram, s);
    upload_pinned(spec_device_.p, spec_host_.data(), 4ULL * spec_layout_.licensed, s);
    if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this round
    settle_round(); // a deferred update precedes the next forward
    residency_->before_round(s, /*landing=*/true, /*walking=*/walk_.active);
    if (mixed_.active) {
        const execution::ForwardVerify view = verify_view(batch, W);
        run_mixed(batch, W, &view);
    } else {
        const execution::NgramRowGate gate = row_gate();
        {
            const nvtx::ScopedRange target(nvtx::Name::DecodeDFlashTarget, nvtx::Category::Decode,
                                           static_cast<std::uint64_t>(batch * W));
            replay(graph, [&] {
                const execution::ForwardVerify view = verify_view(batch, W);
                forward_call(batch, W, batch * W, &view, nullptr, nullptr, &gate, false, /*decode_round=*/true);
            });
        }
        // Nothing that can wait for the device sits between the gated launch and its publish
        // (the gate would deadlock): the rows are read and released before anything else queues.
        if (gated) {
            read_behind_gate(lanes_[lanes[0]], word, io_prefix(batch * W) - io_layout_.ngram, read_rows);
        }
        residency_->enqueue_route_download(s, batch * W);
    }

    // Acceptance on the device, from the logits plain decode samples (FP32, or BF16: design
    // §17.7); the constraints mask them in place.
    Tensor narrow = read_logits(batch * W);
    Tensor target(spec_device(spec_layout_.target), DType::I32, {batch * W});
    ops::argmax(narrow, target, token_domain_, s);
    const bool constrained = stage_constraints(lanes, W);
    if (constrained) { constrain(narrow, &target); }
    auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
    for (std::int32_t b = 0; b < batch; ++b) {
        configs[b]      = lanes_[lanes[b]].sampling;
        configs[b].mask = grammar_mask(static_cast<std::size_t>(b), drafts_[b]);
        reasoning_ban(static_cast<std::size_t>(b), configs[b]);
    }
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
    Tensor drafts(spec_device(spec_layout_.drafts), DType::I32, {K, batch});
    Tensor extents(spec_device(spec_layout_.extents), DType::I32, {batch});
    Tensor lengths(spec_device(spec_layout_.lengths), DType::I32, {batch});
    Tensor anchors(spec_device(spec_layout_.anchors), DType::I32, {batch});
    Tensor licensed(spec_device(spec_layout_.licensed), DType::I32, {W, batch});
    Tensor counts(spec_device(spec_layout_.counts), DType::I32, {batch});
    Tensor accepted(spec_device(spec_layout_.accepted), DType::I32, {batch});
    // A sampled row's DFlash2 drafts were drawn from the selector's laws: such a round verifies
    // by sparse rejection sampling with them (greedy rows' laws are one-hot, n-gram rows' copies
    // one-hot too). A round of greedy rows drafted deterministically and takes the greedy Op.
    bool dflash_sampled = false;
    const bool drafted  = dflash_round_; // the round's commit may leave its cache update to the next proposal
    if (std::exchange(dflash_round_, false)) {
        for (std::int32_t b = 0; b < batch; ++b) {
            dflash_sampled = dflash_sampled || lanes_[lanes[b]].sampling.temperature > 0.0F;
        }
    }
    const bool sampled = std::exchange(mtp_sampled_round_, false);
    const bool sparse  = sampled || dflash_sampled;
    if (dflash_sampled) {
        constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
        const auto [at_candidates, at_q] = dflash2_verify_planes(batch, K);
        Tensor candidates(dflash_device(at_candidates), DType::I32, {C, K, batch});
        Tensor q(dflash_device(at_q), DType::FP32, {C, K, batch});
        auto scope = work_->scope();
        ops::speculative_accept_sparse_drafts(target.view({W, batch}), narrow.view({vocab_, W, batch}), drafts,
                                              candidates, q, extents, lengths, anchors, licensed, counts, accepted,
                                              token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                              ops::SpeculativeAcceptExecutionEnvelope{}, *work_, s);
    } else if (sampled) {
        // Sampled drafts: each row's 16 candidates and q per verified position, [16, K, batch]: an
        // MTP row's from its draft steps, an n-gram row's one-hot at its copied token.
        constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
        std::int32_t* host_candidates = mtp_host(mtp_io_.verify_candidates);
        auto* host_q                  = reinterpret_cast<float*>(mtp_host(mtp_io_.verify_q));
        for (std::int32_t b = 0; b < batch; ++b) {
            const auto n = static_cast<std::int32_t>(drafts_[b].size());
            if (n == 0) { continue; }
            const std::size_t row = static_cast<std::size_t>(b) * K * C;
            if (from_ngram_[b]) {
                for (std::int32_t j = 0; j < n; ++j) {
                    const std::int32_t token = drafts_[b][static_cast<std::size_t>(j)];
                    for (std::int32_t c = 0; c < C; ++c) {
                        host_candidates[row + j * C + c] = (token + c) % token_domain_;
                        host_q[row + j * C + c]          = c == 0 ? 1.0F : 0.0F;
                    }
                }
                upload_pinned(mtp_device(mtp_io_.verify_candidates + row), host_candidates + row, 4ULL * C * n, s);
                upload_pinned(mtp_device(mtp_io_.verify_q + row), host_q + row, 4ULL * C * n, s);
                continue;
            }
            // Draft step j's plane of row b is at (j * batch + b) * C.
            for (const auto [from, to] : {std::pair{mtp_io_.draft_candidates, mtp_io_.verify_candidates},
                                          std::pair{mtp_io_.draft_q, mtp_io_.verify_q}}) {
                CUDA_CHECK(cudaMemcpy2DAsync(mtp_device(to + row), 4ULL * C,
                                             mtp_device(from + static_cast<std::size_t>(C) * b), 4ULL * C * batch,
                                             4ULL * C, static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice, s));
            }
        }
        Tensor candidates(mtp_device(mtp_io_.verify_candidates), DType::I32, {C, K, batch});
        Tensor q(mtp_device(mtp_io_.verify_q), DType::FP32, {C, K, batch});
        auto scope = work_->scope();
        ops::speculative_accept_sparse_drafts(target.view({W, batch}), narrow.view({vocab_, W, batch}), drafts,
                                              candidates, q, extents, lengths, anchors, licensed, counts, accepted,
                                              token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                              ops::SpeculativeAcceptExecutionEnvelope{}, *work_, s);
    } else {
        auto scope = work_->scope();
        ops::speculative_accept_greedy_drafts(target.view({W, batch}), narrow.view({vocab_, W, batch}), drafts,
                                              extents, lengths, anchors, licensed, counts, accepted,
                                              token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                              *work_, s);
    }
    if (sparse) {
        // The sparse acceptance leaves token counts to the caller (the greedy-draft Op counts each licensed token
        // itself): each row's licensed tokens count before the next round's penalties.
        ops::increment_licensed_token_counts(licensed, counts,
                                             static_cast<const ops::SamplingConfig*>(configs_.p), s);
    }
    CUDA_CHECK(cudaMemcpyAsync(spec_host(spec_layout_.licensed), licensed.data,
                               4ULL * (spec_layout_.commit - spec_layout_.licensed), cudaMemcpyDeviceToHost, s));
    {
        const nvtx::ScopedRange wait(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control);
        device_.synchronize();
    }
    check_device_error();

    const std::int32_t* licensed_host = spec_host(spec_layout_.licensed);
    for (std::int32_t b = 0; b < batch; ++b) {
        Lane& lane     = lanes_[lanes[b]];
        const auto n   = static_cast<std::uint64_t>(drafts_[b].size());
        const auto L   = spec_host(spec_layout_.counts)[b];
        const auto A   = spec_host(spec_layout_.accepted)[b];
        if (L < 1 || L > W || A != L - 1) { throw std::logic_error("Qwen4Exp: acceptance returned an invalid extent"); }
        for (std::int32_t j = 0; j < W; ++j) {
            pending_tokens_[static_cast<std::size_t>(b * W + j)] = j < L ? licensed_host[b * W + j] : 0;
            live_[static_cast<std::size_t>(b * W + j)]           = j <= A ? 1 : 0;
        }
        pending_counts_[b] = L;
        if (constrained) {
            take_constraint_draws(lane, static_cast<std::size_t>(b * W),
                                  std::span<const std::int32_t>(&pending_tokens_[static_cast<std::size_t>(b * W)],
                                                                static_cast<std::size_t>(L)));
        }
        auto& stats        = lane.speculative;
        if (n == 0) {
            ++stats.fallback_steps;
            continue;
        }
        ++stats.rounds;
        stats.drafted_tokens += n;
        stats.accepted_tokens += static_cast<std::uint64_t>(A);
        if (!from_ngram_[b]) {
            // Positions 1..A were accepted; position A + 1 (when drafted) was the first rejected.
            for (std::uint64_t j = 0; j < n && j <= static_cast<std::uint64_t>(A); ++j) {
                double& a = lane.mtp_accept[static_cast<std::size_t>(j)];
                a = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * (j < static_cast<std::uint64_t>(A) ? 1.0 : 0.0);
            }
            if (dflash2_) {
                lane.dflash_calibration.observe(std::span<const float>(dflash_confidence_[b].data(), n),
                                                static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(A),
                                                kAcceptanceWeight);
            }
        }
        if (from_ngram_[b]) {
            ++stats.ngram_rounds;
            stats.ngram_drafted_tokens += n;
            stats.ngram_accepted_tokens += static_cast<std::uint64_t>(A);
            // The bucket's acceptance per position.
            if (ngram_bucket_[b] >= 0) {
                double& a = lane.ngram_accept[static_cast<std::size_t>(ngram_bucket_[b])];
                for (std::uint64_t j = 0; j < n && j <= static_cast<std::uint64_t>(A); ++j) {
                    a = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * (j < static_cast<std::uint64_t>(A) ? 1.0 : 0.0);
                }
            }
        }
        for (std::int32_t j = 0; j < A; ++j) { ++stats.accepted_per_position[static_cast<std::size_t>(j)]; }
    }
    std::uint32_t advanced = 1;
    for (std::int32_t b = 0; b < batch; ++b) {
        advanced = std::max<std::uint32_t>(advanced, static_cast<std::uint32_t>(pending_counts_[b]));
    }
    defer_round(batch * W, advanced, true, W, lanes, drafted && !mtp_);
    round_width_ = W;
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    for (std::int32_t b = 0; b < batch; ++b) {
        lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
        lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
    }
    pending_transaction_ = ++next_transaction_;
    auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                {pending_tokens_.data(), static_cast<std::size_t>(batch * W)},
                                                timing, {pending_counts_.data(), static_cast<std::size_t>(batch)},
                                                static_cast<std::uint32_t>(W));
    mark_row_failures(pending, {pending_counts_.data(), static_cast<std::size_t>(batch)});
    return pending;
}

CommitResult ProgramImpl::commit_verified(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
    const cudaStream_t s = device_.stream;
    const auto rows      = ContractAccess::rows(pending);
    const auto batch     = static_cast<std::int32_t>(rows.size());
    const std::int32_t W = round_width_;
    CommitResult out;
    out.row_count = rows.size();
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::int32_t* commit = spec_host(spec_layout_.commit);
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto index = ContractAccess::lane(rows[row]);
        Lane& lane       = lanes_.at(index);
        const auto& d    = decisions[static_cast<std::size_t>(row)];
        fold_rows[row]   = {static_cast<std::int32_t>(index), static_cast<std::int32_t>(index), 0};
        commit[row]      = 0;
        out.rows[row].speculative_counters = speculative_counters(lane.speculative);
        if (d.cancelled || d.failed) {
            lane.constraint_round.clear();
            out.rows[row].timings     = timings(lane);
            out.rows[row].speculative = lane.speculative;
            out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                 : runtime::CommitDisposition::CancelledReleased;
            continue;
        }
        const auto k = static_cast<std::int32_t>(d.accepted_tokens);
        if (k < 1 || k > pending_counts_[row]) { throw std::logic_error("Qwen4Exp: commit exceeds the licensed tokens"); }
        commit_constraint_draws(lane, static_cast<std::size_t>(k));
        for (std::int32_t j = 0; j < k; ++j) {
            const std::int32_t token = pending_tokens_[static_cast<std::size_t>(row * W + j)];
            lane.history.push_back(token);
            if (lane.proposer) { lane.proposer->append(token); }
        }
        fold_rows[row].commit_columns = k;
        commit[row]                   = k;
        lane.state_tokens += static_cast<std::uint32_t>(k);
        if (d.terminal) {
            lane.phase                = Phase::Finishable;
            out.rows[row].disposition = runtime::CommitDisposition::Finishable;
        } else {
            out.rows[row].disposition = runtime::CommitDisposition::Active;
        }
    }
    upload_pinned(spec_device(spec_layout_.commit), commit, 4ULL * batch, s);
    fold(W).execute(std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), static_cast<std::size_t>(batch)), s);
    auto* io = static_cast<std::byte*>(io_device_.p);
    const Tensor commit_columns(spec_device(spec_layout_.commit), DType::I32, {batch});
    const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
    const Tensor positions(io + io_layout_.positions, DType::I32, {W, batch});
    const auto layers = dim(c_.attention_layers);
    Tensor tails(tails_backing_.p, DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), layers});
    ops::qsa_commit_tails(Tensor(qsa_records_.p, DType::BF16, {di_, W, batch, layers}), positions, commit_columns,
                          tails, slots, execution::qsa_geometry(c_), s);
    Tensor ple_states(ple_backing_.p, DType::BF16, {width_, span_, dim(options_.max_concurrency)});
    ops::ple_conv_commit(Tensor(ple_records_.p, DType::BF16, {width_, W, batch}), commit_columns, ple_states,
                         slots, s);
    if (mtp_) { mtp_catch_up(rows, W, commit); }
    if (dflash2_) { dflash2_catch_up(rows, W, commit); }
    for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto& d = decisions[static_cast<std::size_t>(row)];
        if (!d.cancelled && !d.failed) {
            Lane& lane = lanes_[ContractAccess::lane(rows[row])];
            prefix_publish(lane, prefix_frontier(lane, false));
        }
    }
    // A drafted DFlash2 round's next is most likely one too, whose proposal settles this round's cache update
    // while it drafts (DeferredRound); a cancelled or failed row's release settles it first.
    bool releases = false;
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto& d = decisions[static_cast<std::size_t>(row)];
        releases      = releases || d.cancelled || d.failed;
    }
    if (!deferred_.to_proposal || releases) { settle_round(); }
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto& d = decisions[static_cast<std::size_t>(row)];
        if (d.cancelled || d.failed) { release(ContractAccess::lane(rows[row])); }
    }
    ContractAccess::consume(pending);
    pending_transaction_ = 0;
    round_width_         = 1;
    return out;
}

} // namespace infernix::models::qwen4_exp::detail
