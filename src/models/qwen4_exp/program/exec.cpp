// Shared execution helpers: io staging, n-gram staging, Forward calls and their CUDA graphs,
// and the wide workspace lease.

#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp::detail {

std::uint32_t ProgramImpl::lane_of(const SequenceHandle& sequence) const {
    const auto lane = ContractAccess::lane(sequence);
    if (ContractAccess::owner(sequence) != this || lane >= options_.max_concurrency ||
        lanes_[lane].phase == Phase::Free || lanes_[lane].epoch != ContractAccess::epoch(sequence)) {
        throw std::logic_error("Qwen4Exp: stale or foreign sequence handle");
    }
    return lane;
}

GenerationTimings ProgramImpl::timings(const Lane& lane) const {
    GenerationTimings out;
    out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
    out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
    out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
    out.vision_seconds       = vision_seconds_of(lane);
    return out;
}

void ProgramImpl::release(std::uint32_t index) noexcept {
    Lane& lane = lanes_[index];
    if (walk_.active && walk_.lane == index) { walk_abandon(); }
    if (stream_lease_.valid() && !prefilling_besides(index)) {
        try { return_stream_lease(); } catch (...) {}
    }
    if (!prefilling_besides(index)) {
        try { return_wide_work(); } catch (...) {}
    }
    prefix_release(lane);
    lane.pages.clear();
    lane.reservation.reset();
    lane.row = KVExecutionRowLease{};
    lane.history.clear();
    lane.history.shrink_to_fit();
    lane.rope = {};
    if (lane.vision) { vision_release(lane); }
    lane.proposer.reset();
    lane.readout.clear();
    lane.constraint.reset();
    lane.prompt.reset();
    lane.constraint_round.clear();
    lane.constraint_trace.clear();
    lane.phase        = Phase::Free;
    lane.state_tokens = 0;
    lane.prefill_end  = 0;
    lane.replay       = false;
    lane.extent       = 0;
    ++revision_;
    kv_busy();
    try { shrink_kv(); } catch (...) {}
}

void ProgramImpl::reset_slot(std::uint32_t slot) {
    const cudaStream_t s = device_.stream;
    for (std::uint32_t l = 0; l < c_.gdn_layers; ++l) {
        Tensor conv      = gdn_->conv_slot(l, static_cast<std::int32_t>(slot));
        Tensor recurrent = gdn_->recurrent_slot(l, static_cast<std::int32_t>(slot));
        CUDA_CHECK(cudaMemsetAsync(conv.data, 0, conv.bytes(), s));
        CUDA_CHECK(cudaMemsetAsync(recurrent.data, 0, recurrent.bytes(), s));
    }
    const std::size_t ple_bytes = static_cast<std::size_t>(width_) * span_ * 2;
    CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(ple_backing_.p) + slot * ple_bytes, 0, ple_bytes, s));
    const auto lanes            = static_cast<std::size_t>(options_.max_concurrency);
    const std::size_t tail_bytes = static_cast<std::size_t>(di_) * (r_ - 1) * 2;
    for (std::uint32_t l = 0; l < c_.attention_layers + (mtp_ ? 1U : 0U); ++l) {
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(tails_backing_.p) + (l * lanes + slot) * tail_bytes, 0,
                                   tail_bytes, s));
        if (window_geometry_.present()) { // the window's tags: no earlier history's row is read
            const std::size_t tags = window_geometry_.lane_bytes(4);
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(window_backing_.p) +
                                           window_geometry_.plane_offset(l, 4) + slot * tags,
                                       0, tags, s));
        }
    }
    CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + slot * static_cast<std::size_t>(token_domain_),
                               0, 4ULL * token_domain_, s));
}

void ProgramImpl::stage_call_rope(const Lane& lane, std::uint32_t first, std::int32_t count, std::int32_t columns,
                                  std::int32_t column, std::int32_t batch, std::int32_t b) const {
    stage_rope(lane.rope, first, count, io_words(io_layout_.rope, 3ULL * columns), columns, column);
    stage_rope_value(block_start_rope(lane.rope, first, static_cast<std::uint32_t>(r_)),
                     io_words(io_layout_.block_rope, 3ULL * batch), batch, b);
}

void ProgramImpl::stage_ngram(Lane& lane, const std::vector<std::int32_t>& history, std::int32_t begin,
                              std::int32_t count, std::size_t column) {
    ngram_window(history, begin, count);
    read_ngram(lane, count, column);
}

void ProgramImpl::ngram_window(const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count) {
    const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
    window_.clear();
    for (std::int32_t p = begin - context; p < begin + count; ++p) {
        window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : history[static_cast<std::size_t>(p)]);
    }
}

std::int32_t ProgramImpl::verify_token(const Lane& lane, std::size_t b, std::int32_t p, std::int32_t q) const {
    if (q < 0) { return static_cast<std::int32_t>(c_.eos_token_id); }
    if (q <= p) { return lane.history[static_cast<std::size_t>(q)]; }
    const auto& d = drafts_[b];
    const auto j  = static_cast<std::size_t>(q - p - 1);
    return j < d.size() ? d[j] : (d.empty() ? lane.history.back() : d.back());
}

void ProgramImpl::queue_verify_ngram(Lane& lane, std::size_t b, std::int32_t p, std::int32_t W, std::int32_t from,
                                     std::int32_t to) {
    if (from >= to) { return; }
    const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
    window_.clear();
    for (std::int32_t q = p + from - context; q < p + to; ++q) { window_.push_back(verify_token(lane, b, p, q)); }
    queue_ngram(lane, to - from, b * static_cast<std::size_t>(W) + static_cast<std::size_t>(from));
}

void ProgramImpl::queue_ngram(Lane& lane, std::int32_t count, std::size_t column) {
    const std::size_t heads = hash_.heads(), at = queued_ids_.size(), rows = static_cast<std::size_t>(count) * heads;
    queued_ids_.resize(at + rows);
    hash_.row_ids(window_, static_cast<std::size_t>(count), queued_ids_.data() + at);
    std::uint64_t misses = 0;
    for (std::size_t r = at; r < at + rows; ++r) { misses += volume_.cached({queued_ids_.data() + r, 1}) ? 0 : 1; }
    queued_.push_back({.lane = &lane, .column = column, .rows = rows, .misses = misses});
}

void ProgramImpl::read_queued_ngram() {
    if (queued_.empty()) { return; }
    const nvtx::ScopedRange rows_range(nvtx::Name::PleRows, nvtx::Category::Control, queued_ids_.size());
    const std::size_t heads = hash_.heads(), row_bytes = ngram_table().row_bytes;
    queued_rows_.resize(queued_ids_.size() * row_bytes);
    const NgramVolume::Counters before = volume_.counters();
    volume_.read_rows(queued_ids_, queued_rows_);
    const NgramVolume::Counters& after = volume_.counters();
    std::uint64_t misses = 0;
    for (const QueuedRows& q : queued_) { misses += q.misses; }
    const std::uint64_t reads = after.reads - before.reads, read_ns = after.read_ns - before.read_ns;
    std::uint64_t reads_left = reads, ns_left = read_ns, misses_left = misses;
    std::size_t at = 0;
    for (const QueuedRows& q : queued_) {
        std::memcpy(host_io() + io_layout_.ngram + q.column * heads * row_bytes, queued_rows_.data() + at * row_bytes,
                    q.rows * row_bytes);
        at += q.rows;
        auto& counters = q.lane->ngram;
        counters.rows += q.rows;
        counters.hits += q.rows - q.misses;
        if (q.misses == 0) { continue; }
        // The last segment with misses takes the remainders, so the lanes' sums equal the call's.
        const bool last       = q.misses == misses_left;
        const std::uint64_t r = last ? reads_left : reads * q.misses / misses;
        const std::uint64_t t = last ? ns_left : read_ns * q.misses / misses;
        counters.reads += r;
        counters.read_ns += t;
        reads_left -= r;
        ns_left -= t;
        misses_left -= q.misses;
    }
    queued_.clear();
    queued_ids_.clear();
}

void ProgramImpl::read_ngram_to(Lane& lane, std::int32_t count, std::byte* out) {
    const nvtx::ScopedRange rows_range(nvtx::Name::PleRows, nvtx::Category::Control, static_cast<std::uint64_t>(count));
    const std::size_t heads = hash_.heads();
    row_ids_.resize(static_cast<std::size_t>(count) * heads);
    hash_.row_ids(window_, static_cast<std::size_t>(count), row_ids_.data());
    const std::size_t row_bytes = ngram_table().row_bytes;
    const NgramVolume::Counters before = volume_.counters();
    volume_.read_rows(row_ids_, std::span<std::byte>(out, static_cast<std::size_t>(count) * heads * row_bytes));
    const NgramVolume::Counters& after = volume_.counters();
    lane.ngram.rows += after.rows - before.rows;
    lane.ngram.hits += after.hits - before.hits;
    lane.ngram.reads += after.reads - before.reads;
    lane.ngram.read_ns += after.read_ns - before.read_ns;
}

void ProgramImpl::stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width,
                                 const std::vector<std::int32_t>& history, bool rows) {
    auto* ids       = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
    auto* positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
    auto* columns   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
    for (std::int32_t i = 0; i < width; ++i) {
        ids[i]       = history[static_cast<std::size_t>(begin + i)];
        positions[i] = begin + i;
    }
    columns[0] = width - 1;
    stage_call_rope(lanes_[lane], static_cast<std::uint32_t>(begin), width, width, 0, 1, 0);
    reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots)[0] = static_cast<std::int32_t>(lane);
    reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows)[0]  = static_cast<std::int32_t>(lane);
    if (rows) { stage_ngram(lanes_[lane], history, begin, width, 0); }
    host_lanes_[0] = static_cast<std::int32_t>(lane);
}

void ProgramImpl::stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
    auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
    auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
    auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
    auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
    auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
    for (std::size_t b = 0; b < lanes.size(); ++b) {
        Lane& lane       = lanes_[lanes[b]];
        ids[b]           = lane.history[static_cast<std::size_t>(positions[b])];
        pos[b]           = positions[b];
        columns[b]       = static_cast<std::int32_t>(b);
        slots[b]         = static_cast<std::int32_t>(lanes[b]);
        rows[b]          = static_cast<std::int32_t>(lanes[b]);
        host_lanes_[b]   = static_cast<std::int32_t>(lanes[b]);
        stage_call_rope(lane, static_cast<std::uint32_t>(positions[b]), 1, static_cast<std::int32_t>(lanes.size()),
                        static_cast<std::int32_t>(b), static_cast<std::int32_t>(lanes.size()), static_cast<std::int32_t>(b));
    }
}

void ProgramImpl::stage_decode_rows(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
    for (std::size_t b = 0; b < lanes.size(); ++b) {
        Lane& lane = lanes_[lanes[b]];
        ngram_window(lane.history, positions[b], 1);
        queue_ngram(lane, 1, b);
    }
    read_queued_ngram();
}

bool ProgramImpl::decode_rows_cached(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
    const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
    for (std::size_t b = 0; b < lanes.size(); ++b) {
        const Lane& lane = lanes_[lanes[b]];
        window_.clear();
        for (std::int32_t p = positions[b] - context; p <= positions[b]; ++p) {
            window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : lane.history[static_cast<std::size_t>(p)]);
        }
        row_ids_.resize(hash_.heads());
        hash_.row_ids(window_, 1, row_ids_.data());
        if (!volume_.cached(row_ids_)) { return false; }
    }
    return true;
}

std::uint32_t ProgramImpl::next_gate_word() {
    if (++gate_sequence_ == 0) { ++gate_sequence_; }
    const std::uint32_t word = gate_sequence_;
    *reinterpret_cast<std::uint32_t*>(host_io() + io_layout_.gate) = word;
    return word;
}

void ProgramImpl::read_behind_gate(Lane& lane, std::uint32_t word, std::size_t bytes,
                                   const std::function<void()>& read) {
    GateRelease release_gate{this, word, bytes};
    device_.flush();
    const NgramVolume::Counters before = volume_.counters();
    read();
    const NgramVolume::Counters& after = volume_.counters();
    release_gate.owner = nullptr;
    publish_pinned_word(gate_ready(), word);
    ++lane.gate.rounds;
    lane.gate.reads += after.reads - before.reads;
    lane.gate.read_ns += after.read_ns - before.read_ns;
    if (after.read_ns - before.read_ns > kSlowGatedReadNs) {
        diagnostic("n-gram rows behind the gate took " + std::to_string((after.read_ns - before.read_ns) / 1000000) +
                       " ms (the GPU waited at layer 1)",
                   DiagnosticLevel::Warning);
    }
}

void ProgramImpl::run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                      const execution::MtpChunk* chunk, const execution::VisionInput* vision, bool streamed,
                      const execution::NgramRowGate* gate) {
    const cudaStream_t s = device_.stream;
    if (gate != nullptr) {
        // With `gate` the n-gram rows are not staged yet: a helper thread reads them (run_rows_behind)
        // while this call runs, so the upload takes everything but them and the rows land behind
        // the gate word (which precedes them in the layout, so it is already in place).
        upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
        upload_pinned(static_cast<std::byte*>(io_device_.p) + io_layout_.mtp_ids,
                      host_io() + io_layout_.mtp_ids, io_layout_.bytes - io_layout_.mtp_ids, s);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(io_device_.p, io_host_.data(), io_layout_.bytes, cudaMemcpyHostToDevice, s));
    }
    if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this call
    settle_round(); // a deferred update precedes the next forward
    residency_->before_round(s);
    // The ring's copies queue on the copy engine behind the io upload above. A call the CPU can
    // share (design §10.6) gates the stream on each layer's routing.
    split_call_ = streamed && split_policy_ && batch * width <= split_columns();
    if (streamed) { expert_stream_->begin(ring_span(), residency_->host_table(), s, split_call_); }
    const WideWork wide(*this, batch * width);
    try {
        forward_call(batch, width, logit_columns, nullptr, chunk, vision, gate, streamed);
        if (dflash2_) { dflash2_call_context(batch, width); }
    } catch (...) {
        // A failed call (an unreadable record, say) still returns the stream's slots: left
        // active, the next streamed chunk could never begin and the release would refuse.
        if (streamed) { expert_stream_->end(); }
        split_call_ = false;
        throw;
    }
    if (streamed) { expert_stream_->end(); }
    split_call_ = false;
    residency_->enqueue_route_download(s, batch * width);
}

void ProgramImpl::run_rows_behind(Lane& lane, std::int32_t begin, std::int32_t width, const execution::MtpChunk* chunk,
                                  const execution::VisionInput* vision) {
    const std::uint32_t word = next_gate_word();
    const execution::NgramRowGate gate = row_gate();
    std::exception_ptr failed;
    std::thread reader([&, word] {
        try {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(width));
            stage_ngram(lane, lane.history, begin, width, 0);
        } catch (...) {
            failed = std::current_exception();
            std::memset(host_io() + io_layout_.ngram, 0, io_prefix(width) - io_layout_.ngram);
        }
        publish_pinned_word(gate_ready(), word);
    });
    try {
        run(1, width, 1, chunk, vision, true, &gate);
    } catch (...) {
        reader.join();
        throw;
    }
    reader.join();
    if (failed) { std::rethrow_exception(failed); }
}

prefix::CallCost ProgramImpl::call_cost() const {
    return prefix::CallCost{.model            = options_.prefix_cost,
                            .span_seconds     = options_.prefix_span_seconds,
                            .walk_min_columns = static_cast<std::uint32_t>(kStreamMinColumns),
                            .walk_max_tokens  = kWalkMaxTokens};
}

bool ProgramImpl::stream_chunk(std::int32_t width, std::size_t walk_bytes) {
    if (!expert_stream_ || width < kStreamMinColumns || residency_->frames() == 0) { return false; }
    if (!stream_lease_.valid()) {
        const std::int32_t* table = residency_->host_table();
        const std::uint32_t E     = c_.moe.experts;
        std::uint32_t most        = 0;
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
            std::uint32_t absent = 0;
            for (std::uint32_t e = 0; e < E; ++e) { absent += table[static_cast<std::size_t>(l) * E + e] < 0 ? 1U : 0U; }
            most = std::max(most, absent);
        }
        // Three leases, each one contiguous run of one pool's frames (the pool with the most room
        // when it is taken): the wide arena whenever the plan has one (a prompt with a streamed
        // call runs wide calls), a walk's area when it fits, then the ring, whose slots hold the
        // largest record (plan_.record_stride) with one more slot's room for the slot tables.
        // The frames a lease takes evict experts of its pool's layers only: each layer's half
        // widens by the most any lease takes per layer of its pool.
        const std::uint64_t slot     = plan_.record_stride;
        const cudaStream_t writers[] = {expert_stream_->stream()};
        const auto per_layer         = [&](const offload::ExpertResidency::FrameLease& lease) {
            const std::uint32_t layers = std::max(1U, residency_->lease_layers(lease));
            return (lease.count + layers - 1) / layers;
        };
        offload::ExpertResidency::FrameLease wide, walk;
        if (wide_work_ > 0) {
            if (wide_work_ + residency_->lend_pitch() > residency_->lendable_bytes()) { return false; }
            wide = residency_->lend(wide_work_, device_.stream, writers);
        }
        if (walk_bytes > 0 && walk_bytes + 2 * (kStreamHalfMin + 1) * slot <= residency_->lendable_bytes()) {
            walk = residency_->lend(walk_bytes, device_.stream, writers);
        }
        const std::uint32_t widened = std::min(E, most + std::max(per_layer(wide), per_layer(walk)));
        // The ring as wide as the room left allows (at most 2 * half + 1 slots).
        const auto room_slots = static_cast<std::uint32_t>(residency_->lendable_bytes() / slot);
        const std::uint32_t half = std::min(widened, room_slots > 1 ? (room_slots - 1) / 2 : 0U);
        if (half < kStreamHalfMin) {
            if (walk.valid()) { residency_->give_back(walk, device_.stream); }
            if (wide.valid()) { residency_->give_back(wide, device_.stream); }
            return false;
        }
        stream_lease_      = residency_->lend((2ULL * half + 1) * slot, device_.stream, writers);
        stream_wide_lease_ = wide;
        stream_walk_lease_ = walk;
        stream_ring_bytes_ = stream_lease_.memory.bytes;
        stream_wide_bytes_ = wide.memory.bytes;
        stream_walk_bytes_ = walk.memory.bytes;
        stream_at_lease_   = expert_stream_->stats();
        split_at_lease_    = forward_->split_stats();
        prefetched_at_lease_ = walk_prefetched_chunks_;
    }
    return true;
}

VramDemand ProgramImpl::expert_frame_demand(std::uint64_t frame_bytes) const {
    VramDemand demand;
    demand.headroom    = options_.vram_headroom;
    demand.elastic     = residency_->elastic() && options_.vram_monitor;
    demand.graphs      = plan_.bytes.graph_bound;
    demand.frame_bytes = frame_bytes;
    demand.max_frames  = residency_->frame_limit();
    return demand;
}

void ProgramImpl::fit_wide_work_to_free(std::uint64_t free_bytes) {
    if (wide_work_ == 0) { return; }
    const std::uint64_t margin = 2ULL << 30; // graph executables, the display and the monitor's headroom
    const std::uint64_t room   = free_bytes > margin ? free_bytes - margin : 0;
    if (wide_work_ <= room) { return; }
    const std::int32_t requested = chunk_;
    std::int32_t chunk           = chunk_ / 2;
    while (chunk > plan_.static_columns &&
           (chunk > plan_.static_columns
                ? workspace_bytes(c_, options_, plan_, plan_.lanes, chunk)
                : 0) > room) {
        chunk /= 2;
    }
    chunk_     = std::max(chunk, plan_.static_columns);
    wide_work_ = chunk_ > plan_.static_columns ? workspace_bytes(c_, options_, plan_, plan_.lanes, chunk_) : 0;
    diagnostic("prefill chunk " + std::to_string(requested) + " needs a workspace beyond what the resident expert "
                   "placement leaves free (" + std::to_string(room >> 20) + " MiB); prefilling at " +
                   std::to_string(chunk_) + " tokens (the output is the same)",
               DiagnosticLevel::Warning);
}

void ProgramImpl::fit_wide_work_to_cache(std::uint64_t lendable, std::uint64_t stride) {
    if (wide_work_ == 0 || lendable == 0 || stride == 0) { return; }
    const auto frames_for        = [&](std::size_t bytes) { return (bytes + stride - 1) / stride; };
    if (frames_for(wide_work_) < lendable) { return; }
    const std::int32_t requested = chunk_;
    const std::size_t needed     = wide_work_;
    std::int32_t chunk           = chunk_ / 2;
    while (chunk > plan_.static_columns &&
           frames_for(workspace_bytes(c_, options_, plan_, plan_.lanes, chunk)) >= lendable) {
        chunk /= 2;
    }
    chunk_     = std::max(chunk, plan_.static_columns);
    wide_work_ = chunk_ > plan_.static_columns ? workspace_bytes(c_, options_, plan_, plan_.lanes, chunk_) : 0;
    diagnostic("prefill chunk " + std::to_string(requested) + " needs " + std::to_string(needed >> 20) +
                   " MiB of workspace, more than the expert cache can lend (" +
                   std::to_string((lendable * stride) >> 20) + " MiB); prefilling at " + std::to_string(chunk_) +
                   " tokens (the output is the same)",
               DiagnosticLevel::Warning);
}

void ProgramImpl::enter_wide() {
    DeviceSpan span{};
    if (stream_lease_.valid() && stream_wide_bytes_ >= wide_work_) {
        span = {stream_wide_lease_.memory.data, wide_work_};
    } else {
        if (!work_lease_.valid() && wide_fallback_.p == nullptr) {
            if (residency_->can_lend(wide_work_)) {
                const cudaStream_t writers[] = {device_.stream};
                work_lease_ = residency_->lend(wide_work_, device_.stream, writers);
            } else {
                // The cache shrank under VRAM pressure: a failed allocation fails this prompt's
                // requests, not the Engine.
                try {
                    wide_fallback_ = DeviceBuffer(wide_work_);
                    char where[64];
                    std::snprintf(where, sizeof(where), " at %p", wide_fallback_.p);
                    diagnostic("the prefill workspace could not be lent from the expert cache; " +
                                   std::to_string(wide_work_ >> 20) + " MiB are allocated for the prompt" + where,
                               DiagnosticLevel::Warning);
                } catch (const std::exception& error) {
                    throw runtime::RecoverableExecutionError(
                        "the prefill workspace (" + std::to_string(wide_work_ >> 20) +
                        " MiB) could neither be lent from the expert cache nor allocated (" + error.what() +
                        "); a smaller --prefill-chunk needs less");
                }
            }
        }
        span = work_lease_.valid() ? DeviceSpan{work_lease_.memory.data, wide_work_}
                                   : DeviceSpan{wide_fallback_.p, wide_work_};
    }
    wide_arena_ = std::make_unique<WorkspaceArena>(span);
    std::swap(*work_, *wide_arena_);
    wide_active_ = true;
}

void ProgramImpl::leave_wide() noexcept {
    std::swap(*work_, *wide_arena_);
    wide_active_ = false;
}

void ProgramImpl::return_wide_work() {
    if (work_lease_.valid()) { residency_->give_back(work_lease_, device_.stream); }
    work_lease_ = {};
    if (wide_fallback_.p != nullptr) {
        device_.synchronize();
        wide_fallback_ = DeviceBuffer{};
    }
}

void ProgramImpl::return_stream_lease() {
    if (!stream_lease_.valid() || walk_.active) { return; }
    const auto frames = stream_lease_.count + stream_wide_lease_.count + stream_walk_lease_.count;
    residency_->give_back(stream_lease_, device_.stream);
    residency_->give_back(stream_wide_lease_, device_.stream);
    residency_->give_back(stream_walk_lease_, device_.stream);
    stream_lease_         = {};
    stream_ring_bytes_    = 0;
    stream_wide_bytes_    = 0;
    stream_walk_bytes_    = 0;
    const auto& now       = expert_stream_->stats();
    const auto streamed   = now.streamed - stream_at_lease_.streamed;
    const auto copies     = now.copies - stream_at_lease_.copies;
    const auto split      = forward_->split_stats();
    char line[400];
    std::snprintf(line, sizeof(line),
                  "prefill expert stream: %llu experts (%.2f GiB, %llu read from the SSD) in %llu copies through "
                  "%u lent frames; CPU split: %llu experts on the CPU, %llu gated copies; n-gram rows of %llu walk "
                  "chunks read during the previous span",
                  static_cast<unsigned long long>(streamed),
                  static_cast<double>(now.bytes - stream_at_lease_.bytes) / (1ULL << 30),
                  static_cast<unsigned long long>(now.ssd - stream_at_lease_.ssd),
                  static_cast<unsigned long long>(copies), frames,
                  static_cast<unsigned long long>(split.cpu_experts - split_at_lease_.cpu_experts),
                  static_cast<unsigned long long>(split.streamed_experts - split_at_lease_.streamed_experts),
                  static_cast<unsigned long long>(walk_prefetched_chunks_ - prefetched_at_lease_));
    diagnostic(line, DiagnosticLevel::Debug);
}

bool ProgramImpl::prefilling_besides(std::uint32_t index) const noexcept {
    for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
        if (i != index && lanes_[i].phase == Phase::Prefill) { return true; }
    }
    return false;
}

void ProgramImpl::run_decode(std::int32_t batch, std::int32_t position) {
    const cudaStream_t s = device_.stream;
    if (one_column_divisor() != 0) { *reinterpret_cast<std::int32_t*>(host_io() + io_layout_.divisor) = plain_divisor(position); }
    upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
    if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this round
    settle_round(); // a deferred update precedes the next forward
    residency_->before_round(s, /*landing=*/true, /*walking=*/walk_.active);
    const execution::NgramRowGate gate = row_gate();
    const nvtx::ScopedRange target(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                                   static_cast<std::uint64_t>(batch));
    replay(graphs_[static_cast<std::size_t>(batch - 1)],
           [&] { forward_call(batch, 1, batch, nullptr, nullptr, nullptr, &gate, false, /*decode_round=*/true); });
    // The round that called this reads and releases its gate and downloads the route itself:
    // until the word is published the GPU waits on the engine thread, so nothing that can wait
    // for the device may be queued from here.
}

void ProgramImpl::follow_expert_route(DecodeGraph& graph) {
    if (graph.executable.ready() && graph.resident != expert_route()) {
        graph.executable.reset();
        graph.definition.reset();
    }
}

void ProgramImpl::replay(DecodeGraph& graph, const std::function<void()>& body) {
    const cudaStream_t s = device_.stream;
    follow_expert_route(graph);
    if (graph.executable.ready()) {
        ++graph_replays_;
        graph.executable.launch(s);
    } else if (!graph.warmed) {
        ++graph_eager_executions_;
        body();
        graph.warmed = true;
    } else {
        ++graph_captures_;
        // A captured body bakes in whatever it reads by address: the arena *work_ points at and
        // the per-layer waits forward_call consumes. Neither may be a call's temporary.
        if (wide_active_ || !next_waits_[0].empty() || !next_waits_[1].empty()) {
            throw std::logic_error("Qwen4Exp: a graph capture inside a wide call or with pending layer waits");
        }
        const std::size_t free_before = trace_ ? trace_free() : 0;
        graph.resident = expert_route();
        graph.definition.capture(s, body);
        graph.executable.instantiate(graph.definition);
        graph.executable.launch(s);
        if (trace_) { trace_graph(graph, free_before); }
    }
}

void ProgramImpl::replay_gated(DecodeGraph& graph, std::int32_t steps, const std::function<void(std::int32_t)>& body,
                               const std::function<void(std::int32_t, cudaGraphConditionalHandle)>& gate) {
    const cudaStream_t s = device_.stream;
    follow_expert_route(graph);
    if (graph.executable.ready()) {
        ++graph_replays_;
        graph.executable.launch(s);
    } else if (!graph.warmed) {
        ++graph_eager_executions_;
        for (std::int32_t j = 0; j < steps; ++j) { body(j); }
        graph.warmed = true;
    } else {
        ++graph_captures_;
        if (wide_active_ || !next_waits_[0].empty() || !next_waits_[1].empty()) {
            throw std::logic_error("Qwen4Exp: a graph capture inside a wide call or with pending layer waits");
        }
        const std::size_t free_before = trace_ ? trace_free() : 0;
        graph.resident = expert_route();
        graph.definition.capture_gated_chain(s, steps, body, gate);
        graph.executable.instantiate(graph.definition);
        graph.executable.launch(s);
        if (trace_) { trace_graph(graph, free_before); }
    }
}

std::size_t ProgramImpl::trace_free() {
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

void ProgramImpl::trace_graph(const DecodeGraph& graph, std::size_t free_before) {
    offload::RouteTraceGraph family = offload::RouteTraceGraph::Decode;
    std::ptrdiff_t at      = offload::route_trace_index(graphs_, &graph);
    std::ptrdiff_t stride  = 1;
    if (at < 0 && (at = offload::route_trace_index(verify_graphs_, &graph)) >= 0) {
        family = offload::RouteTraceGraph::Verify;
        stride = max_width_;
    } else if (at < 0 && ((at = offload::route_trace_index(mtp_draft_graphs_, &graph)) >= 0 ||
                          (at = offload::route_trace_index(mtp_sampled_graphs_, &graph)) >= 0 ||
                          (at = offload::route_trace_index(mtp_gated_graphs_, &graph)) >= 0)) {
        family = offload::RouteTraceGraph::MtpDraft;
        stride = mtp_k_;
    } else if (at < 0 && (at = offload::route_trace_index(mtp_catch_graphs_, &graph)) >= 0) {
        family = offload::RouteTraceGraph::MtpCatchUp;
        stride = max_width_;
    } else if (at < 0 && (at = offload::route_trace_index(dflash_propose_graphs_, &graph)) >= 0) {
        family = offload::RouteTraceGraph::DFlashPropose;
    } else if (at < 0 && (at = offload::route_trace_index(dflash_context_graphs_, &graph)) >= 0) {
        family = offload::RouteTraceGraph::DFlashCatchUp;
        stride = max_width_;
    }
    if (at < 0) { return; }
    trace_->graph(family, static_cast<std::uint32_t>(at / stride + 1), static_cast<std::uint32_t>(at % stride + 1),
                  free_before, trace_free());
}

void ProgramImpl::require_tier_round() const {
    if (fetch_channel_ && tier_ && !tier_->round_open()) {
        throw std::logic_error("Qwen4Exp: a call reading SSD-only experts was enqueued outside the tier's round");
    }
}

void ProgramImpl::forward_call(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                               const execution::ForwardVerify* verify, const execution::MtpChunk* chunk,
                               const execution::VisionInput* vision, const execution::NgramRowGate* gate,
                               bool streamed, bool decode_round) {
    require_tier_round();
    set_expert_route();
    execution::ForwardBatch fb = io_batch(static_cast<std::byte*>(io_device_.p), batch, width, logit_columns);
    fb.decode_round  = decode_round;
    fb.ngram_gate    = gate;
    fb.verify        = verify;
    fb.mtp_chunk     = chunk;
    fb.layer_waits   = next_waits_;
    next_waits_      = {};
    fb.vision        = vision;
    fb.stream        = streamed;
    fb.split         = streamed && split_call_;
    // Decode and verification rounds export their final residuals for the MTP catch-up.
    const std::int32_t cols = batch * width;
    if (mtp_ && chunk == nullptr) { fb.residual_out = Tensor(mtp_residuals_.p, DType::BF16, {width_, cols}); }
    // DFlash2's target features: every column of a decode or verification round, a prefill call's
    // or forced tokens' last window (the ring keeps no earlier position).
    if (dflash2_) {
        const auto [column, n] = decode_round ? std::pair<std::int32_t, std::int32_t>{0, cols} : dflash_call_region(cols);
        fb.draft_features      = dflash_feature_columns(column, n);
    }
    // A scored prompt's calls keep their final residuals for the LM head (causal_score).
    if (scoring_) { fb.residual_out = score_residuals(score_column_, cols); }
    // The zero-token resume's saved head residual: a non-MTP prompt call's last column. The
    // prompt-end snapshot carries it; decode, verification and MTP calls leave it alone (an MTP
    // Program has no buffer — its own saved column cannot serve the resume, S3 of the design).
    if (head_saved_.p != nullptr && chunk == nullptr && verify == nullptr && !decode_round && !scoring_) {
        fb.head_residual_out = head_saved_column(static_cast<std::uint32_t>(host_lanes_[0]));
    }
    Tensor logits(logits32_.p, DType::FP32, {vocab_, logit_columns});
    forward_->run(fb, logits);
    round_logits(logit_columns, device_.stream);
}

Tensor ProgramImpl::read_logits(std::int32_t columns) {
    if (options_.bf16_logits) { return Tensor(logits16_.p, DType::BF16, {vocab_, columns}); }
    return Tensor(logits32_.p, DType::FP32, {vocab_, columns});
}

void ProgramImpl::round_logits(std::int32_t columns, cudaStream_t s) {
    if (!options_.bf16_logits || columns == 0) { return; }
    const Tensor wide(logits32_.p, DType::FP32, {vocab_, columns});
    Tensor narrow(logits16_.p, DType::BF16, {vocab_, columns});
    ops::cast_fp32_to_bf16(wide, narrow, s);
}

execution::ForwardBatch ProgramImpl::io_batch(std::byte* base, std::int32_t batch, std::int32_t width,
                                              std::int32_t logit_columns) const {
    const std::int32_t cols = batch * width;
    execution::ForwardBatch fb;
    fb.ids           = Tensor(base + io_layout_.ids, DType::I32, {cols});
    fb.positions     = Tensor(base + io_layout_.positions, DType::I32, {cols});
    fb.rope_positions   = Tensor(base + io_layout_.rope, DType::I32, {cols, 3});
    fb.block_start_rope = Tensor(base + io_layout_.block_rope, DType::I32, {batch, 3});
    fb.slots         = Tensor(base + io_layout_.slots, DType::I32, {batch});
    fb.table_rows    = Tensor(base + io_layout_.rows, DType::I32, {batch});
    fb.logit_columns = Tensor(base + io_layout_.columns, DType::I32, {logit_columns});
    fb.ngram_rows    = Tensor(base + io_layout_.ngram, DType::U8,
                              {dim(ngram_table().row_bytes), dim(hash_.heads()), cols});
    fb.ngram_format  = ngram_table().format;
    if (ngram_scale_.p != nullptr) { fb.ngram_scale = Tensor(ngram_scale_.p, DType::BF16, {1}); }
    fb.host_slots      = std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(batch));
    fb.host_table_rows = fb.host_slots;
    fb.batch           = batch;
    fb.width           = width;
    return fb;
}

Program::GraphExecutionStats ProgramImpl::graph_execution_stats() const noexcept {
    return {.replays = graph_replays_, .eager = graph_eager_executions_,
            .captures = graph_captures_};
}

execution::NgramRowGate ProgramImpl::row_gate() const {
    auto* io_device = static_cast<std::byte*>(io_device_.p);
    return {.pinned_rows = host_io() + io_layout_.ngram,
            .ready       = gate_ready(),
            .expected    = reinterpret_cast<const std::uint32_t*>(io_device + io_layout_.gate),
            .wait_stats  = static_cast<std::uint64_t*>(gate_stats_.p),
            .wait_row    = reinterpret_cast<const std::int32_t*>(io_device + io_layout_.slots)};
}

} // namespace infernix::models::qwen4_exp::detail
