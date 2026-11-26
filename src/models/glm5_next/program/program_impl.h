#pragma once

// Glm5Next Program internals: device layouts, lanes, request plans and the ProgramImpl that the
// public Program forwards to. Topic files: construct.cpp (device plan and allocations),
// binding.cpp (admission), execution.cpp (prefill, decode, forced tokens, commit, release),
// dflash2.cpp (the DFlash2 drafter's proposals, verification and commit), constraint.cpp
// (/v1/decide: constraints, grammar masks, prompt readouts), score.cpp (causal scoring),
// experts.cpp (the routed experts' VRAM frames and RAM/SSD tier).

#include "models/glm5_next/program/program.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "core/vram_budget.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/token_constraint.h"
#include "models/glm5_next/execution/dflash2.h"
#include "models/glm5_next/execution/forward.h"
#include "models/glm5_next/execution/parameters.h"
#include "models/glm5_next/program/draft_length.h"
#include "models/glm5_next/program/prefix/call_plan.h"
#include "models/glm5_next/program/prefix/prefix_cache.h"
#include "models/offload/expert_residency.h"
#include "models/offload/host_expert_tier.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "runtime/diagnostics.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::glm5_next::detail {

using Clock = std::chrono::steady_clock;

[[nodiscard]] inline std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
[[nodiscard]] inline std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }
[[nodiscard]] inline std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// A TokenConstraint compiled for ops::constrain_logits: its distinct multi-token sets (written into
// the lane's set columns when it is first constrained), each output step's descriptor relative to
// the lane (a set index, or the forced-token encoding -2 - token), and each step's observed tokens
// (empty when the request observes nothing).
struct ConstraintPlan {
    std::vector<std::vector<TokenId>> sets;
    std::vector<std::int32_t> steps;
    std::vector<std::vector<TokenId>> observe;
    std::uint64_t serial = 0; // unique per planned request: a lane's uploaded sets name their plan
};

struct BasePlanImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::vector<TokenId> readout_tokens;              // read at the prompt's frontier (/v1/decide)
    std::shared_ptr<const ConstraintPlan> constraint; // null: unconstrained
    bool reuse              = false; // the prefix cache may resume and publish this request
    std::uint32_t pages     = 0; // KV page groups of the extent
    std::uint32_t positions = 0; // the extent: prompt + effective output, at most max_context
    std::shared_ptr<const PreparedPrompt> prompt; // the request's prompt, moved in by plan_request
};

// One admission's prefix-cache source: the snapshot it resumes from, the blocks it maps, the
// copy-on-write tail it takes, and the suffix's prefill calls.
struct PrefixSelection {
    std::optional<prefix::cache_index::SnapshotRef> snapshot;
    std::uint32_t frontier = 0;
    // The full blocks below the frontier (the anchor included).
    std::vector<prefix::cache_index::NodeRef> path;
    std::uint32_t shared = 0;     // leading blocks mapped from the cache
    bool cow_tail        = false; // a non-block-aligned frontier: the snapshot's tail is copied
    std::uint32_t need   = 0;     // pages the admission newly takes (private + restored)
    bool image_restore   = false;
    std::vector<std::uint32_t> existing; // frontiers of every matched snapshot (tap spacing)
    std::uint32_t cached_tokens = 0;     // the prompt prefix held as cached blocks (diagnostics)
    prefix::CallPlan plan;               // the suffix's prefill calls and taps from `frontier`
};

struct QuoteImpl {
    std::uint32_t pages = 0;
    // The prefix cache's selection (absent: a root start).
    std::optional<PrefixSelection> prefix;
};

// One lane's prefix-cache state.
struct LanePrefix {
    bool reuse = false;
    std::vector<prefix::cache_index::NodeRef>
        path;                    // pinned once each: mapped, then published blocks
    std::uint32_t page_base = 0; // logical block of lane.pages[0]
    std::uint32_t deepest   = 0; // frontier of the lineage's deepest snapshot
    std::uint32_t reused    = 0;
    // The zero-token resume: the lane's state is the prompt's own end snapshot, whose head
    // residual is saved (the first token regenerates from it without prefill).
    bool head_resume = false;
    std::optional<prefix::cache_index::SnapshotRef> resume;
    std::optional<prefix::RestoreTicket> restore; // the first call waits on it
    std::uint64_t capture = 0;                    // its copy-out gates the next call
    // The current request has captured (its first capture superseded the resume snapshot).
    bool captured = false;
    std::uint64_t resident_capture =
        0; // the capture whose image the lane's staging slot still holds
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::vector<runtime::prefix_cache::TapHint> hints; // the prompt's (the generation opener)
    // Block lookup hashes: the prompt's, extended by publication over the generated blocks.
    std::vector<std::uint64_t> hashes;
    std::uint64_t trailing_extra = 0;
    // The admission's diagnostics: the prompt prefix held as cached blocks, and the bytes its
    // restore copied back from the Host tier.
    std::uint32_t cached_tokens  = 0;
    std::uint64_t restored_bytes = 0;
};

// A media prompt's vision encode (vision.cpp): the tower items past the reused prefix (an image,
// or one frame pair of a video), encoded at the lane's first prefill call into one token-ordered
// handoff BF16 [H, V] whose columns the prompt's calls scatter over their token embeddings. The
// handoff lives in lent expert frames (or, when the residency lends none, an allocation) until the
// prompt's last call has completed.
struct LaneVision {
    struct Segment {
        std::span<const std::uint16_t> patches; // BF16 [patch_width, P] of a payload
        std::vector<std::int32_t> positions;    // I32 [2, P]
        std::uint32_t token_begin = 0, tokens = 0;
    };

    std::vector<std::shared_ptr<const PreparedMediaPayload>> payloads; // the segments' patches
    std::vector<Segment> segments;
    // The encoded tokens' positions, ascending: handoff column c is token visual[c].
    std::vector<std::uint32_t> visual;
    bool encoded = false;
    offload::ExpertResidency::FrameLease lease;
    DeviceBuffer fallback;
    Tensor handoff;
};

struct ResumeStateImpl {
    std::uint32_t target = 0;
};

struct ContractAccess {
    static SequenceHandle make_sequence(const void* owner, std::uint32_t lane, std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }
    static const void* owner(const SequenceHandle& h) noexcept { return h.owner_; }
    static std::uint32_t lane(const SequenceHandle& h) noexcept { return h.lane_; }
    static std::uint64_t epoch(const SequenceHandle& h) noexcept { return h.epoch_; }

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                                     std::span<const TokenId> tokens, runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_ = tokens;
        out.timing_ = timing;
        return out;
    }

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction,
                                     std::span<const SequenceHandle> rows,
                                     std::span<const TokenId> tokens,
                                     runtime::ExecutionTiming timing,
                                     std::span<const std::int32_t> counts, std::uint32_t stride) {
        PendingBatch out = make_pending(owner, transaction, rows, tokens, timing);
        out.row_counts_  = counts;
        out.row_stride_  = stride;
        return out;
    }
    static void set_row_failure(PendingBatch& p, std::size_t row, runtime::PendingRowFailure failure) noexcept {
        p.row_failures_[row] = failure;
    }
    static const void* owner(const PendingBatch& p) noexcept { return p.owner_; }
    static std::uint64_t transaction(const PendingBatch& p) noexcept { return p.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& p) noexcept { return {p.rows_.data(), p.row_count_}; }
    static void consume(PendingBatch& p) noexcept {
        p.owner_       = nullptr;
        p.transaction_ = 0;
        p.row_count_   = 0;
        p.tokens_      = {};
        p.row_counts_  = {};
        p.row_stride_  = 1;
    }
};

// Per-call inputs staged in pinned memory and uploaded with one copy: ids and positions [columns],
// KDA slots and block-table rows [batch], logit columns [columns].
struct IoLayout {
    std::size_t ids = 0, positions = 0, slots = 0, rows = 0, columns = 0, bytes = 0;
};

// The DFlash2 rounds' I32 words (offsets in words), staged in pinned memory: the proposal's inputs
// [L] and block [W, L], its drafts and laws, a call's context positions, the acceptance's inputs
// and outputs (licensed tokens, their counts and the accepted drafts adjacent, read back in one
// copy) and the commit's counts; L lanes, W = K + 1.
struct DFlash2Io {
    std::size_t anchors = 0, frontiers = 0, valid = 0, slots = 0, ids = 0, positions = 0,
                drafts = 0, confidence = 0, candidates = 0, q = 0, context_positions = 0,
                context_count = 0,
                context_slot = 0, ones = 0, target = 0, extents = 0, lengths = 0, round_anchors = 0,
                licensed = 0, counts = 0, accepted = 0, commit = 0, words = 0;
};

// Expert promotions one layer's routes may start per decode token (the shared admission budget,
// moe-offload-design.md §4.3).
inline constexpr std::size_t kDecodePromotionsPerLayer = 1;
inline constexpr std::uint64_t kDecodePromotionInterval = 32;

// Causal scoring's logits tile: 64 columns of FP32 logits.
inline constexpr std::int32_t kScoreTileColumns = 64;
// The tile's words: targets and their log-probabilities, then top ids and log-probabilities and
// candidate ids and log-probabilities of up to kMaximumScoreTopTokens per column.
inline constexpr std::size_t kScoreIoWords =
    static_cast<std::size_t>(kScoreTileColumns) * (2 + 4 * kMaximumScoreTopTokens);

// Every fixed device allocation of a Program and the layouts in it. One function
// (ProgramImpl::plan_device) computes it from the options and the configuration alone: before the
// weights are read, for the startup check, and in the constructor, which allocates exactly it.
struct DeviceLayout {
    std::int32_t lanes = 0, token_domain = 0, chunk = 0, columns = 0, pages_per_row = 0;
    std::uint32_t kv_pages = 0;
    LinearAttentionStatePoolLayout kda;
    std::size_t kda_bytes = 0;
    std::size_t index_ring = 0; // per MLA block and lane: the indexer's recent keys and gates
    KVPageGeometry geometry;
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    std::size_t kv = 0;
    IoLayout io;
    std::size_t vision_io = 0;
    std::size_t logits32 = 0, logits16 = 0, token_counts = 0, constraint = 0, grammar = 0, work = 0;
    std::int32_t staging_slots = 0;
    std::size_t staging        = 0;
    std::size_t score_residuals = 0, score_logits = 0, score_logits16 = 0, score_io = 0;
    // DFlash2 (ProgramOptions::draft_tokens): K drafts a round of width W = K + 1, the features of
    // a round's or a call's last window of columns, every lane's context ring and the KDA blocks'
    // verify records of a round of every lane.
    std::int32_t draft = 0, width = 1, feature_columns = 0;
    std::size_t features = 0, ring = 0, records = 0;
    CyclicKVCacheLayout ring_layout;
    DFlash2Io dflash_io;
    // The prefix cache (ProgramOptions::prefix_cache): the prompt-end captures' staging slots
    // (min(lanes, 2) state images) and every lane's saved head residual.
    std::size_t prefix_staging = 0, head_saved = 0;
    ProgramDevicePlan bytes;
};

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Prefill, Decode, Finishable };

    struct Lane {
        Phase phase         = Phase::Free;
        std::uint64_t epoch = 0;
        runtime::BeginSummary begin;
        std::vector<std::int32_t> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        std::uint32_t extent        = 0; // the positions the request may ever hold (prompt + output)
        // The prompt's prefill calls: exclusive ends, the last prompt_tokens.
        std::vector<std::uint32_t> calls;
        std::size_t next_call = 0;
        ops::SamplingConfig sampling;
        // /v1/decide: the tokens the prompt's readout names, the constraint, the draws of the
        // pending round's token and the committed trace they move into.
        std::vector<TokenId> readout;
        std::shared_ptr<const ConstraintPlan> constraint;
        std::shared_ptr<const PreparedPrompt> prompt;
        std::vector<ConstrainedDraw> constraint_round, constraint_trace;
        std::optional<DeviceKVPageReservation> reservation;
        std::vector<DeviceKVPageLease> pages;
        KVExecutionRowLease row;
        LanePrefix prefix;
        std::uint64_t prefill_ns      = 0;
        std::uint64_t decode_ns       = 0;
        std::uint64_t decode_share_ns = 0;
        offload::ExpertResidency::Stats cache_at_admission;
        // Prompt seeding: of the prompt's prefilled tokens counted, how many routed to each (MoE
        // layer, expert); [MoE layers][experts], allocated on first use.
        std::vector<std::uint32_t> prompt_uses;
        std::uint32_t prompt_counted = 0;
        SpeculativeStats speculative;
        // --draft-tokens auto: the request's selector calibration and its speculating rounds (every
        // kProbeInterval-th verifies one draft more than its choice).
        SelectorCalibration calibration;
        std::uint64_t policy_rounds = 0;
        std::unique_ptr<LaneVision> vision; // a media prompt's encode, until its prefill ends
    };

    struct DecodeGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        bool warmed   = false;
        bool resident = false; // the expert route it was captured with
    };

    // ---------------------------------------------------------------- construction (construct.cpp)
    [[nodiscard]] static DeviceLayout plan_device(const ProgramOptions& o, const Config& config,
                                                  std::int32_t public_tokens);
    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options);
    ~ProgramImpl();
    ProgramImpl(const ProgramImpl&)            = delete;
    ProgramImpl& operator=(const ProgramImpl&) = delete;
    [[nodiscard]] const DeviceLayout& device_layout() const noexcept { return plan_; }

    // ---------------------------------------------------------------- admission (binding.cpp)
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] bool feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base,
                                                              std::uint32_t maximum_frontier);
    [[nodiscard]] runtime::ResourceReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                                             const SourceCandidate& source, ResumeState* resume);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool in_transaction() const noexcept { return transaction_lane_.has_value(); }
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] bool start_pause(SequenceHandle sequence);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence);

    // The blocked FIFO head's Host-only blocks are copied into spare Device cache while it waits
    // (hybrid-prefix-cache-spec §6.6).
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    // The snapshots queued requests would resume from, held in admission order (§9.6).
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept;

    // ---------------------------------------------------------------- execution (execution.cpp)
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets, runtime::TokenMaskProvider* masks);
    [[nodiscard]] runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences,
                                                         std::span<const TokenId> tokens, std::uint32_t stride);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions);
    [[nodiscard]] DiscardResult discard(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] bool recover_after_failure() noexcept;
    void release_all() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot usage() const noexcept;
    [[nodiscard]] MemorySummary memory() const noexcept;

    // ---------------------------------------------------------------- causal scoring (score.cpp)
    [[nodiscard]] ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                           const ScoreOptions& options);

    // ---------------------------------------------------------------- experts (experts.cpp)
    void set_maintenance_waker(std::function<void()> waker) { placement_->set_maintenance_waker(std::move(waker)); }
    // Between rounds and while idle (Program::maintain): the placement's pending switch and
    // fills, and the expert state's periodic save.
    void maintain();
    // Saves the expert cache's state (at stop).
    void shutdown_experts() noexcept { placement_->save_expert_state(); }
    [[nodiscard]] const offload::VramSizing& vram_sizing() const noexcept { return sizing_; }
    [[nodiscard]] offload::ExpertPlacementStatus placement_status() const { return placement_->placement_status(); }
    [[nodiscard]] bool request_placement(const offload::ExpertPlacementRequest& request, std::string* error) {
        return placement_->request_placement(request, error);
    }
    [[nodiscard]] bool request_count_placement(std::uint32_t vram, std::uint32_t ram, std::string* error) {
        return placement_->request_count_placement(vram, ram, error);
    }
    [[nodiscard]] bool export_placement(const std::filesystem::path& path, std::string* error) {
        return placement_->export_placement(path, error);
    }

private:
    // ---- construction helpers (construct.cpp)
    void allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally);

    // ---- admission helpers (binding.cpp)
    [[nodiscard]] std::vector<std::uint32_t> plan_calls(std::uint32_t begin, std::uint32_t end) const;
    void reset_slot(std::uint32_t slot);

    // ---- execution helpers (execution.cpp)
    [[nodiscard]] std::uint32_t lane_of(const SequenceHandle& sequence) const;
    [[nodiscard]] SequenceHandle handle(std::uint32_t lane) const noexcept {
        return ContractAccess::make_sequence(this, lane, lanes_[lane].epoch);
    }
    [[nodiscard]] GenerationTimings timings(const Lane& lane) const;
    void release(std::uint32_t index) noexcept;
    void split_beside_decode(Lane& lane, std::uint32_t index);
    void stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width);
    void stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions);
    // One Forward call over the staged io: `batch` sequences of `width` columns, logits of
    // `logit_columns` columns. Decode rounds replay their batch size's graph.
    void run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns, bool decode_round);
    // `verify`: the call verifies drafts (every column's logits, KDA blocks read-only).
    void forward_call(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                      bool verify = false);
    void replay(DecodeGraph& graph, const std::function<void()>& body);
    // Whether this call streams its experts (execution.cpp): a prefill call of at least
    // kStreamMinColumns columns over a residency with frames, lending the ring first when none is
    // held. Decode and verification rounds never stream.
    bool stream_chunk(std::int32_t width);

    // ---- prefill layer walk (prefill_walk.cpp; glm5_3-flash-design.md §11.1)
    // A span of consecutive streamed calls of one prompt, run layer-major: chunk c's layer l runs
    // after chunk c - 1's, and every chunk's layer l before any chunk's layer l + 1. Each step
    // (advance_prefill call) enqueues a few layers of every chunk and synchronizes, so other lanes'
    // rounds run between steps; the span's residual stream and mHC state live in the walk lease.
    struct LayerWalk {
        bool active            = false;
        std::uint32_t lane     = 0;
        std::int32_t host_slot = 0;
        std::size_t first_call = 0, end_call = 0;
        std::vector<std::int32_t> begins, widths, offsets; // per chunk; offsets in span columns
        std::int32_t columns = 0;                          // the span's tokens
        // Area offsets (bytes): the residual stream, mix, post and comb of every column, the ids
        // and positions, each chunk's {slot, row, logit column} and DFlash2 features.
        std::size_t residual = 0, x = 0, post = 0, comb = 0, ids = 0, positions = 0, vision = 0,
                    meta = 0, features = 0, feature_stride = 0, bytes = 0;
        // Per chunk: its media columns' first handoff column and count (vision.cpp).
        std::vector<std::uint32_t> vision_first, vision_count;
        std::uint32_t next_layer = 0, steps = 0;

        // A prefix tap due at an inner chunk's end (block-aligned): its image is copied layer by
        // layer in place and committed after the span.
        struct Tap {
            std::uint32_t frontier = 0;
            prefix::cache_index::SnapshotKind kind{};
            std::uint64_t staged = 0; // 0: none at this chunk's end
        };

        std::vector<Tap> taps; // per chunk
    };

    // Spans end at kWalkMaxTokens and at a call whose end an unaligned prefix tap is due at
    // (captured after the span, at the state the chunk-major run has there); aligned taps at inner
    // chunk ends are captured in place.
    static constexpr std::uint32_t kWalkMaxTokens = 32768;
    [[nodiscard]] LayerWalk walk_layout(const Lane& lane, std::size_t end) const;
    // The exclusive end of the span starting at lane.next_call (lane.next_call: chunk-major),
    // lending the ring and the walk area.
    [[nodiscard]] std::size_t walk_span_end(Lane& lane);
    [[nodiscard]] bool tap_due(const Lane& lane, std::uint32_t B,
                               std::size_t next_call) const noexcept;
    void walk_begin(Lane& lane, std::uint32_t index, std::size_t end);
    // The span's io, the lane's pending copies and the residency round its stream begins in.
    void walk_stage(Lane& lane, std::uint32_t index);
    [[nodiscard]] PrefillProgress walk_step(Lane& lane, std::uint32_t index,
                                            Clock::time_point start);
    [[nodiscard]] execution::ForwardBatch walk_batch(std::size_t c) const;
    [[nodiscard]] execution::Forward::CallColumns walk_columns(std::size_t c) const;
    void walk_abandon() noexcept;
    // The taps due at the span's inner chunk ends, as the chunk-major calls would take them.
    void walk_plan_taps(Lane& lane);
    void walk_commit_taps(Lane& lane);

    [[nodiscard]] std::byte* walk_area() const noexcept {
        return static_cast<std::byte*>(walk_lease_.memory.data);
    }

    // The prompt's last call is enqueued (by a call or a walk): sample, or settle and report.
    [[nodiscard]] PrefillProgress after_prefill(Lane& lane, std::uint32_t index, std::int32_t begin,
                                                std::int32_t width, std::uint32_t tokens, bool last,
                                                std::size_t promotions, Clock::time_point start);

    // The ring (the stream lease).
    [[nodiscard]] DeviceSpan ring_span() const noexcept {
        return DeviceSpan{stream_lease_.memory.data, stream_ring_bytes_};
    }

    // Returns the ring (after the last streamed call is enqueued: compute waited for every copy),
    // whether one is held.
    void return_stream_lease();
    // Whether a lane other than `index` is prefilling.
    [[nodiscard]] bool prefilling_besides(std::uint32_t index) const noexcept;
    [[nodiscard]] execution::ForwardBatch io_batch(std::int32_t batch, std::int32_t width,
                                                   std::int32_t logit_columns) const;
    [[nodiscard]] Tensor read_logits(std::int32_t columns);
    void round_logits(std::int32_t columns);
    void sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions);
    void note_prompt_routes(Lane& lane, std::int32_t width);
    void seed_prompt_experts(Lane& lane);
    // After a call completed (the stream was synchronized): the expert cache's round boundary.
    void after_call(std::int32_t columns, std::size_t promotions);

    // The promotions a served call of `tokens` columns may make at its round boundary: one per
    // layer every promotion_interval_ tokens.
    [[nodiscard]] std::size_t decode_budget(std::uint32_t tokens) noexcept {
        budget_tokens_ += tokens;
        const std::uint64_t due = budget_tokens_ / promotion_interval_;
        budget_tokens_ %= promotion_interval_;
        return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
    }

    std::uint64_t budget_tokens_      = 0;
    std::uint64_t promotion_interval_ = kDecodePromotionInterval;

    // ---- constraints (constraint.cpp)
    [[nodiscard]] std::shared_ptr<const ConstraintPlan> compile_constraint(const TokenConstraint& constraint);
    [[nodiscard]] bool stage_constraints(std::span<const std::uint32_t> lanes);
    void constrain(Tensor& logits);
    void take_constraint_draws(Lane& lane, std::size_t column, TokenId token);
    void commit_constraint_draws(Lane& lane);
    [[nodiscard]] PromptReadout read_prompt_frontier(const Lane& lane);
    [[nodiscard]] static std::size_t constraint_bytes(std::int32_t lanes) noexcept;
    // Grammar masks of `lanes` rows of `width` positions each.
    [[nodiscard]] static std::size_t
    grammar_mask_bytes(std::int32_t token_domain, std::int32_t lanes, std::int32_t width) noexcept;
    [[nodiscard]] std::size_t observe_offset() const noexcept;
    // Row `row`'s masks for its position and each draft's after it.
    [[nodiscard]] ops::SamplingMask grammar_mask(std::size_t row,
                                                 std::span<const TokenId> drafts = {});
    void reasoning_ban(std::size_t row, ops::SamplingConfig& config) const noexcept;
    // `counts` (empty: one each) are the rows' licensed tokens, the positions a dead end fails at.
    void mark_row_failures(PendingBatch& pending,
                           std::span<const std::int32_t> counts = {}) noexcept;
    struct GrammarScope {
        ProgramImpl& impl;
        GrammarScope(ProgramImpl& owner, runtime::TokenMaskProvider* masks) : impl(owner) { impl.grammar_ = masks; }
        ~GrammarScope() { impl.grammar_ = nullptr; }
        GrammarScope(const GrammarScope&)            = delete;
        GrammarScope& operator=(const GrammarScope&) = delete;
    };

    // ---- DFlash2 (dflash2.cpp)
    void allocate_dflash2(std::uint64_t& allocated);
    [[nodiscard]] Tensor feature_columns(std::int32_t n) const;

    [[nodiscard]] std::int32_t* dflash_host(std::size_t word) const noexcept {
        return static_cast<std::int32_t*>(dflash_host_.data()) + word;
    }

    [[nodiscard]] std::int32_t* dflash_device(std::size_t word) const noexcept {
        return static_cast<std::int32_t*>(dflash_device_.p) + word;
    }

    // The drafts a row may verify this round: K, fewer near its extent or output budget.
    [[nodiscard]] std::int32_t draft_room(const Lane& lane,
                                          const runtime::RoundBudget& budget) const noexcept;
    // A single-row call's last min(width, window) columns into its lane's ring (after its forward).
    void call_context(std::uint32_t lane, std::int32_t begin, std::int32_t width);
    // A plain decode round's columns into its rows' rings (inside the round's graph).
    void round_context(std::int32_t batch);
    // A one-row round's drafts to verify after its proposal: the room, cut at the first draft below
    // draft_min_p, then with auto_draft_length the calibrated length (0: the round decodes plainly).
    [[nodiscard]] std::int32_t verify_length(Lane& lane, std::int32_t room);
    // A one-row round of `width` columns after its proposal: verifies the proposal's first
    // width - 1 drafts and accepts.
    [[nodiscard]] PendingBatch speculate(std::span<const SequenceHandle> sequences,
                                         std::span<const std::uint32_t> lanes,
                                         std::span<const std::int32_t> positions,
                                         std::int32_t width, Clock::time_point start);
    void propose(std::span<const std::uint32_t> lanes, std::int32_t width);
    [[nodiscard]] CommitResult commit_verified(PendingBatch&& pending,
                                               std::span<const runtime::CommitDecision> decisions);
    [[nodiscard]] CommitRowResult::SpeculativeCounters
    speculative_counters(const SpeculativeStats& stats) const noexcept;

    [[nodiscard]] DecodeGraph& graph(std::vector<DecodeGraph>& graphs, std::int32_t batch,
                                     std::int32_t width) {
        return graphs[static_cast<std::size_t>(batch - 1) * static_cast<std::size_t>(plan_.draft) +
                      static_cast<std::size_t>(width - 2)];
    }

    // ---- vision (vision.cpp)
    // At binding: the segments of `prompt` ending past `reused` tokens (none: no encode).
    void vision_plan(Lane& lane, const PreparedPromptData& prompt, std::uint32_t reused);
    // Before the lane's first prefill call: the tower's pass into the handoff.
    void vision_encode(Lane& lane);
    // The handoff columns of the call [begin, begin + width) and, into `columns`, their columns
    // relative to begin: {first, count}.
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t>
    vision_columns(const Lane& lane, std::uint32_t begin, std::uint32_t width,
                   std::int32_t* columns) const;
    // The handoff columns [first, first + count) and their device columns as the batch's media.
    void vision_batch(const Lane& lane, std::uint32_t first, std::uint32_t count,
                      std::byte* columns, execution::ForwardBatch& batch) const;
    // After the prompt's last call has completed (or at release).
    void vision_release(Lane& lane) noexcept;

    // ---- scoring (score.cpp)
    [[nodiscard]] Tensor score_residuals(std::int32_t begin, std::int32_t columns) const;

    // ---------------------------------------------------------------- prefix cache
    // (prefix_program.cpp)
    void create_prefix_cache(std::uint64_t& allocated);
    void pc_make_room(std::uint32_t pages);
    // Chooses the resume source for `prompt` in lane `lane` (kNoLane while quoting): the deepest
    // affordable chunk-grid snapshot, else the root, or nothing when even the root does not fit
    // now.
    std::optional<PrefixSelection> prefix_select(const PreparedPromptData& prompt,
                                                 const BasePlanImpl& base, std::uint32_t lane);
    // The root start of a request with a prefix cache: every page reserved, the cold call grid.
    PrefixSelection prefix_root(const PreparedPromptData& prompt, const BasePlanImpl& base);
    // The prompt's prefill calls from `frontier`, and with `taps` the prefix taps they realize: a
    // tap lands on the chunk grid, so every snapshot a resume takes keeps the publishing run's call
    // boundaries (the phase-aligned chunking of the exactness contract).
    [[nodiscard]] prefix::CallPlan plan_prefill(const PreparedPromptData& prompt,
                                                std::uint32_t frontier,
                                                std::span<const std::uint32_t> existing, bool taps,
                                                std::uint32_t branch = 0) const;
    // What a call costs beyond its tokens, for the call planner's tap admission.
    [[nodiscard]] prefix::CallCost call_cost() const;
    // Pins what a selection reads before room is made for it, so making room cannot evict it.
    void prefix_pin(const PrefixSelection& selection);
    void prefix_unpin(const PrefixSelection& selection) noexcept;
    // Activates a pinned selection in a lane whose pages are reserved: maps shared pages, stages
    // restores and copy-on-write sources, and sets the lane's frontier.
    void prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& selection,
                         DeviceKVPageReservation& reservation, std::uint32_t pages);
    // After a prefill call ending at the lane's frontier: publish its blocks and realize the taps
    // due at this call boundary; after the last one, the prompt-end capture.
    void prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last);
    void prefix_publish(Lane& lane, std::uint32_t frontier);
    void prefix_capture(Lane& lane, std::uint32_t index, prefix::cache_index::SnapshotKind kind);
    [[nodiscard]] prefix::PrefixCache::Capture
    prefix_capture_at(Lane& lane, std::uint32_t F, prefix::cache_index::SnapshotKind kind,
                      bool hand_over_tail, DeviceKVPageLease* fresh_tail = nullptr);
    void prefix_captured(Lane& lane, std::uint64_t id, std::uint32_t F);
    bool prefix_take_taps(Lane& lane, std::uint32_t B, std::size_t next_call, bool& boundary);
    // The zero-token resume's prompt-end capture (design §12): a snapshot at the prompt's own
    // frontier n whose image carries the head's saved residual, that an exact repeat resumes from
    // with no prefill at all.
    void prefix_prompt_end_capture(Lane& lane, std::uint32_t index);
    // Finish or a consistent abort: the remaining blocks, then write-through.
    void prefix_finish(Lane& lane, std::uint32_t index);
    void prefix_release(Lane& lane) noexcept;
    // The lane's pending capture and restore copies, as one event each (empty when none): the
    // lane's next call waits on them before it runs.
    [[nodiscard]] std::span<const cudaEvent_t> prefix_capture_wait(const Lane& lane) const noexcept;
    [[nodiscard]] std::span<const cudaEvent_t> prefix_restore_wait(const Lane& lane) const noexcept;
    [[nodiscard]] std::optional<prefix::cache_index::SnapshotRef>
    prefix_resident(const Lane& lane) const;
    // Credits what needs no transfer, then leaves only sources restorable now.
    void prefix_credit(std::vector<prefix::cache_index::MatchCandidate>& candidates,
                       std::optional<prefix::cache_index::SnapshotRef> resident) const;

    static constexpr std::uint32_t kNoLane = UINT32_MAX;

    // ---- experts (experts.cpp)
    // Builds the frame cache and the RAM/SSD tier and fills `experts` with what the MoE reads.
    void create_experts(execution::ForwardExperts& experts, std::uint64_t& allocated);

    // The prefill expert stream's bindings for the tier as it is now: its records, stream slots
    // and bounce buffers, or the artifact banks' records without a tier. The callbacks capture
    // the tier object, so a tier rebuild (the placement runtime's) retargets the stream with a
    // fresh set.
    struct StreamBindings {
        execution::ExpertStream::RecordOf record_of;
        execution::ExpertStream::Hold hold;
        execution::ExpertStream::SsdReads ssd;
        execution::ExpertStream::Staging staging;
        bool one_allocation = true;
    };

    [[nodiscard]] StreamBindings stream_bindings() const;
    // Before a call: the frame cache's round boundary (decode rounds reserve landing frames).
    void before_call(bool decode_round);
    // After the call's kernels are enqueued: its route log comes back with it.
    void enqueue_routes(std::int32_t columns);
    [[nodiscard]] bool expert_route() const noexcept;
    void set_expert_route();
    // Reads the device's error word after a synchronization; throws the call's fault.
    void check_device_error();
    [[nodiscard]] std::optional<ExpertCacheStats> expert_cache_stats(const Lane& lane) const;

    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const;

    const execution::Parameters& parameters_;
    const TextConfig& c_;
    DeviceContext& device_;
    ProgramOptions options_;
    DeviceLayout plan_;
    std::int32_t token_domain_ = 0, vocab_ = 0, width_ = 0, chunk_ = 0, columns_ = 0;
    std::uint32_t beside_decode_chunk_ = 0;

    std::vector<Lane> lanes_;
    std::uint64_t next_epoch_       = 0;
    std::uint64_t next_transaction_ = 0;
    std::uint64_t pending_transaction_ = 0;
    std::optional<std::uint32_t> transaction_lane_;
    // The pending round's tokens: one a row, or each row's licensed tokens at row * round_width_.
    std::array<TokenId, kMaximumConcurrency * 16> pending_tokens_{};
    std::array<std::int32_t, kMaximumConcurrency> pending_counts_{};
    std::int32_t round_width_ = 1;
    std::array<std::int32_t, kMaximumConcurrency> host_lanes_{};

    DeviceBuffer kda_backing_, index_ring_, kv_backing_, io_device_, logits32_, logits16_, sampled_, sample_pos_, configs_,
        token_counts_, constraint_, grammar_masks_, staging_, score_residuals_, score_logits_, score_logits16_, score_io_;
    PinnedHostBuffer io_host_{1}, host_sampled_{1}, host_configs_{1}, grammar_host_{1}, score_host_{1};
    // A chunk-major call's media columns (I32 [chunk]), uploaded only with media.
    DeviceBuffer vision_io_device_;
    PinnedHostBuffer vision_io_host_{1};
    // The chunk-major call being run carries lane vision_call_lane_'s handoff columns
    // [first, first + count) (-1: none).
    std::int32_t vision_call_lane_ = -1;
    std::pair<std::uint32_t, std::uint32_t> vision_call_{};
    PinnedHostBuffer nonfinite_flags_{sizeof(std::int32_t) * kMaximumConcurrency};
    PinnedHostBuffer device_error_{64};
    [[nodiscard]] std::int32_t* nonfinite_flag(std::uint32_t lane) const noexcept {
        return static_cast<std::int32_t*>(nonfinite_flags_.data()) + lane;
    }
    [[nodiscard]] std::byte* host_io() const noexcept { return static_cast<std::byte*>(io_host_.data()); }
    std::unique_ptr<LinearAttentionStatePool> kda_;
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVExecutionTablePool> tables_;
    std::unique_ptr<WorkspaceArena> work_;
    std::unique_ptr<execution::Forward> forward_;
    std::array<DecodeGraph, kMaximumConcurrency> graphs_;

    // The DFlash2 drafter: the context ring, the round's or call's target features, the KDA verify
    // records and the drafter's io; proposal, verify and context graphs per (batch, width).
    std::unique_ptr<execution::DFlash2> drafter_;
    DeviceBuffer ring_backing_, features_, records_backing_, dflash_device_;
    PinnedHostBuffer dflash_host_{1};
    std::unique_ptr<CyclicKVCache> ring_;
    std::vector<execution::KdaVerifyRecord> records_;
    std::vector<DecodeGraph> propose_graphs_, verify_graphs_, context_graphs_;
    std::array<std::array<TokenId, 16>, kMaximumConcurrency> drafts_{};
    // Each draft's selector probability (temperature 1), read back with the drafts.
    std::array<std::array<float, 16>, kMaximumConcurrency> confidence_{};
    static constexpr std::uint64_t kProbeInterval = 8;
    static constexpr double kCalibrationWeight    = 0.1;

    // The prefix cache: the index and its physical binding, the packed state image of a lane, the
    // prompt-end captures' staging slots and every lane's saved head residual.
    std::unique_ptr<prefix::PrefixCache> prefix_;
    std::optional<prefix::RestoreTicket> prefetch_ticket_;
    std::uint64_t next_capture_ = 0;
    std::unique_ptr<prefix::StateImageLayout> image_layout_;
    std::unique_ptr<prefix::LaneStateImage> lane_image_;
    std::vector<DeviceBuffer> prompt_end_staging_;
    DeviceBuffer head_saved_;
    prefix::KvPageGeometry kv_geometry_;
    bool scoring_              = false;
    std::int32_t score_column_ = 0;
    // The lane whose call keeps its final residual (the zero-token resume's head input); -1 none.
    std::int32_t head_saved_lane_ = -1;

    // Constraints and grammar masks.
    runtime::TokenMaskProvider* grammar_ = nullptr;
    std::array<std::uint64_t, kMaximumConcurrency> grammar_dead_{};
    std::vector<std::int32_t> constraint_host_;
    std::array<std::uint64_t, kMaximumConcurrency> constraint_serials_{};
    std::uint64_t next_constraint_serial_ = 1;
    std::int32_t constraint_columns_      = 0;

    // The routed experts: the shared placement runtime owns the residency, the RAM tier and its
    // fetch channel (the pointers below are its); the budget source outlives it.
    cudaStream_t overlap_stream_ = nullptr;
    std::array<cudaEvent_t, 5> overlap_events_{};
    // Chunks of at least kStreamMinColumns columns route nearly every expert of a layer, so all
    // of a layer's non-resident experts are copied ahead; narrower calls keep the MoE's staging.
    // The ring holds two halves of as many records as the layer with the most non-resident
    // experts has when it is lent (+ one frame for the slot tables), so no streamed layer leaves
    // misses to staging; it is lent by the expert cache for the prompt and returned when the
    // prompt's prefill ends and no lane is prefilling (execution.cpp).
    static constexpr std::int32_t kStreamMinColumns = 256;
    static constexpr std::uint32_t kStreamHalfMin   = 16;
    // The SSD tier's slots the stream borrows for its reads of SSD-only experts (a later layer
    // walk's span; a single call leaves SSD-only experts to the fetch channel, so the tier lends
    // none while only calls stream).
    static constexpr std::uint32_t kStreamReadSlots = 1280;
    // The prefill stream's bounce per part: a chunk streams about a layer's unpinned share of its
    // experts.
    static constexpr std::uint32_t kStreamBounceMin = 64, kStreamBounceMax = 256;
    std::unique_ptr<execution::ExpertStream> expert_stream_;
    // The stream's lease: the ring, a run of one pool's frames.
    offload::ExpertResidency::FrameLease stream_lease_;
    std::size_t stream_ring_bytes_ = 0;
    execution::ExpertStream::Stats stream_at_lease_;
    LayerWalk walk_;
    offload::ExpertResidency::FrameLease walk_lease_; // the active span's area
    PinnedHostBuffer walk_host_{1};                   // the span's ids, positions and chunk words
    std::unique_ptr<VramBudgetSource> vram_;
    std::unique_ptr<offload::ExpertPlacementRuntime> placement_;
    offload::ExpertResidency* residency_                = nullptr;
    offload::HostExpertTier* tier_                      = nullptr;
    ops::offloaded_moe::FetchChannel* fetch_channel_    = nullptr;
    offload::VramSizing sizing_;
    // CPU-served misses of decode and verification calls (moe-offload-design.md §5): declared after
    // the placement runtime, whose tier its jobs read through, so it stops first.
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;
};

} // namespace infernix::models::glm5_next::detail
