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
#include "models/offload/expert_residency.h"
#include "models/offload/host_expert_tier.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
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
    std::uint32_t pages     = 0; // KV page groups of the extent
    std::uint32_t positions = 0; // the extent: prompt + effective output, at most max_context
    std::shared_ptr<const PreparedPrompt> prompt; // the request's prompt, moved in by plan_request
};

struct QuoteImpl {
    std::uint32_t pages = 0;
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
                drafts = 0, candidates = 0, q = 0, context_positions = 0, context_count = 0,
                context_slot = 0, ones = 0, target = 0, extents = 0, lengths = 0, round_anchors = 0,
                licensed = 0, counts = 0, accepted = 0, commit = 0, words = 0;
};

// Expert promotions one layer's routes may start per decode token (the shared admission budget,
// moe-offload-design.md §4.3).
inline constexpr std::size_t kDecodePromotionsPerLayer = 1;

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
        std::uint64_t prefill_ns      = 0;
        std::uint64_t decode_ns       = 0;
        std::uint64_t decode_share_ns = 0;
        offload::ExpertResidency::Stats cache_at_admission;
        // Prompt seeding: of the prompt's prefilled tokens counted, how many routed to each (MoE
        // layer, expert); [MoE layers][experts], allocated on first use.
        std::vector<std::uint32_t> prompt_uses;
        std::uint32_t prompt_counted = 0;
        SpeculativeStats speculative;
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
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base);
    [[nodiscard]] runtime::ResourceReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                                             const SourceCandidate& source, ResumeState* resume);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool in_transaction() const noexcept { return transaction_lane_.has_value(); }
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] bool start_pause(SequenceHandle sequence);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence);

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
    // The promotions a served call of `tokens` columns may make at its round boundary.
    [[nodiscard]] static std::size_t decode_budget(std::uint32_t tokens) noexcept {
        return kDecodePromotionsPerLayer * tokens;
    }

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
    // A round of `width` columns a row: proposes, verifies and accepts.
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

    // ---- scoring (score.cpp)
    [[nodiscard]] Tensor score_residuals(std::int32_t begin, std::int32_t columns) const;

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
    bool scoring_              = false;
    std::int32_t score_column_ = 0;

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
    std::unique_ptr<VramBudgetSource> vram_;
    std::unique_ptr<offload::ExpertPlacementRuntime> placement_;
    offload::ExpertResidency* residency_                = nullptr;
    offload::HostExpertTier* tier_                      = nullptr;
    ops::offloaded_moe::FetchChannel* fetch_channel_    = nullptr;
    offload::VramSizing sizing_;
};

} // namespace infernix::models::glm5_next::detail
