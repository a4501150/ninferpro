#pragma once

// The Glm5Next (GLM-5.3-Flash) Program: the Engine-facing execution contract of the common
// EngineCore over the HybridResourceManager surface (docs/maintainer/engine-architecture.md §2.4,
// §3.3; docs/maintainer/glm5_3-flash-design.md §4.1). It owns the physical lanes (one per
// concurrency slot), their KV pages (the MLA latent and the DSA indexer's planes), their KDA
// recurrent state, sampling state and the one pending model unit, and the routed experts' VRAM
// frames and RAM/SSD tier.
//
// Admission reserves a request's whole extent (prompt plus effective output), so no unit of it
// grows and nothing pauses it; there is no prefix cache yet, so every request starts at the root.
// With the DFlash2 drafter (docs/maintainer/dflash.md) every lane also keeps the drafter's context
// ring, and decode rounds propose and verify drafts.

#include "models/glm5_next/config.h"
#include "models/glm5_next/frontend/frontend.h"
#include "models/glm5_next/frontend/output_session.h"
#include "models/glm5_next/frontend/prepared_prompt.h"
#include "models/offload/expert_placement.h"
#include "models/offload/expert_placement_runtime.h"
#include "infernix/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"
#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix {
struct DeviceContext;
}

namespace infernix::models::glm5_next {

namespace execution {
class Parameters;
}

namespace detail {
class ProgramImpl;
struct BasePlanImpl;
struct QuoteImpl;
struct ResumeStateImpl;
struct ContractAccess;
} // namespace detail

// Read-only diagnostics of the Program's physical stores.
struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
};

// The hybrid prefix cache's statistics, as the HybridResourceManager reads them: all zero (this
// Program has no prefix cache).
struct HybridPrefixCacheStats {
    std::uint32_t nodes                   = 0;
    std::uint32_t snapshots               = 0;
    std::uint32_t device_resident_blocks  = 0;
    std::uint32_t device_evictable_blocks = 0;
    std::uint32_t host_slabs              = 0;
    std::uint32_t host_free_slabs         = 0;
    std::uint64_t host_slab_bytes         = 0;
    std::uint64_t snapshot_hits           = 0;
    std::uint64_t reused_tokens           = 0;
    std::uint64_t blocks_inserted         = 0;
    std::uint64_t blocks_reattached       = 0;
    std::uint64_t blocks_duplicate        = 0;
    std::uint64_t taps_created            = 0;
    std::uint64_t taps_skipped            = 0;
    std::uint64_t prompt_end_captures     = 0;
    std::uint64_t prompt_end_gated        = 0;
    std::uint64_t prompt_end_failed       = 0;
    std::uint64_t endpoints_created       = 0;
    std::uint64_t host_image_writes       = 0;
    std::uint64_t host_block_writes       = 0;
    std::uint64_t host_image_restores     = 0;
    std::uint64_t host_block_restores     = 0;
    std::uint64_t prefetched_blocks       = 0;
    std::uint64_t host_write_bytes        = 0;
    std::uint64_t host_restore_bytes      = 0;
    std::uint64_t evicted_blocks          = 0;
    std::uint64_t host_snapshot_evictions = 0;
    std::uint64_t host_backed_evictions   = 0;
    std::uint64_t host_dead_reclaims      = 0;
    std::uint64_t unbacked_node_losses    = 0;
    std::uint32_t held_snapshots          = 0;
    std::uint64_t held_device_evictions   = 0;
    std::uint64_t held_snapshot_losses    = 0;
    std::uint64_t held_host_refusals      = 0;
    std::uint64_t tool_call_holds         = 0;
    std::uint32_t disk_slots              = 0;
    std::uint32_t disk_free_slots         = 0;
    std::uint64_t disk_image_writes       = 0;
    std::uint64_t disk_block_writes       = 0;
    std::uint64_t disk_write_bytes        = 0;
    std::uint64_t disk_fetches            = 0;
    std::uint64_t disk_image_reads        = 0;
    std::uint64_t disk_block_reads        = 0;
    std::uint64_t disk_read_bytes         = 0;
    std::uint64_t disk_failures           = 0;
    std::uint64_t disk_capture_skips      = 0;
    std::uint64_t disk_spill_skips        = 0;
    std::uint64_t disk_snapshot_evictions = 0;
    std::uint64_t disk_dead_reclaims      = 0;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_   = nullptr;
    std::uint32_t lane_  = 0;
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// One admission source: always a root start (no prefix cache).
struct SourceCandidate {
    std::uint32_t reused_tokens = 0;
    std::shared_ptr<const detail::QuoteImpl> hybrid;
    PrefixReusePath reuse_path = PrefixReusePath::Root;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// A paused request. This Program never pauses one (admission holds the whole extent), so a
// ResumeState only exists to complete the contract's types.
class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    explicit ResumeState(std::unique_ptr<detail::ResumeStateImpl>) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;

    friend class detail::ProgramImpl;
};

enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

// The sampled tokens of one model unit, awaiting the Engine's commit decisions. Its token span
// points into Program storage that stays valid until commit or abort_pending.
class PendingBatch {
public:
    PendingBatch() noexcept = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(std::exchange(other.tokens_, {})),
          row_counts_(std::exchange(other.row_counts_, {})),
          row_stride_(std::exchange(other.row_stride_, 1)),
          timing_(std::exchange(other.timing_, {})), row_failures_(other.row_failures_) {}
    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    // Licensed tokens per row, row b's at tokens()[b * row_stride()]; empty: every row licenses
    // one.
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }
    // Why the row cannot commit (None: it can).
    [[nodiscard]] runtime::PendingRowFailure row_failure(std::size_t row) const { return row_failures_.at(row); }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 1;
    runtime::ExecutionTiming timing_;
    std::array<runtime::PendingRowFailure, kMaximumConcurrency> row_failures_{};

    friend struct detail::ContractAccess;
};

class RequestBasePlan {
public:
    explicit RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept;
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();
    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

private:
    std::unique_ptr<detail::BasePlanImpl> impl_;

    friend class detail::ProgramImpl;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    // No frames are lent to a prompt's calls, so nothing returns between them.
    bool capacity_released = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<PromptReadout> readout;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;

    struct SpeculativeCounters {
        std::uint64_t rounds          = 0;
        std::uint64_t drafted_tokens  = 0;
        std::uint64_t accepted_tokens = 0;
        std::uint64_t fallback_steps  = 0;
    } speculative_counters;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::vector<ConstrainedDraw> constrained_draws;
    std::optional<ExpertCacheStats> expert_cache;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReplayProgress {
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    runtime::ExecutionTiming timing;
};

enum class ContextOperationKind : std::uint8_t { Bind, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced             = false;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

// Device memory a Program allocates besides its expert frames, from the options and the model's
// configuration alone, so startup checks it before the weights are read. The constructor
// allocates exactly these sizes.
struct ProgramDevicePlan {
    std::uint64_t kv        = 0; // KV pages (latent and indexer planes) and the block tables
    std::uint64_t workspace = 0; // the forward and sampling arena
    std::uint64_t staging   = 0; // the expert miss staging slots
    std::uint64_t state     = 0; // KDA convolution and recurrent state of every lane
    std::uint64_t io        = 0; // per-call inputs, logits, sampling, constraint and scoring arrays
    std::uint64_t draft     = 0; // the DFlash2 context ring, target features and KDA verify records
    std::uint64_t prefix    = 0; // the prefix cache's staging slots and saved head residuals
    std::uint64_t residency = 0; // the expert frame tables and route log
    std::uint64_t expert_record_stride = 0; // one expert record (staging slots and frames)
    std::uint32_t max_frames           = 0; // the most frames the expert cache uses
    std::uint32_t graph_bound          = 0; // CUDA graph executables the Program may instantiate

    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept {
        return kv + workspace + staging + state + io + draft + prefix + residency;
    }
};

struct ProgramOptions {
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 1;
    // Teacher-forced scoring (EnginePurpose::CausalScoring): the device plan adds the final
    // residuals of a max_context window and one tile of FP32 logits.
    bool causal_scoring         = false;
    std::uint32_t prefill_chunk = 4096;
    // Prompt tokens per call while another lane decodes; 0 or at least prefill_chunk: no cap.
    std::uint32_t prefill_chunk_beside_decode = 0;
    // A prefill chunk of at least 256 columns streams its non-resident experts to the device by
    // copy-engine DMA two layers ahead of compute, into frames the expert cache lends
    // (glm5_3-flash-design.md §11.2); off, the MoE stages them with its SM kernel as before. The
    // records a chunk copies are the same bytes either way, so the outputs do not change.
    bool prefill_stream = true;
    // Consecutive streamed calls of a prompt run layer-major (prefill_walk.cpp): each expert
    // crosses the link once per span instead of once per call, with the same bits.
    bool prefill_walk = true;
    // CPU-served misses of decode and verification calls: host expert-engine workers (0
    // disables), the most of a layer call's misses they take, and the link's share of a call's
    // host-memory misses in 256ths (the rest go to the CPU).
    std::uint32_t cpu_expert_workers = 8;
    std::uint32_t cpu_expert_jobs    = 4;
    std::int32_t cpu_link_share256   = 128;
    // Main KV capacity in tokens, shared by every lane.
    std::uint32_t kv_capacity_tokens = 0;
    // The MLA latent's storage: BF16 (1024 B a token and layer) or Int8Group64 (528 B; the
    // persistent INT8 group codec of mla_sparse_attention.h). The indexer planes stay BF16.
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    // FP32 logits (the default), or each sum rounded once to BF16 (EngineOptions::logits_dtype).
    bool bf16_logits = false;
    // Decode rounds replay a CUDA graph per batch size (off: every round runs eagerly).
    bool decode_graphs = true;
    // DFlash2 drafts per round, 1..15 (the bound companion drafter's; 0: no speculation). A
    // one-row round proposes this many, fewer only near its request's extent, and verifies them
    // all unless draft_min_p or auto_draft_length cuts them; a round of several rows or of a
    // token-constrained row decodes plainly.
    std::uint32_t draft_tokens = 0;
    // --draft-tokens auto (glm5_3-flash-design.md §10.2): a one-row round verifies the prefix of
    // its drafts that maximizes the expected tokens per round time, from each draft's selector
    // probability calibrated per request; none decodes the round plainly.
    bool auto_draft_length = false;
    // --spec-min-p: a one-row round verifies its drafts up to the first whose selector probability
    // (temperature 1) is below this value; 0 verifies every draft.
    double draft_min_p = 0.0;
    // Expert placement: the shared runtime's options (models/offload/expert_placement_runtime.h),
    // with the instance's placement, warm start, VRAM headroom and RAM tier sizing; the Program
    // fills the family's own fields (label, geometry, staging, chunk, tier floor, stripe identity).
    offload::ExpertPlacementRuntime::Options placement;
    // The prefix cache (glm5_3-flash-design.md §12): requests resume from and publish to a shared
    // block tree with state snapshots in a pinned Host slab pool of prefix_host_bytes (the family's
    // only snapshot residency; there is no Disk tier). GLM resumes only from prefill-computed state
    // on the chunk grid — its projection routes are chosen by call width, so a snapshot's frontier
    // is always a call boundary of the publishing run — plus the prompt-end captures of the
    // zero-token resume, whose state is the prefill transition itself.
    bool prefix_cache               = false;
    std::uint64_t prefix_host_bytes = 0;
    runtime::prefix_cache::TapPlannerConfig prefix_taps;
    runtime::prefix_cache::CacheCostModel prefix_cost;
    DiagnosticObserver diagnostics;
};

class Program {
public:
    Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options);
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    // The device plan of a Program with these options over a model with this configuration and
    // public token count; validates the options.
    [[nodiscard]] static ProgramDevicePlan plan_device(const ProgramOptions& options, const Config& config,
                                                       std::uint32_t public_tokens);
    [[nodiscard]] const ProgramDevicePlan& device_plan() const noexcept;

    // Expert-cache maintenance outside rounds: the engine calls maintain() from its worker,
    // between rounds and under its execution lock, when the waker asks.
    void set_maintenance_waker(std::function<void()> waker);
    void maintain();

    // The expert frames' startup sizing (the instance's VRAM summary reads it).
    [[nodiscard]] const offload::VramSizing& vram_sizing() const noexcept;
    // The expert placement (/admin/expert-placement): the shared placement runtime's status, the
    // file and counts forms of a switch (applied at the next maintenance boundary), and the
    // current placement written in the load format.
    using ExpertPlacementStatus = offload::ExpertPlacementStatus;
    [[nodiscard]] ExpertPlacementStatus expert_placement_status() const;
    [[nodiscard]] bool request_expert_placement(const offload::ExpertPlacementRequest& request,
                                                std::string* error = nullptr);
    [[nodiscard]] bool request_expert_count(std::uint32_t vram_experts, std::uint32_t ram_experts,
                                            std::string* error = nullptr);
    [[nodiscard]] bool export_expert_placement(const std::filesystem::path& path, std::string* error = nullptr);

    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    // ---- admission: sources over the prefix cache, start_binding reserves, poll_context activates
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base,
                                                              std::uint32_t maximum_frontier);
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                                   const SourceCandidate& source,
                                                   ResumeState* resume           = nullptr,
                                                   ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                                                   std::uint32_t resume_tokens   = 1);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    // Reclaims cached Device pages for a shortage (evicting unpinned cached blocks).
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage shortage);
    // The blocked FIFO head's Host-only blocks are copied into spare Device cache while it waits
    // (hybrid-prefix-cache-spec §6.6).
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    // Holds the snapshots queued requests would resume from, admission order first (§9.6).
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    void hybrid_hold_continuation(std::chrono::steady_clock::time_point) {
        throw std::logic_error("Glm5Next: its prefix cache has no tool-call holds");
    }

    [[nodiscard]] bool hybrid_holds_stale() const noexcept;
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;

    // ---- execution units: a binding reserves the request's whole extent ----
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }

    // ---- pause and replay: never needed (no unit grows); a pause is refused ----
    [[nodiscard]] bool start_pause(SequenceHandle sequence, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence, runtime::ExecutionTiming* = nullptr);

    // `masks` (null when no row is constrained) supplies the rows' grammar masks for this call.
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
                         std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;

    void fail_all_cleanup() noexcept;
    // After a runtime::RecoverableExecutionError from a call: releases every lane. False when
    // that fails (the Engine fails everything).
    [[nodiscard]] bool recover_after_failure() noexcept;
    // Releases every lane and saves the expert cache's state.
    void shutdown_cleanup() noexcept;

    // Teacher-forced scoring of a text prompt (ProgramOptions::causal_scoring; an idle Program).
    [[nodiscard]] ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                           const ScoreOptions& options);
    // Drafter-training capture (bench/models/glm5_next/drafter_capture.cpp): after this, every
    // causal_score also keeps each position's stream-mean residual after `layers` and its final
    // stream mean (the final norm's input), read with take_features. Allocates
    // (layers + 1) x hidden x max_context BF16 on the device.
    void enable_feature_capture(std::vector<std::uint32_t> layers);
    // The last causal_score's capture, position-major: features [n][layers x hidden] and final
    // [n][hidden], BF16 bits.
    void take_features(std::vector<std::uint16_t>& features, std::vector<std::uint16_t>& final);

    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept {}

private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};

// The common EngineCore is instantiated once for this model.
struct RuntimeTypes {
    using Frontend          = glm5_next::Frontend;
    using PreparedPrompt    = glm5_next::PreparedPrompt;
    using OutputSession     = glm5_next::OutputSession;
    using PublishedOutput   = glm5_next::PublishedOutput;
    using RequestBasePlan   = glm5_next::RequestBasePlan;
    using SequenceHandle    = glm5_next::SequenceHandle;
    using SourceCandidate   = glm5_next::SourceCandidate;
    using ResumeState       = glm5_next::ResumeState;
    using ExecutionUnit     = glm5_next::ExecutionUnit;
    using ExecutionUnitKind = glm5_next::ExecutionUnitKind;
    using ContextProgress   = glm5_next::ContextProgress;
    using PendingBatch      = glm5_next::PendingBatch;
    using PrefillProgress   = glm5_next::PrefillProgress;
    using ReplayProgress    = glm5_next::ReplayProgress;
    using CommitResult      = glm5_next::CommitResult;
    using DiscardResult     = glm5_next::DiscardResult;
    using FinishResult      = glm5_next::FinishResult;
    using AbortResult       = glm5_next::AbortResult;
    using Program           = glm5_next::Program;
};

} // namespace infernix::models::glm5_next
