#pragma once

#include "models/qwen4_exp/program/internal.h"

namespace infernix::models::qwen4_exp {
namespace detail {

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Prefill, Decode, Finishable };

    struct Lane {
        Phase phase          = Phase::Free;
        std::uint64_t epoch  = 0;
        runtime::BeginSummary begin; // the Begin of the binding
        std::vector<std::int32_t> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        // Where the prefill phase ends: the prompt, or a resumed lane's paused frontier, which its
        // replay reaches without sampling (design §15.7).
        std::uint32_t prefill_end = 0;
        bool replay               = false;
        std::uint32_t extent      = 0; // the positions the request may ever hold (prompt + output)
        // The prompt's planned prefill calls (design §15.3): exclusive ends, the last prompt_tokens.
        std::vector<std::uint32_t> calls;
        std::size_t next_call = 0;
        LaneRope rope;                    // RoPE positions of the lane's tokens (design §16.3)
        std::unique_ptr<LaneVision> vision; // the encode window and handoff of a media prompt
        double vision_seconds = 0;          // the tower's GPU time
        // The tower's timer when its stop had not completed at release: read later
        // (vision_seconds_of), never waited for.
        std::optional<CudaEventTimer> vision_timer;
        ops::SamplingConfig sampling;
        // /v1/decide: the tokens the prompt's readout names, the constraint, the draws of the pending
        // round's tokens (one per licensed token) and the committed trace they move into.
        std::vector<TokenId> readout;
        std::shared_ptr<const ConstraintPlan> constraint;
        std::shared_ptr<const PreparedPrompt> prompt; // a waiting sibling compares its media
        std::vector<ConstrainedDraw> constraint_round, constraint_trace;
        std::optional<DeviceKVPageReservation> reservation;
        std::vector<DeviceKVPageLease> pages;
        KVExecutionRowLease row;
        std::uint64_t prefill_ns      = 0;
        std::uint64_t decode_ns       = 0;
        std::uint64_t decode_share_ns = 0;
        offload::ExpertResidency::Stats cache_at_admission;
        std::uint64_t cpu_served_at_admission = 0;
        offload::HostExpertTier::Stats tier_at_admission{}; // SSD tier mode
        // The Program's verification-round counters (rounds, columns, padded columns) at
        // admission, for the request's share of the rounds' padding.
        std::array<std::uint64_t, 3> verify_at_admission{};
        NgramVolume::Counters ngram; // this request's n-gram row traffic
        GateTraffic gate;            // its verification rounds' rows read behind the n-gram gate
        std::unique_ptr<qwen_common::detail::NgramProposer> proposer; // copy proposals over history
        SpeculativeStats speculative;
        // MTP drafter (design 11.2; rule MR1 of 15.2): its KV cells [0, mtp_cells) are final. At
        // every settled point the lane's saved column holds the residual of position
        // state_tokens - 1, whose cell is written (mtp_cells == state_tokens) or pending
        // (state_tokens - 1: the next call or round writes it).
        std::uint32_t mtp_cells = 0;
        bool mtp_live           = false;
        // DFlash2: the context ring holds every position below this one (state_tokens at every
        // settled point, except after a resume from a tap inside a prefill layer walk).
        std::uint32_t dflash_frontier = 0;
        // Draft-length policy (design 11.4): conditional acceptance of each draft position.
        std::array<double, 8> mtp_accept{};
        std::uint64_t mtp_policy_rounds = 0;
        // DFlash2 (--draft-tokens auto): a draft's acceptance (given the earlier ones') by its
        // selector probability's decile.
        SelectorCalibration dflash_calibration;
        // Conditional acceptance of an n-gram copy
        // position by the proposal's match length (kNgramBuckets).
        std::array<double, 4> ngram_accept{};
        LanePrefix prefix;
        // Prompt seeding (design §9.2): of the prompt's prefilled tokens counted, how many
        // routed to each (layer, expert); [layers][experts], allocated on first use.
        std::vector<std::uint32_t> prompt_uses;
        std::uint32_t prompt_counted = 0;
    };

    // ---------------------------------------------------------------- construction (program_impl.cpp)
    // Everything the Program allocates on the device besides the expert frames; validates the
    // options.
    static DeviceLayout plan_device(const ProgramOptions& o, const Config& config, std::int32_t public_tokens);

    // The chunks of the KV backing that pages [0, pages) of every plane and the block tables need.
    static std::vector<std::uint8_t> kv_chunks(const DeviceLayout& d, std::uint32_t pages);
    static std::size_t kv_mapped_bytes(const DeviceLayout& d, std::uint32_t pages);

    // 16-bit-activation experts' wide route under these options, which sizes the workspace: the
    // expert-a4-residual fast path's (NVFP4 W4A16 banks) or ggml-bf16-prefill's (GGUF banks' BF16
    // planes, a superset of the exact layout).
    static ops::offloaded_moe::WideA16 wide_a16(const ProgramOptions& o) noexcept;

    // The workspace arena of a Program with `lanes` lanes and the plan's widths: the forward at its
    // widest call (a prefill chunk or every lane's widest round), sampling, acceptance and the
    // drafter. The plan sizes the arena with the Program's lane count; the per-lane report compares
    // lane counts.
    static std::size_t workspace_bytes(const TextConfig& c, const ProgramOptions& o, const DeviceLayout& d,
                                       std::int32_t lanes, std::int32_t columns);

    // Whether the sparse acceptance Op (sampled drafts' verifier) serves this token domain and these
    // verify widths and row counts.
    static bool sparse_acceptance(std::int32_t token_domain, std::int32_t W, std::int32_t lanes);

    static void allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally);

    // The startup failure for device allocations the driver placed in system memory.
    static std::string spill_message(std::uint64_t bytes);

    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options);

    const DeviceLayout& device_layout() const noexcept { return plan_; }
    const VramSizing& vram_sizing() const noexcept { return sizing_; }

    // ---------------------------------------------------------------- admission (binding.cpp)
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);

    bool feasible(const RequestBasePlan& base) const noexcept;

    // The positions one round may write past the lane's frontier: a verification's columns and the
    // MTP drafter's cells (design §15.7).
    std::uint32_t round_positions() const noexcept {
        return static_cast<std::uint32_t>(std::max(max_width_, mtp_ ? mtp_k_ + 1 : 1)) + 1U;
    }

    // The KV pages a binding of `base` whose prefill ends at `end` reserves: through one round past
    // it, within the request's extent.
    std::uint32_t binding_pages(const BasePlanImpl& base, std::uint32_t end) const noexcept;

    static PrefixReusePath reuse_path_for(prefix::cache_index::SnapshotKind kind) noexcept;

    // A source over `selection` (absent: a root start without a prefix cache).
    SourceCandidate make_source(const BasePlanImpl& base, const qwen_common::PreparedPromptData& data,
                                std::optional<PrefixSelection> selection) const;

    // The prefix cache's choice for the request (without a lane: a lane-resident snapshot is credited
    // at start_binding), then a root start.
    std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier);

    // Stages the binding of `source` on `lane`: re-selects against the cache as it is now (crediting
    // a lane-resident snapshot), reserves the prompt plus one round and maps the cached blocks. The Begin
    // the Engine published names the source's frontier, so a changed choice is refused and the Engine
    // falls back to the next source (the root).
    runtime::ResourceReservation start_binding(const RequestBasePlan& base, runtime::LaneId destination,
                                               const SourceCandidate& source, ResumeState* resume);

    // ---------------------------------------------------------------- pause and replay (§15.7; binding.cpp)

    // The lane's request-level state a pause keeps (everything but the model state and the pages).
    void save_request(const Lane& lane, ResumeStateImpl& r) const;

    void restore_request(Lane& lane, const ResumeStateImpl& r);

    // The resumed request's ledger as a prompt: its tokens through the next input, their RoPE
    // positions, and its lookup keys (the prompt's, chained over the output blocks as
    // prefix_publish chains them); no tap hints (a replay publishes blocks, not taps).
    qwen_common::PreparedPromptData ledger_prompt(const qwen_common::PreparedPromptData& prompt,
                                              const ResumeStateImpl& r) const;

    // Binds a request paused after its first token: the deepest cached state of its ledger up to
    // the paused frontier (the pause's own capture when it is still cached), then a replay of the
    // ledger to that frontier, without sampling. The pages cover the replay and one round.
    runtime::ResourceReservation bind_replay(const BasePlanImpl& b, std::uint32_t index, ResumeStateImpl& r);

    // Penalty counts over the request's output so far (the sampler counts output tokens only),
    // counted on the host: a binding may overlap a round that still uses the workspace.
    void rebuild_token_counts(const Lane& lane);

    // Pauses a resident at its committed boundary (design §15.7): the prefix cache receives the
    // lane's blocks and its state at the frontier, exactly as a consistent abort leaves them, the
    // lane is released and a Pause transaction hands the Engine the ResumeState.
    bool start_pause(SequenceHandle sequence);

    // One prefill call of a resumed lane's ledger toward its paused frontier, without sampling.
    ReplayProgress advance_replay(SequenceHandle sequence);

    // Grows each unit's lane to its next round (design §15.7): a decode or control unit needs the
    // positions it writes, a verification's columns and the drafter's cells, within the request's
    // extent. A shortage cached blocks and the elastic pool cannot cover is reported, and the Engine
    // reclaims or pauses the youngest resident.
    runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);

    // Advances the binding: a media prompt's encode window one step per call (design §16.4; other
    // lanes run between steps), then the lane is published.
    ContextProgress poll_context(runtime::CancellationFlagView cancellation);

    [[nodiscard]] bool in_transaction() const noexcept { return transaction_lane_.has_value() || pause_.has_value(); }

    // Whether the open binding holds `sequence` back (its lane is still being published).
    // A lane's context is blocked while it is being bound, and a prefilling lane while another
    // lane's layer walk holds the expert stream and the lent workspace (the walk's layers are at
    // different positions until its span ends, so no other prefill call may run in between).
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;

    // The Engine's reclaim for a binding shortage: evict cached Device blocks until it is covered,
    // then grow the elastic pool, taking back a prefill's loans first if only they block growth.
    bool hybrid_reclaim(runtime::ContextResourceUsage shortage);

    // Whether every lent frame is held by a lease a round boundary can return (the prefill stream
    // ring and the wide arena; not a layer walk's area or a vision window), so growth needs only
    // their return (hybrid_reclaim).
    [[nodiscard]] bool leases_returnable() const noexcept;

    // ---------------------------------------------------------------- execution (decode.cpp)
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks);

    // While another lane decodes, a planned call wider than the beside-decode chunk runs as
    // several: the engine runs one decode round per prefill call (a mixed round splits the same
    // way, mixable_width), so the decoding lanes keep streaming between them. The split lands on the GDN grid (the exact route's bits do not
    // depend on it). Exact taps already end calls; a flexible tap is realized at the first
    // boundary at or past it, which a split can bring closer to its position.
    void split_beside_decode(Lane& lane, std::uint32_t index);

    // ---- mixed rounds: a prompt call in a decode round's Forward ----
    // The width of the prefilling lane's next call when it rides in a decode round, or 0 when it
    // cannot. A mixed round has room for the call's columns beside every lane's widest round
    // (columns >= chunk), so the call is split on the GDN grid as split_beside_decode splits it.
    // The prompt's last call stays alone: it samples the first token, which a decode round's
    // pending batch does not carry. A layer walk's span, a replay, an image prompt and the MTP
    // drafter's chunk cells keep their own calls.
    std::int32_t mixable_width(Lane& lane, std::uint32_t index);

    bool can_mix(SequenceHandle prefill);

    MixedProgress mixed(SequenceHandle prefill, std::span<const SequenceHandle> sequences,
                        std::span<const runtime::RoundBudget> budgets, runtime::TokenMaskProvider* masks);

    // The mixed round's prompt call after the round's `rows` rows of `decode_columns` columns: its
    // tokens, positions and n-gram rows at the following columns, its RoPE positions as their own
    // [width, 3] block after the rows' and its block start, slot and table row after theirs.
    void stage_mixed_prefill(std::int32_t rows, std::int32_t decode_columns);

    // The mixed round's Forward (eager: the prompt call's width needs the wide arena, which no graph
    // may capture): the decode or verification rows as their round runs them, the prompt call after
    // them, then the call's DFlash2 window into its lane's ring. The caller staged and uploaded the
    // io (io_prefix of every column) and ran before_round.
    void run_mixed(std::int32_t batch, std::int32_t width, const execution::ForwardVerify* verify);

    // Prompt seeding (design §9.2): a prefill call's per-token routes, read from the route log
    // its after_round is about to apply, add to the lane's prompt uses. A layer walk's span counts
    // its last chunk, the one its route log holds; a mixed round's call starts at column `first`.
    void note_prompt_routes(Lane& lane, std::int32_t width, std::int32_t first = 0);

    // At the prompt's end, after its last after_round: an expert that every counted token routed to
    // gains kPromptSeedCredit of LFRU count, the others in proportion.
    void seed_prompt_experts(Lane& lane);
    static void clear_prompt_uses(Lane& lane);

    // The progress of a prefill unit whose calls were enqueued (the last ends at `begin + width`,
    // `tokens` prompt tokens in all): for the prompt's last call the first token is sampled and the
    // pending batch returned. Non-last units have already synchronized and applied their routes.
    // The zero-token resume's prefill step (an exact repeat of a prompt whose own end snapshot is
    // cached): no prefill call runs — the snapshot holds the lane's whole state — and the first
    // token regenerates from the snapshot's saved head residual through head_logits, the same head
    // function at the same logit width the cold run's finish used, so its logits are byte-equal and
    // the draw (keyed by the same logical position, n - 1) reproduces the cold first token
    // bit-for-bit, greedy and seeded. The head runs after the restore's event: its column is the
    // snapshot's. Nothing else of a prefill call runs — no stream lease, no wide work, no route log
    // (seed_prompt_experts and note_prompt_routes are performance only; the cold run's published
    // blocks already serve the cache).
    PrefillProgress resume_from_saved_head(Lane& lane, std::uint32_t index, Clock::time_point start);

    PrefillProgress prefill_progress(Lane& lane, std::uint32_t index, std::int32_t begin, std::int32_t width,
                                     std::uint32_t tokens, bool last, std::size_t promotions,
                                     Clock::time_point start);

    // The Begin the Engine admitted the request with, fixed at binding (a snapshot evicted since
    // must not change it).
    const runtime::BeginSummary& prefill_summary(const Lane& lane) const noexcept { return lane.begin; }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::TokenMaskProvider* masks);

    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride);

    // The request's cumulative speculative work, which the Engine differences per commit into its
    // live counters. The round's drafts and acceptance were counted when it was submitted. Without
    // speculation (max_width_ == 1) every plain round still counts a fallback step in the request's
    // stats; that is no speculative work, so the live counters stay at zero.
    CommitRowResult::SpeculativeCounters speculative_counters(const SpeculativeStats& stats) const noexcept;

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions);

    DiscardResult discard(PendingBatch&& pending) noexcept;

    FinishResult finish(SequenceHandle sequence) noexcept;

    AbortResult abort(SequenceHandle sequence) noexcept;

    // After a RecoverableExecutionError: every lane is released without publishing anything (the
    // failed round's state is unknown), the cache's transfers in flight land, and a cache the
    // failure may have published into is emptied. False when that fails (the device itself failed;
    // the Engine then fails everything).
    bool recover_after_failure() noexcept;

    void release_all() noexcept;

    PhysicalUsageSnapshot usage() const noexcept;

    MemorySummary memory() const noexcept;

    std::uint64_t revision() const noexcept { return revision_; }

private:
    // Promotions per layer call (design section 9.2): one per decode round measured fastest; a
    // promotion moves as many PCIe bytes as serving the expert once zero-copy.
    static constexpr std::size_t kDecodePromotionsPerLayer  = 1;
    // Once the frames are full, decode promotes only every kDecodePromotionInterval-th round on the
    // reference link (promotion_interval scales it to the measured one): a promotion moves as many
    // PCIe bytes as serving its expert once, and fewer of them measured faster at 512 tokens (design
    // section 9.2). Until then every round promotes, so a cold cache fills at the full rate.
    static constexpr std::uint64_t kDecodePromotionInterval = 4;
    // Misses staged per pass of a layer's experts: every decode and verify call's misses in one
    // pass; a prefill chunk's in several. The least, and with the SSD tier the count (plan_device);
    // the fork route takes at most 512 (ops::moe_experts).
    static constexpr std::int32_t kStagingSlots = 64, kMaxStagingSlots = 512;
    // The SSD tier's smallest RAM (design §9.8: 1,024 slots, 2.6 GiB).
    static constexpr std::uint32_t kMinTierSlots = 1024;
    // RAM slots a layer walk's span borrows for its stream reads (at most 1/8 of the SSD tier's): with
    // reads kReadAhead layers ahead, each layer streams up to 1,280 / 4 = 320 SSD-only experts. The
    // count that streams a 32K prefill's SSD-only experts whole — lent per span, so it costs decode
    // nothing — and the smaller counts' costs are measured in design §9.5.
    static constexpr std::uint32_t kStreamReadSlots = 1280;
    // Unpinned records a layer may stream through the bounce, per part: the layer's unpinned share
    // of its experts, within these bounds (the bounce comes out of the pinned share; the measured
    // tok/s at each bound are design §9.5).
    static constexpr std::uint32_t kStreamBounceMin = 64, kStreamBounceMax = 256;
    // Promotions follow the tokens a round advances (the longest row's), not rounds: a verification
    // round that advances four tokens promotes what four decode rounds would, so a speculative
    // round's cache warms per token as plain decode does.
    std::size_t decode_budget(std::uint32_t tokens);
    // The interval at the measured link (design §9.2): a promotion's bytes cost less wall time
    // on a faster link, whose rounds leave it more idle, so it promotes proportionally more often
    // (x8 4, x16 2; the x16 value is derived, not measured).
    std::uint64_t promotion_interval() const noexcept;
    std::uint64_t budget_tokens_ = 0;

    // A decode or verification round's cache update (routes, LFRU, promotions) runs on the host
    // after its commit has enqueued its GPU work (the GDN fold, tail commits, the drafter's
    // catch-up), so the policy overlaps that work instead of idling the GPU after the round. A
    // DFlash2 verification's update waits longer: the next round's proposal settles it while it
    // drafts (dflash2_propose's while_drafting), the proposal being the next GPU work. Every other
    // path settles it first: each round's launch before its residency's before_round, the
    // assignment of the next round's update, and the Program's other entry points (program.cpp).
    // The update reads the round's routes, which no proposal or commit overwrites, its live
    // columns (live_, rewritten only by the next verification's acceptance) and the width and
    // lanes it was submitted with. Outputs never depend on it: expert placement is bitwise
    // invariant.
    struct DeferredRound {
        std::int32_t columns  = 0;
        std::uint32_t tokens  = 0;
        bool live             = false; // only the verification's accepted columns count
        bool pending          = false;
        std::int32_t prefill_columns = 0; // a mixed round's prompt call, after the rows' columns
        std::int32_t width    = 1;     // the round's columns per row
        std::array<std::int32_t, kMaximumConcurrency> lanes{}; // its rows' lanes
        // A DFlash2 proposal drafted the round (MTP off): its commit leaves the update to the next proposal.
        bool to_proposal      = false;
    };
    DeferredRound deferred_;

    // The round's update, registered after any earlier one has settled. The mixed round's prompt
    // columns are added to it afterward (mixed's prefill_columns).
    void defer_round(std::int32_t columns, std::uint32_t tokens, bool live, std::int32_t width,
                     std::span<const std::uint32_t> lanes, bool to_proposal = false);
    // The mixed round in flight (mixed): its prompt call's lane and columns.
    struct MixedPrefill {
        bool active        = false;
        std::uint32_t lane = 0;
        std::int32_t slot  = 0; // the lane, as the Forward's host slot
        std::int32_t begin = 0, width = 0;
        bool streamed      = false;
    };
    MixedPrefill mixed_;

    void settle_round();

public:
    // The Program's entry points that can touch the expert cache, launch a forward or report on it settle a deferred
    // update first (program.cpp; DeferredRound). Planning and accessors do not.
    void settle_deferred();
    // The same for an entry point that may not throw: an update that fails is dropped, as a failed round's is.
    void settle_deferred_or_drop() noexcept;

private:
    // ---------------------------------------------------------------- the placement runtime's hooks (placement.cpp)
    // The shared placement runtime (models/offload/expert_placement_runtime.h) owns the VRAM
    // monitor, the warm start and the saved state, the RAM tier's lifecycle and fills, and the
    // placement switches; the hooks carry what stays with the family.

    // The calibrated order (qwen4_exp/calibrated_order.h) ranks Qwen3.8-Flash-Next's experts: it
    // applies to every model of that shape (NInfer's guard), whatever the artifact's formats.
    [[nodiscard]] bool has_calibrated_order() const noexcept;

    // The monitor's start hands the sizing baseline over: the spare-demand inputs grow_kv reads
    // (design §9.7), moved by the elastic KV pool's mappings afterwards.
    void capture_sizing_inputs(const VramDemand& demand, const VramSnapshot& snapshot);

    // The runtime's tier hook: the CPU miss service reads its records through the tier, and a
    // rebuilt tier retargets the prefill expert stream's callbacks (they captured the old one).
    void on_tier_changed(offload::HostExpertTier* tier, bool rebuilt);

public:
    void set_maintenance_waker(std::function<void()> waker) {
        placement_runtime_->set_maintenance_waker(std::move(waker));
    }

public:
    // After every synchronization of a round: a call that could not serve an expert (design §9.3:
    // an unreadable record, a host service that stopped answering), an n-gram gate whose rows were
    // never published or a token id outside the vocabulary left undefined outputs, so the round's
    // requests fail and the engine keeps serving (no device trap: the context survives). Prefix-cache
    // entries the round may have published are dropped when the lanes are released.
    void check_device_error();

public:
    // Writes the saved state (at stop, and from maintain_expert_state). Never throws.
    void save_expert_state() noexcept { placement_runtime_->save_expert_state(); }
    // Between requests: saves when rounds ran since the last save and 10 minutes have passed.
    void maintain_expert_state() noexcept { placement_runtime_->maintain_expert_state(); }
    // The prefix cache's Disk tier work outside requests (Program::background_poll).
    [[nodiscard]] bool prefix_background_pending() const noexcept { return prefix_ && prefix_->background_pending(); }
    void prefix_background_poll() {
        if (prefix_) { prefix_->poll(); }
    }

    // At the end of a run: reports the expert cache's routed uses and hits per layer (the whole
    // run, not one request) as one Engine diagnostic, beside report_cache's per-request line.
    void report_cache_layers() noexcept;

public:
    // Prefill calls up to this width are CPU-assisted (design §10.6); wider chunks route nearly
    // every expert, and their misses stream or stage on the GPU.
    static constexpr int kAssistMaxColumns = 255;

    // ---- expert placement (/admin/expert-placement; design §9.6) ----
    // The shared placement runtime owns the switches, the RAM tier's lifecycle and the fills
    // (models/offload/expert_placement_runtime.h); the family keeps the elastic KV interplay
    // (apply_vram_target's tail) and the stream's bindings.
    [[nodiscard]] bool request_placement(const offload::ExpertPlacementRequest& request, std::string* error) {
        return placement_runtime_->request_placement(request, error);
    }

    // The counts form (/admin/expert-placement {"vram": N, "ram": M}): the current ranking's top N
    // hold VRAM frames and the next M the RAM tier's slots -- adaptive within the counts (the
    // per-round policy swaps hot experts in), never pinned, so a workload shift moves the content
    // while the counts stay what the client named. M 0 with N the whole cache releases the tier.
    [[nodiscard]] bool request_count_placement(std::uint32_t vram_experts, std::uint32_t ram_experts,
                                               std::string* error) {
        return placement_runtime_->request_count_placement(vram_experts, ram_experts, error);
    }

    [[nodiscard]] offload::ExpertPlacementStatus placement_status() const {
        return placement_runtime_->placement_status();
    }

    [[nodiscard]] bool export_placement(const std::filesystem::path& path, std::string* error) {
        return placement_runtime_->export_placement(path, error);
    }

    // The maintenance waker from the engine thread: request_maintenance() only flags and notifies,
    // so a tick that is not done can re-arm it and the fills also advance while the engine
    // idles, when no round would call apply_vram_target.
    void wake_maintenance() { placement_runtime_->wake_maintenance(); }

    // A placement drifts from its named counts between switches (the runtime's
    // placement_drifted): settled when the engine idles.
    [[nodiscard]] bool placement_drifted() const { return placement_runtime_->placement_drifted(); }

    // The prefill expert stream's bindings for the tier as it is now: its records, stream slots and
    // bounce buffers, or the artifact banks' records without a tier. The callbacks capture the tier
    // object, so a tier rebuild (the runtime's ensure_tier) retargets the stream with a fresh set;
    // the banks' record strides come from host_strides_.
    struct StreamBindings {
        execution::ExpertStream::RecordOf record_of;
        execution::ExpertStream::Hold hold;
        execution::ExpertStream::SsdReads ssd;
        execution::ExpertStream::Staging staging;
        bool one_allocation = true;
    };
    [[nodiscard]] StreamBindings stream_bindings() const;

    // At a boundary (no round's kernels in flight; `idle`: no request active): applies a pending
    // expert-placement switch, advances the placement grow's refill and the RAM tier's fill,
    // settles the placement when idle, then resizes the expert cache to the control law's
    // decision (the runtime's apply_placement returns whether the elastic KV control may run).
    void apply_vram_target(bool idle);

private:
    // Calls this narrow are CPU-served (design §10.1).
    static constexpr std::int32_t kServedCallColumns = 8;
    // Prefill calls up to this many columns run in the static workspace; wider ones in the wide arena.
    static constexpr std::int32_t kStaticWorkColumns = 512;

    // ---------------------------------------------------------------- reports and diagnostics (reports.cpp)
    // Appends the round whose routes after_round is about to apply to the internal route trace
    // (ProgramOptions::route_trace); rows' lanes are `lanes`, else host_lanes_ as staged for the round.
    void trace_round(offload::RouteTraceKind kind, std::int32_t rows, std::int32_t width, std::uint32_t tokens,
                     std::size_t budget, std::uint32_t position, std::span<const std::uint8_t> live = {},
                     std::span<const std::int32_t> lanes = {});

    // Reports the expert cache over the finished request (and since start) as an Engine diagnostic,
    // or on stderr without an observer.
    void report_cache(const Lane& lane) noexcept;

    // The workspace arena of a Program with `lanes` lanes: the forward at its widest call (a prefill
    // chunk or every lane's widest round), sampling, acceptance and the drafter.
    std::size_t workspace_capacity(std::int32_t lanes) const {
        return workspace_bytes(c_, options_, plan_, lanes, plan_.static_columns);
    }

    // Reports at startup what each lane of --max-concurrency takes from the expert cache, in frames
    // (design 9.7): its KV extent unless --kv-capacity fixes the pool; its recurrent, convolution
    // and QSA-tail state; its verification and drafter records (DFlash2: its context ring and round
    // feature columns); its logit columns and penalty counts; and its share of the workspace. Every
    // other allocation is independent of the lane count, so one more lane costs these frames.
    void report_lane_cost(std::int32_t lanes, std::uint64_t record_stride) noexcept;

    // Reports the finished request's n-gram row traffic (design 12.3). Every row is read on the host
    // before the call that consumes it is launched.
    void report_ngram(const Lane& lane) noexcept;

    // The verification rounds' padding over the finished request, and since start: a padded
    // column is one past its row's drafts, carried through every layer's MoE. The share is what
    // a column mask (dispatching nothing for such columns) could keep off the expert path.
    void report_verify_rounds(const Lane& lane) noexcept;

    std::uint32_t* gate_ready() const { return static_cast<std::uint32_t*>(gate_host_.data()); }

    // The n-gram gate of a decode or verification graph: the pinned rows behind the round's word.
    execution::NgramRowGate row_gate() const;

    std::uint32_t lane_index(const Lane& lane) const noexcept {
        return static_cast<std::uint32_t>(&lane - lanes_.data());
    }

    // Publishes a gated round's word if reading its rows fails: the rows are zeroed (the round's
    // results are discarded with the error), so the enqueued gate passes and the stream drains.
    struct GateRelease {
        ProgramImpl* owner = nullptr;
        std::uint32_t word = 0;
        std::size_t bytes  = 0;
        GateRelease(ProgramImpl* o, std::uint32_t w, std::size_t b) : owner(o), word(w), bytes(b) {}
        GateRelease(const GateRelease&)            = delete;
        GateRelease& operator=(const GateRelease&) = delete;
        ~GateRelease() {
            if (owner == nullptr) { return; }
            std::memset(owner->host_io() + owner->io_layout_.ngram, 0, bytes);
            publish_pinned_word(owner->gate_ready(), word);
        }
    };

    // A diagnostic (Info unless stated) for the Engine's observer, or stderr without one.
    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const noexcept;

    // A --debug trace line of `category`; nothing is formatted while the category is off.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    void trace(DebugCategory category, const char* format, ...) const noexcept;

    // ---------------------------------------------------------------- execution helpers (exec.cpp)
    // The io bytes a decode or verification round of `columns` columns reads: everything before
    // the n-gram rows (ids, positions, slots, table rows, logit columns) and its columns' rows.
    // Prefill and forced-token calls copy the whole layout (their MTP cells follow the rows).
    std::size_t io_prefix(std::int32_t columns) const {
        return io_layout_.ngram + static_cast<std::size_t>(columns) * hash_.heads() * ngram_table().row_bytes;
    }

    SequenceHandle handle(std::uint32_t lane) const noexcept {
        return ContractAccess::make_sequence(this, lane, lanes_[lane].epoch);
    }

    std::uint32_t lane_of(const SequenceHandle& sequence) const;

    static double vision_seconds_of(const Lane& lane);

    GenerationTimings timings(const Lane& lane) const;

    void release(std::uint32_t index) noexcept;

    // Whether the elastic pool could grow to make `need` pages available (the quote's check).
    bool kv_growable(std::uint32_t need) const noexcept {
        return static_cast<std::uint64_t>(pool_->available_pages()) + kv_grow_pages() >= need;
    }

    // The pages one growth could add at most: grow_kv takes up to every frame, and none while frames
    // are lent, unless only leases hybrid_reclaim can return hold them (an estimate; grow_kv
    // decides).
    std::uint32_t kv_grow_pages() const noexcept;

    // Zeroes a slot's recurrent state, PLE convolution history, QSA tails (and exact-window tags) and
    // penalty counts.
    void reset_slot(std::uint32_t slot);

    std::byte* host_io() const { return static_cast<std::byte*>(io_host_.data()); }

    // `count` I32 words of the pinned io at byte offset `offset`.
    std::span<std::int32_t> io_words(std::size_t offset, std::size_t count) const {
        return {reinterpret_cast<std::int32_t*>(host_io() + offset), count};
    }

    // RoPE positions of `count` consecutive tokens of `lane` from `first`, as columns column ..
    // column + count - 1 of a call of `columns` columns over `batch` sequences, and the pooled-block
    // start of the lane's sequence b.
    void stage_call_rope(const Lane& lane, std::uint32_t first, std::int32_t count, std::int32_t columns,
                         std::int32_t column, std::int32_t batch, std::int32_t b) const;

    // The n-gram rows of `count` positions starting at `begin`, from the tokens before them; the
    // traffic counts toward `lane`'s request.
    void stage_ngram(Lane& lane, const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count,
                     std::size_t column);

    // window_ = the tokens of positions [begin - (n - 1), begin + count), EOS before the sequence.
    void ngram_window(const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count);

    // Verification row b's token at position q: its history up to the anchor (position p), then its
    // drafts, padded past their extent with the last draft (the anchor without drafts).
    std::int32_t verify_token(const Lane& lane, std::size_t b, std::int32_t p, std::int32_t q) const;

    // Queues the n-gram rows of verification row b's columns [from, to) (W columns from position p)
    // for read_queued_ngram.
    void queue_verify_ngram(Lane& lane, std::size_t b, std::int32_t p, std::int32_t W, std::int32_t from, std::int32_t to);

    // Queues the rows of the `count` positions whose tokens (with their context) are in window_ for io
    // columns from `column`. read_queued_ngram reads every queued row in one volume call, so the
    // misses of all a round's rows are in flight together: read row by row, their NVMe latencies
    // add up behind the gate (about 0.6 ms of a six-lane decode round).
    void queue_ngram(Lane& lane, std::int32_t count, std::size_t column);

    // Reads the queued rows into their io columns. Each segment's lane counts its rows and hits, and
    // the call's block reads and read time in proportion to its misses.
    void read_queued_ngram();

    // Reads the rows of the `count` positions whose tokens (with their context) are in window_ into
    // io columns from `column`.
    void read_ngram(Lane& lane, std::int32_t count, std::size_t column) {
        read_ngram_to(lane, count, host_io() + io_layout_.ngram + column * hash_.heads() * ngram_table().row_bytes);
    }

    // As read_ngram, into `out` (count * heads rows).
    void read_ngram_to(Lane& lane, std::int32_t count, std::byte* out);

    // `rows`: also read the n-gram rows (a walk chunk whose rows were prefetched skips them).
    void stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width,
                        const std::vector<std::int32_t>& history, bool rows = true);

    void stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions);

    // The n-gram rows of a plain decode round (column b: lane b's input token).
    void stage_decode_rows(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions);

    // Whether every n-gram row of a plain decode round is in the host row cache.
    bool decode_rows_cached(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions);

    std::uint32_t next_gate_word();

    // Right after a gated launch: submits it, reads the rows (`read`, into the io's first `bytes` of
    // n-gram rows), releases the gate and counts the reads toward `lane` (the round's first row).
    // Until the word is published the GPU waits on this thread, so nothing may be queued before it:
    // a call that waits for the device deadlocks the round. A kernel's first launch is one, as the
    // driver loads its code then and, with the process past its OS budget, waits for the device
    // to do so (the sampler and acceptance kernels queued here before the port of upstream
    // c2afefb2: the gate trapped after kPinnedGateTimeoutNs in 5 of 7 --vram-past-budget starts;
    // since the port of upstream 3b34b6e7 a gate past its bound fails the round with
    // kErrorRowGateTimeout instead).
    void read_behind_gate(Lane& lane, std::uint32_t word, std::size_t bytes, const std::function<void()>& read);

    // Runs one eager Forward call over the staged inputs (prefill chunks, forced tokens); logits of
    // `logit_columns` columns land in logits32_.
    void run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
             const execution::MtpChunk* chunk = nullptr, const execution::VisionInput* vision = nullptr,
             bool streamed = false, const execution::NgramRowGate* gate = nullptr);

    // run() for a streamed prefill call staged without its n-gram rows: a helper thread reads them
    // into the io layout and publishes the gate's word while this thread enqueues the call, so the
    // GPU embeds and runs the layers before ple.layer (whose experts stream) during the read, where
    // it otherwise waited for the whole read before the call's first kernel. The helper owns the
    // row scratch (window_, row_ids_), the n-gram volume and the lane's n-gram counters until it is
    // joined; this thread touches none of them meanwhile, and its own host waits (a split call's
    // routing, the stream's bounce) cannot block the helper's publish. A failed read publishes zero
    // rows (the call's results are discarded with its error). The same rows and the same arithmetic:
    // no bit changes.
    void run_rows_behind(Lane& lane, std::int32_t begin, std::int32_t width, const execution::MtpChunk* chunk,
                         const execution::VisionInput* vision);

    // ---- prefill expert streaming (design §13.3) ----
    // Chunks of at least kStreamMinColumns route nearly every expert of a layer, so all of a layer's
    // non-resident experts are copied ahead; narrower chunks keep the MoE's staging. The ring holds
    // two halves of as many records as the layer with the most non-resident experts has when it is
    // lent (+ one frame for the slot tables), so no streamed layer leaves misses to staging; it is
    // lent by the expert cache for the prompt and returned when no lane is prefilling. Halves of
    // 384 records (a cold cache misses all 512 of a layer) were 9 % slower at pp4096 (design
    // §13.3, ring size).
    static constexpr std::int32_t kStreamMinColumns = 256;
    static constexpr std::uint32_t kStreamHalfMin   = 16;

    // The call planner's costs: the fitted call costs, and the walk's spans as walk_span_end forms them.
    [[nodiscard]] prefix::CallCost call_cost() const;

    // Whether this chunk streams; lends the ring first when none is held. With `walk_bytes`, the
    // lease also holds a layer walk's area after the ring (stream_walk_bytes_; zero when the
    // lendable frames do not cover it): the experts its frames evict widen each half by their
    // per-layer share.
    bool stream_chunk(std::int32_t width, std::size_t walk_bytes = 0);

    // The ring (the stream lease) and the walk's area (its own lease).
    DeviceSpan ring_span() const noexcept { return DeviceSpan{stream_lease_.memory.data, stream_ring_bytes_}; }
    std::byte* walk_area() const noexcept { return static_cast<std::byte*>(stream_walk_lease_.memory.data); }

    // ---- the wide workspace (design §13.1, VRAM item 1) ----
    // A call wider than the static arena's columns runs in the wide arena: the stream lease's part
    // when it has one, else frames lent for it alone, else (no lendable frames: the VRAM monitor
    // shrank the cache, or there are no frames) a temporary device allocation, with a warning. The
    // Forward keeps its arena reference; the arenas' contents are swapped for the call.
    class WideWork {
    public:
        WideWork(ProgramImpl& program, std::int32_t columns) : program_(program) {
            if (columns > program.plan_.static_columns && program.wide_work_ > 0) {
                program.enter_wide();
                active_ = true;
            }
        }
        ~WideWork() {
            if (active_) { program_.leave_wide(); }
        }
        WideWork(const WideWork&)            = delete;
        WideWork& operator=(const WideWork&) = delete;

    private:
        ProgramImpl& program_;
        bool active_ = false;
    };

    // The expert cache's VRAM demand: what the fixed allocations, the graph executables' reserve and
    // the display headroom leave becomes frames of `frame_bytes`.
    VramDemand expert_frame_demand(std::uint64_t frame_bytes) const;

    // A prefill chunk whose wide workspace the expert cache could never lend (more than the three
    // quarters of its frames serving may give up) would allocate it for every long prompt, which
    // fails on a full card after admission (a 65,536-token chunk on the SSD tier wanted 27 GiB).
    // Prefill then runs at the widest halving of the chunk that can be lent (or the static arena's
    // width): its exact route is chunk-invariant, so only speed changes. The buffers planned for the
    // wider chunk stay as they are. `lendable` counts frames of `stride` bytes the cache can lend (a
    // cache of several frame pools lends from one: offload::ExpertResidency::lend_capacity_bytes).
    // A resident placement lends no frames: the wide workspace comes from free device memory, so
    // the chunk halves until it fits what startup left (the output is the same).
    void fit_wide_work_to_free(std::uint64_t free_bytes);

    void fit_wide_work_to_cache(std::uint64_t lendable, std::uint64_t stride);

    void enter_wide();

    void leave_wide() noexcept;
    bool wide_active_ = false; // *work_ is the wide arena (enter_wide .. leave_wide)

    // After a prompt's last wide call is enqueued: the wide arena's own lease or allocation goes back
    // (the stream lease's part goes back with it).
    void return_wide_work();

    // Returns the ring (after the last streamed chunk is enqueued: compute waited for every copy).
    void return_stream_lease();

    // Whether a lane other than `index` is prefilling.
    [[nodiscard]] bool prefilling_besides(std::uint32_t index) const noexcept;

    // One decode round of `batch` sequences: its inputs are staged at fixed device addresses and
    // every row is chosen on the device, so one CUDA graph per batch size serves every round. The
    // first round of a size runs eagerly (it also performs the Ops' one-time setup); the next is
    // captured and later rounds replay it.
    // `position`: the first sequence's (a one-column round's context, for the plain divisor).
    void run_decode(std::int32_t batch, std::int32_t position);

    struct DecodeGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        bool warmed   = false;
        bool resident = false; // the expert route it was captured with (expert_route)
    };

    // The expert route of the next Forward call (execution::ForwardExperts::resident): resident
    // while the device table gives every expert a frame, so no call can miss. Read after the call's
    // frame leases, which evict experts.
    [[nodiscard]] bool expert_route() const noexcept { return residency_ && residency_->every_key_ready(); }
    void set_expert_route() { forward_->set_experts_resident(expert_route()); }

    // A captured graph keeps the expert route it was captured with: one whose route is not the
    // current one is captured again (a placement grew to every expert, or a switch or lease took
    // some away).
    void follow_expert_route(DecodeGraph& graph);

    // Runs a fixed-address round: the first time eagerly (the Ops' one-time setup), the second
    // captured, then by replay.
    void replay(DecodeGraph& graph, const std::function<void()>& body);

    // replay for a chain of `steps` bodies whose body j > 0 runs only while the gate after body
    // j - 1 passes (DecodeGraphDefinition::capture_gated_chain). The eager first run runs every body.
    void replay_gated(DecodeGraph& graph, std::int32_t steps, const std::function<void(std::int32_t)>& body,
                      const std::function<void(std::int32_t, cudaGraphConditionalHandle)>& gate);

    // Device free bytes, and the route trace's record of a graph executable just instantiated
    // (its family and shape follow from the slot it occupies).
    static std::size_t trace_free();
    void trace_graph(const DecodeGraph& graph, std::size_t free_before);

    // A call whose MoE layers may publish fetch requests runs inside an open tier round: the read
    // agent answers them only then, and a closed round would read as a silent host after a second
    // per layer.
    void require_tier_round() const;

    void forward_call(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                      const execution::ForwardVerify* verify = nullptr, const execution::MtpChunk* chunk = nullptr,
                      const execution::VisionInput* vision = nullptr, const execution::NgramRowGate* gate = nullptr,
                      bool streamed = false, bool decode_round = false);

    // The head's logits of `columns` columns as the Forward writes them: FP32 (logits32_).
    Tensor head_logits(std::int32_t columns) { return Tensor(logits32_.p, DType::FP32, {vocab_, columns}); }
    // The logits the sampler, verification, constraints and the prompt readout read, `columns`
    // columns: FP32, or their BF16 rounding (ProgramOptions::bf16_logits).
    Tensor read_logits(std::int32_t columns);
    // BF16 logits: the head's FP32 sums of `columns` columns rounded once to BF16.
    void round_logits(std::int32_t columns, cudaStream_t s);

    // A call's inputs as the Forward reads them from a staged io image at `base` (the io, or one
    // chunk's copy in a layer walk's area).
    execution::ForwardBatch io_batch(std::byte* base, std::int32_t batch, std::int32_t width,
                                     std::int32_t logit_columns) const;

    // ---------------------------------------------------------------- speculative verification (verify.cpp)
    void allocate_verification(std::int32_t lanes, std::uint64_t& allocated);

    std::int32_t* spec_host(std::size_t offset) const { return static_cast<std::int32_t*>(spec_host_.data()) + offset; }
    std::int32_t* spec_device(std::size_t offset) const { return static_cast<std::int32_t*>(spec_device_.p) + offset; }

    execution::ForwardVerify verify_view(std::int32_t batch, std::int32_t width) const;

    const ops::GdnReplayFoldPlan& fold(std::int32_t width);

    // One verification round: every row's anchor and drafts (rows with fewer drafts are padded with
    // filler drafts past their extent), one forward over batch x width columns that leaves all
    // recurrent state in place, then on-device acceptance. The pending batch licenses each row's
    // accepted drafts and its correction or bonus token; commit folds the accepted prefix in.
    //
    // The n-gram rows not yet staged (`staged[b]` columns of row b are) are read on the host after
    // the launch, while the GPU embeds and runs layer 0, and released to the round's gate before
    // ple_embed (design §12.4, n-gram S2). A round that captures or warms its graph reads them
    // first and releases the gate before the launch. Every call that writes the landing area
    // ends in a stream sync before the next round stages, so a gate never reads it across calls.
    PendingBatch verify(std::span<const SequenceHandle> sequences, std::span<const std::uint32_t> lanes,
                        std::span<const std::int32_t> positions, std::int32_t width,
                        std::span<const std::int32_t> staged, Clock::time_point start);

    // Commits a verification round: each row keeps its first accepted_tokens licensed tokens, so
    // the model state advances by that many columns (the anchor and the accepted drafts before the
    // last kept token). The GDN fold, the QSA tails and the PLE history replay those columns' records.
    CommitResult commit_verified(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions);

    // ---------------------------------------------------------------- MTP drafter (design 11.2; mtp.cpp)
    void allocate_mtp(std::int32_t lanes, std::uint64_t& allocated);

    std::int32_t* mtp_host(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_host_.data()) + offset; }
    std::int32_t* mtp_device(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_device_.p) + offset; }
    Tensor saved_column(std::uint32_t lane) const {
        return Tensor(static_cast<std::byte*>(mtp_saved_.p) + 2ULL * width_ * lane, DType::BF16, {width_});
    }

    // The zero-token resume's saved head residual (a non-MTP Program's lanes; see finish).
    Tensor head_saved_column(std::uint32_t lane) const {
        return Tensor(static_cast<std::byte*>(head_saved_.p) + 2ULL * width_ * lane, DType::BF16, {width_});
    }

    // MR1's settled invariant (see Lane): the pending cell is the last processed position's.
    bool mtp_settled(const Lane& lane) const noexcept;
    // A drafter whose cells fall out of step with the model state would only propose worse drafts,
    // which verification rejects: output is unaffected. The request keeps decoding without drafts
    // and the defect is reported, rather than failing the request.
    void require_mtp_settled(Lane& lane) const noexcept;

    // The MTP cells of a chunk (prefill or forced tokens) of one lane. The pending cell (the
    // position before the chunk) is prepended unless written; the chunk's last cell is included
    // only when the token after it is known (forced tokens). Stages ids and cells into the io.
    std::optional<execution::MtpChunk> stage_mtp_chunk(Lane& lane, std::uint32_t index, std::int32_t begin,
                                                       std::int32_t width);

    // MTP drafts for every live lane of a decode round: the pending cell's K/V when unwritten,
    // then mtp_k_ chained full steps from the saved residual and the anchor token.
    // Draft-length policy (design 11.4): the K in [0, mtp_k_] that maximizes expected tokens per
    // round time. Expected tokens are 1 + the sum of the products of the lane's conditional
    // position acceptances (learned online); a round of K drafts costs 1 + kWidthCost * K plain
    // rounds, the slope measured on the 5090 (design 11.4), so the plain round time cancels. Every
    // kProbeInterval rounds a row drafts one token more than its choice, so the estimate of the
    // next position stays current (K = 0 then probes one draft). A fixed draft length (no
    // auto_draft_length) drafts mtp_k_ every round.
    std::int32_t choose_draft_length(Lane& lane) const;

    // The cost of one draft in K = 0 rounds (design 11.4), learned online: consecutive one-row MTP
    // rounds' start-to-start times (drafting, verification, commit and host work) against their
    // drafter steps K, fit as t = a + b K by exponentially weighted least squares; a draft then costs
    // b / a rounds. The probes (one draft more every kProbeInterval rounds) keep K varied. Until
    // the fit has kCostMinRounds rounds of spread K (or when it is not positive), the measured prior
    // kWidthCost stands. A cycle past kCostGapRatio predicted rounds held an idle gap and is skipped.
    struct RoundCostFit {
        double n = 0.0, k = 0.0, kk = 0.0, t = 0.0, kt = 0.0;
    };
    RoundCostFit round_cost_;
    Clock::time_point last_round_start_{};
    std::int32_t last_round_steps_ = -1; // -1: the previous round was not a one-row MTP round
    // Batched DFlash2 rounds (choose_batch_dflash_length): each round size's measured cost, from the
    // start-to-start seconds of its consecutive decode rounds. A drafting round costs a plain round
    // of the same size plus a fixed overhead (the drafter's proposal pass, the selector, sampling)
    // plus a marginal per extra verify column, so the zero-column probe rounds and the drafting
    // rounds are fitted separately: an EMA of the plain rounds (a_p) and a regression of the
    // drafting rounds against their extra columns (batch * (width - 1)), gap-filtered like
    // round_cost_. A single line through both would hide the fixed overhead in the intercept and
    // understate the slope -- a single line priced drafts far under their cost and the policy
    // drafted where it lost (design §11.5).
    struct BatchCostFit {
        RoundCostFit draft;     // drafting rounds: columns > 0
        double plain_n = 0.0;  // plain rounds: EMA count
        double plain_t = 0.0;  // and total seconds
    };
    std::array<BatchCostFit, kMaximumConcurrency + 1> batch_cost_{};
    std::int32_t last_round_batch_ = 0; // the previous decode round's size (0: none yet)
    std::int32_t last_round_extra_ = -1; // its extra verify columns; -1: not a sample (set by the round)

    // The round cost fits describe the expert placement they were measured under: a plain round
    // timed under one placement can price drafts from another wildly wrong, and the plain EMA
    // corrects only slowly (the measured failure is design §11.5). A placement switch, and the
    // resident refill that completes a grow, start them again from the priors.
    void reset_round_costs(const char* why);

    // The plain round time a_p, the drafting round's fixed overhead `fixed` (seconds) and marginal
    // per column `b` for rounds of `batch` rows; false while either side is not determined (the
    // drafting fit also needs positive a_d and b, and the overhead cannot be negative -- noise can
    // only understate it, never invent one).
    [[nodiscard]] bool batch_cost_model(std::int32_t batch, double& a_p, double& fixed,
                                        double& b) const noexcept;

    // The fit's a and b (seconds, seconds per draft); false while it is not determined or positive.
    [[nodiscard]] bool round_cost_fit(double& a, double& b) const noexcept;

    [[nodiscard]] double width_cost() const noexcept;

    // At a decode round's start: the previous round's cycle when both are one-row MTP rounds (an
    // n-gram round's width does not follow its drafter steps: not sampled).
    void observe_round(Clock::time_point start, std::int32_t batch);

    // Two-row rounds (design 11.5): the K in [0, kPairMaxDrafts] that maximizes both rows' expected
    // tokens per round time, where a round of K drafts per row costs 1 + kPairWidthCost * K plain
    // two-row rounds. Every kProbeInterval pair rounds it drafts one token more than its choice, so
    // the rows' estimates of the next position stay current. A fixed draft length drafts the most a
    // pair round allows.
    std::int32_t choose_pair_draft_length(const Lane& a, const Lane& b);

    // Batched DFlash2 rounds, any size >= 2 (the adaptive-speculation fast path): the one shared
    // draft length in [0, min(dflash_k_, max_width_ - 1)] that maximizes the round's expected
    // tokens per time (shared_draft_length, draft_length.h): every drafting row contributes its
    // per-lane acceptance EWMA (the same lanes' estimates the pair policy reads, which the commit
    // path keeps for DFlash2), over the round's cost in plain rounds of the same size -- the
    // one-row prior (kDFlashDraftCost) until the size's own rounds are timed, then the measured fit
    // (batch_cost_), so a size whose wide rounds lose -- the offloaded setups the priors came
    // from -- measures that and stays plain. The probe alternates: one draft more than the choice
    // every second kProbeInterval rounds, and a plain round between them, so the lengths'
    // acceptances, the plain round's cost and the drafts' marginal cost keep being measured. A
    // fixed draft length drafts the most a round allows.
    std::int32_t choose_batch_dflash_length(std::span<const std::uint32_t> lanes);

    // `while_drafting` runs on the host after the draft steps are submitted, before their wait
    // (only when draft steps run).
    void mtp_draft(std::span<const std::uint32_t> lanes, std::int32_t steps,
                   const std::function<void()>& while_drafting = {});

    // After a verification round's commit: the drafter's cells of the verified window from its
    // residuals and committed tokens (index-key tails advance by the committed count), and each
    // row's last committed residual becomes its pending cell, already written.
    void mtp_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit);

    // Samples the next token of each row (row b = column b) into pending_tokens_.
    void sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions);

    const execution::Parameters& parameters_;
    DeviceContext& device_;
    const TextConfig& c_;
    ProgramOptions options_;
    NgramVolume volume_;
    NgramHash hash_;
    DeviceLayout plan_;
    std::unique_ptr<VramBudgetSource> vram_;
    VramSizing sizing_;
    // The sizing law's inputs for grow_kv's spare allowance (design §9.7): set at startup when
    // the pool is elastic, by the placement runtime's monitor start (capture_sizing_inputs).
    // `usage_at_sizing_` moves with the elastic KV pool's mappings, which are
    // this process's growth but not the pool's.
    VramDemand spare_demand_{};
    std::uint64_t usage_at_sizing_ = 0;
    std::uint64_t pool_at_sizing_  = 0;

    std::int32_t vocab_ = 0, token_domain_ = 0, chunk_ = 0, columns_ = 0, pages_per_row_ = 0;
    std::uint32_t beside_decode_chunk_ = 0; // prompt tokens per call while another lane decodes (0: no cap)
    std::int32_t width_ = 0, span_ = 0, di_ = 0, r_ = 0;
    std::uint32_t kv_pages_ = 0;

    // ---------------------------------------------------------------- prefix cache (prefix_program.cpp)
    void create_prefix_cache();
    // A lane index meaning "no lane yet" (a quote before the Engine picks the destination).
    static constexpr std::uint32_t kNoLane = 0xFFFFFFFFU;
    // The root start of a request with a prefix cache: every page reserved, the cold call grid.
    PrefixSelection prefix_root(const qwen_common::PreparedPromptData& prompt, const BasePlanImpl& base);
    std::optional<PrefixSelection> prefix_select(const qwen_common::PreparedPromptData& prompt, const BasePlanImpl& base,
                                                 std::uint32_t lane);
    void pc_make_room(std::uint32_t pages);
    // Whether a fresh prompt reusing `reuse` tokens should wait for a prefilling sibling's snapshot
    // inside their shared prefix; plans that snapshot as a boundary tap of the sibling when so.
    bool prefix_await_sibling(const qwen_common::PreparedPromptData& prompt, std::uint32_t reuse);
    // A restore batch prefetching a blocked head's blocks (none once it has landed).
    std::optional<prefix::RestoreTicket> prefetch_ticket_;
    [[nodiscard]] bool prefetch_landing() const noexcept {
        return prefetch_ticket_ && !prefix_->restore_events(*prefetch_ticket_).empty();
    }

public:
    std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;

private:
    void prefix_pin(const PrefixSelection& selection);
    void prefix_unpin(const PrefixSelection& selection) noexcept;
    void prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& selection,
                         DeviceKVPageReservation& reservation, std::uint32_t pages);
    [[nodiscard]] std::uint32_t prefix_frontier(const Lane& lane, bool finishing) const noexcept;
    void prefix_publish(Lane& lane, std::uint32_t frontier);
    void prefix_capture(Lane& lane, std::uint32_t index, prefix::cache_index::SnapshotKind kind, bool hand_over_tail);
    // The capture of the lane's state at frontier F (the lane's published path reaches F's anchor);
    // `mtp_written`: the MTP cell F - 1 is final. `fresh_tail`: the prompt-end capture's private
    // copy of the lane's tail page, which the snapshot takes instead of the lane's.
    [[nodiscard]] prefix::PrefixCache::Capture prefix_capture_at(Lane& lane, std::uint32_t F,
                                                                 prefix::cache_index::SnapshotKind kind,
                                                                 bool hand_over_tail, bool mtp_written,
                                                                 DeviceKVPageLease* fresh_tail = nullptr);
    // The zero-token resume's prompt-end capture: a snapshot at the prompt's own frontier whose
    // image carries the head's saved residual. Free cache resume and no MTP drafter only (the
    // MTP prompt-end cell encodes the cold run's first token, MR5/MR6).
    void prefix_prompt_end_capture(Lane& lane, std::uint32_t index);
    // Records a capture the cache accepted as the lane's latest.
    void prefix_captured(Lane& lane, std::uint64_t id, std::uint32_t F);
    // Removes the lane's taps realized at the call boundary B before calls[next_call]; returns whether
    // any was, and sets `boundary` when one is a boundary tap.
    bool prefix_take_taps(Lane& lane, std::uint32_t B, std::size_t next_call, bool& boundary);
    void prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last);
    void prefix_finish(Lane& lane, std::uint32_t index);
    void prefix_release(Lane& lane) noexcept;
    // The cumulative Vision key of every item of the prompt: the key of each block after it.
    [[nodiscard]] static std::uint64_t prefix_trailing_extra(const qwen_common::PreparedPromptData& prompt);
    // Every Vision item's token span: no tap and no resume lies strictly inside one.
    [[nodiscard]] static std::vector<runtime::prefix_cache::TapExclusion>
    prefix_exclusions(const qwen_common::PreparedPromptData& prompt);
    [[nodiscard]] std::optional<prefix::cache_index::SnapshotRef> prefix_resident(const Lane& lane) const;
    [[nodiscard]] std::array<std::span<const cudaEvent_t>, 2> prefix_waits(const Lane& lane) const noexcept;
    void mtp_flush_cell(Lane& lane, std::uint32_t index);

    // Starts reading a fresh request's chosen source from the Disk tier when it is held only there,
    // or finds such a read running: the request is then deferred (hybrid_sources).
    // `maximum_frontier`: a restore's paused frontier, UINT32_MAX for a fresh request.
    [[nodiscard]] bool prefix_disk_fetch(const qwen_common::PreparedPromptData& prompt, std::uint32_t maximum_frontier);
    [[nodiscard]] bool prefix_disk_pending(const qwen_common::PreparedPromptData& prompt, std::uint32_t maximum_frontier);
    // Credits a lane-resident image and a landed fetch's bytes, then drops candidates that still
    // need a Disk read.
    void prefix_credit(std::vector<prefix::cache_index::MatchCandidate>& candidates,
                       std::optional<prefix::cache_index::SnapshotRef> resident) const;

public:
    // The Disk tier (--prefix-cache-disk).
    PrefixCacheDiskOpen attach_prefix_cache_disk(const std::filesystem::path& directory, std::string_view fingerprint);
    void spill_prefix_cache_for_shutdown() noexcept;
    [[nodiscard]] std::optional<PrefixCacheDiskSpill> prefix_shutdown_spill() const { return prefix_shutdown_spill_; }

private:
    std::optional<PrefixCacheDiskSpill> prefix_shutdown_spill_;

public:
    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept;
    [[nodiscard]] Program::GraphExecutionStats graph_execution_stats() const noexcept;

private:
    // The prompt's prefill calls from `frontier`, and with `taps` the prefix taps they realize. A
    // nonzero `branch` (past the frontier) adds an exact boundary tap there: where the prompt leaves
    // a cached path (prefix_select).
    [[nodiscard]] prefix::CallPlan plan_prefill(const qwen_common::PreparedPromptData& prompt, std::uint32_t frontier,
                                                std::span<const std::uint32_t> existing, bool taps,
                                                std::uint32_t branch = 0) const;
    // A branch tap (prefix_select) needs the prompt to share at least this many cached tokens past
    // its resume frontier: what a later prompt diverging further along re-prefills without it.
    static constexpr std::uint32_t kBranchTapMinTokens = 1024;
    DeviceBuffer state_backing_, ple_backing_, tails_backing_, kv_backing_, staging_;
    DeviceBuffer head_saved_; // the zero-token resume's saved head residual, per lane (!mtp_)
    // The prompt-end capture's fixed staging images (one packed state image each).
    std::vector<DeviceBuffer> prompt_end_staging_;
    // The vq2/k4v2 exact window (design §14.3), every KV layer and lane.
    DeviceBuffer window_backing_;
    prefix::KvWindowGeometry window_geometry_;

    // ---- the host-to-device link (design §10.4: bandwidth-dependent choices follow the machine; cpu_rates.cpp) ----
    // The rate at which the prefill and prefix cost constants were fitted (RTX 5090 at PCIe 5.0 x8).
    static constexpr double kReferenceLinkBytesPerSecond = 27.5e9;
    double link_bytes_per_second_ = kReferenceLinkBytesPerSecond;
    // Measures the link once, by copying expert records from the pinned banks (the SSD tier's pinned
    // slots when the banks stay in the artifact) into the staging slots before any call uses them
    // (or takes ProgramOptions::link_bytes_per_second), and
    // rescales the link-bound costs: the prefix cost's restore rate and the walk-span cost (one pass
    // of the experts over the link).
    void measure_link();

    // ---- the CPU expert team's rates (design §10.4) ----
    // For cpu_expert_workers workers, measured at startup while the link copies records (the staging's
    // DRAM reads compete with the CPU's): experts per second on one-column jobs (memory-bound, as
    // decode's misses are) and expert columns per second on 8-column jobs (the narrow route's widest).
    // The decode miss divisor and the prefill CPU split both use them. They are measured once the
    // experts' warm start is done, on the pinned banks or the SSD tier's filled RAM copies, and
    // repeat until they settle (cpu_rates.h); until then (and without any host record) the reference
    // rates stand (ProgramOptions::cpu_split_*_rate, the i9-13900K at 6 workers on NVFP4 records).
    double cpu_expert_rate_ = 0.0; // 0 until measured
    double cpu_column_rate_ = 0.0;
    // One reading (nullopt without expert records in RAM), and the readings until they settle
    // (cpu_rates.h settle_cpu_rates) with their report.
    [[nodiscard]] std::optional<CpuRateReading> measure_cpu_rate(std::uint64_t record_stride);
    void measure_cpu_rate_settled(std::uint64_t record_stride);
    [[nodiscard]] double cpu_expert_rate() const noexcept {
        return cpu_expert_rate_ > 0.0 ? cpu_expert_rate_ : options_.cpu_split_expert_rate;
    }
    [[nodiscard]] double cpu_column_rate() const noexcept {
        return cpu_column_rate_ > 0.0 ? cpu_column_rate_ : options_.cpu_split_column_rate;
    }

    // The share of a decode call's misses the PCIe stage keeps, misses / d, from C t (C experts per
    // second on the CPU, t seconds per record on the link): d = floor(1.65 + C t). The balance point
    // of the two parts is d = 1 + C t; the measured fastest divisors put the switches between
    // them, so d steps from 2 to 3 at C t = 1.35, midway, where the startup measurement's noise
    // stays clear of it (the step evidence is MoE offload design §5.6). A whole divisor, as a
    // fractional share flips calls' splits at floor(M share) between startups. A positive
    // cpu_pcie_divisor fixes d (one_column_divisor below overrides d in a GGUF import's one-column
    // calls).
    // The CPU expert team's workers: a GGUF import's banks take their own count (ProgramOptions).
    [[nodiscard]] std::uint32_t cpu_workers() const noexcept {
        return c_.exact_experts ? options_.cpu_ggml_expert_workers : options_.cpu_expert_workers;
    }

    [[nodiscard]] int pcie_divisor() const noexcept;

    // The link's share in 256ths of a decode call's host-memory misses (MoeCpuChannel::pcie_share),
    // and the wide calls' (wide_link_share): a CPU expert's cost past its first column is compute
    // that the one-column rate the rules take hides, so wide verification rounds give the CPU more.
    // Fixed family constants, not measured rates — a MEASURED fractional share flipped calls' splits
    // at floor(M * share) between startups (3 workers: 100.8 against 107.5 tok/s, 2026-10-09), which
    // is why the divisor rule above steps at a measured boundary instead. The decode share is set
    // for this machine on weak evidence, its final T3 inside boot-to-boot spread, and re-derivation
    // with a lower-noise measure is still owed (design §10.5 records the sweep and the caveat). The
    // wide share is still upstream's (design §10.5 holds its derivation); not re-derived on this
    // machine. A positive cpu_pcie_divisor leaves every width on the divisor rule; banks other than
    // NVFP4 keep the rule everywhere.
    [[nodiscard]] int pcie_share() const noexcept;

    // {share, from}: calls of at least `from` columns keep share/256 of the call's misses on the
    // link (MoeCpuChannel::wide_share, wide_share_from); {0, 0}: every width takes pcie_share /
    // pcie_divisor. Every NVFP4 bank (either activation): the wide rule replaces the bank's
    // one-column rule in wide calls.
    [[nodiscard]] std::pair<std::int32_t, std::int32_t> wide_link_share() const;
    static constexpr std::int32_t kDecodeLinkShare256 = 85; // this machine's x16 link, NVFP4 W4A16 banks (see above)
    static constexpr std::int32_t kWideLinkShare256  = 51;  // upstream's, every NVFP4 bank

    // Without speculation, a GGUF import's one-column calls (plain decode) take the round's divisor
    // by context (plain_divisor; 1 until the first round sets it). 0 (pcie_divisor) for other banks,
    // with speculation and when cpu_pcie_divisor fixes d: MTP keeps d everywhere, as d = 1 in its
    // rounds' one-column calls measured slower (design §10.5).
    [[nodiscard]] int one_column_divisor() const noexcept;

    // Plain decode's one-column divisor: every host-memory miss on the link (d = 1) below
    // kPlainDivisorContext context tokens, half of them on the CPU (d = 2) from there; each round
    // writes it into the device io word its graph reads. The d = 1 / d = 2 measurements behind the
    // boundary, and the hit-rate-following divisor they rejected, are design §10.5.
    static constexpr std::int32_t kPlainDivisorContext = 1536;
    [[nodiscard]] static std::int32_t plain_divisor(std::int32_t position) noexcept {
        return position >= kPlainDivisorContext ? 2 : 1;
    }

    // The widest call the prefill CPU split takes (ProgramOptions::cpu_split_columns on the reference
    // link, inversely to the measured one; at most the CPU channel's call width).
    [[nodiscard]] std::int32_t split_columns() const noexcept;

public:
    [[nodiscard]] double link_bytes_per_second() const noexcept { return link_bytes_per_second_; }

private:
    // ---- elastic KV (design §14.4; kv_pool.cpp) ----
    static constexpr std::size_t kKvChunkBytes   = 2ULL << 20;
    static constexpr std::uint32_t kKvBaseTokens = 32768; // backed at startup
    static constexpr std::uint32_t kKvGrowPages  = 64;    // growth unit (4,096 tokens)
    static constexpr std::chrono::seconds kKvIdleShrink{60};
    // The last admission, release or idle shrink (steady-clock ticks), and whether the pool backs more
    // than its base: the monitor thread reads both to wake an idle engine for the shrink.
    std::atomic<std::int64_t> kv_busy_at_{0};
    std::atomic<bool> kv_grown_{false};
    void kv_busy() noexcept { kv_busy_at_.store(std::chrono::steady_clock::now().time_since_epoch().count()); }
    bool kv_idle_due() const noexcept;
    std::unique_ptr<VmmRange> kv_vmm_;
    DeviceSpan kv_span_{};

    // Maps (and zeroes) the chunks pages [0, pages) need; false when the device has no memory.
    bool map_kv(std::uint32_t pages);

    // Raises the pool's backed pages so `need` pages are available, taking the memory from the expert
    // frames, as many as the growth needs: a fixed pool at the capacity would take them all at startup,
    // and growing in steps reaches the same pool. False when it cannot: a lease holds the frames, the
    // frames do not cover it, or the pool is at its capacity.
    bool grow_kv(std::uint32_t need);

    // Lowers the pool's backed pages to the free top (not below the base), and returns the memory to the
    // expert frames. Between rounds: nothing reads free pages. `evict` (every lane free for
    // kKvIdleShrink): the cache's idle device blocks that have host copies are released first, in
    // the index's order (nothing is lost: a later match restores them from the Host tier), so a long
    // conversation that has ended gives its memory back to the frames. The delay keeps a follow-up
    // turn from paying a restore and an expert reload for memory the idle gap never used.
    void shrink_kv(bool evict = false);
    DeviceBuffer io_device_, logits32_, logits16_, sampled_, sample_pos_, configs_, token_counts_;

    // The n-gram table the volume holds: the artifact's, or another volume's (ProgramOptions::ngram_table).
    [[nodiscard]] const NgramTableConfig& ngram_table() const noexcept {
        return options_.ngram_table ? *options_.ngram_table : c_.ple.table;
    }

    // ---- causal scoring (score.cpp) ----
    // While a scored prompt prefills, each call's final residuals land in score_residuals_ at its
    // first position (score_column_: forward_call's; a layer walk's chunks carry their own).
    DeviceBuffer score_residuals_, score_logits_, score_logits16_, score_io_;
    PinnedHostBuffer score_host_{1};
    bool scoring_                = false;
    std::int32_t score_column_   = 0;
    // Columns [begin, begin + columns) of score_residuals_.
    [[nodiscard]] Tensor score_residuals(std::int32_t begin, std::int32_t columns) const;

public:
    ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options);

private:
    // ---- /v1/decide (constraint.cpp) ----
    // The constraint controls: each lane's kMaximumConstraintSets sets of up to
    // ops::kTokenConstraintChoices tokens and their counts, then one descriptor, one probability
    // record, one observe list and one observation record per round column (lanes x max width).
    static std::size_t constraint_bytes(std::int32_t lanes, std::int32_t width) noexcept;
    [[nodiscard]] std::shared_ptr<const ConstraintPlan> compile_constraint(const TokenConstraint& constraint);
    // Writes the round's descriptors (`width` columns per row, rows in `lanes` order) and the
    // constrained lanes' sets; false (and nothing written) when no row is constrained.
    bool stage_constraints(std::span<const std::uint32_t> lanes, std::int32_t width);
    // The word offset of the observe lists in constraint_ (the observation records follow them).
    [[nodiscard]] std::size_t observe_offset() const noexcept;
    // Masks the staged round's columns of `logits` [vocab, columns] and rewrites `argmax` for them.
    void constrain(Tensor& logits, Tensor* argmax);
    // After the round synchronized: the draws of `tokens`, chosen by consecutive columns from
    // `first_column` for the lane's next output steps (none for an unconstrained lane).
    void take_constraint_draws(Lane& lane, std::size_t first_column, std::span<const std::int32_t> tokens);
    // Moves the first `accepted` draws of the pending round into the lane's trace.
    static void commit_constraint_draws(Lane& lane, std::size_t accepted);
    // The readout of the lane's prompt from its last prefill call's logits (column 0).
    [[nodiscard]] PromptReadout read_prompt_frontier(const Lane& lane);
    // ---- Constrained decoding (GBNF, JSON, JSON Schema, choice, regex, tool calls; constraint.cpp) ----
    // The Engine's grammar matchers fill packed vocabulary masks for a row's positions 0..drafts
    // (TokenMaskProvider); the sampler and draft acceptance read them through the row's sampling
    // config. Sampling and acceptance run on the host-launched path after each forward, where every
    // row's drafts are known, so no graph split is needed.
    static std::size_t grammar_mask_bytes(std::int32_t token_domain, std::int32_t lanes, std::int32_t width) noexcept;
    // Row `row`'s masks for positions 0..drafts.size(), uploaded; an empty mask when the row is
    // unconstrained. Records which positions are dead ends (no legal token).
    ops::SamplingMask grammar_mask(std::size_t row, std::span<const TokenId> drafts);
    // Row `row`'s end-of-turn exclusion while its thinking is open (runtime::ReasoningBan; none
    // without the round's provider), for its sampler and verification; the drafters propose
    // without it, and a drafted end-of-turn is rejected.
    void reasoning_ban(std::size_t row, ops::SamplingConfig& config) const noexcept;
    // After the round's synchronize. A row fails when its sampler found no finite logit (the lane's
    // flag, cleared here for its next round; its tokens are placeholders), or when the tokens it
    // emits reach a dead-end position (one mask position per token).
    void mark_row_failures(PendingBatch& pending, std::span<const std::int32_t> counts) noexcept;
    // Per lane: the sampler's non-finite flag (ops::SamplingConfig::nonfinite), written by the
    // round's kernels through the host mapping.
    PinnedHostBuffer nonfinite_flags_{sizeof(std::int32_t) * kMaximumConcurrency};
    [[nodiscard]] std::int32_t* nonfinite_flag(std::uint32_t lane) const noexcept {
        return static_cast<std::int32_t*>(nonfinite_flags_.data()) + lane;
    }
    // Set for the duration of one decode or prefill call.
    struct GrammarScope {
        ProgramImpl& impl;
        GrammarScope(ProgramImpl& owner, runtime::TokenMaskProvider* masks) : impl(owner) { impl.grammar_ = masks; }
        ~GrammarScope() { impl.grammar_ = nullptr; }
    };
    DeviceBuffer grammar_masks_;
    PinnedHostBuffer grammar_host_{1};
    runtime::TokenMaskProvider* grammar_ = nullptr;
    std::array<std::uint64_t, kMaximumConcurrency> grammar_dead_{};

    DeviceBuffer constraint_;
    std::vector<std::int32_t> constraint_host_;
    std::int32_t constraint_columns_ = 0; // the staged round's columns
    std::array<std::uint64_t, kMaximumConcurrency> constraint_serials_{};
    std::uint64_t next_constraint_serial_ = 1;
    PinnedHostBuffer io_host_{1}, host_sampled_{1}, host_configs_{1};
    IoLayout io_layout_;
    std::unique_ptr<LinearAttentionStatePool> gdn_;
    prefix::KvPageGeometry kv_geometry_;
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVExecutionTablePool> tables_;
    std::size_t work_capacity_ = 0;
    std::unique_ptr<WorkspaceArena> work_;
    // SSD tier mode (design §9.5): the fetch channel and the tier outlive everything that reads
    // their memory (the residency's promotion copies, the stream, the CPU service). The placement
    // runtime owns them (and, with an expert stripe, the tier's store -- the Model's with each
    // record's tail in the stripe file); these point at what it built, kept current by its hooks.
    ops::offloaded_moe::FetchChannel* fetch_channel_ = nullptr;
    offload::HostExpertTier* tier_ = nullptr;
    PinnedHostBuffer device_error_{1}; // mapped u32: every call's MoE error word and Forward's faults (check_device_error)
    bool clear_prefix_on_release_ = false;
    offload::ExpertResidency* residency_ = nullptr;
    // The shared expert placement runtime (models/offload/expert_placement_runtime.h): the
    // residency, the RAM tier and its fetch channel, the VRAM monitor, the warm start and the
    // saved state, the placement switches and their fills.
    std::unique_ptr<offload::ExpertPlacementRuntime> placement_runtime_;
    std::unique_ptr<execution::ExpertStream> expert_stream_;
    std::vector<std::uint64_t> host_strides_; // each bank's own record stride (a GGUF import's vary)
    // The stream's leases: the ring, the wide arena and a walk's area (each a run of one pool).
    offload::ExpertResidency::FrameLease stream_lease_, stream_wide_lease_, stream_walk_lease_;
    std::size_t stream_ring_bytes_ = 0, stream_wide_bytes_ = 0, stream_walk_bytes_ = 0;
    std::size_t wide_work_ = 0;              // the plan's wide arena bytes (0: none)
    offload::ExpertResidency::FrameLease work_lease_; // the wide arena lent alone (no stream lease)
    DeviceBuffer wide_fallback_;             // the wide arena when nothing could be lent
    std::unique_ptr<WorkspaceArena> wide_arena_; // the arena swapped with *work_ around a wide call
    execution::ExpertStream::Stats stream_at_lease_;
    execution::Forward::SplitStats split_at_lease_;

    // ---------------------------------------------------------------- layer walk (prefill_walk.cpp)
    // Design §13.2: a span of consecutive streamed calls of one prompt runs layer-major (every
    // decoder layer over all of the span's chunks, then the next), so each layer's non-resident
    // experts are streamed once per span instead of once per call. The chunk grid, and with it
    // every output bit, is the chunk-major one. The span's residual stream [S*H, tokens] and each
    // chunk's staged io image live in the stream lease after the ring.
    struct LayerWalk {
        bool active = false;
        std::uint32_t lane = 0;
        std::size_t first_call = 0, end_call = 0;
        std::vector<std::int32_t> begins, widths, offsets; // per chunk: position, width, residual column
        std::uint32_t next_layer      = 0;                // the next decoder layer to enqueue
        std::uint32_t layers_per_step = 1;
        std::size_t io_stride = 0, residual_bytes = 0;
        std::int32_t host_slot = 0; // the lane's state slot as the Forward reads it on the host
        std::vector<std::optional<execution::MtpChunk>> mtp;
        std::vector<std::optional<execution::VisionInput>> vision;
        std::vector<std::vector<execution::VisionInput>> mtp_vision;
        std::uint64_t steps = 0;
        // MoE groups: consecutive chunks [first, end) whose MoE runs as one call over their columns
        // (at most kWalkMoeColumns, and what the workspace holds), its input in the area at moe_offset.
        std::vector<std::pair<std::size_t, std::size_t>> groups;
        std::size_t moe_offset = 0;
        // Prefix taps realized at a chunk end inside the span (walk_plan_taps): each chunk's staged
        // capture (0: none), copied group by group as the walk enqueues that layer of the chunk and
        // committed after the span; the next chunk's layers wait on its events.
        struct Tap {
            std::uint32_t frontier = 0;
            prefix::cache_index::SnapshotKind kind = prefix::cache_index::SnapshotKind::Tap;
            std::uint64_t staged = 0;
        };
        std::vector<Tap> taps; // per chunk
    } walk_;
    PinnedHostBuffer walk_host_{1}; // every chunk's io image, uploaded once per span
    // The next span's n-gram rows, read into walk_host_'s chunk slots while the current span's steps
    // run on the device (prefill_walk.cpp): chunk k is call first_call + k, its rows stand at slot k's
    // n-gram columns, and key is a hash of the tokens its rows hash.
    struct WalkPrefetch {
        bool valid = false;
        std::uint32_t lane = 0;
        std::size_t first_call = 0;
        std::vector<std::int32_t> begins, widths;
        std::vector<std::uint64_t> keys;
        bool done = false; // no further chunk can join the next span
    } walk_prefetch_;
    void walk_prefetch_rows(Lane& lane, std::uint32_t index, std::size_t chunks);
    [[nodiscard]] bool walk_prefetched(std::uint32_t index, std::size_t first_call, std::size_t c, std::int32_t begin,
                                       std::int32_t width);
    [[nodiscard]] std::uint64_t window_key() const noexcept;
    std::uint64_t walk_spans_ = 0, walk_calls_ = 0;
    std::uint64_t walk_prefetched_chunks_ = 0, prefetched_at_lease_ = 0; // chunks whose rows were prefetched
    static constexpr std::uint32_t kWalkMaxTokens = 65536;
    // A walk's MoE calls take up to this many columns (several chunks'): at 16,384 tokens the
    // experts run more effective TOPS than at 4,096 columns, and more again at 32,768; the
    // measured shares and the end-to-end gains are design §13.2.
    static constexpr std::int32_t kWalkMoeColumns = 32768;

    [[nodiscard]] std::size_t walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept;
    std::size_t walk_span_end(const Lane& lane, std::uint32_t index);
    void walk_begin(Lane& lane, std::uint32_t index, std::size_t end);
    PrefillProgress walk_step(Lane& lane, std::uint32_t index, Clock::time_point start);
    [[nodiscard]] execution::ForwardBatch walk_batch(std::size_t chunk);
    [[nodiscard]] Tensor walk_residual(std::size_t first, std::size_t end);
    void walk_mixer(std::size_t chunk, std::uint32_t layer);
    void walk_moe(std::size_t group, std::uint32_t layer);
    // The span's taps: realizes the lane's taps due at its inner chunk ends (as prefix_after_prefill_call
    // would after chunk-major calls) by opening a staged capture each; committed by walk_commit_taps.
    void walk_plan_taps(Lane& lane);
    void walk_capture(std::size_t chunk, std::uint32_t layer);
    void walk_commit_taps(Lane& lane);
    void walk_enqueue(std::uint32_t to_layer);
    void walk_abandon() noexcept;
    std::unique_ptr<offload::RouteTrace> trace_;
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;

    // ---- prefill CPU split (design §10.6) ----
    // Gives the CPU the thinnest non-resident experts of a layer while the link carries the rest:
    // narrow ones (n <= 8) in order of n, then expert id, as many as minimise the later of the
    // CPU's predicted time and the link's time for the experts left to it.
    // The model charges the link a record's copy per kept expert, but the stream copies most of a
    // layer's experts ahead, so the GPU side finishes far sooner than modelled and full rates overload
    // the CPU — a trace measured the GPU waiting ~the whole CPU time. Planning at kSplitRateShare of
    // the measured rates was fastest at every worker count measured (design §10.6 holds the sweep). A
    // line fitted to measured layer times starved the CPU instead (the kept experts' cost is not
    // linear).
    static constexpr double kSplitRateShare = 0.3;
    class SplitPolicy final : public execution::CpuSplitPolicy {
    public:
        explicit SplitPolicy(const ProgramImpl& program) : program_(program) {}
        std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                             std::span<std::uint8_t> cpu) const override {
            const auto& p              = program_;
            const std::uint32_t E      = p.c_.moe.experts;
            const std::int32_t* frames = p.residency_->host_table() + static_cast<std::size_t>(layer) * E;
            const double link          = static_cast<double>(p.residency_->frame_pitch(layer)) / p.link_bytes_per_second_;
            // The team's measured rates (cpu_rates.cpp: one-column jobs fix the per-expert cost, 8-column
            // jobs the per-column cost) at kSplitRateShare, the share of them a gated layer realises.
            const double per_expert    = 1.0 / (p.cpu_expert_rate() * kSplitRateShare);
            const double per_column    = 1.0 / (p.cpu_column_rate() * kSplitRateShare);
            constexpr int kNarrow      = ops::offloaded_moe::kMaxCpuColumns;
            std::array<std::uint32_t, kNarrow + 1> width_count{};
            std::uint32_t absent = 0;
            for (std::uint32_t e = 0; e < E; ++e) {
                if (columns[e] <= 0 || frames[e] >= 0) { continue; }
                ++absent;
                if (columns[e] <= kNarrow) { ++width_count[static_cast<std::size_t>(columns[e])]; }
            }
            // The best count: the CPU's time grows and the link's shrinks with each expert moved.
            double best_time = absent * link, cpu_time = 0.0;
            std::uint32_t best = 0, k = 0;
            for (int n = 1; n <= kNarrow; ++n) {
                for (std::uint32_t i = 0; i < width_count[static_cast<std::size_t>(n)]; ++i) {
                    if (k == static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs)) { break; }
                    cpu_time += std::max(per_expert, n * per_column);
                    ++k;
                    const double time = std::max(cpu_time, (absent - k) * link);
                    if (time < best_time) {
                        best_time = time;
                        best      = k;
                    }
                }
            }
            // Mark the first `best` in that order.
            std::uint32_t marked = 0;
            for (int n = 1; n <= kNarrow && marked < best; ++n) {
                for (std::uint32_t e = 0; e < E && marked < best; ++e) {
                    if (columns[e] == n && frames[e] < 0) {
                        cpu[e] = 1;
                        ++marked;
                    }
                }
            }
            return marked;
        }

    private:
        const ProgramImpl& program_;
    };
    std::unique_ptr<SplitPolicy> split_policy_;
    bool split_call_ = false;
    // Prefill staging overlap and the shared expert's stream (see ForwardExperts); destroyed after
    // every call has completed.
    struct OverlapResources {
        ~OverlapResources() {
            for (auto event : events) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            for (auto event : shared_events) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            if (stream != nullptr) { cudaStreamDestroy(stream); }
            if (shared_stream != nullptr) { cudaStreamDestroy(shared_stream); }
        }
        cudaStream_t stream = nullptr;
        std::array<cudaEvent_t, 5> events{};
        cudaStream_t shared_stream = nullptr;
        std::array<cudaEvent_t, 4> shared_events{};
    };
    OverlapResources overlap_;
    std::unique_ptr<execution::Forward> forward_;
    std::array<DecodeGraph, kMaximumConcurrency> graphs_;
    std::vector<DecodeGraph> mtp_draft_graphs_; // (batch - 1) * mtp_k_ + steps - 1
    // With sampled drafts, the same for rounds with a sampled row (greedy rounds keep the argmax).
    std::vector<DecodeGraph> mtp_sampled_graphs_;
    std::vector<DecodeGraph> mtp_gated_graphs_; // one-row argmax rounds: steps - 1
    // Whether this round's MTP drafts were sampled (their proposals wait in the drafter's io).
    bool mtp_sampled_round_ = false;
    std::vector<DecodeGraph> mtp_catch_graphs_; // (batch - 1) * max_width_ + width - 1
    std::vector<DecodeGraph> verify_graphs_; // (batch - 1) * max_width_ + width - 1

    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::array<std::int32_t, kMaximumConcurrency> host_lanes_{};
    std::vector<TokenId> pending_tokens_ = std::vector<TokenId>(kMaximumConcurrency, 0);
    std::array<std::int32_t, kMaximumConcurrency> pending_counts_{};

    // MTP drafter.
    bool mtp_            = false;
    std::int32_t mtp_k_  = 0;
    std::int32_t mtp_columns_ = 0;
    DeviceBuffer mtp_residuals_, mtp_saved_, mtp_chain_, mtp_records_, mtp_ones_, mtp_device_;
    PinnedHostBuffer mtp_host_{1};
    MtpIo mtp_io_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> mtp_drafts_;
    std::array<std::int32_t, kMaximumConcurrency> mtp_confident_{}; // per row: drafts above draft_min_logprob_
    std::array<bool, kMaximumConcurrency> from_ngram_{};
    // Copy proposals are chosen against the MTP drafts by expected tokens per round time, their
    // acceptance learned per match-length bucket (4-7, 8-11, 12-15, 16+ tokens; Strata's priors).
    // IQ3_S import, 4K + 512 tokens, MTP 4 and 7 n-gram drafts: 190.2 -> 194.2 tok/s (2026-10-15).
    static int ngram_bucket(std::uint32_t matched) { return matched < 8 ? 0 : matched < 12 ? 1 : matched < 16 ? 2 : 3; }
    static constexpr std::array<double, 4> kNgramAcceptancePriors{0.75, 0.88, 0.93, 0.96};
    std::array<int, kMaximumConcurrency> ngram_bucket_{};
    // Draft-length policy: acceptance prior and EWMA weight, and the cost of each draft column in
    // plain rounds (warm tg512 on the 5090: W = 3, 4, 5 rounds took 1.84, 2.13 and 2.54 plain
    // rounds, design 11.4; re-measured 2026-10-10 at 1.61-1.79, 1.89-2.11 and 2.42-2.53 between 8 %
    // and 20 % expert misses, where a miss-rate cost gained no more than it lost; W = 5 and 6 then
    // became 9-17 % cheaper with the one-column expert kernels up to 8 columns, not re-fitted here).
    static constexpr double kAcceptancePrior      = 0.75;
    static constexpr double kAcceptanceWeight     = 0.1;
    static constexpr double kWidthCost            = 0.38;
    static constexpr std::uint64_t kProbeInterval = 8;
    // DFlash2's calibrated length (--draft-tokens auto): a verified draft's column and a proposal in
    // plain one-row rounds. The priors stand on their own measurement; the online fit chose shorter
    // lengths and was slower (design §11.4).
    static constexpr double kDFlashDraftCost    = 0.31;
    static constexpr double kDFlashProposalCost = 0.1;
    // How much a batched round's drafting must beat its plain alternative by before it is chosen:
    // the fit decides on tens of noisy rounds and thin decisions lose — the pre-margin fit drafted
    // above its score and measured below it (design §11.5).
    static constexpr double kBatchDraftMargin = 0.05;
    // The online draft cost (width_cost): sample decay (about 50 rounds), the rounds and the share of
    // K's variance (of the mean square) it needs, its ceiling, and the cycle that counts as a gap.
    static constexpr double kCostDecay     = 0.98;
    static constexpr double kCostMinRounds = 16.0;
    // The batched fit's plain EMA needs only its own mean to settle, not a regression's spread, and
    // the probe contributes one plain round every 2 * kProbeInterval -- kCostMinRounds of them would
    // take a phase no bench and few requests reach, so the mean settles for fewer.
    static constexpr double kPlainMinRounds = 6.0;
    // How much rarer the probe becomes at a size whose choice is plain: kProbeLossFactor cycles of
    // 2 * kProbeInterval rounds between one- and two-draft probes.
    static constexpr std::uint64_t kProbeLossFactor = 8;
    static constexpr double kCostMinSpread = 0.02;
    static constexpr double kCostMax       = 2.0;
    static constexpr double kCostGapRatio  = 3.0;
    // Two-row rounds: at most 2 drafts per row, each costing kPairWidthCost plain two-row rounds
    // (the measured per-K costs are design §11.5).
    static constexpr std::int32_t kPairMaxDrafts = 2;
    // The GDN prefill's sub-chunk: prefill calls starting on multiples of it give every position the
    // same bits whatever the call boundaries (request invariance's resume grid).
    static constexpr std::uint32_t kResumeGrid = 16;
    static constexpr double kPairWidthCost       = 0.47;
    // A one-row round verifies its drafts up to the first whose draft-head probability is below
    // ProgramOptions::draft_min_p, held as its log (-inf at 0: every draft). The 0.5 default is
    // Strata's --spec-min-p rule, measured best on the 5090's AIME sweeps (design §11.4).
    double draft_min_logprob_ = 0.0;
    // Prompt seeding (seed_prompt_experts): the LFRU count an expert gains when every token of the
    // prompt routed to it. Mixed-topic Dense8 decode, credit only: 64 -0.9 %, 256 -1.8 % time, and
    // replayed decode hits +0.34 / +0.59 / +0.54 points at 64 / 256 / 1,024 (design §9.2).
    static constexpr double kPromptSeedCredit = 256.0;
    // sampled-mtp-drafts (ProgramOptions::sampled_mtp_drafts) on a token domain the sparse
    // acceptance serves: sampled rows draw their drafts and skip the confidence cut.
    bool sampled_drafts_ = false;
    std::uint64_t pair_policy_rounds_            = 0;
    std::uint64_t batch_policy_rounds_           = 0;
    bool plain_round_ = false;
    // Verification rounds' shape, program-wide: rounds, columns, columns past their row's
    // drafts; verify_by_width_[W] holds the same per round width. Report_verify_rounds reads
    // them against a lane's admission snapshot.
    std::uint64_t verify_rounds_  = 0;
    std::uint64_t verify_columns_ = 0;
    std::uint64_t verify_padded_  = 0;
    std::vector<std::array<std::uint64_t, 3>> verify_by_width_;

    // The decode path's CUDA graph executions (replay/replay_gated; reported through
    // graph_execution_stats): replays of a captured executable, bodies run eagerly (a shape's
    // first round) and bodies captured into a new graph.
    std::uint64_t graph_replays_          = 0;
    std::uint64_t graph_eager_executions_ = 0;
    std::uint64_t graph_captures_         = 0;

    // The neural drafter's most drafts per round (MTP's or DFlash2's), and whether a lane drafts.
    [[nodiscard]] std::int32_t neural_k() const noexcept { return mtp_ ? mtp_k_ : dflash_k_; }
    [[nodiscard]] bool drafter_live(const Lane& lane) const noexcept { return mtp_ ? lane.mtp_live : dflash2_; }

    // ---------------------------------------------------------------- DFlash2 drafter (dflash2_program.cpp)
    void allocate_dflash2(std::int32_t lanes, std::uint64_t& allocated);
    void create_dflash2_drafter();
    [[nodiscard]] std::int32_t dflash_feature_rows() const noexcept {
        return dim(plan_.dflash2_config->feature_width(c_.hidden_size));
    }
    // Feature columns [column, column + n) (BF16 [taps * H, n]).
    [[nodiscard]] Tensor dflash_feature_columns(std::int32_t column, std::int32_t n) const;
    // Where a prefill call or forced tokens of `cols` columns leave their features: the call region's
    // last window. During another lane's layer walk, whose span holds the call region, forced tokens
    // use the round region (no round is pending between rounds) and keep its columns.
    [[nodiscard]] std::pair<std::int32_t, std::int32_t> dflash_call_region(std::int32_t cols) const noexcept;
    std::int32_t* dflash_host(std::size_t offset) const { return static_cast<std::int32_t*>(dflash_host_.data()) + offset; }
    std::int32_t* dflash_device(std::size_t offset) const { return static_cast<std::int32_t*>(dflash_device_.p) + offset; }
    // One proposal of dflash_k_ drafts for every row of a decode round (the rows' drafts land in
    // dflash_drafts_, their candidates and laws in the io); `while_drafting` runs on the host after
    // it is submitted, before its wait.
    void dflash2_propose(std::span<const std::uint32_t> lanes, const std::function<void()>& while_drafting);
    // After a round's commit: each row's committed columns' features into its ring (`commit`: the
    // committed columns per row, already on the device at spec_layout_.commit).
    void dflash2_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit);
    // After a prefill call or forced tokens (one row): its last window's features into the ring.
    void dflash2_call_context(std::int32_t batch, std::int32_t width);
    // A layer walk: a tapped layer's completed residual for the span's last window, and after the
    // span the window into the ring.
    void dflash2_walk_features(std::size_t first, std::size_t end, std::uint32_t layer, const Tensor& residual);
    void dflash2_walk_context(Lane& lane, std::uint32_t index);
    // n consecutive positions from `first` of lane `slot`, whose features stand at `column`, into the ring.
    void dflash2_context(std::int32_t column, std::int32_t n, std::int32_t first, std::int32_t slot);
    // The stream waits for the prefix cache's copies of the lane's ring in flight (a restore, a copy-out).
    void dflash2_wait_copies(const Lane& lane);
    // A DFlash2 round's verification candidates and laws [16, K, batch] (their io offsets): a DFlash2
    // row's first drafts' from the proposal, an n-gram row's one-hot at its copied tokens. The
    // proposal's own planes when every drafting row is DFlash2's and K is the proposal's.
    std::pair<std::size_t, std::size_t> dflash2_verify_planes(std::int32_t batch, std::int32_t K);
    bool dflash2_          = false;
    bool dflash_round_     = false; // this verification's neural drafts are DFlash2's
    std::int32_t dflash_k_ = 0;
    DeviceBuffer dflash_ring_backing_, dflash_features_, dflash_device_;
    PinnedHostBuffer dflash_host_{1};
    DFlash2Io dflash_io_;
    std::unique_ptr<CyclicKVCache> dflash_ring_;
    std::unique_ptr<execution::DFlash2> drafter_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> dflash_drafts_;
    std::array<std::int32_t, kMaximumConcurrency> dflash_confident_{}; // per row: drafts above draft_min_p
    std::array<std::array<float, 8>, kMaximumConcurrency> dflash_confidence_{}; // per row: each draft's
    // --draft-tokens auto: the calibrated length of row 0's proposal (draft_length.h), one draft more
    // every kProbeInterval rounds, so the deciles of longer drafts keep being measured.
    std::int32_t dflash2_auto_length(Lane& lane, std::int32_t most);
    std::vector<DecodeGraph> dflash_propose_graphs_; // batch - 1
    std::vector<DecodeGraph> dflash_context_graphs_; // (batch - 1) * max_width_ + width - 1
    // The DFlash2 frontier of the lane a layer walk started (its inner taps capture that ring).
    std::uint32_t walk_dflash_frontier_ = 0;

    // Speculative verification (max_width_ > 1).
    std::int32_t max_width_   = 1;
    std::int32_t round_width_ = 1;
    std::size_t proposer_tokens_ = 0;
    DeviceBuffer records_backing_, ple_records_, qsa_records_, spec_device_;
    PinnedHostBuffer spec_host_{1};
    SpecLayout spec_layout_;
    ops::GdnReplayRecords records_;
    std::vector<std::optional<ops::GdnReplayFoldPlan>> folds_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> drafts_;
    // The n-gram gate of verification rounds (design §12.4): the pinned ready word, its last
    // published value (strictly increasing, never 0) and per-lane device wait statistics.
    PinnedHostBuffer gate_host_{64};
    std::uint32_t gate_sequence_ = 0;
    DeviceBuffer gate_stats_;
    DeviceBuffer ngram_scale_; // BF16 [1]: NgramTableConfig::fp8_scale, when the volume records it
    std::vector<std::uint8_t> live_;
    std::vector<std::int32_t> window_;
    std::vector<std::uint32_t> row_ids_;
    // queue_ngram's rows, read by read_queued_ngram: per segment its lane, first io column, row count
    // and rows not in the host cache, then every segment's row ids and their codes.
    struct QueuedRows {
        Lane* lane          = nullptr;
        std::size_t column  = 0;
        std::size_t rows    = 0;
        std::uint64_t misses = 0;
    };
    std::vector<QueuedRows> queued_;
    std::vector<std::uint32_t> queued_ids_;
    std::vector<std::byte> queued_rows_;
    std::optional<std::uint32_t> transaction_lane_;
    std::optional<ResumeState> pause_; // the open Pause transaction's result
    bool replaying_ = false;           // the open binding replays toward a paused frontier
    std::uint64_t next_epoch_          = 0;
    std::uint64_t next_transaction_    = 0;
    std::uint64_t pending_transaction_ = 0;
    std::uint64_t revision_            = 1;
    // ---------------------------------------------------------------- vision (vision_program.cpp)
    // The encode window of the items ending past the reused prefix [0, reused).
    [[nodiscard]] std::shared_ptr<const VisionAdmission> plan_vision(const qwen_common::PreparedPromptData& prompt,
                                                                     std::uint32_t reused = 0) const;
    void vision_reserve(Lane& lane, std::shared_ptr<const VisionAdmission> admission,
                        const qwen_common::PreparedPromptData& prompt);
    [[nodiscard]] static bool vision_encoded(const Lane& lane) noexcept { return !lane.vision || lane.vision->encoded; }
    void vision_step(Lane& lane);
    void vision_release(Lane& lane) noexcept;
    // The image columns of a prefill call [begin, begin + width), staged into the io; null without any.
    const execution::VisionInput* stage_vision(Lane& lane, std::uint32_t begin, std::int32_t width);
    // Per MTP sub-chunk: the cells whose next token is an image token; null without media.
    const execution::VisionInput* stage_mtp_vision(Lane& lane, std::uint32_t first, std::int32_t cells,
                                                   std::int32_t subchunks);
    // The call inputs that point into the io.
    execution::VisionInput vision_input_;
    std::vector<execution::VisionInput> mtp_vision_inputs_;

    // The prefix cache, declared last so it is destroyed first: it returns its page leases to pool_
    // and its lane image reads gdn_ and the lane buffers.
    prefix::StateImageLayout image_layout_;
    std::unique_ptr<prefix::LaneStateImage> lane_image_;
    std::unique_ptr<prefix::PrefixCache> prefix_;
    std::uint64_t next_capture_ = 0;
    std::array<std::span<const cudaEvent_t>, 2> next_waits_{};
    PinnedHostBuffer flush_host_{1};
    DeviceBuffer flush_device_;
    cudaEvent_t flush_uploaded_ = nullptr;

    // Last: its thread reads the source, the control law and the residency, so it stops first.
    std::unique_ptr<offload::VramMonitor> monitor_;
};

} // namespace detail
} // namespace infernix::models::qwen4_exp
