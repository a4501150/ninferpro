#pragma once

// Qwen4Exp Program internals shared by ProgramImpl and its topic files: device and io layouts,
// per-lane prefix and vision state, request plans, and small helpers.


#include "models/qwen4_exp/program/cpu_rates.h"
#include "models/qwen4_exp/program/program.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "core/arena.h"
#include "runtime/diagnostics.h"
#include "core/cuda_vmm.h"
#include "core/cyclic_kv_cache.h"
#include "core/vmm_arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/gdn_replay_records.h"
#include "core/layout.h"
#include "core/link_probe.h"
#include "core/linear_attention_state.h"
#include "core/nvtx.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/vram_budget.h"
#include "core/host_memory.h"
#include "core/weight_view.h"
#include "models/qwen_common/frontend/prepared_prompt.h"
#include "models/qwen_common/program/ngram_proposer.h"
#include "models/qwen4_exp/execution/dflash2.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/program/draft_length.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/offload/expert_residency.h"
#include "models/offload/host_expert_tier.h"
#include "models/qwen4_exp/calibrated_order.h"
#include "models/offload/expert_state.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/prefix/call_plan.h"
#include "models/qwen4_exp/program/prefix/prefix_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "models/qwen4_exp/program/vision_window.h"
#include "models/qwen_common/execution/vision_tower.h"
#include "models/qwen_common/execution/vision_weight_stream.h"
#include "models/qwen_common/program/vision_control.h"
#include "models/offload/route_trace.h"
#include "models/offload/vram_monitor.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/cast.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/qsa.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/token_constraint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace infernix::models::qwen4_exp {
namespace detail {

// A prompt with media: its items' control plan and the encode window it needs (design §16.4).
struct VisionAdmission {
    qwen_common::VisionControlPlan control;
    VisionWindowPlan window;
};

// A TokenConstraint compiled for ops::constrain_logits: its distinct multi-token sets (written into
// the lane's set columns when it is first constrained) and each output step's descriptor relative to
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
    std::vector<TokenId> readout_tokens;               // read at the prompt's frontier (/v1/decide)
    std::shared_ptr<const ConstraintPlan> constraint;  // null: unconstrained
    std::uint32_t pages     = 0; // KV page groups the request needs at most (its extent)
    std::uint32_t positions = 0; // the extent: prompt + effective output, at most max_context
    bool reuse              = false; // the prefix cache may resume and publish this request
    std::shared_ptr<const VisionAdmission> vision;
    std::shared_ptr<const PreparedPrompt> prompt; // the request's prompt, moved in by plan_request
};

// The copy-on-write source of a resume (design §15.2, rule MR6): the snapshot's tail page (the
// frontier inside a block), the anchor block when its last MTP cell encodes another continuation,
// or none.
enum class PrefixCow : std::uint8_t { None, Tail, Anchor };

// A resume source chosen by the quote: a snapshot at `frontier` over `path`, or the root.
struct PrefixSelection {
    std::optional<prefix::cache_index::SnapshotRef> snapshot;
    std::vector<prefix::cache_index::NodeRef> path; // matched blocks up to the snapshot's anchor
    std::uint32_t frontier = 0;
    std::uint32_t shared   = 0; // leading blocks mapped from the cache
    PrefixCow cow          = PrefixCow::None;
    std::uint32_t need     = 0; // pages the admission newly takes (private + restored)
    bool image_restore     = false;
    std::vector<std::uint32_t> existing; // frontiers of every matched snapshot (tap spacing)
    std::uint32_t cached_tokens = 0;     // the prompt prefix held as cached blocks (diagnostics)
    prefix::CallPlan plan;               // the suffix's prefill calls and taps from `frontier`
};

// One lane's prefix-cache state.
struct LanePrefix {
    bool reuse = false;
    std::vector<prefix::cache_index::NodeRef> path; // pinned once each: mapped, then published blocks
    std::uint32_t page_base = 0;                       // logical block of lane.pages[0]
    std::uint32_t deepest   = 0;                       // frontier of the lineage's deepest snapshot
    std::uint32_t reused    = 0;
    // The zero-token resume: the lane's state is the prompt's own end snapshot, whose head
    // residual is saved (resume_from_saved_head regenerates the first token from it).
    bool head_resume = false;
    // The prompt-end capture waiting for its anchor block (MTP, a block-aligned prompt end): the
    // lane resolves it when it publishes that block, or abandons it.
    std::uint64_t pending_anchor = 0;
    std::optional<prefix::cache_index::SnapshotRef> resume;
    std::optional<prefix::RestoreTicket> restore;     // the first call waits on it per layer
    std::uint64_t capture          = 0;               // its copy-out gates the next call per layer
    // The current request has captured (its first capture superseded the resume snapshot). `capture`
    // may still name an earlier request's capture on this slot (its copy-out must land before the slot
    // is rewritten).
    bool captured                  = false;
    std::uint64_t resident_capture = 0;               // the endpoint the slot still holds
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::vector<runtime::prefix_cache::TapHint> hints;
    std::vector<runtime::prefix_cache::TapExclusion> exclusions; // Vision spans: no snapshot inside
    // Block lookup hashes: the prompt's, extended by publication over the generated blocks; the
    // prompt's cumulative Vision keys, and the key every block after the prompt carries.
    std::vector<std::uint64_t> hashes, extras;
    std::uint64_t trailing_extra = 0;
    // The admission's diagnostics: the prompt prefix held as cached blocks, and the bytes its
    // restore copied back from the Host tier.
    std::uint32_t cached_tokens  = 0;
    std::uint64_t restored_bytes = 0;
};

struct QuoteImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t pages = 0;
    std::uint32_t lane  = 0;
    std::optional<PrefixSelection> prefix;
    std::shared_ptr<const VisionAdmission> vision;
};

// A media prompt's encode window and handoff (design §16.4): filled at reserve, encoded over
// admission steps, released once the prompt's last call is enqueued or the lane is released.
struct LaneVision {
    std::shared_ptr<const VisionAdmission> admission;
    qwen_common::VisionControl control;            // the encoded items' controls
    std::vector<std::shared_ptr<const qwen_common::PreparedMediaPayload>> payloads; // per encoded item
    std::vector<std::uint32_t> visual;         // the encoded visual tokens' positions, ascending
    offload::ExpertResidency::FrameLease lease;         // the handoff (W), or handoff and window (L)
    DeviceBuffer fallback;                     // the same bytes when the residency lends no frame
    Tensor handoff;                            // BF16 [out, V_encoded]
    std::unique_ptr<qwen_common::execution::VisionWeightStream> stream;
    std::optional<qwen_common::execution::VisionParameters> weights; // the stream's rebased view
    std::unique_ptr<qwen_common::execution::VisionPassLayout> layout;
    std::unique_ptr<qwen_common::execution::VisionTowerPass> pass;
    std::optional<CudaEventTimer> timer;
    bool encoded = false;
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

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction,
                                     std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                                     runtime::ExecutionTiming timing, std::span<const std::int32_t> counts = {},
                                     std::uint32_t stride = 1) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = counts;
        out.row_stride_ = stride;
        out.timing_     = timing;
        return out;
    }
    static void set_row_failure(PendingBatch& p, std::size_t row,
                                 runtime::PendingRowFailure failure) noexcept {
        p.row_failures_[row] = failure;
    }
    static const void* owner(const PendingBatch& p) noexcept { return p.owner_; }
    static std::uint64_t transaction(const PendingBatch& p) noexcept { return p.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& p) noexcept {
        return {p.rows_.data(), p.row_count_};
    }
    static void consume(PendingBatch& p) noexcept {
        p.owner_       = nullptr;
        p.transaction_ = 0;
        p.row_count_   = 0;
        p.tokens_      = {};
        p.row_counts_  = {};
        p.row_stride_  = 0;
    }
};

using Clock = std::chrono::steady_clock;

inline std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// Per-call inputs staged through pinned memory, one device copy. Decode and verification rounds
// upload only the prefix before `ngram` plus their rows (io_prefix), so every array they read,
// the RoPE positions and pooled-block starts included, precedes `ngram`. `rope` holds a call's
// [columns, 3] axis-major RoPE positions, `block_rope` its [batch, 3] block starts; `mtp_rope` and
// `mtp_block_rope` the MTP chunk's per sub-chunk (execution::MtpChunk). `gate` is the U32 a gated
// verification round's rows wait for (design §12.4): it precedes `ngram`, so the round uploads it
// with the prefix and its rows go through the gate instead.
struct IoLayout {
    std::size_t ids = 0, positions = 0, rope = 0, block_rope = 0, slots = 0, rows = 0, columns = 0, gate = 0, divisor = 0,
                ngram = 0,
                mtp_ids = 0, mtp_cells = 0, mtp_rope = 0, mtp_block_rope = 0, vision_columns = 0,
                mtp_vision_columns = 0, bytes = 0;
};

// A request's verification rounds whose rows were read behind the n-gram gate, and those reads.
struct GateTraffic {
    std::uint64_t rounds = 0, reads = 0, read_ns = 0;
};
inline constexpr std::size_t kGateStatsBytes     = 16;          // per lane: U64 wait ns, U64 waits
inline constexpr std::uint64_t kSlowGatedReadNs  = 100000000;   // a gated round's reads warn past 100 ms

// Device I32 arrays of a verification round, each with room for every lane and width.
struct SpecLayout {
    std::size_t target = 0, drafts = 0, extents = 0, lengths = 0, anchors = 0, licensed = 0, counts = 0,
                accepted = 0, commit = 0, words = 0;
};

// I32 MTP drafter io: chain ids [B], cells [K, B], per draft step the cells' RoPE positions and
// pooled-block starts [K][3][B] (axis-major per step), drafts [K, B], catch-up ids [W, B], gather
// columns [B], catch-up cells [W, B], the drafts' FP32 log-probabilities [K, B]. Everything before
// `drafts` is uploaded per draft round. Sampled drafts add, per draft step, the head's top ranks
// and their FP32 log-probabilities [kSampledDraftRanks, B] and the proposal's candidates and FP32 q
// [16, B] (step j at j * rows * B), then the verify layout of candidates and q [16, W - 1, B].
struct MtpIo {
    std::size_t ids = 0, cells = 0, rope_cells = 0, block_rope_cells = 0, drafts = 0, up_ids = 0, gather = 0,
                up_cells = 0, logprobs = 0, sample_rows = 0, sample_logprobs = 0, draft_candidates = 0, draft_q = 0,
                verify_candidates = 0, verify_q = 0, words = 0;
};

// I32 DFlash2 drafter io (FP32 words where noted): per row its anchor, frontier, live block columns
// and ring lane [L] (L lanes; a round of B rows uses the first B of each); the block's ids and
// positions [W, L] (W = drafts + 1); the drafts [K, L] and their FP32 selector probabilities [K, L],
// their candidates and FP32 laws [16, K, L];
// the verification layout of candidates and laws [16, V - 1, L] (V the widest round: a round
// verifying fewer drafts than proposed, n-gram rows' one-hot laws); then a prefill call's, forced
// tokens' or layer walk's context positions [window], its count and its lane. Everything before `ids`
// is uploaded per proposal.
struct DFlash2Io {
    std::size_t anchors = 0, frontiers = 0, valid = 0, slots = 0, ids = 0, positions = 0, drafts = 0, confidence = 0,
                candidates = 0,
                q = 0, verify_candidates = 0, verify_q = 0, context_positions = 0, context_count = 0, context_slot = 0,
                words = 0;
};

// Sampled MTP drafts (sampled-mtp-drafts) draw from the draft head's four most probable tokens,
// the rank count the sampler trials kept (model-quality.md §6.7, `sampled-mtp-drafts`).
inline constexpr std::int32_t kSampledDraftRanks = 4;

// Every fixed device allocation of a Program and the layouts in it (design §9.7). One function
// (ProgramImpl::plan_device) computes it from the options and the configuration alone: before the
// weights are read, for the startup check, and in the constructor, which allocates exactly it.
struct DeviceLayout {
    std::int32_t lanes = 0, token_domain = 0, head_rows = 0, chunk = 0, mtp_k = 0, max_width = 1, columns = 0,
                 pages_per_row = 0, mtp_columns = 0;
    bool mtp = false;
    std::uint32_t kv_pages = 0, kv_layers = 0;
    std::uint64_t record_stride = 0; // the largest expert record
    std::uint64_t frame_bytes   = 0; // the expert cache's average frame
    LinearAttentionStatePoolLayout gdn;
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    ops::GdnReplayRecordLayout records;
    IoLayout io;
    SpecLayout spec;
    MtpIo mtp_io;
    std::size_t gdn_bytes = 0, ple = 0, tails = 0, kv = 0, records_bytes = 0, ple_records = 0, qsa_records = 0;
    // The vq2/k4v2 exact window of every KV layer and lane (prefix::KvWindowGeometry; zero otherwise).
    std::size_t window = 0;
    std::size_t logits32 = 0, logits16 = 0, token_counts = 0, work = 0, staging = 0;
    // Causal scoring: the BF16 tile copy beside score_logits (ProgramOptions::bf16_logits).
    std::int32_t staging_slots = 0; // misses staged per pass of a layer's experts
    // The arena wide prefill calls swap in (columns above static_columns), lent from expert frames
    // while a prompt runs such calls; zero when the static arena covers every call.
    std::size_t wide_work = 0;
    std::int32_t static_columns = 0;
    // Elastic KV (design §14.4): the backing reserves `kv` bytes of address space and maps the
    // chunks pages [0, n) of every plane (and the block tables) need; `kv_base_pages` are mapped at
    // startup and counted as fixed, the rest grows from the expert frames when admissions need it.
    bool kv_elastic             = false;
    std::uint32_t kv_base_pages = 0;
    std::size_t mtp_column_bytes = 0, mtp_ones = 0;
    // DFlash2 (docs/maintainer/dflash.md): drafts per proposal; the feature columns, a round region
    // (every column of a decode or verification round) then a call region (a prefill call's or layer
    // walk's last window), and their bytes; the context ring of every lane, the drafter's io and its
    // configuration.
    bool dflash2 = false;
    std::int32_t dflash_k = 0, dflash_round_columns = 0, dflash_columns = 0;
    std::size_t dflash_features = 0, dflash_ring = 0;
    CyclicKVCacheLayout dflash_ring_layout;
    DFlash2Io dflash_io;
    std::optional<DFlash2Config> dflash2_config;
    // Causal scoring: the window's final residuals (BF16 [S*H, max_context]), one tile's FP32 logits
    // and its targets, log-probabilities and distributions (ProgramImpl::causal_score).
    std::size_t score_residuals = 0, score_logits = 0, score_logits16 = 0, score_io = 0;
    ProgramDevicePlan bytes;
};

// Causal scoring's logits tile: 64 columns of FP32 logits (64 MB at 248,320 rows).
inline constexpr std::int32_t kScoreTileColumns = 64;
// The tile's I32/FP32 words: targets and their log-probabilities, then top ids and log-probabilities
// and candidate ids and log-probabilities of up to kMaximumScoreTopTokens per column.
inline constexpr std::size_t kScoreIoWords =
    static_cast<std::size_t>(kScoreTileColumns) * (2 + 4 * kMaximumScoreTopTokens);

inline std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

inline std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

inline ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
    if (!(s.temperature >= 0.0F) || !(s.top_p > 0.0F && s.top_p <= 1.0F) || s.min_p < 0.0F || s.min_p > 1.0F) {
        throw std::invalid_argument("Qwen4Exp: sampling parameters are out of range");
    }
    ops::SamplingConfig out;
    out.temperature       = s.temperature;
    out.top_k             = s.top_k;
    out.top_p             = s.top_p;
    out.min_p             = s.min_p;
    out.presence_penalty  = s.presence_penalty;
    out.frequency_penalty = s.frequency_penalty;
    out.seed              = s.seed;
    return out;
}

// A paused request (design §15.7): what its lane held besides model state. The pause published
// the state at `target` to the prefix cache (a request the cache publishes); a resume binds the
// ledger from the deepest cached state up to `target` and replays the rest without sampling.
struct ResumeStateImpl {
    std::uint32_t target = 0;   // the paused frontier: the positions the lane's state held
    bool generated       = false; // paused after its first token (the ledger runs past the prompt)
    std::vector<std::int32_t> history; // prompt, then committed output
    std::uint32_t prompt_tokens = 0;
    // Prefix-cache lookup keys over the ledger (the prompt's, extended over published output
    // blocks), and the capture the pause published.
    std::vector<std::uint64_t> hashes;
    std::uint64_t capture = 0;
    runtime::BeginSummary begin;
    std::uint64_t prefill_ns = 0, decode_ns = 0, decode_share_ns = 0;
    double vision_seconds = 0;
    SpeculativeStats speculative;
    std::array<double, 8> mtp_accept{};
    std::uint64_t mtp_policy_rounds = 0;
    SelectorCalibration dflash_calibration; // DFlash2: acceptance by selector-probability decile
    std::array<double, 4> ngram_accept{}; // the n-gram copies' acceptance per match-length bucket
    std::vector<ConstrainedDraw> constraint_trace;
    offload::ExpertResidency::Stats cache_at_admission;
    std::uint64_t cpu_served_at_admission = 0;
    NgramVolume::Counters ngram;
    GateTraffic gate;
};

} // namespace detail
} // namespace infernix::models::qwen4_exp
