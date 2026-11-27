#pragma once

// The Glm5Next forward pass (docs/maintainer/glm5_3-flash-design.md §2): the token embedding
// expanded to four mHC streams, 45 blocks of KDA or sparse MLA (with its DSA indexer) then a dense
// or offloaded MoE FFN, the stream mean, the final norm and the LM head. One call runs `batch`
// sequences of `width` consecutive positions: a prefill chunk is one sequence of many positions, a
// decode round up to eight sequences of one. The attention is NoPE: no position enters a mixer
// except through the caches' positions.

#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "infernix/ops/dsa_indexer.h"
#include "infernix/ops/mla_sparse_attention.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "models/glm5_next/execution/expert_stream.h"
#include "models/glm5_next/execution/parameters.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::glm5_next::execution {

// Faults the Forward's own kernels report in the experts' error word (ForwardExperts::error), as
// layer << 16 | code like the MoE's.
inline constexpr std::uint32_t kErrorTokenId = 0xFF11; // an id outside the vocabulary reached the token embedding

// Program-owned recurrent state of every slot: one KDA layer per KDA block, its convolution state
// BF16 [3 * width, taps - 1, slots] (q | k | v channels) and its FP32 [128, 128, heads, slots]
// recurrence.
struct ForwardState {
    LinearAttentionStatePool* kda = nullptr;
};

// One MLA block's paged caches: the shared latent and the indexer's keys, gates and pooled keys,
// token-paged through ForwardKV::block_tables.
struct MlaLayerCache {
    ops::MlaLatentBatchView latent;
    ops::DsaIndexerCacheView index;
};

struct ForwardKV {
    std::vector<MlaLayerCache> layers; // per MLA block
    Tensor block_tables;               // I32 [pages per row, rows]
};

// Where each MoE layer's routed experts live: device frames for resident experts, the pinned host
// bank otherwise. Indexed by MoE layer (layer - dense_layers).
struct ForwardExperts {
    // Per MoE layer: its frame pool's base and pitch (frame f at frame_bases[l] + f * pitch; empty:
    // no frames), and its device I32 [E] table (frame or -1).
    std::vector<const std::uint8_t*> frame_bases;
    std::vector<std::uint64_t> frame_pitches;
    std::vector<const std::int32_t*> frames;
    // Optional route log for the expert cache: MoE layer l's routed ids (I32 [k, T]) are copied to
    // route_log + l * route_stride.
    std::int32_t* route_log  = nullptr;
    std::size_t route_stride = 0;
    // RAM tier (banks streamed from the artifact): per MoE layer, the device table of host record
    // pointers (null: the record is only in the artifact), which replaces the pinned bank. Empty: the
    // banks are pinned.
    std::vector<const std::uint8_t* const*> host_tables;
    // SSD tier: per MoE layer, the channel the device requests SSD-only records through (empty:
    // every record outside the frames is in RAM).
    std::vector<ops::MoeFetchChannel> fetch;
    // Per MoE layer, the host expert engine's channel for CPU-served misses (empty: none).
    std::vector<ops::MoeCpuChannel> cpu;
    // Device slots non-resident records are staged into before they are computed (ops::MoeExpertSource).
    std::uint8_t* staging_base = nullptr;
    std::int32_t staging_slots = 0;
    // Landing frames of decode rounds (offload::ExpertResidency::landing_table): device I32
    // [MoE layers][landing_slots] and the landed log of the same shape; null: none.
    const std::int32_t* landing = nullptr;
    std::int32_t* landed        = nullptr;
    std::int32_t landing_slots  = 0;
    // Mapped word a call that could not serve an expert, or an out-of-vocabulary id, writes; the
    // Program checks it after each synchronization. Null: none.
    std::uint32_t* error = nullptr;
    // Prefill chunks stage their next pass of misses on this stream while one pass computes;
    // decode rounds stage their misses on it while the resident experts compute.
    cudaStream_t overlap_stream = nullptr;
    std::array<cudaEvent_t, 5> overlap_events{};
    // Every routed expert holds a frame: calls run without staging or the fork.
    bool resident = false;
    // FastPath::GgmlBf16Prefill: a GGUF bank's wide-call jobs run the BF16 tensor-core tiles
    // (ops::offloaded_moe::WideA16::kGgmlBf16) instead of the exact A16 tiles.
    bool ggml_bf16_prefill = false;
    // FastPath::ExpertA4Residual: an NVFP4 bank's wide-call jobs run the 3-term residual A4
    // tensor-core tiles (ops::offloaded_moe::WideA16::kA4Residual) instead of the exact A16 tiles.
    bool expert_a4_residual = false;
    // The prefill expert stream (expert_stream.h): when it is active around a call, its ring
    // carries the chunk's non-resident experts, copied ahead by copy-engine DMA.
    ExpertStream* stream = nullptr;
};

// Speculative verification (docs/maintainer/dflash.md): a call of `batch` sequences of `width`
// columns whose KDA blocks read their lanes' states without writing them, recording per KDA block
// the columns' convolution input and the recurrence's k, v, g and beta; the commit folds a prefix
// of each row into its lane (Forward::commit_verified). Each tensor holds the widest round's
// columns; a call uses the first batch * width.
struct KdaVerifyRecord {
    Tensor qkv;                // BF16 [3 * width, columns]: the convolution input
    Tensor key, value, forget; // BF16 [width, columns]
    Tensor beta;               // BF16 [heads, columns]
};

struct ForwardBatch {
    Tensor ids;           // I32 [T]
    Tensor positions;     // I32 [T]: cache position of each column
    Tensor slots;         // I32 [batch]: KDA state slot of each sequence, updated in place
    Tensor table_rows;    // I32 [batch]: block-table row of each sequence
    Tensor logit_columns; // I32 [n]: the columns whose logits are produced
    std::span<const std::int32_t> host_slots; // the same slots on the host
    std::int32_t batch = 0;
    std::int32_t width = 0;
    // When set: BF16 [S*H, T] copy of the final residual.
    Tensor residual_out;
    // When set: BF16 [S*H, 1] copy of the final residual of the call's last column (the prefix
    // cache's prompt-end capture: the zero-token resume regenerates the first token from it).
    Tensor head_residual_out;
    // DFlash2 target features (docs/maintainer/dflash.md): when set, BF16 [taps * H, n] receives,
    // for the call's last n columns, the stream mean of each tapped layer's completed residual
    // (tap i at rows [i * H, (i + 1) * H)). n <= T.
    Tensor draft_features;
    // When set, the call verifies (one record per KDA block): width > 1 with any batch.
    const std::vector<KdaVerifyRecord>* verify = nullptr;
    // Media embeddings (the vision tower's output, BF16 [H, n]) that replace the token embeddings
    // of the call's columns vision_columns (I32 [n]) before the stream expansion. Empty: none.
    Tensor vision_embeddings;
    Tensor vision_columns;
};

// Optional observation of every block, for reference comparison.
struct ForwardTap {
    std::vector<Tensor>* residuals = nullptr; // receives BF16 [S*H, T] copies, one per block
    std::vector<Tensor>* routes    = nullptr; // receives I32 [top k, T] routed ids, one per MoE block
};

[[nodiscard]] ops::MlaAttentionGeometry mla_geometry(const TextConfig& config);
[[nodiscard]] ops::DsaIndexerGeometry indexer_geometry(const TextConfig& config);

class Forward {
public:
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardState state,
            ForwardKV kv, ForwardExperts experts, std::int32_t max_context);
    Forward(const Forward&)            = delete;
    Forward& operator=(const Forward&) = delete;

    // FP32 logits [V, n] of batch.logit_columns.
    void run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap = nullptr);

    // The steps run() composes, for a prefill layer walk (glm5_3-flash-design.md §11.1): a walk
    // runs each layer over every chunk of its span before the next layer, so the chunk's residual
    // stream and its mHC state live in caller memory between steps. Chunk c's layer l reads what
    // the chunk-major call c reads there (its own columns, the layer's caches and state after
    // chunks 0..c-1), so every bit equals the chunk-major prefill's.
    struct CallColumns {
        Tensor residual; // BF16 [S*H, T]
        Tensor x;        // BF16 [H, T]: the next half-layer's mHC mix
        Tensor post;     // FP32 [S, T]
        Tensor comb;     // FP32 [S, S, T]
    };

    // The expert stream's layer boundaries a step crosses: a walk's layer waits for the layer's
    // copies before its first chunk and frees the ring half after its last.
    struct StreamUse {
        bool first = true;
        bool last  = true;
    };

    // Validates the call, selects its expert route and resets the workspace.
    void begin_call(const ForwardBatch& batch);
    // The token embedding, the stream expansion and layer 0's attention mix.
    void embed(const ForwardBatch& batch, CallColumns& columns);
    // Layer l (with its DFlash2 feature and tap copies).
    void step(std::uint32_t l, const ForwardBatch& batch, CallColumns& columns, StreamUse stream,
              const ForwardTap* tap = nullptr);
    // The final-residual copies the batch names and, with `logits`, the head over its logit
    // columns.
    void finish(const ForwardBatch& batch, const CallColumns& columns, Tensor* logits);

    // FP32 logits [V, n] of n final residuals (BF16 [S*H, n], ForwardBatch::residual_out): the
    // stream mean, the final norm and the LM head, as run() applies them to its logit columns.
    void head_logits(const Tensor& residuals, Tensor& logits);

    // Workspace of one call of up to `columns` columns at `max_context`.
    [[nodiscard]] static std::size_t workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                     std::int32_t max_context);

    // Commits a verified call's KDA blocks: row b of the call (its lane slots[b] on the host)
    // folds its first columns[b] columns into its state and convolution window; zero leaves it.
    void commit_verified(const std::vector<KdaVerifyRecord>& records, std::int32_t batch,
                         std::int32_t width, std::span<const std::int32_t> slots,
                         std::span<const std::int32_t> columns);

    // Drafter-training capture without a bound drafter: the layers whose stream-mean residuals
    // ForwardBatch::draft_features receives (a bound drafter's target layers otherwise).
    void set_feature_taps(std::vector<std::uint32_t> layers) { draft_taps_ = std::move(layers); }

    [[nodiscard]] bool experts_resident() const noexcept { return experts_.resident; }
    void set_experts_resident(bool resident) noexcept { experts_.resident = resident; }

private:
    void layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, Tensor& post, Tensor& comb,
               Tensor& x, const ForwardTap* tap);
    Tensor kda(const KdaParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    Tensor mla(const MlaParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    Tensor mlp(const MlpParameters& p, const Tensor& x);
    Tensor moe(const MoeParameters& p, const Tensor& x, std::uint32_t moe_layer, Tensor* route_tap);
    void mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor& x, Tensor& post, Tensor& comb);
    void update_mix(const HyperConnectionParameters* next, const Tensor& y, Tensor& residual, Tensor& x,
                    Tensor& post, Tensor& comb);
    void capture_draft_features(std::uint32_t layer, const Tensor& residual, const Tensor& features);
    void head(const Tensor& residual, Tensor& logits);

    const Parameters& parameters_;
    const TextConfig& config_;
    DeviceContext& device_;
    WorkspaceArena& work_;
    ForwardState state_;
    ForwardKV kv_;
    ForwardExperts experts_;
    std::int32_t max_context_;
    std::vector<std::uint32_t> draft_taps_; // the DFlash2 drafter's target layers (empty without it)
    bool eager_ = false;                    // the call is a prefill chunk (one sequence, several positions)
    StreamUse stream_{};                    // the current step's expert stream boundaries
};

} // namespace infernix::models::glm5_next::execution
