#pragma once

// Glm5Next (GLM-5.3-Flash) instance configuration as recorded by the converter
// (tools/convert/glm5_next.py). The mathematics is fixed by this implementation; the config only
// supplies dimensions (docs/maintainer/glm5_3-flash-design.md §2).

#include "models/load_options.h"
#include "models/offload/geometry.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace infernix::artifact {
struct Directory;
}

namespace infernix::models::glm5_next {

enum class MixerKind : std::uint8_t { Kda, Mla };

// Kimi Delta Attention: q, k and v each heads x head_dim wide, with their own depthwise causal
// convolutions; the forget gate is lower_bound * sigmoid(exp(A_log) * (f + dt_bias)).
struct KdaConfig {
    std::uint32_t heads       = 0;
    std::uint32_t head_dim    = 0;
    std::uint32_t conv_kernel = 0;
    float lower_bound         = 0;

    [[nodiscard]] std::uint32_t width() const noexcept { return heads * head_dim; }
};

// Multi-head latent attention without RoPE, read in its absorbed form against the shared latent.
struct MlaConfig {
    std::uint32_t heads      = 0;
    std::uint32_t q_lora     = 0;
    std::uint32_t kv_lora    = 0; // the latent each token stores
    std::uint32_t head_dim   = 0; // query/key width per head (qk_nope_head_dim)
    std::uint32_t value_dim  = 0;

    [[nodiscard]] std::uint32_t query_width() const noexcept { return heads * head_dim; }
    [[nodiscard]] std::uint32_t value_width() const noexcept { return heads * value_dim; }
};

// The DSA indexer: keys pooled `pool` tokens at a time, `top_k` tokens (top_k / pool pools) plus
// the incomplete tail pool selected per query.
struct IndexerConfig {
    std::uint32_t heads    = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t top_k    = 0;
    std::uint32_t pool     = 0;

    [[nodiscard]] std::uint32_t pool_budget() const noexcept { return top_k / pool; }
    // Tokens a query attends to at most: the selected pools plus its incomplete tail pool.
    [[nodiscard]] std::uint32_t max_selected() const noexcept { return top_k + pool - 1; }
};

struct MoeConfig {
    std::uint32_t experts             = 0;
    std::uint32_t top_k               = 0;
    std::uint32_t intermediate        = 0;
    std::uint32_t shared_intermediate = 0; // one ungated shared expert
    float routed_scale                = 0;
};

struct HyperConnectionConfig {
    std::uint32_t streams         = 0;
    std::uint32_t sinkhorn_iters  = 0;
    float eps                     = 0;
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t dense_layers            = 0; // the first layers run a dense MLP
    std::uint32_t dense_intermediate      = 0;
    float rms_norm_eps                    = 0;
    float swiglu_limit                    = 0;
    std::vector<std::int32_t> eos_token_ids;
    std::vector<MixerKind> layer_types;
    // Index of each layer among the layers of its kind.
    std::vector<std::uint32_t> compact_layer_indices;
    std::uint32_t kda_layers = 0;
    std::uint32_t mla_layers = 0;
    KdaConfig kda;
    MlaConfig mla;
    IndexerConfig indexer;
    MoeConfig moe;
    HyperConnectionConfig hc;

    [[nodiscard]] std::uint32_t moe_layers() const noexcept {
        return num_hidden_layers - dense_layers;
    }
    [[nodiscard]] std::uint32_t context_ceiling() const noexcept { return max_position_embeddings; }
    [[nodiscard]] std::uint32_t residual_width() const noexcept { return hc.streams * hidden_size; }
};

// A DFlash2 drafter (docs/maintainer/dflash.md) from a companion artifact (--draft-model): its own
// block geometry; the hidden width, vocabulary, token embedding and head are the target's. Every
// layer is a sliding-window layer.
struct DFlash2Config {
    std::uint32_t layers         = 0;
    std::uint32_t heads          = 0;
    std::uint32_t kv_heads       = 0;
    std::uint32_t head_dim       = 0;
    std::uint32_t intermediate   = 0;
    std::uint32_t window         = 0; // the sliding window and context ring capacity
    float rms_norm_eps           = 0;
    float rope_theta             = 0; // full-head 1-D RoPE at absolute positions
    std::vector<std::uint32_t> target_layer_ids; // the completed target layers whose stream means are the features
    std::uint32_t mask_token_id  = 0;
    std::uint32_t conv_kernel    = 0; // dynamic convolution taps
    std::uint32_t conv_group     = 0; // channels sharing one dynamic coefficient
    std::uint32_t selector_rank  = 0;
    std::uint32_t selector_top_k = 0;

    // The feature width: the target hidden width per tapped layer.
    [[nodiscard]] std::uint32_t feature_width(std::uint32_t hidden) const noexcept {
        return hidden * static_cast<std::uint32_t>(target_layer_ids.size());
    }
};

// The vision tower (docs/maintainer/glm5_3-flash-design.md §15.2): 24 blocks of RMSNorm, Q/K/V
// with biases, per-head q/k norms, 2-D axial RoPE and a clamped SwiGLU MLP, then a post norm,
// the m x m downsample and a gated merger into the text width. Patches are 3 x t x p x p pixels.
struct VisionConfig {
    std::uint32_t depth                        = 0;
    std::uint32_t hidden_size                  = 0;
    std::uint32_t num_heads                    = 0;
    std::uint32_t intermediate_size            = 0;
    std::uint32_t out_hidden_size              = 0; // the text width
    std::uint32_t projection_intermediate_size = 0; // the merger's gate/up rows
    std::uint32_t patch_size                   = 0;
    std::uint32_t temporal_patch_size          = 0;
    std::uint32_t spatial_merge_size           = 0;
    float rms_norm_eps                         = 0;
    float swiglu_limit                         = 0;
    float rope_theta                           = 0;
    std::uint32_t image_token_id = 0, image_start_token_id = 0, image_end_token_id = 0;
    std::uint32_t video_token_id = 0, video_start_token_id = 0, video_end_token_id = 0;

    [[nodiscard]] std::uint32_t head_dim() const noexcept { return hidden_size / num_heads; }

    [[nodiscard]] std::uint32_t patch_width() const noexcept {
        return 3U * temporal_patch_size * patch_size * patch_size;
    }

    [[nodiscard]] std::uint32_t merge_area() const noexcept {
        return spatial_merge_size * spatial_merge_size;
    }

    // The downsample's input: one merged token's m x m patches.
    [[nodiscard]] std::uint32_t merged_width() const noexcept { return merge_area() * hidden_size; }
};

struct Config {
    TextConfig text;
    std::optional<DFlash2Config> dflash2; // the DFlash2 drafter is bound (--spec dflash2 --draft-model)
    std::optional<VisionConfig> vision;   // the tower is bound (--vision)
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

// The DFlash2 drafter of a companion artifact (a directory without text whose dflash2 component
// records the target identity it was converted for, tools/convert/glm5_next.py), checked against
// the target's text config.
[[nodiscard]] DFlash2Config parse_dflash2_config(const artifact::Directory& companion,
                                                 const TextConfig& target);

// The offload tier's view of this family's expert banks (models/offload/geometry.h): one bank per
// MoE layer (the layers after the dense ones), bank b serving layer dense_layers + b.
[[nodiscard]] inline offload::ExpertBankGeometry bank_geometry(const TextConfig& config) noexcept {
    return {.banks = config.moe_layers(), .experts_per_bank = config.moe.experts,
            .top_k = config.moe.top_k};
}

} // namespace infernix::models::glm5_next
