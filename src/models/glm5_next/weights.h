#pragma once

// Logical parameter handles of a loaded Glm5Next model. Names follow the converter
// (tools/convert/glm5_next.py); handles index the frozen Model's bound weights.

#include "core/weight_view.h"
#include "infernix/ops/linear.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace infernix::models::glm5_next {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
    [[nodiscard]] bool valid() const noexcept { return index != std::numeric_limits<std::size_t>::max(); }
};

struct BoundWeight {
    std::string name;
    WeightView view;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only; // of the first Use, if any
};

// One mHC mixer (ops::mhc_mix): fn [(2+S)S, S*H] (BF16), base FP32 [(2+S)S], scale FP32 [3].
struct HyperConnectionWeights {
    WeightId fn, base, scale;
};

// Kimi Delta Attention: q/k/v share one input parent, as do the forget and gate down
// projections with beta; one causal convolution BF16 [taps, 3 * width] holds q's, k's and v's
// kernels in the parent's channel order.
struct KdaWeights {
    WeightId query, key, value;
    WeightId convolution;
    WeightId forget_down, forget_up, gate_down, gate_up, beta;
    WeightId a_log, dt_bias; // FP32 [heads], FP32 [width]
    WeightId output_norm, output;
};

// Sparse MLA with its DSA indexer. key_up and value_up are kv_b_proj's per-head W_uk (transposed)
// and W_uv, the BF16 operands of the absorbed MLA Ops; query_down shares a parent with latent, and index_key with index_gate and
// index_weights.
struct MlaWeights {
    WeightId query_down, query_norm, query_up;
    WeightId latent, latent_norm, key_up, value_up, output;
    WeightId index_query, index_key, index_key_norm, index_key_bias;
    WeightId index_gate, index_position, index_weights;
};

struct MlpWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    WeightId router;      // BF16 [experts, H], FP32 logits
    WeightId router_bias; // FP32 [experts]: e_score_correction_bias, selection only
    MlpWeights shared;    // the ungated shared expert
    WeightId experts;     // nvfp4_mul bank in nvfp4_expert_rg16_v1 (W4A16), pinned host memory
};

struct BlockWeights {
    HyperConnectionWeights attn_hc, ffn_hc;
    WeightId input_norm, post_attention_norm;
    std::variant<KdaWeights, MlaWeights> mixer;
    std::variant<MlpWeights, MoeWeights> ffn;
};

// One DFlash2 layer (docs/maintainer/dflash.md): its dynamic convolutions' base kernels BF16
// [2 sides, taps, H] and coefficient projections, the query block's Q/K/V parent and the context
// K/V (the converter shares the K/V words between the two roles), and the gate/up parent.
struct DFlash2LayerWeights {
    WeightId input_norm, post_attention_norm;
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
    WeightId gate, up, down;
    WeightId attention_base, attention_projection, mlp_base, mlp_projection;
};

// The DFlash2 drafter of a companion artifact (--draft-model); it shares the token embedding and
// the head with the text model.
struct DFlash2Weights {
    WeightId feature_projection; // [H, taps * H]
    WeightId context_norm, final_norm;
    std::vector<DFlash2LayerWeights> layers;
    WeightId selector_hidden;                          // [rank, H]
    WeightId predecessor_codebook, successor_codebook; // BF16 [vocab, rank]
};

struct TextWeights {
    WeightId token_embedding; // BF16, pinned host memory
    WeightId final_norm;      // over the stream mean
    WeightId output_head;
    std::vector<BlockWeights> layers;
    std::optional<DFlash2Weights> dflash2; // bound from the companion artifact
};

} // namespace infernix::models::glm5_next
