#pragma once

// Native execution parameters of a loaded Glm5Next model: one Weight per fused projection (the
// converter packs every projection group that shares an input into one parent), BF16 vectors for
// norms and convolutions, and each MoE layer's expert source.

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/weight_input.h"
#include "models/glm5_next/model.h"

#include <optional>
#include <variant>
#include <vector>

namespace infernix::models::glm5_next::execution {

struct LinearParameters {
    Weight weight;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

// One mHC mixer (ops::mhc_mix).
struct HyperConnectionParameters {
    Tensor fn;    // BF16 [S*H, (2+S)S]: one row of the 24 per column
    Tensor base;  // FP32 [(2+S)S]
    Tensor scale; // FP32 [3]
};

struct KdaParameters {
    LinearParameters qkv;     // [q | k | v, H]
    LinearParameters control; // [forget down | gate down | beta, H]
    Tensor convolution;       // BF16 [3 * width, taps]: q | k | v channels
    LinearParameters forget_up, gate_up; // [width, head_dim]
    Tensor a_log;   // FP32 [heads]
    Tensor dt_bias; // FP32 [width]
    Tensor output_norm; // BF16 [head_dim]
    LinearParameters output; // [H, width]
};

struct MlaParameters {
    LinearParameters query_latent; // [q_lora | kv_lora, H]
    Tensor query_norm;             // BF16 [q_lora]
    Tensor latent_norm;            // BF16 [kv_lora]
    LinearParameters query_up;     // [heads * head_dim, q_lora]
    LinearParameters index_query;  // [index heads * index dim, q_lora]
    Tensor key_up;                 // BF16 [head_dim, kv_lora, heads]: W_uk transposed per head
    Tensor value_up;               // BF16 [kv_lora, value_dim, heads]: W_uv per head
    LinearParameters output;       // [H, heads * value_dim]
    LinearParameters index_inputs; // [index key | index gate | index weights, H]
    Tensor index_key_norm, index_key_bias; // BF16 [index dim]: the keys' LayerNorm
    Tensor index_position;                 // FP32 [index dim, pool]: the pooling's APE
};

struct MlpParameters {
    LinearParameters gate_up; // [2I, H]: gate rows, then up rows
    LinearParameters down;    // [H, I]
};

struct MoeParameters {
    Tensor router;       // BF16 [H, E]: one expert row per column (FP32 logits)
    Tensor router_bias;  // FP32 [E]
    MlpParameters shared; // the ungated shared expert
    const ExpertBank* bank = nullptr;
    const ops::offloaded_moe::ExpertScales* device_scales = nullptr; // [E]
};

// One DFlash2 layer (docs/maintainer/dflash.md "DFlash2 dynamic convolution").
struct DFlash2LayerParameters {
    Tensor input_norm, post_attention_norm;      // BF16 [H]
    Weight query_key_value;                      // [q | k | v, H] for ops::attn_input_proj
    LinearParameters context_key, context_value; // [kv, H] (the K/V words, shared)
    Tensor query_norm, key_norm;                 // BF16 [head_dim]
    LinearParameters output;                     // [H, q]
    LinearParameters gate_up, down;              // [2I, H], [H, I]
    Tensor attention_base, mlp_base;             // BF16 [H, taps, sides]
    LinearParameters attention_projection, mlp_projection; // [2 * taps * H / group, H]
};

// The DFlash2 drafter of a companion artifact (--draft-model).
struct DFlash2Parameters {
    ops::PreparedRope rope;              // full-head 1-D, the drafter's theta
    LinearParameters feature_projection; // [H, taps * H]
    Tensor context_norm, final_norm;     // BF16 [H]
    std::vector<DFlash2LayerParameters> layers;
    LinearParameters selector_hidden;                // [rank, H]
    Tensor predecessor_codebook, successor_codebook; // BF16 [rank, vocab]
};

// One vision block (VisionBlockWeights): the MLP's gate/up parent with its biases joined.
struct VisionBlockParameters {
    Tensor norm1, norm2;                     // BF16 [H]
    LinearParameters query, key, value;      // [H, H] each
    Tensor query_bias, key_bias, value_bias; // BF16 [H]
    Tensor query_norm, key_norm;             // BF16 [head_dim]
    LinearParameters output;                 // [H, H]
    Tensor output_bias;                      // BF16 [H]
    LinearParameters gate_up;                // [2I, H]: gate rows, then up rows
    Tensor gate_up_bias;                     // BF16 [2I]
    LinearParameters down;                   // [H, I]
    Tensor down_bias;                        // BF16 [H]
};

// The vision tower (execution/vision.h).
struct VisionParameters {
    LinearParameters patch_embedding; // [H, patch_width]
    Tensor patch_embedding_bias;      // BF16 [H]
    std::vector<VisionBlockParameters> layers;
    Tensor post_norm;                     // BF16 [H]
    LinearParameters downsample;          // [O, m*m*H]: patch-major (the loader's permutation)
    Tensor downsample_bias;               // BF16 [O]
    LinearParameters merger_projection;   // [O, O]
    Tensor merger_norm, merger_norm_bias; // BF16 [O]: LayerNorm
    LinearParameters merger_gate_up;      // [2PI, O]
    LinearParameters merger_down;         // [O, PI]
};

struct BlockParameters {
    HyperConnectionParameters attn_hc, ffn_hc;
    Tensor input_norm, post_attention_norm; // BF16 [H]
    std::variant<KdaParameters, MlaParameters> mixer;
    std::variant<MlpParameters, MoeParameters> ffn;
};

class Parameters {
public:
    explicit Parameters(const Model& model);
    ~Parameters();
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;

    const Model& model;
    Weight token_embedding; // BF16 rows in pinned host memory, read zero-copy
    Tensor final_norm;      // BF16 [H]
    Tensor output_head;     // BF16 [H, V]: one vocabulary row per column (FP32 logits)
    std::vector<BlockParameters> layers;
    std::optional<DFlash2Parameters> dflash2;
    std::optional<VisionParameters> vision;

private:
    // Device copy of every MoE layer's ExpertScales, [MoE layers][E].
    DeviceBuffer scales_;
    // The vision blocks' joined gate/up biases, [layers][2I].
    DeviceBuffer vision_biases_;
};

} // namespace infernix::models::glm5_next::execution
