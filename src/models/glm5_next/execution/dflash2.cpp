#include "models/glm5_next/execution/dflash2.h"
#include "models/glm5_next/execution/forward.h"

#include "core/nvtx.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/context_kv_materialize.h"
#include "infernix/ops/dynamic_grouped_conv.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/prepare_masked_block.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rmsnorm_pack_tail.h"
#include "infernix/ops/rmsnorm_rope.h"
#include "infernix/ops/sliding_window_attention.h"
#include "infernix/ops/top_logprobs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::models::glm5_next::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

// The drafter's top-16 candidates per position (the selector's lattice).
constexpr std::int32_t kCandidates = 16;

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work,
             cudaStream_t s) {
    ops::linear(x, p.weight, out, p.policy, work, s);
}

// The norm and dynamic-conv prepare in an arena of its own carved from `work`; a call that needs no
// scratch runs in `work` itself, as an arena cannot be empty.
void prepare(const Tensor& residual, const Tensor& norm, float eps, const Tensor& base,
             const Weight& projection, Tensor& prepared, Tensor& delta, WorkspaceArena& work,
             cudaStream_t s) {
    const std::size_t bytes = ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
        residual.ne[0], residual.ne[1], residual.ne[1], residual.ne[2], residual.ne[2]);
    if (bytes == 0) {
        ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, eps, base, projection, prepared,
                                                  delta, work, s);
        return;
    }
    WorkspaceArena scratch(work.alloc_bytes(bytes));
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, eps, base, projection, prepared,
                                              delta, scratch, s);
}

// The fused Ops take Q8_G32_FP16 RowSplit projections (the companion recipe's).
void require_q8(const Weight& weight, const char* name) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(
            std::string("Glm5Next DFlash2: ") + name +
            " must be q8_g32_fp16 (tools.convert recipe glm5_3_flash_dflash2)");
    }
}

} // namespace

DFlash2::DFlash2(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
                 std::vector<CyclicKVCacheLayerView> ring, std::uint32_t max_context,
                 std::uint32_t* error)
    : parameters_(parameters), p_(parameters.dflash2.value()),
      config_(*parameters.model.config().dflash2), text_(parameters.model.config().text),
      device_(device), work_(work), ring_(std::move(ring)), max_context_(max_context),
      error_(error) {
    if (ring_.size() != config_.layers || p_.layers.size() != config_.layers) {
        throw std::invalid_argument("Glm5Next DFlash2: one context ring layer per drafter layer");
    }
    require_q8(p_.feature_projection.weight, "the feature projection");
    for (const auto& layer : p_.layers) {
        require_q8(layer.query_key_value, "the query/key/value projections");
        require_q8(layer.context_key.weight, "the context key projection");
        require_q8(layer.context_value.weight, "the context value projection");
        require_q8(layer.output.weight, "the attention output projection");
        require_q8(layer.gate_up.weight, "the MLP gate/up projections");
        require_q8(layer.down.weight, "the MLP down projection");
    }
    // The fused Ops' contract (context_kv_materialize: five local layers; the selector's 16
    // candidates; two-tap convolutions of 16-channel groups).
    if (config_.layers != ops::kContextKVMaterializeLayers ||
        config_.selector_top_k != static_cast<std::uint32_t>(kCandidates) ||
        config_.conv_kernel != 2 || config_.conv_group != 16) {
        throw std::invalid_argument(
            "Glm5Next DFlash2: the drafter is outside the fused Ops' registered geometry");
    }
}

void DFlash2::materialize(const DFlash2ContextCall& call) {
    const cudaStream_t s = device_.stream;
    const std::int32_t H = dim(text_.hidden_size), W = call.features.ne[1], B = call.features.ne[2];
    if (call.features.dtype != DType::BF16 ||
        call.features.ne[0] != dim(config_.feature_width(text_.hidden_size)) || W <= 0 || B <= 0 ||
        call.max_count > static_cast<std::uint32_t>(W) || call.min_count > call.max_count) {
        throw std::invalid_argument("Glm5Next DFlash2: context call geometry is invalid");
    }
    const nvtx::ScopedRange range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(W * B));
    work_.reset();
    Tensor projected = work_.alloc(DType::BF16, {H, W * B});
    project(call.features.view({call.features.ne[0], W * B}), p_.feature_projection, projected,
            work_, s);
    Tensor context = work_.alloc(DType::BF16, {H, W, B});
    Tensor flat    = context.view({H, W * B});
    ops::rmsnorm(projected, p_.context_norm, config_.rms_norm_eps, false, flat, s);
    std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = p_.layers[l];
        layers[l]         = {layer.context_key.weight, layer.context_value.weight, layer.key_norm,
                             ring_[l]};
    }
    ops::context_kv_materialize(context, call.positions, call.counts, call.slots, layers,
                                config_.rope_theta, config_.rms_norm_eps,
                                {call.min_count, call.max_count}, work_, s);
}

void DFlash2::propose(const DFlash2ProposalCall& call) {
    const cudaStream_t s = device_.stream;
    const std::int32_t H = dim(text_.hidden_size), W = call.width, B = call.batch, K = W - 1;
    const std::int32_t D = dim(config_.head_dim), Hq = dim(config_.heads),
                       Hk = dim(config_.kv_heads), I = dim(config_.intermediate);
    const std::int32_t G = H / dim(config_.conv_group), columns = W * B;
    if (W < 2 || W > 16 || B < 1 || B > static_cast<std::int32_t>(kMaximumConcurrency) ||
        call.configs == nullptr) {
        throw std::invalid_argument("Glm5Next DFlash2: proposal geometry is invalid");
    }
    const nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(columns));
    work_.reset();
    Tensor ids       = call.ids;
    Tensor positions = call.positions;
    ops::prepare_masked_block(call.anchors, call.frontiers, call.valid_columns,
                              dim(config_.mask_token_id), ids, positions, s);
    Tensor residual = work_.alloc(DType::BF16, {H, W, B});
    {
        Tensor flat = residual.view({H, columns});
        ops::embedding(ids.view({columns}), parameters_.token_embedding, flat, s,
                       DeviceFault{.word = error_, .value = kErrorTokenId});
    }
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{0, max_context_};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    for (std::size_t l = 0; l < p_.layers.size(); ++l) {
        const auto& layer = p_.layers[l];
        const nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash, l);
        {
            // Attention: prepare (norm, dynamic conv), Q/K/V, Q/K norm and RoPE, attention over the
            // ring and the block, then the output projection, conv finish and residual add.
            auto scope      = work_.scope();
            Tensor prepared = work_.alloc(DType::BF16, {H, W, B});
            Tensor delta    = work_.alloc(DType::BF16, {G, 2, W, B});
            prepare(residual, layer.input_norm, config_.rms_norm_eps, layer.attention_base,
                    layer.attention_projection.weight, prepared, delta, work_, s);
            Tensor q = work_.alloc(DType::BF16, {D, Hq, W, B});
            Tensor k = work_.alloc(DType::BF16, {D, Hk, W, B});
            Tensor v = work_.alloc(DType::BF16, {D, Hk, W, B});
            {
                Tensor qf = q.view({D * Hq, columns}), kf = k.view({D * Hk, columns}),
                       vf = v.view({D * Hk, columns});
                ops::attn_input_proj(prepared.view({H, columns}), layer.query_key_value, qf, kf, vf,
                                     s);
            }
            ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, p_.rope,
                              config_.rms_norm_eps, q, k, s);
            Tensor attention = work_.alloc(DType::BF16, {D, Hq, W, B});
            ops::sliding_window_attention(q, k, v, positions, call.valid_columns, call.slots,
                                          {D, Hq, Hk}, config_.window, scale, ring_[l], envelope,
                                          work_, attention, s);
            WorkspaceArena scratch(
                work_.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                    H, D * Hq, W, W, B, B)));
            ops::linear_dynamic_grouped_conv_add(attention.view({D * Hq, W, B}),
                                                 layer.output.weight, layer.attention_base, delta,
                                                 residual, scratch, s);
        }
        {
            // MLP: prepare, SwiGLU(gate, up), then down, conv finish and residual add.
            auto scope      = work_.scope();
            Tensor prepared = work_.alloc(DType::BF16, {H, W, B});
            Tensor delta    = work_.alloc(DType::BF16, {G, 2, W, B});
            prepare(residual, layer.post_attention_norm, config_.rms_norm_eps, layer.mlp_base,
                    layer.mlp_projection.weight, prepared, delta, work_, s);
            Tensor product = work_.alloc(DType::BF16, {I, W, B});
            {
                Tensor flat = product.view({I, columns});
                ops::linear_swiglu(prepared.view({H, columns}), layer.gate_up.weight, flat,
                                   layer.gate_up.policy, work_, s);
            }
            WorkspaceArena scratch(work_.alloc_bytes(
                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(H, I, W, W, B, B)));
            ops::linear_dynamic_grouped_conv_add(product, layer.down.weight, layer.mlp_base, delta,
                                                 residual, scratch, s);
        }
    }
    // The mask columns' final hidden, the BF16 head's FP32 logits and their top log-probabilities
    // (a per-column shift of the logits, to which the selector's softmax and argmax are invariant),
    // then the selector's conditional path.
    Tensor hidden = work_.alloc(DType::BF16, {H, K * B});
    ops::rmsnorm_pack_tail(residual, p_.final_norm, config_.rms_norm_eps, hidden, s);
    Tensor candidates = call.candidates;
    Tensor ids_flat   = candidates.view({kCandidates, K * B});
    Tensor scores     = work_.alloc(DType::FP32, {kCandidates, K * B});
    {
        auto scope           = work_.scope();
        Tensor logits        = work_.alloc(DType::FP32, {dim(text_.vocab_size), K * B});
        const Tensor* rows[] = {&parameters_.output_head};
        ops::projection_fp32(hidden, rows, logits, s);
        ops::top_logprobs(logits, dim(parameters_.model.resources().public_token_count), ids_flat,
                          scores, s);
    }
    Tensor projected = work_.alloc(DType::BF16, {dim(config_.selector_rank), K * B});
    project(hidden, p_.selector_hidden, projected, work_, s);
    Tensor drafts     = call.drafts;
    Tensor q          = call.proposal_q;
    Tensor confidence = call.confidence;
    ops::candidate_selector_path(candidates, scores.view({kCandidates, K, B}),
                                 projected.view({dim(config_.selector_rank), K, B}), call.anchors,
                                 p_.predecessor_codebook, p_.successor_codebook, call.frontiers,
                                 call.configs, drafts, q,
                                 confidence.data != nullptr ? &confidence : nullptr, work_, s);
}

std::size_t DFlash2::workspace_bytes(const TextConfig& text, const DFlash2Config& config,
                                     std::int32_t max_batch, std::int32_t max_width,
                                     std::int32_t context_columns, std::uint32_t max_context) {
    constexpr auto kQ8           = QType::Q8_G32_FP16;
    constexpr auto kA16          = ops::LinearPolicy::A16Only;
    constexpr std::size_t kSlack = 256; // per allocation
    const std::size_t H = text.hidden_size, I = config.intermediate, V = text.vocab_size;
    const std::size_t q  = static_cast<std::size_t>(config.heads) * config.head_dim;
    const std::size_t kv = static_cast<std::size_t>(config.kv_heads) * config.head_dim;
    const std::size_t G = H / config.conv_group, F = config.feature_width(text.hidden_size);
    // materialize: the projected and normalized context, the projection and the materializer, for
    // the batched profile (up to kBatchedContextColumns per row) and one row's widest call.
    std::size_t context = 0;
    for (const std::int32_t columns :
         {std::min(context_columns, kBatchedContextColumns), context_columns}) {
        const std::int32_t rows = columns <= kBatchedContextColumns ? max_batch : 1;
        const std::size_t cells =
            static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows);
        context = std::max(
            context,
            2 * (H * cells * 2 + kSlack) +
                ops::linear_workspace_capacity_bytes(kQ8, dim(H), dim(F), kA16, 1, dim(cells)) +
                ops::context_kv_materialize_workspace_capacity_bytes(rows, 1, columns));
    }
    // propose at the widest block (every buffer grows with W and B): the residual, one branch's
    // buffers and scratch, then the head's logits and the selector.
    const std::size_t W     = static_cast<std::size_t>(max_width),
                      B     = static_cast<std::size_t>(max_batch);
    const std::size_t cells = W * B, masks = (W - 1) * B;
    const std::size_t branch = (H * cells * 2 + kSlack) + (G * 2 * cells * 2 + kSlack);
    std::size_t prepare = 0, add_attention = 0, add_mlp = 0, window = 0;
    for (std::int32_t w = 2; w <= max_width; ++w) {
        for (std::int32_t b = 1; b <= max_batch; ++b) {
            prepare = std::max(prepare,
                               ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                   dim(H), w, w, b, b));
            add_attention = std::max(add_attention,
                                     ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                         dim(H), dim(q), w, w, b, b));
            add_mlp =
                std::max(add_mlp, ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                      dim(H), dim(I), w, w, b, b));
            window = std::max(window,
                              ops::sliding_window_attention_workspace_capacity_bytes(
                                  {dim(config.head_dim), dim(config.heads), dim(config.kv_heads)},
                                  config.window, {0, max_context}, w, w, b));
        }
    }
    const std::size_t swiglu =
        ops::linear_swiglu_workspace_capacity_bytes(kQ8, dim(2 * I), dim(H), kA16, 1, dim(cells));
    const std::size_t attention = branch + (q + 2 * kv) * cells * 2 + 3 * kSlack +
                                  (q * cells * 2 + kSlack) + std::max(prepare, window) +
                                  add_attention + 2 * kSlack;
    const std::size_t mlp =
        branch + (I * cells * 2 + kSlack) + std::max(prepare, swiglu) + add_mlp + 2 * kSlack;
    const std::size_t head = V * masks * 4 + kSlack;
    std::size_t selector =
        ops::candidate_selector_path_workspace_capacity_bytes(1, max_width - 1, 1, max_batch);
    for (const QType format : {kQ8, QType::BF16}) {
        try {
            selector = std::max(
                selector, ops::linear_workspace_capacity_bytes(format, dim(config.selector_rank),
                                                               dim(H), kA16, 1, dim(masks)));
        } catch (const std::invalid_argument&) {}
    }
    const std::size_t tail = (H * masks * 2 + kSlack) + (kCandidates * masks * 4 + kSlack) +
                             (config.selector_rank * masks * 2 + kSlack) + std::max(head, selector);
    const std::size_t propose = (H * cells * 2 + kSlack) + std::max({attention, mlp, tail});
    return std::max(context, propose) + (1ULL << 20); // alignment slack
}

} // namespace infernix::models::glm5_next::execution
