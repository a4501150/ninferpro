#include "models/glm5_next/execution/forward.h"

#include "infernix/ops/causal_conv1d_silu.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/kimi_delta_attention.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/mhc.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rows.h"
#include "infernix/ops/scatter.h"
#include "infernix/ops/silu_mul.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::glm5_next::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work, cudaStream_t s) {
    ops::linear(x, p.weight, out, p.policy, work, s);
}

// project() with the output's consecutive row ranges written to `parts`.
void project_rows(const Tensor& x, const LinearParameters& p, std::span<Tensor* const> parts, WorkspaceArena& work,
                  cudaStream_t s) {
    ops::linear_rows(x, p.weight, parts, p.policy, work, s);
}

ops::offloaded_moe::Geometry expert_geometry(const TextConfig& c) {
    return {.hidden = dim(c.hidden_size), .intermediate = dim(c.moe.intermediate), .swiglu_limit = c.swiglu_limit};
}

constexpr std::size_t kAlign = 256; // the arena's per-allocation alignment

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// Prefill calls of at least this many columns attend on the tensor-core route.
constexpr std::int32_t kPrefillAttentionWidth = 64;

} // namespace

ops::MlaAttentionGeometry mla_geometry(const TextConfig& c) {
    return {.heads = dim(c.mla.heads), .head_dim = dim(c.mla.head_dim), .latent_dim = dim(c.mla.kv_lora)};
}

ops::DsaIndexerGeometry indexer_geometry(const TextConfig& c) {
    return {.heads     = dim(c.indexer.heads),
            .head_dim  = dim(c.indexer.head_dim),
            .pool      = dim(c.indexer.pool),
            .top_pools = dim(c.indexer.pool_budget()),
            .eps       = 1e-6F};
}

Forward::Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardState state,
                 ForwardKV kv, ForwardExperts experts, std::int32_t max_context)
    : parameters_(parameters), config_(parameters.model.config().text), device_(device), work_(work),
      state_(std::move(state)), kv_(std::move(kv)), experts_(std::move(experts)), max_context_(max_context) {
    if (state_.kda == nullptr || state_.kda->layer_count() != config_.kda_layers ||
        kv_.layers.size() != config_.mla_layers || experts_.frames.size() != config_.moe_layers()) {
        throw std::invalid_argument("Glm5Next forward: state, caches or expert residency is incomplete");
    }
    if (const auto& dflash2 = parameters_.model.config().dflash2) { draft_taps_ = dflash2->target_layer_ids; }
}

std::size_t Forward::workspace_bytes(const TextConfig& c, std::int32_t columns, std::int32_t max_context) {
    // The residual, the block input and the mixer's post/comb live across the call; each layer
    // half runs in its own scope, so the peak is those plus the largest half (and the head).
    const std::size_t t = static_cast<std::size_t>(std::max(columns, 1));
    const std::size_t h = c.hidden_size, s = c.hc.streams, bf = 2, f32 = 4;
    const std::size_t mix_ws =
        aligned(ops::mhc_mix_workspace_capacity_bytes(dim(s), dim(h), 1, static_cast<std::int32_t>(t)));
    const std::size_t across = aligned(s * h * t * bf) + aligned(h * t * bf) + aligned(s * t * f32) +
                               aligned(s * s * t * f32) + mix_ws;
    const std::size_t xn = aligned(h * t * bf); // the normed block input
    const std::size_t y  = aligned(h * t * bf);

    const std::size_t w = c.kda.width(), d = c.kda.head_dim;
    const std::size_t kda = aligned(3 * w * t * bf) + 2 * aligned(d * t * bf) + aligned(c.kda.heads * t * bf) +
                            2 * aligned(w * t * bf) + 3 * aligned(w * t * bf) + aligned(w * t * bf) +
                            aligned(w * t * bf) + y +
                            aligned(ops::kimi_delta_attention_workspace_capacity_bytes(
                                dim(c.kda.heads), dim(c.kda.heads), 1, static_cast<std::int32_t>(t)));

    const auto& m = c.mla;
    const auto mg = mla_geometry(c);
    const auto ig = indexer_geometry(c);
    const std::size_t index_rows = 2 * static_cast<std::size_t>(c.indexer.head_dim) + c.indexer.heads;
    const std::size_t mla =
        2 * aligned(m.q_lora * t * bf) + 2 * aligned(m.kv_lora * t * bf) +
        aligned(m.query_width() * t * bf) +
        aligned(static_cast<std::size_t>(c.indexer.heads) * c.indexer.head_dim * t * bf) +
        aligned(index_rows * t * bf) + aligned(static_cast<std::size_t>(ig.top_pools) * t * 4) +
        aligned(t * 4) + aligned(static_cast<std::size_t>(m.kv_lora) * m.heads * t * bf) +
        aligned(m.value_width() * t * bf) + y +
        aligned(ops::dsa_indexer_select_workspace_bytes(ig, static_cast<std::int32_t>(t),
                                                        max_context)) +
        aligned(std::max(ops::mla_sparse_attention_workspace_bytes(mg, static_cast<std::int32_t>(t),
                                                                   max_context, {}),
                         ops::mla_sparse_attention_workspace_bytes(mg, static_cast<std::int32_t>(t),
                                                                   max_context,
                                                                   {.tensor_cores = true})));

    const auto mlp_bytes = [&](std::size_t intermediate) { return 3 * aligned(intermediate * t * bf) + y; };
    const std::size_t e = c.moe.experts, k = c.moe.top_k;
    const auto entries  = static_cast<std::int32_t>(k * t);
    const std::int32_t max_jobs = std::min(static_cast<std::int32_t>(e), entries);
    const std::size_t moe =
        aligned(e * t * f32) + aligned(k * t * 4) + aligned(k * t * f32) + aligned(t * f32) +
        aligned(ops::moe_dispatch_bytes(static_cast<std::int32_t>(e), entries)) + aligned(h * k * t * bf) +
        aligned(std::max(ops::moe_experts_workspace_bytes(max_jobs, entries, static_cast<std::int32_t>(t),
                                                          ops::offloaded_moe::ExpertActivation::kA16,
                                                          ops::offloaded_moe::WideA16::kExact, expert_geometry(c)),
                         ops::moe_experts_workspace_bytes(max_jobs, entries, static_cast<std::int32_t>(t),
                                                          ops::offloaded_moe::ExpertActivation::kA16,
                                                          ops::offloaded_moe::WideA16::kGgmlBf16, expert_geometry(c)))) +
        mlp_bytes(c.moe.shared_intermediate) + y;
    const std::size_t head = aligned(s * h * t * bf) + 2 * aligned(h * t * bf);

    const std::size_t phase =
        xn + std::max({kda, mla, mlp_bytes(c.dense_intermediate), moe, head});
    // The BF16 projections' split-K partials and alignment slack.
    constexpr std::size_t kLinearSlack = 64ULL << 20;
    return across + phase + kLinearSlack;
}

void Forward::run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap) {
    if (logits.dtype != DType::FP32 || logits.ne[0] != dim(config_.vocab_size) ||
        logits.ne[1] != batch.logit_columns.ne[0]) {
        throw std::invalid_argument("Glm5Next forward: batch geometry is invalid");
    }
    begin_call(batch);
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size), S = dim(config_.hc.streams);
    CallColumns columns{.residual = work_.alloc(DType::BF16, {dim(config_.residual_width()), T}),
                        .x        = work_.alloc(DType::BF16, {H, T}),
                        .post     = work_.alloc(DType::FP32, {S, T}),
                        .comb     = work_.alloc(DType::FP32, {S, S, T})};
    embed(batch, columns);
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        step(l, batch, columns, {}, tap);
    }
    finish(batch, columns, &logits);
}

void Forward::begin_call(const ForwardBatch& batch) {
    const std::int32_t T = batch.ids.ne[0];
    const bool verify    = batch.verify != nullptr;
    if (batch.batch <= 0 || batch.width <= 0 || batch.batch * batch.width != T ||
        (batch.width > 1 && batch.batch != 1 && !verify) || batch.batch > 8 ||
        batch.positions.ne[0] != T || batch.slots.ne[0] != batch.batch ||
        batch.table_rows.ne[0] != batch.batch ||
        (batch.width > 1 && !verify && batch.host_slots.size() != 1) ||
        (verify && (batch.width > 16 || batch.verify->size() != config_.kda_layers))) {
        throw std::invalid_argument("Glm5Next forward: batch geometry is invalid");
    }
    // A verify round runs like a decode round (the expert fork and landing frames).
    eager_ = batch.width > 1 && !verify;
    work_.reset();
}

void Forward::embed(const ForwardBatch& batch, CallColumns& columns) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size),
                       S = dim(config_.hc.streams);
    {
        auto scope = work_.scope();
        Tensor x0  = work_.alloc(DType::BF16, {H, T});
        ops::embedding(batch.ids, parameters_.token_embedding, x0, s,
                       DeviceFault{.word = experts_.error, .value = kErrorTokenId});
        if (batch.vision_columns.data != nullptr) {
            ops::scatter(batch.vision_embeddings, batch.vision_columns, x0, s);
        }
        ops::hyper_connection_expand(x0, S, columns.residual, s);
    }
    mix(parameters_.layers[0].attn_hc, columns.residual, columns.x, columns.post, columns.comb);
}

void Forward::step(std::uint32_t l, const ForwardBatch& batch, CallColumns& columns,
                   StreamUse stream, const ForwardTap* tap) {
    const std::int32_t T = batch.ids.ne[0];
    stream_              = stream;
    try {
        layer(l, batch, columns.residual, columns.post, columns.comb, columns.x, tap);
    } catch (const std::exception& error) {
        throw std::runtime_error("glm5_next/layers/" + std::to_string(l) + ": " + error.what());
    }
    if (batch.draft_features.data != nullptr) {
        const std::int32_t n = batch.draft_features.ne[1];
        if (n > T) {
            throw std::invalid_argument("Glm5Next forward: DFlash2 features exceed the call");
        }
        capture_draft_features(l, columns.residual.slice(1, T - n, n), batch.draft_features);
    }
    if (tap != nullptr && tap->residuals != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(tap->residuals->at(l).data, columns.residual.data,
                                   columns.residual.bytes(), cudaMemcpyDeviceToDevice,
                                   device_.stream));
    }
}

void Forward::finish(const ForwardBatch& batch, const CallColumns& columns, Tensor* logits) {
    const cudaStream_t s   = device_.stream;
    const Tensor& residual = columns.residual;
    if (batch.residual_out.data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(batch.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    if (batch.head_residual_out.data != nullptr) {
        // The call's last column, as the head reads it (the saved column is BF16 [S*H]).
        const std::size_t width = static_cast<std::size_t>(dim(config_.residual_width()));
        CUDA_CHECK(cudaMemcpyAsync(batch.head_residual_out.data,
                                   static_cast<std::byte*>(residual.data) +
                                       (static_cast<std::size_t>(residual.ne[1]) - 1U) * width * 2U,
                                   width * 2U, cudaMemcpyDeviceToDevice, s));
    }
    if (logits == nullptr) { return; }
    auto scope     = work_.scope();
    Tensor gathered =
        work_.alloc(DType::BF16, {dim(config_.residual_width()), batch.logit_columns.ne[0]});
    ops::gather_columns(residual, batch.logit_columns, gathered, s);
    head(gathered, *logits);
}

void Forward::head_logits(const Tensor& residuals, Tensor& logits) {
    if (residuals.dtype != DType::BF16 || residuals.ne[0] != dim(config_.residual_width()) ||
        logits.dtype != DType::FP32 || logits.ne[0] != dim(config_.vocab_size) || logits.ne[1] != residuals.ne[1]) {
        throw std::invalid_argument("Glm5Next forward: head logits need BF16 [S*H, n] residuals and FP32 [V, n] logits");
    }
    work_.reset();
    head(residuals, logits);
}

void Forward::head(const Tensor& residual, Tensor& logits) {
    const cudaStream_t s = device_.stream;
    const std::int32_t n = residual.ne[1], H = dim(config_.hidden_size);
    auto scope  = work_.scope();
    Tensor mean = work_.alloc(DType::BF16, {H, n});
    Tensor xf   = work_.alloc(DType::BF16, {H, n});
    ops::hyper_connection_stream_mean(residual, dim(config_.hc.streams), mean, s);
    ops::rmsnorm(mean, parameters_.final_norm, config_.rms_norm_eps, /*unit_offset=*/false, xf, s);
    const Tensor* rows[] = {&parameters_.output_head};
    ops::projection_fp32(xf, rows, logits, s);
}

void Forward::mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor& x, Tensor& post, Tensor& comb) {
    ops::mhc_mix(residual, p.fn, p.base, p.scale, dim(config_.hc.streams), dim(config_.hc.sinkhorn_iters),
                 config_.rms_norm_eps, config_.hc.eps, x, &post, &comb, work_, device_.stream);
}

void Forward::update_mix(const HyperConnectionParameters* next, const Tensor& y, Tensor& residual, Tensor& x,
                         Tensor& post, Tensor& comb) {
    if (next == nullptr) {
        ops::mhc_update(y, post, comb, residual, residual, device_.stream);
        return;
    }
    ops::mhc_update_mix(y, post, comb, residual, residual, next->fn, next->base, next->scale,
                        dim(config_.hc.streams), dim(config_.hc.sinkhorn_iters), config_.rms_norm_eps,
                        config_.hc.eps, x, &post, &comb, work_, device_.stream);
}

void Forward::layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, Tensor& post, Tensor& comb,
                    Tensor& x, const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const auto& block    = parameters_.layers[layer];
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    {
        // x is the attention mix of the residual (the previous update's fused mix).
        auto scope = work_.scope();
        Tensor xa  = work_.alloc(DType::BF16, {H, T});
        ops::rmsnorm(x, block.input_norm, config_.rms_norm_eps, /*unit_offset=*/false, xa, s);
        const Tensor y = config_.layer_types[layer] == MixerKind::Kda
                             ? kda(std::get<KdaParameters>(block.mixer), xa, compact, batch)
                             : mla(std::get<MlaParameters>(block.mixer), xa, compact, batch);
        update_mix(&block.ffn_hc, y, residual, x, post, comb);
    }
    {
        auto scope = work_.scope();
        Tensor xm  = work_.alloc(DType::BF16, {H, T});
        ops::rmsnorm(x, block.post_attention_norm, config_.rms_norm_eps, /*unit_offset=*/false, xm, s);
        Tensor y;
        if (const auto* dense = std::get_if<MlpParameters>(&block.ffn)) {
            y = mlp(*dense, xm);
        } else {
            const std::uint32_t moe_layer = layer - config_.dense_layers;
            Tensor* route_tap = tap != nullptr && tap->routes != nullptr ? &tap->routes->at(moe_layer) : nullptr;
            y = moe(std::get<MoeParameters>(block.ffn), xm, moe_layer, route_tap);
        }
        const bool last = layer + 1 == config_.num_hidden_layers;
        update_mix(last ? nullptr : &parameters_.layers[layer + 1].attn_hc, y, residual, x, post, comb);
    }
}

void Forward::capture_draft_features(std::uint32_t layer, const Tensor& residual, const Tensor& features) {
    const auto it = std::find(draft_taps_.begin(), draft_taps_.end(), layer);
    if (it == draft_taps_.end()) { return; }
    const std::int32_t H = dim(config_.hidden_size);
    if (features.ne[1] != residual.ne[1] || features.ne[0] != H * static_cast<std::int32_t>(draft_taps_.size())) {
        throw std::invalid_argument("Glm5Next forward: DFlash2 features do not match the residual");
    }
    Tensor out = features.slice(0, static_cast<std::int32_t>(it - draft_taps_.begin()) * H, H);
    ops::hyper_connection_stream_mean(residual, dim(config_.hc.streams), out, device_.stream);
}

Tensor Forward::kda(const KdaParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& k        = config_.kda;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t W = dim(k.width()), NH = dim(k.heads), D = dim(k.head_dim), C = 3 * W;
    // A verify call writes the inputs its commit folds into the block's records.
    const KdaVerifyRecord* record = batch.verify != nullptr ? &batch.verify->at(index) : nullptr;
    const auto take               = [&](const Tensor KdaVerifyRecord::* field, std::int32_t rows) {
        return record != nullptr ? (record->*field).slice(1, 0, T)
                                 : work_.alloc(DType::BF16, {rows, T});
    };
    Tensor qkv = take(&KdaVerifyRecord::qkv, C);
    project(x, p.qkv, qkv, work_, s);
    Tensor forget_down = work_.alloc(DType::BF16, {D, T});
    Tensor gate_down   = work_.alloc(DType::BF16, {D, T});
    Tensor beta        = take(&KdaVerifyRecord::beta, NH);
    {
        Tensor* parts[] = {&forget_down, &gate_down, &beta};
        project_rows(x, p.control, parts, work_, s);
    }
    Tensor forget = take(&KdaVerifyRecord::forget, W);
    Tensor gate   = work_.alloc(DType::BF16, {W, T});
    project(forget_down, p.forget_up, forget, work_, s);
    project(gate_down, p.gate_up, gate, work_, s);

    Tensor q = work_.alloc(DType::BF16, {W, T});
    Tensor key           = take(&KdaVerifyRecord::key, W);
    Tensor v             = take(&KdaVerifyRecord::value, W);
    Tensor o = work_.alloc(DType::BF16, {D, NH, T});
    const float scale    = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    const Tensor dt_bias = p.dt_bias.view({D, NH});
    if (record != nullptr) {
        // Verification: every row's columns from its lane's window and state, which stay unwritten.
        const std::int32_t B = batch.batch, L = batch.width;
        auto layer = state_.kda->layer_view(index);
        ops::causal_conv1d_silu_from_states_split(qkv.view({C, L, B}), p.convolution, layer.conv,
                                                  batch.slots, q, key, v, s);
        Tensor o_rows = o.view({D, NH, L, B});
        ops::kimi_delta_attention_batch_verify(
            q.view({D, NH, L, B}), key.view({D, NH, L, B}), v.view({D, NH, L, B}),
            forget.view({D, NH, L, B}), beta.view({NH, L, B}), p.a_log, dt_bias, k.lower_bound,
            scale, layer.recurrent, batch.slots, o_rows, s);
    } else if (batch.width > 1) {
        // One sequence of several positions (a prefill chunk): the chunked recurrence on its slot.
        const std::int32_t slot = batch.host_slots[0];
        Tensor conv_state       = state_.kda->conv_slot(index, slot);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state, conv_state, q, key, v, s);
        Tensor state = state_.kda->recurrent_slot(index, slot);
        ops::kimi_delta_attention(q.view({D, NH, T}), key.view({D, NH, T}), v.view({D, NH, T}),
                                  forget.view({D, NH, T}), beta, p.a_log, dt_bias, k.lower_bound, scale, work_, state,
                                  o, device_.execution_view());
    } else {
        // One position per sequence (decode): the recurrent update on device-chosen slots.
        const std::int32_t B = T;
        auto layer           = state_.kda->layer_view(index);
        ops::causal_conv1d_silu_snapshot_split(qkv.view({C, 1, B}), p.convolution, layer.conv, Tensor{}, batch.slots,
                                               batch.slots, q, key, v, s);
        Tensor o_rows = o.view({D, NH, 1, B});
        ops::kimi_delta_attention_batch_update(q.view({D, NH, 1, B}), key.view({D, NH, 1, B}), v.view({D, NH, 1, B}),
                                               forget.view({D, NH, 1, B}), beta.view({NH, 1, B}), p.a_log, dt_bias,
                                               k.lower_bound, scale, layer.recurrent, batch.slots, o_rows, s);
    }
    Tensor normed = work_.alloc(DType::BF16, {D, NH * T});
    ops::gated_rmsnorm(o.view({D, NH * T}), p.output_norm, gate.view({D, NH * T}), ops::RmsGate::Sigmoid,
                       config_.rms_norm_eps, normed, device_.execution_view().on_stream(s));
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(normed.view({W, T}), p.output, y, work_, s);
    return y;
}

void Forward::commit_verified(const std::vector<KdaVerifyRecord>& records, std::int32_t batch,
                              std::int32_t width, std::span<const std::int32_t> slots,
                              std::span<const std::int32_t> columns) {
    const auto& k        = config_.kda;
    const std::int32_t W = dim(k.width()), NH = dim(k.heads), D = dim(k.head_dim), C = 3 * W,
                       T = batch * width;
    if (records.size() != config_.kda_layers || slots.size() != static_cast<std::size_t>(batch) ||
        columns.size() != slots.size() || batch < 1 || batch > 8 || width < 1 || width > 16) {
        throw std::invalid_argument(
            "Glm5Next forward: the verified commit does not match its call");
    }
    std::array<ops::KimiDeltaAttentionFoldRow, 8> fold{};
    std::array<ops::CausalConvCommitRow, 8> window{};
    for (std::int32_t b = 0; b < batch; ++b) {
        const auto i = static_cast<std::size_t>(b);
        fold[i]      = {slots[i], slots[i], columns[i]};
        window[i]    = {slots[i], slots[i], columns[i]};
    }
    const auto rows = static_cast<std::size_t>(batch);
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        if (config_.layer_types[l] != MixerKind::Kda) { continue; }
        const auto& p          = std::get<KdaParameters>(parameters_.layers[l].mixer);
        const std::uint32_t at = config_.compact_layer_indices[l];
        const auto& r          = records[at];
        auto layer             = state_.kda->layer_view(at);
        const auto rows4       = [&](const Tensor& t, std::int32_t heads) {
            return t.slice(1, 0, T).view({D, heads, width, batch});
        };
        ops::kimi_delta_attention_fold(rows4(r.key, NH), rows4(r.value, NH), rows4(r.forget, NH),
                                       r.beta.slice(1, 0, T).view({NH, width, batch}), p.a_log,
                                       p.dt_bias.view({D, NH}), k.lower_bound, layer.recurrent,
                                       std::span(fold.data(), rows), device_.stream);
        ops::causal_conv1d_commit_window(r.qkv.slice(1, 0, T).view({C, width, batch}), layer.conv,
                                         std::span(window.data(), rows), device_.stream);
    }
}

Tensor Forward::mla(const MlaParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.mla;
    const auto& ix       = config_.indexer;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t QR = dim(m.q_lora), KR = dim(m.kv_lora), NH = dim(m.heads), DN = dim(m.head_dim);
    const std::int32_t IH = dim(ix.heads), ID = dim(ix.head_dim);
    const auto geometry       = mla_geometry(config_);
    const auto index_geometry = indexer_geometry(config_);
    const MlaLayerCache& cache = kv_.layers.at(index);
    const ops::MlaSparseBatch sparse{.block_tables = kv_.block_tables,
                                     .table_rows   = batch.table_rows,
                                     .positions    = batch.positions,
                                     .batch        = batch.batch,
                                     .width        = batch.width};

    Tensor query_rank = work_.alloc(DType::BF16, {QR, T});
    Tensor latent     = work_.alloc(DType::BF16, {KR, T});
    {
        auto scope        = work_.scope();
        Tensor query_raw  = work_.alloc(DType::BF16, {QR, T});
        Tensor latent_raw = work_.alloc(DType::BF16, {KR, T});
        Tensor* parts[]   = {&query_raw, &latent_raw};
        project_rows(x, p.query_latent, parts, work_, s);
        ops::rmsnorm(query_raw, p.query_norm, config_.rms_norm_eps, /*unit_offset=*/false, query_rank, s);
        ops::rmsnorm(latent_raw, p.latent_norm, config_.rms_norm_eps, /*unit_offset=*/false, latent, s);
    }
    ops::mla_latent_append(latent.view({KR, batch.width, batch.batch}), sparse, cache.latent, s);

    // The indexer: its keys and gates join the cache (completing pools), then every column selects
    // its pools from the queries of the shared query rank.
    Tensor pools  = work_.alloc(DType::I32, {index_geometry.top_pools, T});
    Tensor counts = work_.alloc(DType::I32, {T});
    {
        auto scope         = work_.scope();
        Tensor raw_key     = work_.alloc(DType::BF16, {ID, T});
        Tensor key_gate    = work_.alloc(DType::BF16, {ID, T});
        Tensor head_weight = work_.alloc(DType::BF16, {IH, T});
        Tensor* parts[]    = {&raw_key, &key_gate, &head_weight};
        project_rows(x, p.index_inputs, parts, work_, s);
        ops::dsa_indexer_update(raw_key, key_gate, p.index_key_norm, p.index_key_bias, p.index_position, cache.index,
                                sparse, index_geometry, s);
        Tensor index_query = work_.alloc(DType::BF16, {ID * IH, T});
        project(query_rank, p.index_query, index_query, work_, s);
        // The reference's 1 / sqrt(heads) on the head weights scales every score alike, so the
        // selection omits it.
        const std::size_t bytes = ops::dsa_indexer_select_workspace_bytes(index_geometry, T, max_context_);
        const DeviceSpan workspace = work_.alloc_bytes(bytes);
        ops::dsa_indexer_select(index_query.view({ID, IH, T}), head_weight, cache.index, sparse, index_geometry,
                                static_cast<float>(1.0 / std::sqrt(static_cast<double>(ID))), max_context_,
                                workspace.data, bytes, pools, counts, s);
    }

    Tensor attended = work_.alloc(DType::BF16, {DN, NH, T});
    {
        auto scope  = work_.scope();
        Tensor q    = work_.alloc(DType::BF16, {DN * NH, T});
        project(query_rank, p.query_up, q, work_, s);
        Tensor qa = work_.alloc(DType::BF16, {KR, NH, T});
        ops::mla_absorbed_q_projection(q.view({DN, NH, T}), p.key_up, qa, s);
        const ops::MlaSelectedPools selected{.pools = &pools, .counts = &counts, .pool = index_geometry.pool};
        // A prefill call's columns fill the card one column per CTA, all heads sharing its tokens;
        // the split route's partials would leave the workspace a few columns per launch.
        const ops::MlaAttentionExecutionEnvelope envelope{
            .tensor_cores = batch.verify == nullptr && batch.width >= kPrefillAttentionWidth};
        const std::size_t bytes = ops::mla_sparse_attention_workspace_bytes(geometry, T, max_context_, envelope);
        const DeviceSpan workspace = work_.alloc_bytes(bytes);
        ops::mla_sparse_attention(qa, p.value_up, cache.latent, sparse, selected, geometry,
                                  static_cast<float>(1.0 / std::sqrt(static_cast<double>(DN))), max_context_, envelope,
                                  workspace.data, bytes, attended, s);
    }
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(attended.view({dim(m.value_width()), T}), p.output, y, work_, s);
    return y;
}

Tensor Forward::mlp(const MlpParameters& p, const Tensor& x) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t I = p.down.weight.k;
    Tensor y   = work_.alloc(DType::BF16, {H, T});
    auto scope = work_.scope();
    Tensor gate    = work_.alloc(DType::BF16, {I, T});
    Tensor up      = work_.alloc(DType::BF16, {I, T});
    Tensor product = work_.alloc(DType::BF16, {I, T});
    {
        Tensor* parts[] = {&gate, &up};
        project_rows(x, p.gate_up, parts, work_, s);
    }
    ops::silu_mul_clamped(gate, up, config_.swiglu_limit, product, s);
    project(product, p.down, y, work_, s);
    return y;
}

Tensor Forward::moe(const MoeParameters& p, const Tensor& x, std::uint32_t moe_layer, Tensor* route_tap) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    auto scope    = work_.scope();
    Tensor logits = work_.alloc(DType::FP32, {E, T});
    ops::MoeRouting routing{work_.alloc(DType::I32, {K, T}), work_.alloc(DType::FP32, {K, T}),
                            work_.alloc(DType::FP32, {T})};
    const DeviceSpan dispatch_bytes = work_.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
    Tensor outputs                  = work_.alloc(DType::BF16, {H, K * T});
    const std::int32_t max_jobs     = std::min(E, K * T);
    const auto geometry             = expert_geometry(config_);
    const auto& planes              = p.bank->planes;
    const bool ggml_bank            = planes.gate_up_format != QType::NVFP4_MUL;
    const auto wide_a16 = ggml_bank && experts_.ggml_bf16_prefill ? ops::offloaded_moe::WideA16::kGgmlBf16
                                                                  : ops::offloaded_moe::WideA16::kExact;
    const DeviceSpan expert_ws = work_.alloc_bytes(ops::moe_experts_workspace_bytes(
        max_jobs, K * T, T, ops::offloaded_moe::ExpertActivation::kA16, wide_a16, geometry));

    std::int32_t* route_log = nullptr;
    if (experts_.route_log != nullptr) {
        if (static_cast<std::size_t>(K) * T > experts_.route_stride) {
            throw std::invalid_argument("Glm5Next MoE: route log is too small for this call");
        }
        route_log = experts_.route_log + moe_layer * experts_.route_stride;
    }
    const std::uint64_t pitch = experts_.frame_pitches.empty() ? 0 : experts_.frame_pitches.at(moe_layer);
    if (pitch != 0 && pitch != planes.record_stride) {
        throw std::invalid_argument("Glm5Next MoE: the layer's frame pitch differs from its bank's record stride");
    }
    ops::MoeExpertSource source{.frame_base    = experts_.frame_bases.empty() ? nullptr : experts_.frame_bases.at(moe_layer),
                                .frames        = experts_.frames.at(moe_layer),
                                .host_records  = reinterpret_cast<const std::uint8_t*>(planes.device_records),
                                .record_stride = planes.record_stride,
                                .scales        = p.device_scales,
                                .activation    = ops::offloaded_moe::ExpertActivation::kA16,
                                .staging_base  = experts_.staging_base,
                                .staging_slots = experts_.staging_slots,
                                .cpu           = experts_.cpu.empty() ? ops::MoeCpuChannel{} : experts_.cpu.at(moe_layer)};
    if (!experts_.host_tables.empty()) {
        source.host_records = nullptr; // the banks hold no records in tier mode
        source.host_table   = experts_.host_tables.at(moe_layer);
        if (!experts_.fetch.empty()) { source.fetch = experts_.fetch.at(moe_layer); }
    }
    source.error        = experts_.error;
    source.geometry     = geometry;
    source.wide_a16     = wide_a16;
    source.frame_stride = pitch;
    if (ggml_bank) {
        // A GGUF import's bank: its own record size in frames of its pool's pitch, the exact A16
        // kernels (or, under ggml-bf16-prefill, the BF16 tiles for wide jobs) and the exact CPU
        // route for misses. Stages copy the record rounded up to 16 bytes (the stride's zero padding).
        source.ggml = ops::MoeGgmlBank{.gate_up_format = planes.gate_up_format,
                                       .down_format    = planes.down_format,
                                       .gate_row_bytes = static_cast<std::uint32_t>(planes.gate_row_bytes),
                                       .down_row_bytes = static_cast<std::uint32_t>(planes.down_row_bytes),
                                       .record_bytes   = (planes.record_bytes + 15) / 16 * 16};
    }
    // Every expert holds a frame: no call can miss, so it runs as one pass over every job with no
    // staging, fork, landing, CPU or fetch channel (the same arithmetic per expert, the same bits).
    if (experts_.resident) {
        source.staging_slots = 0;
        source.cpu           = ops::MoeCpuChannel{};
        source.fetch         = ops::MoeFetchChannel{};
    } else if (eager_ && experts_.overlap_stream != nullptr) {
        source.overlap_stream = experts_.overlap_stream;
        for (int i = 0; i < 5; ++i) { source.overlap_events[i] = experts_.overlap_events[static_cast<std::size_t>(i)]; }
    } else if (experts_.overlap_stream != nullptr) {
        source.fork_stream    = experts_.overlap_stream;
        source.fork_events[0] = experts_.overlap_events[0];
        source.fork_events[1] = experts_.overlap_events[1];
        if (experts_.landing != nullptr) {
            const auto offset    = static_cast<std::size_t>(moe_layer) * static_cast<std::size_t>(experts_.landing_slots);
            source.landing       = experts_.landing + offset;
            source.landed        = experts_.landed + offset;
            source.landing_slots = experts_.landing_slots;
        }
    }
    // While the host computes this call's misses, the CPU wait warms the next layer's weights into
    // L2 in the order it reads them: its attention mix, then its mixer's input projections. A
    // one-column call warms up to 40 MiB, a wider call 16 MiB of the first projection
    // (Qwen3.8-Flash-Next's measured budgets, its design §8.6).
    const std::uint32_t layer = moe_layer + config_.dense_layers;
    if (source.cpu.max_jobs > 0 && layer + 1 < config_.num_hidden_layers) {
        const auto& next = parameters_.layers[layer + 1];
        const bool whole_chain = T == 1;
        std::size_t left       = whole_chain ? 40ULL << 20 : 16ULL << 20;
        int spans              = 0;
        const auto add         = [&](const void* data, std::size_t bytes) {
            if (spans == ops::MoeL2Warm::kSpans || left < 16 || data == nullptr ||
                reinterpret_cast<std::uintptr_t>(data) % 16 != 0 || bytes < 16) {
                return;
            }
            const std::size_t take       = std::min(bytes, left);
            source.l2_warm.ptr[spans]   = data;
            source.l2_warm.bytes[spans] = take;
            left -= take;
            ++spans;
        };
        const auto add_linear = [&](const LinearParameters& linear) {
            add(linear.weight.payload, linear.weight.payload_bytes);
        };
        if (whole_chain) { add(next.attn_hc.fn.data, next.attn_hc.fn.bytes()); }
        if (config_.layer_types[layer + 1] == MixerKind::Kda) {
            const auto& kda = std::get<KdaParameters>(next.mixer);
            add_linear(kda.qkv);
            if (whole_chain) { add_linear(kda.control); }
        } else {
            const auto& mla = std::get<MlaParameters>(next.mixer);
            add_linear(mla.query_latent);
            if (whole_chain) {
                add_linear(mla.index_inputs);
                add_linear(mla.query_up);
            }
        }
    }

    // A streamed prefill chunk (expert_stream.h): the layer's copied-ahead experts are read from
    // the ring like resident ones. An expert not in the ring stages or fetches as before.
    ExpertStream* const streaming =
        eager_ && !experts_.resident && experts_.stream != nullptr && experts_.stream->active()
            ? experts_.stream
            : nullptr;
    if (streaming != nullptr) {
        if (stream_.first) { streaming->before_experts(moe_layer, s); }
        source.prefetched      = streaming->slots(moe_layer);
        source.prefetch_base   = streaming->base(moe_layer);
        source.prefetch_stride = streaming->pitch(moe_layer);
        // Every non-resident expert at chunk/walk start has a ring slot, and residency does not
        // evict frames while a stream is active: nothing needs staging, fetching, or CPU service,
        // so the layer runs in one pass over every job instead of staging passes of half the slots.
        if (streaming->complete(moe_layer)) {
            source.staging_slots  = 0;
            source.overlap_stream = nullptr;
            source.fetch          = ops::MoeFetchChannel{};
            source.cpu            = ops::MoeCpuChannel{};
        }
    }
    const Tensor* rows[] = {&p.router};
    ops::projection_fp32(x, rows, logits, s);
    const ops::MoeRoutePlan planned =
        ops::moe_route_dispatch_plan_noaux_tc(logits, p.router_bias, m.routed_scale, K, routing, dispatch, route_log,
                                              x, source, max_jobs, expert_ws.data, outputs, s);
    if (route_tap != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(route_tap->data, routing.ids.data, routing.ids.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s, /*wait_for_cpu=*/false, planned);
    if (streaming != nullptr && stream_.last) { streaming->after_experts(moe_layer, s); }
    // The shared expert runs while the host computes the CPU-served misses. It is ungated:
    // noaux_tc routing sets every shared gate to 1.
    const Tensor shared = mlp(p.shared, x);
    ops::moe_experts_cpu_wait(x, dispatch, source, max_jobs, expert_ws.data, outputs, s);
    ops::moe_combine(outputs, routing, shared, y, s);
    return y;
}

} // namespace infernix::models::glm5_next::execution
