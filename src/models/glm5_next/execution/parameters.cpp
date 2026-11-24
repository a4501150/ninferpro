#include "models/glm5_next/execution/parameters.h"

#include "core/device.h"
#include "core/weight_view.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::models::glm5_next::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    ops::WeightInput input(WeightId id) const {
        const auto& bound = model_.weight(id);
        return {bound.view, bound.policy, std::nullopt};
    }

    // One projection over consecutive parameters that share an input; the converter packs each
    // such group into one parent, and the rows keep this order.
    LinearParameters linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> rows;
        for (const auto id : ids) { rows.push_back(input(id)); }
        return with_context(model_.weight(*ids.begin()).name, [&] {
            const auto single = ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows));
            return LinearParameters{single.weight, single.policy};
        });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) { throw std::invalid_argument("direct parameter exceeds Tensor rank"); }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    HyperConnectionParameters hc(const HyperConnectionWeights& w) const {
        return {tensor(w.fn), tensor(w.base), tensor(w.scale)};
    }

    KdaParameters kda(const KdaWeights& k) const {
        return {linear({k.query, k.key, k.value}),
                linear({k.forget_down, k.gate_down, k.beta}),
                tensor(k.convolution),
                linear({k.forget_up}),
                linear({k.gate_up}),
                tensor(k.a_log),
                tensor(k.dt_bias),
                tensor(k.output_norm),
                linear({k.output})};
    }

    MlaParameters mla(const MlaWeights& a) const {
        return {linear({a.query_down, a.latent}),
                tensor(a.query_norm),
                tensor(a.latent_norm),
                linear({a.query_up}),
                linear({a.index_query}),
                tensor(a.key_up),
                tensor(a.value_up),
                linear({a.output}),
                linear({a.index_key, a.index_gate, a.index_weights}),
                tensor(a.index_key_norm),
                tensor(a.index_key_bias),
                tensor(a.index_position)};
    }

    MlpParameters mlp(const MlpWeights& m) const {
        return {linear({m.gate, m.up}), linear({m.down})};
    }

    // The DFlash2 drafter (docs/maintainer/dflash.md): the query block's fused Q/K/V parent, BF16
    // convolution and selector parameters.
    DFlash2Parameters dflash2(const DFlash2Weights& w, const DFlash2Config& d) const {
        DFlash2Parameters out;
        // Full-head 1-D RoPE at absolute positions (the drafter's own unscaled rotation).
        out.rope               = ops::prepare_rope(static_cast<int>(d.head_dim), d.rope_theta);
        out.feature_projection = linear({w.feature_projection});
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("layers/" + std::to_string(i), [&] {
                const auto& l = w.layers[i];
                DFlash2LayerParameters layer;
                layer.input_norm           = tensor(l.input_norm);
                layer.post_attention_norm  = tensor(l.post_attention_norm);
                layer.query_key_value      = ops::prepare_attn_input_proj_weights(input(l.query), input(l.key),
                                                                                  input(l.value)).weight;
                layer.context_key          = linear({l.context_key});
                layer.context_value        = linear({l.context_value});
                layer.query_norm           = tensor(l.query_norm);
                layer.key_norm             = tensor(l.key_norm);
                layer.output               = linear({l.output});
                layer.gate_up              = linear({l.gate, l.up});
                layer.down                 = linear({l.down});
                layer.attention_base       = tensor(l.attention_base);
                layer.attention_projection = linear({l.attention_projection});
                layer.mlp_base             = tensor(l.mlp_base);
                layer.mlp_projection       = linear({l.mlp_projection});
                return layer;
            }));
        }
        out.selector_hidden      = linear({w.selector_hidden});
        out.predecessor_codebook = tensor(w.predecessor_codebook);
        out.successor_codebook   = tensor(w.successor_codebook);
        return out;
    }

    // The vision tower; the blocks' gate/up biases are joined into `biases` ([layers][2I]).
    VisionParameters vision(const VisionWeights& w, const VisionConfig& v,
                            DeviceBuffer& biases) const {
        VisionParameters out;
        out.patch_embedding      = linear({w.patch_embedding});
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        const std::size_t joined = 2ULL * v.intermediate_size * sizeof(std::uint16_t);
        biases                   = DeviceBuffer(joined * w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& l = w.layers[i];
                VisionBlockParameters layer;
                layer.norm1       = tensor(l.norm1);
                layer.norm2       = tensor(l.norm2);
                layer.query       = linear({l.query});
                layer.key         = linear({l.key});
                layer.value       = linear({l.value});
                layer.query_bias  = tensor(l.query_bias);
                layer.key_bias    = tensor(l.key_bias);
                layer.value_bias  = tensor(l.value_bias);
                layer.query_norm  = tensor(l.query_norm);
                layer.key_norm    = tensor(l.key_norm);
                layer.output      = linear({l.output});
                layer.output_bias = tensor(l.output_bias);
                layer.gate_up     = linear({l.gate, l.up});
                layer.down        = linear({l.down});
                layer.down_bias   = tensor(l.down_bias);
                auto* joined_bias = static_cast<std::byte*>(biases.p) + joined * i;
                const Tensor gate = tensor(l.gate_bias), up = tensor(l.up_bias);
                // The biases stay resident with vision offload, so they are not rebased.
                CUDA_CHECK(cudaMemcpy(joined_bias, gate.data, joined / 2, cudaMemcpyDefault));
                CUDA_CHECK(
                    cudaMemcpy(joined_bias + joined / 2, up.data, joined / 2, cudaMemcpyDefault));
                layer.gate_up_bias =
                    Tensor(joined_bias, DType::BF16, {dim(2ULL * v.intermediate_size)});
                return layer;
            }));
        }
        out.post_norm = tensor(w.post_norm);
        {
            // The permuted kernel read as the [O, m*m*H] matrix of one GEMM.
            const auto& bound = model_.weight(w.downsample);
            WeightView view   = bound.view;
            view.shape        = {v.out_hidden_size, v.merged_width()};
            out.downsample    = with_context(bound.name, [&] {
                const auto single =
                    ops::prepare_linear_weight(ops::WeightInput{view, bound.policy, std::nullopt});
                return LinearParameters{single.weight, single.policy};
            });
        }
        out.downsample_bias   = tensor(w.downsample_bias);
        out.merger_projection = linear({w.merger_projection});
        out.merger_norm       = tensor(w.merger_norm);
        out.merger_norm_bias  = tensor(w.merger_norm_bias);
        out.merger_gate_up    = linear({w.merger_gate, w.merger_up});
        out.merger_down       = linear({w.merger_down});
        return out;
    }

private:
    static std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(source);
    const auto& w = source.weights();
    const auto& c = source.config().text;
    const auto& embedding = source.weight(w.token_embedding);
    token_embedding = with_context(embedding.name, [&] { return device_native_weight(embedding.view); });
    if (token_embedding.qtype != QType::BF16) {
        throw std::invalid_argument(embedding.name + ": the token embedding must be BF16 rows");
    }
    final_norm  = prepare.tensor(w.final_norm);
    output_head = prepare.tensor(w.output_head);

    const auto banks = source.expert_banks();
    if (banks.size() != c.moe_layers()) { throw std::logic_error("Glm5Next: one expert bank per MoE layer"); }
    const std::size_t per_layer = sizeof(ops::offloaded_moe::ExpertScales) * c.moe.experts;
    scales_ = DeviceBuffer(per_layer * banks.size());
    for (std::size_t i = 0; i < banks.size(); ++i) {
        scales_.copy_from_host(banks[i].scales.data(), per_layer, per_layer * i);
    }

    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& layer = w.layers[i];
        layers.push_back(with_context("text/layers/" + std::to_string(i), [&] {
            BlockParameters out;
            out.attn_hc             = prepare.hc(layer.attn_hc);
            out.ffn_hc              = prepare.hc(layer.ffn_hc);
            out.input_norm          = prepare.tensor(layer.input_norm);
            out.post_attention_norm = prepare.tensor(layer.post_attention_norm);
            if (const auto* k = std::get_if<KdaWeights>(&layer.mixer)) {
                out.mixer = prepare.kda(*k);
            } else {
                out.mixer = prepare.mla(std::get<MlaWeights>(layer.mixer));
            }
            if (const auto* m = std::get_if<MlpWeights>(&layer.ffn)) {
                out.ffn = prepare.mlp(*m);
            } else {
                const auto& moe   = std::get<MoeWeights>(layer.ffn);
                const std::size_t bank = i - c.dense_layers;
                MoeParameters p;
                p.router        = prepare.tensor(moe.router);
                p.router_bias   = prepare.tensor(moe.router_bias);
                p.shared        = prepare.mlp(moe.shared);
                p.bank          = &banks[bank];
                p.device_scales = reinterpret_cast<const ops::offloaded_moe::ExpertScales*>(
                    static_cast<const std::byte*>(scales_.p) + per_layer * bank);
                out.ffn = std::move(p);
            }
            return out;
        }));
    }
    if (w.dflash2) {
        dflash2 = with_context("dflash2", [&] { return prepare.dflash2(*w.dflash2, *model.config().dflash2); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] {
            return prepare.vision(*w.vision, *model.config().vision, vision_biases_);
        });
    }
}

Parameters::~Parameters() = default;

} // namespace infernix::models::glm5_next::execution
