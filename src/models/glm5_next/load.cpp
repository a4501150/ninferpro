#include "models/glm5_next/load.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/common/vision/overlay_layout.h"
#include "models/registry.h"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace infernix::models::glm5_next {
namespace {

using artifact::ArtifactError;
using artifact::Residency;
using artifact::Shape;

ops::LinearPolicy linear_policy(artifact::ActivationPolicy policy) {
    switch (policy) {
    case artifact::ActivationPolicy::A16Only:
        return ops::LinearPolicy::A16Only;
    case artifact::ActivationPolicy::AllowA8:
        return ops::LinearPolicy::AllowA8;
    case artifact::ActivationPolicy::AllowA4:
        return ops::LinearPolicy::AllowA4;
    }
    throw ArtifactError("unknown activation policy");
}

struct PendingWeight {
    artifact::ParameterReference reference;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

// Logical demands of the selected model; each parameter is declared exactly once.
class Bindings {
public:
    // a16_activations binds every Use A16Only (LoadOptions::a16_activations). first_id: the
    // WeightId of this binder's first parameter (a companion artifact's follow the main artifact's).
    Bindings(artifact::Binder& binder, bool a16_activations, std::size_t first_id = 0)
        : binder_(binder), a16_activations_(a16_activations), first_id_(first_id) {}

    WeightId parameter(std::string name, Shape shape, std::string_view input = {},
                       std::optional<QType> exact_format = {},
                       Residency residency                = Residency::Device) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), residency, exact_format);
        if (!input.empty()) {
            const auto& use = binder_.use(name, input);
            if (!use.activation_policy) {
                throw ArtifactError(name + "@" + std::string(input) + ": missing activation policy");
            }
            if (!use.auxiliaries.empty()) {
                throw ArtifactError(name + ": Glm5Next weights carry no Use auxiliaries");
            }
            pending.policy = a16_activations_ ? ops::LinearPolicy::A16Only
                                              : linear_policy(*use.activation_policy);
        }
        const WeightId id{first_id_ + weights.size()};
        ids_.emplace(std::move(name), id);
        weights.push_back(std::move(pending));
        return id;
    }

    WeightId direct(std::string name, Shape shape, QType format = QType::BF16,
                    Residency residency = Residency::Device) {
        return parameter(std::move(name), std::move(shape), {}, format, residency);
    }

    std::vector<PendingWeight> weights;

private:
    artifact::Binder& binder_;
    bool a16_activations_ = false;
    std::size_t first_id_ = 0;
    std::map<std::string, WeightId, std::less<>> ids_;
};

// The mixer weights the mhc Op reads directly: fn BF16 (used in FP32), base and scale FP32.
HyperConnectionWeights bind_hc(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t width = c.residual_width(), rows = (2ULL + c.hc.streams) * c.hc.streams;
    HyperConnectionWeights out;
    out.fn    = b.parameter(prefix + "fn", {rows, width}, prefix + "input", QType::BF16);
    out.base  = b.direct(prefix + "base", {rows}, QType::FP32);
    out.scale = b.direct(prefix + "scale", {3}, QType::FP32);
    return out;
}

KdaWeights bind_kda(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, w = c.kda.width(), d = c.kda.head_dim,
                        heads = c.kda.heads, taps = c.kda.conv_kernel;
    const std::string input = prefix + "mixer_input", p = prefix + "kda/";
    KdaWeights out;
    out.query             = b.parameter(p + "query", {w, h}, input);
    out.key               = b.parameter(p + "key", {w, h}, input);
    out.value             = b.parameter(p + "value", {w, h}, input);
    out.convolution       = b.direct(p + "convolution", {taps, 3 * w});
    out.forget_down       = b.parameter(p + "forget_down", {d, h}, input);
    out.forget_up         = b.parameter(p + "forget_up", {w, d}, p + "forget_rank");
    out.gate_down         = b.parameter(p + "gate_down", {d, h}, input);
    out.gate_up           = b.parameter(p + "gate_up", {w, d}, p + "gate_rank");
    out.beta              = b.parameter(p + "beta", {heads, h}, input);
    out.a_log             = b.direct(p + "a_log", {heads}, QType::FP32);
    out.dt_bias           = b.direct(p + "dt_bias", {w}, QType::FP32);
    out.output_norm       = b.direct(p + "output_norm", {d});
    out.output            = b.parameter(p + "output", {h, w}, p + "gated_output");
    return out;
}

MlaWeights bind_mla(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, q_rank = c.mla.q_lora, kv_rank = c.mla.kv_lora,
                        q = c.mla.query_width(), v = c.mla.value_width(),
                        iq = std::uint64_t(c.indexer.heads) * c.indexer.head_dim,
                        id = c.indexer.head_dim;
    const std::string input = prefix + "mixer_input", p = prefix + "attention/";
    MlaWeights out;
    out.query_down     = b.parameter(p + "query_down", {q_rank, h}, input);
    out.query_norm     = b.direct(p + "query_norm", {q_rank});
    out.query_up       = b.parameter(p + "query_up", {q, q_rank}, p + "query_rank");
    out.latent         = b.parameter(p + "latent", {kv_rank, h}, input);
    out.latent_norm    = b.direct(p + "latent_norm", {kv_rank});
    out.key_up         = b.direct(p + "key_up", {c.mla.heads, kv_rank, c.mla.head_dim});
    out.value_up       = b.direct(p + "value_up", {c.mla.heads, c.mla.value_dim, kv_rank});
    out.output         = b.parameter(p + "output", {h, v}, p + "attention_output");
    out.index_query    = b.parameter(p + "index_query", {iq, q_rank}, p + "query_rank");
    out.index_key      = b.parameter(p + "index_key", {id, h}, input);
    out.index_key_norm = b.direct(p + "index_key_norm", {id});
    out.index_key_bias = b.direct(p + "index_key_bias", {id});
    out.index_gate     = b.parameter(p + "index_gate", {id, h}, input);
    out.index_position = b.direct(p + "index_position", {c.indexer.pool, id}, QType::FP32);
    out.index_weights  = b.parameter(p + "index_weights", {c.indexer.heads, h}, input);
    return out;
}

MlpWeights bind_mlp(Bindings& b, const std::string& p, std::uint64_t h, std::uint64_t intermediate,
                    const std::string& input) {
    MlpWeights out;
    out.gate = b.parameter(p + "gate", {intermediate, h}, input);
    out.up   = b.parameter(p + "up", {intermediate, h}, input);
    out.down = b.parameter(p + "down", {h, intermediate}, p + "product");
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& prefix, Residency experts) {
    const std::uint64_t h = c.hidden_size, e = c.moe.experts;
    const std::string input = prefix + "ffn_input", p = prefix + "moe/";
    MoeWeights out;
    out.router      = b.parameter(p + "router", {e, h}, input);
    out.router_bias = b.direct(p + "router_bias", {e}, QType::FP32);
    out.shared      = bind_mlp(b, p + "shared/", h, c.moe.shared_intermediate, input);
    // An NVFP4 bank (nvfp4_mul in nvfp4_expert_rg16_v1); plan_load checks the layout.
    out.experts = b.parameter(p + "experts", {e, h, c.moe.intermediate}, input, std::nullopt, experts);
    if (b.weights.at(out.experts.index).policy != ops::LinearPolicy::A16Only) {
        throw ArtifactError(p + "experts: Glm5Next expert banks are W4A16 (an A16-only use)");
    }
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& c, Residency experts) {
    TextWeights out;
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    out.token_embedding = b.parameter("text/token_embedding", {v, h}, {}, QType::BF16, Residency::HostPinned);
    out.final_norm      = b.direct("text/final_norm", {h});
    out.output_head     = b.parameter("text/output_head", {v, h}, "text/final_hidden");
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string prefix = "text/layers/" + std::to_string(i) + "/";
        BlockWeights block;
        block.attn_hc             = bind_hc(b, c, prefix + "attn_hc/");
        block.ffn_hc              = bind_hc(b, c, prefix + "ffn_hc/");
        block.input_norm          = b.direct(prefix + "input_norm", {h});
        block.post_attention_norm = b.direct(prefix + "post_attention_norm", {h});
        if (c.layer_types[i] == MixerKind::Kda) {
            block.mixer = bind_kda(b, c, prefix);
        } else {
            block.mixer = bind_mla(b, c, prefix);
        }
        if (i < c.dense_layers) {
            block.ffn = bind_mlp(b, prefix + "mlp/", h, c.dense_intermediate, prefix + "ffn_input");
        } else {
            block.ffn = bind_moe(b, c, prefix, experts);
        }
        out.layers.push_back(std::move(block));
    }
    return out;
}

// The DFlash2 drafter of a companion artifact (tools/convert/glm5_next.py _Builder.draft names);
// the context K/V roles share the query block's K/V words.
DFlash2Weights bind_dflash2(Bindings& b, const DFlash2Config& d, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, q = std::uint64_t(d.heads) * d.head_dim,
                        k = std::uint64_t(d.kv_heads) * d.head_dim, i = d.intermediate,
                        rows = 2ULL * d.conv_kernel * (h / d.conv_group);
    DFlash2Weights out;
    out.feature_projection = b.parameter("dflash2/feature_projection", {h, d.feature_width(c.hidden_size)},
                                         "dflash2/target_features");
    out.context_norm       = b.direct("dflash2/context_norm", {h});
    out.final_norm         = b.direct("dflash2/final_norm", {h});
    for (std::uint32_t l = 0; l < d.layers; ++l) {
        const std::string p = "dflash2/layers/" + std::to_string(l) + "/", a = p + "attention/";
        DFlash2LayerWeights layer;
        layer.input_norm           = b.direct(p + "input_norm", {h});
        layer.post_attention_norm  = b.direct(p + "post_attention_norm", {h});
        layer.query                = b.parameter(a + "query", {q, h}, p + "query_projection_input");
        layer.key                  = b.parameter(a + "key", {k, h}, p + "query_projection_input");
        layer.value                = b.parameter(a + "value", {k, h}, p + "query_projection_input");
        layer.context_key          = b.parameter(a + "context_key", {k, h}, "dflash2/context_input");
        layer.context_value        = b.parameter(a + "context_value", {k, h}, "dflash2/context_input");
        layer.query_norm           = b.direct(a + "query_norm", {d.head_dim});
        layer.key_norm             = b.direct(a + "key_norm", {d.head_dim});
        layer.output               = b.parameter(a + "output", {h, q}, p + "attention_output");
        layer.gate                 = b.parameter(p + "mlp/gate", {i, h}, p + "mlp_input");
        layer.up                   = b.parameter(p + "mlp/up", {i, h}, p + "mlp_input");
        layer.down                 = b.parameter(p + "mlp/down", {h, i}, p + "mlp_product");
        layer.attention_base       = b.direct(p + "attention_conv/base_kernel", {2, d.conv_kernel, h});
        layer.attention_projection = b.parameter(p + "attention_conv/kernel_projection", {rows, h},
                                                 p + "attention_conv_input");
        layer.mlp_base             = b.direct(p + "mlp_conv/base_kernel", {2, d.conv_kernel, h});
        layer.mlp_projection       = b.parameter(p + "mlp_conv/kernel_projection", {rows, h}, p + "mlp_conv_input");
        out.layers.push_back(layer);
    }
    out.selector_hidden      = b.parameter("dflash2/candidate_selector/hidden_projection", {d.selector_rank, h},
                                           "dflash2/final_hidden");
    out.predecessor_codebook = b.direct("dflash2/candidate_selector/predecessor_codebook", {c.vocab_size, d.selector_rank});
    out.successor_codebook   = b.direct("dflash2/candidate_selector/successor_codebook", {c.vocab_size, d.selector_rank});
    return out;
}

// The vision tower (tools/convert/glm5_next.py _Builder.vision names): on the device, or with
// vision offload in pinned host memory.
VisionWeights bind_vision(Bindings& b, const VisionConfig& v, Residency r) {
    const std::uint64_t h = v.hidden_size, i = v.intermediate_size, o = v.out_hidden_size,
                        pi = v.projection_intermediate_size, d = v.head_dim(),
                        m = v.spatial_merge_size;
    VisionWeights out;
    out.patch_embedding =
        b.parameter("vision/patch_embedding", {h, v.patch_width()}, "vision/patch_input", {}, r);
    out.patch_embedding_bias = b.direct("vision/patch_embedding_bias", {h}, QType::BF16, r);
    for (std::uint32_t l = 0; l < v.depth; ++l) {
        const std::string p = "vision/layers/" + std::to_string(l) + "/", a = p + "attention/";
        VisionBlockWeights layer;
        layer.norm1       = b.direct(p + "norm1", {h}, QType::BF16, r);
        layer.norm2       = b.direct(p + "norm2", {h}, QType::BF16, r);
        layer.query       = b.parameter(a + "query", {h, h}, p + "attention_input", {}, r);
        layer.key         = b.parameter(a + "key", {h, h}, p + "attention_input", {}, r);
        layer.value       = b.parameter(a + "value", {h, h}, p + "attention_input", {}, r);
        layer.query_bias  = b.direct(a + "query_bias", {h}, QType::BF16, r);
        layer.key_bias    = b.direct(a + "key_bias", {h}, QType::BF16, r);
        layer.value_bias  = b.direct(a + "value_bias", {h}, QType::BF16, r);
        layer.query_norm  = b.direct(a + "query_norm", {d}, QType::BF16, r);
        layer.key_norm    = b.direct(a + "key_norm", {d}, QType::BF16, r);
        layer.output      = b.parameter(a + "output", {h, h}, p + "attention_output", {}, r);
        layer.output_bias = b.direct(a + "output_bias", {h}, QType::BF16, r);
        layer.gate        = b.parameter(p + "mlp/gate", {i, h}, p + "mlp_input", {}, r);
        layer.gate_bias   = b.direct(p + "mlp/gate_bias", {i}, QType::BF16, r);
        layer.up          = b.parameter(p + "mlp/up", {i, h}, p + "mlp_input", {}, r);
        layer.up_bias     = b.direct(p + "mlp/up_bias", {i}, QType::BF16, r);
        layer.down        = b.parameter(p + "mlp/down", {h, i}, p + "mlp_product", {}, r);
        layer.down_bias   = b.direct(p + "mlp/down_bias", {h}, QType::BF16, r);
        out.layers.push_back(layer);
    }
    out.post_norm  = b.direct("vision/post_norm", {h}, QType::BF16, r);
    out.downsample = b.parameter("vision/downsample", {o, h, m, m}, "vision/merged_input", {}, r);
    out.downsample_bias = b.direct("vision/downsample_bias", {o}, QType::BF16, r);
    out.merger_projection =
        b.parameter("vision/merger/projection", {o, o}, "vision/merger/input", {}, r);
    out.merger_norm      = b.direct("vision/merger/post_projection_norm", {o}, QType::BF16, r);
    out.merger_norm_bias = b.direct("vision/merger/post_projection_norm_bias", {o}, QType::BF16, r);
    out.merger_gate = b.parameter("vision/merger/gate", {pi, o}, "vision/merger/activation", {}, r);
    out.merger_up   = b.parameter("vision/merger/up", {pi, o}, "vision/merger/activation", {}, r);
    out.merger_down = b.parameter("vision/merger/down", {o, pi}, "vision/merger/product", {}, r);
    return out;
}

// The downsample's conv kernel [O, H, m, m] becomes [O, m*m, H] in place: the column of a merged
// token is its m x m patches' H features one patch after another, so the convolution is one GEMM.
void permute_downsample(const BoundWeight& weight, const VisionConfig& v) {
    const WeightRegion region = contiguous_weight_region(weight.view);
    const std::uint64_t o = v.out_hidden_size, h = v.hidden_size, area = v.merge_area();
    if (region.parent == nullptr || region.parent->geometry.format != QType::BF16 ||
        region.end - region.begin != o * h * area) {
        throw ArtifactError(weight.name + ": the downsample must be one plain BF16 kernel");
    }
    auto* data = const_cast<std::byte*>(region.parent->data) + region.begin * sizeof(std::uint16_t);
    const std::size_t bytes = o * h * area * sizeof(std::uint16_t);
    std::vector<std::uint16_t> stored(o * h * area), permuted(stored.size());
    CUDA_CHECK(cudaMemcpy(stored.data(), data, bytes, cudaMemcpyDefault)); // device or pinned
    for (std::uint64_t row = 0; row < o; ++row) {
        const std::uint16_t* in = stored.data() + row * h * area;
        std::uint16_t* out      = permuted.data() + row * h * area;
        for (std::uint64_t c = 0; c < h; ++c) {
            for (std::uint64_t j = 0; j < area; ++j) { out[j * h + c] = in[c * area + j]; }
        }
    }
    CUDA_CHECK(cudaMemcpy(data, permuted.data(), bytes, cudaMemcpyDefault));
}

// The pinned groups the encode streams (vision offload): the patch embedding, each layer, and
// everything after the last layer.
common::VisionOverlayLayout vision_overlay_layout(const VisionWeights& vision,
                                                  const std::vector<PendingWeight>& pending,
                                                  const artifact::MaterializationPlan& plan) {
    const auto handles = [&](std::initializer_list<WeightId> ids) {
        std::vector<artifact::ObjectHandle> out;
        for (const WeightId id : ids) {
            for (const auto& part : pending.at(id.index).reference.binding.parts) {
                out.push_back(part.object);
            }
        }
        return out;
    };
    std::vector<std::vector<artifact::ObjectHandle>> layers;
    for (const VisionBlockWeights& l : vision.layers) {
        layers.push_back(
            handles({l.norm1, l.norm2, l.query, l.key, l.value, l.query_bias, l.key_bias,
                     l.value_bias, l.query_norm, l.key_norm, l.output, l.output_bias, l.gate,
                     l.gate_bias, l.up, l.up_bias, l.down, l.down_bias}));
    }
    return common::make_vision_overlay_layout(
        plan, handles({vision.patch_embedding, vision.patch_embedding_bias}), layers,
        handles({vision.post_norm, vision.downsample, vision.downsample_bias,
                 vision.merger_projection, vision.merger_norm, vision.merger_norm_bias,
                 vision.merger_gate, vision.merger_up, vision.merger_down}));
}

FrontendResources bind_resources(artifact::Binder& binder, const Config& config) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("text", "tokenizer.json");
    out.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.generation_config_json = resource("text", "generation_config.json");
    if (config.vision) { out.processor_config_json = resource("vision", "processor_config.json"); }
    // The frontend validates the token domain against the embedding rows.
    parse_resources(out, {.vocab_size = config.text.vocab_size});
    if (config.dflash2 && config.dflash2->selector_top_k > out.public_token_count) {
        throw ArtifactError("DFlash2 selector_top_k exceeds the public token domain");
    }
    return out;
}

// A W4A16 bank's scalars: no input scales, alpha = the stored multipliers.
std::vector<ops::offloaded_moe::ExpertScales> a16_expert_scales(const ExpertBankPlanes& planes,
                                                               const std::string& name) {
    std::vector<ops::offloaded_moe::ExpertScales> out(planes.experts);
    for (std::uint32_t e = 0; e < planes.experts; ++e) {
        const float* m = planes.multipliers + 3 * std::size_t(e);
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(m[i]) || m[i] <= 0) {
                throw ArtifactError(name + ": expert multipliers must be positive finite FP32");
            }
        }
        out[e] = {0.0F, 0.0F, 0.0F, m[0], m[1], m[2]};
    }
    return out;
}

const MoeWeights& moe_of(const BlockWeights& block) { return std::get<MoeWeights>(block.ffn); }

} // namespace

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    TextWeights weights;
    std::vector<PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::uint64_t embedding_bytes = 0; // the token embedding's stored bytes (pinned)
    std::optional<common::VisionOverlayLayout> vision_overlay; // vision offload
    // The DFlash2 companion (--draft-model): its Reader, its plan, and the first of its weights in
    // `pending` (the main artifact's come first).
    std::unique_ptr<artifact::Reader> draft_reader;
    std::optional<artifact::MaterializationPlan> draft_materialization;
    std::size_t draft_first = 0;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

// Besides the expert banks: the token embedding (bind_text) and, with vision offload, the tower.
std::uint64_t LoadPlan::pinned_other_bytes() const {
    return impl_->embedding_bytes +
           (impl_->vision_overlay ? impl_->vision_overlay->tower_bytes() : 0);
}

std::uint64_t LoadPlan::device_bytes() const {
    return impl_->materialization.device_capacity_bytes +
           (impl_->draft_materialization ? impl_->draft_materialization->device_capacity_bytes : 0);
}

std::uint32_t LoadPlan::public_token_count() const {
    return static_cast<std::uint32_t>(impl_->resources.public_token_count);
}

std::uint64_t LoadPlan::pinned_expert_bytes() const {
    const std::uint64_t pinned = impl_->materialization.pinned_capacity_bytes;
    const std::uint64_t other  = pinned_other_bytes();
    return pinned > other ? pinned - other : 0;
}

void LoadPlan::set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes) {
    impl_->materialization.host_reserve_bytes = reserve_bytes;
    impl_->materialization.later_pinned_bytes = later_pinned_bytes;
}

bool is_glm5_next(const artifact::Reader& reader) {
    const auto& config = reader.directory().component("text").config;
    if (!config.contains("architectures") || !config.contains("model_type")) { return false; }
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1 || !names[0].is_string() ||
        !config.at("model_type").is_string()) {
        return false;
    }
    try {
        return resolve_architecture(names[0].get<std::string>(),
                                    config.at("model_type").get<std::string>()) ==
               Architecture::Glm5Next;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    if (options.dflash2() && options.purpose == EnginePurpose::Generation) {
        out->draft_reader   = std::make_unique<artifact::Reader>(options.draft_model);
        out->config.dflash2 = parse_dflash2_config(out->draft_reader->directory(), out->config.text);
    }
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder, out->config);
    Bindings bindings(binder, options.a16_activations);
    out->weights = bind_text(bindings, out->config.text,
                             options.stream_experts ? Residency::Streamed : Residency::HostPinned);
    if (out->config.vision) {
        out->weights.vision =
            bind_vision(bindings, *out->config.vision,
                        options.overlay_vision() ? Residency::HostPinned : Residency::Device);
    }
    for (std::size_t i = out->config.text.dense_layers; i < out->weights.layers.size(); ++i) {
        for (const auto& part : bindings.weights.at(moe_of(out->weights.layers[i]).experts.index).reference.binding.parts) {
            const auto& tensor = reader.directory().tensor(part.object);
            const auto g = weight_geometry(artifact::parse_format(tensor.format),
                                           artifact::parse_layout(tensor.layout), tensor.shape);
            if (g.layout != QuantLayout::ExpertRg16 || g.format != QType::NVFP4_MUL) {
                throw ArtifactError("Glm5Next: routed experts must be nvfp4_mul banks in nvfp4_expert_rg16_v1");
            }
        }
    }
    {
        std::vector<artifact::ObjectHandle> objects;
        for (const auto& part : bindings.weights.at(out->weights.token_embedding.index).reference.binding.parts) {
            if (std::find(objects.begin(), objects.end(), part.object) == objects.end()) { objects.push_back(part.object); }
        }
        for (const auto object : objects) {
            out->embedding_bytes += artifact::object_bytes(reader.directory().object(object));
        }
    }
    out->pending = std::move(bindings.weights);
    if (out->config.dflash2) {
        // The companion's weights, after every main-artifact weight; it declares its Use of the
        // target head its drafts read.
        artifact::Binder draft_binder(*out->draft_reader);
        (void)draft_binder.use("text/output_head", "dflash2/final_hidden");
        Bindings draft(draft_binder, options.a16_activations, out->pending.size());
        out->draft_first     = out->pending.size();
        out->weights.dflash2 = bind_dflash2(draft, *out->config.dflash2, out->config.text);
        for (auto& weight : draft.weights) { out->pending.push_back(std::move(weight)); }
        out->draft_materialization = std::move(draft_binder).finish();
    }
    out->materialization = std::move(binder).finish();
    if (out->weights.vision && options.overlay_vision()) {
        out->vision_overlay =
            vision_overlay_layout(*out->weights.vision, out->pending, out->materialization);
    }
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(Architecture::Glm5Next)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    if (out->draft_reader) { out->info.draft_artifact_id = out->draft_reader->artifact_id(); }
    return LoadPlan(std::move(out));
}

Model::Model(Config config, LoadOptions options, TextWeights weights,
             std::vector<BoundWeight> bound, std::vector<ExpertBank> banks,
             FrontendResources resources, InstanceInfo info,
             std::optional<common::VisionOverlayAssets> overlay_vision,
             artifact::MaterializedArtifact backing,
             std::optional<offload::ExpertStore> expert_store,
             std::optional<artifact::MaterializedArtifact> draft_backing)
    : backing_(std::move(backing)), draft_backing_(std::move(draft_backing)),
      expert_store_(std::move(expert_store)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), banks_(std::move(banks)),
      resources_(std::move(resources)), info_(std::move(info)),
      overlay_vision_(std::move(overlay_vision)) {}

Model::~Model() = default;

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data                      = std::move(plan.impl_);
    const artifact::Reader& reader = *data->materialization.source;
    auto backing = artifact::materialize(reader, std::move(data->materialization), device, observer);
    std::optional<artifact::MaterializedArtifact> draft_backing;
    if (data->draft_materialization) {
        draft_backing = artifact::materialize(*data->draft_reader, std::move(*data->draft_materialization),
                                              device, observer);
    }
    std::vector<BoundWeight> bound;
    bound.reserve(data->pending.size());
    for (std::size_t i = 0; i < data->pending.size(); ++i) {
        auto& item = data->pending[i];
        BoundWeight weight;
        weight.name   = item.reference.name;
        weight.policy = item.policy;
        weight.view   = artifact::bind_view(item.reference,
                                            draft_backing && i >= data->draft_first ? *draft_backing : backing);
        bound.push_back(std::move(weight));
    }
    if (data->weights.vision) {
        permute_downsample(bound.at(data->weights.vision->downsample.index), *data->config.vision);
    }
    std::vector<ExpertBank> banks;
    const auto& c = data->config.text;
    // Streamed banks (the SSD tier): the records stay in the artifact; each layer's multipliers
    // (the scale tail) are read here into Model memory.
    const bool streamed = data->options.stream_experts;
    std::vector<artifact::ObjectHandle> bank_objects;
    std::vector<std::uint64_t> bank_strides;
    std::vector<std::vector<float>> multipliers;
    for (std::size_t i = c.dense_layers; i < data->weights.layers.size(); ++i) {
        const WeightId experts = moe_of(data->weights.layers[i]).experts;
        const auto& view       = bound.at(experts.index).view;
        if (view.parts.size() != 1 || !is_complete_weight(view)) {
            throw ArtifactError(bound.at(experts.index).name + ": an expert bank must be one complete parent");
        }
        ExpertBank bank;
        if (streamed) {
            const auto object  = data->pending.at(experts.index).reference.binding.parts.front().object;
            const auto& parent = *view.parts.front().parent;
            std::vector<float> words(parent.geometry.scale_bytes / sizeof(float));
            reader.read_into(artifact::object_offset(reader.directory().object(object)) + parent.geometry.scale_offset,
                             std::as_writable_bytes(std::span(words)));
            bank.planes = expert_bank_layout(parent.geometry, words.data());
            bank_objects.push_back(object);
            bank_strides.push_back(bank.planes.record_stride);
            multipliers.push_back(std::move(words));
        } else {
            bank.planes = expert_bank_planes(*view.parts.front().parent);
        }
        if (bank.planes.experts != c.moe.experts || bank.planes.hidden != c.hidden_size ||
            bank.planes.intermediate != c.moe.intermediate) {
            throw ArtifactError("expert bank geometry differs from the text config");
        }
        bank.scales = a16_expert_scales(bank.planes, bound.at(experts.index).name);
        banks.push_back(std::move(bank));
    }
    std::optional<offload::ExpertStore> store;
    if (streamed) {
        store.emplace(backing.stream_source(), bank_objects, std::move(bank_strides), c.moe.experts,
                      std::move(multipliers)); // the vectors move, so the planes' pointers stay valid
    }
    std::optional<common::VisionOverlayAssets> overlay;
    if (data->vision_overlay) {
        const auto pinned = backing.pinned_bytes_range();
        if (pinned.empty()) {
            throw ArtifactError("vision offload requires a pinned materialization");
        }
        overlay = common::VisionOverlayAssets{.pinned_block = pinned.data(),
                                              .pinned_bytes = pinned.size(),
                                              .layout       = std::move(*data->vision_overlay)};
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(banks), std::move(data->resources), std::move(data->info), std::move(overlay),
        std::move(backing), std::move(store), std::move(draft_backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace infernix::models::glm5_next
