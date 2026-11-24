#include "models/glm5_next/config.h"

#include "artifact/schema.h"
#include "models/registry.h"

#include <cmath>
#include <limits>
#include <string>

namespace infernix::models::glm5_next {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float finite_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result)) { throw ArtifactError(std::string(name) + " must be finite FP32"); }
    return result;
}

float positive_float(const Json& config, const char* name) {
    const float result = finite_float(config, name);
    if (result <= 0) { throw ArtifactError(std::string(name) + " must be positive"); }
    return result;
}

TextConfig text(const Json& value) {
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
         "max_position_embeddings", "intermediate_size", "moe_intermediate_size",
         "n_routed_experts", "num_experts_per_tok", "first_k_dense_replace", "num_attention_heads",
         "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim", "index_n_heads",
         "index_head_dim", "index_topk", "index_kpool", "hc_mult", "hc_sinkhorn_iters",
         "tie_word_embeddings", "rms_norm_eps", "hc_eps", "routed_scaling_factor", "swiglu_limit",
         "layer_types", "linear_num_heads", "linear_head_dim", "linear_conv_kernel_dim",
         "linear_lower_bound", "eos_token_id", "mtp_layers"},
        {}, "text config");
    const auto& names = value.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    if (resolve_architecture(artifact::require_id(names[0], "architecture"),
                             artifact::require_id(value.at("model_type"), "model_type")) !=
        Architecture::Glm5Next) {
        throw ArtifactError("text component is not Glm5Next");
    }
    if (value.at("tie_word_embeddings") != false) {
        throw ArtifactError("Glm5Next requires untied embeddings");
    }
    TextConfig out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.dense_layers            = dimension(value, "first_k_dense_replace");
    out.dense_intermediate      = dimension(value, "intermediate_size");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    out.swiglu_limit            = positive_float(value, "swiglu_limit");
    if (out.dense_layers >= out.num_hidden_layers) {
        throw ArtifactError("first_k_dense_replace leaves no MoE layer");
    }
    const auto& eos = value.at("eos_token_id");
    if (!eos.is_array() || eos.empty()) { throw ArtifactError("eos_token_id must list token ids"); }
    for (const auto& id : eos) {
        const auto token = integer(id, "eos", false);
        if (token >= out.vocab_size) { throw ArtifactError("eos_token_id exceeds the vocabulary"); }
        out.eos_token_ids.push_back(static_cast<std::int32_t>(token));
    }
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every text block");
    }
    for (const auto& layer : layers) {
        if (layer == "linear_attention") {
            out.layer_types.push_back(MixerKind::Kda);
            out.compact_layer_indices.push_back(out.kda_layers++);
        } else if (layer == "sparse_attention") {
            out.layer_types.push_back(MixerKind::Mla);
            out.compact_layer_indices.push_back(out.mla_layers++);
        } else {
            throw ArtifactError("unknown text layer_type");
        }
    }
    out.kda = {dimension(value, "linear_num_heads"), dimension(value, "linear_head_dim"),
               dimension(value, "linear_conv_kernel_dim"), finite_float(value, "linear_lower_bound")};
    if (!(out.kda.lower_bound >= -5.0F && out.kda.lower_bound < 0.0F)) {
        throw ArtifactError("linear_lower_bound must lie in [-5, 0)");
    }
    out.mla = {dimension(value, "num_attention_heads"), dimension(value, "q_lora_rank"),
               dimension(value, "kv_lora_rank"), dimension(value, "qk_nope_head_dim"),
               dimension(value, "v_head_dim")};
    out.indexer = {dimension(value, "index_n_heads"), dimension(value, "index_head_dim"),
                   dimension(value, "index_topk"), dimension(value, "index_kpool")};
    if (out.indexer.top_k % out.indexer.pool) {
        throw ArtifactError("index_topk must be whole pools");
    }
    out.moe = {dimension(value, "n_routed_experts"), dimension(value, "num_experts_per_tok"),
               dimension(value, "moe_intermediate_size"), dimension(value, "moe_intermediate_size"),
               positive_float(value, "routed_scaling_factor")};
    if (out.moe.top_k > out.moe.experts) { throw ArtifactError("selected experts exceed expert count"); }
    out.hc = {dimension(value, "hc_mult"), dimension(value, "hc_sinkhorn_iters"),
              positive_float(value, "hc_eps")};
    if (out.hc.streams < 2) { throw ArtifactError("hc_mult must exceed one"); }
    // The MTP layer (mtp_layers 1) stays in the artifact; only --spec mtp would bind it, and this
    // family has no MTP drafter.
    if (integer(value.at("mtp_layers"), "mtp_layers", false) > 1) {
        throw ArtifactError("mtp_layers must be 0 or 1");
    }
    return out;
}

} // namespace

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.speculative != SpeculativeBackend::None && !options.dflash2()) {
            throw ArtifactError("Glm5Next speculative decoding uses a DFlash2 drafter "
                                "(--spec dflash2 --draft-model FILE)");
        }
        if (options.dflash2() && options.draft_model.empty()) {
            throw ArtifactError("--spec dflash2 needs the DFlash2 companion artifact (--draft-model FILE)");
        }
        if (!options.dflash2() && !options.draft_model.empty()) {
            throw ArtifactError("--draft-model is the DFlash2 drafter: it needs --spec dflash2");
        }
        if (options.vision) { throw ArtifactError("Glm5Next has no vision tower yet"); }
        if (options.rope_yarn_factor != 1.0F) {
            throw ArtifactError("Glm5Next attention has no RoPE to scale (rope_yarn_factor)");
        }
        if (options.proposal_head != ProposalHead::Full) {
            throw ArtifactError("Glm5Next drafts read the full output head (no --lm-head-draft)");
        }
        Config out;
        out.text = text(directory.component("text").config);
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Glm5Next config: ") + error.what());
    }
}

DFlash2Config parse_dflash2_config(const artifact::Directory& companion, const TextConfig& target) {
    try {
        if (!companion.companion || companion.components.size() != 1 ||
            !companion.components.contains("dflash2")) {
            throw ArtifactError("the draft model must be a DFlash2 companion artifact (one dflash2 component, no text)");
        }
        const Json& value = companion.component("dflash2").config;
        require_members(value,
                        {"architectures", "model_type", "intermediate_size", "num_attention_heads",
                         "num_key_value_heads", "head_dim", "num_hidden_layers", "max_position_embeddings",
                         "rms_norm_eps", "rope_parameters", "layer_types", "sliding_window", "dflash_config",
                         "target_config"},
                        {}, "DFlash2 config");
        const auto& names = value.at("architectures");
        if (!names.is_array() || names.size() != 1 || names[0] != "DFlash2DraftModel" ||
            value.at("model_type") != "qwen3") {
            throw ArtifactError("the draft model is not a DFlash2DraftModel");
        }
        // The target it was converted for: this architecture and geometry.
        const Json& t = value.at("target_config");
        require_members(t, {"architectures", "hidden_size", "vocab_size", "num_hidden_layers"}, {},
                        "DFlash2 target");
        if (!t.at("architectures").is_array() || t.at("architectures").size() != 1 ||
            t.at("architectures")[0] != "Glm5NextForCausalLM" ||
            dimension(t, "hidden_size") != target.hidden_size ||
            dimension(t, "vocab_size") != target.vocab_size ||
            dimension(t, "num_hidden_layers") != target.num_hidden_layers) {
            throw ArtifactError("the DFlash2 drafter was converted for another target (" + t.dump() + ")");
        }
        DFlash2Config out;
        out.layers       = dimension(value, "num_hidden_layers");
        out.heads        = dimension(value, "num_attention_heads");
        out.kv_heads     = dimension(value, "num_key_value_heads");
        out.head_dim     = dimension(value, "head_dim");
        out.intermediate = dimension(value, "intermediate_size");
        out.window       = dimension(value, "sliding_window");
        out.rms_norm_eps = positive_float(value, "rms_norm_eps");
        require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "DFlash2 RoPE");
        out.rope_theta = positive_float(value.at("rope_parameters"), "rope_theta");
        if (out.heads % out.kv_heads || out.head_dim % 2) {
            throw ArtifactError("invalid DFlash2 attention geometry");
        }
        const auto& layers = value.at("layer_types");
        if (!layers.is_array() || layers.size() != out.layers) {
            throw ArtifactError("DFlash2 layer_types length mismatch");
        }
        for (const auto& layer : layers) {
            if (layer != "sliding_attention") {
                throw ArtifactError("the Glm5Next DFlash2 drafter has sliding-window layers only");
            }
        }
        const Json& d = value.at("dflash_config");
        require_members(d,
                        {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size",
                         "selector_rank", "selector_top_k"},
                        {}, "DFlash2 dflash_config");
        const auto& taps = d.at("target_layer_ids");
        if (!taps.is_array() || taps.empty()) { throw ArtifactError("target_layer_ids must be nonempty"); }
        for (const auto& tap : taps) {
            const auto layer = integer(tap, "target layer", false);
            if (layer >= target.num_hidden_layers ||
                (!out.target_layer_ids.empty() && layer <= out.target_layer_ids.back())) {
                throw ArtifactError("target_layer_ids must be ascending target layers");
            }
            out.target_layer_ids.push_back(layer);
        }
        out.mask_token_id  = integer(d.at("mask_token_id"), "mask token", false);
        out.conv_kernel    = dimension(d, "conv_kernel_size");
        out.conv_group     = dimension(d, "conv_group_size");
        out.selector_rank  = dimension(d, "selector_rank");
        out.selector_top_k = dimension(d, "selector_top_k");
        if (out.mask_token_id >= target.vocab_size) {
            throw ArtifactError("the mask token exceeds the embedding rows");
        }
        if (target.hidden_size % out.conv_group) {
            throw ArtifactError("the DFlash2 conv group does not divide the hidden width");
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("DFlash2 companion config: ") + error.what());
    }
}

} // namespace infernix::models::glm5_next
