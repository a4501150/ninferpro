#include "models/qwen3_5/measurement.h"

#include "models/qwen_common/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <vector>

namespace infernix::models::qwen3_5 {
namespace {

constexpr runtime::ContextPrefillCost kGroupwise27bPrefill{
    .chunk_ns              = 40'813'570,
    .token_ns_q32          = 1'012'273'154'411'951,
    .attention_pair_ns_q32 = 30'497'396'515,
    .vision_item_ns        = 5'986'585,
    .vision_patch_ns_q32   = 23'710'212'854'694,
};
constexpr runtime::ContextPrefillCost kNvfp4Fp8Prefill{
    .chunk_ns              = 14'672'989,
    .token_ns_q32          = 375'800'765'711'778,
    .attention_pair_ns_q32 = 8'200'474'657,
    .vision_item_ns        = 5'860'255,
    .vision_patch_ns_q32   = 23'832'529'381'413,
};

// Existing measurements apply to these Text bindings with Vision disabled/enabled. The signature
// excludes trained values and the selected speculative backend; it describes the primary
// reconstruction work priced by ContextPrefillCost.
constexpr std::array<runtime::ContextCompiledPrefill, 4> kCompiledPrefill{{
    {"nvidia-geforce-rtx-5090-sm120",
     "200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2", kGroupwise27bPrefill},
    {"nvidia-geforce-rtx-5090-sm120",
     "badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b", kGroupwise27bPrefill},
    {"nvidia-geforce-rtx-5090-sm120",
     "e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a", kNvfp4Fp8Prefill},
    {"nvidia-geforce-rtx-5090-sm120",
     "953e22d9b9a6639c09f7389d49084b059dea07720bc0121dfd9c154002459915", kNvfp4Fp8Prefill},
}};

} // namespace

runtime::ContextPrefillDefaults context_prefill_defaults() {
    // The slower measured 27B configuration is the conservative estimate for an unmeasured
    // configuration: it still receives continuous recomputation costs.
    return {.fallback = kGroupwise27bPrefill, .compiled = kCompiledPrefill};
}

std::string prefill_signature(const Model& model) {
    using Json         = nlohmann::json;
    const auto& config = model.config();
    const auto& t      = config.text;
    Json facts{{"implementation", "qwen3_5-prefill-1"},
               {"architecture", architecture_name(t.architecture)},
               {"hidden_size", t.hidden_size},
               {"vocab_size", t.vocab_size},
               {"public_token_count", model.resources().public_token_count},
               {"num_hidden_layers", t.num_hidden_layers},
               {"rms_norm_eps", t.rms_norm_eps},
               {"layer_types", t.layer_types}};
    if (t.attention) {
        const auto& a      = *t.attention;
        facts["attention"] = {a.num_attention_heads, a.num_key_value_heads, a.head_dim};
    }
    if (t.rope_parameters) {
        const auto& r = *t.rope_parameters;
        facts["rope"] = {r.rope_theta, r.rotary_dim, r.pair_axes};
    }
    if (t.gdn) {
        const auto& g = *t.gdn;
        facts["gdn"]  = {g.linear_num_key_heads, g.linear_key_head_dim, g.linear_num_value_heads,
                         g.linear_value_head_dim, g.linear_conv_kernel_dim};
    }
    if (const auto* d = std::get_if<DenseConfig>(&t.ffn)) {
        facts["ffn"] = {d->intermediate_size};
    } else {
        const auto& m = std::get<MoeConfig>(t.ffn);
        facts["ffn"]  = {m.num_experts, m.num_experts_per_tok, m.moe_intermediate_size,
                         m.shared_expert_intermediate_size};
    }
    if (config.vision) {
        const auto& v   = *config.vision;
        facts["vision"] = {
            v.depth,      v.hidden_size,         v.intermediate_size,  v.num_heads,
            v.patch_size, v.temporal_patch_size, v.spatial_merge_size, v.num_position_embeddings};
    }
    std::vector<const BoundWeight*> weights;
    // The context-cost model prices primary Text/Vision reconstruction. The coefficients were
    // measured for that work; optional draft/proposal calls do not change the baseline's
    // applicability.
    for (const auto& weight : model.weight_data()) {
        if (weight.name.starts_with("text/") || weight.name.starts_with("vision/")) {
            weights.push_back(&weight);
        }
    }
    std::sort(weights.begin(), weights.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });
    std::map<const WeightParent*, std::size_t> parents;
    auto& inventory = facts["weights"] = Json::array();
    for (const auto* weight : weights) {
        Json item{{"role", weight->name},
                  {"shape", weight->view.shape},
                  {"parts", Json::array()},
                  {"uses", Json::array()}};
        for (const auto& part : weight->view.parts) {
            const auto [it, inserted] = parents.emplace(part.parent, parents.size());
            const auto& geometry      = part.parent->geometry;
            item["parts"].push_back({it->second, part.begin, part.end, geometry.format,
                                     geometry.layout, geometry.shape, geometry.padded_columns,
                                     geometry.group_size});
        }
        for (const auto& use : weight->uses) {
            if (!use.input.starts_with("text/") && !use.input.starts_with("vision/")) { continue; }
            item["uses"].push_back(
                {use.input, use.policy, use.activation_input_divisor.has_value()});
        }
        inventory.push_back(std::move(item));
    }
    return qwen_common::frontend::sha256_hex(qwen_common::frontend::sha256(facts.dump()));
}

} // namespace infernix::models::qwen3_5
