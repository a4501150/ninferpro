// The glm5_next load plan on the real converted artifact: every converter binding resolves to the
// shapes and formats the loader expects, and the parsed config is GLM-5.3-Flash's. Nothing is
// read beyond the directory, so no GPU is needed. INFERNIX_GLM_ARTIFACT names the target artifact
// and INFERNIX_GLM_DRAFT (optional) the DFlash2 companion; without the first the test skips (77).
#include "artifact/reader.h"
#include "models/glm5_next/load.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

namespace glm = infernix::models::glm5_next;

int failures = 0;

template <typename T>
void expect(const char* what, const T& actual, const T& expected) {
    if (actual == expected) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s is %s, expected %s\n", what, std::to_string(actual).c_str(),
                 std::to_string(expected).c_str());
}

void check_text(const glm::TextConfig& t) {
    expect("hidden_size", t.hidden_size, 4096U);
    expect("num_hidden_layers", t.num_hidden_layers, 45U);
    expect("dense_layers", t.dense_layers, 3U);
    expect("moe.experts", t.moe.experts, 288U);
    expect("moe.top_k", t.moe.top_k, 8U);
    expect("moe.intermediate", t.moe.intermediate, 2048U);
    expect("moe.shared_intermediate", t.moe.shared_intermediate, 2048U);
    expect("moe.routed_scale", t.moe.routed_scale, 2.5F);
    expect("swiglu_limit", t.swiglu_limit, 10.0F);
    expect("hc.streams", t.hc.streams, 4U);
    expect("hc.sinkhorn_iters", t.hc.sinkhorn_iters, 20U);
    expect("mla.heads", t.mla.heads, 64U);
    expect("mla.q_lora", t.mla.q_lora, 1536U);
    expect("mla.kv_lora", t.mla.kv_lora, 512U);
    expect("mla.head_dim", t.mla.head_dim, 256U);
    expect("mla.value_dim", t.mla.value_dim, 256U);
    expect("indexer.heads", t.indexer.heads, 32U);
    expect("indexer.head_dim", t.indexer.head_dim, 128U);
    expect("indexer.top_k", t.indexer.top_k, 2048U);
    expect("indexer.pool", t.indexer.pool, 4U);
    expect("kda.heads", t.kda.heads, 64U);
    expect("kda.head_dim", t.kda.head_dim, 128U);
    expect("kda.conv_kernel", t.kda.conv_kernel, 4U);
    expect("kda.lower_bound", t.kda.lower_bound, -5.0F);
    expect("layer_types", t.layer_types.size(), std::size_t{t.num_hidden_layers});
    expect("kda + mla layers", t.kda_layers + t.mla_layers, t.num_hidden_layers);
    std::printf("layers: %u KDA, %u MLA; vocab %u, context %u\n", t.kda_layers, t.mla_layers,
                t.vocab_size, t.max_position_embeddings);
}

void check_dflash2(const glm::DFlash2Config& d) {
    expect("dflash2.layers", d.layers, 5U);
    expect("dflash2.heads", d.heads, 32U);
    expect("dflash2.kv_heads", d.kv_heads, 8U);
    expect("dflash2.head_dim", d.head_dim, 128U);
    expect("dflash2.intermediate", d.intermediate, 12288U);
    expect("dflash2.window", d.window, 2048U);
    expect("dflash2.mask_token_id", d.mask_token_id, 154856U);
    expect("dflash2.conv_kernel", d.conv_kernel, 2U);
    expect("dflash2.conv_group", d.conv_group, 16U);
    expect("dflash2.selector_rank", d.selector_rank, 256U);
    expect("dflash2.selector_top_k", d.selector_top_k, 16U);
    const std::vector<std::uint32_t> taps{5, 14, 24, 33, 42};
    if (d.target_layer_ids != taps) {
        ++failures;
        std::fprintf(stderr, "FAIL: dflash2.target_layer_ids differ from 5,14,24,33,42\n");
    }
}

void report(const char* label, const glm::LoadPlan& plan) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("%s: device %.2f GiB, pinned experts %.2f GiB, pinned other %.2f GiB, %u public tokens\n",
                label, static_cast<double>(plan.device_bytes()) / kGiB,
                static_cast<double>(plan.pinned_expert_bytes()) / kGiB,
                static_cast<double>(plan.pinned_other_bytes()) / kGiB, plan.public_token_count());
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_GLM_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_GLM_ARTIFACT\n");
        return 77;
    }
    try {
        infernix::artifact::Reader reader(artifact);
        if (!glm::is_glm5_next(reader)) throw std::invalid_argument("artifact is not Glm5Next");

        const auto plan = glm::plan_load(reader);
        check_text(plan.config().text);
        if (plan.config().dflash2) {
            ++failures;
            std::fprintf(stderr, "FAIL: a drafter is bound without --draft-model\n");
        }
        report("target", plan);

        if (const char* draft = std::getenv("INFERNIX_GLM_DRAFT")) {
            infernix::models::LoadOptions options;
            options.speculative = infernix::SpeculativeBackend::DFlash2;
            options.draft_model = draft;
            const auto drafted = glm::plan_load(reader, options);
            if (!drafted.config().dflash2) {
                ++failures;
                std::fprintf(stderr, "FAIL: --draft-model bound no DFlash2 drafter\n");
            } else {
                check_dflash2(*drafted.config().dflash2);
            }
            report("with DFlash2", drafted);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
