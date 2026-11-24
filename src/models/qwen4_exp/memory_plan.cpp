#include "models/qwen4_exp/memory_plan.h"

namespace infernix::models::qwen4_exp {

std::uint32_t graph_bound(std::uint32_t lanes, std::uint32_t max_width, std::uint32_t mtp_draft_tokens,
                          bool sampled_drafts, std::uint32_t dflash2_draft_tokens) {
    std::uint32_t graphs = lanes;
    if (max_width > 1) { graphs += lanes * max_width; }
    if (mtp_draft_tokens > 0) {
        graphs += lanes * max_width + lanes * mtp_draft_tokens * (sampled_drafts ? 2 : 1);
    }
    if (dflash2_draft_tokens > 0) { graphs += lanes + lanes * max_width; }
    return graphs;
}

} // namespace infernix::models::qwen4_exp
