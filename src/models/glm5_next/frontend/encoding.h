#pragma once

#include "models/glm5_next/frontend/chat_template.h"
#include "models/glm5_next/frontend/prepared_prompt.h"
#include "models/glm5_next/frontend/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace infernix::models::glm5_next::frontend {

// The bound on recounting an overlong prompt for its error: four times the limit, so a prompt that
// overflows by up to 4x reports its exact length without unbounded tokenization work.
[[nodiscard]] inline std::size_t context_recount_limit(std::size_t maximum_prompt_tokens) noexcept {
    return maximum_prompt_tokens > (std::numeric_limits<std::size_t>::max() - 1U) / 4U
               ? std::numeric_limits<std::size_t>::max()
               : 4U * maximum_prompt_tokens + 1U;
}

// The details for a prompt counted with context_recount_limit(maximum_prompt_tokens).
[[nodiscard]] inline ContextLengthDetails context_length_details(std::size_t counted,
                                                                 std::size_t maximum_prompt_tokens) {
    const std::size_t limit = context_recount_limit(maximum_prompt_tokens);
    return {.prompt_tokens = counted >= limit ? limit - 1U : counted,
            .max_context   = static_cast<std::uint32_t>(maximum_prompt_tokens),
            .exact         = counted < limit};
}

// A rendered chat's token ids, with its byte boundaries mapped to the token frontiers that fall
// exactly on them, and each expanded media run's tokens.
struct EncodedChat {
    std::vector<int> input_ids;

    struct MediaTokenRun {
        TokenSpan tokens;
        Modality modality       = Modality::Image;
        std::size_t item_index  = 0;
        std::size_t frame_index = 0;
    };

    std::vector<MediaTokenRun> media_token_runs;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
};

// Every media run must be expanded (media_placeholders empty) and encode to image_token only.
EncodedChat
encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered,
                     std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max(),
                     TokenId image_token        = -1);

} // namespace infernix::models::glm5_next::frontend
