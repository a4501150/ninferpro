#include "models/glm5_next/frontend/encoding.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace infernix::models::glm5_next::frontend {

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered,
                                 std::size_t maximum_tokens) {
    if (!rendered.media_placeholders.empty() || !rendered.media_token_runs.empty()) {
        throw std::logic_error("a GLM prompt holds no media");
    }
    EncodedChat encoded;
    std::vector<std::size_t> byte_boundaries;
    byte_boundaries.reserve((rendered.rewrite_checkpoint ? 2U : 0U) +
                            rendered.rewrite_execution_boundaries.size() +
                            rendered.message_boundaries.size() + rendered.cache_boundaries.size());
    if (rendered.rewrite_checkpoint) {
        byte_boundaries.push_back(rendered.rewrite_checkpoint->offset);
        byte_boundaries.push_back(rendered.rewrite_checkpoint->recovery_offset);
    }
    byte_boundaries.insert(byte_boundaries.end(), rendered.rewrite_execution_boundaries.begin(),
                           rendered.rewrite_execution_boundaries.end());
    for (const std::optional<std::size_t> boundary : rendered.message_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }
    for (const std::optional<std::size_t> boundary : rendered.cache_boundaries) {
        if (boundary) { byte_boundaries.push_back(*boundary); }
    }

    BoundaryEncodedText tokenized = tokenizer.encode_with_boundaries(
        rendered.text, byte_boundaries, EncodeOptions{.max_tokens = maximum_tokens},
        rendered.literal_spans);
    encoded.input_ids = std::move(tokenized.input_ids);
    if (encoded.input_ids.size() == maximum_tokens) { return encoded; }
    std::size_t boundary_index = 0;
    const auto to_frontier     = [](std::size_t frontier, std::string_view kind) {
        if (frontier > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(std::string(kind) + " token frontier exceeds uint32");
        }
        return static_cast<std::uint32_t>(frontier);
    };
    if (rendered.rewrite_checkpoint) {
        const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
        const TokenBoundaryResult& recovery = tokenized.boundaries.at(boundary_index++);
        if (boundary.exact_frontier && *boundary.exact_frontier > 0) {
            const auto frontier = to_frontier(*boundary.exact_frontier, "rewrite checkpoint");
            encoded.rewrite_checkpoint = RewriteCheckpointSpec{
                .kind              = rendered.rewrite_checkpoint->kind,
                .frontier          = frontier,
                .recovery_frontier = recovery.exact_frontier && *recovery.exact_frontier > 0 &&
                                             *recovery.exact_frontier <= frontier
                                         ? to_frontier(*recovery.exact_frontier, "input recovery")
                                         : frontier};
        }
    }
    encoded.rewrite_execution_frontiers.reserve(rendered.rewrite_execution_boundaries.size());
    for (std::size_t remaining = rendered.rewrite_execution_boundaries.size(); remaining != 0;
         --remaining) {
        const TokenBoundaryResult& result = tokenized.boundaries.at(boundary_index++);
        const std::optional<std::uint32_t> frontier =
            result.exact_frontier ? std::optional<std::uint32_t>(to_frontier(
                                        *result.exact_frontier, "rewrite execution boundary"))
                                  : std::nullopt;
        if (frontier && *frontier != 0 &&
            (encoded.rewrite_execution_frontiers.empty() ||
             encoded.rewrite_execution_frontiers.back() != *frontier)) {
            encoded.rewrite_execution_frontiers.push_back(*frontier);
        }
    }
    encoded.message_boundaries.resize(rendered.message_boundaries.size());
    for (std::size_t index = 0; index < rendered.message_boundaries.size(); ++index) {
        if (rendered.message_boundaries[index]) {
            const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier) {
                encoded.message_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "message boundary");
            }
        }
    }
    encoded.cache_boundaries.resize(rendered.cache_boundaries.size());
    for (std::size_t index = 0; index < rendered.cache_boundaries.size(); ++index) {
        if (rendered.cache_boundaries[index]) {
            const TokenBoundaryResult& boundary = tokenized.boundaries.at(boundary_index++);
            if (boundary.exact_frontier)
                encoded.cache_boundaries[index] =
                    to_frontier(*boundary.exact_frontier, "cache boundary");
        }
    }
    if (boundary_index != tokenized.boundaries.size()) {
        throw std::logic_error("rendered token boundary result count changed during encoding");
    }
    return encoded;
}

} // namespace infernix::models::glm5_next::frontend
