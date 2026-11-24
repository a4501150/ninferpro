#pragma once

// The GLM-5.3-Flash media processor: Glm5NextImageProcessor and Glm5NextVideoProcessor (smart
// resize to whole 28-pixel merge blocks, content resize and zero padding, CLIP normalization,
// 14-pixel patches in merge-block order) and Glm5NextProcessor's placeholder expansion.

#include "models/glm5_next/frontend/chat_template.h"
#include "models/glm5_next/frontend/encoding.h"
#include "models/glm5_next/frontend/prepared_prompt.h"
#include "models/glm5_next/frontend/tokenizer.h"

#include "infernix/types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::glm5_next::frontend {

class MediaPreprocessCache;

enum class ProcessorErrorKind {
    BudgetExceeded,
    ContextLengthExceeded,
    InvalidMedia,
};

class ProcessorError final : public std::runtime_error {
public:
    ProcessorError(ProcessorErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    // ProcessorErrorKind::ContextLengthExceeded with its counts.
    explicit ProcessorError(ContextLengthDetails details)
        : std::runtime_error(RequestError(details).what()),
          kind_(ProcessorErrorKind::ContextLengthExceeded), context_length_(details) {}

    [[nodiscard]] ProcessorErrorKind kind() const noexcept { return kind_; }

    [[nodiscard]] const std::optional<ContextLengthDetails>& context_length() const noexcept {
        return context_length_;
    }

private:
    ProcessorErrorKind kind_;
    std::optional<ContextLengthDetails> context_length_;
};

struct ProcessorOptions {
    // Merged-token bounds of one image and of one whole video (min_image_tokens and
    // max_image_tokens of processor_config.json's two processors).
    std::uint64_t image_min_tokens = 16;
    std::uint64_t image_max_tokens = 8000;
    std::uint64_t video_min_tokens = 16;
    std::uint64_t video_max_tokens = 240000;
    double video_fps               = 2.0;
    int video_max_frames           = 2048;
    // Encoded bytes are aggregate per prompt. Decode limits are per item; the fixed worker pool
    // bounds concurrently decoded media.
    std::size_t max_encoded_media_bytes    = kMaximumPromptMediaBytes;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::uint64_t max_raw_patches          = kMaximumPromptVisionRawPatches;
    std::uint64_t max_vision_tokens        = kMaximumPromptVisionTokens;
};

struct ProcessedInput {
    bool starts_in_reasoning = false;
    std::string continuation_content;
    std::vector<int> input_ids;
    // Glm5NextProcessor's mm_token_type_ids: 1 for an image's tokens, 2 for a video's.
    std::vector<std::uint8_t> token_types;
    std::vector<VisionItem> vision_items;
    // One immutable [patches, 1176] BF16 payload per Vision item.
    std::vector<std::shared_ptr<const PreparedMediaPayload>> media_payloads;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    PrepareStats stats;
};

class Processor {
public:
    Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              const ProcessorOptions& options, std::shared_ptr<MediaPreprocessCache> media_cache);

    [[nodiscard]] std::size_t count_tokens(std::vector<ChatMessage> messages,
                                           ChatRenderOptions render_options  = {},
                                           const PreparationControl& control = {}) const;

    ProcessedInput
    process(std::vector<ChatMessage> messages, ChatRenderOptions render_options = {},
            const PreparationControl& control = {},
            std::size_t maximum_prompt_tokens = std::numeric_limits<std::size_t>::max()) const;

private:
    const Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    const ProcessorOptions& options_;
    std::shared_ptr<MediaPreprocessCache> media_cache_;
    TokenId image_token_ = -1;
};

} // namespace infernix::models::glm5_next::frontend
