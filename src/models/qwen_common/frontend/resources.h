#pragma once

#include "models/qwen_common/frontend/tokenizer.h"
#include "models/qwen_common/vision_config.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace infernix::models::qwen_common {

// Byte views borrow the model's Host backing. Tokenizer owns its decoded vocabulary and tables;
// loading and request preparation share this one immutable interpretation.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::string_view preprocessor_config_json;
    std::string_view video_preprocessor_config_json;
    std::shared_ptr<const frontend::Tokenizer> tokenizer;
    std::uint32_t public_token_count = 0;
};

// The model geometry a family checks its frontend resources against: the embedding rows, the
// bound Vision tower and a bound DFlash2 drafter's selector width.
struct FrontendDomain {
    std::uint32_t vocab_size   = 0;
    const VisionConfig* vision = nullptr;
    std::optional<std::uint32_t> selector_top_k;
};

void parse_resources(FrontendResources& resources, const FrontendDomain& domain);

} // namespace infernix::models::qwen_common
