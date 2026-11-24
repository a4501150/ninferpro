#pragma once

#include "models/glm5_next/frontend/tokenizer.h"
#include <cstdint>
#include <memory>
#include <string_view>

namespace infernix::models::glm5_next {

// Byte views borrow the model's Host backing. Tokenizer owns its decoded vocabulary and tables;
// loading and request preparation share this one immutable interpretation.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::shared_ptr<const frontend::Tokenizer> tokenizer;
    std::uint32_t public_token_count = 0;
};

// The model geometry a family checks its frontend resources against: the embedding rows.
struct FrontendDomain {
    std::uint32_t vocab_size = 0;
};

void parse_resources(FrontendResources& resources, const FrontendDomain& domain);

} // namespace infernix::models::glm5_next
