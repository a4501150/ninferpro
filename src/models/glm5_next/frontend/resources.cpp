#include "models/glm5_next/frontend/resources.h"

#include "artifact/schema.h"

namespace infernix::models::glm5_next {

void parse_resources(FrontendResources& resources, const FrontendDomain& domain) {
    resources.tokenizer = std::make_shared<const frontend::Tokenizer>(
        frontend::TokenizerResources{resources.tokenizer_json, resources.tokenizer_config_json,
                                     resources.generation_config_json});
    const auto count = resources.tokenizer->vocab_size();
    if (!count || count > domain.vocab_size ||
        !resources.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    resources.public_token_count = static_cast<std::uint32_t>(count);
    for (const auto token : resources.tokenizer->default_stop_token_ids()) {
        if (!resources.tokenizer->is_valid_token(token)) {
            throw artifact::ArtifactError("stop token is outside the public token domain");
        }
    }
}

} // namespace infernix::models::glm5_next
