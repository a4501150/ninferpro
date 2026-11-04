#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "infernix/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <functional>
#include <span>
#include <string>
#include <vector>

namespace infernix::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<infernix::OwnedMedia(const ContentPart&)>;

struct ResolvedPromptSemantics {
    std::optional<bool> enable_thinking;
    std::optional<infernix::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server);

infernix::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media);

// The text a turn that loads tools ends with: the definitions in the function form of the listed
// tools, framed like the prompt's tool section. It depends on the definitions alone, so a history
// replayed by a later request renders the same tokens.
std::string render_loaded_tools(std::span<const ToolDefinition> tools);

// Each declaration's protocol schema path in Engine declaration order (listed tools, then deferred
// ones), for mapping an Engine tool-schema error back to the request.
std::vector<std::string> tool_declaration_params(const GenerationRequest& req);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
infernix::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

} // namespace infernix::serve
