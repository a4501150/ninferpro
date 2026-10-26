#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "infernix/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <functional>

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

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
infernix::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

// The thinking budget ServeOptions::thinking_answer_room gives a thinking request that has no
// budget of its own, or none when the answer room is off or the output is too short. The output
// is the request's limit bounded by the context the prepared prompt leaves, so a prompt near
// max_context still leaves its answer a quarter of what can actually be generated.
std::optional<std::uint32_t> thinking_answer_room_budget(const ServeOptions& server,
                                                         std::uint32_t requested_output_tokens,
                                                         std::uint32_t prompt_tokens);

} // namespace infernix::serve
