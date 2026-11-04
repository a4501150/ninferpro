#include "serve/server_tools.h"

#include <algorithm>
#include <utility>

namespace infernix::serve {

GenerationOutcome combine_round_outcomes(std::span<const ServerToolRound> rounds,
                                         GenerationOutcome final_round) {
    if (rounds.empty()) { return final_round; }
    GenerationOutcome total = std::move(final_round);
    std::vector<infernix::TokenId> generated;
    for (const ServerToolRound& round : rounds) {
        const GenerationOutcome& earlier = round.outcome;
        total.prompt_tokens += earlier.prompt_tokens;
        total.completion_tokens += earlier.completion_tokens;
        total.reasoning_tokens += earlier.reasoning_tokens;
        generated.insert(generated.end(), earlier.generated_token_ids.begin(),
                         earlier.generated_token_ids.end());

        GenerationMetrics& total_metrics = total.metrics;
        const GenerationMetrics& earlier_metrics = earlier.metrics;
        total_metrics.computed_prefill_tokens += earlier_metrics.computed_prefill_tokens;
        total_metrics.prepare_seconds += earlier_metrics.prepare_seconds;
        total_metrics.vision_seconds += earlier_metrics.vision_seconds;
        total_metrics.prefill_seconds += earlier_metrics.prefill_seconds;
        total_metrics.decode_seconds += earlier_metrics.decode_seconds;
        total_metrics.decode_share_seconds += earlier_metrics.decode_share_seconds;
        total_metrics.prompt_wall_seconds += earlier_metrics.prompt_wall_seconds;
        total_metrics.generation_wall_seconds += earlier_metrics.generation_wall_seconds;
        total_metrics.total_seconds += earlier_metrics.total_seconds;
        total_metrics.vision_offload_window_seconds += earlier_metrics.vision_offload_window_seconds;
        total_metrics.vision_offload_evict_seconds += earlier_metrics.vision_offload_evict_seconds;
        total_metrics.vision_offload_restore_seconds +=
            earlier_metrics.vision_offload_restore_seconds;
        total_metrics.vision_offload_evicted_bytes += earlier_metrics.vision_offload_evicted_bytes;
        total_metrics.vision_offload_staged_bytes += earlier_metrics.vision_offload_staged_bytes;

        total_metrics.speculative_rounds += earlier_metrics.speculative_rounds;
        total_metrics.speculative_draft_tokens += earlier_metrics.speculative_draft_tokens;
        total_metrics.speculative_accepted_tokens += earlier_metrics.speculative_accepted_tokens;
        total_metrics.speculative_fallback_steps += earlier_metrics.speculative_fallback_steps;
        for (std::size_t position = 0; position < earlier_metrics.speculative_accepted_per_position
                                              .size();
             ++position) {
            if (total_metrics.speculative_accepted_per_position.size() <= position) {
                total_metrics.speculative_accepted_per_position.resize(position + 1U, 0);
            }
            total_metrics.speculative_accepted_per_position[position] +=
                earlier_metrics.speculative_accepted_per_position[position];
        }
        for (std::size_t length = 0;
             length < earlier_metrics.speculative_rounds_by_draft_length.size(); ++length) {
            if (total_metrics.speculative_rounds_by_draft_length.size() <= length) {
                total_metrics.speculative_rounds_by_draft_length.resize(length + 1U, 0);
            }
            total_metrics.speculative_rounds_by_draft_length[length] +=
                earlier_metrics.speculative_rounds_by_draft_length[length];
        }

        total_metrics.ngram_rounds += earlier_metrics.ngram_rounds;
        total_metrics.ngram_drafted_tokens += earlier_metrics.ngram_drafted_tokens;
        total_metrics.ngram_accepted_tokens += earlier_metrics.ngram_accepted_tokens;
        total_metrics.ngram_archive_rounds += earlier_metrics.ngram_archive_rounds;
        total_metrics.ngram_archive_drafted_tokens += earlier_metrics.ngram_archive_drafted_tokens;
        total_metrics.ngram_archive_accepted_tokens +=
            earlier_metrics.ngram_archive_accepted_tokens;
    }
    // The response's first token came from the first generation.
    total.metrics.ttft_seconds = rounds.front().outcome.metrics.ttft_seconds;
    generated.insert(generated.end(), total.generated_token_ids.begin(),
                     total.generated_token_ids.end());
    total.generated_token_ids = std::move(generated);
    return total;
}

void append_server_round(GenerationRequest& request, const ServerToolRound& round,
                         std::span<const std::string> server_tool_names) {
    const GenerationOutcome& outcome = round.outcome;
    ContentPart text;
    text.kind     = ContentKind::Text;
    text.type_raw = "text";
    text.text     = outcome.text;

    // An Assistant prefill was continued by the round's output: the call ends that same turn.
    const bool continued = request.continuation ==
                               infernix::PromptContinuationMode::ContinueFinalAssistant &&
                           !request.messages.empty() &&
                           request.messages.back().role == ChatRole::Assistant;
    if (continued) {
        ChatTurn& assistant = request.messages.back();
        assistant.reasoning_content += outcome.reasoning;
        if (!text.text.empty()) { assistant.content.push_back(std::move(text)); }
        assistant.tool_calls.insert(assistant.tool_calls.end(), round.calls.begin(),
                                    round.calls.end());
    } else {
        ChatTurn assistant;
        assistant.role              = ChatRole::Assistant;
        assistant.reasoning_content = outcome.reasoning;
        if (!text.text.empty()) { assistant.content.push_back(std::move(text)); }
        assistant.tool_calls = round.calls;
        request.messages.push_back(std::move(assistant));
    }
    request.messages.insert(request.messages.end(), round.results.begin(), round.results.end());
    request.continuation = infernix::PromptContinuationMode::NewAssistantTurn;
    request.max_tokens   = std::max(0, request.max_tokens - outcome.completion_tokens);

    ToolChoice& choice = request.tool_choice;
    if (choice.mode == ToolChoiceMode::Required) { choice.mode = ToolChoiceMode::Auto; }
    if (choice.allowed_names &&
        std::all_of(choice.allowed_names->begin(), choice.allowed_names->end(),
                    [&](const std::string& name) {
                        return std::find(server_tool_names.begin(), server_tool_names.end(),
                                         name) != server_tool_names.end();
                    })) {
        choice.allowed_names.reset();
    }
}

} // namespace infernix::serve
