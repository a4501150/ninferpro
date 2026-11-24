#include "models/glm5_next/frontend/frontend.h"

#include "models/glm5_next/frontend/resources.h"
#include "models/glm5_next/frontend/prepared_prompt.h"

#include "models/glm5_next/frontend/chat_template.h"
#include "models/glm5_next/frontend/encoding.h"
#include "models/glm5_next/frontend/test_access.h"
#include "models/glm5_next/frontend/tokenizer.h"
#include "models/glm5_next/frontend/tool_call_parser.h"
#include "models/common/frontend/tool_contract.h"
#include "models/glm5_next/frontend/tool_grammar.h"
#include "models/glm5_next/program/prefix/block_keys.h"
#include "text/unicode.h"
#include "text/grammar.h"
#include <mutex>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::models::glm5_next {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend;

constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>";
static_assert(kThinkingControl.ends_with(fi::kCanonicalReasoningCloseSerialization));
Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

double number_or_default(const Json& object, std::string_view field, std::string_view resource,
                         double default_value) {
    const std::string key(field);
    return object.contains(key) ? require_number(object, field, resource) : default_value;
}

std::int64_t integer_or_default(const Json& object, std::string_view field,
                                std::string_view resource, std::int64_t default_value) {
    const std::string key(field);
    return object.contains(key) ? require_integer(object, field, resource) : default_value;
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_tokenizer_config(const FrontendResources& resources) {
    const Json tokenizer_config =
        parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    // The tokenizers backend adds neither unless the config asks; tokenizer.json's own
    // post-processor and pre-tokenizer add nothing either (the Tokenizer checks those).
    if (tokenizer_config.value("add_bos_token", false) ||
        tokenizer_config.value("add_prefix_space", false)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match the GLM tokenizer prefix semantics");
    }
    if (!tokenizer_config.contains("pad_token") || !tokenizer_config.at("pad_token").is_string() ||
        tokenizer_config.at("pad_token").get<std::string>() != "<|endoftext|>") {
        throw std::invalid_argument(
            "tokenizer_config.json does not use the official <|endoftext|> pad token");
    }
}

fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources,
                                               const std::filesystem::path& override_path) {
    validate_tokenizer_config(resources);
    nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
    const auto config             = nlohmann::ordered_json::parse(resources.tokenizer_config_json);
    for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                            "cls_token", "mask_token"}) {
        if (!config.contains(key) || config[key].is_null()) continue;
        const auto& value = config[key];
        if (value.is_string())
            tokens[key] = value;
        else if (value.is_object() && value.contains("content"))
            tokens[key] = value.at("content");
    }
    if (config.contains("additional_special_tokens"))
        tokens["additional_special_tokens"] = config["additional_special_tokens"];
    std::string source(resources.chat_template_jinja);
    std::string name = "artifact:chat_template.jinja";
    if (!override_path.empty()) {
        name = override_path.string();
        std::ifstream file(override_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::invalid_argument("cannot read chat template: " + name);
        const auto size = file.tellg();
        if (size <= 0 || size > 16 * 1024 * 1024) {
            throw std::invalid_argument(
                "chat template must be a nonempty UTF-8 source file up to 16 MiB: " + name);
        }
        source.resize(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(source.data(), static_cast<std::streamsize>(source.size()))) {
            throw std::invalid_argument("cannot read chat template: " + name);
        }
    }
    return fi::CompiledChatTemplate::resolve(source, std::move(name), std::move(tokens));
}

[[noreturn]] void throw_context_length_exceeded(std::size_t counted, std::uint32_t max_context) {
    throw RequestError(fi::context_length_details(counted, max_context));
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role              = source.role;
        target.reasoning_content = std::move(source.reasoning_content);
        target.tool_call_id      = std::move(source.tool_call_id);
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            if (part.kind != MessagePartKind::Text) {
                throw std::invalid_argument(
                    "GLM-5.3-Flash is a text-only model: image, video and audio inputs are not "
                    "supported");
            }
            target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options,
                                     std::span<const PromptCacheMarker> cache_markers = {}) {
    fi::ChatRenderOptions rendered{.continuation              = options.continuation,
                                   .enable_thinking           = options.enable_thinking,
                                   .reasoning_effort          = options.reasoning_effort,
                                   .preserve_thinking         = options.preserve_thinking,
                                   .chat_template_kwargs_json = options.chat_template_kwargs_json,
                                   .add_vision_id             = options.add_vision_id,
                                   .tool_jsons                = options.tool_jsons};
    rendered.cache_markers.assign(cache_markers.begin(), cache_markers.end());
    return rendered;
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("prepared prompt must contain at least one token");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("prepared prompt token count exceeds the target domain");
    }
    return static_cast<std::uint32_t>(count);
}

void assign_text_positions(PreparedPromptData& prompt) {
    const std::size_t count = prompt.token_ids.size();
    prompt.token_types.assign(count, 0);
    prompt.positions.resize(count * 3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < count; ++index) {
            if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prepared prompt position exceeds int32 range");
            }
            prompt.positions[axis * count + index] = static_cast<std::int32_t>(index);
        }
    }
    prompt.rope_delta = 0;
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                          std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)infernix::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

std::optional<std::uint32_t> leading_instruction_boundary(std::span<const ChatRole> roles) {
    std::uint32_t leading = 0;
    while (leading < roles.size() &&
           (roles[leading] == ChatRole::System || roles[leading] == ChatRole::Developer)) {
        ++leading;
    }
    return leading != 0 ? std::optional<std::uint32_t>(leading) : std::nullopt;
}

void validate_context_cache_hints(const ContextCacheHints& hints, std::size_t message_count,
                                  std::span<const PromptCacheMarker> rendered_markers,
                                  std::span<const std::optional<std::uint32_t>> cache_boundaries) {
    if (hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    if (cache_boundaries.size() != rendered_markers.size()) {
        throw std::logic_error("rendered cache marker count changed during preparation");
    }

    for (const PromptCacheMarker marker : hints.markers) {
        switch (marker.kind) {
        case PromptCacheMarkerKind::SharedStablePrefix:
        case PromptCacheMarkerKind::PrivateLongAnchor:
            break;
        default:
            throw std::invalid_argument("context cache marker kind is invalid");
        }
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count > message_count ||
                marker.leading_instruction_bytes != 0 || marker.after_tool_count != 0 ||
                marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache message marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count == 0 || marker.after_message_count > message_count ||
                marker.after_message_part_count == 0 || marker.leading_instruction_bytes != 0 ||
                marker.after_tool_count != 0) {
                throw std::invalid_argument("context cache message-part marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (marker.after_message_count != 0 || marker.leading_instruction_bytes == 0 ||
                marker.after_tool_count != 0 || marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache leading-instruction marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_message_count != 0 || marker.leading_instruction_bytes != 0 ||
                marker.after_tool_count == 0 || marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache tool marker is invalid");
            }
            break;
        default:
            throw std::invalid_argument("context cache marker location is invalid");
        }
        if (marker.kind == PromptCacheMarkerKind::SharedStablePrefix &&
            marker.evidence == SharedCandidateEvidence::None) {
            throw std::invalid_argument("shared context cache marker evidence is empty");
        }
    }
}

// Boundary facts for hybrid prefix-cache taps. The Program floors each frontier to a page
// boundary and applies its own priority, spacing and budget rules; this only reports where the
// rendered prompt has reusable structure.
PreparedTapHints prepare_tap_hints(const ContextCacheHints& hints,
                                   std::span<const std::optional<std::uint32_t>> message_boundaries,
                                   std::span<const std::optional<std::uint32_t>> cache_boundaries,
                                   std::optional<std::size_t> engine_tool_marker_index,
                                   std::optional<std::uint32_t> leading_boundary,
                                   const std::optional<RewriteCheckpointSpec>& rewrite_checkpoint) {
    using runtime::prefix_cache::TapHint;
    using runtime::prefix_cache::TapHintKind;
    // Taps are cheap and content-deduplicated, so explicit markers only add taps. A protocol
    // write-policy mode (OpenAI explicit mode, Anthropic cache_control) withholds only the Engine's
    // tool-boundary marker; the other automatic placements are never suppressed.
    PreparedTapHints out;
    out.hints.reserve(message_boundaries.size() + hints.markers.size() + 3U);
    const auto add = [&](std::optional<std::uint32_t> frontier, TapHintKind kind) {
        if (frontier && *frontier != 0) { out.hints.push_back(TapHint{*frontier, kind}); }
    };
    for (std::size_t index = 0; index < hints.markers.size(); ++index) {
        const PromptCacheMarker& marker = hints.markers[index];
        // Only a client-named breakpoint is honored unconditionally; protocol-automatic markers
        // (OpenAI default caching, Anthropic automatic cache_control) compete with the Engine's
        // structural boundaries and mark the conversation's latest turn.
        const TapHintKind kind = has_shared_candidate_evidence(
                                     marker.evidence, SharedCandidateEvidence::ExplicitBoundary)
                                     ? TapHintKind::Explicit
                                     : TapHintKind::Automatic;
        if (marker.location == PromptCacheMarkerLocation::MessageBoundary) {
            if (marker.after_message_count < message_boundaries.size()) {
                add(message_boundaries[marker.after_message_count], kind);
            }
        } else if (index < cache_boundaries.size()) {
            add(cache_boundaries[index], kind);
        }
    }
    if (engine_tool_marker_index && *engine_tool_marker_index < cache_boundaries.size()) {
        add(cache_boundaries[*engine_tool_marker_index], TapHintKind::Structural);
    }
    if (leading_boundary && *leading_boundary < message_boundaries.size()) {
        add(message_boundaries[*leading_boundary], TapHintKind::Structural);
    }
    if (rewrite_checkpoint) { add(rewrite_checkpoint->frontier, TapHintKind::GenerationOpener); }
    for (const std::optional<std::uint32_t> boundary : message_boundaries) {
        add(boundary, TapHintKind::MessageBoundary);
    }
    return out;
}

// The reasoning-end exclusion's tokens (OutputSession::reasoning_ban): the close marker as one
// token, and the turn's end tokens, which take no probability while a reply's thinking is open.
fi::OutputMarkerTokens output_marker_tokens(const fi::Tokenizer& tokenizer,
                                            const std::vector<TokenId>& default_stops) {
    const std::vector<int> close = tokenizer.encode(kThinkClose);
    if (close.size() != 1) {
        throw std::invalid_argument(
            "GLM tokenizer must encode the reasoning close marker as one token");
    }
    fi::OutputMarkerTokens markers;
    markers.reasoning_close = close.front();
    // A GLM turn ends at the next user turn or, after its calls, at the tool results; the third
    // default stop, <|endoftext|>, ends a document, not a chat turn.
    constexpr std::array<std::string_view, 2> kTurnEnds{"<|user|>", "<|observation|>"};
    for (std::size_t index = 0; index < kTurnEnds.size(); ++index) {
        const std::vector<int> end = tokenizer.encode(kTurnEnds[index]);
        if (end.size() != 1 ||
            std::find(default_stops.begin(), default_stops.end(), end.front()) ==
                default_stops.end()) {
            throw std::invalid_argument("GLM tokenizer must stop at " +
                                        std::string(kTurnEnds[index]) + " as one token");
        }
        markers.turn_ends[index] = end.front();
    }
    return markers;
}

} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, FrontendOptions options)
        : chat_template(compile_chat_template(resources, options.chat_template_path)),
          tokenizer(resources.tokenizer), max_context(options.max_context) {
        if (options.max_context == 0) {
            throw std::invalid_argument("frontend max_context must be nonzero");
        }
        if (!tokenizer || resources.public_token_count != tokenizer->vocab_size()) {
            throw std::invalid_argument(
                "Frontend requires the parsed model tokenizer and public token domain");
        }
        sampling            = options.sampling;
        grammar_cache_bytes = options.grammar_cache_bytes;
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        markers = output_marker_tokens(*tokenizer, defaults.token_ids);
        std::string control_suffix(kThinkingControl);
        const char* control_name = "canonical thinking control suffix";
        if (!options.thinking_budget_message.empty()) {
            control_name = "thinking budget message";
            control_suffix = options.thinking_budget_message;
            if (fi::contains_unquoted_tool_marker(control_suffix)) {
                throw std::invalid_argument(
                    "thinking budget message holds a tool-call marker outside Markdown code, which "
                    "the tool constraint grammar refuses inside thinking");
            }
            if (!control_suffix.ends_with(fi::kCanonicalReasoningCloseSerialization)) {
                control_suffix += fi::kCanonicalReasoningCloseSerialization;
            }
        }
        std::vector<TokenId> encoded = tokenizer->encode(control_suffix);
        if (encoded.empty()) {
            throw std::invalid_argument(std::string("GLM tokenizer cannot encode the ") +
                                        control_name);
        }
        const std::string exact =
            tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = false});
        const std::string presented =
            tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = true});
        if (exact != control_suffix || presented.find(kThinkClose) == std::string::npos) {
            throw std::invalid_argument(std::string("GLM tokenizer cannot present the ") +
                                        control_name);
        }
        for (const TokenId token : encoded) {
            if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
                defaults.token_ids.end()) {
                throw std::invalid_argument(std::string(control_name) +
                                            " contains a default terminal token");
            }
        }
        thinking_control_tokens = std::make_shared<const std::vector<TokenId>>(std::move(encoded));
    }

    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy defaults;
    fi::OutputMarkerTokens markers; // the reasoning-end exclusion's tokens (reasoning_ban)
    ModelSamplingDefaults sampling;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    std::uint32_t max_context       = 0;
    std::size_t grammar_cache_bytes = 0;
    mutable std::once_flag grammar_once;
    mutable std::unique_ptr<text::GrammarCompiler> grammar_compiler;

    text::GrammarCompiler& grammars() const {
        std::call_once(grammar_once, [&] {
            std::vector<std::string> vocab(tokenizer->vocab_size());
            for (std::size_t id = 0; id < vocab.size(); ++id) {
                if (!tokenizer->is_valid_token(static_cast<TokenId>(id))) { continue; }
                const auto decoded = tokenizer->decoded_token(static_cast<TokenId>(id));
                if (!decoded.special) { vocab[id] = decoded.bytes; }
            }
            grammar_compiler = std::make_unique<text::GrammarCompiler>(
                std::move(vocab), defaults.token_ids, grammar_cache_bytes);
        });
        return *grammar_compiler;
    }
};

std::span<const std::int32_t> PreparedPromptData::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != token_ids.size() * 3) {
        throw std::out_of_range("invalid prepared-prompt position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * token_ids.size(), token_ids.size());
}

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    return PromptSummary{.starts_in_reasoning = data_->starts_in_reasoning,
                         .prompt_tokens       = checked_token_count(data_->token_ids.size()),
                         .has_media           = false};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    const PrepareStats& stats = data_->prepare;
    return PromptPreparationStats{
        .seconds                       = stats.seconds,
        .media_preprocess_seconds      = stats.media_preprocess_seconds,
        .media_preprocess_work_seconds = stats.media_preprocess_work_seconds,
        .tokenize_seconds              = stats.tokenize_seconds,
        .media_items                   = stats.media_items,
        .media_bytes                   = stats.media_bytes,
        .raw_patches                   = stats.raw_patches,
        .vision_tokens                 = stats.vision_tokens,
        .patch_bytes                   = stats.patch_bytes,
        .media_cache_hits              = stats.media_cache_hits,
        .media_cache_misses            = stats.media_cache_misses,
        .media_singleflight_waits      = stats.media_singleflight_waits,
        .built_patch_bytes             = stats.built_patch_bytes,
        .reused_patch_bytes            = stats.reused_patch_bytes,
    };
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept {
    return impl_->sampling;
}

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

const PreparedPromptData& FrontendTestAccess::inspect(const PreparedPrompt& prompt) {
    return PreparedPromptAccess::view(prompt);
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    fi::check_preparation_control(control);
    const auto start              = Clock::now();
    const PromptOptions options   = input.options;
    ContextCacheHints cache_hints = std::move(input.context_cache);
    if (cache_hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    std::vector<ChatRole> message_roles;
    message_roles.reserve(input.messages.size());
    for (const ChatMessage& message : input.messages) { message_roles.push_back(message.role); }
    const auto tool_call_output = common::frontend::build_tool_call_output_contract(
        options.tool_jsons, options.deferred_tool_jsons, options.open_tool_names);
    const std::optional<std::uint32_t> leading_boundary =
        leading_instruction_boundary(message_roles);
    std::vector<PromptCacheMarker> rendered_markers = cache_hints.markers;
    std::optional<std::size_t> engine_tool_marker_index;
    if (cache_hints.allow_engine_automatic_shared_prefixes && !options.tool_jsons.empty()) {
        engine_tool_marker_index = rendered_markers.size();
        rendered_markers.push_back(PromptCacheMarker{
            .kind             = PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = SharedCandidateEvidence::EngineStructural,
            .location         = PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(options.tool_jsons.size()),
        });
    }
    const std::size_t message_count       = input.messages.size();
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.tool_call_output    = tool_call_output;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    const fi::RenderedChat rendered =
        impl_->chat_template.render(messages, render_options(options, rendered_markers), control);
    result.starts_in_reasoning  = rendered.starts_in_reasoning;
    result.continuation_content = rendered.continuation_content;
    const auto tokenize_started = Clock::now();
    fi::EncodedChat encoded     = fi::encode_rendered_chat(
        *impl_->tokenizer, rendered, static_cast<std::size_t>(impl_->max_context) + 1U);
    result.prepare.tokenize_seconds =
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    fi::check_preparation_control(control, "tokenization");
    if (encoded.input_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(
            fi::encode_rendered_chat(*impl_->tokenizer, rendered,
                                     fi::context_recount_limit(impl_->max_context))
                .input_ids.size(),
            impl_->max_context);
    }
    result.token_ids                   = std::move(encoded.input_ids);
    result.identity.rewrite_checkpoint = encoded.rewrite_checkpoint;
    result.identity.rewrite_execution_frontiers = std::move(encoded.rewrite_execution_frontiers);
    message_boundaries                          = std::move(encoded.message_boundaries);
    cache_boundaries                            = std::move(encoded.cache_boundaries);
    assign_text_positions(result);
    (void)checked_token_count(result.token_ids.size());
    result.identity.reusable = true;
    result.tap_hints         = prepare_tap_hints(cache_hints, message_boundaries, cache_boundaries,
                                                 engine_tool_marker_index, leading_boundary,
                                                 result.identity.rewrite_checkpoint);
    validate_context_cache_hints(cache_hints, message_count, rendered_markers, cache_boundaries);
    detail::prompt_block_keys(result, result.block_hashes, result.block_extras);
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    fi::check_preparation_control(control);
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const fi::RenderedChat rendered =
        impl_->chat_template.render(messages, render_options(options), control);
    const std::uint32_t count = checked_token_count(
        fi::encode_rendered_chat(*impl_->tokenizer, rendered).input_ids.size());
    fi::check_preparation_control(control, "tokenization");
    return count;
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (token_ids.size() > impl_->max_context) {
        throw RequestError(ContextLengthDetails{.prompt_tokens = token_ids.size(), .max_context = impl_->max_context});
    }
    (void)checked_token_count(token_ids.size());
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids           = std::move(token_ids);
    assign_text_positions(result);
    detail::prompt_block_keys(result, result.block_hashes, result.block_extras);
    result.identity.reusable = allow_prefix_identity;
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking,
                                            const std::optional<OutputConstraint>& constraint,
                                            const ToolChoice& tool_choice) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    auto tool_contract =
        common::frontend::select_tool_call_contract(prompt.data_->tool_call_output, tool_choice);
    if (constraint && tool_contract && tool_contract->tools.empty()) tool_contract.reset();
    const bool combined = constraint && tool_contract && !tool_contract->tools.empty();
    if (combined) {
        if (constraint->kind != OutputConstraintKind::JsonObject &&
            constraint->kind != OutputConstraintKind::JsonSchema)
            throw RequestError(text::constraint_error_kind(constraint->kind),
                               "active tools can be combined with JSON output constraints");
        if (!tool_contract->constrained) {
            auto structured =
                std::make_shared<common::frontend::ToolCallOutputContract>(*tool_contract);
            structured->constrained = true;
            tool_contract           = std::move(structured);
        }
    }
    const bool tool_constraint = tool_contract && tool_contract->constrained;
    std::unique_ptr<text::GrammarSession> matcher;
    if (constraint || tool_constraint) {
        if (impl_->defaults.token_ids.empty() || !caller_stop.token_ids.empty() ||
            !caller_stop.strings.empty() || !caller_stop.include_model_defaults ||
            caller_stop.publish_stop_token || output.raw || output.preserve_special_tokens) {
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                                               : text::constraint_error_kind(constraint->kind),
                               "constraints require default EOS, text output, no custom stops, and "
                               "one output language");
        }
        try {
            const auto close = prompt.data_->starts_in_reasoning
                                   ? fi::kCanonicalReasoningCloseSerialization
                                   : std::string_view{};
            if (tool_constraint) {
                for (const auto& tool : tool_contract->tools) {
                    if (tool.name.size() > output.tool_name_max_length ||
                        !std::all_of(tool.name.begin(), tool.name.end(), [](unsigned char c) {
                            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9') || c == '_' || c == '-';
                        }))
                        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                                           "tool name is outside the output contract: " +
                                               tool.name);
                }
                matcher = fi::compile_tool_grammar(impl_->grammars(), *tool_contract, close,
                                                   prompt.data_->continuation_content, constraint);
            } else {
                matcher = impl_->grammars().compile(*constraint, close,
                                                    prompt.data_->continuation_content);
            }
        } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                                               : text::constraint_error_kind(constraint->kind),
                               error.what());
        }
    }
    return OutputSession(impl_->tokenizer, std::move(policy), output,
                         prompt.data_->starts_in_reasoning, thinking,
                         impl_->thinking_control_tokens, std::move(tool_contract),
                         impl_->markers, std::move(matcher), prompt.data_->continuation_content,
                         combined);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

} // namespace infernix::models::glm5_next
