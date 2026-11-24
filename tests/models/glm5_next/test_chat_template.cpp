// Renders representative conversations through the glm5_next chat template and compares the
// output against a Python jinja2 oracle (chat_template_oracle.json, generated from the same
// template source with the transformers JSON conventions). Covers the reasoning-effort
// preamble, thinking preservation and clearing, tool declarations, tool calls with object
// arguments, tool responses in their sorted and tolerant forms, and assistant continuation.
// Tool outputs are text here: the engine's ChatMessage carries no structured tool output.
#include "models/glm5_next/frontend/chat_template.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fi = infernix::models::glm5_next::frontend;
using infernix::ChatRole;
using infernix::PromptContinuationMode;
using infernix::ReasoningEffort;
using Json = nlohmann::ordered_json;

int failures = 0;

void check(bool ok, const std::string& message) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::invalid_argument("cannot read " + path); }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

fi::ChatMessage message_from_json(const Json& value) {
    fi::ChatMessage message;
    const std::string role = value.at("role").get<std::string>();
    if (role == "system") {
        message.role = ChatRole::System;
    } else if (role == "user") {
        message.role = ChatRole::User;
    } else if (role == "assistant") {
        message.role = ChatRole::Assistant;
    } else {
        message.role = ChatRole::Tool;
    }
    message.parts.push_back(fi::ChatPart::text_part(value.at("content").get<std::string>()));
    if (value.contains("reasoning_content")) {
        message.reasoning_content = value.at("reasoning_content").get<std::string>();
    }
    if (value.contains("tool_call_id")) {
        message.tool_call_id = value.at("tool_call_id").get<std::string>();
    }
    if (value.contains("tool_calls")) {
        for (const auto& call : value.at("tool_calls")) {
            message.tool_calls.push_back(fi::ToolCall{
                .id             = call.at("id").get<std::string>(),
                .name           = call.at("function").at("name").get<std::string>(),
                .arguments_json = call.at("function").at("arguments").dump(),
            });
        }
    }
    return message;
}

fi::ChatRenderOptions options_from_json(const Json& value) {
    fi::ChatRenderOptions options;
    options.add_generation_prompt = value.value("add_generation_prompt", false);
    if (value.value("continue_final_message", false)) {
        options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    }
    if (value.contains("reasoning_effort")) {
        const std::string name = value.at("reasoning_effort").get<std::string>();
        if (name == "low") {
            options.reasoning_effort = ReasoningEffort::Low;
        } else if (name == "high") {
            options.reasoning_effort = ReasoningEffort::High;
        } else if (name == "medium") {
            options.reasoning_effort = ReasoningEffort::Medium;
        } else if (name == "max") {
            options.reasoning_effort = ReasoningEffort::Max;
        }
    }
    // The template reads clear_thinking from its context; the product passes it as a chat
    // template keyword argument.
    if (value.contains("clear_thinking")) {
        options.chat_template_kwargs_json = Json{{"clear_thinking", value.at("clear_thinking")}}.dump();
    }
    if (value.contains("tools")) {
        for (const auto& tool : value.at("tools")) { options.tool_jsons.push_back(tool.dump()); }
    }
    return options;
}

} // namespace

int main() {
    const std::string source =
        read_file(INFERNIX_SOURCE_DIR "/tests/models/glm5_next/chat_template.jinja");
    const Json fixture =
        Json::parse(read_file(INFERNIX_SOURCE_DIR "/tests/models/glm5_next/chat_template_oracle.json"));
    const Json special_tokens{{"eos_token", "\n"}, {"pad_token", "\n"}};
    const auto compiled = fi::CompiledChatTemplate::resolve(source, "chat_template.jinja", special_tokens);

    for (const auto& item : fixture.at("cases")) {
        std::vector<fi::ChatMessage> messages;
        for (const auto& entry : item.at("messages")) {
            messages.push_back(message_from_json(entry));
        }
        const auto rendered = compiled.render(messages, options_from_json(item.at("options")));
        const std::string expected = item.at("text").get<std::string>();
        check(rendered.text == expected,
              item.value("name", "") + ": rendered text differs; expected " +
                  Json(expected).dump() + " got " + Json(rendered.text).dump());
        check(rendered.starts_in_reasoning == expected.ends_with("<think>"),
              item.value("name", "") + ": reasoning start state differs");
        if (item.contains("continuation_content")) {
            check(rendered.continuation_content ==
                      item.at("continuation_content").get<std::string>(),
                  item.value("name", "") + ": continuation content differs");
        }
    }

    if (failures == 0) {
        std::cout << "chat template oracle passed\n";
        return 0;
    }
    std::cout << failures << " failures\n";
    return 1;
}
