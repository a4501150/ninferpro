#pragma once

// The DFlash2 drafter-training corpus records shared by drafter_generate.cpp and
// drafter_capture.cpp (tools/drafter/README.md): one JSON object per line. A prompt record holds
// {"id", "system"?, "user", "thinking", "tools"}; generation adds "output_ids" (the target's own
// tokens). An off-policy record holds {"id", "messages": [{"role", "content"}...]} ending in an
// assistant turn, rendered whole.

#include "infernix/types.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

namespace drafter_corpus {

// The tool a "tools" prompt declares (the agentic loop's run_bash; nothing is ever run).
inline const char* kRunBashTool =
    R"({"type":"function","function":{"name":"run_bash","description":"Run a bash command in a fresh sandbox; stdout/stderr go to output. Python via `python3` is available.","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"timeout":{"type":"integer","description":"seconds to wait; default 30"}},"required":["cmd"]}}})";

inline infernix::ChatRole role(const std::string& name) {
    if (name == "system") { return infernix::ChatRole::System; }
    if (name == "user") { return infernix::ChatRole::User; }
    if (name == "assistant") { return infernix::ChatRole::Assistant; }
    throw std::invalid_argument("unknown role " + name);
}

inline infernix::ChatMessage message(infernix::ChatRole r, const std::string& text) {
    infernix::ChatMessage m;
    m.role = r;
    m.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = text});
    return m;
}

// The prompt a record renders: a prompt record's new assistant turn, or an off-policy record's
// whole conversation continued from its final assistant turn.
inline infernix::PromptInput prompt_input(const nlohmann::json& record) {
    infernix::PromptInput input;
    if (record.contains("messages")) {
        for (const auto& m : record["messages"]) {
            input.messages.push_back(message(role(m["role"].get<std::string>()), m["content"].get<std::string>()));
        }
        input.options.continuation    = infernix::PromptContinuationMode::ContinueFinalAssistant;
        input.options.enable_thinking = false;
        return input;
    }
    if (record.contains("system")) {
        input.messages.push_back(message(infernix::ChatRole::System, record["system"].get<std::string>()));
    }
    input.messages.push_back(message(infernix::ChatRole::User, record["user"].get<std::string>()));
    input.options.enable_thinking = record.value("thinking", true);
    if (record.value("tools", false)) { input.options.tool_jsons.push_back(kRunBashTool); }
    return input;
}

} // namespace drafter_corpus
