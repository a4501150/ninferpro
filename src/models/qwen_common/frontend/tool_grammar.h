#pragma once

#include "models/qwen_common/frontend/tool_contract.h"
#include "text/grammar.h"

namespace infernix::models::qwen_common::frontend {
[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const ToolCallOutputContract& contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body = {});
} // namespace infernix::models::qwen_common::frontend
