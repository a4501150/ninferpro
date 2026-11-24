// GLM tool calls against the chat template's own spelling: calls the template renders parse back to
// their arguments, the constraint grammar accepts that spelling and nothing looser, and the
// streaming decoder agrees with the whole-output parser at every cut.
#include "models/glm5_next/frontend/chat_template.h"
#include "models/glm5_next/frontend/tool_call_parser.h"
#include "models/common/frontend/tool_contract.h"
#include "models/glm5_next/frontend/tool_grammar.h"
#include "text/grammar.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fi = infernix::models::glm5_next::frontend;
namespace tool_contract = infernix::models::common::frontend;
using Json   = nlohmann::ordered_json;

int failures = 0;

void check(bool ok, const std::string& message) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::invalid_argument("cannot read " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

infernix::text::GrammarCompiler byte_compiler() {
    std::vector<std::string> tokens;
    for (int i = 0; i < 256; ++i) tokens.emplace_back(1, static_cast<char>(i));
    tokens.emplace_back();
    return {std::move(tokens), {256}, 32 * 1024 * 1024};
}

bool accepts(infernix::text::GrammarCompiler& compiler,
             const tool_contract::ToolCallOutputContract& contract, std::string_view text,
             std::string_view reasoning_close = {}) {
    auto matcher = fi::compile_tool_grammar(compiler, contract, reasoning_close, {});
    std::vector<std::uint32_t> words(matcher->mask_words());
    for (unsigned char c : text) {
        if (matcher->masks({}, words) || !(words[c / 32] & (1u << (c % 32)))) return false;
        matcher->accept(c);
        matcher->confirm();
    }
    return matcher->masks({}, words) == 0 && (words[256 / 32] & 1u);
}

const Json kWeather{
    {"type", "function"},
    {"function",
     {{"name", "get_weather"},
      {"parameters",
       {{"type", "object"},
        {"properties",
         {{"city", {{"type", "string"}}},
          {"days", {{"type", "integer"}}},
          {"units",
           {{"type", "object"},
            {"properties", {{"metric", {{"type", "boolean"}}}}},
            {"additionalProperties", false}}},
          {"hours", {{"type", "array"}, {"items", {{"type", "number"}}}}},
          {"note", {{"type", "string"}}}}},
        {"required", {"city"}},
        {"additionalProperties", false}}}}}};

const Json kSearch{
    {"type", "function"},
    {"function",
     {{"name", "search"},
      {"parameters",
       {{"type", "object"},
        {"properties", {{"query", {{"type", "string"}}}}},
        {"required", {"query"}},
        {"additionalProperties", false}}}}}};

Json strict(Json tool) {
    tool["function"]["strict"] = true;
    return tool;
}

std::shared_ptr<const tool_contract::ToolCallOutputContract>
contract_for(const std::vector<Json>& tools, infernix::ToolChoice choice = {}) {
    std::vector<std::string> jsons;
    for (const auto& tool : tools) jsons.push_back(tool.dump());
    return tool_contract::select_tool_call_contract(
        tool_contract::build_tool_call_output_contract(jsons), choice);
}

// The template's spelling of an assistant turn with these calls: the final turn's bytes from its
// first call (the tool section before it shows an example call).
std::string rendered_calls(const fi::CompiledChatTemplate& compiled, const std::vector<Json>& tools,
                           const std::vector<std::pair<std::string, Json>>& calls) {
    fi::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(fi::ChatPart::text_part("q"));
    fi::ChatMessage assistant;
    assistant.role = infernix::ChatRole::Assistant;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        assistant.tool_calls.push_back(fi::ToolCall{.id             = "call_" + std::to_string(i),
                                                    .name           = calls[i].first,
                                                    .arguments_json = calls[i].second.dump()});
    }
    fi::ChatRenderOptions options;
    options.add_generation_prompt = false;
    for (const auto& tool : tools) options.tool_jsons.push_back(tool.dump());
    const auto text = compiled.render({user, assistant}, options).text;
    const auto begin = text.find("<tool_call>", text.rfind("<|assistant|>"));
    if (begin == std::string::npos) throw std::runtime_error("template rendered no call");
    return text.substr(begin);
}

void round_trips(const fi::CompiledChatTemplate& compiled,
                 infernix::text::GrammarCompiler& compiler) {
    const Json weather{{"city", "São Paulo <north>"},
                       {"days", 3},
                       {"units", {{"metric", true}}},
                       {"hours", {6.5, 12, -1e-3}},
                       {"note", "line one\n  line two "}};
    const Json query{{"query", "{\"not\": json}"}};
    const std::vector<std::pair<std::string, Json>> calls{{"get_weather", weather},
                                                          {"search", query}};
    for (const bool is_strict : {false, true}) {
        const std::vector<Json> tools = is_strict ? std::vector<Json>{strict(kWeather), strict(kSearch)}
                                                  : std::vector<Json>{kWeather, kSearch};
        const auto spelled  = rendered_calls(compiled, tools, calls);
        const auto contract = contract_for(tools);
        const std::string label = is_strict ? "strict" : "non-strict";

        const auto parsed = fi::parse_glm_tool_call_output(spelled, 64, *contract);
        check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
              label + ": the template's calls did not parse: " + Json(spelled).dump());
        for (std::size_t i = 0; i < parsed.tool_calls.size() && i < calls.size(); ++i) {
            check(parsed.tool_calls[i].name == calls[i].first, label + ": call name changed");
            check(Json::parse(parsed.tool_calls[i].arguments_json) == calls[i].second,
                  label + ": arguments changed: " + parsed.tool_calls[i].arguments_json);
        }

        // Only the strict grammar types the values; both accept the template's spelling.
        check(accepts(compiler, *contract, spelled),
              label + ": grammar refused the template's spelling: " + Json(spelled).dump());
        check(accepts(compiler, *contract, "Let me check.\n" + spelled),
              label + ": grammar refused text before the calls");

        fi::ToolCallOutputDecoder decoder(contract, 64);
        std::string content;
        for (const char c : "Let me check.\n" + spelled) content += decoder.feed(std::string(1, c));
        const auto terminal = decoder.finish();
        content += terminal.content;
        check(content == "Let me check.", label + ": streamed content was " + Json(content).dump());
        check(terminal.tool_calls.size() == 2, label + ": streamed decoder lost a call");
        for (std::size_t i = 0; i < terminal.tool_calls.size() && i < calls.size(); ++i) {
            check(Json::parse(terminal.tool_calls[i].arguments_json) == calls[i].second,
                  label + ": streamed arguments changed: " + terminal.tool_calls[i].arguments_json);
        }

        // An output cut anywhere keeps exactly the calls completed before the cut.
        const auto first_end = spelled.find("</tool_call>") + std::string_view("</tool_call>").size();
        for (std::size_t cut = 0; cut <= spelled.size(); ++cut) {
            fi::ToolCallOutputDecoder partial(contract, 64);
            (void)partial.feed(std::string_view(spelled).substr(0, cut));
            const auto result = partial.finish(infernix::FinishReason::OutputLimit);
            const std::size_t expected = cut == spelled.size() ? 2 : cut >= first_end ? 1 : 0;
            check(result.tool_calls.size() == expected,
                  label + ": cut at " + std::to_string(cut) + " kept " +
                      std::to_string(result.tool_calls.size()) + " calls");
        }
    }
}

void grammar_bounds(infernix::text::GrammarCompiler& compiler) {
    const auto contract = contract_for({strict(kWeather)});
    const std::string call =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
    check(accepts(compiler, *contract, call), "grammar refused a minimal call");
    check(!accepts(compiler, *contract,
                   "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Oslo</arg_value>"
                   "</tool_call>"),
          "grammar admitted whitespace the template never writes");
    check(!accepts(compiler, *contract,
                   "<tool_call>get_weather<arg_key>days</arg_key><arg_value>3</arg_value>"
                   "</tool_call>"),
          "grammar admitted a call without its required argument");
    check(!accepts(compiler, *contract,
                   "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value>"
                   "<arg_key>days</arg_key><arg_value>three</arg_value></tool_call>"),
          "grammar admitted a string for an integer argument");
    check(!accepts(compiler, *contract,
                   "<tool_call>missing<arg_key>city</arg_key><arg_value>Oslo</arg_value>"
                   "</tool_call>"),
          "grammar admitted an undeclared tool");
    check(accepts(compiler, *contract, "Oslo is cold.</think>" + call, "</think>"),
          "grammar refused a call after the reasoning close");
    check(!accepts(compiler, *contract, "maybe " + call + "</think>", "</think>"),
          "grammar admitted a call inside thinking");
    check(accepts(compiler, *contract, call + call), "parallel calls refused");

    infernix::ToolChoice required;
    required.mode = infernix::ToolChoiceMode::Required;
    const auto forced = contract_for({strict(kWeather)}, required);
    check(!accepts(compiler, *forced, "No call."), "a required call admitted text");
    check(accepts(compiler, *forced, call), "a required call refused a call");
}

void free_output() {
    const auto contract = contract_for({kWeather});
    const std::string call =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";

    const auto quoted = fi::parse_glm_tool_call_output("Write `" + call + "` to call it.", 64,
                                                       *contract);
    check(!quoted.is_tool_call_response, "a call inside inline code became a call");

    const auto spaced = fi::parse_glm_tool_call_output(
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n"
        "</tool_call>",
        64, *contract);
    check(spaced.is_tool_call_response && spaced.tool_calls.size() == 1 &&
              Json::parse(spaced.tool_calls[0].arguments_json) == Json{{"city", "Oslo"}},
          "free output with whitespace between pieces did not parse");

    const std::string cut =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Os";
    check(!fi::parse_glm_tool_call_output(cut, 64, *contract).is_tool_call_response,
          "the strict parser accepted a truncated call");
    const auto tolerant = fi::parse_glm_tool_call_output(cut, 64, *contract, true);
    check(tolerant.is_tool_call_response && tolerant.tool_calls.size() == 1 &&
              tolerant.diagnostics.fallback_reason ==
                  infernix::ToolCallParseFallbackReason::TruncatedTail,
          "tolerant mode did not keep a truncated call");

    const auto undeclared = fi::parse_glm_tool_call_output(
        "<tool_call>missing<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>", 64,
        *contract);
    check(!undeclared.is_tool_call_response &&
              undeclared.diagnostics.fallback_reason ==
                  infernix::ToolCallParseFallbackReason::UndeclaredTool,
          "an undeclared tool was not reported");

    check(fi::contains_unquoted_tool_marker("think <tool_call>"), "a bare opener went unseen");
    check(!fi::contains_unquoted_tool_marker("```\n<tool_call>\n```"),
          "an opener inside a code fence was reported");
}

} // namespace

int main() {
    try {
        const auto compiled = fi::CompiledChatTemplate::resolve(
            read_file(INFERNIX_SOURCE_DIR "/tests/models/glm5_next/chat_template.jinja"),
            "chat_template.jinja", Json{{"eos_token", "\n"}, {"pad_token", "\n"}});
        auto compiler = byte_compiler();
        round_trips(compiled, compiler);
        grammar_bounds(compiler);
        free_output();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    if (failures == 0) {
        std::cout << "glm tool calls passed\n";
        return 0;
    }
    std::cout << failures << " failures\n";
    return 1;
}
