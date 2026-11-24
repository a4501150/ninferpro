// The glm5_next frontend on the real GLM-5.3-Flash tokenizer and chat template. Prepared prompts
// must equal the Hugging Face tokenizer's ids for the template oracle's text
// (prompt_tokens_oracle.json), and a generated reply must split into reasoning, content and tool
// calls the way the template writes them. GLM_TOKENIZER_RESOURCES names the directory holding
// tokenizer.json, tokenizer_config.json, generation_config.json and chat_template.jinja; without it
// the test skips (77).
#include "models/glm5_next/frontend/frontend.h"
#include "models/glm5_next/frontend/resources.h"
#include "models/glm5_next/frontend/test_access.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace glm = infernix::models::glm5_next;
using Json    = nlohmann::ordered_json;

int failures = 0;

void check(bool ok, const std::string& message) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

infernix::ChatMessage message_from_json(const Json& value) {
    infernix::ChatMessage message;
    const std::string role = value.at("role");
    message.role = role == "system"      ? infernix::ChatRole::System
                   : role == "user"      ? infernix::ChatRole::User
                   : role == "assistant" ? infernix::ChatRole::Assistant
                                         : infernix::ChatRole::Tool;
    message.parts.push_back({.text = value.at("content").get<std::string>()});
    if (value.contains("reasoning_content"))
        message.reasoning_content = value.at("reasoning_content");
    if (value.contains("tool_call_id")) message.tool_call_id = value.at("tool_call_id");
    for (const auto& call : value.value("tool_calls", Json::array())) {
        message.tool_calls.push_back({.id             = call.at("id"),
                                      .name           = call.at("function").at("name"),
                                      .arguments_json = call.at("function").at("arguments").dump()});
    }
    return message;
}

infernix::PromptInput input_from_json(const Json& item) {
    infernix::PromptInput input;
    for (const auto& entry : item.at("messages")) input.messages.push_back(message_from_json(entry));
    const auto& options = item.at("options");
    if (options.contains("reasoning_effort")) {
        const std::string effort = options.at("reasoning_effort");
        input.options.reasoning_effort = effort == "low"    ? infernix::ReasoningEffort::Low
                                         : effort == "high" ? infernix::ReasoningEffort::High
                                         : effort == "max"  ? infernix::ReasoningEffort::Max
                                                            : infernix::ReasoningEffort::Medium;
    }
    for (const auto& tool : options.value("tools", Json::array()))
        input.options.tool_jsons.push_back(tool.dump());
    return input;
}

struct Reply {
    std::string reasoning, content;
    std::vector<infernix::GeneratedToolCall> calls;
    infernix::FinishReason finish = infernix::FinishReason::None;
};

// Feeds a reply one token per round, as decode does, then ends it at the stop token if the reply
// holds one.
Reply run(glm::OutputSession& session, const std::vector<infernix::TokenId>& tokens) {
    Reply reply;
    const auto collect = [&](const glm::PublishedOutput& published) {
        for (const auto& delta : published) {
            (delta.channel == infernix::OutputChannel::Reasoning ? reply.reasoning
                                                                 : reply.content) += delta.text;
        }
    };
    for (std::size_t i = 0; i < tokens.size() && reply.finish == infernix::FinishReason::None; ++i) {
        const auto remaining = static_cast<std::uint32_t>(tokens.size() - i + 8);
        const auto decision  = session.preview_model(std::span(&tokens[i], 1), remaining,
                                                     infernix::FinishReason::OutputLimit);
        reply.finish         = decision.finish_reason;
        collect(session.commit_preview());
    }
    if (reply.finish == infernix::FinishReason::None) {
        reply.finish = session.preview_terminal(infernix::FinishReason::OutputLimit).finish_reason;
        collect(session.commit_preview());
    }
    reply.calls = session.take_tool_calls();
    return reply;
}

} // namespace

int main() {
    const char* resources_dir = std::getenv("GLM_TOKENIZER_RESOURCES");
    if (resources_dir == nullptr || std::string(resources_dir).empty()) {
        std::cout << "skipped: GLM_TOKENIZER_RESOURCES is not set\n";
        return 77;
    }
    const std::string dir(resources_dir);
    try {
        const std::string tokenizer_json  = read_file(dir + "/tokenizer.json");
        const std::string tokenizer_config = read_file(dir + "/tokenizer_config.json");
        const std::string chat_template    = read_file(dir + "/chat_template.jinja");
        const std::string generation       = read_file(dir + "/generation_config.json");
        glm::FrontendResources resources{.tokenizer_json         = tokenizer_json,
                                         .tokenizer_config_json  = tokenizer_config,
                                         .chat_template_jinja    = chat_template,
                                         .generation_config_json = generation};
        glm::parse_resources(resources, {.vocab_size = 154880});
        glm::FrontendOptions options;
        options.max_context = 32768;
        const auto frontend = glm::make_frontend(resources, options);

        const Json templates = Json::parse(
            read_file(INFERNIX_SOURCE_DIR "/tests/models/glm5_next/chat_template_oracle.json"));
        const Json oracle = Json::parse(
            read_file(INFERNIX_SOURCE_DIR "/tests/models/glm5_next/prompt_tokens_oracle.json"));
        for (const auto& expected : oracle.at("cases")) {
            const std::string name = expected.at("name");
            const auto item = std::find_if(templates.at("cases").begin(), templates.at("cases").end(),
                                           [&](const Json& c) { return c.at("name") == name; });
            if (item == templates.at("cases").end()) throw std::runtime_error("no case " + name);
            const auto prompt = frontend.prepare(input_from_json(*item));
            const auto& data  = glm::FrontendTestAccess::inspect(prompt);
            check(data.token_ids == expected.at("ids").get<std::vector<infernix::TokenId>>(),
                  name + ": prompt tokens differ from the tokenizer oracle");
            check(data.starts_in_reasoning, name + ": the generation prompt opens thinking");
        }

        // A reply as the model writes it: thinking, its close, an answer, one call, then the
        // tool-results turn marker that ends it.
        const Json tools = Json::array({templates.at("cases")[7].at("options").at("tools")[0]});
        infernix::PromptInput input;
        input.messages.push_back({.role = infernix::ChatRole::User, .parts = {{.text = "Oslo?"}}});
        input.options.tool_jsons.push_back(tools[0].dump());
        const auto prompt = frontend.prepare(input);
        std::vector<infernix::TokenId> reply_tokens =
            frontend.tokenize_text("Two plus two is four.</think>The answer is 4.<tool_call>"
                                   "get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value>"
                                   "</tool_call>");
        check(reply_tokens ==
                  std::vector<infernix::TokenId>{11608, 5519, 1378, 374, 3040, 13, 154842, 785,
                                                 4226, 374, 220, 19, 13, 154843, 455, 68852,
                                                 154847, 8923, 154848, 154849, 28764, 385, 154850,
                                                 154844},
              "the reply's tokens differ from the tokenizer oracle");
        reply_tokens.push_back(154829); // <|observation|>
        auto session = frontend.make_output_session(prompt, infernix::StopPolicy{});
        const auto ban = session.reasoning_ban();
        check(ban.close == 154842 && ban.ends[0] == 154827 && ban.ends[1] == 154829,
              "open thinking does not ban </think>'s alternatives <|user|> and <|observation|>");
        const Reply reply = run(session, reply_tokens);
        check(reply.reasoning == "Two plus two is four.",
              "reasoning was " + Json(reply.reasoning).dump());
        check(reply.content == "The answer is 4.", "content was " + Json(reply.content).dump());
        check(reply.calls.size() == 1 && reply.calls[0].name == "get_weather" &&
                  Json::parse(reply.calls[0].arguments_json) == Json{{"city", "Oslo"}},
              "the tool call did not parse");
        check(reply.finish == infernix::FinishReason::StopToken,
              "<|observation|> did not end the reply");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    if (failures == 0) {
        std::cout << "real frontend passed\n";
        return 0;
    }
    std::cout << failures << " failures\n";
    return 1;
}
