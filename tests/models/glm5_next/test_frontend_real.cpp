// The glm5_next frontend on the real GLM-5.3-Flash tokenizer and chat template. Prepared prompts
// must equal the Hugging Face tokenizer's ids for the template oracle's text
// (prompt_tokens_oracle.json), and a generated reply must split into reasoning, content and tool
// calls the way the template writes them. GLM_TOKENIZER_RESOURCES names the directory holding
// tokenizer.json, tokenizer_config.json, generation_config.json, chat_template.jinja and
// processor_config.json; without it the test skips (77). With GLM_PROCESSOR_REFERENCE (the --out
// directory of tools/glm5_next/processor_reference.py) each media prompt's token ids, token types,
// grids and BF16 patches must equal the Hugging Face processor's exactly.
#include "models/glm5_next/frontend/frontend.h"
#include "models/glm5_next/frontend/resources.h"
#include "models/glm5_next/frontend/test_access.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

double bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    const std::string bytes = [&] {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot read " + path);
        std::ostringstream buffer;
        buffer << stream.rdbuf();
        return buffer.str();
    }();
    return {bytes.begin(), bytes.end()};
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

// Each processor_reference.py case against the frontend's prepared prompt.
void check_processor(const glm::Frontend& frontend, const std::string& dir) {
    const Json oracle = Json::parse(read_file(dir + "/processor.json"));
    for (std::size_t k = 0; k < oracle.at("cases").size(); ++k) {
        const Json& expected   = oracle.at("cases")[k];
        const std::string name = "media case " + std::to_string(k);
        infernix::ChatMessage message{.role = infernix::ChatRole::User};
        for (std::size_t m = 0; m < expected.at("media").size(); ++m) {
            const std::string path = expected.at("media")[m];
            const bool video       = expected.at("kinds")[m] == "video";
            message.parts.push_back(
                {.kind  = infernix::MessagePartKind::Media,
                 .media = {.kind  = video ? infernix::MediaKind::Video : infernix::MediaKind::Image,
                           .bytes = read_bytes(path),
                           .source_name = path}});
        }
        message.parts.push_back({.text = expected.at("text").get<std::string>()});
        infernix::PromptInput input;
        input.messages.push_back(std::move(message));
        const auto prompt = frontend.prepare(std::move(input));
        const auto& data  = glm::FrontendTestAccess::inspect(prompt);
        check(data.token_ids == expected.at("ids").get<std::vector<infernix::TokenId>>(),
              name + ": prompt tokens differ from the processor");
        check(data.token_types == expected.at("token_types").get<std::vector<std::uint8_t>>(),
              name + ": token types differ from the processor");
        // The oracle's pixels list the images, then the videos.
        std::vector<std::size_t> order;
        for (const bool video : {false, true}) {
            for (std::size_t i = 0; i < data.vision_items.size(); ++i) {
                if ((data.vision_items[i].modality == glm::PromptModality::Video) == video) {
                    order.push_back(i);
                }
            }
        }
        const auto pixels  = read_bytes(dir + "/case-" + std::to_string(k) + ".bf16");
        std::size_t offset = 0, differing = 0, compared = 0;
        double largest = 0; // in normalized units: one 8-bit level is about 0.015
        check(order.size() == expected.at("grids").size(), name + ": media item count differs");
        for (std::size_t j = 0; j < order.size() && j < expected.at("grids").size(); ++j) {
            const glm::VisionItem& item = data.vision_items[order[j]];
            const auto& grid            = expected.at("grids")[j];
            check(item.grid.temporal == grid[0].get<int>() &&
                      item.grid.height == grid[1].get<int>() &&
                      item.grid.width == grid[2].get<int>(),
                  name + ": item " + std::to_string(j) + " grid differs");
            const auto patches = data.media_payloads[order[j]]->span();
            for (std::size_t i = 0; i < patches.size() && offset + 2 * i + 1 < pixels.size(); ++i) {
                const auto want = static_cast<std::uint16_t>(pixels[offset + 2 * i] |
                                                             (pixels[offset + 2 * i + 1] << 8));
                if (patches[i] != want) {
                    ++differing;
                    largest =
                        std::max(largest, std::abs(bf16_value(patches[i]) - bf16_value(want)));
                }
            }
            compared += patches.size();
            offset += 2 * patches.size();
        }
        check(offset == pixels.size(), name + ": patch count differs from the processor");
        // FFmpeg's JPEG IDCT is not libjpeg's: its pixels may differ from PIL's by a few levels.
        bool jpeg = false;
        for (const auto& path : expected.at("media")) {
            jpeg |= path.get<std::string>().ends_with(".jpg");
        }
        check(jpeg ? largest <= 4.0 / (255.0 * 0.26) : differing == 0,
              name + ": " + std::to_string(differing) + " of " + std::to_string(compared) +
                  " patch values differ");
        std::cout << name << ": " << data.token_ids.size() << " tokens, " << differing << " of "
                  << compared << " patch values differ (largest by " << largest << ")\n";
    }
}

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
        const std::string processor        = read_file(dir + "/processor_config.json");
        glm::FrontendResources resources{.tokenizer_json         = tokenizer_json,
                                         .tokenizer_config_json  = tokenizer_config,
                                         .chat_template_jinja    = chat_template,
                                         .generation_config_json = generation,
                                         .processor_config_json  = processor};
        glm::parse_resources(resources, {.vocab_size = 154880});
        glm::FrontendOptions options;
        options.max_context    = 32768;
        options.vision_enabled = true;
        const auto frontend = glm::make_frontend(resources, options);
        if (const char* reference = std::getenv("GLM_PROCESSOR_REFERENCE")) {
            check_processor(frontend, reference);
        }

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
