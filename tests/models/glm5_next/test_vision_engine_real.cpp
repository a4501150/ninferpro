// GLM-5.3-Flash's media prompts through the public Engine, in process, on the real artifact
// (INFERNIX_GLM_ARTIFACT; without it the test skips, 77) with the examples' media
// (examples/cli/media):
//
// - An image: the chart's title is read back, also when 512-token prefill calls split the image
//   across calls (the handoff's columns are scattered call by call).
// - An image and a video in one prompt: the title's first word and the video's last number
//   (examples/cli manifest case mixed_image_video, "NIFER-9").
// - Two images and the video side by side, two requests at once: each gives its tokens alone.
// - The prefix cache: a repeat resumes past the media (no encode), and with 512-token calls a
//   prompt sharing the image but asking another question resumes inside the image; both give
//   their cold runs' tokens.
// - Vision offload: these Engines stream the tower from pinned memory (--vision-offload auto); one
//   with the tower in device memory gives the same tokens.

#include "infernix/engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    failures += ok ? 0 : 1;
}

std::vector<std::uint8_t> read_media(const std::string& name) {
    const std::string path = INFERNIX_SOURCE_DIR "/examples/cli/media/" + name;
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot read " + path); }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

infernix::MessagePart media(const std::string& name) {
    const bool video = name.ends_with(".mp4");
    return {.kind  = infernix::MessagePartKind::Media,
            .media = {.kind  = video ? infernix::MediaKind::Video : infernix::MediaKind::Image,
                      .bytes = read_media(name),
                      .source_name = name}};
}

infernix::MessagePart text(std::string value) {
    return {.kind = infernix::MessagePartKind::Text, .text = std::move(value)};
}

infernix::PreparedPrompt prompt_of(infernix::Engine& engine,
                                   std::vector<infernix::MessagePart> parts) {
    infernix::PromptInput input;
    input.messages.push_back({.role = infernix::ChatRole::User, .parts = std::move(parts)});
    input.options.reasoning_effort = infernix::ReasoningEffort::Low; // GLM always thinks
    return engine.prepare(std::move(input));
}

infernix::RequestOptions greedy(std::uint32_t tokens, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    return options;
}

infernix::EngineOptions
engine_options(const char* artifact, std::uint32_t chunk,
               infernix::VisionOffload offload = infernix::VisionOffload::Auto) {
    infernix::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 8192;
    options.max_concurrency                   = 2;
    options.max_pending_requests              = 4;
    options.prefill_chunk                     = chunk;
    options.enable_vision                     = true;
    options.vision_offload                    = offload;
    options.kv_capacity                       = infernix::KvCapacityPolicy::automatic();
    options.experts_spec                      = "vram=40%,ram=60%"; // production's split
    options.context_cache.host_capacity_bytes = 512ULL << 20;
    return options;
}

// The reply's reasoning and content.
std::string reply(const infernix::GenerationResult& result) {
    return result.reasoning + (result.reasoning.empty() ? "" : " | ") + result.content;
}

std::string joined(const Tokens& tokens) {
    std::string out;
    for (const auto t : tokens) { out += (out.empty() ? "" : " ") + std::to_string(t); }
    return out;
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_GLM_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_GLM_ARTIFACT\n");
        return 77;
    }
    try {
        const auto chart = [] {
            return std::vector{media("visual_chart.png"),
                               text("What is the title at the top of this image? Output only the "
                                    "title.")};
        };
        const auto mixed = [] {
            return std::vector{text("先从静态图标题中读取第一个单词。"), media("visual_chart.png"),
                               text("再观察视频末尾 END 标记中的数字。"),
                               media("temporal_events.mp4"),
                               text("将第一个单词与末尾数字用连字符拼接，只输出拼接结果。")};
        };
        const auto many = [] {
            return std::vector{media("compare_left.png"), media("compare_right.png"),
                               media("temporal_events.mp4"),
                               text("How many red circles does each image show? Answer briefly.")};
        };
        constexpr std::uint32_t kAnswer = 512; // the thinking and the answer
        Tokens chart_tokens;
        {
            infernix::Engine engine(engine_options(artifact, 4096));
            const auto start = std::chrono::steady_clock::now();
            auto result      = engine.generate(prompt_of(engine, chart()), greedy(kAnswer, false));
            std::printf(
                "chart (%u prompt tokens, %.2f s): %s\n", result.prompt.prompt_tokens,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                reply(result).c_str());
            check(reply(result).find("NIFER VISION 731") != std::string::npos,
                  "the chart's title is read back");

            result = engine.generate(prompt_of(engine, mixed()), greedy(kAnswer, false));
            std::printf("image and video (%u prompt tokens): %s\n", result.prompt.prompt_tokens,
                        reply(result).c_str());
            check(reply(result).find("NIFER-9") != std::string::npos,
                  "the title's first word and the video's last number");

            chart_tokens =
                engine.generate(prompt_of(engine, chart()), greedy(16, false)).generated_token_ids;
            const auto alone = engine.generate(prompt_of(engine, many()), greedy(24, false));
            std::printf("two images and a video: %s\n", reply(alone).c_str());
            infernix::GenerationResult first, second;
            std::thread other(
                [&] { second = engine.generate(prompt_of(engine, chart()), greedy(16, false)); });
            first = engine.generate(prompt_of(engine, many()), greedy(24, false));
            other.join();
            check(first.generated_token_ids == alone.generated_token_ids &&
                      second.generated_token_ids == chart_tokens,
                  "two media requests at once give their tokens alone");

            // The prefix cache: the cold run publishes and a repeat resumes past the media.
            const auto cold  = engine.generate(prompt_of(engine, chart()), greedy(16, true));
            const auto again = engine.generate(prompt_of(engine, chart()), greedy(16, true));
            std::printf("repeat: %u of %u prompt tokens reused\n", again.reused_prompt_tokens,
                        again.prompt.prompt_tokens);
            check(cold.generated_token_ids == chart_tokens &&
                      again.generated_token_ids == chart_tokens && again.reused_prompt_tokens > 0,
                  "a resumed media prompt gives the cold run's tokens");
        }
        {
            // Calls of 512 tokens split the image's 532 tokens across calls; a tap at 512 lies
            // inside the image, so the same image with another question resumes there and encodes
            // the image again for its remaining columns.
            infernix::Engine engine(engine_options(artifact, 512));
            const auto result = engine.generate(prompt_of(engine, chart()), greedy(kAnswer, true));
            std::printf("chart in 512-token calls: %s\n", reply(result).c_str());
            check(reply(result).find("NIFER VISION 731") != std::string::npos,
                  "the title is read back when the image spans calls");
            const auto other_question = [] {
                return std::vector{media("visual_chart.png"),
                                   text("How many red circles are in this image? Answer with a "
                                        "number.")};
            };
            const auto fresh =
                engine.generate(prompt_of(engine, other_question()), greedy(16, false));
            const auto shared =
                engine.generate(prompt_of(engine, other_question()), greedy(16, true));
            std::printf("shared image: %u of %u prompt tokens reused: %s\n",
                        shared.reused_prompt_tokens, shared.prompt.prompt_tokens,
                        reply(shared).c_str());
            check(shared.generated_token_ids == fresh.generated_token_ids,
                  "a prompt resumed from another's image prefix gives its cold tokens");
            check(shared.reused_prompt_tokens == 512,
                  "the shared prompt resumes at the call boundary inside the image");
        }
        {
            infernix::Engine engine(engine_options(artifact, 4096, infernix::VisionOffload::Off));
            check(engine.generate(prompt_of(engine, chart()), greedy(16, false))
                          .generated_token_ids == chart_tokens,
                  "the tower in device memory gives the streamed tower's tokens");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
