// The >2,051-token multimodal wiring oracle (design §19.3.2, test VT6c; AGENTS.md "Verification"):
// QSA equals causal attention below 2,051 visible tokens, so a media prompt past that threshold is
// the only route whose M-RoPE staging, vision scatter and pooled-key selection a short prompt
// cannot hide. The prompt is ~200 text tokens, a 32 x 32 merged-grid image (1,024 tokens) and ~900
// text tokens (~2,130 in all), prepared through the real frontend (tokenizer, chat template,
// processor) on the real artifact; the test dumps the prepared prompt's token ids, M-RoPE positions
// and rope delta, the Vision tower's BF16 patches and the patch grid for the independent FP64
// reference, and runs the same prompt's first greedy tokens through the public Engine's exact
// route. Skips unless INFERNIX_QWEN4_ARTIFACT is set (INFERNIX_QWEN4_NGRAM: the n-gram volume,
// default <artifact>.ngram).
//
//   infernix_qwen4_exp_vision_oracle_real_test OUT_DIR
//
// OUT_DIR receives ids.txt (whitespace-separated), positions.txt ("delta D", then one "a0 a1 a2"
// line per token), patches.bin (raw little-endian BF16 bits, P x 1536), grid.txt (t h w and the
// merged token count) and greedy.txt (the Engine's first greedy token ids). Compare with the FP64
// reference (positions exact, greedy token 1 exact) through tools/flash_next/vision_oracle.py:
//
//   .venv-convert/bin/python tools/flash_next/vision_oracle.py \
//       --model ~/.cache/.../snapshots/HASH --dump OUT_DIR
//
// The FP64 reference run needs the host RAM the engine holds; run it after the test exits.

#include "infernix/engine.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/load_options.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen4_exp/config.h"
#include "models/registry.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q35 = infernix::models::qwen3_5;
namespace q4  = infernix::models::qwen4_exp;

using Tokens = std::vector<infernix::TokenId>;

// The test's own geometry: 1024 x 1024 keeps smart_resize (factor 32) unchanged, so the image
// packs into a 64 x 64 raw patch grid and merges to 32 x 32 = 1,024 Vision tokens.
constexpr int kImageSide = 1024;

// Striped blocks over a diagonal gradient: color edges at every block boundary and a smooth
// gradient within, so both sharp and smooth structure reach the tower.
std::vector<std::uint8_t> striped_ppm() {
    std::vector<std::uint8_t> ppm;
    for (const char byte : std::string("P6\n1024 1024\n255\n")) {
        ppm.push_back(static_cast<std::uint8_t>(byte));
    }
    for (int y = 0; y < kImageSide; ++y) {
        for (int x = 0; x < kImageSide; ++x) {
            const std::uint8_t r = (x / 128) % 2 == 0 ? 210 : 40;
            const std::uint8_t g = (y / 128) % 2 == 0 ? 200 : 50;
            const std::uint8_t b = static_cast<std::uint8_t>((x + y) / 8);
            ppm.insert(ppm.end(), {r, g, b});
        }
    }
    return ppm;
}

// ~200 tokens before the image, ~900 after (the design's VT6c shape). Fixed text, not sampled: the
// dump and the reference must see the same ids.
std::string head_text() {
    std::string text = "Study the image and answer the questions after it. This machine's maintenance "
                       "manual covers the pump assembly, the coolant loop and the operator console. ";
    for (int i = 0; i < 3; ++i) {
        text += "Section " + std::to_string(i) +
                ". The maintenance log records the pump pressure, the coolant temperature and the "
                "operator on duty every hour, so that a later reader can reconstruct the shift. ";
    }
    return text;
}

std::string tail_text() {
    std::string text;
    for (int i = 0; i < 28; ++i) {
        text += "Record " + std::to_string(i) +
                ". The night crew replaced a valve gasket, logged the coolant temperature, and noted "
                "that the pump pressure held steady while the console showed no warnings at all. ";
    }
    return text;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

infernix::EngineOptions oracle_options(const char* artifact, const char* ngram) {
    infernix::EngineOptions options;
    options.artifact_path       = artifact;
    if (ngram != nullptr && *ngram != '\0') { options.ngram_volume_path = ngram; }
    options.kv_cache            = infernix::KvCacheStorage::BFloat16;
    options.max_context         = 8192;
    options.kv_capacity         = infernix::KvCapacityPolicy::explicit_capacity(8192);
    options.max_concurrency     = 1;
    options.max_pending_requests = 1;
    options.fast_paths          = false; // the exact route: request-invariant, reference-comparable
    options.enable_vision       = true;
    return options;
}

infernix::PromptInput media_input(const std::vector<std::uint8_t>& image) {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = head_text()});
    message.parts.push_back({.kind = infernix::MessagePartKind::Media,
                             .media = infernix::OwnedMedia{.kind      = infernix::MediaKind::Image,
                                                           .bytes    = image,
                                                           .media_type = "image/x-portable-pixmap"}});
    message.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = tail_text()});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

// The Engine's own frontend (construct_qwen4_exp) binds the artifact's tokenizer and processor
// resources without any weight; the same binding here prepares the prompt without a second model
// boot. The frontend's resource string views borrow the Reader's host backing, so the holder keeps
// the Reader and Binder alive beside it. The field list mirrors construct_qwen4_exp's FrontendOptions.
struct FrontendHolder {
    std::unique_ptr<infernix::artifact::Reader> reader;
    std::unique_ptr<infernix::artifact::Binder> binder;
    q35::Frontend frontend;
};

FrontendHolder artifact_frontend(const char* artifact, const infernix::EngineOptions& options) {
    auto reader  = std::make_unique<infernix::artifact::Reader>(artifact);
    auto binder  = std::make_unique<infernix::artifact::Binder>(*reader);
    const q4::Config config =
        q4::parse_config(reader->directory(),
                         infernix::models::load_options(options, infernix::models::Architecture::Qwen4Exp));
    if (!config.vision) { throw std::runtime_error("the artifact carries no Vision component"); }
    const auto bytes = [&](std::string_view component, std::string_view role) {
        const auto span = binder->host_object(binder->resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(span.data()), span.size());
    };
    q35::FrontendResources resources;
    resources.tokenizer_json                 = bytes("text", "tokenizer.json");
    resources.tokenizer_config_json          = bytes("text", "tokenizer_config.json");
    resources.chat_template_jinja            = bytes("text", "chat_template.jinja");
    resources.generation_config_json         = bytes("text", "generation_config.json");
    resources.preprocessor_config_json       = bytes("vision", "preprocessor_config.json");
    resources.video_preprocessor_config_json = bytes("vision", "video_preprocessor_config.json");
    q35::Config domain;
    domain.text.vocab_size = config.text.vocab_size;
    domain.vision          = config.vision;
    q35::parse_resources(resources, domain);
    auto frontend = q35::make_frontend(resources, {.chat_template_path       = options.chat_template_path,
                                                   .architecture             = infernix::models::Architecture::Qwen4Exp,
                                                   .vision_enabled           = true,
                                                   .max_context              = options.max_context,
                                                   .media_cache_bytes        = options.media_cache_bytes,
                                                   .media_live_bytes         = options.media_live_bytes,
                                                   .media_preprocess_threads = options.media_preprocess_threads,
                                                   .vision_max_merged_tokens = options.vision_max_merged_tokens,
                                                   .thinking_budget_message  = options.thinking_budget_message});
    return {std::move(reader), std::move(binder), std::move(frontend)};
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OUT_DIR\n", argv[0]);
        return 2;
    }
    try {
        const std::string out_dir = argv[1];
        const infernix::EngineOptions options = oracle_options(artifact, std::getenv("INFERNIX_QWEN4_NGRAM"));
        int failures = 0;

        std::printf("== prepared prompt through the real frontend\n");
        const auto image = striped_ppm();
        const auto input = media_input(image);
        auto frontend    = artifact_frontend(artifact, options);
        auto prepared    = frontend.frontend.prepare(input);
        if (!prepared) { throw std::runtime_error("the frontend refused the media prompt"); }
        const q35::PreparedPromptData& data = q35::PreparedPromptAccess::view(prepared);

        const std::size_t n = data.token_ids.size();
        const q35::VisionItem& item = data.vision_items.front();
        const auto grid = item.grid;
        failures += check(data.has_media() && data.vision_items.size() == 1, "the prompt carries one image");
        failures += check(item.modality == q35::PromptModality::Image && grid.temporal == 1 &&
                              grid.height == 64 && grid.width == 64,
                          "the image packs into a 64 x 64 raw patch grid (t 1)");
        failures += check(static_cast<std::uint64_t>(n) > 2051,
                          "the prompt passes the 2,051-token QSA threshold (" + std::to_string(n) + " tokens)");
        failures += check(data.prepare.vision_tokens == 1024,
                          "the image merges into 1,024 Vision tokens (" +
                              std::to_string(data.prepare.vision_tokens) + ")");
        failures += check(item.patch_count == 1ULL * 64 * 64 &&
                              data.media_payloads.front()->patch_elements ==
                                  item.patch_count * q35::kPreparedVisionPatchFeatures,
                          "the item's patches match its grid (" + std::to_string(item.patch_count) +
                              " patches, " + std::to_string(data.media_payloads.front()->patch_elements) +
                              " features)");
        failures += check(data.rope_delta < 0, "the media prompt's rope delta is negative (" +
                          std::to_string(data.rope_delta) + ")");
        const std::int64_t maximum = [&] {
            std::int64_t max = 0;
            for (int a = 0; a < 3; ++a) {
                for (const std::int32_t p : data.position_axis(a)) { max = std::max<std::int64_t>(max, p); }
            }
            return max;
        }();
        failures += check(maximum + 1 - static_cast<std::int64_t>(n) == data.rope_delta,
                          "the rope delta closes the position maximum (max " + std::to_string(maximum) + ")");

        {
            std::ofstream file(out_dir + "/ids.txt");
            for (const auto id : data.token_ids) { file << id << ' '; }
            std::ofstream positions(out_dir + "/positions.txt");
            positions << "delta " << data.rope_delta << '\n';
            for (std::size_t i = 0; i < n; ++i) {
                positions << data.position_axis(0)[i] << ' ' << data.position_axis(1)[i] << ' '
                          << data.position_axis(2)[i] << '\n';
            }
            const auto& patches = data.media_payloads.front();
            std::ofstream raw(out_dir + "/patches.bin", std::ios::binary);
            raw.write(reinterpret_cast<const char*>(patches->patches.get()),
                      static_cast<std::streamsize>(patches->patch_elements * sizeof(std::uint16_t)));
            std::ofstream grid_file(out_dir + "/grid.txt");
            grid_file << grid.temporal << ' ' << grid.height << ' ' << grid.width << ' '
                      << data.prepare.vision_tokens << '\n';
            if (!positions || !raw || !grid_file) {
                throw std::runtime_error("cannot write the dump into " + out_dir);
            }
        }
        std::printf("  dumped ids, positions, patches and grid into %s\n", out_dir.c_str());

        std::printf("== first greedy tokens through the public Engine (exact route)\n");
        infernix::Engine engine(options);
        infernix::RequestOptions greedy;
        greedy.execution.requested_output_tokens = 4;
        greedy.execution.sampling.temperature    = 0.0F;
        greedy.execution.allow_prefix_reuse      = false;
        greedy.stop.include_model_defaults       = false;
        const auto result = engine.generate(engine.prepare(input), greedy);
        failures += check(result.prompt.prompt_tokens == n,
                          "the Engine's prompt equals the frontend's (" +
                              std::to_string(result.prompt.prompt_tokens) + " vs " + std::to_string(n) + ")");
        failures += check(result.generated_token_ids.size() == 4, "the Engine generated four greedy tokens");
        {
            std::ofstream file(out_dir + "/greedy.txt");
            for (const auto id : result.generated_token_ids) { file << id << ' '; }
        }
        std::printf("  greedy:");
        for (const auto id : result.generated_token_ids) { std::printf(" %d", static_cast<int>(id)); }
        std::printf("\n%s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
