// Compares the glm5_next tokenizer against the Hugging Face tokenizers oracle over a committed
// corpus. The 20 MB tokenizer.json is not committed: point GLM_TOKENIZER_RESOURCES at the
// directory holding tokenizer.json, tokenizer_config.json and generation_config.json. Without
// the variable the test skips (77); the corpus with expected ids lives in
// tests/models/glm5_next/tokenizer_oracle.json and was produced by `tokenizers` itself.
#include "models/glm5_next/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fi = infernix::models::glm5_next::frontend;

using Json = nlohmann::json;

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

} // namespace

int main() {
    const char* resources_dir = std::getenv("GLM_TOKENIZER_RESOURCES");
    if (resources_dir == nullptr || std::string(resources_dir).empty()) {
        std::cout << "skipped: GLM_TOKENIZER_RESOURCES is not set\n";
        return 77;
    }
    const std::string dir(resources_dir);

    const fi::Tokenizer tokenizer(fi::TokenizerResources{
        read_file(dir + "/tokenizer.json"), read_file(dir + "/tokenizer_config.json"),
        read_file(dir + "/generation_config.json")});

    const Json fixture =
        Json::parse(read_file(INFERNIX_SOURCE_DIR "/tests/models/glm5_next/tokenizer_oracle.json"));
    int failures = 0;
    const auto check = [&](bool ok, const std::string& message) {
        if (!ok) {
            ++failures;
            std::cerr << "FAIL: " << message << "\n";
        }
    };

    check(tokenizer.vocab_size() == 154856,
          "vocabulary size is 154820 vocabulary ids plus 36 added tokens");
    check((tokenizer.default_stop_token_ids() == std::vector<int>{154820, 154827, 154829}),
          "generation_config eos ids are the default stop ids");
    check(tokenizer.is_special_token(154828), "<|assistant|> is special");
    check(!tokenizer.is_special_token(154842), "</think> is an ordinary added token");
    check(!tokenizer.is_special_token(154843), "<tool_call> is an ordinary added token");
    check((tokenizer.encode("a</think>b<tool_call>") ==
           std::vector<int>{tokenizer.encode("a")[0], 154842, tokenizer.encode("b")[0], 154843}),
          "</think> and <tool_call> encode as their added tokens inside text");

    std::size_t index  = 0;
    std::size_t tokens = 0;
    for (const auto& item : fixture.at("cases")) {
        const std::string text       = item.at("text");
        const std::vector<int> ids = item.at("ids").get<std::vector<int>>();
        tokens += ids.size();
        const auto actual = tokenizer.encode(text);
        if (actual != ids) {
            ++failures;
            std::cerr << "FAIL: case " << index << " ids differ for text "
                      << Json(text).dump() << "\n";
        }
        if (tokenizer.decode(ids) != text) {
            ++failures;
            std::cerr << "FAIL: case " << index << " decode round trip differs\n";
        }
        ++index;
    }
    std::cout << "checked " << index << " cases, " << tokens << " tokens\n";
    if (failures == 0) {
        std::cout << "real tokenizer oracle passed\n";
        return 0;
    }
    std::cout << failures << " failures\n";
    return 1;
}
