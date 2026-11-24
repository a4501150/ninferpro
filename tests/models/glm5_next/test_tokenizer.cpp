#include "models/glm5_next/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = infernix::models::glm5_next::frontend;

static constexpr std::string_view kGlmSplitPattern =
    R"glm((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)glm";

int failures = 0;

void check(bool ok, const std::string& message) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

// The GPT-2 byte-to-unicode alphabet the ByteLevel pre-tokenizer and decoder use: visible
// ASCII/latin bytes map to themselves, the rest to consecutive codepoints from 256.
std::string byte_encoder_byte(unsigned char byte) {
    const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                         (byte >= 174 && byte <= 255);
    if (visible && byte < 0x80) { return std::string(1, static_cast<char>(byte)); }
    // A visible high byte maps to its own codepoint; an invisible one to consecutive
    // codepoints from 256 upward, exactly as the ByteLevel alphabet defines them.
    const unsigned int codepoint =
        visible ? byte : [&] {
            unsigned int value = 256;
            for (int b = 0; b < byte; ++b) {
                const bool v =
                    (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
                if (!v) { ++value; }
            }
            return value;
        }();
    std::string out;
    if (codepoint < 0x80) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return out;
}

std::vector<int> byte_vocab_ids() {
    std::vector<int> ids(256);
    for (int byte = 0; byte < 256; ++byte) { ids[byte] = byte; }
    return ids;
}

struct Fixture {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
};

Fixture build_fixture(const Json& extra_vocab, const Json& merges, const Json& added_tokens) {
    Json vocab = Json::object();
    for (int byte = 0; byte < 256; ++byte) {
        vocab[byte_encoder_byte(static_cast<unsigned char>(byte))] = byte;
    }
    for (const auto& item : extra_vocab.items()) { vocab[item.key()] = item.value(); }

    Json split{{"type", "Split"},
               {"pattern", Json{{"Regex", kGlmSplitPattern}}},
               {"behavior", "Isolated"},
               {"invert", false}};
    Json bytes{{"type", "ByteLevel"},
               {"add_prefix_space", false},
               {"trim_offsets", true},
               {"use_regex", false}};
    Json root;
    root["version"] = "1.0";
    root["truncation"] = nullptr;
    root["padding"] = nullptr;
    root["added_tokens"] = added_tokens;
    root["normalizer"] = nullptr;
    root["pre_tokenizer"] = Json{{"type", "Sequence"},
                                 {"pretokenizers", Json::array({split, bytes})}};
    root["post_processor"] = nullptr;
    root["decoder"] = Json{{"type", "ByteLevel"}};
    root["model"] = Json{{"type", "BPE"},
                         {"dropout", nullptr},
                         {"unk_token", nullptr},
                         {"continuing_subword_prefix", nullptr},
                         {"end_of_word_suffix", nullptr},
                         {"fuse_unk", false},
                         {"byte_fallback", false},
                         {"ignore_merges", true},
                         {"vocab", vocab},
                         {"merges", merges}};
    Json config;
    config["added_tokens_decoder"] = Json::object();
    Json generation{{"eos_token_id", Json::array({154820, 154827})}};
    return Fixture{root.dump(), config.dump(), generation.dump()};
}

std::vector<int> encode(const fi::Tokenizer& tokenizer, const std::string& text) {
    return tokenizer.encode(text);
}

// Runs the GLM-specific checks over one tokenizer fixture.
void exercise(const fi::Tokenizer& tokenizer) {
    // Digit runs split into groups of at most three (\p{N}{1,3}, not Qwen's single \p{N}).
    check(encode(tokenizer, "1").size() == 1, "single digit is one token");
    check(encode(tokenizer, "123").size() == 3, "three digits are three byte tokens");
    check(encode(tokenizer, "1234").size() == 4, "four digits split 3+1");
    check(encode(tokenizer, "1234567890").size() == 10, "ten digits split 3+3+3+1");

    // Marks do not continue letter runs (Qwen's \p{L}\p{M} vs GLM's \p{L}).
    check(encode(tokenizer, "\u00e9").size() == 2, "a precomposed letter is its own run");
    check(encode(tokenizer, "e\u0301").size() == 3,
          "a combining mark splits the letter run (e, then the mark's bytes)");

    // Contractions, case-insensitive.
    check(encode(tokenizer, "it's").size() == 4, "contraction splits after the apostrophe");
    check(encode(tokenizer, "IT'S").size() == 4, "upper-case contraction splits the same way");

    // Whitespace: a run ending in newlines groups together; an interior run keeps the last
    // space with the next word.
    check(encode(tokenizer, "a\n\n\nb").size() == 5, "a newline run is one pre-token");
    check(encode(tokenizer, "a b").size() == 3, "single interior space joins the next word");
    check(encode(tokenizer, "x\r\ny").size() == 4, "CRLF is its own pre-token");

    // Decode round trips exactly.
    const std::string text = "Mix 12345 \u4e2d\u6587 \U0001f44d\U0001f3fd \r\n\t\u00e9 <|thing|>";
    const auto ids = encode(tokenizer, text);
    check(tokenizer.decode(ids) == text, "decode round trip");

    // Added tokens split as units.
    const auto with_token = encode(tokenizer, "pre<|thing|>post");
    const auto pre = encode(tokenizer, "pre");
    const auto post = encode(tokenizer, "post");
    check(with_token.size() == pre.size() + 1 + post.size(), "added token is a single unit");
    check(with_token.at(pre.size()) == 300, "added token carries its id");

    // ignore_merges: a word whose whole byte-level form is a vocabulary entry is one token,
    // while a word only reachable through merges still merges normally.
    check(encode(tokenizer, "hello").size() == 1,
          "ignore_merges returns a whole-vocabulary word directly");
    check(encode(tokenizer, "helo").size() > 1, "ordinary words still merge from bytes");
    check(tokenizer.decode(encode(tokenizer, "hello!")) == "hello!",
          "a suffix after an ignored word decodes back");

    // Invalid UTF-8 input is rejected, as under the Qwen tokenizer.
    bool threw = false;
    try {
        (void)encode(tokenizer, std::string("a\xff" "b"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "invalid UTF-8 input is rejected");
}

} // namespace

int main() {
    // Vocabulary beyond the 256 byte tokens: "he", "lo", "hel", "hello" as merge targets, and
    // two added tokens (one appearing inside ordinary text).
    Json extra_vocab = Json::object();
    extra_vocab["he"] = 256;
    extra_vocab["lo"] = 257;
    extra_vocab["hel"] = 258;
    extra_vocab["hello"] = 259;
    const Json merges      = Json::array({"h e", "he l", "hel lo"});
    const Json added_tokens = Json::array({Json{{"id", 300},
              {"content", "<|thing|>"},
              {"single_word", false},
              {"lstrip", false},
              {"rstrip", false},
              {"normalized", false},
              {"special", false}},
         Json{{"id", 301},
              {"content", "<|other|>"},
              {"single_word", false},
              {"lstrip", false},
              {"rstrip", false},
              {"normalized", false},
              {"special", true}}});
    const Fixture fixture = build_fixture(extra_vocab, merges, added_tokens);
    const fi::Tokenizer tokenizer({fixture.tokenizer_json, fixture.tokenizer_config_json,
                                   fixture.generation_config_json});
    exercise(tokenizer);

    check((tokenizer.default_stop_token_ids() == std::vector<int>{154820, 154827}),
          "generation config stop ids are read");
    check(tokenizer.vocab_size() == 302, "vocabulary size covers bytes, model, added");
    check(tokenizer.is_special_token(301) && !tokenizer.is_special_token(300),
          "added-token special flags are honored");

    const auto reject = [&](auto mutate, const std::string& what) {
        Json changed = Json::parse(fixture.tokenizer_json);
        mutate(changed);
        const fi::TokenizerResources resources{changed.dump(), fixture.tokenizer_config_json,
                                               fixture.generation_config_json};
        bool threw = false;
        try {
            const fi::Tokenizer refused(resources);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, what);
    };
    reject([](Json& j) { j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\w+"; },
           "a changed pre-tokenizer pattern is refused");
    reject([](Json& j) { j["model"]["ignore_merges"] = false; },
           "ignore_merges=false is refused");
    reject([](Json& j) { j["normalizer"] = Json{{"type", "NFC"}}; },
           "an explicit normalizer is refused");

    if (failures == 0) {
        std::cout << "tokenizer tests passed\n";
        return 0;
    }
    std::cout << failures << " failures\n";
    return 1;
}
