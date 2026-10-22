// The T2 start-up and correctness smoke test (AGENTS.md "Iteration tiers"): the public Engine on the
// real artifact with production's options (tools/../super-quant serve_infernix.sh: fp8 KV, 6 lanes,
// DFlash2 with an automatic draft length and the proposal head, the prefix cache), in process. Skips
// unless INFERNIX_QWEN4_ARTIFACT and INFERNIX_QWEN4_DFLASH2 are set (INFERNIX_QWEN4_NGRAM: the n-gram
// volume, default <artifact>.ngram).
//
//   production  The fast paths production selects: the engine starts; six concurrent greedy requests
//               complete with DFlash2 drafting and accepting; a repeated prompt resumes from the
//               prefix cache; an expert-placement switch to half the experts in VRAM applies and the
//               engine keeps generating; the switch back applies.
//   exact       The exact route (--disable-fast-paths): each prompt's greedy tokens alone equal the
//               golden tokens (--check FILE; --record FILE writes them), and equal them again with all
//               six in flight, resumed from the prefix cache, and after the placement switch.
//
//   infernix_qwen4_exp_smoke_real_test [--check FILE | --record FILE] [--tokens N]

#include "infernix/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

constexpr const char* kPrompts[] = {
    "Write a Python function that returns the n-th Fibonacci number, then explain its complexity.",
    "The three most important causes of the French Revolution were",
    "Explain to a ten-year-old why the sky is blue, in two short paragraphs.",
    "List five practical tips for writing a clear bug report:",
    "In C++, the difference between std::vector::reserve and std::vector::resize is",
    "The old lighthouse keeper climbed the stairs one last time before the storm. At the top, he",
};
constexpr std::size_t kLanes = std::size(kPrompts);

// A prompt long enough to leave prefix-cache taps (calls of 512 tokens): a paragraph repeated.
Tokens long_prompt(infernix::Engine& engine) {
    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "Section " + std::to_string(i) +
                ". The maintenance log records the pump pressure, the coolant temperature and the "
                "operator on duty every hour, so that a later reader can reconstruct the shift. ";
    }
    Tokens tokens = engine.tokenize_text(text);
    tokens.resize(std::min<std::size_t>(tokens.size(), 1400));
    return tokens;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

std::string first_difference(const Tokens& a, const Tokens& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at token " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

struct Setup {
    const char* artifact = nullptr;
    const char* ngram    = nullptr;
    const char* drafter  = nullptr;
};

infernix::EngineOptions production_options(const Setup& setup, bool fast_paths) {
    infernix::EngineOptions options;
    options.artifact_path = setup.artifact;
    if (setup.ngram != nullptr && *setup.ngram != '\0') { options.ngram_volume_path = setup.ngram; }
    options.kv_cache                       = infernix::KvCacheStorage::Fp8E4M3Row256;
    options.max_context                    = 262144;
    options.max_concurrency                = kLanes;
    options.max_pending_requests           = 2 * kLanes;
    options.fast_paths                     = fast_paths;
    options.speculative.backend            = infernix::SpeculativeBackend::DFlash2;
    options.speculative.draft_model        = setup.drafter;
    options.speculative.draft_tokens_auto  = true;
    options.speculative.proposal_head      = infernix::ProposalHead::Optimized;
    return options;
}

infernix::RequestOptions greedy(std::uint32_t tokens, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<infernix::GenerationResult> run_all(infernix::Engine& engine, const std::vector<Tokens>& prompts,
                                                std::uint32_t tokens, bool reuse) {
    std::vector<infernix::GenerationHandle> handles;
    for (const Tokens& prompt : prompts) { handles.push_back(engine.submit(engine.prepare_tokens(prompt), greedy(tokens, reuse))); }
    std::vector<infernix::GenerationResult> out;
    for (auto& handle : handles) { out.push_back(handle.wait()); }
    return out;
}

// Requests a placement switch and waits until it has applied: the report's last-switch message
// changes once the engine applies a switch at a round boundary (an idle engine at its next request).
bool switch_placement(infernix::Engine& engine, std::uint32_t vram_experts, std::string& message) {
    const std::string before = engine.expert_placement_status().last_switch_message;
    std::string error;
    if (!engine.request_expert_placement({.vram_experts = vram_experts}, &error)) {
        message = "refused: " + error;
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        (void)engine.generate(engine.prepare_tokens(engine.tokenize_text("Hello")), greedy(1, false));
        const auto status = engine.expert_placement_status();
        if (!status.switching && status.last_switch_message != before) {
            message = status.last_switch_message;
            return status.last_switch_ok;
        }
        if (seconds_since(start) > 120) {
            message = "not applied after 120 s";
            return false;
        }
    }
}

std::map<std::string, Tokens> read_golden(const std::filesystem::path& path) {
    std::map<std::string, Tokens> out;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        const auto eq = line.find(" = ");
        if (line.empty() || line.front() == '#' || eq == std::string::npos) { continue; }
        Tokens tokens;
        std::stringstream list(line.substr(eq + 3));
        std::string item;
        while (std::getline(list, item, ',')) { tokens.push_back(static_cast<infernix::TokenId>(std::stol(item))); }
        out[line.substr(0, eq)] = std::move(tokens);
    }
    return out;
}

void write_golden(const std::filesystem::path& path, const std::map<std::string, Tokens>& golden) {
    std::ofstream file(path);
    file << "# T2 golden greedy tokens (infernix_qwen4_exp_smoke_real_test --record), exact route.\n";
    for (const auto& [key, tokens] : golden) {
        file << key << " = ";
        for (std::size_t i = 0; i < tokens.size(); ++i) { file << (i == 0 ? "" : ",") << tokens[i]; }
        file << '\n';
    }
    if (!file) { throw std::runtime_error("cannot write " + path.string()); }
}

int production(const Setup& setup, std::uint32_t tokens) {
    std::printf("== production options (fast paths)\n");
    const auto t0 = std::chrono::steady_clock::now();
    infernix::Engine engine(production_options(setup, true));
    std::printf("  started in %.1f s\n", seconds_since(t0));
    std::vector<Tokens> prompts;
    for (const char* text : kPrompts) { prompts.push_back(engine.tokenize_text(text)); }
    int failures = 0;
    const auto t1      = std::chrono::steady_clock::now();
    const auto results = run_all(engine, prompts, tokens, false);
    std::uint64_t drafted = 0, accepted = 0;
    bool complete = true;
    for (const auto& r : results) {
        complete = complete && r.generated_token_ids.size() == tokens;
        drafted += r.speculative.drafted_tokens;
        accepted += r.speculative.accepted_tokens;
    }
    std::printf("  %zu concurrent requests in %.2f s: drafted %llu, accepted %llu\n", kLanes, seconds_since(t1),
                static_cast<unsigned long long>(drafted), static_cast<unsigned long long>(accepted));
    failures += check(complete, "every concurrent request generated " + std::to_string(tokens) + " tokens");
    failures += check(drafted > 0 && accepted > 0, "DFlash2 drafted and its drafts were accepted");
    const Tokens long_text = long_prompt(engine);
    (void)run_all(engine, {long_text}, tokens, true);
    const auto again = run_all(engine, {long_text}, tokens, true).front();
    failures += check(again.reused_prompt_tokens > 0,
                      "a repeated " + std::to_string(long_text.size()) + "-token prompt resumes from the prefix cache (reused " +
                          std::to_string(again.reused_prompt_tokens) + ")");
    const auto status = engine.expert_placement_status();
    failures += check(status.supported, "the engine reports an expert placement");
    std::string message;
    const std::uint32_t half = status.total_experts / 2, all = status.total_experts;
    bool switched = switch_placement(engine, half, message);
    failures += check(switched, "the switch to " + std::to_string(half) + " VRAM experts applies (" + message + ")");
    const auto after = run_all(engine, {prompts[1]}, tokens, false).front();
    failures += check(after.generated_token_ids.size() == tokens, "the engine generates after the switch");
    switched = switch_placement(engine, all, message);
    failures += check(switched, "the switch back to every expert in VRAM applies (" + message + ")");
    return failures;
}

int exact(const Setup& setup, std::uint32_t tokens, const std::filesystem::path& check_path,
          const std::filesystem::path& record_path) {
    std::printf("== exact route\n");
    const auto t0 = std::chrono::steady_clock::now();
    infernix::Engine engine(production_options(setup, false));
    std::printf("  started in %.1f s\n", seconds_since(t0));
    std::vector<Tokens> prompts;
    for (const char* text : kPrompts) { prompts.push_back(engine.tokenize_text(text)); }
    int failures = 0;
    std::vector<Tokens> solo;
    for (const Tokens& prompt : prompts) { solo.push_back(run_all(engine, {prompt}, tokens, false).front().generated_token_ids); }
    const std::string artifact = std::filesystem::path(setup.artifact).filename().string();
    const auto key = [&](std::size_t i) { return artifact + " prompt=" + std::to_string(i) + " tokens=" + std::to_string(tokens); };
    if (!check_path.empty()) {
        const auto golden = read_golden(check_path);
        for (std::size_t i = 0; i < kLanes; ++i) {
            const auto it = golden.find(key(i));
            failures += check(it != golden.end() && it->second == solo[i],
                              "prompt " + std::to_string(i) + " equals its golden tokens (" +
                                  (it == golden.end() ? std::string("none recorded") : first_difference(solo[i], it->second)) + ")");
        }
    }
    if (!record_path.empty()) {
        auto golden = read_golden(record_path);
        for (std::size_t i = 0; i < kLanes; ++i) { golden[key(i)] = solo[i]; }
        write_golden(record_path, golden);
        std::printf("  recorded %zu prompts into %s\n", kLanes, record_path.c_str());
    }
    const auto together = run_all(engine, prompts, tokens, false);
    for (std::size_t i = 0; i < kLanes; ++i) {
        failures += check(together[i].generated_token_ids == solo[i],
                          "prompt " + std::to_string(i) + " with six in flight equals it alone (" +
                              first_difference(together[i].generated_token_ids, solo[i]) + ")");
    }
    // The cold run that publishes the prompt's tail tap splits its last call there, as the resumed
    // run's call does (the exact route is bit-exact for the same call boundaries).
    const Tokens long_text = long_prompt(engine);
    const auto cold        = run_all(engine, {long_text}, tokens, true).front();
    const auto resumed     = run_all(engine, {long_text}, tokens, true).front();
    failures += check(cold.reused_prompt_tokens == 0 && resumed.reused_prompt_tokens > 0 &&
                          resumed.generated_token_ids == cold.generated_token_ids,
                      "the long prompt resumed from the prefix cache (reused " + std::to_string(resumed.reused_prompt_tokens) +
                          ") equals the run that published it (" +
                          first_difference(resumed.generated_token_ids, cold.generated_token_ids) + ")");
    std::string message;
    const std::uint32_t half = engine.expert_placement_status().total_experts / 2;
    const bool switched      = switch_placement(engine, half, message);
    failures += check(switched, "the switch to " + std::to_string(half) + " VRAM experts applies (" + message + ")");
    const auto placed = run_all(engine, {prompts[3]}, tokens, false).front();
    failures += check(placed.generated_token_ids == solo[3], "prompt 3 after the switch equals it before (" +
                                                                 first_difference(placed.generated_token_ids, solo[3]) + ")");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    Setup setup;
    setup.artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    setup.drafter  = std::getenv("INFERNIX_QWEN4_DFLASH2");
    setup.ngram    = std::getenv("INFERNIX_QWEN4_NGRAM");
    if (setup.artifact == nullptr || *setup.artifact == '\0' || setup.drafter == nullptr || *setup.drafter == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT and INFERNIX_QWEN4_DFLASH2 are not both set\n");
        return 77;
    }
    std::filesystem::path check_path, record_path;
    std::uint32_t tokens = 48;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--check" && i + 1 < argc) {
            check_path = argv[++i];
        } else if (arg == "--record" && i + 1 < argc) {
            record_path = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            tokens = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        int failures  = production(setup, tokens);
        failures += exact(setup, tokens, check_path, record_path);
        std::printf("%s in %.1f s\n", failures == 0 ? "PASS" : "FAIL", seconds_since(t0));
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
