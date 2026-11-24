// The T2 start-up and correctness smoke test (AGENTS.md "Iteration tiers") for GLM-5.3-Flash: the
// public Engine on the real artifact with production's options
// (~/src/super-quant/configs/GLM-5.3-Flash/model.env: INT8 KV, 4 x 262,144 tokens,
// vram=40%,ram=60%, the prefix cache), in process. Skips unless INFERNIX_GLM_ARTIFACT and
// INFERNIX_GLM_DRAFT are set.
//
//   production  With DFlash2 (three draft tokens): the engine starts; four concurrent greedy
//   requests
//               complete; one request alone drafts and accepts; a repeated long prompt resumes from
//               the prefix cache; an expert-placement switch to half the VRAM experts applies, the
//               engine keeps generating, and the switch back applies.
//   golden      Without the drafter (a drafting round's width depends on its acceptance): each
//               prompt's greedy tokens alone equal the golden tokens (--check FILE; --record FILE
//               writes them), and equal them again with all four in flight and after the placement
//               switch; a long prompt resumed from its prefix-cache tap equals its cold run. GLM
//               has no exact route: these are the default route's bits, which do not depend on the
//               round's other rows, the experts' tiers or the resume.
//
//   infernix_glm5_next_smoke_real_test [--check FILE | --record FILE] [--tokens N]

#include "infernix/engine.h"

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
    "The old lighthouse keeper climbed the stairs one last time before the storm. At the top, he",
};
constexpr std::size_t kLanes = std::size(kPrompts);

// A prompt of several 4,096-token calls, so it leaves prefix-cache taps on the chunk grid.
Tokens long_prompt(infernix::Engine& engine) {
    std::string text;
    for (int i = 0; i < 260; ++i) {
        text += "Section " + std::to_string(i) +
                ". The maintenance log records the pump pressure, the coolant temperature and the "
                "operator on duty every hour, so that a later reader can reconstruct the shift. ";
    }
    return engine.tokenize_text(text);
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

infernix::EngineOptions production_options(const char* artifact, const char* drafter) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
    options.max_context          = 262144;
    options.kv_capacity          = infernix::KvCapacityPolicy::automatic();
    options.max_concurrency      = kLanes;
    options.max_pending_requests = 2 * kLanes;
    options.experts_spec         = "vram=40%,ram=60%";
    if (drafter != nullptr) {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_model  = drafter;
        options.speculative.draft_tokens = 3;
    }
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

std::vector<infernix::GenerationResult> run_all(infernix::Engine& engine,
                                                const std::vector<Tokens>& prompts,
                                                std::uint32_t tokens, bool reuse) {
    std::vector<infernix::GenerationHandle> handles;
    for (const Tokens& prompt : prompts) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompt), greedy(tokens, reuse)));
    }
    std::vector<infernix::GenerationResult> out;
    for (auto& handle : handles) { out.push_back(handle.wait()); }
    return out;
}

// Requests a placement switch (the RAM tier holds `ram_slots` resident records) and waits until it
// has applied: the report's last-switch message changes once the engine applies a switch at a round
// boundary (an idle engine at its next request).
bool switch_placement(infernix::Engine& engine, std::uint32_t vram_experts, std::uint32_t ram_slots,
                      std::string& message) {
    const std::string before = engine.expert_placement_status().last_switch_message;
    std::string error;
    if (!engine.request_expert_placement({.vram_experts = vram_experts, .ram_experts = ram_slots},
                                         &error)) {
        message = "refused: " + error;
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        (void)engine.generate(engine.prepare_tokens(engine.tokenize_text("Hello")),
                              greedy(1, false));
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
        while (std::getline(list, item, ',')) {
            tokens.push_back(static_cast<infernix::TokenId>(std::stol(item)));
        }
        out[line.substr(0, eq)] = std::move(tokens);
    }
    return out;
}

void write_golden(const std::filesystem::path& path, const std::map<std::string, Tokens>& golden) {
    std::ofstream file(path);
    file << "# T2 golden greedy tokens (infernix_glm5_next_smoke_real_test --record), default "
            "route without the drafter.\n";
    for (const auto& [key, tokens] : golden) {
        file << key << " = ";
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            file << (i == 0 ? "" : ",") << tokens[i];
        }
        file << '\n';
    }
    if (!file) { throw std::runtime_error("cannot write " + path.string()); }
}

int production(const char* artifact, const char* drafter, std::uint32_t tokens) {
    std::printf("== production options (DFlash2)\n");
    const auto t0 = std::chrono::steady_clock::now();
    infernix::Engine engine(production_options(artifact, drafter));
    std::printf("  started in %.1f s\n", seconds_since(t0));
    std::vector<Tokens> prompts;
    for (const char* text : kPrompts) { prompts.push_back(engine.tokenize_text(text)); }
    int failures       = 0;
    const auto t1      = std::chrono::steady_clock::now();
    const auto results = run_all(engine, prompts, tokens, false);
    bool complete      = true;
    for (const auto& r : results) { complete = complete && r.generated_token_ids.size() == tokens; }
    std::printf("  %zu concurrent requests in %.2f s\n", kLanes, seconds_since(t1));
    failures +=
        check(complete, "every concurrent request generated " + std::to_string(tokens) + " tokens");
    const auto alone = run_all(engine, {prompts[0]}, tokens, false).front();
    std::printf("  one request: drafted %llu, accepted %llu\n",
                static_cast<unsigned long long>(alone.speculative.drafted_tokens),
                static_cast<unsigned long long>(alone.speculative.accepted_tokens));
    failures += check(alone.speculative.drafted_tokens > 0 && alone.speculative.accepted_tokens > 0,
                      "DFlash2 drafted for one request and its drafts were accepted");
    const Tokens long_text = long_prompt(engine);
    (void)run_all(engine, {long_text}, tokens, true);
    const auto again = run_all(engine, {long_text}, tokens, true).front();
    failures += check(again.reused_prompt_tokens > 0,
                      "a repeated " + std::to_string(long_text.size()) +
                          "-token prompt resumes from the prefix cache (reused " +
                          std::to_string(again.reused_prompt_tokens) + ")");
    const auto status = engine.expert_placement_status();
    failures += check(status.supported, "the engine reports an expert placement");
    std::string message;
    const std::uint32_t frames = status.frames, half = frames / 2,
                        ram = static_cast<std::uint32_t>(status.tier_residents);
    bool switched           = switch_placement(engine, half, ram, message);
    failures += check(switched, "the switch to " + std::to_string(half) + " VRAM experts and " +
                                    std::to_string(ram) + " RAM slots applies (" + message + ")");
    const auto after = run_all(engine, {prompts[1]}, tokens, false).front();
    failures +=
        check(after.generated_token_ids.size() == tokens, "the engine generates after the switch");
    switched = switch_placement(engine, frames, ram, message);
    failures += check(switched, "the switch back to " + std::to_string(frames) +
                                    " VRAM experts applies (" + message + ")");
    return failures;
}

int golden(const char* artifact, std::uint32_t tokens, const std::filesystem::path& check_path,
           const std::filesystem::path& record_path) {
    std::printf("== golden tokens (no drafter)\n");
    const auto t0 = std::chrono::steady_clock::now();
    infernix::Engine engine(production_options(artifact, nullptr));
    std::printf("  started in %.1f s\n", seconds_since(t0));
    std::vector<Tokens> prompts;
    for (const char* text : kPrompts) { prompts.push_back(engine.tokenize_text(text)); }
    int failures = 0;
    std::vector<Tokens> solo;
    for (const Tokens& prompt : prompts) {
        solo.push_back(run_all(engine, {prompt}, tokens, false).front().generated_token_ids);
    }
    const std::string name = std::filesystem::path(artifact).filename().string();
    const auto key         = [&](std::size_t i) {
        return name + " prompt=" + std::to_string(i) + " tokens=" + std::to_string(tokens);
    };
    if (!check_path.empty()) {
        const auto recorded = read_golden(check_path);
        for (std::size_t i = 0; i < kLanes; ++i) {
            const auto it = recorded.find(key(i));
            failures += check(it != recorded.end() && it->second == solo[i],
                              "prompt " + std::to_string(i) + " equals its golden tokens (" +
                                  (it == recorded.end() ? std::string("none recorded")
                                                        : first_difference(solo[i], it->second)) +
                                  ")");
        }
    }
    if (!record_path.empty()) {
        auto recorded = read_golden(record_path);
        for (std::size_t i = 0; i < kLanes; ++i) { recorded[key(i)] = solo[i]; }
        write_golden(record_path, recorded);
        std::printf("  recorded %zu prompts into %s\n", kLanes, record_path.c_str());
    }
    const auto together = run_all(engine, prompts, tokens, false);
    for (std::size_t i = 0; i < kLanes; ++i) {
        failures += check(together[i].generated_token_ids == solo[i],
                          "prompt " + std::to_string(i) + " with four in flight equals it alone (" +
                              first_difference(together[i].generated_token_ids, solo[i]) + ")");
    }
    const Tokens long_text = long_prompt(engine);
    const auto cold        = run_all(engine, {long_text}, tokens, false).front();
    (void)run_all(engine, {long_text}, tokens, true);
    Tokens shared(long_text.begin(), long_text.end() - 8);
    const Tokens tail = engine.tokenize_text(" Summarize the log.");
    shared.insert(shared.end(), tail.begin(), tail.end());
    const auto shared_cold    = run_all(engine, {shared}, tokens, false).front();
    const auto shared_resumed = run_all(engine, {shared}, tokens, true).front();
    failures += check(
        cold.generated_token_ids.size() == tokens && shared_resumed.reused_prompt_tokens > 0 &&
            shared_resumed.generated_token_ids == shared_cold.generated_token_ids,
        "a prompt sharing the long prompt's prefix resumed from its tap (reused " +
            std::to_string(shared_resumed.reused_prompt_tokens) + ") equals its cold run (" +
            first_difference(shared_resumed.generated_token_ids, shared_cold.generated_token_ids) +
            ")");
    std::string message;
    const auto status        = engine.expert_placement_status();
    const std::uint32_t half = status.frames / 2;
    const bool switched =
        switch_placement(engine, half, static_cast<std::uint32_t>(status.tier_residents), message);
    failures += check(switched, "the switch to " + std::to_string(half) +
                                    " VRAM experts applies (" + message + ")");
    const auto placed = run_all(engine, {prompts[3]}, tokens, false).front();
    failures += check(placed.generated_token_ids == solo[3],
                      "prompt 3 after the switch equals it before (" +
                          first_difference(placed.generated_token_ids, solo[3]) + ")");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* artifact = std::getenv("INFERNIX_GLM_ARTIFACT");
    const char* drafter  = std::getenv("INFERNIX_GLM_DRAFT");
    if (artifact == nullptr || *artifact == '\0' || drafter == nullptr || *drafter == '\0') {
        std::printf("SKIP: INFERNIX_GLM_ARTIFACT and INFERNIX_GLM_DRAFT are not both set\n");
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
        int failures  = production(artifact, drafter, tokens);
        failures += golden(artifact, tokens, check_path, record_path);
        std::printf("%s in %.1f s\n", failures == 0 ? "PASS" : "FAIL", seconds_since(t0));
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
