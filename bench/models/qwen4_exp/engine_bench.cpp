// The in-process speed check between T1 and T3 (AGENTS.md "Iteration tiers"): the public Engine on
// the real artifacts with production's options (as tests/models/qwen4_exp/test_smoke_real.cpp), and
// bench_infernix.py's concurrent phase: C chat requests at once (default 6) of N output tokens
// (default 256), greedy, thinking off, each prompt with a unique prefix so the prefix cache serves
// none of it. Its prompt asks for a long essay, so the requests decode all N tokens (the C6
// phase's paragraph often ends after about 100). One warm-up set, then --repeat sets (default 3). Each set reports its aggregate speed as
// bench_infernix.py's C6_agg does (output tokens over the time from the first submission to the
// last completion) and the DFlash2 drafting; the summary gives the median and the range. Reads
// INFERNIX_QWEN4_ARTIFACT, INFERNIX_QWEN4_DFLASH2 and INFERNIX_QWEN4_NGRAM (`make t2-bench`).
//
//   infernix_qwen4_exp_engine_bench [--concurrency C] [--tokens N] [--repeat R]

#include "infernix/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMaxConcurrency = 8;

infernix::EngineOptions production_options(const char* artifact, const char* ngram, const char* drafter,
                                           std::uint32_t lanes) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr && *ngram != '\0') { options.ngram_volume_path = ngram; }
    options.kv_cache                      = infernix::KvCacheStorage::Fp8E4M3Row256;
    options.max_context                   = 262144;
    options.max_concurrency               = lanes;
    options.max_pending_requests          = 2 * lanes;
    options.speculative.backend           = infernix::SpeculativeBackend::DFlash2;
    options.speculative.draft_model       = drafter;
    options.speculative.draft_tokens_auto = true;
    options.speculative.proposal_head     = infernix::ProposalHead::Optimized;
    return options;
}

infernix::PreparedPrompt chat_prompt(infernix::Engine& engine, const std::string& text) {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = text});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return engine.prepare(std::move(input));
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    return options;
}

struct SetResult {
    double seconds = 0, tokens_per_second = 0;
    std::uint64_t tokens = 0, drafted = 0, accepted = 0, rounds = 0, fallback = 0;
};

SetResult run_set(infernix::Engine& engine, std::uint32_t concurrency, std::uint32_t tokens, std::mt19937_64& rng,
                  const char* tag) {
    std::vector<infernix::PreparedPrompt> prompts;
    for (std::uint32_t i = 0; i < concurrency; ++i) {
        char nonce[17];
        std::snprintf(nonce, sizeof nonce, "%08llx", static_cast<unsigned long long>(rng() & 0xFFFFFFFFULL));
        prompts.push_back(chat_prompt(engine, "[bench-" + std::string(tag) + "-c6-" + std::to_string(i) + "-" + nonce +
                                                  "] Write a detailed technical essay of at least 800 words on distributed systems topic " +
                                                  std::to_string(i) +
                                                  ": consensus, replication, partitioning and failure handling."));
    }
    const auto start = std::chrono::steady_clock::now();
    std::vector<infernix::GenerationHandle> handles;
    for (auto& prompt : prompts) { handles.push_back(engine.submit(std::move(prompt), greedy(tokens))); }
    SetResult out;
    for (auto& handle : handles) {
        const infernix::GenerationResult r = handle.wait();
        out.tokens += r.generated_token_ids.size();
        out.drafted += r.speculative.drafted_tokens;
        out.accepted += r.speculative.accepted_tokens;
        out.rounds += r.speculative.rounds;
        out.fallback += r.speculative.fallback_steps;
    }
    out.seconds           = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    out.tokens_per_second = static_cast<double>(out.tokens) / out.seconds;
    return out;
}

void print_set(const char* what, const SetResult& r) {
    std::printf("  %-8s %7.1f tok/s  %5llu tokens in %.2f s  drafted %llu accepted %llu (%.0f %%)  rounds %llu fallback %llu\n",
                what, r.tokens_per_second, static_cast<unsigned long long>(r.tokens), r.seconds,
                static_cast<unsigned long long>(r.drafted), static_cast<unsigned long long>(r.accepted),
                r.drafted ? 100.0 * static_cast<double>(r.accepted) / static_cast<double>(r.drafted) : 0.0,
                static_cast<unsigned long long>(r.rounds), static_cast<unsigned long long>(r.fallback));
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    const char* drafter  = std::getenv("INFERNIX_QWEN4_DFLASH2");
    const char* ngram    = std::getenv("INFERNIX_QWEN4_NGRAM");
    if (artifact == nullptr || *artifact == '\0' || drafter == nullptr || *drafter == '\0') {
        std::fprintf(stderr, "INFERNIX_QWEN4_ARTIFACT and INFERNIX_QWEN4_DFLASH2 must be set\n");
        return 2;
    }
    std::uint32_t concurrency = 6, tokens = 256, repeat = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s needs a value\n", arg.c_str());
            return 2;
        }
        const auto value = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        if (arg == "--concurrency") {
            concurrency = value;
        } else if (arg == "--tokens") {
            tokens = value;
        } else if (arg == "--repeat") {
            repeat = value;
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (concurrency < 1 || concurrency > kMaxConcurrency || tokens < 1 || repeat < 1) {
        std::fprintf(stderr, "--concurrency 1..8, --tokens and --repeat at least 1\n");
        return 2;
    }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        infernix::Engine engine(production_options(artifact, ngram, drafter, concurrency));
        std::printf("engine started in %.1f s; %u concurrent requests of %u tokens\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), concurrency, tokens);
        std::mt19937_64 rng(std::random_device{}());
        print_set("warm-up", run_set(engine, concurrency, tokens, rng, "warm"));
        std::vector<double> speeds;
        for (std::uint32_t r = 0; r < repeat; ++r) {
            const SetResult set = run_set(engine, concurrency, tokens, rng, "set");
            print_set(("set " + std::to_string(r + 1)).c_str(), set);
            speeds.push_back(set.tokens_per_second);
        }
        std::sort(speeds.begin(), speeds.end());
        std::printf("aggregate: median %.1f tok/s, range %.1f-%.1f (%u sets)\n", speeds[speeds.size() / 2], speeds.front(),
                    speeds.back(), repeat);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
