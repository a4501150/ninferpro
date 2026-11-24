// The GLM-5.3-Flash prefix cache on the real converted artifact, through the public Engine
// (INFERNIX_GLM_ARTIFACT; without it the test skips, 77). Prompts of about 7,700 tokens prefill in
// 1,024-token calls, so their layer walk captures the ladder's taps in place. A request with
// prefix reuse off neither reads nor writes the cache: it is the cold reference on the same Engine.
//
// - Shared prefix: a prompt that leaves a cached one late resumes from a tap on the chunk grid and
//   generates its cold greedy tokens.
// - Exact repeat: the prompt-end snapshot regenerates the first token with no prefill, and the
//   continuation is the first run's.
// - Two requests resuming from the cache at once each give their cold tokens.
// - With INFERNIX_GLM_DRAFT: the same shared-prefix resume on a DFlash2 Engine (the taps carry the
//   drafter's context ring) whose Host tier holds only a few snapshots, so the first prompt's own
//   taps evict each other; the resumed prompt's tokens equal its cold drafting run's.

#include "infernix/engine.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    failures += ok ? 0 : 1;
}

constexpr std::uint32_t kTokens = 32;

infernix::RequestOptions greedy(bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kTokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::mutex log_mutex;
int walk_spans = 0;

infernix::EngineOptions engine_options(const char* artifact) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 16384;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    options.prefill_chunk        = 1024;
    options.kv_capacity          = infernix::KvCapacityPolicy::automatic();
    options.experts_spec         = "vram=40%,ram=60%"; // production's split (no SSD-only experts)
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        const bool walk = d.message.rfind("prefill layer walk:", 0) == 0;
        if (walk || d.level != infernix::DiagnosticLevel::Debug) {
            const std::lock_guard<std::mutex> lock(log_mutex);
            walk_spans += walk ? 1 : 0;
            std::printf("  [engine] %s\n", d.message.c_str());
        }
    };
    return options;
}

// A long user turn: numbered lines of words, then `question`.
infernix::PreparedPrompt prompt_of(infernix::Engine& engine, const std::string& question) {
    static const char* kWords[] = {"lighthouse", "harbor", "keeper", "lens",   "beam",
                                   "storm",      "coast",  "rock",   "signal", "night",
                                   "ship",       "tide",   "wave",   "fog"};
    std::string text;
    for (int line = 0; line < 760; ++line) {
        text += std::to_string(line) + ". ";
        for (int w = 0; w < 6; ++w) { text += std::string(kWords[(line * 7 + w * 3) % 14]) + " "; }
        text += "\n";
    }
    text += question;
    infernix::PromptInput input;
    input.messages.push_back({.role  = infernix::ChatRole::User,
                              .parts = {{.kind = infernix::MessagePartKind::Text, .text = text}}});
    return engine.prepare(std::move(input));
}

struct Run {
    Tokens tokens;
    std::uint32_t reused = 0, computed = 0;
};

Run run(infernix::Engine& engine, const std::string& question, bool reuse, const char* what) {
    const infernix::GenerationResult r =
        engine.generate(prompt_of(engine, question), greedy(reuse));
    std::printf("%s: %u reused, %u computed, prefill %.2f s\n", what, r.reused_prompt_tokens,
                r.computed_prefill_tokens, r.timings.prefill_seconds);
    return {r.generated_token_ids, r.reused_prompt_tokens, r.computed_prefill_tokens};
}

const std::string kFirst  = "Question: which word ends line 300? Answer in one word.";
const std::string kSecond = "Question: which word starts line 500? Answer in one word.";

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* artifact = std::getenv("INFERNIX_GLM_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_GLM_ARTIFACT\n");
        return 77;
    }
    const char* draft = std::getenv("INFERNIX_GLM_DRAFT");
    try {
        {
            infernix::Engine engine(engine_options(artifact));
            const Run cold = run(engine, kSecond, false, "second prompt, cold reference");
            check(cold.reused == 0 && cold.tokens.size() == kTokens,
                  "the cold reference prefills from the root");
            const Run a = run(engine, kFirst, true, "first prompt");
            check(a.reused == 0 && a.tokens.size() == kTokens,
                  "the first prompt prefills from the root");
            check(walk_spans > 0, "the long prompts prefill through the layer walk");

            const Run b = run(engine, kSecond, true, "second prompt, after the first");
            check(b.reused > 0 && b.reused % 1024 == 0,
                  "the second prompt resumes from a tap on the chunk grid");
            check(b.tokens == cold.tokens, "the resumed prompt generates its cold greedy tokens");

            const Run repeat = run(engine, kFirst, true, "first prompt, repeated");
            check(repeat.computed == 0, "an exact repeat prefills nothing");
            check(repeat.tokens == a.tokens, "an exact repeat generates the first run's tokens");

            infernix::GenerationHandle h1 = engine.submit(prompt_of(engine, kFirst), greedy(true));
            infernix::GenerationHandle h2 = engine.submit(prompt_of(engine, kSecond), greedy(true));
            const infernix::GenerationResult r1 = h1.wait(), r2 = h2.wait();
            check(r1.generated_token_ids == a.tokens && r2.generated_token_ids == cold.tokens,
                  "two cached prompts at once each generate their cold tokens");
        }
        if (draft != nullptr) {
            infernix::EngineOptions options  = engine_options(artifact);
            options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
            options.speculative.draft_model  = draft;
            options.speculative.draft_tokens = 7;
            // About three snapshot images: the first prompt's taps evict each other.
            options.context_cache.host_capacity_bytes = 512ULL << 20;
            infernix::Engine engine(options);
            const Run cold = run(engine, kSecond, false, "drafting: second prompt, cold reference");
            const Run a    = run(engine, kFirst, true, "drafting: first prompt");
            check(a.reused == 0 && a.tokens.size() == kTokens,
                  "drafting: the first prompt prefills from the root");
            const Run b = run(engine, kSecond, true, "drafting: second prompt, after the first");
            check(b.reused % 1024 == 0, "drafting: the second prompt resumes on the chunk grid");
            check(b.tokens == cold.tokens,
                  "drafting: the resumed prompt generates its cold drafting tokens");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
