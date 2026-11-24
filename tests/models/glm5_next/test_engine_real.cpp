// GLM-5.3-Flash through the public Engine, in process, on the real converted artifact.
// INFERNIX_GLM_ARTIFACT names the artifact and INFERNIX_GLM_REFERENCE the forward test's reference
// directory (fp64/ and bf16/ runs of one prompt, tools/glm5_next/reference.py --out DIR); without
// both the test skips (77).
//
// - Generation: the reference prompt's greedy tokens, the first the oracle's argmax; the same
//   prompt among four concurrent chat requests gives the same tokens, and every request completes.
// - Expert placement: the engine reports the three tiers; a switch to half the VRAM experts
//   applies, the engine keeps generating, and the greedy tokens do not change (the tiers hold the
//   same records, computed by the same kernels).
// - DFlash2 (with INFERNIX_GLM_DRAFT, the drafter's companion artifact): an Engine drafting 7
//   tokens a round generates the long greedy continuation of a chat prompt; verification licenses
//   drafts (the acceptance is printed) and the tokens follow the plain decode's (a verify call's
//   dense and expert kernels see more columns than a decode round's, so a late divergence is
//   reported, not failed); four concurrent requests, one sampled, each complete (rounds of
//   several rows decode plainly), and a sampled request alone accepts drafts.
// - Causal scoring: an Engine of that purpose scores the prompt against the oracle within the
//   floor the rounded reference sets, with the latent cache in BF16 and in INT8-G64; the INT8
//   scores' distance from the BF16 ones is printed.

#include "infernix/engine.h"
#include "reference_files.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using glm_test::argmax;
using glm_test::log_probability;
using glm_test::read_floats;
using glm_test::read_tokens;

namespace {

using Tokens = std::vector<infernix::TokenId>;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    failures += ok ? 0 : 1;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

infernix::EngineOptions engine_options(const char* artifact) {
    infernix::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.max_concurrency       = 4;
    options.max_pending_requests  = 8;
    options.context_cache.enabled = false;
    options.kv_capacity           = infernix::KvCapacityPolicy::automatic();
    // A RAM tier of 24 GiB: the rest of the experts are read from the artifact (and the test
    // fits its memory cap; every expert pinned would take 160 GiB).
    options.expert_ram_bytes = 24ULL << 30;
    return options;
}

// One user turn of chat text.
infernix::PreparedPrompt chat_of(infernix::Engine& engine, const char* text) {
    infernix::PromptInput input;
    input.messages.push_back({.role  = infernix::ChatRole::User,
                              .parts = {{.kind = infernix::MessagePartKind::Text, .text = text}}});
    return engine.prepare(std::move(input));
}

std::string joined(const Tokens& tokens) {
    std::string out;
    for (const auto t : tokens) { out += (out.empty() ? "" : " ") + std::to_string(t); }
    return out;
}

} // namespace

int main() {
    const char* artifact  = std::getenv("INFERNIX_GLM_ARTIFACT");
    const char* reference = std::getenv("INFERNIX_GLM_REFERENCE");
    if (artifact == nullptr || reference == nullptr) {
        std::printf("SKIP: set INFERNIX_GLM_ARTIFACT and INFERNIX_GLM_REFERENCE\n");
        return 77;
    }
    try {
        const std::string dir = reference;
        const Tokens prompt   = read_tokens(dir + "/fp64/tokens.txt");
        const std::size_t T   = prompt.size();
        constexpr std::uint32_t kTokens = 16;
        const char* draft                   = std::getenv("INFERNIX_GLM_DRAFT");
        constexpr std::uint32_t kLongTokens = 96;
        static constexpr const char* kLongPrompt =
            "List the numbers from one to forty in English words, separated by commas.";
        Tokens alone, plain_long;
        {
            const auto start = std::chrono::steady_clock::now();
            infernix::Engine engine(engine_options(artifact));
            std::printf("engine started in %.1f s\n",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            const std::size_t V = engine.model_metadata().vocab_size;
            const auto ref      = read_floats(dir + "/fp64/logits.f32", T * V);
            // The oracle's argmax over the vocabulary's rows the model may emit.
            const float* last   = ref.data() + (T - 1) * V;
            alone = engine.generate(engine.prepare_tokens(prompt), greedy(kTokens)).generated_token_ids;
            std::printf("greedy: %s\n", joined(alone).c_str());
            check(alone.size() == kTokens, "the reference prompt generated " + std::to_string(kTokens) + " tokens");
            check(!alone.empty() && alone[0] == static_cast<infernix::TokenId>(std::max_element(last, last + V) - last),
                  "the first token is the oracle's argmax");

            // Four concurrent requests: three chat prompts and the reference prompt.
            std::vector<infernix::GenerationHandle> handles;
            for (const char* text : {"Write a haiku about a lighthouse.", "What is 17 * 23? Answer briefly.",
                                     "Name three uses of a paperclip."}) {
                infernix::PromptInput input;
                input.messages.push_back({.role = infernix::ChatRole::User,
                                          .parts = {{.kind = infernix::MessagePartKind::Text, .text = text}}});
                handles.push_back(engine.submit(engine.prepare(std::move(input)), greedy(kTokens)));
            }
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), greedy(kTokens)));
            bool complete = true;
            Tokens beside;
            for (std::size_t i = 0; i < handles.size(); ++i) {
                const infernix::GenerationResult result = handles[i].wait();
                complete = complete && result.generated_token_ids.size() == kTokens;
                if (i + 1 == handles.size()) { beside = result.generated_token_ids; }
                if (i == 0) { std::printf("chat reply: %s\n", result.content.c_str()); }
            }
            check(complete, "four concurrent requests each generated " + std::to_string(kTokens) + " tokens");
            check(beside == alone, "the reference prompt beside three others gives its tokens alone");

            const infernix::ExpertPlacementReport status = engine.expert_placement_status();
            std::printf("placement: %u experts, %u frames (limit %u), %zu in RAM, %u from the artifact\n",
                        status.total_experts, status.frames, status.frame_limit, status.tier_residents,
                        status.ssd_experts);
            check(status.supported && status.total_experts > 0, "the engine reports an expert placement");
            const std::uint32_t half = status.frames / 2;
            std::string error;
            const bool requested = engine.request_expert_placement({.vram_experts = half}, &error);
            check(requested, "a switch to " + std::to_string(half) + " VRAM experts is accepted" +
                                 (requested ? std::string() : " (" + error + ")"));
            infernix::ExpertPlacementReport after = engine.expert_placement_status();
            const auto start_switch                = std::chrono::steady_clock::now();
            while (requested && (after.switching || after.last_switch_message == status.last_switch_message)) {
                (void)engine.generate(engine.prepare_tokens(engine.tokenize_text("Hello")), greedy(1));
                after = engine.expert_placement_status();
                if (std::chrono::steady_clock::now() - start_switch > std::chrono::seconds(120)) { break; }
            }
            check(after.last_switch_ok && after.frames <= half + 1,
                  "the switch applied: " + std::to_string(after.frames) + " frames (" + after.last_switch_message + ")");
            const Tokens switched = engine.generate(engine.prepare_tokens(prompt), greedy(kTokens)).generated_token_ids;
            check(switched == alone, "after the switch the greedy tokens are unchanged");
            if (draft != nullptr) {
                plain_long = engine.generate(chat_of(engine, kLongPrompt), greedy(kLongTokens))
                                 .generated_token_ids;
            }
        }
        {
            // A prompt of more than one prefill call (about 10K tokens at the 4,096-token chunk)
            // streams its experts: its greedy tokens are reported for comparison against a build
            // with the stream off (--prefill-expert-stream off), and the stream's diagnostic
            // reports the copies it made.
            infernix::EngineOptions options = engine_options(artifact);
            options.max_context             = 12288;
            std::mutex diag_mutex;
            std::string stream_line;
            options.diagnostic_observer.callback = [&](const infernix::Diagnostic& d) {
                std::lock_guard lock(diag_mutex);
                if (d.message.find("prefill expert stream:") != std::string::npos) {
                    stream_line = d.message;
                }
            };
            infernix::Engine engine(options);
            std::string text;
            for (int i = 0; i < 550; ++i) {
                text.append(kLongPrompt).append(" Now list them backwards. ");
            }
            const Tokens long_tokens = engine.tokenize_text(text);
            std::printf("long prompt: %zu prompt tokens\n", long_tokens.size());
            check(long_tokens.size() > 8192, "the long prompt spans more than two prefill calls");
            const Tokens long_greedy =
                engine.generate(engine.prepare_tokens(long_tokens), greedy(kTokens))
                    .generated_token_ids;
            {
                std::lock_guard lock(diag_mutex);
                std::printf("long greedy: %s\nstream: %s\n", joined(long_greedy).c_str(),
                            stream_line.empty() ? "(none)" : stream_line.c_str());
            }
            check(long_greedy.size() == kTokens, "the long prompt generated its tokens");
        }
        if (draft != nullptr) {
            infernix::EngineOptions options  = engine_options(artifact);
            options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
            options.speculative.draft_model  = draft;
            options.speculative.draft_tokens = 7;
            infernix::Engine engine(options);
            const infernix::GenerationResult result =
                engine.generate(chat_of(engine, kLongPrompt), greedy(kLongTokens));
            const auto& spec    = result.speculative;
            const auto& drafted = result.generated_token_ids;
            std::size_t agree   = 0;
            while (agree < drafted.size() && agree < plain_long.size() &&
                   drafted[agree] == plain_long[agree]) {
                ++agree;
            }
            std::printf("dflash2: %llu rounds, %llu of %llu drafts accepted (%.2f a round), %llu "
                        "plain steps; %zu of "
                        "%zu tokens agree with plain decode\n",
                        static_cast<unsigned long long>(spec.rounds),
                        static_cast<unsigned long long>(spec.accepted_tokens),
                        static_cast<unsigned long long>(spec.drafted_tokens),
                        spec.rounds ? static_cast<double>(spec.accepted_tokens) /
                                          static_cast<double>(spec.rounds)
                                    : 0.0,
                        static_cast<unsigned long long>(spec.fallback_steps), agree,
                        drafted.size());
            check(drafted.size() == kLongTokens,
                  "the drafting Engine generated " + std::to_string(kLongTokens) + " tokens");
            check(spec.rounds > 0 && spec.accepted_tokens > 0, "verification licensed drafts");
            check(agree >= 16, "the first 16 tokens follow plain decode");
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = 0; i < 4; ++i) {
                infernix::RequestOptions request = greedy(kLongTokens);
                if (i == 3) {
                    request.execution.sampling.temperature = 0.8F;
                    request.execution.sampling.seed        = 7;
                }
                handles.push_back(engine.submit(
                    chat_of(engine, i == 0 ? kLongPrompt : "Explain how a lighthouse lens works."),
                    std::move(request)));
            }
            bool complete          = true;
            std::uint64_t accepted = 0;
            for (auto& handle : handles) {
                const infernix::GenerationResult r = handle.wait();
                complete = complete && r.generated_token_ids.size() == kLongTokens;
                accepted += r.speculative.accepted_tokens;
            }
            std::printf("concurrent drafting requests: %llu drafts accepted (rounds of several "
                        "rows decode plainly)\n",
                        static_cast<unsigned long long>(accepted));
            check(complete,
                  "four concurrent requests (one sampled) beside the drafter each complete");
            {
                // A sampled request alone: its rounds verify by sparse rejection sampling.
                infernix::RequestOptions request       = greedy(kLongTokens);
                request.execution.sampling.temperature = 0.8F;
                request.execution.sampling.seed        = 7;
                const infernix::GenerationResult r     = engine.generate(
                    chat_of(engine, "Explain how a lighthouse lens works."), request);
                std::printf("sampled dflash2: %llu of %llu drafts accepted\n",
                            static_cast<unsigned long long>(r.speculative.accepted_tokens),
                            static_cast<unsigned long long>(r.speculative.drafted_tokens));
                check(r.generated_token_ids.size() == kLongTokens &&
                          r.speculative.accepted_tokens > 0,
                      "a sampled request's drafts are accepted by rejection sampling");
            }
        }
        std::vector<double> bf16_scores;
        for (const auto kv :
             {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64}) {
            const std::string name = kv == infernix::KvCacheStorage::BFloat16 ? "bf16" : "int8";
            infernix::EngineOptions options = engine_options(artifact);
            options.purpose                 = infernix::EnginePurpose::CausalScoring;
            options.max_context = std::max<std::uint32_t>(128, static_cast<std::uint32_t>(T));
            options.kv_cache    = kv;
            infernix::Engine engine(options);
            const std::size_t V      = engine.model_metadata().vocab_size;
            const auto ref           = read_floats(dir + "/fp64/logits.f32", T * V);
            const auto rounded       = read_floats(dir + "/bf16/logits.f32", T * V);
            const auto scored        = engine.score_tokens(prompt, 1);
            const std::size_t domain = V;
            double worst = 0, floor = 0;
            for (std::size_t p = 0; p + 1 < T && p < scored.logprobs.size(); ++p) {
                const double exact = log_probability(ref.data() + p * V, domain, prompt[p + 1]);
                worst = std::max(worst, std::abs(scored.logprobs[p] - exact));
                floor = std::max(floor, std::abs(log_probability(rounded.data() + p * V, domain, prompt[p + 1]) - exact));
            }
            std::printf("scoring, %s KV: worst |log p - oracle| %.4f (rounded reference %.4f)\n",
                        name.c_str(), worst, floor);
            check(scored.logprobs.size() + 1 == T,
                  "the scoring Engine scores every target (" + name + " KV)");
            check(worst < 2 * floor + 0.05,
                  "the scored log-probabilities stay within the rounded reference's floor (" +
                      name + " KV)");
            if (kv == infernix::KvCacheStorage::BFloat16) {
                bf16_scores.assign(scored.logprobs.begin(), scored.logprobs.end());
            } else if (bf16_scores.size() == scored.logprobs.size()) {
                double largest = 0, sum = 0;
                for (std::size_t p = 0; p < bf16_scores.size(); ++p) {
                    const double d = std::abs(scored.logprobs[p] - bf16_scores[p]);
                    largest        = std::max(largest, d);
                    sum += d;
                }
                std::printf("int8 KV against bf16 KV: |log p| difference mean %.4f, largest %.4f "
                            "over %zu targets\n",
                            sum / static_cast<double>(bf16_scores.size()), largest,
                            bf16_scores.size());
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
