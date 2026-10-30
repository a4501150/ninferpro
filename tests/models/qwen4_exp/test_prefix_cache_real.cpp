// Qwen4Exp prefix cache on the real artifact (design §19.3.1). Skips unless INFERNIX_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2; default int8).
// One Engine per drafter mode serves every scenario, so the 64.5 GiB model loads once per mode:
//
//   tap resume   (E3) A request resumes from another's flexible tap at a prefill-chunk boundary; its
//                greedy output equals a cold run of the same prompt (allow_prefix_reuse = false),
//                which has the same call boundaries.
//   host blocks  (E3 through the Host tier) A pressure prompt evicts the cached blocks; the next
//                resume restores them from the slab pool and still equals the cold run.
//   endpoint     (E1) A second turn resumes from the first turn's endpoint left in its lane (state
//                that turn 1's decode computed) and generates what a cold run of turn 2 generates;
//                after pressure evicts everything Device-side, the same second turn resumes through
//                a Host image restore and must generate the same tokens.
//   X5           Chat turn 2 without turn 1's reasoning block resumes at least at turn 1's generation
//                opener tap, off the 16-token grid, and generates what its cold run generates.
//   X6           Sessions sharing only a system block resume at its structural tap (an exact split).
//   X3           A repeat of a chat prompt resumes at its generation opener (an exact off-grid tap) and
//                generates what the capturing run and a cold run (calls at 0, 512, ...) generate.
//                The prompt-end capture is value-gated there (the opener realized within one tap
//                separation of n), so the opener serves the repeat.
//   The comparisons with cold runs need the artifact's exact route to be construction-exact
//   (recipe C, bf16, int8 or fp8 KV; model-quality.md §6.8); under MTP the drafter's kernels are
//   column-invariant there too, so the resumed run's drafted and accepted counts equal the cold run's.
//   X12          (both modes) Engine 1 serves two chat turns and stops, spilling its Host tier to the
//                Disk tier; Engine 2 rebuilds it and resumes turn 2 from it, generating what Engine 1 did.
//   Disk tier    (both modes, own Engines) The zero-token resume through the Disk tier with the RAM
//                tier on (the stop's spill) and off (write-through, fetched back in the same Engine),
//                across a restart, after a damaged journal tail, with unreadable images (a miss), and
//                with another build identity (emptied); and a small RAM tier that spills under
//                pressure and serves a repeat back from the Disk tier. INFERNIX_TEST_DISK_DIR puts
//                the tier on a given filesystem.
//   one context  (MTP mode, own Engine with kv_capacity = max_context) A recomputed cached prefix
//                replaces the cached copies instead of pinning them beside its pages: the request
//                completes, equals its cold run, and the Engine keeps serving.
//   zero token   (both modes, own Engines) The prompt-end snapshot with its saved head residual: an
//                exact repeat reuses the whole prompt with no prefill (computed prefill 0) and
//                reproduces the capturing run bit-for-bit — greedy and seeded, multi-call shapes
//                with block-aligned and misaligned ends (a span whose deepest published frontier
//                sits within one tap separation of n — a one-call span's tail tap, an opener
//                that realized just below n — skips the capture by value and its case asserts the
//                fallback), a constrained repeat (the draws advance on the resumed sample), the
//                repeat-then-repeat-again case, and the same reuse through the Disk tier.
//                A second Engine at concurrency 2 repeats two graph-warmed prompts in parallel.
//                Under MTP the snapshot is the transition with the drafter's pending cell unwritten,
//                and each repeat's rounds, drafted and accepted counts equal its capturing run's
//                (a wrong restored drafter state — its cells, tails or saved residual — shows only
//                there, never in the tokens; the pending cell itself is rewritten by every decode
//                round's step 0, so no count can see it); the block-aligned end exercises the
//                anchor block published after the first round.
//
//   infernix_qwen4_exp_prefix_cache_real_test [--mtp] [--plain] [--zero | --disk]   (default: both
//   modes; --zero runs only the zero-token and persistence scenarios, --disk only the Disk tier's,
//   in the selected modes)

#include "test_temp_path.h"
#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kChunk = 512;

std::vector<infernix::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<infernix::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

infernix::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Run {
    std::vector<infernix::TokenId> tokens;
    std::uint32_t reused = 0;
    infernix::SpeculativeStats speculative;
};

Run generate(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt, std::uint32_t outputs, bool reuse) {
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), greedy(outputs, reuse));
    return Run{result.generated_token_ids, result.reused_prompt_tokens, result.speculative};
}

// A resumed run drafts as its capturing run did: the drafter's restored cells (and the pending
// cell its first round writes) are the capturing run's, so its rounds, drafts and acceptances are.
bool drafts_equal(const infernix::SpeculativeStats& a, const infernix::SpeculativeStats& b) {
    return a.rounds == b.rounds && a.drafted_tokens == b.drafted_tokens && a.accepted_tokens == b.accepted_tokens;
}

std::string drafts_text(const infernix::SpeculativeStats& a, const infernix::SpeculativeStats& b) {
    const auto one = [](const infernix::SpeculativeStats& s) {
        return std::to_string(s.rounds) + "/" + std::to_string(s.drafted_tokens) + "/" + std::to_string(s.accepted_tokens);
    };
    return "rounds/drafted/accepted " + one(a) + " vs " + one(b);
}

const char* mode_name(bool mtp) { return mtp ? "MTP" : "plain"; }

// The MTP drafter as the mode runs it (3 drafts).
void use_mode(infernix::EngineOptions& options, bool mtp) {
    if (!mtp) { return; }
    options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// A resume against a cold run: the target model and the MTP drafter are construction-exact, so the
// resumed run drafts and accepts what the cold run does.
int check_drafts(const char* what, const infernix::SpeculativeStats& resumed, const infernix::SpeculativeStats& cold) {
    return check(drafts_equal(resumed, cold), std::string(what) + ": drafts equal (" + drafts_text(resumed, cold) + ")");
}

std::string first_difference(const std::vector<infernix::TokenId>& a, const std::vector<infernix::TokenId>& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

infernix::EngineOptions base_options(const char* artifact, const char* ngram) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context          = 4096;
    // 72 pages (+1 copy-on-write page): a 3,900-token pressure prompt evicts nearly every block.
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk        = kChunk;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                 : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    // Resumes equal cold runs on the exact route (model-quality.md §6.1); the default route's
    // qualified paths make a column's bits depend on its call. The artifact's exact route is
    // construction-exact (recipe C, bf16, int8 or fp8 KV; model-quality.md §6.8), so it resumes, taps and
    // publishes endpoints at any position itself, and every resume below equals a cold run.
    options.fast_paths           = false;
    return options;
}

std::string rules_text() {
    std::string rules;
    for (int r = 1; r <= 110; ++r) {
        rules += "Rule " + std::to_string(r) + ": items of colour " + std::to_string(r % 9) + " go to shelf " +
                 std::to_string((r * 7) % 13) + ".\n";
    }
    return rules;
}

// User turns longer than kMinimumTapSeparation (64 tokens): an opener closer to the structural tap
// joins its cluster, and only the earlier tap is kept.
std::string user_context() {
    std::string context;
    for (int i = 0; i < 4; ++i) {
        context += "Today a delivery of mixed items arrived at the warehouse and I need to plan where each "
                   "box goes before the afternoon shift starts. ";
    }
    return context;
}

infernix::ChatMessage message(infernix::ChatRole role, std::string text) {
    infernix::ChatMessage m;
    m.role = role;
    m.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = std::move(text)});
    return m;
}

infernix::GenerationResult chat_on(infernix::Engine& engine, std::vector<infernix::ChatMessage> messages,
                                 std::uint32_t outputs, bool reuse = true) {
    infernix::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    infernix::RequestOptions request      = greedy(outputs, reuse);
    request.stop.include_model_defaults = true;
    return engine.generate(engine.prepare(std::move(input)), request);
}

int run_mode(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== %s\n", mtp ? "MTP drafter (--spec mtp, 3 drafts)" : "plain decode");
    infernix::EngineOptions options = base_options(artifact, ngram);
    if (mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    infernix::Engine engine(std::move(options));
    int failures = 0;

    // tap resume: A leaves a flexible tap at the start of its final call (1024); B shares 1,400
    // tokens, so it resumes at 1024 and prefills 1024..1536..1700 as a cold run does.
    const std::vector<infernix::TokenId> a = synthetic_tokens(1500, 11);
    std::vector<infernix::TokenId> b(a.begin(), a.begin() + 1400);
    const auto suffix = synthetic_tokens(300, 12);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const Run cold_b = generate(engine, b, 48, false);
    (void)generate(engine, a, 16, true);
    const Run warm_b = generate(engine, b, 48, true);
    failures += check(warm_b.reused == 1024, "tap resume reuses 1024 tokens (got " + std::to_string(warm_b.reused) + ")");
    failures += check(warm_b.tokens == cold_b.tokens, "tap resume equals the cold run (" +
                                                          first_difference(warm_b.tokens, cold_b.tokens) + ")");
    failures += check_drafts("tap resume against the cold run", warm_b.speculative, cold_b.speculative);

    // host blocks: pressure evicts the Device copies; B now resumes from its own prompt-tail tap
    // (1536) through Host block restores.
    const std::vector<infernix::TokenId> pressure = synthetic_tokens(3900, 13);
    (void)generate(engine, pressure, 4, true);
    const Run host_b = generate(engine, b, 48, true);
    failures += check(host_b.reused >= 1024, "Host resume reuses " + std::to_string(host_b.reused) + " tokens");
    failures += check(host_b.tokens == cold_b.tokens, "Host block restore equals the cold run (" +
                                                          first_difference(host_b.tokens, cold_b.tokens) + ")");

    // endpoint: turn 2 echoes turn 1 and its output; the lane still holds turn 1's endpoint.
    const std::vector<infernix::TokenId> t1 = synthetic_tokens(700, 14);
    const Run turn1 = generate(engine, t1, 96, true);
    std::vector<infernix::TokenId> t2 = t1;
    t2.insert(t2.end(), turn1.tokens.begin(), turn1.tokens.end());
    const auto more = synthetic_tokens(64, 15);
    t2.insert(t2.end(), more.begin(), more.end());
    const Run resident = generate(engine, t2, 48, true);
    const std::uint32_t endpoint = static_cast<std::uint32_t>(t1.size() + turn1.tokens.size() - 1U);
    failures += check(resident.reused == endpoint, "turn 2 resumes at the endpoint " + std::to_string(endpoint) +
                                                       " (got " + std::to_string(resident.reused) + ")");
    // The endpoint holds decode-computed state (turn 1's output, decoded one round at a time);
    // the cold run prefills those positions in its chunks.
    const Run cold_t2 = generate(engine, t2, 48, false);
    failures += check(resident.tokens == cold_t2.tokens, "the endpoint resume equals the cold run (" +
                                                             first_difference(resident.tokens, cold_t2.tokens) + ")");
    failures += check_drafts("the endpoint resume against the cold run", resident.speculative, cold_t2.speculative);
    // Re-create turn 1's endpoint, evict, then resume turn 2 through the Host image.
    (void)generate(engine, t1, 96, true);
    (void)generate(engine, pressure, 4, true);
    const Run restored = generate(engine, t2, 48, true);
    failures += check(restored.reused == endpoint, "turn 2 resumes at the endpoint after eviction (got " +
                                                       std::to_string(restored.reused) + ")");
    failures += check(restored.tokens == resident.tokens, "Host-restored endpoint equals the lane-resident resume (" +
                                                              first_difference(restored.tokens, resident.tokens) + ")");

    // Chat prompts carry the structural (end of the system block) and generation-opener taps.
    const auto chat = [&](std::vector<infernix::ChatMessage> messages, std::uint32_t outputs) {
        return chat_on(engine, std::move(messages), outputs);
    };
    const infernix::ChatMessage system = message(infernix::ChatRole::System, rules_text());
    const std::string context        = user_context();
    const auto user = [&](const std::string& question) {
        return message(infernix::ChatRole::User, context + question);
    };

    // X5 chat no-echo: turn 2 re-renders turn 1's reply without its reasoning block; it resumes at
    // least at turn 1's generation opener (a few tokens before turn 1's prompt end).
    const infernix::GenerationResult s1 =
        chat({system, user("Which shelf takes colour 4?")}, 32);
    const std::uint32_t n1 = s1.prompt.prompt_tokens;
    const infernix::GenerationResult x5 =
        chat({system, user("Which shelf takes colour 4?"), message(infernix::ChatRole::Assistant, s1.content),
              message(infernix::ChatRole::User, "And colour 5?")},
             32);
    std::printf("        X5: turn 1 prompt %u, turn 2 prompt %u reused %u\n", n1, x5.prompt.prompt_tokens,
                x5.reused_prompt_tokens);
    failures += check(x5.reused_prompt_tokens + 16U >= n1, "X5 turn 2 reuses at least turn 1's opener (reused " +
                                                               std::to_string(x5.reused_prompt_tokens) + ", turn 1 prompt " +
                                                               std::to_string(n1) + ")");
    {
        // The opener is off the 16-token grid; the cold run's calls start at 0, 512, 1024, ...
        const infernix::GenerationResult cold =
            chat_on(engine,
                    {system, user("Which shelf takes colour 4?"), message(infernix::ChatRole::Assistant, s1.content),
                     message(infernix::ChatRole::User, "And colour 5?")},
                    32, false);
        failures += check(x5.generated_token_ids == cold.generated_token_ids,
                          "X5 the off-grid resume equals the cold run (" +
                              first_difference(x5.generated_token_ids, cold.generated_token_ids) + ")");
        failures += check_drafts("X5 against the cold run", x5.speculative, cold.speculative);
    }

    const infernix::RuntimeStats before_s2 = engine.runtime_stats();

    // X6 shared preamble: sessions 2 and 3 share only the system block; they resume at its
    // structural tap, an exact split the first session made.
    const infernix::GenerationResult s2 =
        chat({system, user("Summarise rule 12 in one sentence.")}, 32);
    const infernix::RuntimeStats after_s2 = engine.runtime_stats();
    const infernix::GenerationResult s3 = chat({system, user("How many rules send items to shelf 0?")}, 32);
    std::printf("        X6: sessions 2 and 3 reused %u and %u of %u and %u\n", s2.reused_prompt_tokens,
                s3.reused_prompt_tokens, s2.prompt.prompt_tokens, s3.prompt.prompt_tokens);
    failures += check(s2.reused_prompt_tokens >= 1024 && s2.reused_prompt_tokens + 8U < s2.prompt.prompt_tokens &&
                          s3.reused_prompt_tokens == s2.reused_prompt_tokens,
                      "X6 sessions 2 and 3 resume at the shared structural frontier");

    // X3 split equivalence at an exact off-grid tap: session 2 again resumes at its own generation
    // opener; the run that captured it split its prefill there, so the resume repeats its
    // computation. The prompt-end capture is value-gated in both modes (the opener realized
    // within one tap separation of n, and the image would cost more than the short tail call it
    // saves), so the opener serves the repeat; the capture's counters print as evidence.
    const infernix::GenerationResult again = chat({system, user("Summarise rule 12 in one sentence.")}, 32);
    const infernix::RuntimeStats after_again = engine.runtime_stats();
    std::printf("        X3: prompt-end captures %llu -> %llu -> %llu, failed %llu -> %llu -> %llu\n",
                static_cast<unsigned long long>(before_s2.hybrid_prompt_end_captures),
                static_cast<unsigned long long>(after_s2.hybrid_prompt_end_captures),
                static_cast<unsigned long long>(after_again.hybrid_prompt_end_captures),
                static_cast<unsigned long long>(before_s2.hybrid_prompt_end_failed),
                static_cast<unsigned long long>(after_s2.hybrid_prompt_end_failed),
                static_cast<unsigned long long>(after_again.hybrid_prompt_end_failed));
    failures += check(again.reused_prompt_tokens + 16U >= s2.prompt.prompt_tokens &&
                          again.reused_prompt_tokens < s2.prompt.prompt_tokens,
                      "X3 the repeat resumes at the opener (reused " +
                          std::to_string(again.reused_prompt_tokens) + ")");
    failures += check(again.generated_token_ids == s2.generated_token_ids,
                      "X3 the repeat equals the capturing run (" +
                          first_difference(again.generated_token_ids, s2.generated_token_ids) + ")");
    {
        const infernix::GenerationResult cold = chat_on(engine, {system, user("Summarise rule 12 in one sentence.")}, 32, false);
        failures += check(again.generated_token_ids == cold.generated_token_ids,
                          "X3 the opener resume equals the cold run (" +
                              first_difference(again.generated_token_ids, cold.generated_token_ids) + ")");
        failures += check_drafts("X3 against the cold run", again.speculative, cold.speculative);
    }

    // Single-chunk tail snapshot: a prompt that fits one call has no inner boundary for its
    // opener to realize at, and the finish endpoint serves continuations only, so the tail tap
    // publishes the prompt's own tail (grid 16) and the repeat resumes there instead of from the
    // root. The prompt-end capture is skipped by value (the tap sits within one
    // separation of n, and the image would cost more than the short tail call it saves), so the
    // tap serves the repeat; the deeper reuses below it in the file exercise the capture itself.
    {
        const std::vector<infernix::TokenId> p = synthetic_tokens(400, 26);
        const Run first  = generate(engine, p, 32, true);
        const infernix::RuntimeStats stats = engine.runtime_stats();
        std::printf("        prompt-end captures %llu, failed %llu (Host free %llu of %llu bytes)\n",
                    static_cast<unsigned long long>(stats.hybrid_prompt_end_captures),
                    static_cast<unsigned long long>(stats.hybrid_prompt_end_failed),
                    static_cast<unsigned long long>(stats.hybrid_host_capacity_bytes - stats.hybrid_host_used_bytes),
                    static_cast<unsigned long long>(stats.hybrid_host_capacity_bytes));
        const Run repeat = generate(engine, p, 32, true);
        const std::uint32_t tail = 399U / 16U * 16U;
        failures += check(repeat.reused == tail, "a single-chunk repeat resumes at its tail " +
                                                      std::to_string(tail) + " (got " + std::to_string(repeat.reused) + ")");
        failures += check(repeat.tokens == first.tokens, "the single-chunk repeat equals the first run (" +
                                                            first_difference(repeat.tokens, first.tokens) + ")");
    }
    return failures;
}

// The zero-token resume: a prompt's own end snapshot, taken at its prefill -> decode transition,
// carries the lane's whole state and the head's saved residual. An exact repeat reuses the whole
// prompt (computed prefill 0) and regenerates its first token through the saved residual, so it
// reproduces the capturing run bit-for-bit — greedy and seeded, multi-call shapes, block-aligned
// and misaligned ends, a constrained repeat (the draws advance on the resumed sample), and the
// repeat-then-repeat-again case (the snapshot serves its own repeat). A span whose deepest
// published frontier sits within one tap separation of n (a one-call span's tail tap, an opener
// that realized just below n) skips the capture by value — the image costs more than the short
// tail call a repeat re-fills anyway — and its case below asserts the fallback. Under MTP the
// snapshot holds the drafter's pending cell n - 1 unwritten: the repeat's first decode round
// writes it into a private copy of the tail page (or, at a block-aligned end, of the anchor
// block), so its rounds, drafted and accepted counts equal the capturing run's.
int run_zero_token(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== the zero-token resume (%s, free cache resume)\n", mode_name(mtp));
    infernix::EngineOptions zero = base_options(artifact, ngram);
    use_mode(zero, mtp);
    // The four prompts' images (their prompt-end captures and taps) must all fit beside the cached
    // blocks, and the capture's fresh tail page needs a free pool page: refusals of either would
    // make the reuse checks measure capacity, not the zero-token path.
    zero.context_cache.host_capacity_bytes = 8ULL << 30;
    zero.kv_capacity                        = infernix::KvCapacityPolicy::explicit_capacity(9216);
    infernix::Engine engine(zero);
    int failures = 0;
    const auto repeat_case = [&](const char* what, std::size_t tokens, std::uint32_t seed, std::uint32_t outputs) {
        const std::vector<infernix::TokenId> p = synthetic_tokens(tokens, seed);
        const Run first  = generate(engine, p, outputs, true);
        const Run repeat = generate(engine, p, outputs, true);
        const auto n = static_cast<std::uint32_t>(tokens);
        failures += check(repeat.reused == n, std::string(what) + ": the repeat reuses the whole prompt (got " +
                                                 std::to_string(repeat.reused) + ")");
        failures += check(repeat.tokens == first.tokens, std::string(what) + ": the repeat equals the capturing run (" +
                                                           first_difference(repeat.tokens, first.tokens) + ")");
        if (mtp) {
            failures += check(first.speculative.drafted_tokens > 0 && drafts_equal(repeat.speculative, first.speculative),
                              std::string(what) + ": the repeat drafts as the capturing run (" +
                                  drafts_text(repeat.speculative, first.speculative) + ")");
        }
        const Run third = generate(engine, p, outputs, true);
        failures += check(third.reused == n && third.tokens == first.tokens,
                          std::string(what) + ": the repeat's repeat resumes the same snapshot");
        if (mtp) {
            failures += check(drafts_equal(third.speculative, first.speculative),
                              std::string(what) + ": the repeat's repeat drafts as the capturing run (" +
                                  drafts_text(third.speculative, first.speculative) + ")");
        }
    };
    repeat_case("multi-call, misaligned end", 1500, 31, 48);
    repeat_case("multi-call, block-aligned end", 1536, 32, 48);

    // A one-call span's tail tap sits within one separation of n, so the prompt-end capture is
    // skipped by value and the repeat resumes at the tail tap, re-filling its short tail call.
    {
        const std::vector<infernix::TokenId> p = synthetic_tokens(400, 33);
        const Run first  = generate(engine, p, 32, true);
        const Run repeat = generate(engine, p, 32, true);
        const std::uint32_t tail = 399U / 16U * 16U;
        failures += check(repeat.reused == tail, "one-call end: the repeat resumes at the tail tap (got " +
                                                      std::to_string(repeat.reused) + ")");
        failures += check(repeat.tokens == first.tokens, "one-call end: the repeat equals the first run (" +
                                                            first_difference(repeat.tokens, first.tokens) + ")");
    }

    // Seeded sampling: the resumed draw keys by the same logical position, so it reproduces the
    // cold draw bit-for-bit.
    {
        const std::vector<infernix::TokenId> p = synthetic_tokens(900, 34);
        infernix::RequestOptions seeded = greedy(48, true);
        seeded.execution.sampling.temperature = 0.8F;
        seeded.execution.sampling.seed       = 1234U;
        const auto run = [&] {
            return engine.generate(engine.prepare_tokens(p), seeded);
        };
        const infernix::GenerationResult first  = run();
        const infernix::GenerationResult repeat = run();
        const infernix::RuntimeStats seeded_stats = engine.runtime_stats();
        std::printf("        seeded: reused %u (prompt-end captures %llu, failed %llu)\n",
                    repeat.reused_prompt_tokens,
                    static_cast<unsigned long long>(seeded_stats.hybrid_prompt_end_captures),
                    static_cast<unsigned long long>(seeded_stats.hybrid_prompt_end_failed));
        failures += check(repeat.reused_prompt_tokens == 900, "the seeded repeat reuses the whole prompt (got " +
                                                                 std::to_string(repeat.reused_prompt_tokens) + ")");
        failures += check(repeat.generated_token_ids == first.generated_token_ids,
                          "the seeded repeat equals the capturing run (" +
                              first_difference(repeat.generated_token_ids, first.generated_token_ids) + ")");
        if (mtp) {
            failures += check(drafts_equal(repeat.speculative, first.speculative),
                              "the seeded repeat drafts as the capturing run (" +
                                  drafts_text(repeat.speculative, first.speculative) + ")");
        }
    }

    return failures;
}

// The Disk tier's directory: INFERNIX_TEST_DISK_DIR puts it on a given filesystem (the SSD the
// tier is meant for); otherwise it lives in the system temp directory.
std::filesystem::path disk_directory(const char* name) {
    const char* root = std::getenv("INFERNIX_TEST_DISK_DIR");
    if (root != nullptr && *root != '\0') {
        return std::filesystem::path(root) / infernix::test::unique_temp_path(name).filename();
    }
    return infernix::test::unique_temp_path(name);
}

void use_disk(infernix::EngineOptions& options, const std::filesystem::path& directory, const char* identity) {
    options.context_cache.hybrid.disk_directory      = directory;
    options.context_cache.hybrid.disk_capacity_bytes = 4ULL << 30;
    options.context_cache.hybrid.persistent_identity = identity;
}

// Zeroes the slot file's first GiB: the index takes the lowest free slots, so every image these
// scenarios wrote stops matching its journal record (and every block loses its bytes).
void zero_used_slots(const std::filesystem::path& directory) {
    std::fstream slots(directory / "slots.bin", std::ios::binary | std::ios::in | std::ios::out);
    const std::uint64_t size = std::min<std::uint64_t>(std::filesystem::file_size(directory / "slots.bin"), 1ULL << 30);
    const std::vector<char> zeros(1U << 20, 0);
    for (std::uint64_t at = 0; at < size; at += zeros.size()) {
        slots.write(zeros.data(), static_cast<std::streamsize>(std::min<std::uint64_t>(zeros.size(), size - at)));
    }
}

// The same whole-prompt reuse through the Disk tier, in both data paths: with the RAM tier, the
// snapshot reaches the Disk at the Engine's stop (the spill); without it (host 0), every capture
// is written through, so a repeat in the same Engine already reads its image from the Disk. A
// fresh Engine rebuilds the entries from the journal and the repeat regenerates its first token
// with no prefill at all (a tap-free multi-call prompt; under MTP the prompt ends on a block
// boundary, so the anchor is the block published after the first round). Then the tier's
// failures give misses, not errors: a damaged journal tail keeps the records before it, images
// whose header no longer matches their record are not resumed from, and another build identity
// empties the tier. Its own engines, one alive at a time (a second boot beside a live Engine does
// not fit the model's experts in VRAM).
int run_zero_token_disk(const char* artifact, const char* ngram, bool mtp, bool host) {
    std::printf("== the zero-token resume through the Disk tier (%s, RAM tier %s)\n", mode_name(mtp),
                host ? "on" : "off");
    const std::filesystem::path directory = disk_directory(host ? "qwen4_exp_zero_disk_ram" : "qwen4_exp_zero_disk");
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    infernix::EngineOptions zero = base_options(artifact, ngram);
    use_mode(zero, mtp);
    zero.context_cache.host_capacity_bytes = host ? 8ULL << 30 : 0U;
    zero.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(9216);
    use_disk(zero, directory, "zero");
    const std::size_t tokens               = mtp ? 1536 : 1500;
    const std::vector<infernix::TokenId> p = synthetic_tokens(tokens, 38);
    Run saved_run;
    int failures = 0;
    const auto zero_token = [&](infernix::Engine& engine, const std::string& what) {
        const infernix::GenerationResult repeat = engine.generate(engine.prepare_tokens(p), greedy(48, true));
        std::printf("        %s: reused %u, computed %u\n", what.c_str(), repeat.reused_prompt_tokens,
                    repeat.computed_prefill_tokens);
        failures += check(repeat.reused_prompt_tokens == tokens && repeat.computed_prefill_tokens == 0,
                          what + ": a zero-token resume (reused " + std::to_string(repeat.reused_prompt_tokens) +
                              ", computed " + std::to_string(repeat.computed_prefill_tokens) + ")");
        failures += check(repeat.generated_token_ids == saved_run.tokens,
                          what + ": equals the capturing run (" +
                              first_difference(repeat.generated_token_ids, saved_run.tokens) + ")");
        if (mtp) {
            failures += check(drafts_equal(repeat.speculative, saved_run.speculative),
                              what + ": drafts as the capturing run (" +
                                  drafts_text(repeat.speculative, saved_run.speculative) + ")");
        }
    };
    {
        infernix::Engine engine(zero);
        const infernix::LoadSummary::PrefixCacheDisk disk = engine.load_summary().prefix_disk;
        std::printf("        Disk tier: cuFile %s (%s)\n", disk.mode.c_str(), disk.reset_reason.c_str());
        failures += check(disk.attached && !disk.reset_reason.empty() && disk.snapshots == 0,
                          "a new Disk tier starts empty");
        saved_run = generate(engine, p, 48, true);
        failures += check(saved_run.reused == 0, "the capturing run is cold");
        if (!host) {
            zero_token(engine, "written through, the repeat");
            const infernix::RuntimeStats stats = engine.runtime_stats();
            std::printf("        Disk writes %llu images, %llu blocks; fetches %llu, %llu bytes read\n",
                        static_cast<unsigned long long>(stats.hybrid_disk_image_writes),
                        static_cast<unsigned long long>(stats.hybrid_disk_block_writes),
                        static_cast<unsigned long long>(stats.hybrid_disk_fetches),
                        static_cast<unsigned long long>(stats.hybrid_disk_read_bytes));
            failures += check(stats.hybrid_disk_image_writes > 0 && stats.hybrid_disk_fetches > 0 &&
                                  stats.hybrid_disk_failures == 0,
                              "without the RAM tier the snapshot is written through and fetched back");
        }
    } // the stop spills the Host tier's snapshots (RAM tier on)
    {
        infernix::Engine engine(zero);
        const infernix::LoadSummary::PrefixCacheDisk disk = engine.load_summary().prefix_disk;
        std::printf("        rebuilt %llu snapshots and %llu blocks in %.3f s\n",
                    static_cast<unsigned long long>(disk.snapshots), static_cast<unsigned long long>(disk.blocks),
                    disk.seconds);
        failures += check(disk.attached && disk.reset_reason.empty() && disk.snapshots > 0 && !disk.damaged,
                          "a restarted Engine rebuilds the Disk tier's snapshots");
        zero_token(engine, "after the restart, the repeat");
        // RAM tier on: the fetched snapshot was promoted to the Host tier, which serves this one.
        zero_token(engine, "after the restart, the repeat's repeat");
        failures += check(engine.runtime_stats().hybrid_disk_fetches > 0, "the restarted Engine read the Disk tier");
    }
    {
        // A torn record at the journal's end: the records before it still rebuild.
        std::ofstream journal(directory / "catalog.log", std::ios::binary | std::ios::app);
        journal.write("\x02\x40\x00\x00\x00torn", 9);
    }
    {
        infernix::Engine engine(zero);
        const infernix::LoadSummary::PrefixCacheDisk disk = engine.load_summary().prefix_disk;
        std::printf("        after a damaged journal tail: rebuilt %llu snapshots and %llu blocks\n",
                    static_cast<unsigned long long>(disk.snapshots), static_cast<unsigned long long>(disk.blocks));
        failures += check(disk.damaged && disk.snapshots > 0, "a damaged journal tail keeps the records before it");
        zero_token(engine, "after a damaged journal tail, the repeat");
    }
    zero_used_slots(directory);
    {
        infernix::Engine engine(zero);
        const Run cold = generate(engine, p, 48, true);
        const infernix::RuntimeStats stats = engine.runtime_stats();
        std::printf("        damaged images: reused %u, Disk failures %llu\n", cold.reused,
                    static_cast<unsigned long long>(stats.hybrid_disk_failures));
        failures += check(cold.reused == 0 && cold.tokens == saved_run.tokens && stats.hybrid_disk_failures > 0,
                          "an image whose header no longer matches its record is not resumed from (" +
                              first_difference(cold.tokens, saved_run.tokens) + ")");
    }
    {
        infernix::EngineOptions other = zero;
        other.context_cache.hybrid.persistent_identity = "another build";
        infernix::Engine engine(other);
        const infernix::LoadSummary::PrefixCacheDisk disk = engine.load_summary().prefix_disk;
        failures += check(!disk.reset_reason.empty() && disk.snapshots == 0 && disk.blocks == 0,
                          "another build identity empties the Disk tier (" + disk.reset_reason + ")");
    }
    std::filesystem::remove_all(directory, ignored);
    return failures;
}

// The RAM tier under pressure (room for three snapshot images): each new prompt's snapshots evict
// the coldest Host entries, which were spilled to the Disk tier while the requests ran, so a
// repeat of the first prompt still resumes with no prefill, reading its snapshot back from the Disk.
int run_disk_pressure(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== the RAM tier spilling to the Disk tier under pressure (%s)\n", mode_name(mtp));
    const std::filesystem::path directory = disk_directory("qwen4_exp_disk_pressure");
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    infernix::EngineOptions options = base_options(artifact, ngram);
    use_mode(options, mtp);
    options.context_cache.host_capacity_bytes = 400ULL << 20;
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(9216);
    use_disk(options, directory, "pressure");
    int failures = 0;
    {
        infernix::Engine engine(options);
        std::vector<Run> first;
        for (std::uint32_t i = 0; i < 4; ++i) { first.push_back(generate(engine, synthetic_tokens(1500, 50 + i), 48, true)); }
        const infernix::RuntimeStats spilled = engine.runtime_stats();
        std::printf("        Disk writes %llu images, %llu blocks; Host evictions %llu\n",
                    static_cast<unsigned long long>(spilled.hybrid_disk_image_writes),
                    static_cast<unsigned long long>(spilled.hybrid_disk_block_writes),
                    static_cast<unsigned long long>(spilled.hybrid_host_snapshot_evictions));
        failures += check(spilled.hybrid_disk_image_writes > 0 && spilled.hybrid_host_snapshot_evictions > 0,
                          "the RAM tier spilled snapshots before evicting them");
        const Run repeat = generate(engine, synthetic_tokens(1500, 50), 48, true);
        const infernix::RuntimeStats read = engine.runtime_stats();
        std::printf("        repeat of the first prompt: reused %u; fetches %llu\n", repeat.reused,
                    static_cast<unsigned long long>(read.hybrid_disk_fetches - spilled.hybrid_disk_fetches));
        failures += check(repeat.reused == 1500 && read.hybrid_disk_fetches > spilled.hybrid_disk_fetches,
                          "the first prompt's spilled snapshot serves its repeat from the Disk tier");
        failures += check(repeat.tokens == first[0].tokens,
                          "the repeat from the Disk tier equals the capturing run (" +
                              first_difference(repeat.tokens, first[0].tokens) + ")");
        if (mtp) {
            failures += check(drafts_equal(repeat.speculative, first[0].speculative),
                              "the repeat from the Disk tier drafts as the capturing run (" +
                                  drafts_text(repeat.speculative, first[0].speculative) + ")");
        }
    }
    std::filesystem::remove_all(directory, ignored);
    return failures;
}

// A constrained repeat: the grammar mask applies to the resumed sample and the constraint's draws
// advance on it, one per licensed token. Its own engine (one model fits the GPU): the shared one's
// pool is full of the earlier scenarios' cached blocks by now, and the prompt-end capture
// legitimately skips for a free page (counted), which would not leave the zero-token path here.
int run_zero_token_constrained(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== the zero-token resume, constrained (%s, free cache resume)\n", mode_name(mtp));
    infernix::EngineOptions options = base_options(artifact, ngram);
    use_mode(options, mtp);
    infernix::Engine engine(std::move(options));
    int failures = 0;
    const std::vector<infernix::TokenId> p = synthetic_tokens(800, 35);
    infernix::RequestOptions constrained = greedy(48, true);
    constrained.stop.include_model_defaults = true; // constraints require the default EOS
    constrained.constraint = infernix::OutputConstraint::grammar("root ::= \"yes\" | \"no\"");
    const auto run = [&] {
        return engine.generate(engine.prepare_tokens(p), constrained);
    };
    const infernix::GenerationResult first  = run();
    const infernix::RuntimeStats between = engine.runtime_stats();
    std::printf("        first: %zu tokens, %zu draws, reused %u (prompt-end captures %llu, failed %llu)\n",
                first.generated_token_ids.size(), first.constrained_draws.size(), first.reused_prompt_tokens,
                static_cast<unsigned long long>(between.hybrid_prompt_end_captures),
                static_cast<unsigned long long>(between.hybrid_prompt_end_failed));
    const infernix::GenerationResult repeat = run();
    std::printf("        repeat: %zu tokens, %zu draws, reused %u\n", repeat.generated_token_ids.size(),
                repeat.constrained_draws.size(), repeat.reused_prompt_tokens);
    failures += check(repeat.reused_prompt_tokens == 800, "the constrained repeat reuses the whole prompt (got " +
                                                         std::to_string(repeat.reused_prompt_tokens) + ")");
    failures += check(repeat.generated_token_ids == first.generated_token_ids,
                      "the constrained repeat equals the capturing run (" +
                          first_difference(repeat.generated_token_ids, first.generated_token_ids) + ")");
    // The grammar's mask visibly governed both samples: a licensed two-word answer, then the
    // model's own stop, in both runs (the draw records are the decide API's surface).
    failures += check(first.finish_reason == infernix::FinishReason::StopToken &&
                          first.generated_token_ids.size() <= 4,
                      "the constraint licenses the resumed sample's short answer");
    return failures;
}

// The zero-token resume at concurrency: two threads repeat two graph-warmed prompts on one Engine
// while a third warms its decode graphs, and both repeats equal their capturing runs.
int run_zero_token_concurrency(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== the zero-token resume at concurrency 2 (%s, free cache resume)\n", mode_name(mtp));
    infernix::EngineOptions options = base_options(artifact, ngram);
    use_mode(options, mtp);
    options.max_concurrency         = 2;
    options.max_pending_requests    = 2;
    infernix::Engine engine(std::move(options));
    int failures = 0;
    const std::vector<infernix::TokenId> a = synthetic_tokens(1200, 36);
    const std::vector<infernix::TokenId> b = synthetic_tokens(1000, 37);
    const Run cold_a = generate(engine, a, 64, true);
    const Run cold_b = generate(engine, b, 64, true);
    std::vector<infernix::TokenId> repeat_a, repeat_b;
    std::uint32_t reused_a = 0, reused_b = 0;
    std::thread thread_a([&] {
        const Run r = generate(engine, a, 64, true);
        repeat_a    = r.tokens;
        reused_a    = r.reused;
    });
    std::thread thread_b([&] {
        const Run r = generate(engine, b, 64, true);
        repeat_b    = r.tokens;
        reused_b    = r.reused;
    });
    thread_a.join();
    thread_b.join();
    failures += check(reused_a == 1200 && reused_b == 1000,
                      "both concurrent repeats reuse their whole prompts (got " + std::to_string(reused_a) +
                          " and " + std::to_string(reused_b) + ")");
    failures += check(repeat_a == cold_a.tokens, "the concurrent repeat of A equals its capturing run (" +
                                                     first_difference(repeat_a, cold_a.tokens) + ")");
    failures += check(repeat_b == cold_b.tokens, "the concurrent repeat of B equals its capturing run (" +
                                                     first_difference(repeat_b, cold_b.tokens) + ")");
    return failures;
}

// One context of KV (issue #1: --kv-capacity == --max-context): B shares A's first 2,000 tokens but
// no snapshot on that path, so it recomputes them from the root while the admission eviction leaves
// A's shallowest blocks cached. Each recomputed block must replace its unpinned cached copy: pinning
// the copy beside B's page fills the pool, and B's first decode page past its admission then has
// no source (the capacity contract error, formerly fatal for the Engine).
int run_one_context(const char* artifact, const char* ngram) {
    std::printf("== one context of KV (MTP drafter, kv_capacity = max_context)\n");
    infernix::EngineOptions options = base_options(artifact, ngram);
    options.kv_capacity               = infernix::KvCapacityPolicy::explicit_capacity(4096);
    options.speculative.backend       = infernix::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    infernix::Engine engine(std::move(options));
    int failures = 0;
    const std::vector<infernix::TokenId> a = synthetic_tokens(3000, 21);
    std::vector<infernix::TokenId> b(a.begin(), a.begin() + 2000);
    const auto suffix = synthetic_tokens(1300, 22);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const Run cold_b = generate(engine, b, 700, false);
    (void)generate(engine, a, 16, true);
    const infernix::RuntimeStats before = engine.runtime_stats();
    Run warm_b;
    try {
        warm_b = generate(engine, b, 700, true);
    } catch (const std::exception& error) {
        return check(false, std::string("B completes in one context of KV (") + error.what() + ")");
    }
    const infernix::RuntimeStats after = engine.runtime_stats();
    std::printf("        B reused %u tokens; blocks reattached %llu, duplicate %llu\n", warm_b.reused,
                static_cast<unsigned long long>(after.hybrid_blocks_reattached - before.hybrid_blocks_reattached),
                static_cast<unsigned long long>(after.hybrid_blocks_duplicate - before.hybrid_blocks_duplicate));
    failures += check(warm_b.reused < 2000 && after.hybrid_blocks_reattached > before.hybrid_blocks_reattached,
                      "B recomputes cached blocks and replaces their copies");
    failures += check(warm_b.tokens == cold_b.tokens,
                      "B equals the cold run (" + first_difference(warm_b.tokens, cold_b.tokens) + ")");
    const Run later = generate(engine, synthetic_tokens(200, 23), 8, true);
    failures += check(later.tokens.size() == 8, "the Engine keeps serving");
    return failures;
}

// X12: the Host tier spilled to the Disk tier at an Engine's stop and rebuilt by the next Engine.
int run_persistence(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== persistence (X12, %s)\n", mode_name(mtp));
    const std::filesystem::path directory = disk_directory("qwen4_exp_x12");
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    const auto options = [&] {
        infernix::EngineOptions o = base_options(artifact, ngram);
        use_mode(o, mtp);
        use_disk(o, directory, "x12");
        return o;
    };
    const infernix::ChatMessage system = message(infernix::ChatRole::System, rules_text());
    const std::string context        = user_context();
    const std::vector<infernix::ChatMessage> turn1{system, message(infernix::ChatRole::User, context + "Which shelf takes colour 2?")};
    int failures = 0;
    std::vector<infernix::ChatMessage> turn2 = turn1;
    std::vector<infernix::TokenId> turn1_tokens;
    infernix::GenerationResult before;
    {
        infernix::Engine engine(options());
        failures += check(engine.load_summary().prefix_disk.snapshots == 0, "Engine 1 starts with an empty Disk tier");
        const infernix::GenerationResult r1 = chat_on(engine, turn1, 32);
        turn1_tokens = r1.generated_token_ids;
        turn2.push_back(message(infernix::ChatRole::Assistant, r1.content));
        turn2.push_back(message(infernix::ChatRole::User, context + "And colour 3?"));
        before = chat_on(engine, turn2, 32);
        // The same turn 2 again, in-process: its opener realized within one tap separation of n,
        // so the prompt-end capture is value-gated and the repeat resumes at the opener,
        // re-filling its 7-token tail call. It must still equal `before` (the exact route is
        // request-invariant); the gated capture's own zero-token path runs in the suite's own
        // from-Disk scenario below, on a prompt with no nearby tap.
        const infernix::GenerationResult before2 = chat_on(engine, turn2, 32);
        std::printf("        before2: reused %u, computed %u, %zu tokens (%s)\n",
                    before2.reused_prompt_tokens, before2.computed_prefill_tokens,
                    before2.generated_token_ids.size(),
                    before2.generated_token_ids == before.generated_token_ids ? "equals before" : "DIFFERS");
        failures += check(before2.computed_prefill_tokens <= 16U &&
                              before2.generated_token_ids == before.generated_token_ids,
                          "the repeat of a value-gated turn resumes at its opener (" +
                              first_difference(before2.generated_token_ids, before.generated_token_ids) + ")");
    } // the stop spills the Host tier
    {
        infernix::Engine engine(options());
        const infernix::LoadSummary::PrefixCacheDisk restore = engine.load_summary().prefix_disk;
        std::printf("        rebuilt %llu blocks and %llu snapshots\n",
                    static_cast<unsigned long long>(restore.blocks), static_cast<unsigned long long>(restore.snapshots));
        failures += check(restore.reset_reason.empty() && restore.snapshots > 0,
                          "Engine 2 rebuilds the spilled snapshots");
        // Turn 2 again: Engine 1 left a snapshot at its own generation opener, so the restarted
        // Engine resumes there and repeats Engine 1's computation of turn 2.
        const infernix::GenerationResult after = chat_on(engine, turn2, 32);
        failures += check(after.reused_prompt_tokens + 16U >= before.prompt.prompt_tokens,
                          "after the restart turn 2 resumes at its opener (reused " +
                              std::to_string(after.reused_prompt_tokens) + " of " +
                              std::to_string(after.prompt.prompt_tokens) + ")");
        std::printf("        before: reused %u of %u, %zu tokens (preemptions %llu, replayed %llu); "
                    "after: reused %u, computed %u, %zu tokens\n",
                    before.reused_prompt_tokens, before.prompt.prompt_tokens, before.generated_token_ids.size(),
                    static_cast<unsigned long long>(before.scheduling.preemptions),
                    static_cast<unsigned long long>(before.scheduling.replayed_tokens),
                    after.reused_prompt_tokens, after.computed_prefill_tokens, after.generated_token_ids.size());
        // A cold turn 2 on the restarted engine: whichever of the resumed runs differs from it is
        // the divergent one (the exact route is request-invariant, so all three must agree).
        const infernix::GenerationResult cold = chat_on(engine, turn2, 32, false);
        std::printf("        cold: reused %u, %zu tokens, preemptions %llu (before %s at %s; after %s at %s)\n",
                    cold.reused_prompt_tokens, cold.generated_token_ids.size(),
                    static_cast<unsigned long long>(cold.scheduling.preemptions),
                    cold.generated_token_ids == before.generated_token_ids ? "equals" : "DIFFERS",
                    first_difference(cold.generated_token_ids, before.generated_token_ids).c_str(),
                    cold.generated_token_ids == after.generated_token_ids ? "equals" : "DIFFERS",
                    first_difference(cold.generated_token_ids, after.generated_token_ids).c_str());
        if (after.generated_token_ids != before.generated_token_ids) {
            std::printf("        before tail:");
            for (std::size_t i = 24; i < before.generated_token_ids.size(); ++i) {
                std::printf(" %d", before.generated_token_ids[i]);
            }
            std::printf("\n        after  tail:");
            for (std::size_t i = 24; i < after.generated_token_ids.size(); ++i) {
                std::printf(" %d", after.generated_token_ids[i]);
            }
            std::printf("\n");
        }
        failures += check(after.generated_token_ids == before.generated_token_ids,
                          "the restored resume equals Engine 1's turn 2 (" +
                              first_difference(after.generated_token_ids, before.generated_token_ids) + ")");
        // The same restored opener serves the same repeat again: the from-Disk path is
        // deterministic (a second divergence would be nondeterminism, not a state defect).
        const infernix::GenerationResult after2 = chat_on(engine, turn2, 32);
        failures += check(after2.computed_prefill_tokens <= 16U &&
                              after2.generated_token_ids == after.generated_token_ids,
                          "the restored opener resume repeats itself (" +
                              first_difference(after2.generated_token_ids, after.generated_token_ids) + ")");
    }
    {
        // Turn 1's opener realized within one separation of its end, so its prompt-end capture
        // is value-gated: a from-Disk repeat of turn 1 alone resumes at the opener and re-fills
        // the 7-token cut. Engine 3 re-runs turn 1 (its transition state is r1's bit for bit)
        // and its stop saves the retained snapshots; Engine 4's repeat must still regenerate
        // r1's tokens exactly — the zero-token path itself runs on a tap-free prompt in the
        // zero-token suite's own from-Disk scenario.
        {
            infernix::Engine engine(options());
            const infernix::GenerationResult warm = chat_on(engine, turn1, 32);
            failures += check(warm.generated_token_ids == turn1_tokens,
                              "turn 1's re-run equals its cold run (" +
                                  first_difference(warm.generated_token_ids, turn1_tokens) + ")");
        } // the stop spills turn 1's retained snapshots
        infernix::Engine engine(options());
        const infernix::GenerationResult repeat = chat_on(engine, turn1, 32);
        std::printf("        turn 1 from the Disk tier: reused %u of %u, computed %u, %zu tokens\n",
                    repeat.reused_prompt_tokens, repeat.prompt.prompt_tokens, repeat.computed_prefill_tokens,
                    repeat.generated_token_ids.size());
        failures += check(repeat.computed_prefill_tokens <= 16U,
                          "turn 1's from-Disk repeat resumes at its opener (computed " +
                              std::to_string(repeat.computed_prefill_tokens) + ")");
        failures += check(repeat.generated_token_ids == turn1_tokens,
                          "turn 1's from-Disk repeat equals its cold run (" +
                              first_difference(repeat.generated_token_ids, turn1_tokens) + ")");
    }
    std::filesystem::remove_all(directory, ignored);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    bool plain = true, mtp = true, zero_only = false, disk_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mtp") { plain = false; }
        if (arg == "--plain") { mtp = false; }
        if (arg == "--zero") { zero_only = true; }
        if (arg == "--disk") { disk_only = true; }
    }
    try {
        int failures = 0;
        if (zero_only) {
            for (const bool mode : {false, true}) {
                if (mode ? !mtp : !plain) { continue; }
                failures += run_zero_token(artifact, ngram, mode);
                failures += run_zero_token_disk(artifact, ngram, mode, true);
                failures += run_zero_token_disk(artifact, ngram, mode, false);
                failures += run_zero_token_constrained(artifact, ngram, mode);
                failures += run_zero_token_concurrency(artifact, ngram, mode);
                failures += run_persistence(artifact, ngram, mode);
            }
            std::printf(failures == 0 ? "qwen4_exp zero-token checks passed\n" : "FAIL: %d checks failed\n",
                        failures);
            return failures == 0 ? 0 : 1;
        }
        if (disk_only) {
            for (const bool mode : {false, true}) {
                if (mode ? !mtp : !plain) { continue; }
                failures += run_zero_token_disk(artifact, ngram, mode, false);
                failures += run_zero_token_disk(artifact, ngram, mode, true);
                failures += run_disk_pressure(artifact, ngram, mode);
                failures += run_persistence(artifact, ngram, mode);
            }
            std::printf(failures == 0 ? "qwen4_exp Disk tier checks passed\n" : "FAIL: %d checks failed\n", failures);
            return failures == 0 ? 0 : 1;
        }
        if (plain) { failures += run_mode(artifact, ngram, false); }
        if (mtp) { failures += run_mode(artifact, ngram, true); }
        if (mtp) { failures += run_one_context(artifact, ngram); }
        for (const bool mode : {false, true}) {
            if (mode ? !mtp : !plain) { continue; }
            failures += run_zero_token(artifact, ngram, mode);
            failures += run_zero_token_disk(artifact, ngram, mode, true);
            failures += run_zero_token_disk(artifact, ngram, mode, false);
            failures += run_disk_pressure(artifact, ngram, mode);
            failures += run_zero_token_constrained(artifact, ngram, mode);
            failures += run_zero_token_concurrency(artifact, ngram, mode);
            failures += run_persistence(artifact, ngram, mode);
        }
        std::printf(failures == 0 ? "qwen4_exp prefix cache checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
