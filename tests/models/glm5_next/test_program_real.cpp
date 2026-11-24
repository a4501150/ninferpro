// The Glm5Next Program on the real converted artifact. INFERNIX_GLM_ARTIFACT names the artifact
// and INFERNIX_GLM_REFERENCE the forward test's reference directory (fp64/ and bf16/ runs of one
// prompt, tools/glm5_next/reference.py --out DIR); without both the test skips (77).
//
// The banks stay in the artifact: part of the experts take VRAM frames, part the RAM tier's slots,
// and the rest are read from the artifact when routed, so every call crosses the three tiers.
//
// - Generation: the prompt prefills in several calls, then greedy decode; the first token is the
//   oracle's argmax (a margin of 0.7 logits there).
// - Two lanes with the same prompt decode in one batch: their rows produce the same tokens.
// - Causal scoring of the prompt against the oracle's log-probabilities, within the floor the
//   rounded reference sets.
// - Teacher forcing: scoring the prompt with the generated tokens gives each generated token as the
//   argmax wherever the top two are not a near-tie, so decode and prefill agree on the state.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/glm5_next/execution/parameters.h"
#include "models/glm5_next/frontend/frontend.h"
#include "models/glm5_next/load.h"
#include "models/glm5_next/program/program.h"
#include "reference_files.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
namespace glm = infernix::models::glm5_next;

using glm_test::argmax;
using glm_test::log_probability;
using glm_test::read_floats;
using glm_test::read_tokens;

namespace {

int failures = 0;

void expect(bool ok, const std::string& message) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

DiagnosticObserver printer() {
    DiagnosticObserver observer;
    observer.callback = [](const Diagnostic& d) {
        if (d.level >= DiagnosticLevel::Info) { std::printf("  [program] %s\n", d.message.c_str()); }
    };
    return observer;
}

runtime::ResolvedExecutionOptions greedy(std::uint32_t tokens) {
    runtime::ResolvedExecutionOptions out;
    out.requested_output_tokens = tokens;
    out.allow_prefix_reuse      = false;
    out.sampling.temperature    = 0.0F;
    return out;
}

// Binds the prompt on `lane` and prefills it; returns the sequence and its first token (committed).
glm::SequenceHandle start(glm::Program& program, const glm::Frontend& frontend, const std::vector<std::int32_t>& prompt,
                          std::uint32_t lane, std::uint32_t output, std::int32_t& first, std::uint32_t& calls) {
    glm::RequestBasePlan base = program.plan_request(frontend.prepare_tokens(prompt, false), greedy(output));
    const auto sources        = program.hybrid_sources(base, UINT32_MAX);
    if (!program.start_binding(base, runtime::LaneId{lane}, sources.front()).reserved) {
        throw std::runtime_error("binding refused");
    }
    const glm::ContextProgress bound = program.poll_context({});
    if (!bound.sequence) { throw std::runtime_error("binding did not complete"); }
    calls = 0;
    for (;;) {
        glm::PrefillProgress progress = program.advance_prefill(*bound.sequence);
        ++calls;
        if (!progress.complete) { continue; }
        first                                  = progress.pending->tokens()[0];
        const runtime::CommitDecision commit[] = {{.accepted_tokens = 1}};
        (void)program.commit(std::move(*progress.pending), commit);
        return *bound.sequence;
    }
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
        const auto prompt     = read_tokens(dir + "/fp64/tokens.txt");
        const auto T          = prompt.size();

        DeviceContext device(0);
        artifact::Reader reader(artifact);
        models::LoadOptions load;
        load.stream_experts = true;
        auto model          = glm::materialize_model(glm::plan_load(reader, load), device);
        device.synchronize();
        const glm::execution::Parameters parameters(*model);
        const auto& c      = model->config().text;
        const std::size_t V = c.vocab_size;
        const auto frontend = glm::make_frontend(model->resources(), glm::FrontendOptions{});
        const auto ref      = read_floats(dir + "/fp64/logits.f32", T * V);
        const auto rounded  = read_floats(dir + "/bf16/logits.f32", T * V);
        const std::size_t domain = model->resources().public_token_count;

        constexpr std::uint32_t kGenerated = 8;
        std::vector<std::int32_t> generated;
        {
            glm::ProgramOptions options;
            options.max_concurrency    = 2;
            options.max_context        = 128;
            options.kv_capacity_tokens = 256;
            options.prefill_chunk      = 6; // the 15-token prompt prefills in calls of 6, 6 and 3
            options.placement.expert_frame_target = 2000;
            options.placement.expert_ram_bytes = 24ULL << 30;
            options.diagnostics        = printer();
            glm::Program program(parameters, device, options);

            // One lane: prefill, then greedy decode.
            std::int32_t first  = 0;
            std::uint32_t calls = 0;
            const glm::SequenceHandle one = start(program, frontend, prompt, 0, kGenerated, first, calls);
            generated.push_back(first);
            std::printf("prompt of %zu tokens prefilled in %u calls; first token %d (oracle argmax %zu)\n", T, calls,
                        first, argmax(ref.data() + (T - 1) * V, domain));
            expect(calls == 3, "the prompt did not prefill in three calls");
            expect(static_cast<std::size_t>(first) == argmax(ref.data() + (T - 1) * V, domain),
                   "the first token is not the oracle's argmax");
            for (std::uint32_t step = 1; step < kGenerated; ++step) {
                const glm::SequenceHandle rows[]      = {one};
                const runtime::RoundBudget budget[]   = {{kGenerated - step}};
                glm::PendingBatch pending             = program.decode(rows, budget);
                generated.push_back(pending.tokens()[0]);
                const runtime::CommitDecision commit[] = {{.accepted_tokens = 1, .terminal = step + 1 == kGenerated}};
                (void)program.commit(std::move(pending), commit);
            }
            const glm::FinishResult finished = program.finish(one);
            expect(finished.status == runtime::ConsumeStatus::Consumed, "the request did not finish");
            std::printf("generated:");
            for (const auto t : generated) { std::printf(" %d", t); }
            std::printf("\n");
            if (finished.expert_cache) {
                std::printf("expert cache: %llu of %llu routed experts were in VRAM\n",
                            static_cast<unsigned long long>(finished.expert_cache->hits),
                            static_cast<unsigned long long>(finished.expert_cache->routed));
            }

            // Two lanes, one prompt: one batch decodes both rows.
            std::int32_t first_a = 0, first_b = 0;
            const glm::SequenceHandle a = start(program, frontend, prompt, 0, kGenerated, first_a, calls);
            const glm::SequenceHandle b = start(program, frontend, prompt, 1, kGenerated, first_b, calls);
            expect(first_a == first_b && first_a == generated[0], "the two lanes' first tokens differ");
            std::vector<std::int32_t> row_a{first_a}, row_b{first_b};
            for (std::uint32_t step = 1; step < kGenerated; ++step) {
                const glm::SequenceHandle rows[]    = {a, b};
                const runtime::RoundBudget budget[] = {{kGenerated - step}, {kGenerated - step}};
                glm::PendingBatch pending           = program.decode(rows, budget);
                row_a.push_back(pending.tokens()[0]);
                row_b.push_back(pending.tokens()[1]);
                const bool last = step + 1 == kGenerated;
                const runtime::CommitDecision commit[] = {{.accepted_tokens = 1, .terminal = last},
                                                          {.accepted_tokens = 1, .terminal = last}};
                (void)program.commit(std::move(pending), commit);
            }
            (void)program.finish(a);
            (void)program.finish(b);
            expect(row_a == row_b, "two rows of one batch with the same prompt produced different tokens");
            std::size_t same = 0;
            for (std::size_t i = 0; i < kGenerated; ++i) { same += row_a[i] == generated[i]; }
            std::printf("batched decode: rows equal %s; %zu of %u tokens equal the one-lane run\n",
                        row_a == row_b ? "yes" : "no", same, kGenerated);
        }

        // Causal scoring: the prompt against the oracle, then the prompt with the generated tokens.
        {
            glm::ProgramOptions options;
            options.max_concurrency    = 1;
            options.max_context        = 128;
            options.kv_capacity_tokens = 128;
            options.prefill_chunk      = 64;
            options.causal_scoring     = true;
            options.placement.expert_frame_target = 2000;
            options.placement.expert_ram_bytes = 24ULL << 30;
            options.diagnostics        = printer();
            glm::Program program(parameters, device, options);

            ScoreOptions top;
            top.top_k = 2;
            const ScoreResult scored = program.causal_score(frontend.prepare_tokens(prompt, false), 1, top);
            double worst = 0, floor = 0;
            for (std::size_t p = 0; p + 1 < T; ++p) {
                const double exact = log_probability(ref.data() + p * V, domain, prompt[p + 1]);
                const double near  = log_probability(rounded.data() + p * V, domain, prompt[p + 1]);
                worst              = std::max(worst, std::abs(scored.logprobs[p] - exact));
                floor              = std::max(floor, std::abs(near - exact));
            }
            std::printf("scoring: worst |log p - oracle| %.4f (rounded reference %.4f)\n", worst, floor);
            expect(worst < 2 * floor + 0.05, "the scored log-probabilities deviate from the oracle beyond the floor");

            std::vector<std::int32_t> forced = prompt;
            forced.insert(forced.end(), generated.begin(), generated.end());
            const ScoreResult teacher =
                program.causal_score(frontend.prepare_tokens(forced, false), static_cast<std::uint32_t>(T), top);
            for (std::size_t i = 0; i < generated.size(); ++i) {
                const std::int32_t best   = teacher.top_ids[i * 2];
                const double margin       = teacher.top_logprobs[i * 2] - teacher.top_logprobs[i * 2 + 1];
                const bool equal          = best == generated[i];
                std::printf("  generated %zu: %d, teacher-forced argmax %d (margin %.3f)\n", i, generated[i], best, margin);
                expect(equal || margin < 0.25,
                       "decode token " + std::to_string(i) + " is not the teacher-forced argmax at a clear margin");
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
