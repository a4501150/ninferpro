#include "serve/metrics.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace infernix;
using namespace infernix::serve;

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                  = 2;
    options.context_cache.hybrid.device_snapshot_slots = 1;
    options.speculative.backend             = SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens         = 4;
    MemorySummary memory;
    memory.kv_capacity_page_groups = 256;
    RuntimeStats baseline;
    baseline.generated_tokens         = 4;
    baseline.speculative_draft_tokens = 3;
    // The expert cache's counters at warmup, and the same report later: /metrics must show the
    // serving-time deltas, not the absolute counters.
    ExpertPlacementReport placement_baseline;
    placement_baseline.supported            = true;
    placement_baseline.swap_routed           = 10;
    placement_baseline.swap_hits            = 5;
    placement_baseline.swap_promotion_bytes = 96;
    ExpertPlacementReport placement;
    placement.supported            = true;
    placement.swap_routed          = 100;
    placement.swap_hits           = 80;
    placement.swap_promotion_bytes = 4096;
    placement.swap_demotion_bytes = 512;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, memory, baseline, placement_baseline);
    RuntimeStats running = baseline;
    running.generated_tokens += 5;
    running.speculative_draft_tokens += 6;
    running.speculative_accepted_tokens = 3;
    running.speculative_rounds          = 7;
    running.mixed_rounds                = 3;
    running.decode_graph_replays        = 12;
    running.decode_eager_executions     = 2;
    running.decode_graph_captures       = 1;
    running.running_requests            = 1;
    running.prefix_selections           = 2;
    running.hybrid_disk_capacity_bytes  = 64;
    running.hybrid_disk_write_bytes     = 7;
    running.hybrid_host_used_bytes      = 9;
    running.hybrid_prompt_end_gated     = 4;
    running.hybrid_prompt_end_failed    = 1;
    running.host_work.decode_host_ns        = 2'000'000'000;
    running.host_work.decode_device_wait_ns = 4'000'000'000;
    running.host_work.prefill_host_ns       = 1'000'000'000;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, placement, true);
    check(live.find("infernix_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("infernix_spec_decode_draft_tokens_total{backend=\"dflash2\"} 6\n") !=
              std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("infernix_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("infernix_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("infernix_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(metrics.render(running, placement, true) == live, "scrapes must not consume or reset counters");
    check(live.find("infernix_prefix_selections_total 2\n") != std::string::npos &&
              live.find("infernix_device_state_capacity_slots 3\n") != std::string::npos,
          "prefix-cache selections and StateImage capacity must be observable");
    check(live.find("infernix_prefix_cache_disk_capacity_bytes 64\n") != std::string::npos &&
              live.find("infernix_prefix_cache_disk_write_bytes_total 7\n") != std::string::npos,
          "the prefix cache's Disk tier size and writes must be observable");
    check(live.find("infernix_prefix_cache_host_used_bytes 9\n") != std::string::npos &&
              live.find("infernix_prefix_cache_prompt_end_gated_total 4\n") != std::string::npos &&
              live.find("infernix_prefix_cache_prompt_end_failed_total 1\n") != std::string::npos,
          "the Host tier's use and the prompt-end gate and failures must be observable apart");
    check(live.find("infernix_mixed_rounds_total 3\n") != std::string::npos,
          "mixed rounds must be observable");
    check(live.find("infernix_decode_graph_replays_total 12\n") != std::string::npos &&
              live.find("infernix_decode_eager_executions_total 2\n") != std::string::npos &&
              live.find("infernix_decode_graph_captures_total 1\n") != std::string::npos,
          "decode graph replays, eager executions and captures must be observable apart");
    check(live.find("infernix_spec_decode_rounds_total{backend=\"dflash2\"} 7\n") !=
                  std::string::npos &&
              live.find("infernix_spec_decode_accepted_tokens_total{backend=\"dflash2\"} 3\n") !=
                  std::string::npos &&
              live.find("infernix_spec_decode_fallback_steps_total{backend=\"dflash2\"} 0\n") !=
                  std::string::npos &&
              live.find("infernix_spec_decode_rounds_total{backend=\"ngram\"} 0\n") !=
                  std::string::npos,
          "speculative counters must carry the drafter backend label");
    check(live.find("infernix_spec_decode_accepted_positions_total{position=\"1\"} 0\n") !=
                  std::string::npos &&
              live.find("infernix_spec_decode_accepted_positions_total{position=\"4\"} 0\n") !=
                  std::string::npos &&
              live.find("infernix_spec_decode_accepted_positions_total{position=\"5\"} 0\n") ==
                  std::string::npos,
          "the acceptance positions must cover exactly the configured draft window");
    check(live.find("infernix_phase_work_seconds_total{phase=\"decode\",work=\"host\"} 2\n") !=
                  std::string::npos &&
              live.find(
                  "infernix_phase_work_seconds_total{phase=\"decode\",work=\"device_wait\"} 4\n") !=
                  std::string::npos &&
              live.find("infernix_phase_work_seconds_total{phase=\"prefill\",work=\"host\"} 1\n") !=
                  std::string::npos &&
              live.find("infernix_phase_work_seconds_total{phase=\"control\",work=\"host\"} 0\n") !=
                  std::string::npos,
          "the decode/prefill/control host and device-wait split must be observable");
    check(live.find("infernix_expert_cache_routed_total 90\n") != std::string::npos &&
              live.find("infernix_expert_cache_hits_total 75\n") != std::string::npos &&
              live.find("infernix_expert_cache_promotion_bytes_total 4000\n") !=
                  std::string::npos &&
              live.find("infernix_expert_cache_demotion_bytes_total 512\n") != std::string::npos,
          "expert-cache routed, hits and promotion/demotion bytes must be observable as deltas");

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.metrics.total_seconds                    = 2.0;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    outcome.metrics.ngram_rounds                     = 2;
    outcome.metrics.ngram_drafted_tokens             = 8;
    outcome.metrics.ngram_accepted_tokens            = 5;
    outcome.metrics.speculative_accepted_per_position = {3, 2, 1};
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, placement, false);
    check(final.find("infernix_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("infernix_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("infernix_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("infernix_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("infernix_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("infernix_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("infernix_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("infernix_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("infernix_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    outcome.constraint = ConstraintObservation{.complete          = true,
                                               .terminated        = false,
                                               .cache             = ConstraintCacheAccess::Built,
                                               .mask_positions    = 5,
                                               .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, placement, false);
    check(constrained.find(
              "infernix_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("infernix_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("infernix_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    check(constrained.find("infernix_spec_decode_rounds_total{backend=\"ngram\"} 804\n") !=
                  std::string::npos &&
              constrained.find("infernix_spec_decode_draft_tokens_total{backend=\"ngram\"} 3216\n") !=
                  std::string::npos &&
              constrained.find(
                  "infernix_spec_decode_accepted_tokens_total{backend=\"ngram\"} 2010\n") !=
                  std::string::npos,
          "n-gram drafting must aggregate across settled requests");
    check(constrained.find("infernix_spec_decode_accepted_positions_total{position=\"1\"} 1206\n") !=
                  std::string::npos &&
              constrained.find(
                  "infernix_spec_decode_accepted_positions_total{position=\"2\"} 804\n") !=
                  std::string::npos &&
              constrained.find(
                  "infernix_spec_decode_accepted_positions_total{position=\"3\"} 402\n") !=
                  std::string::npos &&
              constrained.find(
                  "infernix_spec_decode_accepted_positions_total{position=\"4\"} 0\n") !=
                  std::string::npos,
          "the acceptance positions must aggregate the acceptance-length distribution");
    return failures == 0 ? 0 : 1;
}
