#pragma once

#include "serve/generation_service.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

namespace infernix::serve {

// Fixed-size request aggregates. Runtime counters remain owned and published by Engine.
class Metrics {
public:
    void configure(std::string model, const EngineOptions& options, const MemorySummary& memory,
                   RuntimeStats baseline, const infernix::ExpertPlacementReport& placement_baseline);
    void first_token(const GenerationFirstTokenObservation& observation);
    void done(const GenerationOutcome& outcome);
    void rejected();
    void failed(bool cancelled);
    void response_failed();
    [[nodiscard]] std::string render(const RuntimeStats& runtime,
                                     const infernix::ExpertPlacementReport& placement, bool ready) const;

private:
    struct Histogram {
        // Disjoint bins while recording; cumulative buckets are produced only when scraped.
        std::array<std::uint64_t, 21> bins{};
        double sum = 0.0;
        void observe(double seconds);
    };

    // The widest draft position a round can accept: neural drafts (draft tokens and tree widths)
    // and n-gram copy proposals (ngram_draft_tokens, at most 63) both bounded by the options.
    static constexpr std::size_t kAcceptedPositionCapacity = 64;

    struct Requests {
        std::uint64_t completed         = 0;
        std::uint64_t cancelled         = 0;
        std::uint64_t failed            = 0;
        std::uint64_t rejected          = 0;
        std::uint64_t response_failures = 0;
        std::array<std::uint64_t, 3> constraint_outcomes{};
        std::array<std::uint64_t, 3> constraint_cache{};
        double constraint_prepare_seconds     = 0;
        double constraint_mask_seconds        = 0;
        double constraint_matcher_seconds     = 0;
        std::uint64_t constraint_positions    = 0;
        std::uint64_t constraint_upload_bytes = 0;
        // N-gram copy-proposal drafting (beside the configured neural backend, or alone on
        // Qwen3.8-Flash-Next), aggregated at request settlement like the constraint work.
        std::uint64_t ngram_rounds          = 0;
        std::uint64_t ngram_drafted_tokens  = 0;
        std::uint64_t ngram_accepted_tokens = 0;
        // Draft acceptance by position, aggregated across requests: entry i counts the rounds
        // whose accepted sequence reached position i + 1 (the acceptance-length distribution).
        std::array<std::uint64_t, kAcceptedPositionCapacity> accepted_positions{};
        Histogram ttft;
        Histogram duration;
        Histogram queue;
    };

    // The expert cache's absolute counters at startup warmup, subtracted from the placement
    // report's values so /metrics counts only serving work.
    struct ExpertCacheBaseline {
        std::uint64_t routed          = 0;
        std::uint64_t hits            = 0;
        std::uint64_t promotion_bytes = 0;
        std::uint64_t demotion_bytes  = 0;
    };

    std::string model_;
    EngineOptions options_;
    MemorySummary memory_;
    RuntimeStats baseline_;
    ExpertCacheBaseline expert_cache_baseline_;
    double start_seconds_ = 0.0;
    mutable std::mutex mutex_;
    Requests requests_;
};

} // namespace infernix::serve
