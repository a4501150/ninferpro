#pragma once

// Glm5Next's DFlash2 draft-length policy (--draft-tokens auto; glm5_3-flash-design.md §10.2). A
// proposal drafts its whole block in one pass, so a round's choice is how many of its drafts to
// verify, made after the proposal from each draft's selector probability (the candidate selector's
// temperature-1 softmax at the drafted rank): the k that maximizes the expected tokens per round
// time. A draft's acceptance given the earlier ones' is read from its probability's decile, learned
// per request, since the selector's probability over its 16 candidates is not calibrated on its own.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix::models::glm5_next::detail {

struct SelectorCalibration {
    static constexpr std::size_t kBins = 10;
    // Conditional acceptance of a draft whose selector probability falls in decile i.
    std::array<double, kBins> accept{};

    // Each decile's centre: a probability's face value until the request's rounds say otherwise.
    void reset() noexcept {
        for (std::size_t i = 0; i < kBins; ++i) { accept[i] = (static_cast<double>(i) + 0.5) / kBins; }
    }

    [[nodiscard]] static std::size_t bin(float probability) noexcept {
        return static_cast<std::size_t>(
            std::clamp(static_cast<int>(probability * kBins), 0, static_cast<int>(kBins) - 1));
    }

    // A verified round: drafts [0, verified) carried `probabilities`; the first `accepted` were
    // accepted and, when accepted < verified, draft `accepted` was the first rejected (later drafts
    // say nothing about their own acceptance). EWMA with `weight`.
    void observe(std::span<const float> probabilities, std::uint32_t verified, std::uint32_t accepted,
                 double weight) noexcept {
        for (std::uint32_t j = 0; j < verified && j <= accepted && j < probabilities.size(); ++j) {
            double& a = accept[bin(probabilities[j])];
            a         = (1.0 - weight) * a + weight * (j < accepted ? 1.0 : 0.0);
        }
    }
};

// A one-row round's time by the drafts it verifies, in plain rounds, the proposal included: the
// BF16-dense artifact at vram=42%,ram=58%, one request, 512-token essays (design §10.3: 28 ms plain,
// 42 / 50 / 55 / 64 / 75 ms at K = 1..5). Used as the warm-up prior before the online fit settles.
inline constexpr std::array<double, 8> kRoundCost = {1.07, 1.50, 1.79, 1.96, 2.29, 2.68, 3.07, 3.46};

// Online one-row round-time model: learns the proposal overhead, the plain-step time (k = 0), and
// the verifying-step time t_verify(k) = a_v + b * k (k >= 1) by exponentially weighted least
// squares. Keeping k = 0 separate from k >= 1 preserves the fixed verify/commit overhead (so k = 1
// is not underpriced against k = 0 when b / a is low). Until enough spread verify rounds arrive
// (or when the fit is non-positive), kRoundCost stands. Reset on expert placement switches.
struct RoundCostModel {
    static constexpr double kDecay         = 0.97; // ~33-round window: smooths miss variance while shedding warmup
    static constexpr double kMinRounds     = 14.0;
    static constexpr double kMinPlain      = 4.0;
    static constexpr double kMinSpread     = 0.25;
    static constexpr double kGapRatio      = 3.0;
    static constexpr double kMinSlope      = 0.05;
    static constexpr double kMaxSlope      = 0.80;
    static constexpr double kMaxProposal   = 0.50;
    static constexpr double kPriorVerifyFx = 0.08; // verify/commit overhead over plain before k=0 settles

    double n = 0.0, k = 0.0, kk = 0.0, t = 0.0, kt = 0.0; // k >= 1 verify steps
    double plain_n = 0.0, plain_t = 0.0;                  // k == 0 plain steps
    double prop_n = 0.0, prop_t = 0.0;                    // proposal + prior commit

    void reset() noexcept { *this = {}; }

    void observe_proposal(double seconds) noexcept {
        if (!(seconds > 0.0)) { return; }
        const double predict = prop_n > 0.0 ? prop_t / prop_n : seconds;
        if (prop_n >= 4.0 && seconds > kGapRatio * predict) { return; }
        prop_n = kDecay * prop_n + 1.0;
        prop_t = kDecay * prop_t + seconds;
    }

    void observe_step(std::int32_t drafts, double seconds) noexcept {
        if (drafts < 0 || !(seconds > 0.0)) { return; }
        if (drafts == 0) {
            const double predict = plain_n > 0.0 ? plain_t / plain_n : seconds;
            if (plain_n >= 4.0 && seconds > kGapRatio * predict) { return; }
            plain_n = kDecay * plain_n + 1.0;
            plain_t = kDecay * plain_t + seconds;
            return;
        }
        double a = 0.0, b = 0.0;
        const double kd      = static_cast<double>(drafts);
        const double predict = fit_step(a, b) ? (a + b * kd) : (n > 0.0 ? t / n : seconds);
        if (n >= 4.0 && seconds > kGapRatio * predict) { return; }
        n  = kDecay * n + 1.0;
        k  = kDecay * k + kd;
        kk = kDecay * kk + kd * kd;
        t  = kDecay * t + seconds;
        kt = kDecay * kt + kd * seconds;
    }

    [[nodiscard]] bool fit_step(double& a, double& b) const noexcept {
        const double spread = n * kk - k * k;
        if (n < kMinRounds || spread < kMinSpread * n * n) { return false; }
        b = (n * kt - k * t) / spread;
        a = (t - b * k) / n;
        return a > 0.0 && b > 0.0;
    }

    // Normalized proposal, fixed verify overhead (k >= 1 over k = 0), and per-draft slope, in
    // plain-step units.
    [[nodiscard]] bool normalized(double& proposal, double& fixed, double& slope) const noexcept {
        double a_v = 0.0, b = 0.0;
        if (!fit_step(a_v, b)) { return false; }
        const double a_p =
            plain_n >= kMinPlain ? std::min(a_v, plain_t / plain_n) : (a_v / (1.0 + kPriorVerifyFx));
        if (!(a_p > 0.0)) { return false; }
        fixed    = std::clamp((a_v - a_p) / a_p, kPriorVerifyFx, kMaxProposal);
        slope    = std::clamp(b / a_p, kMinSlope, kMaxSlope);
        proposal = prop_n > 0.0 ? std::clamp((prop_t / prop_n) / a_p, 0.0, kMaxProposal) : (kRoundCost[0] - 1.0);
        return true;
    }

    // Cost of verifying `drafts` (0 = plain after a spent proposal), in plain rounds.
    [[nodiscard]] double cost(std::int32_t drafts) const noexcept {
        double proposal = 0.0, fixed = 0.0, slope = 0.0;
        if (normalized(proposal, fixed, slope)) {
            if (drafts <= 0) { return 1.0 + proposal; }
            return (1.0 + proposal + fixed) + slope * static_cast<double>(drafts);
        }
        const auto u = static_cast<std::size_t>(std::max<std::int32_t>(0, drafts));
        if (u < kRoundCost.size()) { return kRoundCost[u]; }
        const double tail = kRoundCost.back() - kRoundCost[kRoundCost.size() - 2];
        return kRoundCost.back() + tail * static_cast<double>(u + 1 - kRoundCost.size());
    }
};

// The number of drafts, in [0, most], a one-row round verifies: maximizing (1 + sum over j <= k of
// the product of the first j drafts' calibrated acceptances) / round_cost.cost(k).
[[nodiscard]] inline std::int32_t calibrated_draft_length(const SelectorCalibration& calibration,
                                                          const RoundCostModel& round_cost,
                                                          std::span<const float> probabilities,
                                                          std::int32_t most) noexcept {
    std::int32_t best = 0;
    double best_score = 1.0 / round_cost.cost(0), reach = 1.0, tokens = 1.0;
    const auto limit  = std::min<std::int32_t>(
        {most, static_cast<std::int32_t>(probabilities.size()), static_cast<std::int32_t>(kRoundCost.size()) - 1});
    for (std::int32_t k = 1; k <= limit; ++k) {
        reach *= calibration.accept[SelectorCalibration::bin(probabilities[static_cast<std::size_t>(k - 1)])];
        tokens += reach;
        const double score = tokens / round_cost.cost(k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

[[nodiscard]] inline std::int32_t calibrated_draft_length(const SelectorCalibration& calibration,
                                                          std::span<const float> probabilities,
                                                          std::int32_t most) noexcept {
    return calibrated_draft_length(calibration, RoundCostModel{}, probabilities, most);
}

} // namespace infernix::models::glm5_next::detail
