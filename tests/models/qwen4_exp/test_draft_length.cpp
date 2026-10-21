// Qwen4Exp's DFlash2 draft-length policy (program/draft_length.h): the calibrated length against a
// direct evaluation of expected tokens per round time, and the calibration's updates.

#include "models/qwen4_exp/program/draft_length.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::models::qwen4_exp::detail::calibrated_draft_length;
using infernix::models::qwen4_exp::detail::SelectorCalibration;
using infernix::models::qwen4_exp::detail::shared_draft_length;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

// The k in [0, most] with the highest expected tokens per round time, by exhaustive evaluation.
int brute_force(const SelectorCalibration& c, const std::vector<float>& p, int most, double w, double f) {
    int best = 0;
    double best_score = 1.0 / (1.0 + f);
    for (int k = 1; k <= most; ++k) {
        double tokens = 1.0;
        for (int j = 1; j <= k; ++j) {
            double reach = 1.0;
            for (int i = 0; i < j; ++i) { reach *= c.accept[SelectorCalibration::bin(p[static_cast<std::size_t>(i)])]; }
            tokens += reach;
        }
        const double score = tokens / (1.0 + f + w * k);
        if (score > best_score + 1e-12) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

} // namespace

int main() {
    try {
        SelectorCalibration c;
        c.reset();
        require(std::abs(c.accept[0] - 0.05) < 1e-12 && std::abs(c.accept[9] - 0.95) < 1e-12, "decile centres");
        require(SelectorCalibration::bin(0.0F) == 0 && SelectorCalibration::bin(0.999F) == 9 &&
                    SelectorCalibration::bin(1.0F) == 9 && SelectorCalibration::bin(-1.0F) == 0,
                "probabilities map to deciles 0..9");
        // Confident drafts are all verified; unlikely ones are not; a low draft ends the run.
        const std::vector<float> sure(7, 0.97F), doubtful(7, 0.05F);
        require(calibrated_draft_length(c, sure, 7, 0.31, 0.1) == 7, "seven confident drafts are verified");
        require(calibrated_draft_length(c, doubtful, 7, 0.31, 0.1) == 0, "unlikely drafts are not verified");
        const std::vector<float> broken{0.97F, 0.96F, 0.05F, 0.97F, 0.97F, 0.97F, 0.97F};
        require(calibrated_draft_length(c, broken, 7, 0.31, 0.1) == 2, "a doubtful third draft ends the run");
        require(calibrated_draft_length(c, sure, 3, 0.31, 0.1) == 3, "the length never exceeds `most`");
        // Against the exhaustive choice over a grid of probabilities and costs.
        std::uint32_t state = 7;
        for (int trial = 0; trial < 2000; ++trial) {
            std::vector<float> p(7);
            for (float& v : p) {
                state = state * 1664525U + 1013904223U;
                v     = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            }
            const double w = 0.1 + 0.05 * (trial % 9), f = 0.05 * (trial % 4);
            const int most = 1 + trial % 7;
            require(calibrated_draft_length(c, p, most, w, f) == brute_force(c, p, most, w, f),
                    "the policy's choice equals the exhaustive one (trial " + std::to_string(trial) + ")");
        }
        // Updates: an accepted draft raises its decile, the first rejected lowers its own, later
        // drafts are untouched.
        SelectorCalibration u;
        u.reset();
        const std::array<float, 4> probabilities{0.92F, 0.71F, 0.55F, 0.33F};
        u.observe(probabilities, 4, 1, 0.5);
        require(std::abs(u.accept[9] - (0.5 * 0.95 + 0.5)) < 1e-12, "an accepted draft raises its decile");
        require(std::abs(u.accept[7] - 0.5 * 0.75) < 1e-12, "the first rejected draft lowers its decile");
        require(u.accept[5] == 0.55 && u.accept[3] == 0.35, "drafts after the first rejection are untouched");
        u.reset();
        u.observe(probabilities, 2, 2, 0.5);
        require(u.accept[9] > 0.95 && u.accept[7] > 0.75 && u.accept[5] == 0.55,
                "an all-accepted round raises every verified draft's decile and nothing past it");

        // The shared multi-row length (shared_draft_length) against a direct evaluation of
        // expected tokens per round time: drafting rows contribute their acceptance chains, plain
        // rows only their one token, and k = 0 is the plain round. A drafting round also pays the
        // fixed overhead `fixed` before its first column, and must beat plain by `margin`.
        const auto batch_brute = [](const std::vector<std::vector<double>>& rows, int plain, int most,
                                    double w, double fixed, double margin) {
            int best = 0;
            const double base = static_cast<double>(plain + static_cast<int>(rows.size()));
            double best_score = base * (1.0 + margin);
            for (int k = 1; k <= most; ++k) {
                double tokens = base;
                for (const auto& row : rows) {
                    double reach = 1.0;
                    for (int j = 0; j < k; ++j) {
                        reach *= row[static_cast<std::size_t>(j)];
                        tokens += reach; // each verified position's expected acceptance
                    }
                }
                const double score = tokens / (1.0 + fixed + w * k);
                if (score > best_score + 1e-12) {
                    best_score = score;
                    best       = k;
                }
            }
            return best;
        };
        const std::vector<double> hot(7, 0.9), cold(7, 0.05);
        const std::array<std::span<const double>, 3> cold_rows{
            std::span<const double>(cold), std::span<const double>(cold), std::span<const double>(cold)};
        const std::array<std::span<const double>, 2> sure_rows{
            std::span<const double>(hot), std::span<const double>(hot)};
        require(shared_draft_length(std::span<const std::span<const double>>(cold_rows), 3, 7, 0.31) == 0,
                "unlikely drafts keep the plain round");
        require(shared_draft_length(std::span<const std::span<const double>>(sure_rows), 0, 7, 0.0) == 7,
                "free verification drafts the most");
        require(shared_draft_length(std::span<const std::span<const double>>(sure_rows), 0, 7, 0.0, 10.0) == 0,
                "a fixed overhead past the yield keeps the plain round");
        require(shared_draft_length(std::span<const std::span<const double>>(sure_rows), 0, 7, 0.0, 0.0, 0.05) == 7,
                "a clear win drafts through the margin");
        const std::vector<double> thin(7, 0.35);
        const std::array<std::span<const double>, 2> thin_rows{
            std::span<const double>(thin), std::span<const double>(thin)};
        require(shared_draft_length(std::span<const std::span<const double>>(thin_rows), 5, 7, 0.05) > 0,
                "without a margin a thin win still drafts");
        require(shared_draft_length(std::span<const std::span<const double>>(thin_rows), 5, 7, 0.05, 0.0, 0.4) == 0,
                "a thin win stays plain behind the margin");
        std::uint32_t bstate = 13;
        for (int trial = 0; trial < 2000; ++trial) {
            std::vector<std::vector<double>> backing;
            const int drafting = 1 + static_cast<int>(bstate >> 30) % 6; // 1..6 drafting rows
            for (int r = 0; r < drafting; ++r) {
                std::vector<double> row(7);
                for (double& v : row) {
                    bstate = bstate * 1664525U + 1013904223U;
                    v      = static_cast<double>(bstate >> 8) / static_cast<double>(1U << 24);
                }
                backing.push_back(std::move(row));
            }
            std::vector<std::span<const double>> spans;
            for (const auto& row : backing) { spans.emplace_back(row); }
            const int plain = static_cast<int>(bstate >> 28) % 3;
            const int most  = 1 + static_cast<int>(trial % 7);
            const double w  = 0.05 + 0.1 * (trial % 10);
            bstate = bstate * 1664525U + 1013904223U;
            const double fixed  = 0.25 * static_cast<double>(bstate >> 8) / static_cast<double>(1U << 24);
            const double margin = 0.1 * static_cast<double>((bstate >> 8) & 0xFU) / 15.0;
            const int got  = shared_draft_length(std::span<const std::span<const double>>(spans), plain, most,
                                                 w, fixed, margin);
            const int want = batch_brute(backing, plain, most, w, fixed, margin);
            if (got != want) {
                throw std::runtime_error("the shared policy's choice equals the exhaustive one (trial " +
                                         std::to_string(trial) + ": " + std::to_string(got) + " vs " +
                                         std::to_string(want) + ")");
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp draft length policy checks passed\n");
    return 0;
}
