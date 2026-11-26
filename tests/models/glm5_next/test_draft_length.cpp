// Glm5Next's DFlash2 draft-length policy (program/draft_length.h): the calibrated length against a
// direct evaluation of expected tokens per round time over the measured round-time table, and the
// calibration's updates.

#include "models/glm5_next/program/draft_length.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::models::glm5_next::detail::calibrated_draft_length;
using infernix::models::glm5_next::detail::kRoundCost;
using infernix::models::glm5_next::detail::SelectorCalibration;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

// The k in [0, most] with the highest expected tokens per round time, by exhaustive evaluation.
int brute_force(const SelectorCalibration& c, const std::vector<float>& p, int most) {
    int best          = 0;
    double best_score = 1.0 / kRoundCost[0];
    for (int k = 1; k <= most && k < static_cast<int>(kRoundCost.size()); ++k) {
        double tokens = 1.0;
        for (int j = 1; j <= k; ++j) {
            double reach = 1.0;
            for (int i = 0; i < j; ++i) { reach *= c.accept[SelectorCalibration::bin(p[static_cast<std::size_t>(i)])]; }
            tokens += reach;
        }
        const double score = tokens / kRoundCost[static_cast<std::size_t>(k)];
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
        for (std::size_t k = 1; k < kRoundCost.size(); ++k) {
            require(kRoundCost[k] > kRoundCost[k - 1], "a longer round costs more");
        }
        SelectorCalibration c;
        c.reset();
        // Confident drafts are verified up to the most; unlikely ones decode the round plainly; a
        // doubtful draft ends the run.
        const std::vector<float> sure(7, 0.97F), doubtful(7, 0.05F);
        // At the deciles' face value (0.95) four and five drafts score within 0.1 %; past five the
        // table's columns cost more than they yield.
        require(calibrated_draft_length(c, sure, 5) >= 4, "confident drafts are verified long");
        require(calibrated_draft_length(c, sure, 7) <= 5, "confident drafts stop where columns stop paying");
        require(calibrated_draft_length(c, doubtful, 5) == 0, "unlikely drafts decode the round plainly");
        const std::vector<float> broken{0.97F, 0.96F, 0.05F, 0.97F, 0.97F, 0.97F, 0.97F};
        require(calibrated_draft_length(c, broken, 5) == 2, "a doubtful third draft ends the run");
        require(calibrated_draft_length(c, sure, 3) == 3, "the length never exceeds `most`");
        require(calibrated_draft_length(c, std::vector<float>(2, 0.97F), 5) == 2,
                "the length never exceeds the drafts");
        std::uint32_t state = 7;
        for (int trial = 0; trial < 2000; ++trial) {
            std::vector<float> p(7);
            for (float& v : p) {
                state = state * 1664525U + 1013904223U;
                v     = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            }
            const int most = 1 + trial % 7;
            require(calibrated_draft_length(c, p, most) == brute_force(c, p, most),
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
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("glm5_next draft length policy checks passed\n");
    return 0;
}
