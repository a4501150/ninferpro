#pragma once

// The CPU expert team's measured rates and the rule that settles them
// (docs/maintainer/moe-offload-design.md §5.5): the rates set the decode miss split for the whole
// session, and memory is slow for some seconds after the banks' load or the SSD tier's fill, or
// after another process frees tens of GiB, so the readings repeat until they settle. Used by
// qwen4_exp's Program (program_impl.h measure_cpu_rate_settled); host only.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::models::qwen4_exp::detail {

// One reading of the CPU expert team's rates: experts/s on one-column jobs (decode's misses), expert
// columns/s on the narrow route's widest jobs (the prefill split's per-column cost), and how it was
// taken.
struct CpuRateReading {
    double experts    = 0.0;
    double columns    = 0.0;
    double busy_share = -1.0; // other processes' CPU share when they kept the CPU busy through it, else < 0
    std::string note;         // how it was taken (the startup report's parenthesis)
};

// The rates the decode miss split and the prefill CPU split take for the session.
struct SettledCpuRates {
    double experts = 0.0; // the best of the readings
    double columns = 0.0;
    std::vector<double> readings; // each reading's experts/s in order (none: no expert record in RAM, the
                                  // reference rates stand)
    double busy_share = -1.0;     // the last reading's when every reading was busy (the startup warns), else < 0
    std::string note;             // the last reading's
    // The startup report's account of the readings: "; the best of 2 readings (9377, 18942 experts/s)", or "".
    [[nodiscard]] std::string readings_text() const {
        if (readings.size() < 2) { return {}; }
        std::string text = "; the best of " + std::to_string(readings.size()) + " readings (";
        for (std::size_t i = 0; i < readings.size(); ++i) {
            text += (i > 0 ? ", " : "") + std::to_string(std::lround(readings[i]));
        }
        return text + " experts/s)";
    }
};

// Right after the expert banks' load or the SSD tier's fill (tens of GiB of direct reads), or while
// the OS clears memory another process freed, memory is slow for some seconds, and a low reading
// keeps too many decode misses on the link for the whole session: upstream's Flash-Next Dense8 read
// 9,377 experts/s right after its tier's fill and 18,942 5 s later (decode -26 %, 2026-10-27); its
// Qwen3.5-122B read 4,923 against 9,515-12,805 settled (decode -24 to -40 %, 2026-10-30). So the
// rates are read again kRateSettle after the first reading, and again while a reading beats the best
// experts/s by more than kRateSettleGain, at most kRateSettleReadings readings; the best of each rate
// stands. `read` takes one reading (nullopt without expert records in RAM); `pause` waits between
// readings (empty: kRateSettle).
inline constexpr std::chrono::seconds kRateSettle{4};
inline constexpr int kRateSettleReadings = 3;
inline constexpr double kRateSettleGain  = 0.10;
[[nodiscard]] inline SettledCpuRates settle_cpu_rates(const std::function<std::optional<CpuRateReading>()>& read,
                                                      const std::function<void()>& pause = {}) {
    SettledCpuRates out;
    bool quiet = false; // some reading was taken on a quiet CPU
    while (out.readings.size() < static_cast<std::size_t>(kRateSettleReadings)) {
        if (!out.readings.empty()) {
            if (pause) {
                pause();
            } else {
                std::this_thread::sleep_for(kRateSettle);
            }
        }
        const std::optional<CpuRateReading> reading = read();
        if (!reading) { break; }
        const bool improved = reading->experts > out.experts * (1.0 + kRateSettleGain);
        out.experts         = std::max(out.experts, reading->experts);
        out.columns         = std::max(out.columns, reading->columns);
        out.note            = reading->note;
        quiet               = quiet || reading->busy_share < 0.0;
        out.busy_share      = quiet ? -1.0 : reading->busy_share;
        out.readings.push_back(reading->experts);
        if (!improved) { break; }
    }
    return out;
}

} // namespace infernix::models::qwen4_exp::detail
