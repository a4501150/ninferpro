// The rule that settles the CPU expert team's measured rates (models/qwen4_exp/program/cpu_rates.h):
// a second reading 4 s after the first, a third 4 s later only if the second beat the first's
// experts/s by more than 10 %, the best of each rate standing, on scripted readings. Host only.

#include "models/qwen4_exp/program/cpu_rates.h"

#include <cstdio>
#include <optional>
#include <vector>

namespace {

using infernix::models::qwen4_exp::detail::CpuRateReading;
using infernix::models::qwen4_exp::detail::settle_cpu_rates;
using infernix::models::qwen4_exp::detail::SettledCpuRates;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

} // namespace

int main() {
    const auto settle = [](std::vector<CpuRateReading> readings, int* pauses) {
        std::size_t next = 0;
        *pauses          = 0;
        return settle_cpu_rates(
            [&]() -> std::optional<CpuRateReading> {
                if (next == readings.size()) { return std::nullopt; }
                return readings[next++];
            },
            [&] { ++*pauses; });
    };
    int pauses = 0;
    SettledCpuRates rates = settle({{10'000.0, 80'000.0, -1.0, "a"}, {10'500.0, 78'000.0, -1.0, "b"}}, &pauses);
    check(rates.readings.size() == 2 && pauses == 1, "a quiet start takes two readings");
    check(rates.experts == 10'500.0 && rates.columns == 80'000.0, "each rate keeps its best reading");
    check(rates.note == "b" && rates.busy_share < 0.0, "the report takes the last reading's note");
    check(rates.readings_text() == "; the best of 2 readings (10000, 10500 experts/s)", "the report lists the readings");
    rates = settle({{12'000.0, 1.0, -1.0, ""}, {9'000.0, 1.0, -1.0, ""}}, &pauses);
    check(rates.readings.size() == 2 && rates.experts == 12'000.0, "a lower second reading leaves the first");
    // A tier right after its fill (upstream's Dense8, 2026-10-27): 4,923 experts/s, then the settled rate.
    rates = settle({{4'923.0, 1.0, -1.0, ""}, {9'515.0, 1.0, -1.0, ""}, {12'805.0, 1.0, -1.0, ""}, {20'000.0, 1.0, -1.0, ""}},
                   &pauses);
    check(rates.readings.size() == 3 && pauses == 2 && rates.experts == 12'805.0,
          "a recovering memory reads at most three times");
    rates = settle({{5'000.0, 1.0, -1.0, ""}, {5'400.0, 1.0, -1.0, ""}, {20'000.0, 1.0, -1.0, ""}}, &pauses);
    check(rates.readings.size() == 2 && rates.experts == 5'400.0, "a gain of 10 % or less ends the readings");
    rates = settle({}, &pauses);
    check(rates.readings.empty() && pauses == 0 && rates.experts == 0.0 && rates.readings_text().empty(),
          "no expert record in RAM: no reading, no pause");
    rates = settle({{9'000.0, 1.0, 0.40, ""}, {10'000.0, 1.0, -1.0, ""}}, &pauses);
    check(rates.busy_share < 0.0, "a quiet reading clears the busy warning");
    rates = settle({{9'000.0, 1.0, -1.0, ""}, {10'000.0, 1.0, 0.40, ""}, {9'500.0, 1.0, 0.50, ""}}, &pauses);
    check(rates.busy_share < 0.0, "a busy reading after a quiet one does not warn");
    rates = settle({{5'000.0, 1.0, 0.40, ""}, {9'000.0, 1.0, 0.50, ""}, {9'500.0, 1.0, 0.45, ""}}, &pauses);
    check(rates.busy_share == 0.45, "every reading busy: the warning gives the last one's share");
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("cpu rates: all checks passed\n");
    return 0;
}
