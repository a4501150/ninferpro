#pragma once

// A reader of the Prometheus text exposition format, as served by infernix-serve's /metrics.

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::monitor {

struct MetricSample {
    std::string name;
    std::map<std::string, std::string> labels;
    double value = 0.0;
};

// Comment lines and lines that do not parse are skipped.
std::vector<MetricSample> parse_prometheus(std::string_view text);

class Metrics {
public:
    Metrics() = default;
    explicit Metrics(std::vector<MetricSample> samples) : samples_(std::move(samples)) {}

    // The sum over every sample of `name` whose labels include `label`=`value` (any sample when
    // `label` is empty); nullopt when no sample matches.
    std::optional<double> sum(std::string_view name, std::string_view label = {},
                              std::string_view value = {}) const;
    // The value of `label` on the first sample of `name`; empty when there is none.
    std::string label(std::string_view name, std::string_view label) const;
    bool empty() const noexcept { return samples_.empty(); }

private:
    std::vector<MetricSample> samples_;
};

} // namespace infernix::monitor
