#include "prometheus.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>

namespace infernix::monitor {
namespace {

bool name_char(char c) noexcept {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == ':';
}

void skip_spaces(std::string_view line, std::size_t& i) noexcept {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) { ++i; }
}

std::optional<double> parse_value(std::string_view token) {
    if (token == "+Inf" || token == "Inf") { return HUGE_VAL; }
    if (token == "-Inf") { return -HUGE_VAL; }
    if (token == "NaN") { return std::nan(""); }
    double value = 0.0;
    const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || end != token.data() + token.size()) { return std::nullopt; }
    return value;
}

// Parses `{a="x",b="y"}` starting at line[i] == '{'; leaves i past the closing brace.
bool parse_labels(std::string_view line, std::size_t& i, std::map<std::string, std::string>& labels) {
    ++i;
    while (true) {
        skip_spaces(line, i);
        if (i < line.size() && line[i] == '}') {
            ++i;
            return true;
        }
        const std::size_t key_start = i;
        while (i < line.size() && name_char(line[i])) { ++i; }
        if (i == key_start) { return false; }
        std::string key(line.substr(key_start, i - key_start));
        skip_spaces(line, i);
        if (i >= line.size() || line[i] != '=') { return false; }
        ++i;
        skip_spaces(line, i);
        if (i >= line.size() || line[i] != '"') { return false; }
        ++i;
        std::string value;
        while (i < line.size() && line[i] != '"') {
            if (line[i] == '\\' && i + 1 < line.size()) {
                ++i;
                value.push_back(line[i] == 'n' ? '\n' : line[i]);
            } else {
                value.push_back(line[i]);
            }
            ++i;
        }
        if (i >= line.size()) { return false; }
        ++i;
        labels.emplace(std::move(key), std::move(value));
        skip_spaces(line, i);
        if (i < line.size() && line[i] == ',') { ++i; }
    }
}

std::optional<MetricSample> parse_line(std::string_view line) {
    std::size_t i = 0;
    skip_spaces(line, i);
    if (i >= line.size() || line[i] == '#') { return std::nullopt; }
    MetricSample sample;
    const std::size_t name_start = i;
    while (i < line.size() && name_char(line[i])) { ++i; }
    if (i == name_start) { return std::nullopt; }
    sample.name.assign(line.substr(name_start, i - name_start));
    if (i < line.size() && line[i] == '{' && !parse_labels(line, i, sample.labels)) { return std::nullopt; }
    skip_spaces(line, i);
    std::size_t end = i;
    while (end < line.size() && line[end] != ' ' && line[end] != '\t' && line[end] != '\r') { ++end; }
    const auto value = parse_value(line.substr(i, end - i));
    if (!value) { return std::nullopt; }
    sample.value = *value;
    return sample;
}

} // namespace

std::vector<MetricSample> parse_prometheus(std::string_view text) {
    std::vector<MetricSample> samples;
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        const std::string_view line = text.substr(0, newline);
        if (auto sample = parse_line(line)) { samples.push_back(std::move(*sample)); }
        if (newline == std::string_view::npos) { break; }
        text.remove_prefix(newline + 1);
    }
    return samples;
}

std::optional<double> Metrics::sum(std::string_view name, std::string_view label,
                                   std::string_view value) const {
    std::optional<double> total;
    for (const MetricSample& sample : samples_) {
        if (sample.name != name) { continue; }
        if (!label.empty()) {
            const auto it = sample.labels.find(std::string(label));
            if (it == sample.labels.end() || it->second != value) { continue; }
        }
        total = total.value_or(0.0) + sample.value;
    }
    return total;
}

std::string Metrics::label(std::string_view name, std::string_view label) const {
    for (const MetricSample& sample : samples_) {
        if (sample.name != name) { continue; }
        const auto it = sample.labels.find(std::string(label));
        if (it != sample.labels.end()) { return it->second; }
    }
    return {};
}

} // namespace infernix::monitor
