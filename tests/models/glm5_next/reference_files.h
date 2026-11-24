#pragma once

// The reference directory's files (tools/glm5_next/reference.py --out DIR) and the log-probability
// the real-artifact tests compare against them.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace glm_test {

inline std::vector<std::int32_t> read_tokens(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    return {std::istream_iterator<std::int32_t>(in), std::istream_iterator<std::int32_t>()};
}

inline std::vector<float> read_floats(const std::string& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    std::vector<float> out(count);
    if (!in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
        throw std::runtime_error("cannot read " + path);
    }
    return out;
}

// log softmax(logits)[token] over the first `domain` rows.
inline double log_probability(const float* logits, std::size_t domain, std::int32_t token) {
    double peak = -INFINITY;
    for (std::size_t i = 0; i < domain; ++i) { peak = std::max(peak, static_cast<double>(logits[i])); }
    double sum = 0;
    for (std::size_t i = 0; i < domain; ++i) { sum += std::exp(static_cast<double>(logits[i]) - peak); }
    return static_cast<double>(logits[token]) - peak - std::log(sum);
}

inline std::size_t argmax(const float* v, std::size_t n) {
    return static_cast<std::size_t>(std::max_element(v, v + n) - v);
}

} // namespace glm_test
