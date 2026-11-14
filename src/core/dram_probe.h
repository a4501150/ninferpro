#pragma once

// The device-memory read rate (DRAM as streaming kernels see it), measured on the running GPU so
// speed-of-light figures follow the card instead of a datasheet constant.

#include <cstddef>

namespace infernix {

// Allocates `bytes` of device memory (much larger than L2), fills it with distinct words, streams it
// through every SM `repetitions` times after a warm-up and returns the best timed read rate in bytes
// per second. Runs on a private stream and frees the buffer before it returns.
[[nodiscard]] double measure_device_read_bytes_per_second(std::size_t bytes, int repetitions);

} // namespace infernix
