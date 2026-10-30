#pragma once

#include "infernix/types.h"

#include <memory>

namespace spdlog {
class logger;
}

namespace infernix::product {

// Renders Engine runtime diagnostics as ordinary operational records ("engine | ...") at the level
// the Engine assigned, so --log-level filters them and a pinned terminal panel stays beneath them.
// `debug`: the DebugCategory bits whose trace lines the Engine publishes (--debug).
[[nodiscard]] DiagnosticObserver engine_diagnostic_observer(std::shared_ptr<spdlog::logger> logger,
                                                            std::uint32_t debug = 0);

} // namespace infernix::product
