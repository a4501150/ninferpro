#include "product/logging/engine_diagnostics.h"

#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include <utility>

namespace infernix::product {

DiagnosticObserver engine_diagnostic_observer(std::shared_ptr<spdlog::logger> logger, std::uint32_t debug) {
    return DiagnosticObserver{
        .callback = [logger = std::move(logger)](const Diagnostic& diagnostic) {
            const std::string message = format_pretty_text(diagnostic.message);
            if (diagnostic.category != DebugCategory::None) {
                // A --debug trace line: asked for by its category, so shown at any log level.
                logger->info("{} | {}", debug_category_name(diagnostic.category), message);
                return;
            }
            switch (diagnostic.level) {
            case DiagnosticLevel::Debug:
                logger->debug("engine | {}", message);
                break;
            case DiagnosticLevel::Info:
                logger->info("engine | {}", message);
                break;
            case DiagnosticLevel::Warning:
                logger->warn("engine | {}", message);
                break;
            case DiagnosticLevel::Error:
                logger->error("engine | {}", message);
                break;
            }
        },
        .debug = debug};
}

} // namespace infernix::product
