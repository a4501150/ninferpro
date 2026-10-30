#pragma once

#include "infernix/types.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>

namespace infernix::runtime {

inline void publish_diagnostic_v(const DiagnosticObserver& observer, DiagnosticLevel level, DebugCategory category,
                                 const char* format, std::va_list args) noexcept {
    std::va_list copy;
    va_copy(copy, args);
    const int length = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (length < 0) { return; }
    std::string message;
    try {
        message.resize(static_cast<std::size_t>(length));
    } catch (...) { return; }
    std::vsnprintf(message.data(), message.size() + 1, format, args);
    if (!observer.callback) {
        if (level != DiagnosticLevel::Debug || category != DebugCategory::None) {
            std::fprintf(stderr, "[%s] %s\n", debug_category_name(category), message.c_str());
        }
        return;
    }
    try {
        observer.callback(Diagnostic{.level = level, .category = category, .message = std::move(message)});
    } catch (...) {}
}

// Publishes one Engine runtime diagnostic through the product's observer. Without an observer,
// Info and above are written to stderr. It never throws: a logging failure must not disturb the
// worker or a destructor.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 3, 4)))
#endif
inline void publish_diagnostic(const DiagnosticObserver& observer, DiagnosticLevel level,
                               const char* format, ...) noexcept {
    std::va_list args;
    va_start(args, format);
    publish_diagnostic_v(observer, level, DebugCategory::None, format, args);
    va_end(args);
}

// Publishes one --debug trace line of `category`; nothing is formatted while the category is off.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 3, 4)))
#endif
inline void publish_trace(const DiagnosticObserver& observer, DebugCategory category, const char* format,
                          ...) noexcept {
    if (!observer.tracing(category)) [[likely]] { return; }
    std::va_list args;
    va_start(args, format);
    publish_diagnostic_v(observer, DiagnosticLevel::Debug, category, format, args);
    va_end(args);
}

} // namespace infernix::runtime
