#include "records.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <ctime>

namespace infernix::monitor {
namespace {

using Json = nlohmann::json;

template <typename T>
T field(const Json& object, const char* key, T fallback = T{}) {
    if (!object.is_object()) { return fallback; }
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) { return fallback; }
    try {
        return it->get<T>();
    } catch (const Json::exception&) {
        return fallback;
    }
}

const Json& member(const Json& object, const char* key) {
    static const Json null_json;
    if (!object.is_object()) { return null_json; }
    const auto it = object.find(key);
    return it == object.end() ? null_json : *it;
}

std::string strip_ansi(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\x1b' && i + 1 < text.size() && text[i + 1] == '[') {
            i += 2;
            while (i < text.size() && !(text[i] >= '@' && text[i] <= '~')) { ++i; }
            continue;
        }
        out.push_back(text[i]);
    }
    return out;
}

bool digits(std::string_view text, std::size_t at, std::size_t count) {
    if (at + count > text.size()) { return false; }
    for (std::size_t i = at; i < at + count; ++i) {
        if (std::isdigit(static_cast<unsigned char>(text[i])) == 0) { return false; }
    }
    return true;
}

int number(std::string_view text, std::size_t at, std::size_t count) {
    int value = 0;
    for (std::size_t i = at; i < at + count; ++i) { value = value * 10 + (text[i] - '0'); }
    return value;
}

} // namespace

std::optional<RequestLogRecord> parse_request_log_line(std::string_view line) {
    const Json record = Json::parse(line, nullptr, false);
    if (record.is_discarded() || !record.is_object()) { return std::nullopt; }
    RequestLogRecord out;
    out.server_instance = field<std::string>(record, "server_instance_id");
    const std::string event = field<std::string>(record, "event");
    if (event != "request_done" && event != "request_error") { return out; }
    out.event                  = RequestLogEvent::Finished;
    FinishedRequest& r         = out.request;
    r.time                     = field<double>(record, "timestamp_unix_ms") / 1000.0;
    const Json& request        = member(record, "request");
    r.id                       = field<std::uint64_t>(request, "request_id");
    r.protocol                 = field<std::string>(request, "protocol");
    if (event == "request_error") {
        r.finish = "error";
        return out;
    }
    const Json& result          = member(record, "result");
    r.finish                    = field<std::string>(result, "finish_reason");
    r.prompt_tokens             = field<std::uint64_t>(result, "prompt_tokens");
    r.completion_tokens         = field<std::uint64_t>(result, "completion_tokens");
    r.computed_prefill_tokens   = field<std::uint64_t>(result, "computed_prefill_tokens");
    r.prefix_hit_tokens         = field<std::uint64_t>(result, "prefix_cache_hit_tokens");
    const Json& timings         = member(record, "timings_seconds");
    r.ttft_s                    = field<double>(timings, "ttft");
    r.prefill_s                 = field<double>(timings, "prefill");
    r.decode_s                  = field<double>(timings, "decode");
    r.total_s                   = field<double>(timings, "total");
    const Json& speculative     = member(record, "speculative");
    r.drafted_tokens            = field<std::uint64_t>(speculative, "drafted_tokens");
    r.accepted_tokens           = field<std::uint64_t>(speculative, "accepted_tokens");
    return out;
}

LogLine parse_log_line(std::string_view raw) {
    const std::string text = strip_ansi(raw);
    const std::string_view line(text);
    LogLine out;
    // 2026-11-17 18:41:36.591  INFO  message
    const bool stamped = line.size() > 24 && digits(line, 0, 4) && line[4] == '-' && digits(line, 5, 2) &&
                         line[7] == '-' && digits(line, 8, 2) && line[10] == ' ' && digits(line, 11, 2) &&
                         line[13] == ':' && digits(line, 14, 2) && line[16] == ':' && digits(line, 17, 2) &&
                         line[19] == '.' && digits(line, 20, 3);
    if (!stamped) {
        out.message = text;
        return out;
    }
    std::tm tm{};
    tm.tm_year  = number(line, 0, 4) - 1900;
    tm.tm_mon   = number(line, 5, 2) - 1;
    tm.tm_mday  = number(line, 8, 2);
    tm.tm_hour  = number(line, 11, 2);
    tm.tm_min   = number(line, 14, 2);
    tm.tm_sec   = number(line, 17, 2);
    tm.tm_isdst = -1;
    out.time    = static_cast<double>(std::mktime(&tm)) + number(line, 20, 3) / 1000.0;
    std::size_t i = 23;
    while (i < line.size() && line[i] == ' ') { ++i; }
    const std::size_t level_start = i;
    while (i < line.size() && line[i] != ' ') { ++i; }
    const std::string_view level = line.substr(level_start, i - level_start);
    if (level == "WARN") {
        out.level = LogLevel::Warn;
    } else if (level == "ERROR" || level == "FATAL") {
        out.level = LogLevel::Error;
    } else if (level != "INFO" && level != "DEBUG" && level != "TRACE") {
        out.message = std::string(line.substr(level_start));
        return out;
    }
    while (i < line.size() && line[i] == ' ') { ++i; }
    out.message = std::string(line.substr(i));
    return out;
}

} // namespace infernix::monitor
