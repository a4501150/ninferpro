#pragma once

// The two engine files the monitor reads: the --request-log-jsonl records and the console log.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace infernix::monitor {

// One finished request, from a request_done or request_error record.
struct FinishedRequest {
    double time = 0.0; // Unix seconds the record was written
    std::uint64_t id = 0;
    std::string protocol;
    std::string finish; // the finish reason, or "error"
    std::uint64_t prompt_tokens = 0, completion_tokens = 0;
    std::uint64_t computed_prefill_tokens = 0, prefix_hit_tokens = 0;
    double ttft_s = 0.0, prefill_s = 0.0, decode_s = 0.0, total_s = 0.0;
    std::uint64_t drafted_tokens = 0, accepted_tokens = 0;
};

enum class RequestLogEvent { Other, Finished };

struct RequestLogRecord {
    RequestLogEvent event = RequestLogEvent::Other;
    std::string server_instance;
    FinishedRequest request; // set for Finished
};

// nullopt for a line that is not a JSON object.
std::optional<RequestLogRecord> parse_request_log_line(std::string_view line);

enum class LogLevel { Info, Warn, Error };

struct LogLine {
    std::optional<double> time; // Unix seconds; absent on a line without a timestamp
    LogLevel level = LogLevel::Info;
    std::string message;
};

// Reads `YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message` (local time). A line in another form (a
// continuation, a launcher banner) is kept whole as an Info message. ANSI colour codes are removed.
LogLine parse_log_line(std::string_view line);

} // namespace infernix::monitor
