// infernix-monitor's readers: the Prometheus text the engine serves, the request-log records,
// the console log lines, and following a file through appends, a partial line and replacement;
// and its controls: the control command runner, the placement shares and the POST guard.

#include "control.h"
#include "file_tail.h"
#include "prometheus.h"
#include "records.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <unistd.h>

using namespace infernix::monitor;

namespace {

int failures = 0;

void check(bool pass, const char* message) {
    if (!pass) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void append(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out << text;
}

void test_prometheus() {
    const Metrics metrics(parse_prometheus(
        "# HELP infernix_decode_tokens_total Committed decode tokens.\n"
        "# TYPE infernix_decode_tokens_total counter\n"
        "infernix_decode_tokens_total 1234\n"
        "infernix_model_info{model_name=\"Qwen \\\"Flash\\\"\",speculative_backend=\"dflash2\"} 1\n"
        "infernix_spec_decode_draft_tokens_total{backend=\"dflash2\"} 300\n"
        "infernix_spec_decode_draft_tokens_total{backend=\"ngram\"} 20\r\n"
        "infernix_ttft_seconds_bucket{le=\"+Inf\"} 7\n"
        "infernix_engine_ready 1e0\n"
        "broken{label=\"unterminated} 3\n"
        "not_a_number abc\n"));
    check(metrics.sum("infernix_decode_tokens_total") == 1234.0, "plain counter");
    check(metrics.sum("infernix_spec_decode_draft_tokens_total") == 320.0, "sum over labels");
    check(metrics.sum("infernix_spec_decode_draft_tokens_total", "backend", "ngram") == 20.0, "label filter");
    check(metrics.sum("infernix_ttft_seconds_bucket", "le", "+Inf") == 7.0, "+Inf label value");
    check(metrics.sum("infernix_engine_ready") == 1.0, "exponent value");
    check(!metrics.sum("broken") && !metrics.sum("not_a_number"), "malformed lines are skipped");
    check(!metrics.sum("infernix_missing"), "missing metric");
    check(metrics.label("infernix_model_info", "model_name") == "Qwen \"Flash\"", "escaped label value");
    check(metrics.label("infernix_model_info", "speculative_backend") == "dflash2", "second label");
}

void test_request_log() {
    const auto done = parse_request_log_line(
        R"({"event":"request_done","timestamp_unix_ms":1700000000500,"server_instance_id":"abc",)"
        R"("request":{"request_id":42,"protocol":"openai_chat"},)"
        R"("result":{"finish_reason":"length","prompt_tokens":1000,"completion_tokens":256,)"
        R"("computed_prefill_tokens":200,"prefix_cache_hit_tokens":800},)"
        R"("timings_seconds":{"ttft":0.25,"prefill":0.2,"decode":2.0,"total":2.3},)"
        R"("speculative":{"drafted_tokens":400,"accepted_tokens":300}})");
    check(done && done->event == RequestLogEvent::Finished, "request_done is a finished request");
    if (done) {
        const FinishedRequest& r = done->request;
        check(done->server_instance == "abc", "server instance");
        check(std::abs(r.time - 1700000000.5) < 1e-6, "record time in seconds");
        check(r.id == 42 && r.protocol == "openai_chat" && r.finish == "length", "request identity");
        check(r.prompt_tokens == 1000 && r.completion_tokens == 256 && r.computed_prefill_tokens == 200 &&
                  r.prefix_hit_tokens == 800,
              "token counts");
        check(r.decode_s == 2.0 && r.prefill_s == 0.2 && r.ttft_s == 0.25 && r.total_s == 2.3, "timings");
        check(r.drafted_tokens == 400 && r.accepted_tokens == 300, "speculative counts");
    }
    const auto error = parse_request_log_line(
        R"({"event":"request_error","timestamp_unix_ms":5,"request":{"request_id":7},"error":{"message":"x"}})");
    check(error && error->event == RequestLogEvent::Finished && error->request.finish == "error" &&
              error->request.id == 7,
          "request_error is a finished request with finish error");
    const auto throughput = parse_request_log_line(R"({"event":"throughput","server_instance_id":"abc"})");
    check(throughput && throughput->event == RequestLogEvent::Other && throughput->server_instance == "abc",
          "other events keep their server instance");
    check(!parse_request_log_line("not json"), "a line that is not JSON");
    const auto odd = parse_request_log_line(R"({"event":"request_done","result":{"prompt_tokens":"many"}})");
    check(odd && odd->request.prompt_tokens == 0, "a mistyped field reads as zero");
}

void test_log_lines() {
    const LogLine warn = parse_log_line("2026-11-17 18:41:36.591  WARN  engine | prefill chunk 16384");
    check(warn.time.has_value() && warn.level == LogLevel::Warn && warn.message == "engine | prefill chunk 16384",
          "WARN line");
    const LogLine info = parse_log_line("\x1b[32m2026-11-17 18:41:43.311  INFO \x1b[0m engine ready | model");
    check(info.time.has_value() && info.level == LogLevel::Info && info.message == "engine ready | model",
          "coloured INFO line");
    check(info.time && warn.time && std::abs(*info.time - *warn.time - 6.72) < 1e-6, "timestamps keep milliseconds");
    const LogLine error = parse_log_line("2026-11-17 18:41:43.311  ERROR req#3 failed");
    check(error.level == LogLevel::Error && error.message == "req#3 failed", "ERROR line");
    const LogLine banner = parse_log_line("=== Infernix ===");
    check(!banner.time && banner.level == LogLevel::Info && banner.message == "=== Infernix ===",
          "a line without a timestamp stays whole");
}

void test_file_tail() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("infernix_monitor_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const auto first = dir / "a.log", second = dir / "b.log", link = dir / "current.log";

    append(first, "old line 1\nold line 2\nold line 3\n");
    std::filesystem::create_symlink(first, link);
    FileTail backlog(link, 14); // the last 14 bytes start inside "old line 2"
    auto lines = backlog.read_lines();
    check(lines.size() == 1 && lines[0] == "old line 3", "backlog drops the line it starts inside");

    FileTail tail(link, 1 << 20);
    lines = tail.read_lines();
    check(lines.size() == 3 && !tail.reopened(), "first read returns the whole small file");
    append(first, "new 1\npart");
    lines = tail.read_lines();
    check(lines.size() == 1 && lines[0] == "new 1", "a partial line waits");
    append(first, "ial\r\n");
    lines = tail.read_lines();
    check(lines.size() == 1 && lines[0] == "partial", "the partial line completes, CR removed");
    check(tail.read_lines().empty(), "nothing new");

    append(second, "fresh\n");
    std::filesystem::remove(link);
    std::filesystem::create_symlink(second, link);
    lines = tail.read_lines();
    check(tail.reopened() && lines.size() == 1 && lines[0] == "fresh", "a repointed symlink reads the new file");

    std::filesystem::resize_file(second, 0);
    check(tail.read_lines().empty(), "a truncated file reads nothing");
    append(second, "after\n");
    lines = tail.read_lines();
    check(lines.size() == 1 && lines[0] == "after", "a truncated file is read from its start");

    std::filesystem::remove_all(dir);
    FileTail missing(dir / "none.log", 100);
    check(missing.read_lines().empty(), "a missing file reads nothing");
}

void test_run_command() {
    std::vector<std::string> lines;
    const LineSink collect = [&](const std::string& line, bool) { lines.push_back(line); };
    CommandResult r = run_command({"/bin/sh", "-c", "echo one; echo two >&2; printf 'three\\r\\nfour'; exit 3"},
                                  std::chrono::seconds(10), collect);
    check(r.exit_code == 3 && !r.timed_out, "the command's exit status");
    check(lines == std::vector<std::string>{"one", "two", "three", "four"},
          "stdout and stderr lines, CR removed, the unterminated last line kept");
    check(r.last_line == "four", "the last line");

    // The monitor's own descriptors (its listening socket) must not reach the server a start launches.
    const int fd = ::dup2(::open("/dev/null", O_RDONLY), 57);
    lines.clear();
    r = run_command({"/bin/sh", "-c", "test -e /proc/$$/fd/57 && echo open || echo closed"}, std::chrono::seconds(10),
                    collect);
    ::close(fd);
    check(r.exit_code == 0 && lines == std::vector<std::string>{"closed"}, "no inherited descriptor");

    const auto begin = std::chrono::steady_clock::now();
    r = run_command({"/bin/sh", "-c", "echo waiting; sleep 30"}, std::chrono::seconds(1), collect);
    const auto took = std::chrono::steady_clock::now() - begin;
    check(r.timed_out && r.exit_code == 128 + 9 && took < std::chrono::seconds(5),
          "past the timeout the command and its children are killed");

    const auto detached = std::chrono::steady_clock::now();
    r = run_command({"/bin/sh", "-c", "(sleep 4 &); echo left"}, std::chrono::seconds(10), collect);
    check(r.exit_code == 0 && !r.timed_out && std::chrono::steady_clock::now() - detached < std::chrono::seconds(3),
          "a program the command leaves holding the output does not hold the result");

    r = run_command({"/nonexistent/command"}, std::chrono::seconds(10), collect);
    check(r.exit_code == 127, "a command that cannot run exits 127");
}

void test_placement_shares() {
    check(placement_body(100, 0) == nlohmann::json{{"vram", "all"}}, "100 % VRAM is vram=all");
    check(placement_body(60, 25) == nlohmann::json{{"vram", "60%"}, {"ram", "25%"}, {"ssd", "15%"}},
          "the shares add up to 100 %");
    check(!placement_body(0, 50) && !placement_body(60, 41) && !placement_body(101, 0) && !placement_body(50, -1),
          "VRAM below 1 %, a sum past 100 % or a negative share is refused");
    check(experts_spec(100, 0) == std::string() && experts_spec(40, 60) == "vram=40%,ram=60%,ssd=0%",
          "the start spec: empty for the engine's default");
}

void test_page_request() {
    const std::vector<std::string> names{"box", "box.local"};
    const char* json = "application/json";
    check(page_request_allowed("127.0.0.1:8090", "", json, names), "loopback, no Origin");
    check(page_request_allowed("localhost:8090", "http://localhost:8090", json, names), "the page's own Origin");
    check(page_request_allowed("[::1]:8090", "http://[::1]:8090", "application/json; charset=utf-8", names), "IPv6");
    check(page_request_allowed("Box:8090", "", json, names), "the machine's name");
    check(!page_request_allowed("127.0.0.1:8090", "http://evil.example", json, names), "another site's Origin");
    check(!page_request_allowed("evil.example:8090", "http://evil.example:8090", json, names),
          "a DNS name rebound to this address");
    check(!page_request_allowed("127.0.0.1:8090", "", "text/plain", names), "a form or text POST");
    check(!page_request_allowed("", "", json, names), "no Host");
}

} // namespace

int main() {
    test_prometheus();
    test_request_log();
    test_log_lines();
    test_file_tail();
    test_run_command();
    test_placement_shares();
    test_page_request();
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
