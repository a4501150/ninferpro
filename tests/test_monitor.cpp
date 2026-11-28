// infernix-monitor's readers, TOML profile catalog, reserve-based placement policy, native
// server lifecycle controller (load/switch/unload/fast-fail), and POST CSRF guard.

#include "control.h"
#include "file_tail.h"
#include "profiles.h"
#include "prometheus.h"
#include "records.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

int pick_free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return 18490; }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    int port             = 18490;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
        socklen_t len = sizeof(addr);
        if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) { port = ntohs(addr.sin_port); }
    }
    ::close(fd);
    return port;
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
        std::filesystem::temp_directory_path() / ("infernix_monitor_test_tail_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
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

void test_profiles() {
    const auto def = parse_experts_shares("");
    check(def && def->vram == 100 && def->ram == 0, "empty experts defaults to 100/0");
    const auto g40 = parse_experts_shares("vram=40%,ram=60%");
    check(g40 && g40->vram == 40 && g40->ram == 60, "vram=40%,ram=60% parses");
    const auto mix = parse_experts_shares("vram=65%,ram=15%,ssd=20%");
    check(mix && mix->vram == 65 && mix->ram == 15, "three-tier percentages parse");
    const auto all = parse_experts_shares("vram=all");
    check(all && all->vram == 100 && all->ram == 0, "vram=all parses to 100/0");
    check(!parse_experts_shares("vram=50%,ram=60%"), "percentages past 100 refused");
    check(!parse_experts_shares("vram=1000,ram=500"), "raw counts return nullopt");

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("infernix_monitor_test_prof_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto dummy_model = dir / "dummy.infernix";
    write_file(dummy_model, "x");
    write_file(dir / "Beta.toml",
               "model = \"" + dummy_model.string() + "\"\n"
               "kv-dtype = \"int8\"\n"
               "spec = \"dflash2\"\n"
               "[serve]\n"
               "served-model-name = [\"beta-alias\"]\n"
               "experts = \"vram=40%,ram=60%\"\n"
               "max-context = 262144\n"
               "max-concurrency = 4\n"
               "port = 8123\n");
    write_file(dir / "Alpha.toml",
               "model = \"" + (dir / "missing.infernix").string() + "\"\n"
               "[serve]\n"
               "port = 8000\n");
    write_file(dir / ".hidden.toml", "model = \"ignored\"\n");
    write_file(dir / "Broken.toml", "port = [unterminated\n");

    const auto profiles = scan_profiles(dir);
    check(profiles.size() == 3, "scans non-hidden *.toml profiles sorted alphabetically");
    if (profiles.size() == 3) {
        check(profiles[0].name == "Alpha" && !profiles[0].artifact_exists && profiles[0].default_shares &&
                  profiles[0].default_shares->vram == 100,
              "Alpha profile missing artifact and defaults to 100% VRAM");
        check(profiles[1].name == "Beta" && profiles[1].artifact_exists && profiles[1].served_name == "beta-alias" &&
                  profiles[1].port == 8123 && profiles[1].kv_dtype == "int8" && profiles[1].spec == "dflash2" &&
                  profiles[1].max_context == 262144 && profiles[1].max_concurrency == 4 &&
                  profiles[1].default_shares && profiles[1].default_shares->vram == 40 &&
                  profiles[1].default_shares->ram == 60,
              "Beta profile reads top-level and [serve] metadata accurately");
        check(profiles[2].name == "Broken" && !profiles[2].parse_error.empty(), "Broken profile captures parse error");
    }
    std::filesystem::remove_all(dir);
}

void test_placement_shares() {
    check(placement_body(100, 0) == nlohmann::json{{"vram", "all"}}, "100 % VRAM is vram=all");
    check(placement_body(60, 25) == nlohmann::json{{"vram", "60%"}, {"ram", "25%"}, {"ssd", "15%"}},
          "the shares add up to 100 %");
    check(!placement_body(0, 50) && !placement_body(60, 41) && !placement_body(101, 0) && !placement_body(50, -1),
          "VRAM below 1 %, a sum past 100 % or a negative share is refused");
    check(experts_spec(100, 0) == "vram=100%" && experts_spec(40, 60) == "vram=40%,ram=60%,ssd=0%",
          "explicit --experts override at 100% and mixed tiers");
}

void test_reserve_plan() {
    constexpr std::uint64_t kGiB = 1ULL << 30;
    const nlohmann::json report{
        {"supported", true},
        {"total_experts", 1000},
        {"mechanism",
         {{"total_experts", 1000},
          {"vram", {{"bytes", 60 * kGiB}, {"frame_bytes", 100ULL << 20}}}, // 100 MiB/expert -> 10.24/GiB
          {"ram", {{"record_bytes", 100ULL << 20}}},
          {"vram_free", 10 * kGiB},        // 60 + 10 = 70 GiB total expert budget
          {"host_available", 50 * kGiB}}}, // 50 GiB available RAM
    };
    std::string err;
    // Reserve 32 GiB VRAM -> 38 GiB left -> floor(38 * 1024 / 100) = 389 experts in VRAM.
    // Reserve 24 GiB RAM -> 26 GiB left -> floor(26 * 1024 / 100) = 266 experts in RAM, 345 on SSD.
    const auto gaming = compute_reserve_plan(report, 32.0, 24.0, &err);
    check(gaming.has_value(), "gaming reserve plan computed");
    if (gaming) {
        check(gaming->vram_experts == 389 && gaming->ram_experts == 266 && gaming->ssd_experts == 345,
              "gaming reserve counts match exact VRAM/RAM reserve math");
        check(gaming->body == nlohmann::json{{"vram", 389}, {"ram", 266}}, "gaming reserve POST body");
    }
    // Huge budget with 0 reserve -> fits all 1000 experts in VRAM -> {"vram": "all"}.
    nlohmann::json roomy                    = report;
    roomy["mechanism"]["vram_free"]         = 100 * kGiB;
    const auto full = compute_reserve_plan(roomy, 0.0, 0.0, &err);
    check(full && full->vram_experts == 1000 && full->ram_experts == 0 &&
              full->body == nlohmann::json{{"vram", "all"}},
          "full VRAM fit collapses to vram=all");

    check(!compute_reserve_plan(nlohmann::json{{"supported", false}}, 32.0, 24.0, &err),
          "unsupported model refused");
    check(!compute_reserve_plan(report, -1.0, 0.0, &err), "negative reserve refused");
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

void test_native_controller() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("infernix_monitor_test_ctl_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "configs");
    std::filesystem::create_directories(dir / "logs");

    const auto dummy_model = dir / "mock.infernix";
    write_file(dummy_model, "mock");
    const int port = pick_free_port();

    write_file(dir / "configs" / "MockA.toml",
               "model = \"" + dummy_model.string() + "\"\n[serve]\nport = " + std::to_string(port) + "\n");
    write_file(dir / "configs" / "MockB.toml",
               "model = \"" + dummy_model.string() + "\"\n[serve]\nport = " + std::to_string(port) + "\n");
    write_file(dir / "configs" / "MockCrash.toml",
               "model = \"" + dummy_model.string() + "\"\n[serve]\nport = " + std::to_string(port) + "\n");

    const auto args_dump = dir / "last_args.txt";
    const auto mock_bin  = dir / "mock-infernix-serve";
    write_file(
        mock_bin,
        "#!/usr/bin/env python3\n"
        "import http.server, os, signal, sys\n"
        "args = sys.argv[1:]\n"
        "with open(" + nlohmann::json(args_dump.string()).dump() + ", 'w') as f:\n"
        "    f.write('\\n'.join(args) + '\\n')\n"
        "    f.write('COREDUMP=' + os.environ.get('CUDA_ENABLE_COREDUMP_ON_EXCEPTION', '') + '\\n')\n"
        "    f.write('FD57=' + ('open' if os.path.exists('/proc/self/fd/57') else 'closed') + '\\n')\n"
        "if any('MockCrash.toml' in a for a in args):\n"
        "    print('FATAL: simulated crash', flush=True)\n"
        "    sys.exit(2)\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'{\"data\":[]}')\n"
        "    def log_message(self, *a): pass\n"
        "http.server.HTTPServer.allow_reuse_address = True\n"
        "srv = http.server.HTTPServer(('127.0.0.1', " + std::to_string(port) + "), H)\n"
        "signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
        "print('server ready on port " + std::to_string(port) + "', flush=True)\n"
        "srv.serve_forever()\n");
    ::chmod(mock_bin.c_str(), 0755);

    ManagerConfig cfg;
    cfg.configs_dir     = dir / "configs";
    cfg.server_bin      = mock_bin;
    cfg.work_dir        = dir;
    cfg.log_dir         = dir / "logs";
    cfg.default_profile = "MockA";
    cfg.load_timeout    = std::chrono::seconds(15);
    cfg.unload_timeout  = std::chrono::seconds(10);

    const auto wait_idle = [](const Controller& ctl) {
        for (int i = 0; i < 150; ++i) {
            const auto st = ctl.state();
            if (st["busy"].is_null()) { return st; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return ctl.state();
    };

    const int extra_fd = ::dup2(::open("/dev/null", O_RDONLY), 57);
    {
        Controller ctl(cfg, [](const std::string&, bool) {});
        check(!ctl.load("MockA").has_value(), "load MockA queued");
        const auto st1 = wait_idle(ctl);
        check(st1["last"]["ok"] == true && st1["active_profile"] == "MockA" && st1["pid"].is_number_integer(),
              "MockA loaded and tracked");
        check(std::filesystem::is_symlink(cfg.log_dir / "latest.log"), "latest.log symlink created");
        check(std::filesystem::exists(cfg.log_dir / "infernix-serve.pid"), "pidfile written");

        // Verify env, request-log, and closed descriptor 57.
        std::ifstream in(args_dump);
        const std::string dumped((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        check(dumped.find("--request-log-jsonl") != std::string::npos, "request-log enabled by default");
        check(dumped.find("COREDUMP=1") != std::string::npos, "CUDA coredump env set");
        check(dumped.find("FD57=closed") != std::string::npos, "extra fd not inherited by server");

        // Switch directly from MockA to MockB with explicit --experts override.
        const pid_t pid_a = static_cast<pid_t>(st1["pid"].get<std::int64_t>());
        check(!ctl.load("MockB", "vram=100%").has_value(), "switch to MockB queued");
        const auto st2 = wait_idle(ctl);
        check(st2["last"]["ok"] == true && st2["active_profile"] == "MockB" &&
                  st2["pid"].get<std::int64_t>() != pid_a,
              "switched cleanly from MockA to MockB");
        std::ifstream in2(args_dump);
        const std::string dumped2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
        check(dumped2.find("--experts\nvram=100%") != std::string::npos, "explicit --experts passed on switch");

        // Unload cleanly.
        const pid_t pid_b = static_cast<pid_t>(st2["pid"].get<std::int64_t>());
        check(!ctl.unload().has_value(), "unload queued");
        const auto st3 = wait_idle(ctl);
        check(st3["last"]["ok"] == true && st3["active_profile"].is_null() && st3["pid"].is_null(),
              "unloaded cleanly");
        check(::kill(pid_b, 0) != 0, "unloaded server process exited");
        check(!std::filesystem::exists(cfg.log_dir / "infernix-serve.pid"), "pidfile removed on unload");

        // Fast-fail on premature server exit.
        const auto t0 = std::chrono::steady_clock::now();
        check(!ctl.load("MockCrash").has_value(), "load MockCrash queued");
        const auto st4 = wait_idle(ctl);
        const auto dt  = std::chrono::steady_clock::now() - t0;
        check(st4["last"]["ok"] == false && dt < std::chrono::seconds(5) &&
                  st4["last"]["message"].get<std::string>().find("simulated crash") != std::string::npos,
              "crashed server fails fast with last log line");
    }
    if (extra_fd >= 0) { ::close(extra_fd); }
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    test_prometheus();
    test_request_log();
    test_log_lines();
    test_file_tail();
    test_profiles();
    test_placement_shares();
    test_reserve_plan();
    test_page_request();
    test_native_controller();
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
