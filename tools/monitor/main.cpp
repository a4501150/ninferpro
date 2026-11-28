// infernix-monitor: first-class native server manager and live status page for infernix-serve.
// Starts idle by default (0 VRAM), discovers TOML serving profiles under --configs, spawns and
// unloads infernix-serve directly, applies placement and reserve policies, and provides both the
// web UI and CLI control subcommands (status, load, unload, placement, stop).

#include "monitor.h"
#include "monitor_web.h"
#include "product/config/config_file.h"

#include <fcntl.h>
#include <httplib.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using infernix::monitor::Monitor;
using infernix::monitor::MonitorOptions;
using Json = nlohmann::json;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

struct ServerCliOptions {
    MonitorOptions monitor;
    std::string host = "127.0.0.1";
    int port         = 8090;
    bool daemon      = false;
    bool autoload    = false;
    std::filesystem::path pid_file;
};

std::filesystem::path executable_path() {
    std::error_code ec;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path{} : exe;
}

std::filesystem::path detect_repo_root(const std::filesystem::path& exe) {
    std::error_code ec;
    auto has_configs = [&](const std::filesystem::path& dir) {
        const auto cfg = dir / "configs";
        if (!std::filesystem::is_directory(cfg, ec) || ec) { return false; }
        for (const auto& entry : std::filesystem::directory_iterator(cfg, ec)) {
            if (!ec && entry.is_regular_file(ec) && entry.path().extension() == ".toml") { return true; }
        }
        return false;
    };
    if (!exe.empty()) {
        std::filesystem::path cur = exe.parent_path();
        for (int i = 0; i < 5 && !cur.empty(); ++i) {
            if (has_configs(cur)) { return cur; }
            const auto parent = cur.parent_path();
            if (parent == cur) { break; }
            cur = parent;
        }
    }
    const auto cwd = std::filesystem::current_path(ec);
    if (!ec && has_configs(cwd)) { return cwd; }
    return cwd;
}

std::filesystem::path default_pid_file() {
    return std::filesystem::path(infernix::product::expand_user_path("~/.cache/ninferpro/monitor.pid"));
}

int parse_port_value(const std::string& text) {
    char* end       = nullptr;
    const long port = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || port < 1 || port > 65535) {
        std::cerr << "--port must be 1-65535\n";
        std::exit(2);
    }
    return static_cast<int>(port);
}

double parse_nonneg_double(const std::string& text, const char* flag) {
    char* end        = nullptr;
    const double val = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || !std::isfinite(val) || val < 0.0) {
        std::cerr << flag << " must be a non-negative number\n";
        std::exit(2);
    }
    return val;
}

void usage(std::ostream& out) {
    out << "usage:\n"
           "  infernix-monitor [options]              run the native manager and web UI (starts idle)\n"
           "  infernix-monitor status [--port N]      print live manager, GPU, and placement status\n"
           "  infernix-monitor load [PROFILE] [--experts SPEC] [--port N]\n"
           "                                          load (or switch to) a model profile and wait until ready\n"
           "  infernix-monitor unload [--port N]      unload the active model (0 VRAM) and wait until exited\n"
           "  infernix-monitor placement [--status | --shares SPEC | --reserve-vram-gb N --reserve-ram-gb M |\n"
           "                              --gaming | --full] [--port N]\n"
           "                                          inspect or apply runtime expert placement\n"
           "  infernix-monitor stop [--with-server] [--port N]\n"
           "                                          stop a background --daemon manager\n"
           "\n"
           "manager options:\n"
           "  --daemon                  detach to background (writes ~/.cache/ninferpro/monitor.pid)\n"
           "  --autoload                load the default profile immediately on startup\n"
           "  --configs DIR             directory of *.toml serving profiles (default: <repo>/configs)\n"
           "  --default NAME            default model profile (default: GLM-5.3-Flash)\n"
           "  --server-bin PATH         infernix-serve binary (default: sibling of this executable)\n"
           "  --log-dir DIR             directory for dated server logs and requests.jsonl\n"
           "  --host ADDR               address for the web UI and control API (default: 127.0.0.1)\n"
           "  --port N                  port for the web UI and control API (default: 8090)\n"
           "  --gaming-vram-gb N        VRAM GiB reserved by the Gaming preset (default: 32)\n"
           "  --gaming-ram-gb N         host RAM GiB reserved by the Gaming preset (default: 24)\n"
           "  --no-request-log          do not enable --request-log-jsonl automatically\n"
           "  --api-key KEY             engine --api-key sent as a bearer token\n"
           "  --engine URL              read-only mode against an external server (disables lifecycle control\n"
           "                            unless --configs is also given)\n"
           "  --log-file FILE           override console log file to follow\n"
           "  --request-log FILE        override request JSONL file to follow\n";
}

// Parses an --experts spec string ("vram=60%,ram=25%,ssd=15%" or "vram=all" or "vram=8000,ram=4000")
// into the JSON body POST /admin/expert-placement expects.
std::optional<Json> parse_shares_spec_body(std::string_view spec, std::string& error) {
    Json body      = Json::object();
    std::size_t at = 0;
    while (at < spec.size()) {
        const std::size_t comma = spec.find(',', at);
        std::string_view term   = spec.substr(at, comma == std::string_view::npos ? std::string_view::npos : comma - at);
        at                      = comma == std::string_view::npos ? spec.size() : comma + 1;
        const auto trim         = [](std::string_view s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) { s.remove_prefix(1); }
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) { s.remove_suffix(1); }
            return s;
        };
        term                 = trim(term);
        const std::size_t eq = term.find('=');
        if (eq == std::string_view::npos) {
            error = "bad placement term '" + std::string(term) + "' (expected tier=value)";
            return std::nullopt;
        }
        const std::string tier(trim(term.substr(0, eq)));
        const std::string val(trim(term.substr(eq + 1)));
        if (tier.empty() || val.empty()) {
            error = "bad placement term '" + std::string(term) + "'";
            return std::nullopt;
        }
        if (val == "all" || val.back() == '%') {
            body[tier] = val;
        } else {
            char* end             = nullptr;
            const unsigned long n = std::strtoul(val.c_str(), &end, 10);
            if (end == val.c_str() || *end != '\0') {
                error = "bad value '" + val + "' for tier '" + tier + "'";
                return std::nullopt;
            }
            body[tier] = n;
        }
    }
    if (body.empty()) {
        error = "empty placement spec";
        return std::nullopt;
    }
    return body;
}

ServerCliOptions parse_server_options(int argc, char** argv, int start_idx) {
    ServerCliOptions options;
    const auto exe  = executable_path();
    const auto root = detect_repo_root(exe);
    options.monitor.manager.configs_dir = root / "configs";
    options.monitor.manager.work_dir    = root;
    options.monitor.manager.log_dir     = root / "logs" / "infernix";
    options.monitor.manager.server_bin  = exe.empty() ? (root / "build" / "apps" / "infernix-serve")
                                                      : (exe.parent_path() / "infernix-serve");
    options.pid_file                    = default_pid_file();

    bool engine_explicit  = false;
    bool configs_explicit = false;
    bool log_dir_explicit = false;

    for (int i = start_idx; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << arg << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--daemon" || arg == "-d") {
            options.daemon = true;
        } else if (arg == "--autoload") {
            options.autoload = true;
        } else if (arg == "--configs") {
            options.monitor.manager.configs_dir = infernix::product::expand_user_path(value());
            configs_explicit                    = true;
        } else if (arg == "--default") {
            options.monitor.manager.default_profile = value();
        } else if (arg == "--server-bin") {
            options.monitor.manager.server_bin = infernix::product::expand_user_path(value());
        } else if (arg == "--log-dir") {
            options.monitor.manager.log_dir = infernix::product::expand_user_path(value());
            log_dir_explicit                = true;
        } else if (arg == "--no-request-log") {
            options.monitor.manager.enable_request_log = false;
        } else if (arg == "--gaming-vram-gb") {
            options.monitor.gaming_reserve_vram_gb = parse_nonneg_double(value(), "--gaming-vram-gb");
        } else if (arg == "--gaming-ram-gb") {
            options.monitor.gaming_reserve_ram_gb = parse_nonneg_double(value(), "--gaming-ram-gb");
        } else if (arg == "--engine") {
            options.monitor.engine_url = value();
            while (!options.monitor.engine_url.empty() && options.monitor.engine_url.back() == '/') {
                options.monitor.engine_url.pop_back();
            }
            engine_explicit = true;
        } else if (arg == "--api-key") {
            options.monitor.api_key         = value();
            options.monitor.manager.api_key = options.monitor.api_key;
        } else if (arg == "--request-log") {
            options.monitor.request_log = infernix::product::expand_user_path(value());
        } else if (arg == "--log-file") {
            options.monitor.log_file = infernix::product::expand_user_path(value());
        } else if (arg == "--host") {
            options.host = value();
        } else if (arg == "--port") {
            options.port = parse_port_value(value());
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            std::exit(0);
        } else {
            std::cerr << "unknown option " << arg << '\n';
            usage(std::cerr);
            std::exit(2);
        }
    }

    if (engine_explicit && !configs_explicit) { options.monitor.manage_server = false; }
    if (configs_explicit) {
        std::error_code ec;
        const auto abs_cfg = std::filesystem::weakly_canonical(options.monitor.manager.configs_dir, ec);
        if (!ec && !abs_cfg.parent_path().empty()) {
            options.monitor.manager.work_dir = abs_cfg.parent_path();
            if (!log_dir_explicit) { options.monitor.manager.log_dir = abs_cfg.parent_path() / "logs" / "infernix"; }
        }
    }
    return options;
}

httplib::Server* g_server = nullptr;

void on_signal(int) {
    if (g_server != nullptr) { g_server->stop(); }
}

void reply_raw(httplib::Response& res, int status, const std::string& body) {
    res.status = status;
    res.set_header("Cache-Control", "no-store");
    res.set_content(body, "application/json");
}

void reply(httplib::Response& res, int status, const Json& body) { reply_raw(res, status, body.dump()); }

// The page's shares: whole percents of the experts, the rest on SSD.
bool read_shares(const Json& body, int& vram, int& ram) {
    if (!body.contains("vram") || !body["vram"].is_number_integer()) { return false; }
    if (body.contains("ram") && !body["ram"].is_number_integer()) { return false; }
    vram = body["vram"].get<int>();
    ram  = body.value("ram", 0);
    return true;
}

void write_daemon_pid(const std::filesystem::path& pid_file, pid_t pid) {
    std::error_code ec;
    std::filesystem::create_directories(pid_file.parent_path(), ec);
    std::ofstream out(pid_file, std::ios::binary | std::ios::trunc);
    if (out) { out << pid << '\n'; }
}

std::string read_proc_cmdline(pid_t pid) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/cmdline", std::ios::binary);
    if (!in) { return {}; }
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (char& c : raw) {
        if (c == '\0') { c = ' '; }
    }
    return raw;
}

// ---- CLI client helpers ----

Json cli_get_state(int port) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(5, 0);
    const auto res = client.Get("/api/state");
    if (!res) {
        std::cerr << "cannot reach infernix-monitor on http://127.0.0.1:" << port << " ("
                  << httplib::to_string(res.error()) << ")\n";
        std::exit(1);
    }
    if (res->status != 200) {
        std::cerr << "GET /api/state failed: HTTP " << res->status << " " << res->body << '\n';
        std::exit(1);
    }
    const Json j = Json::parse(res->body, nullptr, false);
    if (!j.is_object()) {
        std::cerr << "invalid JSON from infernix-monitor\n";
        std::exit(1);
    }
    return j;
}

std::pair<int, Json> cli_post(int port, const char* path, const Json& body, int read_timeout_s = 30) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(read_timeout_s, 0);
    const auto res = client.Post(path, body.dump(), "application/json");
    if (!res) {
        std::cerr << "cannot reach infernix-monitor on http://127.0.0.1:" << port << " ("
                  << httplib::to_string(res.error()) << ")\n";
        std::exit(1);
    }
    Json parsed = Json::parse(res->body, nullptr, false);
    if (!parsed.is_object()) { parsed = Json{{"raw", res->body}}; }
    return {res->status, std::move(parsed)};
}

void print_placement_report(const Json& pl) {
    if (!pl.is_object() || !pl.value("supported", false)) {
        std::cout << "placement:         not active\n";
        return;
    }
    const Json& mech = pl.contains("mechanism") && pl["mechanism"].is_object() ? pl["mechanism"] : Json{};
    const auto total = mech.value("total_experts", std::uint64_t{0});
    const Json& vram = mech.contains("vram") && mech["vram"].is_object() ? mech["vram"] : Json{};
    const Json& ram  = mech.contains("ram") && mech["ram"].is_object() ? mech["ram"] : Json{};
    const Json& ssd  = mech.contains("ssd") && mech["ssd"].is_object() ? mech["ssd"] : Json{};
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "total experts:     " << total << '\n';
    std::cout << "vram:              " << vram.value("experts", 0) << " experts  "
              << (static_cast<double>(vram.value("bytes", std::uint64_t{0})) / kGiB) << " GiB ("
              << vram.value("pinned", 0) << " pinned, frame " << vram.value("frame_bytes", 0) << " B)\n";
    std::cout << "ram:               " << ram.value("experts", 0) << " experts  in " << ram.value("slots", 0)
              << " slots (record " << ram.value("record_bytes", 0) << " B)\n";
    std::cout << "ssd:               " << ssd.value("experts", 0) << " experts\n";
    std::cout << std::setprecision(1);
    std::cout << "vram free/total:   " << (static_cast<double>(mech.value("vram_free", std::uint64_t{0})) / kGiB)
              << " / " << (static_cast<double>(mech.value("vram_total", std::uint64_t{0})) / kGiB) << " GiB\n";
    std::cout << "host available:    " << (static_cast<double>(mech.value("host_available", std::uint64_t{0})) / kGiB)
              << " GiB\n";
    if (pl.value("switching", false)) {
        std::cout << "switching:         target " << pl.value("target", 0) << " frames (fill "
                  << pl.value("fill_seeded", 0) << "/" << pl.value("fill_total", 0) << ")\n";
    }
}

int cmd_status(int argc, char** argv) {
    int port = 8090;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = parse_port_value(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            return 0;
        } else {
            std::cerr << "unknown option " << arg << '\n';
            return 2;
        }
    }
    const Json state = cli_get_state(port);
    const Json& live = state["live"];
    const Json& eng  = state["engine"];
    const Json& ctl  = state["control"];
    std::cout << "manager:           http://127.0.0.1:" << port << '\n';
    std::cout << "state:             " << live.value("state", "unknown") << '\n';
    std::cout << "engine:            " << eng.value("url", "")
              << (eng.value("reachable", false) ? " (reachable)" : " (not running)") << '\n';
    if (ctl.contains("active_profile") && ctl["active_profile"].is_string()) {
        std::cout << "active profile:    " << ctl["active_profile"].get<std::string>();
        if (ctl.contains("pid") && ctl["pid"].is_number_integer()) {
            std::cout << " (PID " << ctl["pid"].get<std::int64_t>() << ")";
        }
        std::cout << '\n';
    } else if (eng.contains("model") && eng["model"].is_string() && !eng["model"].get<std::string>().empty()) {
        std::cout << "model:             " << eng["model"].get<std::string>() << '\n';
    }
    if (state.contains("default_profile") && state["default_profile"].is_string()) {
        std::cout << "default profile:   " << state["default_profile"].get<std::string>() << '\n';
    }
    if (state.contains("profiles") && state["profiles"].is_array() && !state["profiles"].empty()) {
        std::cout << "profiles:          ";
        bool first = true;
        for (const Json& p : state["profiles"]) {
            if (!first) { std::cout << ", "; }
            first = false;
            std::cout << p.value("name", "");
            if (!p.value("artifact_exists", false)) { std::cout << " (missing)"; }
        }
        std::cout << '\n';
    }
    if (state.contains("hardware") && state["hardware"].contains("gpus") && state["hardware"]["gpus"].is_array()) {
        for (const Json& g : state["hardware"]["gpus"]) {
            if (g.contains("mem_used") && g["mem_used"].is_number() && g.contains("mem_total") &&
                g["mem_total"].is_number()) {
                std::cout << std::fixed << std::setprecision(1) << "gpu " << g.value("index", 0) << " vram:        "
                          << (g["mem_used"].get<double>() / kGiB) << " / " << (g["mem_total"].get<double>() / kGiB)
                          << " GiB\n";
            }
        }
    }
    if (state.contains("placement") && state["placement"].is_object()) { print_placement_report(state["placement"]); }
    return 0;
}

int wait_control_action(int port, const std::string& action_label) {
    std::string last_busy;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const Json state = cli_get_state(port);
        const Json& ctl  = state["control"];
        if (ctl.contains("busy") && ctl["busy"].is_string()) {
            const std::string busy = ctl["busy"].get<std::string>();
            if (busy != last_busy) {
                std::cout << busy << "..." << std::endl;
                last_busy = busy;
            }
            continue;
        }
        const Json& last = ctl["last"];
        if (last.is_object()) {
            const bool ok       = last.value("ok", false);
            const std::string m = last.value("message", "");
            std::cout << action_label << ": " << (ok ? "ok" : "failed") << (m.empty() ? "" : " — " + m) << '\n';
            return ok ? 0 : 1;
        }
        return 0;
    }
}

int cmd_load(int argc, char** argv) {
    int port = 8090;
    std::string profile;
    std::string experts;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = parse_port_value(argv[++i]);
        } else if (arg == "--experts" && i + 1 < argc) {
            experts = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            return 0;
        } else if (!arg.starts_with("-") && profile.empty()) {
            profile = arg;
        } else {
            std::cerr << "unknown argument " << arg << '\n';
            return 2;
        }
    }
    Json body{{"action", "load"}};
    if (!profile.empty()) { body["profile"] = profile; }
    if (!experts.empty()) { body["experts"] = experts; }
    const auto [status, reply_json] = cli_post(port, "/api/control", body);
    if (status != 202) {
        std::cerr << "load failed (HTTP " << status << "): " << reply_json.value("error", reply_json.dump()) << '\n';
        return 1;
    }
    return wait_control_action(port, profile.empty() ? "load" : "load " + profile);
}

int cmd_unload(int argc, char** argv) {
    int port = 8090;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = parse_port_value(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            return 0;
        } else {
            std::cerr << "unknown option " << arg << '\n';
            return 2;
        }
    }
    const auto [status, reply_json] = cli_post(port, "/api/control", Json{{"action", "unload"}});
    if (status != 202) {
        std::cerr << "unload failed (HTTP " << status << "): " << reply_json.value("error", reply_json.dump()) << '\n';
        return 1;
    }
    return wait_control_action(port, "unload");
}

int cmd_placement(int argc, char** argv) {
    int port       = 8090;
    bool do_status = false;
    bool do_gaming = false;
    bool do_full   = false;
    std::string shares_spec;
    std::optional<double> reserve_vram;
    std::optional<double> reserve_ram;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need       = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << arg << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--port") {
            port = parse_port_value(need());
        } else if (arg == "--status") {
            do_status = true;
        } else if (arg == "--gaming") {
            do_gaming = true;
        } else if (arg == "--full") {
            do_full = true;
        } else if (arg == "--shares") {
            shares_spec = need();
        } else if (arg == "--reserve-vram-gb") {
            reserve_vram = parse_nonneg_double(need(), "--reserve-vram-gb");
        } else if (arg == "--reserve-ram-gb") {
            reserve_ram = parse_nonneg_double(need(), "--reserve-ram-gb");
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            return 0;
        } else {
            std::cerr << "unknown option " << arg << '\n';
            return 2;
        }
    }

    if (do_status || (!do_gaming && !do_full && shares_spec.empty() && !reserve_vram && !reserve_ram)) {
        const Json state = cli_get_state(port);
        if (!state.contains("placement") || !state["placement"].is_object()) {
            std::cerr << "no active model with expert placement\n";
            return 1;
        }
        print_placement_report(state["placement"]);
        return 0;
    }

    Json req = Json::object();
    if (do_gaming) {
        const Json state  = cli_get_state(port);
        const Json& g     = state["gaming_reserve"];
        req["reserve_vram_gb"] = g.value("vram_gb", 32.0);
        req["reserve_ram_gb"]  = g.value("ram_gb", 24.0);
    } else if (do_full) {
        req["vram"] = 100;
        req["ram"]  = 0;
    } else if (!shares_spec.empty()) {
        req["shares"] = shares_spec;
    } else {
        req["reserve_vram_gb"] = reserve_vram.value_or(0.0);
        req["reserve_ram_gb"]  = reserve_ram.value_or(0.0);
    }

    Json dry_req       = req;
    dry_req["dry_run"] = true;
    const auto [d_status, preview] = cli_post(port, "/api/placement", dry_req);
    if (d_status != 200) {
        std::cerr << "placement dry-run failed (HTTP " << d_status
                  << "): " << preview.value("error", preview.dump()) << '\n';
        return 1;
    }
    const Json& pl = preview["placement"];
    const Json& pr = preview["predicted"];
    std::cout << std::fixed << std::setprecision(1) << "dry run: vram=" << pl.value("vram", 0)
              << " ram=" << pl.value("ram", 0) << " ssd=" << pl.value("ssd", 0) << " -> vram "
              << (static_cast<double>(pr.value("vram_expert_bytes", std::uint64_t{0})) / kGiB) << " GiB (free after "
              << (static_cast<double>(pr.value("vram_free_after", std::uint64_t{0})) / kGiB) << " GiB), ram "
              << (static_cast<double>(pr.value("ram_expert_bytes", std::uint64_t{0})) / kGiB) << " GiB\n";
    if (!pr.value("fits", true)) {
        std::cerr << "the placement does not fit in VRAM; nothing applied\n";
        return 1;
    }
    req["dry_run"]                 = false;
    const auto [a_status, applied] = cli_post(port, "/api/placement", req);
    if (a_status != 200) {
        std::cerr << "placement failed (HTTP " << a_status << "): " << applied.value("error", applied.dump()) << '\n';
        return 1;
    }
    std::cout << "applied: " << applied.dump() << '\n';
    return 0;
}

int cmd_stop(int argc, char** argv) {
    int port         = 8090;
    bool with_server = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = parse_port_value(argv[++i]);
        } else if (arg == "--with-server") {
            with_server = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            return 0;
        } else {
            std::cerr << "unknown option " << arg << '\n';
            return 2;
        }
    }
    if (with_server) {
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(1, 0);
        client.set_read_timeout(5, 0);
        if (const auto res = client.Post("/api/control", Json{{"action", "unload"}}.dump(), "application/json");
            res && res->status == 202) {
            (void)wait_control_action(port, "unload");
        }
    }
    const auto pid_file = default_pid_file();
    std::ifstream in(pid_file);
    if (!in) {
        std::cout << "infernix-monitor is not running (no pidfile).\n";
        return 0;
    }
    pid_t pid = -1;
    in >> pid;
    if (pid <= 0 || ::kill(pid, 0) != 0) {
        std::error_code ec;
        std::filesystem::remove(pid_file, ec);
        std::cout << "infernix-monitor is not running.\n";
        return 0;
    }
    const std::string cmdline = read_proc_cmdline(pid);
    if (cmdline.find("infernix-monitor") == std::string::npos) {
        std::cerr << "PID " << pid << " in " << pid_file << " is not infernix-monitor; refusing to signal it\n";
        return 1;
    }
    ::kill(pid, SIGTERM);
    for (int i = 0; i < 50; ++i) {
        if (::kill(pid, 0) != 0 && errno == ESRCH) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::error_code ec;
    std::filesystem::remove(pid_file, ec);
    std::cout << "infernix-monitor stopped (PID " << pid << ").\n";
    return 0;
}

int run_server(const ServerCliOptions& options, int ready_write_fd) {
    const auto report_ready = [&](const std::string& msg) {
        if (ready_write_fd >= 0) {
            [[maybe_unused]] const auto ignored = ::write(ready_write_fd, msg.data(), msg.size());
            ::close(ready_write_fd);
            ready_write_fd = -1;
        }
    };

    Monitor monitor(options.monitor);
    monitor.tick(); // populate initial state immediately before accepting requests

    httplib::Server server;
    const auto serve = [&](const char* path, std::string_view body, const char* type) {
        server.Get(path, [body, type](const httplib::Request&, httplib::Response& res) {
            res.set_content(body.data(), body.size(), type);
        });
    };
    serve("/", infernix::monitor::web::kIndexHtml, "text/html; charset=utf-8");
    serve("/app.js", infernix::monitor::web::kAppJs, "text/javascript; charset=utf-8");
    serve("/app.css", infernix::monitor::web::kAppCss, "text/css; charset=utf-8");
    server.Get("/api/state", [&](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
        res.set_content(monitor.state_json(), "application/json");
    });

    // What a POST's Host may name besides IP literals and localhost.
    std::vector<std::string> names;
    if (char name[256] = {}; gethostname(name, sizeof(name) - 1) == 0 && name[0] != '\0') {
        names.push_back(name);
        names.push_back(std::string(name) + ".local");
    }
    if (options.host != "0.0.0.0" && options.host != "::") { names.push_back(options.host); }

    const auto post = [&](const char* path, auto handler) {
        server.Post(path, [&, handler](const httplib::Request& req, httplib::Response& res) {
            if (!infernix::monitor::page_request_allowed(req.get_header_value("Host"), req.get_header_value("Origin"),
                                                         req.get_header_value("Content-Type"), names)) {
                reply(res, 403, {{"error", "only this page or local client may change the server"}});
                return;
            }
            const Json body = Json::parse(req.body, nullptr, false);
            if (!body.is_object()) {
                reply(res, 400, {{"error", "the body must be a JSON object"}});
                return;
            }
            try {
                handler(body, res);
            } catch (const Json::type_error& e) {
                reply(res, 400, {{"error", std::string("a field has the wrong type: ") + e.what()}});
            }
        });
    };

    // Supports percentage shares, reserve GiB policy, or an explicit --experts spec string.
    post("/api/placement", [&](const Json& body, httplib::Response& res) {
        const bool dry_run = body.value("dry_run", false);
        if (body.contains("reserve_vram_gb") || body.contains("reserve_ram_gb")) {
            const double rv   = body.value("reserve_vram_gb", 0.0);
            const double rr   = body.value("reserve_ram_gb", 0.0);
            const auto result = monitor.apply_reserve_placement(rv, rr, dry_run);
            reply_raw(res, result.status, result.body);
            return;
        }
        if (body.contains("shares") && body["shares"].is_string()) {
            std::string err;
            const auto parsed = parse_shares_spec_body(body["shares"].get<std::string>(), err);
            if (!parsed) {
                reply(res, 400, {{"error", err}});
                return;
            }
            const auto result = monitor.post_placement_body(*parsed, dry_run);
            reply_raw(res, result.status, result.body);
            return;
        }
        int vram = 0, ram = 0;
        if (!read_shares(body, vram, ram)) {
            reply(res, 400, {{"error", "vram (whole percent), reserve_vram_gb, or shares is required"}});
            return;
        }
        const auto result = monitor.switch_placement(vram, ram, dry_run);
        reply_raw(res, result.status, result.body);
    });

    // {"action": "load", "profile": NAME, "vram": V, "ram": R | "experts": SPEC} or {"action": "unload"}.
    post("/api/control", [&](const Json& body, httplib::Response& res) {
        auto* controller = monitor.controller();
        if (controller == nullptr) {
            reply(res, 404, {{"error", "manager lifecycle control is disabled in read-only --engine mode"}});
            return;
        }
        const std::string action = body.value("action", "");
        if (action != "load" && action != "unload") {
            reply(res, 400, {{"error", "action must be load or unload"}});
            return;
        }
        std::optional<std::string> error;
        if (action == "unload") {
            error = controller->unload();
        } else {
            const std::string profile = body.value("profile", "");
            std::optional<std::string> experts;
            if (body.contains("experts") && body["experts"].is_string() && !body["experts"].get<std::string>().empty()) {
                experts = body["experts"].get<std::string>();
            } else if (body.contains("vram")) {
                int vram = 0, ram = 0;
                if (!read_shares(body, vram, ram)) {
                    reply(res, 400, {{"error", infernix::monitor::kSharesError}});
                    return;
                }
                const auto spec = infernix::monitor::experts_spec(vram, ram);
                if (!spec) {
                    reply(res, 400, {{"error", infernix::monitor::kSharesError}});
                    return;
                }
                experts = *spec;
            }
            error = controller->load(profile, experts);
        }
        if (error) {
            reply(res, 409, {{"error", *error}});
            return;
        }
        reply(res, 202, controller->state());
    });

    server.Get("/api/logs", [&](const httplib::Request& req, httplib::Response& res) {
        std::uint64_t after = 0;
        if (req.has_param("after")) { after = std::strtoull(req.get_param_value("after").c_str(), nullptr, 10); }
        res.set_header("Cache-Control", "no-store");
        res.set_content(monitor.logs_json(after), "application/json");
    });

    if (!server.bind_to_port(options.host, options.port)) {
        const std::string err = "cannot bind " + options.host + ":" + std::to_string(options.port);
        report_ready("ERR:" + err + "\n");
        std::cerr << err << '\n';
        return 1;
    }

    if (options.daemon) { write_daemon_pid(options.pid_file, ::getpid()); }

    std::atomic<bool> running{true};
    std::thread sampler([&] {
        auto next = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (running.load()) {
            while (running.load() && std::chrono::steady_clock::now() < next) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (!running.load()) { break; }
            monitor.tick();
            next += std::chrono::seconds(1);
            const auto now = std::chrono::steady_clock::now();
            if (next < now) { next = now; }
        }
    });

    if (options.autoload && monitor.controller() != nullptr) {
        (void)monitor.controller()->load(options.monitor.manager.default_profile);
    }

    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    report_ready("OK:" + std::to_string(::getpid()) + "\n");
    std::cout << "infernix-monitor on http://" << options.host << ':' << options.port;
    if (options.monitor.manage_server) {
        std::cout << " | configs " << options.monitor.manager.configs_dir.string() << " | default "
                  << options.monitor.manager.default_profile;
    } else {
        std::cout << " | engine " << options.monitor.engine_url;
    }
    std::cout << std::endl;

    const bool listened = server.listen_after_bind();
    running.store(false);
    sampler.join();
    if (options.daemon) {
        std::error_code ec;
        std::filesystem::remove(options.pid_file, ec);
    }
    return listened ? 0 : 1;
}

int launch_daemon(const ServerCliOptions& options) {
    std::error_code ec;
    const auto log_dir = options.monitor.manager.log_dir.empty()
                             ? std::filesystem::path(infernix::product::expand_user_path("~/.cache/ninferpro"))
                             : options.monitor.manager.log_dir;
    std::filesystem::create_directories(log_dir, ec);
    const auto monitor_log = log_dir / "monitor.log";

    // Refuse if a live daemon is already recorded in pid_file.
    if (std::ifstream in(options.pid_file); in) {
        pid_t old_pid = -1;
        in >> old_pid;
        if (old_pid > 0 && ::kill(old_pid, 0) == 0 &&
            read_proc_cmdline(old_pid).find("infernix-monitor") != std::string::npos) {
            std::cout << "infernix-monitor is already running (PID " << old_pid << "): http://" << options.host << ':'
                      << options.port << '\n';
            return 0;
        }
    }

    int ready_pipe[2];
    if (::pipe2(ready_pipe, O_CLOEXEC) != 0) {
        std::cerr << "pipe failed: " << std::strerror(errno) << '\n';
        return 1;
    }
    const int log_fd  = ::open(monitor_log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    const int null_fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (log_fd < 0 || null_fd < 0) {
        std::cerr << "cannot open " << monitor_log << ": " << std::strerror(errno) << '\n';
        return 1;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        std::cerr << "fork failed: " << std::strerror(errno) << '\n';
        return 1;
    }
    if (pid == 0) {
        ::close(ready_pipe[0]);
        ::setsid();
        ::dup2(null_fd, 0);
        ::dup2(log_fd, 1);
        ::dup2(log_fd, 2);
        ::close(null_fd);
        ::close(log_fd);
        std::exit(run_server(options, ready_pipe[1]));
    }

    ::close(ready_pipe[1]);
    ::close(null_fd);
    ::close(log_fd);
    char buf[512]{};
    const ssize_t n = ::read(ready_pipe[0], buf, sizeof(buf) - 1);
    ::close(ready_pipe[0]);
    const std::string reply_line(n > 0 ? buf : "");
    if (reply_line.starts_with("OK:")) {
        std::cout << "infernix-monitor: http://" << options.host << ':' << options.port << " (PID " << pid
                  << ", log " << monitor_log.string() << ")\n";
        return 0;
    }
    std::cerr << "failed to start infernix-monitor: "
              << (reply_line.starts_with("ERR:") ? reply_line.substr(4) : "exited unexpectedly (see " + monitor_log.string() + ")\n");
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::string_view cmd = argv[1];
        if (cmd == "status") { return cmd_status(argc, argv); }
        if (cmd == "load") { return cmd_load(argc, argv); }
        if (cmd == "unload") { return cmd_unload(argc, argv); }
        if (cmd == "placement") { return cmd_placement(argc, argv); }
        if (cmd == "stop") { return cmd_stop(argc, argv); }
        if (cmd == "serve") {
            const ServerCliOptions options = parse_server_options(argc, argv, 2);
            return options.daemon ? launch_daemon(options) : run_server(options, -1);
        }
    }
    const ServerCliOptions options = parse_server_options(argc, argv, 1);
    return options.daemon ? launch_daemon(options) : run_server(options, -1);
}
