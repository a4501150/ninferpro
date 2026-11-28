#include "control.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <httplib.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <system_error>

namespace infernix::monitor {

using Json = nlohmann::json;

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

double unix_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string timestamp_slug() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d-%02d%02d%02d", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                  local.tm_hour, local.tm_min, local.tm_sec);
    return buf;
}

// The host part of a Host header: "[::1]:8090" -> "::1", "box:8090" -> "box".
std::string_view host_name(std::string_view host) {
    if (!host.empty() && host.front() == '[') {
        const auto close = host.find(']');
        return close == std::string_view::npos ? std::string_view() : host.substr(1, close - 1);
    }
    const auto colon = host.rfind(':');
    return colon == std::string_view::npos ? host : host.substr(0, colon);
}

bool ip_literal(std::string_view name) {
    const std::string text(name);
    unsigned char buffer[16];
    return ::inet_pton(AF_INET, text.c_str(), buffer) == 1 || ::inet_pton(AF_INET6, text.c_str(), buffer) == 1;
}

bool iequal(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
        if (lower(a[i]) != lower(b[i])) { return false; }
    }
    return true;
}

bool loopback_port_in_use(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { return false; }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<std::uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const int rc         = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::close(fd);
    return rc == 0;
}

bool http_ready(const std::string& url, const std::string& api_key) {
    httplib::Client client(url);
    client.set_connection_timeout(1, 0);
    client.set_read_timeout(2, 0);
    if (!api_key.empty()) { client.set_bearer_token_auth(api_key); }
    const auto res = client.Get("/v1/models");
    return res && res->status == 200;
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

std::filesystem::path pid_state_path(const std::filesystem::path& log_dir) { return log_dir / "infernix-serve.pid"; }

// Builds a child environment before fork() so the child calls only async-signal-safe functions.
std::vector<std::string> build_child_env(const std::vector<std::pair<std::string, std::string>>& overrides) {
    std::vector<std::string> env;
    if (environ != nullptr) {
        for (char** p = environ; *p != nullptr; ++p) {
            const std::string_view entry(*p);
            const std::size_t eq = entry.find('=');
            const std::string_view key = eq == std::string_view::npos ? entry : entry.substr(0, eq);
            const bool replaced =
                std::any_of(overrides.begin(), overrides.end(), [&](const auto& kv) { return kv.first == key; });
            if (!replaced) { env.emplace_back(entry); }
        }
    }
    for (const auto& [k, v] : overrides) { env.push_back(k + "=" + v); }
    return env;
}

} // namespace

std::optional<Json> placement_body(int vram_percent, int ram_percent) {
    if (vram_percent < 1 || ram_percent < 0 || vram_percent + ram_percent > 100) { return std::nullopt; }
    if (vram_percent == 100) { return Json{{"vram", "all"}}; }
    const auto share = [](int percent) { return std::to_string(percent) + "%"; };
    return Json{{"vram", share(vram_percent)},
                {"ram", share(ram_percent)},
                {"ssd", share(100 - vram_percent - ram_percent)}};
}

std::optional<std::string> experts_spec(int vram_percent, int ram_percent) {
    if (vram_percent < 1 || ram_percent < 0 || vram_percent + ram_percent > 100) { return std::nullopt; }
    if (vram_percent == 100) { return std::string("vram=100%"); }
    return "vram=" + std::to_string(vram_percent) + "%,ram=" + std::to_string(ram_percent) +
           "%,ssd=" + std::to_string(100 - vram_percent - ram_percent) + "%";
}

std::optional<ReservePlan> compute_reserve_plan(const Json& report, double reserve_vram_gb, double reserve_ram_gb,
                                                std::string* error) {
    const auto fail = [&](std::string msg) -> std::optional<ReservePlan> {
        if (error != nullptr) { *error = std::move(msg); }
        return std::nullopt;
    };
    if (!report.is_object() || !report.value("supported", false)) {
        return fail("this model runs without the expert cache");
    }
    if (reserve_vram_gb < 0.0 || reserve_ram_gb < 0.0 || !std::isfinite(reserve_vram_gb) ||
        !std::isfinite(reserve_ram_gb)) {
        return fail("reserve GiB values must be non-negative");
    }
    const Json& mech = report.contains("mechanism") && report["mechanism"].is_object() ? report["mechanism"] : Json{};
    const std::uint64_t total = report.value("total_experts", mech.value("total_experts", std::uint64_t{0}));
    const Json& vram_obj      = mech.contains("vram") && mech["vram"].is_object() ? mech["vram"] : Json{};
    const Json& ram_obj       = mech.contains("ram") && mech["ram"].is_object() ? mech["ram"] : Json{};
    const std::uint64_t frame_bytes  = vram_obj.value("frame_bytes", std::uint64_t{0});
    const std::uint64_t record_bytes = ram_obj.value("record_bytes", std::uint64_t{0});
    if (total == 0 || frame_bytes == 0) { return fail("the server reports no expert sizes"); }

    const double vram_free  = static_cast<double>(mech.value("vram_free", std::uint64_t{0}));
    const double vram_bytes = static_cast<double>(vram_obj.value("bytes", std::uint64_t{0}));
    const double vram_budget = vram_free + vram_bytes - reserve_vram_gb * kGiB;
    const std::int64_t raw_vram =
        vram_budget <= 0.0 ? 1 : static_cast<std::int64_t>(std::floor(vram_budget / static_cast<double>(frame_bytes)));
    const std::uint64_t vram = static_cast<std::uint64_t>(
        std::clamp<std::int64_t>(raw_vram, 1, static_cast<std::int64_t>(total)));

    const double host_avail = static_cast<double>(mech.value("host_available", std::uint64_t{0}));
    const double ram_budget = std::max(0.0, host_avail - reserve_ram_gb * kGiB);
    std::uint64_t ram       = 0;
    if (record_bytes != 0 && vram < total) {
        const auto raw_ram = static_cast<std::int64_t>(std::floor(ram_budget / static_cast<double>(record_bytes)));
        ram = static_cast<std::uint64_t>(
            std::clamp<std::int64_t>(raw_ram, 0, static_cast<std::int64_t>(total - vram)));
    }
    const std::uint64_t ssd = total - vram - ram;

    ReservePlan plan;
    plan.vram_experts  = vram;
    plan.ram_experts   = ram;
    plan.ssd_experts   = ssd;
    plan.total_experts = total;
    if (vram >= total) {
        plan.body = Json{{"vram", "all"}};
    } else {
        plan.body = Json{{"vram", vram}, {"ram", ram}};
    }
    return plan;
}

bool page_request_allowed(std::string_view host, std::string_view origin, std::string_view content_type,
                          const std::vector<std::string>& names) {
    if (content_type.substr(0, std::string_view("application/json").size()) != "application/json") { return false; }
    const std::string_view name = host_name(host);
    if (name.empty()) { return false; }
    bool known = iequal(name, "localhost") || ip_literal(name);
    for (const std::string& n : names) { known = known || iequal(name, n); }
    if (!known) { return false; }
    return origin.empty() || origin == "http://" + std::string(host);
}

Controller::Controller(ManagerConfig config, LineSink on_line, TargetSink on_target)
    : config_(std::move(config)), on_line_(std::move(on_line)), on_target_(std::move(on_target)), last_(nullptr) {
    adopt_existing_server();
    thread_ = std::thread([this] { work(); });
}

Controller::~Controller() {
    {
        const std::lock_guard lock(mutex_);
        cancel_.store(true);
    }
    wake_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
}

std::optional<std::string> Controller::load(const std::string& profile_name,
                                            std::optional<std::string> experts_override) {
    return queue(Action{"load", profile_name.empty() ? config_.default_profile : profile_name,
                        std::move(experts_override)});
}

std::optional<std::string> Controller::unload() { return queue(Action{"unload", {}, std::nullopt}); }

std::optional<std::string> Controller::queue(Action action) {
    {
        const std::lock_guard lock(mutex_);
        if (pending_ || running_) { return "another action is running"; }
        pending_ = std::move(action);
    }
    wake_.notify_all();
    return std::nullopt;
}

Json Controller::state() const {
    const std::lock_guard lock(mutex_);
    // Reap / refresh tracked pid liveness lazily without mutating fields (just report truth).
    const bool alive = tracked_pid_ > 0 && pid_alive_and_matches(tracked_pid_);
    Json busy        = nullptr;
    if (!busy_phase_.empty()) {
        busy = busy_phase_;
    } else if (pending_) {
        busy = pending_->kind == "unload" ? "unloading" : (alive ? "switching" : "loading");
    }
    return Json{{"enabled", true},
                {"busy", busy},
                {"active_profile", alive && !active_profile_.empty() ? Json(active_profile_) : Json(nullptr)},
                {"pid", alive ? Json(static_cast<std::int64_t>(tracked_pid_)) : Json(nullptr)},
                {"last", last_}};
}

std::optional<std::string> Controller::active_profile() const {
    const std::lock_guard lock(mutex_);
    if (tracked_pid_ > 0 && pid_alive_and_matches(tracked_pid_) && !active_profile_.empty()) {
        return active_profile_;
    }
    return std::nullopt;
}

std::optional<pid_t> Controller::server_pid() const {
    const std::lock_guard lock(mutex_);
    if (tracked_pid_ > 0 && pid_alive_and_matches(tracked_pid_)) { return tracked_pid_; }
    return std::nullopt;
}

void Controller::reap_exited() {
    std::string note;
    bool error = false;
    {
        const std::lock_guard lock(mutex_);
        if (running_ || tracked_pid_ <= 0) { return; }
        bool gone = false;
        int status = 0;
        if (direct_child_) {
            const pid_t w = ::waitpid(tracked_pid_, &status, WNOHANG);
            if (w == tracked_pid_) {
                gone = true;
                const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
                note  = "manager: " + (active_profile_.empty() ? "server" : active_profile_) + " (PID " +
                       std::to_string(tracked_pid_) + ") exited with status " + std::to_string(code);
                error = code != 0;
            } else if (w < 0 && errno == ECHILD) {
                gone = true;
            }
        } else if (!pid_alive_and_matches(tracked_pid_)) {
            gone = true;
            note = "manager: adopted server PID " + std::to_string(tracked_pid_) + " is no longer running";
        }
        if (gone) {
            tracked_pid_  = -1;
            direct_child_ = false;
            active_profile_.clear();
            remove_pid_state();
        }
    }
    if (!note.empty() && on_line_) { on_line_(note, error); }
}

bool Controller::pid_alive_and_matches(pid_t pid) const {
    if (pid <= 0) { return false; }
    if (::kill(pid, 0) != 0 && errno != EPERM) { return false; }
    const std::string cmdline = read_proc_cmdline(pid);
    if (cmdline.empty()) { return false; }
    const std::string bin_name = config_.server_bin.filename().string();
    return cmdline.find(bin_name.empty() ? "infernix-serve" : bin_name) != std::string::npos;
}

void Controller::adopt_existing_server() {
    std::ifstream in(pid_state_path(config_.log_dir));
    if (!in) { return; }
    pid_t pid = -1;
    std::string profile;
    int port = 8000;
    in >> pid;
    std::getline(in, profile); // consume newline
    std::getline(in, profile);
    in >> port;
    if (pid <= 0 || !pid_alive_and_matches(pid)) {
        remove_pid_state();
        return;
    }
    tracked_pid_    = pid;
    direct_child_   = false;
    active_profile_ = profile;
    active_port_    = (port >= 1 && port <= 65535) ? port : 8000;
    if (on_line_) {
        on_line_("manager: adopted running server PID " + std::to_string(pid) +
                     (profile.empty() ? "" : " (" + profile + ")"),
                 false);
    }
    if (on_target_) {
        EngineTarget target;
        target.engine_url = "http://127.0.0.1:" + std::to_string(active_port_);
        target.log_file   = config_.log_dir / "latest.log";
        if (config_.enable_request_log) { target.request_log = config_.log_dir / "requests.jsonl"; }
        on_target_(target);
    }
}

void Controller::write_pid_state(pid_t pid, const std::string& profile_name, int port) const {
    std::error_code ec;
    std::filesystem::create_directories(config_.log_dir, ec);
    const auto path = pid_state_path(config_.log_dir);
    const auto tmp  = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) { return; }
    out << pid << '\n' << profile_name << '\n' << port << '\n';
    out.close();
    std::filesystem::rename(tmp, path, ec);
}

void Controller::remove_pid_state() const {
    std::error_code ec;
    std::filesystem::remove(pid_state_path(config_.log_dir), ec);
}

void Controller::prune_old_logs() const {
    if (config_.log_keep_days <= 0) { return; }
    std::error_code ec;
    if (!std::filesystem::is_directory(config_.log_dir, ec) || ec) { return; }
    const auto cutoff = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * config_.log_keep_days);
    for (const auto& entry : std::filesystem::directory_iterator(config_.log_dir, ec)) {
        if (ec) { break; }
        if (!entry.is_regular_file(ec) || ec) { continue; }
        const std::string name = entry.path().filename().string();
        const bool match = (name.starts_with("infernix-") && name.ends_with(".log")) || name.starts_with("gpucore-");
        if (!match) { continue; }
        const auto mtime = entry.last_write_time(ec);
        if (!ec && mtime < cutoff) { std::filesystem::remove(entry.path(), ec); }
    }
}

std::string Controller::read_tail_line(const std::filesystem::path& log_path) const {
    std::ifstream in(log_path, std::ios::binary);
    if (!in) { return {}; }
    std::string line, last;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line.find_first_not_of(" \t") != std::string::npos) { last = line; }
    }
    return last;
}

bool Controller::do_unload(std::string& message) {
    pid_t pid        = -1;
    bool direct      = false;
    std::string prof;
    int port = 8000;
    {
        const std::lock_guard lock(mutex_);
        pid    = tracked_pid_;
        direct = direct_child_;
        prof   = active_profile_;
        port   = active_port_;
    }

    if (pid <= 0 || !pid_alive_and_matches(pid)) {
        {
            const std::lock_guard lock(mutex_);
            tracked_pid_  = -1;
            direct_child_ = false;
            active_profile_.clear();
        }
        remove_pid_state();
        if (loopback_port_in_use(port)) {
            message = "port " + std::to_string(port) + " is held by a process not started by this manager";
            return false;
        }
        message = "no model loaded";
        return true;
    }

    if (on_line_) {
        on_line_("manager: unloading " + (prof.empty() ? "server" : prof) + " (PID " + std::to_string(pid) + ")",
                 false);
    }
    if (::kill(pid, SIGTERM) != 0 && errno != ESRCH) {
        message = "failed to signal PID " + std::to_string(pid) + ": " + std::strerror(errno);
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() + config_.unload_timeout;
    bool exited         = false;
    while (std::chrono::steady_clock::now() < deadline && !cancel_.load()) {
        if (direct) {
            int status        = 0;
            const pid_t w     = ::waitpid(pid, &status, WNOHANG);
            if (w == pid || (w < 0 && errno == ECHILD)) {
                exited = true;
                break;
            }
        } else if (::kill(pid, 0) != 0 && errno == ESRCH) {
            exited = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }

    if (!exited) {
        if (on_line_) {
            on_line_("manager: server PID " + std::to_string(pid) + " did not exit in " +
                         std::to_string(config_.unload_timeout.count()) + " s; sending SIGKILL",
                     true);
        }
        ::kill(pid, SIGKILL);
        const auto kill_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < kill_deadline) {
            if (direct) {
                int status    = 0;
                const pid_t w = ::waitpid(pid, &status, WNOHANG);
                if (w == pid || (w < 0 && errno == ECHILD)) {
                    exited = true;
                    break;
                }
            } else if (::kill(pid, 0) != 0 && errno == ESRCH) {
                exited = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    {
        const std::lock_guard lock(mutex_);
        tracked_pid_  = -1;
        direct_child_ = false;
        active_profile_.clear();
    }
    remove_pid_state();
    if (!exited) {
        message = "server PID " + std::to_string(pid) + " did not terminate";
        return false;
    }
    message = "unloaded" + (prof.empty() ? "" : " " + prof);
    return true;
}

bool Controller::do_load(const Action& action, std::string& message, std::string& resolved_profile) {
    const std::vector<ModelProfile> profiles = scan_profiles(config_.configs_dir);
    const auto it = std::find_if(profiles.begin(), profiles.end(),
                                 [&](const ModelProfile& p) { return p.name == action.profile_name; });
    if (it == profiles.end()) {
        message = "unknown profile '" + action.profile_name + "' in " + config_.configs_dir.string();
        return false;
    }
    const ModelProfile& profile = *it;
    resolved_profile            = profile.name;
    if (!profile.parse_error.empty()) {
        message = "profile " + profile.name + " has invalid TOML: " + profile.parse_error;
        return false;
    }
    if (profile.model_path.empty() || !profile.artifact_exists) {
        message = "missing model artifact: " + profile.model_path.string();
        return false;
    }
    if (::access(config_.server_bin.c_str(), X_OK) != 0) {
        message = "missing server binary: " + config_.server_bin.string();
        return false;
    }

    // If a managed server is already running, unload it cleanly first so VRAM returns to 0.
    pid_t existing = -1;
    {
        const std::lock_guard lock(mutex_);
        existing = tracked_pid_;
    }
    if (existing > 0 && pid_alive_and_matches(existing)) {
        std::string unload_msg;
        if (!do_unload(unload_msg)) {
            message = "unload before switch failed: " + unload_msg;
            return false;
        }
    }

    if (loopback_port_in_use(profile.port)) {
        message = "port " + std::to_string(profile.port) + " is already in use";
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(config_.log_dir, ec);
    if (ec) {
        message = "cannot create log directory " + config_.log_dir.string() + ": " + ec.message();
        return false;
    }
    prune_old_logs();

    const std::string stamp               = timestamp_slug();
    const std::filesystem::path log_path  = config_.log_dir / ("infernix-" + stamp + ".log");
    const std::filesystem::path link_path = config_.log_dir / "latest.log";
    const std::filesystem::path link_tmp  = config_.log_dir / ("latest.log.tmp." + std::to_string(::getpid()));
    const std::filesystem::path req_log =
        !profile.request_log_jsonl.empty()
            ? profile.request_log_jsonl
            : (config_.enable_request_log ? (config_.log_dir / "requests.jsonl") : std::filesystem::path{});

    const int log_fd  = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    const int null_fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (log_fd < 0 || null_fd < 0) {
        if (log_fd >= 0) { ::close(log_fd); }
        if (null_fd >= 0) { ::close(null_fd); }
        message = "cannot open server log " + log_path.string() + ": " + std::strerror(errno);
        return false;
    }

    // Atomically repoint latest.log to the new dated log file.
    std::filesystem::remove(link_tmp, ec);
    std::filesystem::create_symlink(log_path, link_tmp, ec);
    if (!ec) { std::filesystem::rename(link_tmp, link_path, ec); }

    std::vector<std::string> argv{config_.server_bin.string(), "--config", profile.config_path.string()};
    if (action.experts_override && !action.experts_override->empty()) {
        argv.push_back("--experts");
        argv.push_back(*action.experts_override);
    }
    if (config_.enable_request_log && profile.request_log_jsonl.empty()) {
        argv.push_back("--request-log-jsonl");
        argv.push_back(req_log.string());
        argv.push_back("--request-log-max-mib");
        argv.push_back(std::to_string(config_.request_log_max_mib));
    }

    std::vector<char*> c_argv;
    c_argv.reserve(argv.size() + 1);
    for (std::string& a : argv) { c_argv.push_back(a.data()); }
    c_argv.push_back(nullptr);

    const std::string coredump_file = (config_.log_dir / ("gpucore-" + stamp + "-%p")).string();
    const std::string work_dir_str  = config_.work_dir.string();
    std::vector<std::string> child_env = build_child_env({
        {"CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "1"},
        {"CUDA_COREDUMP_FILE", coredump_file},
        {"CUDA_COREDUMP_GENERATION_FLAGS",
         "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory"},
    });
    std::vector<char*> c_envp;
    c_envp.reserve(child_env.size() + 1);
    for (std::string& e : child_env) { c_envp.push_back(e.data()); }
    c_envp.push_back(nullptr);

    if (on_target_) {
        on_target_(EngineTarget{profile_engine_url(profile), link_path, req_log});
    }

    const pid_t pid = ::fork();
    if (pid == 0) {
        ::setsid();
        if (!work_dir_str.empty()) { [[maybe_unused]] const int ignored = ::chdir(work_dir_str.c_str()); }
        ::dup2(null_fd, 0);
        ::dup2(log_fd, 1);
        ::dup2(log_fd, 2);
        ::close_range(3, ~0U, 0);
        ::signal(SIGPIPE, SIG_DFL);
        ::execve(c_argv[0], c_argv.data(), c_envp.data());
        static constexpr char kFailed[] = "cannot execute infernix-serve\n";
        [[maybe_unused]] const auto ignored = ::write(2, kFailed, sizeof(kFailed) - 1);
        ::_exit(127);
    }
    ::close(null_fd);
    ::close(log_fd);
    if (pid < 0) {
        message = "fork failed: " + std::string(std::strerror(errno));
        return false;
    }

    {
        const std::lock_guard lock(mutex_);
        tracked_pid_    = pid;
        direct_child_   = true;
        active_profile_ = profile.name;
        active_port_    = profile.port;
    }
    write_pid_state(pid, profile.name, profile.port);

    if (on_line_) {
        on_line_("manager: loading " + profile.name +
                     (action.experts_override && !action.experts_override->empty()
                          ? " (" + *action.experts_override + ")"
                          : "") +
                     " | PID " + std::to_string(pid) + " | log " + log_path.filename().string(),
                 false);
    }

    const std::string engine_url = profile_engine_url(profile);
    const auto started           = std::chrono::steady_clock::now();
    const auto deadline          = started + config_.load_timeout;

    while (std::chrono::steady_clock::now() < deadline) {
        if (cancel_.load()) {
            message = "manager shutting down during load";
            return false;
        }
        int status    = 0;
        const pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            {
                const std::lock_guard lock(mutex_);
                tracked_pid_  = -1;
                direct_child_ = false;
                active_profile_.clear();
            }
            remove_pid_state();
            const std::string tail = read_tail_line(log_path);
            const int code         = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            message                = tail.empty()
                                         ? ("server exited with status " + std::to_string(code))
                                         : ("server exited (" + std::to_string(code) + "): " + tail);
            return false;
        }
        if (http_ready(engine_url, config_.api_key)) {
            const auto secs =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started).count();
            message = "ready in " + std::to_string(secs) + " s (" + profile.served_name + ")";
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }

    // Load timed out: terminate the stuck child so it does not linger.
    ::kill(pid, SIGTERM);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    int status = 0;
    if (::waitpid(pid, &status, WNOHANG) != pid) {
        ::kill(pid, SIGKILL);
        ::waitpid(pid, &status, 0);
    }
    {
        const std::lock_guard lock(mutex_);
        tracked_pid_  = -1;
        direct_child_ = false;
        active_profile_.clear();
    }
    remove_pid_state();
    message = "server did not become ready within " + std::to_string(config_.load_timeout.count()) + " s";
    return false;
}

void Controller::work() {
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return cancel_.load() || pending_.has_value(); });
        if (cancel_.load()) { return; }
        running_ = std::move(pending_);
        pending_.reset();
        const bool had_server = tracked_pid_ > 0 && pid_alive_and_matches(tracked_pid_);
        if (running_->kind == "unload") {
            busy_phase_ = "unloading";
        } else {
            busy_phase_ = had_server ? "switching" : "loading";
        }
        const Action action = *running_;
        lock.unlock();
        run(action);
        lock.lock();
        running_.reset();
        busy_phase_.clear();
    }
}

void Controller::run(const Action& action) {
    bool ok = false;
    std::string message;
    std::string profile = action.profile_name;
    if (action.kind == "unload") {
        ok = do_unload(message);
    } else {
        ok = do_load(action, message, profile);
    }
    if (on_line_) {
        on_line_("manager: " + action.kind + (profile.empty() ? "" : " " + profile) + (ok ? " done" : " failed") +
                     (message.empty() ? "" : ": " + message),
                 !ok);
    }
    const std::lock_guard lock(mutex_);
    last_ = {{"action", action.kind},
             {"profile", profile.empty() ? Json(nullptr) : Json(profile)},
             {"ok", ok},
             {"message", message},
             {"time", unix_seconds()}};
}

} // namespace infernix::monitor
