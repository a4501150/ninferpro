#include "control.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace infernix::monitor {

using Json = nlohmann::json;

namespace {

// A start waits for the server to load (the super-quant script allows 300 s), a stop for it to
// save its state and exit.
constexpr std::chrono::seconds kStartTimeout{420};
constexpr std::chrono::seconds kStopTimeout{120};

double unix_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void kill_group(pid_t pid, int signal) {
    if (::kill(-pid, signal) != 0) { ::kill(pid, signal); }
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

} // namespace

CommandResult run_command(const std::vector<std::string>& argv, std::chrono::seconds timeout, const LineSink& on_line,
                          const std::atomic<bool>* cancel) {
    CommandResult result;
    if (argv.empty()) { return result; }
    // Everything the child touches is prepared before fork: after it only async-signal-safe calls.
    std::vector<char*> args;
    for (const std::string& a : argv) { args.push_back(const_cast<char*>(a.c_str())); }
    args.push_back(nullptr);
    const int null_fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    int pipe_fd[2];
    if (null_fd < 0 || ::pipe2(pipe_fd, O_CLOEXEC) != 0) {
        if (null_fd >= 0) { ::close(null_fd); }
        on_line("cannot start " + argv[0] + ": " + std::strerror(errno), true);
        return result;
    }
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::setpgid(0, 0);
        ::dup2(null_fd, 0);
        ::dup2(pipe_fd[1], 1);
        ::dup2(pipe_fd[1], 2);
        ::close_range(3, ~0U, 0);
        ::signal(SIGPIPE, SIG_DFL);
        ::execv(args[0], args.data());
        static constexpr char kFailed[] = "cannot execute the control command\n";
        [[maybe_unused]] const auto ignored = ::write(2, kFailed, sizeof(kFailed) - 1);
        ::_exit(127);
    }
    ::close(null_fd);
    ::close(pipe_fd[1]);
    if (pid < 0) {
        ::close(pipe_fd[0]);
        on_line("cannot start " + argv[0] + ": " + std::strerror(errno), true);
        return result;
    }
    ::setpgid(pid, pid); // also in the parent, so the group exists before any kill

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string partial;
    const auto emit = [&](std::string line) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line.find_first_not_of(" \t") != std::string::npos) { result.last_line = line; }
        on_line(line, false);
    };
    bool killed = false, exited = false;
    int status  = 0;
    for (;;) {
        if (!killed && (std::chrono::steady_clock::now() >= deadline || (cancel != nullptr && cancel->load()))) {
            result.timed_out = std::chrono::steady_clock::now() >= deadline;
            kill_group(pid, SIGKILL);
            killed = true;
        }
        pollfd poll_fd{pipe_fd[0], POLLIN, 0};
        const int ready = ::poll(&poll_fd, 1, 200);
        if (ready < 0 && errno != EINTR) { break; }
        if (ready <= 0) {
            // The command has exited but something it started still holds the pipe: do not wait
            // for that program's end.
            if (ready == 0 && ::waitpid(pid, &status, WNOHANG) == pid) {
                exited = true;
                break;
            }
            continue;
        }
        char buffer[4096];
        const ssize_t n = ::read(pipe_fd[0], buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { break; } // every writer closed: the command and anything it left behind
        partial.append(buffer, static_cast<std::size_t>(n));
        for (std::size_t newline; (newline = partial.find('\n')) != std::string::npos;) {
            emit(partial.substr(0, newline));
            partial.erase(0, newline + 1);
        }
    }
    if (!partial.empty()) { emit(std::move(partial)); }
    ::close(pipe_fd[0]);
    while (!exited && ::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    return result;
}

std::optional<Json> placement_body(int vram_percent, int ram_percent) {
    if (vram_percent < 1 || ram_percent < 0 || vram_percent + ram_percent > 100) { return std::nullopt; }
    if (vram_percent == 100) { return Json{{"vram", "all"}}; }
    const auto share = [](int percent) { return std::to_string(percent) + "%"; };
    return Json{{"vram", share(vram_percent)},
                {"ram", share(ram_percent)},
                {"ssd", share(100 - vram_percent - ram_percent)}};
}

std::optional<std::string> experts_spec(int vram_percent, int ram_percent) {
    const auto body = placement_body(vram_percent, ram_percent);
    if (!body) { return std::nullopt; }
    if (vram_percent == 100) { return std::string(); }
    return "vram=" + (*body)["vram"].get<std::string>() + ",ram=" + (*body)["ram"].get<std::string>() +
           ",ssd=" + (*body)["ssd"].get<std::string>();
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

Controller::Controller(std::filesystem::path command, LineSink on_line)
    : command_(std::move(command)), on_line_(std::move(on_line)), last_(nullptr) {
    thread_ = std::thread([this] { work(); });
}

Controller::~Controller() {
    {
        const std::lock_guard lock(mutex_);
        cancel_.store(true);
    }
    wake_.notify_all();
    thread_.join();
}

std::optional<std::string> Controller::start(const std::string& experts) { return queue(Action{"start", experts}); }

std::optional<std::string> Controller::stop() { return queue(Action{"stop", {}}); }

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
    const std::optional<Action>& action = running_ ? running_ : pending_;
    return Json{{"enabled", true}, {"busy", action ? Json(action->kind) : Json(nullptr)}, {"last", last_}};
}

void Controller::work() {
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return cancel_.load() || pending_.has_value(); });
        if (cancel_.load()) { return; }
        running_ = std::move(pending_);
        pending_.reset();
        lock.unlock();
        run(*running_);
        lock.lock();
        running_.reset();
    }
}

void Controller::run(const Action& action) {
    std::vector<std::string> argv{command_.string(), action.kind};
    if (!action.experts.empty()) { argv.push_back(action.experts); }
    on_line_("control: " + action.kind + (action.experts.empty() ? "" : " " + action.experts), false);
    const auto timeout = action.kind == "start" ? kStartTimeout : kStopTimeout;
    const CommandResult result =
        run_command(argv, timeout, [&](const std::string& line, bool error) { on_line_("control | " + line, error); },
                    &cancel_);
    const bool ok       = result.exit_code == 0;
    std::string message = result.last_line;
    if (result.timed_out) {
        message = action.kind + " did not finish within " + std::to_string(timeout.count()) + " s";
    } else if (!ok && message.empty()) {
        message = action.kind + " failed with status " + std::to_string(result.exit_code);
    }
    on_line_("control: " + action.kind + (ok ? " done" : " failed") + (message.empty() ? "" : ": " + message), !ok);
    const std::lock_guard lock(mutex_);
    last_ = {{"action", action.kind}, {"ok", ok}, {"message", message}, {"time", unix_seconds()}};
}

} // namespace infernix::monitor
