#pragma once

// The page's Start and Stop. A control command (--control) owns how the server starts and stops;
// the monitor runs it, one action at a time, and shows its output in the Logs pane. The command's
// interface is in tools/monitor/README.md.

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace infernix::monitor {

using LineSink = std::function<void(const std::string& line, bool error)>;

struct CommandResult {
    int exit_code = -1; // 128 + N when killed by signal N; -1 when it did not start
    bool timed_out = false;
    std::string last_line; // the last non-empty output line
};

// Runs argv[0] with argv, its stdin /dev/null and its stdout and stderr fed line by line to
// on_line, in a process group of its own and with no other descriptor of the monitor's (the
// server a start launches must not inherit the page's listening socket). Past the timeout, or
// once *cancel turns true, the group is killed.
CommandResult run_command(const std::vector<std::string>& argv, std::chrono::seconds timeout, const LineSink& on_line,
                          const std::atomic<bool>* cancel = nullptr);

inline constexpr const char* kSharesError = "VRAM takes 1-100 % and VRAM plus RAM at most 100 %";

// The engine's placement body for whole-percent shares, the rest on SSD: {"vram":"all"} at 100,
// else {"vram":"V%","ram":"R%","ssd":"S%"}. nullopt unless 1 <= vram and vram + ram <= 100.
std::optional<nlohmann::json> placement_body(int vram_percent, int ram_percent);
// The same shares as an --experts spec for a start; empty at vram 100 (the engine's default).
std::optional<std::string> experts_spec(int vram_percent, int ram_percent);

// Whether a POST may change the server: it comes from this page (an Origin, when sent, equal to
// http://Host), asks by JSON (so another site's request needs a CORS preflight the monitor never
// answers), and names the monitor by IP literal, localhost or one of `names` (a DNS name another
// site controls could be rebound to this address).
bool page_request_allowed(std::string_view host, std::string_view origin, std::string_view content_type,
                          const std::vector<std::string>& names);

class Controller {
public:
    // `command` is the control command's path; output lines and outcomes go to on_line.
    Controller(std::filesystem::path command, LineSink on_line);
    ~Controller();

    // Queue an action; the error when another one is running. `experts` is an --experts spec,
    // empty for the engine's default.
    std::optional<std::string> start(const std::string& experts);
    std::optional<std::string> stop();

    nlohmann::json state() const;

private:
    struct Action {
        std::string kind; // "start" or "stop"
        std::string experts;
    };

    std::optional<std::string> queue(Action action);
    void work();
    void run(const Action& action);

    std::filesystem::path command_;
    LineSink on_line_;
    std::atomic<bool> cancel_{false};

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::optional<Action> pending_, running_;
    nlohmann::json last_; // the last action's outcome, or null
    std::thread thread_;
};

} // namespace infernix::monitor
