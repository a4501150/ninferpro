#pragma once

// Native server lifecycle control and client-side placement policy for infernix-monitor.
// Manages `infernix-serve` directly from TOML profiles (load, switch, unload), handles dated logs,
// GPU core dumps and PID adoption across monitor restarts, and computes reserve-based placement.

#include "profiles.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <thread>
#include <vector>

namespace infernix::monitor {

using LineSink = std::function<void(const std::string& line, bool error)>;

// Invoked when the controller starts a server on a (possibly new) engine URL or log path so
// Monitor can point its poller and log tails at it immediately.
struct EngineTarget {
    std::string engine_url;
    std::filesystem::path log_file;
    std::filesystem::path request_log;
};
using TargetSink = std::function<void(const EngineTarget&)>;

struct ManagerConfig {
    std::filesystem::path configs_dir;
    std::filesystem::path server_bin;
    std::filesystem::path work_dir;
    std::filesystem::path log_dir;
    std::string default_profile = "GLM-5.3-Flash";
    std::string api_key;
    bool enable_request_log           = true;
    std::uint32_t request_log_max_mib = 256;
    int log_keep_days                 = 30;
    std::chrono::seconds load_timeout{420};
    std::chrono::seconds unload_timeout{60};
};

inline constexpr const char* kSharesError = "VRAM takes 1-100 % and VRAM plus RAM at most 100 %";

// The engine's placement body for whole-percent shares, the rest on SSD: {"vram":"all"} at 100,
// else {"vram":"V%","ram":"R%","ssd":"S%"}. nullopt unless 1 <= vram and vram + ram <= 100.
std::optional<nlohmann::json> placement_body(int vram_percent, int ram_percent);

// Explicit --experts override for a server launch ("vram=100%" at 100, else
// "vram=V%,ram=R%,ssd=S%"). Returns nullopt on invalid percentages.
std::optional<std::string> experts_spec(int vram_percent, int ram_percent);

struct ReservePlan {
    std::uint64_t vram_experts  = 0;
    std::uint64_t ram_experts   = 0;
    std::uint64_t ssd_experts   = 0;
    std::uint64_t total_experts = 0;
    nlohmann::json body; // ready to POST to /admin/expert-placement
};

// Client-side placement policy: largest VRAM share that leaves `reserve_vram_gb` GiB free for
// other programs, then largest RAM share that leaves `reserve_ram_gb` GiB of host RAM free, the
// rest on SSD. Returns nullopt and populates `error` when the report lacks expert sizes.
std::optional<ReservePlan> compute_reserve_plan(const nlohmann::json& placement_report, double reserve_vram_gb,
                                                double reserve_ram_gb, std::string* error = nullptr);

// Whether a POST may change the server: it comes from this page (an Origin, when sent, equal to
// http://Host), asks by JSON (so another site's request needs a CORS preflight the monitor never
// answers), and names the monitor by IP literal, localhost or one of `names`.
bool page_request_allowed(std::string_view host, std::string_view origin, std::string_view content_type,
                          const std::vector<std::string>& names);

class Controller {
public:
    Controller(ManagerConfig config, LineSink on_line, TargetSink on_target = {});
    ~Controller();

    // Queue a Load (or Switch when another server is currently running) of `profile_name`
    // (empty uses config_.default_profile). `experts_override` overrides the profile's [serve].experts
    // when non-empty; empty/nullopt keeps the profile's own placement.
    // Returns an error message when another action is already running.
    std::optional<std::string> load(const std::string& profile_name,
                                    std::optional<std::string> experts_override = std::nullopt);
    std::optional<std::string> unload();

    nlohmann::json state() const;
    std::optional<std::string> active_profile() const;
    std::optional<pid_t> server_pid() const;
    // Reaps a tracked child that exited outside an explicit unload (e.g. crash or external signal).
    void reap_exited();

private:
    struct Action {
        std::string kind; // "load" or "unload"
        std::string profile_name;
        std::optional<std::string> experts_override;
    };

    std::optional<std::string> queue(Action action);
    void adopt_existing_server();
    void write_pid_state(pid_t pid, const std::string& profile_name, int port) const;
    void remove_pid_state() const;
    bool pid_alive_and_matches(pid_t pid) const;
    void prune_old_logs() const;
    std::string read_tail_line(const std::filesystem::path& log_path) const;

    bool do_unload(std::string& message);
    bool do_load(const Action& action, std::string& message, std::string& resolved_profile);

    void work();
    void run(const Action& action);

    ManagerConfig config_;
    LineSink on_line_;
    TargetSink on_target_;
    std::atomic<bool> cancel_{false};

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::optional<Action> pending_, running_;
    std::string busy_phase_; // "loading", "unloading", "switching", or empty
    pid_t tracked_pid_ = -1;
    bool direct_child_ = false;
    std::string active_profile_;
    int active_port_ = 8000;
    nlohmann::json last_; // last action outcome, or null
    std::thread thread_;
};

} // namespace infernix::monitor
