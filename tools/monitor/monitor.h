#pragma once

// The monitor's state: once a second it samples the host and GPUs, polls the engine's /metrics
// and /admin/expert-placement, reads new lines from the request log and the console log, and
// refreshes the TOML model profile catalog. The page and CLI read state_json() and logs_json().

#include "control.h"
#include "file_tail.h"
#include "profiles.h"
#include "prometheus.h"
#include "records.h"
#include "system_sampler.h"

#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace infernix::monitor {

class EngineClient;

struct MonitorOptions {
    std::string engine_url = "http://127.0.0.1:8000";
    std::string api_key;
    std::filesystem::path request_log; // empty: derived from manager when manage_server is true
    std::filesystem::path log_file;    // empty: derived from manager when manage_server is true
    bool manage_server = true;
    ManagerConfig manager;
    double gaming_reserve_vram_gb = 32.0;
    double gaming_reserve_ram_gb  = 24.0;
};

// The engine's answer to a request the monitor passes on: its HTTP status (502 when the engine
// did not answer) and body.
struct EngineReply {
    int status = 502;
    std::string body;
};

// Counter totals the page turns into rates; every field is a monotonic engine counter.
struct EngineCounters {
    double time = 0.0; // steady-clock seconds of the reading
    double decode_tokens = 0, prefill_tokens = 0;
    double prompt_tokens = 0, prompt_cached_tokens = 0;
    double expert_routed = 0, expert_hits = 0;
    double drafted = 0, accepted = 0;
};

EngineCounters read_counters(const Metrics& metrics, double time);

class Monitor {
public:
    static constexpr std::size_t kHistory  = 60; // one sample a second
    static constexpr std::size_t kRequests = 200;
    static constexpr std::size_t kLogLines = 2000;

    explicit Monitor(MonitorOptions options);
    ~Monitor();

    void tick();
    std::string state_json() const;
    std::string logs_json(std::uint64_t after) const;

    // POST /admin/expert-placement with whole-percent shares (control.h: placement_body).
    EngineReply switch_placement(int vram_percent, int ram_percent, bool dry_run);
    // Client-side placement policy: compute largest VRAM/RAM counts leaving reserves free, then
    // POST /admin/expert-placement (or ?dry_run=1).
    EngineReply apply_reserve_placement(double reserve_vram_gb, double reserve_ram_gb, bool dry_run);
    // POST /admin/expert-placement with an arbitrary validated body (e.g. from CLI --shares).
    EngineReply post_placement_body(const nlohmann::json& body, bool dry_run);

    // nullptr when running in read-only monitor mode (manage_server == false).
    Controller* controller() noexcept { return controller_.get(); }
    const MonitorOptions& options() const noexcept { return options_; }
    void add_log(LogLevel level, std::string message);

private:
    struct Gpu {
        std::deque<std::optional<double>> util, power, temp;
    };
    struct Best {
        double tok_s     = 0.0;
        std::uint64_t id = 0, tokens = 0;
        double time = 0.0;
    };
    struct LogEntry {
        std::uint64_t seq = 0;
        double time       = 0.0;
        LogLevel level    = LogLevel::Info;
        std::string message;
    };

    void schedule_target_update(const EngineTarget& target);
    void apply_pending_target();
    std::string current_engine_url() const;
    void refresh_profiles(double now, bool force);
    void poll_engine(double now);
    void read_request_log();
    void read_console_log();
    void add_request(const FinishedRequest& request);
    nlohmann::json build_state() const;

    MonitorOptions options_;
    SystemSampler sampler_;
    std::unique_ptr<EngineClient> engine_;
    std::string active_engine_url_;
    std::filesystem::path active_log_file_;
    std::filesystem::path active_request_log_;
    std::unique_ptr<FileTail> request_tail_, log_tail_;
    std::unique_ptr<Controller> controller_;

    // Written by tick() only; read under mutex_ through state_ and logs_.
    std::vector<GpuReading> gpus_;
    HostReading host_;
    std::map<unsigned, Gpu> gpu_history_;
    std::deque<std::optional<double>> decode_rate_, prefill_rate_;
    Metrics metrics_;
    bool reachable_ = false;
    std::string placement_; // the last /admin/expert-placement body; empty when unsupported
    double next_placement_poll_ = 0.0;
    std::deque<EngineCounters> counters_; // the last kHistory + 1 readings
    std::deque<FinishedRequest> requests_; // newest first
    std::optional<Best> best_decode_, best_prefill_;
    std::string server_instance_;
    double last_log_time_       = 0.0;
    std::vector<ModelProfile> profiles_;
    double next_profile_scan_ = 0.0;

    mutable std::mutex mutex_;
    std::string shared_engine_url_;
    std::optional<EngineTarget> pending_target_;
    nlohmann::json state_; // the last tick's readings; state_json() adds live control state
    std::deque<LogEntry> logs_;
    std::uint64_t log_seq_ = 0;
};

} // namespace infernix::monitor
