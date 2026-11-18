#include "monitor.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>

namespace infernix::monitor {

using Json = nlohmann::json;

namespace {

constexpr std::uint64_t kRequestBacklogBytes = 16ull << 20; // records carry their token ids
constexpr std::uint64_t kLogBacklogBytes     = 256ull << 10;
constexpr double kPlacementRetrySeconds      = 30.0;
// A finished request counts toward the session bests only past these sizes, so a two-token
// reply does not set the decode record.
constexpr std::uint64_t kBestDecodeTokens  = 32;
constexpr std::uint64_t kBestPrefillTokens = 256;

double steady_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double unix_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

template <typename T>
Json optional_json(const std::optional<T>& value) {
    return value ? Json(*value) : Json(nullptr);
}

template <typename T>
void push_history(std::deque<T>& series, T value) {
    series.push_back(std::move(value));
    while (series.size() > Monitor::kHistory) { series.pop_front(); }
}

Json series_json(const std::deque<std::optional<double>>& series) {
    Json out = Json::array();
    for (const auto& value : series) { out.push_back(optional_json(value)); }
    return out;
}

const char* level_name(LogLevel level) {
    switch (level) {
    case LogLevel::Warn:
        return "warn";
    case LogLevel::Error:
        return "err";
    default:
        return "info";
    }
}

// The fraction delta(numerator) / delta(denominator) between two readings; nullopt without
// enough of the denominator.
std::optional<double> ratio(double numerator, double denominator, double minimum) {
    if (denominator < minimum) { return std::nullopt; }
    return numerator / denominator;
}

} // namespace

class EngineClient {
public:
    EngineClient(const std::string& url, const std::string& api_key) : client_(url) {
        client_.set_connection_timeout(1, 0);
        client_.set_read_timeout(3, 0);
        client_.set_keep_alive(true);
        if (!api_key.empty()) { client_.set_bearer_token_auth(api_key); }
    }

    // The body of a 200 response; nullopt on any failure.
    std::optional<std::string> get(const char* path) {
        const auto result = client_.Get(path);
        if (!result || result->status != 200) { return std::nullopt; }
        return result->body;
    }

private:
    httplib::Client client_;
};

EngineCounters read_counters(const Metrics& metrics, double time) {
    EngineCounters c;
    c.time                 = time;
    c.decode_tokens        = metrics.sum("infernix_decode_tokens_total").value_or(0.0);
    c.prefill_tokens       = metrics.sum("infernix_prefill_tokens_total").value_or(0.0);
    c.prompt_tokens        = metrics.sum("infernix_prompt_tokens_total").value_or(0.0);
    c.prompt_cached_tokens = metrics.sum("infernix_prompt_tokens_cached_total").value_or(0.0);
    c.expert_routed        = metrics.sum("infernix_expert_cache_routed_total").value_or(0.0);
    c.expert_hits          = metrics.sum("infernix_expert_cache_hits_total").value_or(0.0);
    c.drafted              = metrics.sum("infernix_spec_decode_draft_tokens_total").value_or(0.0);
    c.accepted             = metrics.sum("infernix_spec_decode_accepted_tokens_total").value_or(0.0);
    return c;
}

Monitor::Monitor(MonitorOptions options)
    : options_(std::move(options)), engine_(std::make_unique<EngineClient>(options_.engine_url, options_.api_key)) {
    if (!options_.request_log.empty()) {
        request_tail_ = std::make_unique<FileTail>(options_.request_log, kRequestBacklogBytes);
    }
    if (!options_.log_file.empty()) { log_tail_ = std::make_unique<FileTail>(options_.log_file, kLogBacklogBytes); }
}

Monitor::~Monitor() = default;

void Monitor::tick() {
    const double now = steady_seconds();
    gpus_            = sampler_.sample_gpus();
    host_            = sampler_.sample_host();
    for (const GpuReading& g : gpus_) {
        Gpu& h = gpu_history_[g.index];
        push_history(h.util, g.util);
        push_history(h.power, g.power);
        push_history(h.temp, g.temp);
    }
    poll_engine(now);
    read_request_log();
    read_console_log();
    std::string state = build_state();
    const std::lock_guard lock(mutex_);
    state_ = std::move(state);
}

void Monitor::poll_engine(double now) {
    const auto body = engine_->get("/metrics");
    reachable_      = body.has_value();
    if (!reachable_) {
        metrics_ = Metrics{};
        counters_.clear();
        placement_.clear();
        next_placement_poll_ = 0.0;
        push_history(decode_rate_, std::optional<double>{});
        push_history(prefill_rate_, std::optional<double>{});
        return;
    }
    metrics_                    = Metrics(parse_prometheus(*body));
    const EngineCounters latest = read_counters(metrics_, now);
    // A counter that fell means the engine restarted between two readings.
    if (!counters_.empty() && (latest.decode_tokens < counters_.back().decode_tokens ||
                               latest.prefill_tokens < counters_.back().prefill_tokens)) {
        counters_.clear();
    }
    if (counters_.empty()) {
        push_history(decode_rate_, std::optional<double>{});
        push_history(prefill_rate_, std::optional<double>{});
    } else {
        const EngineCounters& last = counters_.back();
        const double dt            = latest.time - last.time;
        push_history(decode_rate_, std::optional<double>((latest.decode_tokens - last.decode_tokens) / dt));
        push_history(prefill_rate_, std::optional<double>((latest.prefill_tokens - last.prefill_tokens) / dt));
    }
    counters_.push_back(latest);
    while (counters_.size() > kHistory + 1) { counters_.pop_front(); }

    if (now >= next_placement_poll_) {
        const auto placement = engine_->get("/admin/expert-placement");
        const Json parsed    = placement ? Json::parse(*placement, nullptr, false) : Json();
        if (parsed.is_object() && parsed.value("supported", false)) {
            placement_ = *placement;
        } else {
            placement_.clear();
            next_placement_poll_ = now + kPlacementRetrySeconds;
        }
    }
}

void Monitor::read_request_log() {
    if (!request_tail_) { return; }
    for (const std::string& line : request_tail_->read_lines()) {
        const auto record = parse_request_log_line(line);
        if (!record) { continue; }
        // The bests are the running server's: a new instance starts them again.
        if (!record->server_instance.empty() && record->server_instance != server_instance_) {
            best_decode_.reset();
            best_prefill_.reset();
            server_instance_ = record->server_instance;
        }
        if (record->event == RequestLogEvent::Finished) { add_request(record->request); }
    }
}

void Monitor::add_request(const FinishedRequest& r) {
    requests_.push_front(r);
    while (requests_.size() > kRequests) { requests_.pop_back(); }
    if (r.completion_tokens >= kBestDecodeTokens && r.decode_s > 0.0) {
        const double rate = static_cast<double>(r.completion_tokens) / r.decode_s;
        if (!best_decode_ || rate > best_decode_->tok_s) { best_decode_ = Best{rate, r.id, r.completion_tokens, r.time}; }
    }
    if (r.computed_prefill_tokens >= kBestPrefillTokens && r.prefill_s > 0.0) {
        const double rate = static_cast<double>(r.computed_prefill_tokens) / r.prefill_s;
        if (!best_prefill_ || rate > best_prefill_->tok_s) {
            best_prefill_ = Best{rate, r.id, r.computed_prefill_tokens, r.time};
        }
    }
}

void Monitor::read_console_log() {
    if (!log_tail_) { return; }
    std::vector<std::string> lines = log_tail_->read_lines();
    if (lines.empty() && !log_tail_->reopened()) { return; }
    const std::lock_guard lock(mutex_);
    if (log_tail_->reopened()) {
        logs_.push_back(LogEntry{++log_seq_, unix_seconds(), LogLevel::Info,
                                 "-- monitor: " + log_tail_->path().string() + " was replaced; reading the new file --"});
    }
    for (const std::string& raw : lines) {
        LogLine line = parse_log_line(raw);
        if (line.time) { last_log_time_ = *line.time; }
        const double time = line.time.value_or(last_log_time_ != 0.0 ? last_log_time_ : unix_seconds());
        logs_.push_back(LogEntry{++log_seq_, time, line.level, std::move(line.message)});
    }
    while (logs_.size() > kLogLines) { logs_.pop_front(); }
}

std::string Monitor::build_state() const {
    Json state;
    const StaticInfo& info = sampler_.static_info();

    Json gpus_static = Json::array();
    for (const GpuInfo& g : info.gpus) { gpus_static.push_back({{"index", g.index}, {"name", g.name}}); }
    state["hardware_static"] = {
        {"gpus", gpus_static}, {"cpu_name", info.cpu_name}, {"driver", info.driver}, {"cuda", info.cuda}};

    Json gpus = Json::array();
    for (const GpuReading& g : gpus_) {
        gpus.push_back({{"index", g.index},
                        {"util", optional_json(g.util)},
                        {"mem_used", optional_json(g.mem_used)},
                        {"mem_total", optional_json(g.mem_total)},
                        {"power", optional_json(g.power)},
                        {"power_limit", optional_json(g.power_limit)},
                        {"temp", optional_json(g.temp)},
                        {"pcie_gen", optional_json(g.pcie_gen)},
                        {"pcie_gen_max", optional_json(g.pcie_gen_max)},
                        {"pcie_width", optional_json(g.pcie_width)}});
    }
    state["hardware"] = {{"gpus", gpus},
                         {"cpu", optional_json(host_.cpu)},
                         {"ram_used", optional_json(host_.ram_used)},
                         {"ram_total", optional_json(host_.ram_total)},
                         {"disk_read_mb", optional_json(host_.disk_read_mb)}};

    Json gpu_history = Json::array();
    for (const auto& [index, h] : gpu_history_) {
        gpu_history.push_back(
            {{"index", index}, {"util", series_json(h.util)}, {"power", series_json(h.power)}, {"temp", series_json(h.temp)}});
    }
    state["history"] = {{"tok_s", series_json(decode_rate_)},
                        {"prefill_tok_s", series_json(prefill_rate_)},
                        {"gpus", gpu_history}};

    const auto gauge = [&](const char* name) { return optional_json(metrics_.sum(name)); };
    const std::optional<double> decode_now  = decode_rate_.empty() ? std::nullopt : decode_rate_.back();
    const std::optional<double> prefill_now = prefill_rate_.empty() ? std::nullopt : prefill_rate_.back();
    const double running     = metrics_.sum("infernix_requests_running").value_or(0.0);
    const double prefilling  = metrics_.sum("infernix_requests_prefilling").value_or(0.0);
    std::string phase        = "offline";
    if (reachable_) {
        if (metrics_.sum("infernix_engine_ready").value_or(1.0) == 0.0) {
            phase = "loading";
        } else if (decode_now.value_or(0.0) > 0.0 || running > prefilling) {
            phase = "generating";
        } else if (prefill_now.value_or(0.0) > 0.0 || prefilling > 0.0) {
            phase = "reading";
        } else {
            phase = "idle";
        }
    }
    state["live"] = {{"state", phase},
                     {"tok_s", optional_json(decode_now)},
                     {"prefill_tok_s", optional_json(prefill_now)},
                     {"running", gauge("infernix_requests_running")},
                     {"prefilling", gauge("infernix_requests_prefilling")},
                     {"decode_ready", gauge("infernix_requests_decode_ready")},
                     {"waiting", gauge("infernix_requests_waiting")},
                     {"paused", gauge("infernix_requests_paused")},
                     {"max_concurrency", gauge("infernix_max_concurrency")},
                     {"max_context", gauge("infernix_max_context_tokens")}};

    Json window = nullptr, total = nullptr;
    if (!counters_.empty()) {
        const EngineCounters& a = counters_.front();
        const EngineCounters& b = counters_.back();
        // Rates need 100 lookups (experts), 1 prompt token or 20 drafts in the window to show.
        window = {{"seconds", b.time - a.time},
                  {"expert_hit", optional_json(ratio(b.expert_hits - a.expert_hits, b.expert_routed - a.expert_routed, 100))},
                  {"prefix_hit", optional_json(ratio(b.prompt_cached_tokens - a.prompt_cached_tokens,
                                                     b.prompt_tokens - a.prompt_tokens, 1))},
                  {"accept", optional_json(ratio(b.accepted - a.accepted, b.drafted - a.drafted, 20))}};
        total = {{"expert_hit", optional_json(ratio(b.expert_hits, b.expert_routed, 100))},
                 {"prefix_hit", optional_json(ratio(b.prompt_cached_tokens, b.prompt_tokens, 1))},
                 {"accept", optional_json(ratio(b.accepted, b.drafted, 20))},
                 {"decode_tokens", b.decode_tokens},
                 {"prefill_tokens", b.prefill_tokens}};
    }
    const std::optional<double> started = metrics_.sum("infernix_server_start_time_seconds");
    state["engine"] = {{"url", options_.engine_url},
                       {"reachable", reachable_},
                       {"model", metrics_.label("infernix_model_info", "model_name")},
                       {"spec_backend", metrics_.label("infernix_model_info", "speculative_backend")},
                       {"uptime_s", started ? Json(unix_seconds() - *started) : Json(nullptr)},
                       {"window", window},
                       {"total", total}};

    state["placement"] = placement_.empty() ? Json(nullptr) : Json::parse(placement_, nullptr, false);

    Json requests = Json::array();
    for (const FinishedRequest& r : requests_) {
        requests.push_back({{"time", r.time},
                            {"id", r.id},
                            {"protocol", r.protocol},
                            {"finish", r.finish},
                            {"prompt_tokens", r.prompt_tokens},
                            {"read_tokens", r.computed_prefill_tokens},
                            {"reused", r.prefix_hit_tokens},
                            {"output_tokens", r.completion_tokens},
                            {"ttft_s", r.ttft_s},
                            {"prefill_s", r.prefill_s},
                            {"decode_s", r.decode_s},
                            {"duration_s", r.total_s},
                            {"drafted", r.drafted_tokens},
                            {"accepted", r.accepted_tokens}});
    }
    state["requests"] = requests;
    const auto best_json = [](const std::optional<Best>& best) {
        return best ? Json{{"tok_s", best->tok_s}, {"id", best->id}, {"tokens", best->tokens}, {"time", best->time}}
                    : Json(nullptr);
    };
    state["best"]    = {{"decode", best_json(best_decode_)}, {"prefill", best_json(best_prefill_)}};
    state["sources"] = {
        {"request_log", options_.request_log.empty() ? Json(nullptr) : Json(options_.request_log.string())},
        {"log_file", options_.log_file.empty() ? Json(nullptr) : Json(options_.log_file.string())}};
    return state.dump();
}

std::string Monitor::state_json() const {
    const std::lock_guard lock(mutex_);
    return state_.empty() ? std::string("{}") : state_;
}

std::string Monitor::logs_json(std::uint64_t after) const {
    const std::lock_guard lock(mutex_);
    Json lines = Json::array();
    for (const LogEntry& e : logs_) {
        if (e.seq <= after) { continue; }
        lines.push_back({{"t", e.time}, {"lvl", level_name(e.level)}, {"msg", e.message}});
    }
    return Json{{"last", log_seq_}, {"lines", lines}}.dump();
}

} // namespace infernix::monitor
