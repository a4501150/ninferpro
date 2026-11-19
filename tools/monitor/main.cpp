// infernix-monitor: a live status page for infernix-serve. It reads the engine (/metrics,
// /admin/expert-placement, the request log and the console log), switches its expert placement,
// and with --control starts and stops it; it can itself start, stop and restart independently of
// the server.

#include "monitor.h"
#include "monitor_web.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

using infernix::monitor::Monitor;
using infernix::monitor::MonitorOptions;
using Json = nlohmann::json;

struct Options {
    MonitorOptions monitor;
    std::string host = "127.0.0.1";
    int port         = 8090;
};

void usage(std::ostream& out) {
    out << "usage: infernix-monitor [options]\n"
           "  --engine URL         infernix-serve's base URL (default http://127.0.0.1:8000)\n"
           "  --api-key KEY        the engine's --api-key, sent as a bearer token\n"
           "  --request-log FILE   the engine's --request-log-jsonl file: the Requests table and\n"
           "                       the session bests\n"
           "  --log-file FILE      the engine's console log or --log-file: the Logs pane; a\n"
           "                       symlink is followed to each new target\n"
           "  --control CMD        the command that starts and stops the server\n"
           "                       (tools/monitor/README.md); without it the page has no Start\n"
           "                       or Stop\n"
           "  --host ADDR          address to serve the page on (default 127.0.0.1)\n"
           "  --port N             port to serve the page on (default 8090)\n";
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << arg << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--engine") {
            options.monitor.engine_url = value();
            while (!options.monitor.engine_url.empty() && options.monitor.engine_url.back() == '/') {
                options.monitor.engine_url.pop_back();
            }
        } else if (arg == "--api-key") {
            options.monitor.api_key = value();
        } else if (arg == "--request-log") {
            options.monitor.request_log = value();
        } else if (arg == "--log-file") {
            options.monitor.log_file = value();
        } else if (arg == "--control") {
            options.monitor.control = value();
            if (access(options.monitor.control.c_str(), X_OK) != 0) {
                std::cerr << "--control " << options.monitor.control << " is not an executable file\n";
                std::exit(2);
            }
        } else if (arg == "--host") {
            options.host = value();
        } else if (arg == "--port") {
            const std::string text = value();
            char* end              = nullptr;
            const long port        = std::strtol(text.c_str(), &end, 10);
            if (end == text.c_str() || *end != '\0' || port < 1 || port > 65535) {
                std::cerr << "--port must be 1-65535\n";
                std::exit(2);
            }
            options.port = static_cast<int>(port);
        } else if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            std::exit(0);
        } else {
            std::cerr << "unknown option " << arg << '\n';
            usage(std::cerr);
            std::exit(2);
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

} // namespace

int main(int argc, char** argv) {
    const Options options = parse(argc, argv);
    Monitor monitor(options.monitor);

    std::atomic<bool> running{true};
    std::thread sampler([&] {
        auto next = std::chrono::steady_clock::now();
        while (running.load()) {
            monitor.tick();
            next += std::chrono::seconds(1);
            const auto now = std::chrono::steady_clock::now();
            if (next < now) { next = now; } // a slow poll skips ticks rather than bursting
            while (running.load() && std::chrono::steady_clock::now() < next) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    });

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
    // Every POST changes the server: refuse what did not come from this page, and malformed JSON.
    const auto post = [&](const char* path, auto handler) {
        server.Post(path, [&, handler](const httplib::Request& req, httplib::Response& res) {
            if (!infernix::monitor::page_request_allowed(req.get_header_value("Host"), req.get_header_value("Origin"),
                                                         req.get_header_value("Content-Type"), names)) {
                reply(res, 403, {{"error", "only this page may change the server"}});
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
    // {"vram": V, "ram": R, "dry_run": bool}: passed on to POST /admin/expert-placement.
    post("/api/placement", [&](const Json& body, httplib::Response& res) {
        int vram = 0, ram = 0;
        if (!read_shares(body, vram, ram)) {
            reply(res, 400, {{"error", "vram (a whole percent) is required"}});
            return;
        }
        const auto result = monitor.switch_placement(vram, ram, body.value("dry_run", false));
        reply_raw(res, result.status, result.body);
    });
    // {"action": "start", "vram": V, "ram": R} (the placement to start with; without vram the
    // engine's default) or {"action": "stop"}.
    post("/api/control", [&](const Json& body, httplib::Response& res) {
        auto* controller = monitor.controller();
        if (controller == nullptr) {
            reply(res, 404, {{"error", "the monitor runs without --control"}});
            return;
        }
        const std::string action = body.value("action", "");
        if (action != "start" && action != "stop") {
            reply(res, 400, {{"error", "action must be start or stop"}});
            return;
        }
        std::optional<std::string> error;
        if (action == "stop") {
            error = controller->stop();
        } else {
            int vram = 100, ram = 0;
            std::optional<std::string> spec;
            if (!body.contains("vram") || read_shares(body, vram, ram)) { spec = infernix::monitor::experts_spec(vram, ram); }
            if (!spec) {
                reply(res, 400, {{"error", infernix::monitor::kSharesError}});
                return;
            }
            error = controller->start(*spec);
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

    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::cout << "infernix-monitor on http://" << options.host << ':' << options.port << " | engine "
              << options.monitor.engine_url << std::endl;
    const bool listened = server.listen(options.host, options.port);
    running.store(false);
    sampler.join();
    if (!listened) {
        std::cerr << "cannot listen on " << options.host << ':' << options.port << '\n';
        return 1;
    }
    return 0;
}
