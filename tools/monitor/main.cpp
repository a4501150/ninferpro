// infernix-monitor: a live status page for a running infernix-serve. It only reads the engine
// (/metrics, /admin/expert-placement, the request log and the console log), so it can start,
// stop and restart independently of the server.

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

namespace {

using infernix::monitor::Monitor;
using infernix::monitor::MonitorOptions;

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
