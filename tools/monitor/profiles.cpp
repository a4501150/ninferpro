#include "profiles.h"

#include "product/config/config_file.h"

#include <toml.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <system_error>

namespace infernix::monitor {
namespace {

std::string trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) { text.remove_prefix(1); }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) { text.remove_suffix(1); }
    return std::string(text);
}

// Reads a key with [serve] overriding top-level, matching infernix-serve --config precedence.
const toml::node* lookup_serve_key(const toml::table& root, std::string_view key) {
    if (const toml::table* serve = root["serve"].as_table()) {
        if (const toml::node* node = serve->get(key)) { return node; }
    }
    return root.get(key);
}

std::optional<std::string> get_string(const toml::table& root, std::string_view key) {
    const toml::node* node = lookup_serve_key(root, key);
    if (node == nullptr) { return std::nullopt; }
    if (const auto* s = node->as_string()) { return s->get(); }
    return std::nullopt;
}

std::optional<std::string> get_first_string(const toml::table& root, std::string_view key) {
    const toml::node* node = lookup_serve_key(root, key);
    if (node == nullptr) { return std::nullopt; }
    if (const auto* s = node->as_string()) { return s->get(); }
    if (const auto* arr = node->as_array()) {
        for (const toml::node& item : *arr) {
            if (const auto* s = item.as_string()) { return s->get(); }
        }
    }
    return std::nullopt;
}

std::optional<std::int64_t> get_int(const toml::table& root, std::string_view key) {
    const toml::node* node = lookup_serve_key(root, key);
    if (node == nullptr) { return std::nullopt; }
    if (const auto* n = node->as_integer()) { return n->get(); }
    return std::nullopt;
}

ModelProfile parse_profile_file(const std::filesystem::path& path) {
    ModelProfile p;
    p.name        = path.stem().string();
    std::error_code ec;
    p.config_path = std::filesystem::weakly_canonical(path, ec);
    if (ec) { p.config_path = std::filesystem::absolute(path, ec); }
    if (p.config_path.empty()) { p.config_path = path; }
    p.served_name = p.name;

    toml::table root;
    try {
        root = toml::parse_file(path.string());
    } catch (const toml::parse_error& err) {
        p.parse_error = std::string(err.description());
        return p;
    } catch (const std::exception& err) {
        p.parse_error = err.what();
        return p;
    }

    if (const auto model = get_string(root, "model")) {
        p.model_path      = product::expand_user_path(*model);
        p.artifact_exists = std::filesystem::exists(p.model_path, ec) && !ec;
    }
    if (const auto alias = get_first_string(root, "served-model-name")) {
        if (!alias->empty()) { p.served_name = *alias; }
    } else if (const auto id = get_string(root, "model-id")) {
        if (!id->empty()) { p.served_name = *id; }
    }
    if (const auto experts = get_string(root, "experts")) { p.default_experts = *experts; }
    p.default_shares = parse_experts_shares(p.default_experts);

    if (const auto host = get_string(root, "host")) { p.host = *host; }
    if (const auto port = get_int(root, "port"); port && *port >= 1 && *port <= 65535) {
        p.port = static_cast<int>(*port);
    }
    if (const auto kv = get_string(root, "kv-dtype")) { p.kv_dtype = *kv; }
    if (const auto spec = get_string(root, "spec")) { p.spec = *spec; }
    if (const auto ctx = get_int(root, "max-context"); ctx && *ctx > 0) {
        p.max_context = static_cast<std::uint64_t>(*ctx);
    }
    if (const auto conc = get_int(root, "max-concurrency"); conc && *conc > 0) {
        p.max_concurrency = static_cast<std::uint64_t>(*conc);
    }
    if (const auto rlog = get_string(root, "request-log-jsonl"); rlog && !rlog->empty()) {
        p.request_log_jsonl = product::expand_user_path(*rlog);
    }
    return p;
}

} // namespace

std::optional<ExpertsShares> parse_experts_shares(std::string_view spec) {
    const std::string cleaned = trim(spec);
    if (cleaned.empty() || cleaned == "auto") { return ExpertsShares{100, 0}; }

    std::optional<int> vram, ram, ssd;
    std::size_t at = 0;
    while (at < cleaned.size()) {
        const std::size_t comma = cleaned.find(',', at);
        const std::string term =
            trim(std::string_view(cleaned).substr(at, comma == std::string::npos ? std::string::npos : comma - at));
        at = comma == std::string::npos ? cleaned.size() : comma + 1;
        if (term.empty()) { return std::nullopt; }
        const std::size_t eq = term.find('=');
        if (eq == std::string::npos) { return std::nullopt; }
        const std::string tier = trim(std::string_view(term).substr(0, eq));
        const std::string val  = trim(std::string_view(term).substr(eq + 1));

        int pct = -1;
        if (val == "all") {
            pct = 100;
        } else if (!val.empty() && val.back() == '%') {
            const std::string digits = val.substr(0, val.size() - 1);
            if (digits.empty() ||
                !std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
                return std::nullopt;
            }
            const long parsed = std::strtol(digits.c_str(), nullptr, 10);
            if (parsed < 0 || parsed > 100) { return std::nullopt; }
            pct = static_cast<int>(parsed);
        } else {
            return std::nullopt; // counts, not percentage shares
        }

        if (tier == "vram") {
            vram = pct;
        } else if (tier == "ram") {
            ram = pct;
        } else if (tier == "ssd") {
            ssd = pct;
        } else {
            return std::nullopt;
        }
    }

    if (!vram.has_value()) { return std::nullopt; }
    const int v = *vram;
    const int r = ram.value_or(0);
    const int s = ssd.value_or(100 - v - r);
    if (v < 1 || r < 0 || s < 0 || v + r + s != 100) { return std::nullopt; }
    return ExpertsShares{v, r};
}

std::vector<ModelProfile> scan_profiles(const std::filesystem::path& configs_dir) {
    std::vector<ModelProfile> out;
    std::error_code ec;
    if (configs_dir.empty() || !std::filesystem::is_directory(configs_dir, ec) || ec) { return out; }
    for (const auto& entry : std::filesystem::directory_iterator(configs_dir, ec)) {
        if (ec) { break; }
        if (!entry.is_regular_file(ec) || ec) { continue; }
        const std::filesystem::path p = entry.path();
        const std::string filename    = p.filename().string();
        if (filename.empty() || filename.front() == '.' || p.extension() != ".toml") { continue; }
        out.push_back(parse_profile_file(p));
    }
    std::sort(out.begin(), out.end(), [](const ModelProfile& a, const ModelProfile& b) { return a.name < b.name; });
    return out;
}

std::string profile_engine_url(const ModelProfile& profile) {
    return "http://127.0.0.1:" + std::to_string(profile.port);
}

nlohmann::json profile_to_json(const ModelProfile& p) {
    nlohmann::json j{
        {"name", p.name},
        {"config_path", p.config_path.string()},
        {"model_path", p.model_path.string()},
        {"artifact_exists", p.artifact_exists},
        {"served_name", p.served_name},
        {"default_experts", p.default_experts},
        {"host", p.host},
        {"port", p.port},
        {"kv_dtype", p.kv_dtype},
        {"spec", p.spec},
        {"max_context", p.max_context ? nlohmann::json(*p.max_context) : nlohmann::json(nullptr)},
        {"max_concurrency", p.max_concurrency ? nlohmann::json(*p.max_concurrency) : nlohmann::json(nullptr)},
    };
    if (p.default_shares) {
        j["default_shares"] = {{"vram", p.default_shares->vram}, {"ram", p.default_shares->ram}};
    } else {
        j["default_shares"] = nullptr;
    }
    if (!p.parse_error.empty()) { j["error"] = p.parse_error; }
    return j;
}

} // namespace infernix::monitor
