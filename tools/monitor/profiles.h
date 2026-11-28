#pragma once

// Native TOML serving profile discovery for infernix-monitor. Each *.toml file under --configs is
// a standard `infernix-serve --config` file (top-level keys plus optional [serve] overrides).

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::monitor {

struct ExpertsShares {
    int vram = 100;
    int ram  = 0;
};

// Parses an --experts share spec ("vram=40%,ram=60%", "vram=100%", "vram=all", empty => 100/0)
// into whole percents for UI slider defaults. Returns nullopt for count specs or malformed specs.
std::optional<ExpertsShares> parse_experts_shares(std::string_view spec);

struct ModelProfile {
    std::string name;                  // file stem (e.g. "GLM-5.3-Flash")
    std::filesystem::path config_path; // absolute path to the .toml file
    std::filesystem::path model_path;  // expanded `model` path
    bool artifact_exists = false;
    std::string served_name;           // first served-model-name, else name
    std::string default_experts;       // raw `experts` value; empty => engine default
    std::optional<ExpertsShares> default_shares;
    std::string host = "0.0.0.0";
    int port         = 8000;
    std::string kv_dtype;
    std::string spec;
    std::optional<std::uint64_t> max_context;
    std::optional<std::uint64_t> max_concurrency;
    std::filesystem::path request_log_jsonl; // expanded if set in TOML
    std::string parse_error;                 // non-empty if TOML failed to parse
};

// Scans `configs_dir` for *.toml profiles, sorted alphabetically by name. Missing or unreadable
// directories return an empty list.
std::vector<ModelProfile> scan_profiles(const std::filesystem::path& configs_dir);

// Loopback base URL for polling a profile's infernix-serve instance.
std::string profile_engine_url(const ModelProfile& profile);

nlohmann::json profile_to_json(const ModelProfile& profile);

} // namespace infernix::monitor
