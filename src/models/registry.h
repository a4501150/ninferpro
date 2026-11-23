#pragma once

#include <filesystem>
#include <string_view>

namespace infernix {
struct DeviceContext;
struct EngineOptions;
} // namespace infernix

namespace infernix::runtime {
struct FamilyOptionRules;
struct ConstructedBackend;
} // namespace infernix::runtime

// The only shared entry points that name the model families: the artifact probe, each family's
// option rules, and its Engine backend. Startup only.
namespace infernix::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

// The family whose config parser refines and refuses an artifact no probe claims.
inline constexpr Architecture kDefaultArchitecture = Architecture::Qwen3_5;

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

// The artifact's family: the Qwen4Exp probe, with the Qwen3.5 family as the default its config
// parser refines (Qwen3_5 vs Qwen3_5Moe) and where unknown architectures are refused. An
// unreadable artifact throws; the Engine reports it from the model construction.
[[nodiscard]] Architecture detect_artifact_architecture(const std::filesystem::path& artifact);

[[nodiscard]] const runtime::FamilyOptionRules& family_option_rules(Architecture architecture);

// Loads the artifact and builds the family's Engine backend for the options' purpose; the family
// may resolve further options (they are updated in place).
[[nodiscard]] runtime::ConstructedBackend
construct_engine_backend(Architecture architecture, EngineOptions& options, DeviceContext& device);

} // namespace infernix::models
