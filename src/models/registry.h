#pragma once

#include <filesystem>
#include <string_view>

namespace infernix::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

// The artifact's family: the Qwen4Exp probe, with the Qwen3.5 family as the default its config
// parser refines (Qwen3_5 vs Qwen3_5Moe) and where unknown architectures are refused. An
// unreadable artifact throws; the Engine reports it from the model construction.
[[nodiscard]] Architecture detect_artifact_architecture(const std::filesystem::path& artifact);

} // namespace infernix::models
