#pragma once

// The Qwen4Exp (Qwen3.8-Flash-Next) model instance behind the public Engine: its resident model,
// native parameters, the Qwen3.5 frontend it shares the tokenizer and chat template with, and its
// Program. The Engine drives it with the common EngineCore over the HybridResourceManager.

#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/program.h"
#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <filesystem>
#include <memory>

namespace infernix {
struct DeviceContext;
}

namespace infernix::runtime {

struct Qwen4ExpInstance {
    using ModelContract = models::qwen4_exp::RuntimeTypes;

    std::unique_ptr<models::qwen4_exp::Model> model;
    const models::qwen4_exp::execution::Parameters parameters;
    models::qwen4_exp::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen4_exp::Program> program;

    Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> model, const EngineOptions& options);
    ~Qwen4ExpInstance();
    Qwen4ExpInstance(const Qwen4ExpInstance&)            = delete;
    Qwen4ExpInstance& operator=(const Qwen4ExpInstance&) = delete;

    // /admin/expert-placement (see types.h): forwarded to the Program between rounds. The switch
    // lands at the next maintenance boundary; status reports progress.
    [[nodiscard]] ExpertPlacementReport expert_placement_status() const;
    [[nodiscard]] bool request_expert_placement(const ExpertPlacementSwitch& request, std::string* error = nullptr);
    [[nodiscard]] bool export_expert_placement(const std::filesystem::path& path, std::string* error = nullptr);

private:
    [[nodiscard]] const models::qwen4_exp::TextConfig& text_config_() const;
};

struct ConstructedQwen4Exp {
    std::unique_ptr<Qwen4ExpInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    EngineOptions options;
};

[[nodiscard]] ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device);

} // namespace infernix::runtime
