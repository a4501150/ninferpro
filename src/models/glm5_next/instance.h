#pragma once

// The Glm5Next (GLM-5.3-Flash) model instance behind the public Engine: its resident model,
// native parameters, frontend and Program. The Engine drives it with the common EngineCore over
// the HybridResourceManager surface.

#include "infernix/types.h"
#include "models/glm5_next/execution/parameters.h"
#include "models/glm5_next/model.h"
#include "models/glm5_next/program/program.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/engine_backend.h"

#include <filesystem>
#include <memory>
#include <string>

namespace infernix {
struct DeviceContext;
}

namespace infernix::runtime {

struct Glm5NextInstance {
    using ModelContract = models::glm5_next::RuntimeTypes;

    std::unique_ptr<models::glm5_next::Model> model;
    const models::glm5_next::execution::Parameters parameters;
    models::glm5_next::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::glm5_next::Program> program;

    Glm5NextInstance(std::unique_ptr<models::glm5_next::Model> model, const EngineOptions& options);
    ~Glm5NextInstance();
    Glm5NextInstance(const Glm5NextInstance&)            = delete;
    Glm5NextInstance& operator=(const Glm5NextInstance&) = delete;

    // /admin/expert-placement (see types.h): forwarded to the Program's shared placement runtime;
    // a switch lands at the next maintenance boundary, the status reports its progress. Keys
    // count the MoE layers only: a placement file's layer l is MoE layer l (model layer
    // dense_layers + l).
    [[nodiscard]] ExpertPlacementReport expert_placement_status() const;
    [[nodiscard]] bool request_expert_placement(const ExpertPlacementSwitch& request, std::string* error = nullptr);
    [[nodiscard]] bool export_expert_placement(const std::filesystem::path& path, std::string* error = nullptr);
    [[nodiscard]] ExpertPlacementPreview preview_expert_placement(const std::string& experts_spec,
                                                                  const ExpertPlacementReport& now) const;
};

// The family's Engine backend (models::construct_engine_backend); options take the family's
// resolved values.
[[nodiscard]] ConstructedBackend construct_glm5_next_backend(EngineOptions& options, DeviceContext& device);

} // namespace infernix::runtime
