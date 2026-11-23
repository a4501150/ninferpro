#pragma once

// The Qwen3.5 model instance behind the public Engine: its resident model, native parameters, the
// frontend it shares with the Qwen4Exp family, and its Program. The Engine drives it with the
// common EngineCore; its family's option rules live in models/qwen3_5/options.h and the shared,
// artifact-free checks in runtime/engine/engine_options.h.

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/engine_backend.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>
#include <optional>
#include <string>

namespace infernix {
struct DeviceContext;
}

namespace infernix::runtime {

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen_common::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

    // The prefix cache file the Generation core saves at its orderly stop.
    [[nodiscard]] std::string prefix_cache_save_notice(const EngineOptions& options) const;
    [[nodiscard]] std::optional<PrefixCacheSave> prefix_cache_shutdown_save() const;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

// The family's Engine backend (models::construct_engine_backend).
[[nodiscard]] ConstructedBackend construct_qwen3_5_backend(EngineOptions& options,
                                                           DeviceContext& device);

} // namespace infernix::runtime
