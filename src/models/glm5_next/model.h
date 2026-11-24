#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/glm5_next/config.h"
#include "models/glm5_next/frontend/resources.h"
#include "models/glm5_next/weights.h"
#include "models/common/vision/overlay_layout.h"
#include "models/offload/expert_store.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::models::glm5_next {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
    // The DFlash2 companion artifact's id (--draft-model), when one is loaded.
    std::optional<artifact::ArtifactId> draft_artifact_id;
};

// The routed experts of one MoE layer: the pinned host bank and each expert's stored multipliers.
// The banks are W4A16 (records without activation scales), so alpha holds the multipliers.
struct ExpertBank {
    ExpertBankPlanes planes;
    std::vector<ops::offloaded_moe::ExpertScales> scales; // [experts]
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    // One bank per MoE layer: bank b serves layer config().text.dense_layers + b.
    [[nodiscard]] std::span<const ExpertBank> expert_banks() const noexcept { return banks_; }
    // With LoadOptions::stream_experts: where each record lies in the artifact (the banks then
    // hold no records, only their layout and multipliers).
    [[nodiscard]] const std::optional<offload::ExpertStore>& expert_store() const noexcept { return expert_store_; }
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // With vision offload: the pinned tower and its staging layout (no evictable pool: the encode
    // lends its staging from the expert cache).
    [[nodiscard]] const std::optional<common::VisionOverlayAssets>&
    overlay_vision() const noexcept {
        return overlay_vision_;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
          std::vector<ExpertBank> banks, FrontendResources resources, InstanceInfo info,
          std::optional<common::VisionOverlayAssets> overlay_vision,
          artifact::MaterializedArtifact backing, std::optional<offload::ExpertStore> expert_store,
          std::optional<artifact::MaterializedArtifact> draft_backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    std::optional<artifact::MaterializedArtifact> draft_backing_; // the DFlash2 companion's weights
    std::optional<offload::ExpertStore> expert_store_; // the streamed banks' multipliers back banks_
    Config config_;
    LoadOptions options_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBank> banks_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<common::VisionOverlayAssets> overlay_vision_;
};

} // namespace infernix::models::glm5_next
