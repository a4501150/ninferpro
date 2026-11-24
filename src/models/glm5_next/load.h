#pragma once

#include "models/glm5_next/model.h"

#include <filesystem>
#include <memory>

namespace infernix::artifact {
class Reader;
}

namespace infernix::models::glm5_next {

// Cold load plan; it borrows its Reader until materialization.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // Pinned bytes of the expert banks and of the model's other pinned weights (token embedding).
    [[nodiscard]] std::uint64_t pinned_expert_bytes() const;
    [[nodiscard]] std::uint64_t pinned_other_bytes() const;
    // Device bytes of the dense weights, and the tokenizer's public token count.
    [[nodiscard]] std::uint64_t device_bytes() const;
    [[nodiscard]] std::uint32_t public_token_count() const;
    // The Host memory the pinned block must leave free at materialization, and the pins the
    // caller makes after it (both checked before anything is read).
    void set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

// True when the artifact's text component is Glm5Next.
[[nodiscard]] bool is_glm5_next(const artifact::Reader& reader);

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace infernix::models::glm5_next
