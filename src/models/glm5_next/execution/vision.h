#pragma once

// The GLM-5.3-Flash vision tower (docs/maintainer/glm5_3-flash-design.md §21.1): patch embedding,
// 24 blocks, the post norm, the m x m downsample and the gated merger, run as a resumable
// layer-major pass over the items of one prompt. Every Op runs per item with its own patch count,
// so an item's embeddings are the same alone or beside other items. Program-free: the caller owns
// the arena, the patches and the outputs.

#include "core/device.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/glm5_next/config.h"
#include "models/glm5_next/execution/parameters.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::common {
class VisionWeightStream;
}

namespace infernix::models::glm5_next::execution {

// One item of a pass: its patches in merge-block order (BF16 [patch_width, P], pageable host
// memory), the (h, w) patch coordinates matching them (I32 axis-major [2, P]: the P row
// coordinates, then the P column coordinates) and its output BF16 [O, P / m^2].
struct VisionTowerItem {
    std::span<const std::uint16_t> patches;
    std::span<const std::int32_t> positions;
    Tensor output;
};

// Device layout of a pass: each item's residual and coordinates for the whole pass, then one
// shared scope for whichever stage runs, sized for the largest item. The MLP and the merger run
// in column tiles of at most kVisionMlpTile patches (kVisionMlpTile / m^2 merged tokens), so a
// large image's transient bytes grow with its attention buffers only.
inline constexpr std::int32_t kVisionMlpTile = 4096;

struct VisionPassLayout {
    struct Item {
        TensorRegion x, positions;
    };

    std::vector<Item> items;
    // Embedding: the patches. Attention: four [H, P] planes. MLP: the normed input, gate/up and
    // the product of one tile. Merger: the normed residual and the tile's intermediates.
    TensorRegion patches, plane[4], mlp_input, gate_up, product, merged, projected, merger_gate_up,
        merger_product;
    LayoutRegion scratch; // every linear's split-K workspace (one at a time)
    std::int32_t max_patches = 0;
    std::size_t bytes        = 0;
};

[[nodiscard]] VisionPassLayout plan_vision_pass(const VisionConfig& config,
                                                const VisionParameters& parameters,
                                                std::span<const std::size_t> item_patches);

// The tower's parameters rebased onto a weight stream's staging (vision offload): every layer's
// weights point at the slot that holds the layer once the stream admits it; the joined gate/up
// biases stay resident.
[[nodiscard]] VisionParameters vision_window_weights(const common::VisionWeightStream& stream,
                                                     const VisionParameters& host);

// Stage 0 embeds every item, stage 1 + l runs block l for every item, stage depth + 1 merges every
// item into its output. With a weight stream (vision offload), `parameters` is its
// vision_window_weights() view and each stage first waits for its weights' upload; the caller
// calls the stream's finish() after the last stage.
class VisionTowerPass {
public:
    VisionTowerPass(DeviceContext& device, const VisionConfig& config,
                    const VisionParameters& parameters, std::vector<VisionTowerItem> items,
                    DeviceSpan arena, const VisionPassLayout& layout,
                    common::VisionWeightStream* weight_stream = nullptr);

    [[nodiscard]] static std::uint32_t stages(std::uint32_t depth) noexcept { return depth + 2; }

    [[nodiscard]] std::uint32_t next_stage() const noexcept { return next_; }

    [[nodiscard]] bool done() const noexcept { return next_ == stages(config_.depth); }

    // Enqueues stages [next_stage(), end_stage) on the compute stream.
    void advance(std::uint32_t end_stage);

    // An item's residual stream BF16 [H, P] between stages.
    [[nodiscard]] Tensor residual(std::size_t item) const {
        return layout_.items.at(item).x.bind(arena_);
    }

private:
    void embed(std::size_t item);
    void block(std::uint32_t layer, std::size_t item);
    void merge(std::size_t item);

    DeviceContext& device_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
    std::vector<VisionTowerItem> items_;
    DeviceSpan arena_;
    const VisionPassLayout& layout_;
    common::VisionWeightStream* stream_;
    std::uint32_t next_ = 0;
};

// The patch coordinates of an h x w grid in merge-block order (I32 axis-major [2, h*w]): the
// row-major grid regrouped so each m x m block's patches are consecutive. A video's frames are
// separate items (each two-frame group attends only to itself).
[[nodiscard]] std::vector<std::int32_t>
vision_patch_positions(std::uint32_t height, std::uint32_t width, std::uint32_t merge);

} // namespace infernix::models::glm5_next::execution
