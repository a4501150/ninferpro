#include "models/glm5_next/execution/vision.h"

#include "core/arena.h"
#include "core/nvtx.h"
#include "models/common/vision/weight_stream.h"
#include "infernix/ops/add_bias.h"
#include "infernix/ops/gelu.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/residual_add.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/softmax_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace infernix::models::glm5_next::execution {
namespace {

constexpr std::size_t kAlignment = 256;
constexpr float kMergerNormEps   = 1.0e-5F; // nn.LayerNorm's default

std::int32_t dim(std::uint64_t v) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Glm5Next vision: dimension exceeds int32");
    }
    return static_cast<std::int32_t>(v);
}

void copy_host(const void* source, const Tensor& target, cudaStream_t stream) {
    if (target.bytes() == 0) { return; }
    CUDA_CHECK(
        cudaMemcpyAsync(target.data, source, target.bytes(), cudaMemcpyHostToDevice, stream));
}

// The first `columns` columns of a contiguous [rows, capacity] region.
Tensor columns_of(const Tensor& storage, std::int32_t columns) {
    return Tensor(storage.data, storage.dtype, {storage.ne[0], columns});
}

// Columns [first, first + count) of a contiguous [rows, columns] tensor.
Tensor column_slice(const Tensor& t, std::int32_t first, std::int32_t count) {
    return Tensor(static_cast<std::byte*>(t.data) + static_cast<std::size_t>(t.nb[1]) * first,
                  t.dtype, {t.ne[0], count});
}

// Rows [first, first + rows) of every column of a contiguous [R, columns] tensor (strided).
Tensor row_block(const Tensor& t, std::int32_t first, std::int32_t rows) {
    Tensor out(static_cast<std::byte*>(t.data) + static_cast<std::size_t>(t.nb[0]) * first, t.dtype,
               {rows, t.ne[1]});
    out.nb[1] = t.nb[1];
    out.nb[2] = t.nb[2];
    out.nb[3] = t.nb[3];
    return out;
}

void project(const Tensor& x, const LinearParameters& p, Tensor& out, DeviceSpan scratch,
             cudaStream_t stream) {
    WorkspaceArena arena(scratch);
    ops::linear(x, p.weight, out, p.policy, arena, stream);
}

std::size_t capacity(const LinearParameters& p, std::int32_t first, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                first, last);
}

} // namespace

std::vector<std::int32_t> vision_patch_positions(std::uint32_t height, std::uint32_t width,
                                                 std::uint32_t merge) {
    if (height == 0 || width == 0 || height % merge != 0 || width % merge != 0) {
        throw std::invalid_argument("Glm5Next vision: the grid must be whole merge blocks");
    }
    const std::size_t patches = std::size_t(height) * width;
    std::vector<std::int32_t> out(2 * patches);
    std::size_t at = 0;
    for (std::uint32_t by = 0; by < height / merge; ++by) {
        for (std::uint32_t bx = 0; bx < width / merge; ++bx) {
            for (std::uint32_t my = 0; my < merge; ++my) {
                for (std::uint32_t mx = 0; mx < merge; ++mx, ++at) {
                    out[at]           = static_cast<std::int32_t>(by * merge + my);
                    out[patches + at] = static_cast<std::int32_t>(bx * merge + mx);
                }
            }
        }
    }
    return out;
}

VisionPassLayout plan_vision_pass(const VisionConfig& config, const VisionParameters& parameters,
                                  std::span<const std::size_t> item_patches) {
    const std::size_t area = config.merge_area();
    if (item_patches.empty()) {
        throw std::invalid_argument("Glm5Next vision: a pass needs an item");
    }
    std::size_t largest = 0;
    for (const std::size_t p : item_patches) {
        if (p == 0 || p % area != 0) {
            throw std::invalid_argument(
                "Glm5Next vision: an item's patches must be whole merge blocks");
        }
        largest = std::max(largest, p);
    }
    VisionPassLayout out;
    if (config.merged_width() < config.out_hidden_size) {
        throw std::invalid_argument(
            "Glm5Next vision: the merger reuses the merged input for its projection");
    }
    out.max_patches      = dim(largest);
    const std::int32_t P = out.max_patches, H = dim(config.hidden_size),
                       I = dim(config.intermediate_size), O = dim(config.out_hidden_size),
                       PI   = dim(config.projection_intermediate_size);
    const std::int32_t tile = std::min(P, kVisionMlpTile), merged_tile = tile / dim(area),
                       first = dim(area);
    std::size_t scratch      = capacity(parameters.patch_embedding, first, P);
    for (const auto& layer : parameters.layers) {
        for (const LinearParameters* p : {&layer.query, &layer.key, &layer.value, &layer.output}) {
            scratch = std::max(scratch, capacity(*p, first, P));
        }
        scratch = std::max(
            {scratch, capacity(layer.gate_up, first, tile), capacity(layer.down, first, tile)});
    }
    for (const LinearParameters* p : {&parameters.downsample, &parameters.merger_projection,
                                      &parameters.merger_gate_up, &parameters.merger_down}) {
        scratch = std::max(scratch, capacity(*p, 1, merged_tile));
    }
    LayoutBuilder builder;
    for (const std::size_t p : item_patches) {
        VisionPassLayout::Item item;
        item.x = builder.add_tensor(DType::BF16, {H, dim(p)}, kAlignment, "vision residual");
        item.positions =
            builder.add_tensor(DType::I32, {dim(p), 2}, kAlignment, "vision positions");
        out.items.push_back(item);
    }
    // A borrowed WorkspaceArena needs backing even when no linear uses scratch.
    out.scratch =
        builder.add(std::max<std::size_t>(1, scratch), kAlignment, "vision linear scratch");
    {
        auto scope  = builder.scope();
        out.patches = builder.add_tensor(DType::BF16, {dim(config.patch_width()), P}, kAlignment,
                                         "vision patches");
    }
    {
        auto scope = builder.scope();
        for (auto& plane : out.plane) {
            plane = builder.add_tensor(DType::BF16, {H, P}, kAlignment, "vision attention plane");
        }
    }
    {
        auto scope    = builder.scope();
        out.mlp_input = builder.add_tensor(DType::BF16, {H, tile}, kAlignment, "vision MLP input");
        out.gate_up =
            builder.add_tensor(DType::BF16, {2 * I, tile}, kAlignment, "vision MLP gate/up");
        out.product = builder.add_tensor(DType::BF16, {I, tile}, kAlignment, "vision MLP product");
    }
    {
        auto scope = builder.scope();
        out.merged = builder.add_tensor(DType::BF16, {dim(config.merged_width()), merged_tile},
                                        kAlignment, "vision merged input");
        out.projected =
            builder.add_tensor(DType::BF16, {O, merged_tile}, kAlignment, "vision merger hidden");
        out.merger_gate_up = builder.add_tensor(DType::BF16, {2 * PI, merged_tile}, kAlignment,
                                                "vision merger gate/up");
        out.merger_product =
            builder.add_tensor(DType::BF16, {PI, merged_tile}, kAlignment, "vision merger product");
    }
    out.bytes = builder.finish(kAlignment, "vision pass");
    return out;
}

VisionParameters vision_window_weights(const common::VisionWeightStream& stream,
                                       const VisionParameters& host) {
    VisionParameters out       = host;
    out.patch_embedding.weight = stream.prelude(host.patch_embedding.weight);
    out.patch_embedding_bias   = stream.prelude(host.patch_embedding_bias);
    for (std::size_t i = 0; i < host.layers.size(); ++i) {
        const auto l                     = static_cast<std::uint32_t>(i);
        const VisionBlockParameters& src = host.layers[i];
        VisionBlockParameters& dst       = out.layers[i];
        dst.norm1                        = stream.layer(l, src.norm1);
        dst.norm2                        = stream.layer(l, src.norm2);
        dst.query.weight                 = stream.layer(l, src.query.weight);
        dst.key.weight                   = stream.layer(l, src.key.weight);
        dst.value.weight                 = stream.layer(l, src.value.weight);
        dst.query_bias                   = stream.layer(l, src.query_bias);
        dst.key_bias                     = stream.layer(l, src.key_bias);
        dst.value_bias                   = stream.layer(l, src.value_bias);
        dst.query_norm                   = stream.layer(l, src.query_norm);
        dst.key_norm                     = stream.layer(l, src.key_norm);
        dst.output.weight                = stream.layer(l, src.output.weight);
        dst.output_bias                  = stream.layer(l, src.output_bias);
        dst.gate_up.weight               = stream.layer(l, src.gate_up.weight);
        dst.down.weight                  = stream.layer(l, src.down.weight);
        dst.down_bias                    = stream.layer(l, src.down_bias);
    }
    out.post_norm                = stream.merger(host.post_norm);
    out.downsample.weight        = stream.merger(host.downsample.weight);
    out.downsample_bias          = stream.merger(host.downsample_bias);
    out.merger_projection.weight = stream.merger(host.merger_projection.weight);
    out.merger_norm              = stream.merger(host.merger_norm);
    out.merger_norm_bias         = stream.merger(host.merger_norm_bias);
    out.merger_gate_up.weight    = stream.merger(host.merger_gate_up.weight);
    out.merger_down.weight       = stream.merger(host.merger_down.weight);
    return out;
}

VisionTowerPass::VisionTowerPass(DeviceContext& device, const VisionConfig& config,
                                 const VisionParameters& parameters,
                                 std::vector<VisionTowerItem> items, DeviceSpan arena,
                                 const VisionPassLayout& layout,
                                 common::VisionWeightStream* weight_stream)
    : device_(device), config_(config), parameters_(parameters), items_(std::move(items)),
      arena_(arena), layout_(layout), stream_(weight_stream) {
    if (items_.size() != layout_.items.size() || arena_.data == nullptr ||
        arena_.bytes < layout_.bytes || parameters_.layers.size() != config_.depth) {
        throw std::invalid_argument(
            "Glm5Next vision: the pass's items or arena disagree with its layout");
    }
    for (std::size_t i = 0; i < items_.size(); ++i) {
        const VisionTowerItem& item = items_[i];
        const auto P = static_cast<std::size_t>(layout_.items[i].x.bind(arena_).ne[1]);
        if (item.patches.size() != P * config_.patch_width() || item.positions.size() != 2 * P ||
            item.output.dtype != DType::BF16 || item.output.ne[0] != dim(config_.out_hidden_size) ||
            static_cast<std::size_t>(item.output.ne[1]) * config_.merge_area() != P ||
            !item.output.is_contiguous()) {
            throw std::invalid_argument(
                "Glm5Next vision: an item does not match its layout or output");
        }
    }
}

void VisionTowerPass::advance(std::uint32_t end_stage) {
    end_stage = std::min(end_stage, stages(config_.depth));
    for (; next_ < end_stage; ++next_) {
        if (stream_ != nullptr) {
            if (next_ == 0) {
                stream_->prelude_ready(device_.stream);
            } else if (next_ <= config_.depth) {
                stream_->arrive(next_ - 1, device_.stream);
            } else {
                stream_->merger_ready(device_.stream);
            }
        }
        for (std::size_t i = 0; i < items_.size(); ++i) {
            if (next_ == 0) {
                embed(i);
            } else if (next_ <= config_.depth) {
                block(next_ - 1, i);
            } else {
                merge(i);
            }
        }
    }
}

void VisionTowerPass::embed(std::size_t i) {
    const cudaStream_t s        = device_.stream;
    const VisionTowerItem& item = items_[i];
    Tensor x                    = layout_.items[i].x.bind(arena_);
    const std::int32_t P        = x.ne[1];
    nvtx::ScopedRange range(nvtx::Name::VisionPatchEmbedding, nvtx::Category::Vision,
                            static_cast<std::uint64_t>(P));
    copy_host(item.positions.data(), layout_.items[i].positions.bind(arena_), s);
    const Tensor patches = columns_of(layout_.patches.bind(arena_), P);
    copy_host(item.patches.data(), patches, s);
    project(patches, parameters_.patch_embedding, x, layout_.scratch.bind(arena_), s);
    ops::add_bias(parameters_.patch_embedding_bias, x, s);
}

void VisionTowerPass::block(std::uint32_t l, std::size_t i) {
    const cudaStream_t s           = device_.stream;
    const VisionBlockParameters& b = parameters_.layers[l];
    const DeviceSpan scratch       = layout_.scratch.bind(arena_);
    Tensor x                       = layout_.items[i].x.bind(arena_);
    const std::int32_t P = x.ne[1], D = dim(config_.head_dim()), heads = dim(config_.num_heads),
                       I = dim(config_.intermediate_size);
    const float eps      = config_.rms_norm_eps;
    {
        nvtx::ScopedRange range(nvtx::Name::VisionAttention, nvtx::Category::Attention, l);
        // plane 0: the normed input, then q normed; 1: q, then k normed; 2: k, then the attended
        // heads; 3: v, then the output projection.
        Tensor plane[4];
        for (int k = 0; k < 4; ++k) { plane[k] = columns_of(layout_.plane[k].bind(arena_), P); }
        ops::rmsnorm(x, b.norm1, eps, false, plane[0], s);
        project(plane[0], b.query, plane[1], scratch, s);
        ops::add_bias(b.query_bias, plane[1], s);
        project(plane[0], b.key, plane[2], scratch, s);
        ops::add_bias(b.key_bias, plane[2], s);
        project(plane[0], b.value, plane[3], scratch, s);
        ops::add_bias(b.value_bias, plane[3], s);
        const auto heads_of = [&](const Tensor& t) {
            return Tensor(t.data, DType::BF16, {D, heads * P});
        };
        Tensor q_rows = heads_of(plane[0]), k_rows = heads_of(plane[1]);
        ops::rmsnorm(heads_of(plane[1]), b.query_norm, eps, false, q_rows, s);
        ops::rmsnorm(heads_of(plane[2]), b.key_norm, eps, false, k_rows, s);
        Tensor q(plane[0].data, DType::BF16, {D, heads, P}),
            k(plane[1].data, DType::BF16, {D, heads, P});
        const Tensor positions = layout_.items[i].positions.bind(arena_);
        ops::rope(positions, D, config_.rope_theta, q, k, device_.execution_view().on_stream(s));
        const Tensor v(plane[3].data, DType::BF16, {D, heads, P});
        Tensor attended(plane[2].data, DType::BF16, {D, heads, P});
        ops::packed_softmax_attention(q, k, v, {D, heads, heads},
                                      static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))),
                                      P, attended, s);
        project(plane[2], b.output, plane[3], scratch, s);
        ops::add_bias(b.output_bias, plane[3], s);
        ops::residual_add(plane[3], x, s);
    }
    nvtx::ScopedRange range(nvtx::Name::VisionMlp, nvtx::Category::PostMixer, l);
    for (std::int32_t first = 0; first < P; first += kVisionMlpTile) {
        const std::int32_t n = std::min(kVisionMlpTile, P - first);
        Tensor columns       = column_slice(x, first, n);
        Tensor input         = columns_of(layout_.mlp_input.bind(arena_), n);
        Tensor gate_up       = columns_of(layout_.gate_up.bind(arena_), n);
        Tensor product       = columns_of(layout_.product.bind(arena_), n);
        ops::rmsnorm(columns, b.norm2, eps, false, input, s);
        project(input, b.gate_up, gate_up, scratch, s);
        ops::add_bias(b.gate_up_bias, gate_up, s);
        ops::silu_mul_clamped(row_block(gate_up, 0, I), row_block(gate_up, I, I),
                              config_.swiglu_limit, product, s);
        project(product, b.down, input, scratch, s);
        ops::add_bias(b.down_bias, input, s);
        ops::residual_add(input, columns, s);
    }
}

void VisionTowerPass::merge(std::size_t i) {
    const cudaStream_t s     = device_.stream;
    const DeviceSpan scratch = layout_.scratch.bind(arena_);
    const Tensor x           = layout_.items[i].x.bind(arena_);
    Tensor& output           = items_[i].output;
    const std::int32_t area = dim(config_.merge_area()), V = output.ne[1],
                       H = dim(config_.hidden_size), PI = dim(config_.projection_intermediate_size),
                       tile = kVisionMlpTile / area;
    nvtx::ScopedRange range(nvtx::Name::VisionMerge, nvtx::Category::Vision,
                            static_cast<std::uint64_t>(V));
    for (std::int32_t first = 0; first < V; first += tile) {
        const std::int32_t n = std::min(tile, V - first);
        Tensor normed        = columns_of(layout_.merged.bind(arena_), n);
        Tensor normed_patches(normed.data, DType::BF16, {H, n * area});
        ops::rmsnorm(column_slice(x, first * area, n * area), parameters_.post_norm,
                     config_.rms_norm_eps, false, normed_patches, s);
        Tensor hidden = columns_of(layout_.projected.bind(arena_), n);
        project(normed, parameters_.downsample, hidden, scratch, s);
        ops::add_bias(parameters_.downsample_bias, hidden, s);
        Tensor projected(normed.data, DType::BF16, {hidden.ne[0], n}); // the merged input is spent
        project(hidden, parameters_.merger_projection, projected, scratch, s);
        ops::layer_norm(projected, parameters_.merger_norm, parameters_.merger_norm_bias,
                        kMergerNormEps, hidden, s);
        ops::gelu(hidden, ops::GeluMode::Exact, s);
        Tensor gate_up = columns_of(layout_.merger_gate_up.bind(arena_), n);
        Tensor product = columns_of(layout_.merger_product.bind(arena_), n);
        project(hidden, parameters_.merger_gate_up, gate_up, scratch, s);
        ops::silu_mul_clamped(row_block(gate_up, 0, PI), row_block(gate_up, PI, PI),
                              config_.swiglu_limit, product, s);
        Tensor out = column_slice(output, first, n);
        project(product, parameters_.merger_down, out, scratch, s);
    }
}

} // namespace infernix::models::glm5_next::execution
