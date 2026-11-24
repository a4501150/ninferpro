#include "models/glm5_next/program/prefix/state_image.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::models::glm5_next::prefix {
namespace {

constexpr std::uint64_t kHeaderMagic   = 0x31584D49344D4C47ULL; // "GLM4IMX1"
constexpr std::uint32_t kHeaderVersion = 1;

// Header bytes: magic, version, fingerprint, frontier, flags, draft_frontier.
struct PackedHeader {
    std::uint64_t magic       = 0;
    std::uint32_t version     = 0;
    std::uint32_t flags       = 0; // bit 0 opener, bit 1 head_residual
    std::uint64_t fingerprint = 0;
    std::uint64_t image_bytes = 0;
    std::uint32_t frontier    = 0;
    std::uint32_t draft_frontier = 0;
};
static_assert(sizeof(PackedHeader) <= kStateImageHeaderBytes);

std::size_t align_up(std::size_t bytes) {
    return (bytes + kStateImageAlignment - 1U) / kStateImageAlignment * kStateImageAlignment;
}

std::uint64_t fnv1a(std::uint64_t hash, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (8 * i)) & 0xFFU;
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

// One copy per contiguous run between a Device range and the segmented Host image.
template <class Copy>
void for_each_segment_run(std::size_t offset, std::size_t bytes, std::size_t segment_bytes, Copy&& copy) {
    std::size_t done = 0;
    while (done < bytes) {
        const std::size_t at      = offset + done;
        const std::size_t segment = at / segment_bytes;
        const std::size_t within  = at % segment_bytes;
        const std::size_t run     = std::min(bytes - done, segment_bytes - within);
        copy(segment, within, done, run);
        done += run;
    }
}

} // namespace

StateImageSpec state_image_spec(const TextConfig& c, DraftRingGeometry draft_ring, std::size_t head_saved_bytes) {
    StateImageSpec spec;
    spec.layer_types         = c.layer_types;
    // The KDA state pool: the recurrence FP32 [key_head_dim, value_head_dim, value_heads] and the
    // convolution window BF16 [3 * width, kernel - 1] per layer and lane.
    spec.kda_recurrent_bytes = static_cast<std::size_t>(c.kda.head_dim) * c.kda.head_dim * c.kda.heads *
                               sizeof(float);
    spec.kda_conv_bytes      = static_cast<std::size_t>(3ULL * c.kda.width()) * (c.kda.conv_kernel - 1U) * 2U;
    // The indexer's ring: one lane's recent keys, then gates, per MLA block (BF16 [head_dim, 64]).
    spec.index_bytes         = static_cast<std::size_t>(c.indexer.head_dim) * 64U * 2U;
    spec.head_saved_bytes    = head_saved_bytes;
    spec.draft_ring          = draft_ring;
    if (spec.kda_conv_bytes == 0 || spec.index_bytes == 0) {
        throw std::invalid_argument("Glm5Next state image: a state part has no bytes");
    }
    return spec;
}

StateImageLayout plan_state_image(const StateImageSpec& spec) {
    if (spec.layer_types.empty()) { throw std::invalid_argument("Glm5Next state image: no decoder layers"); }
    StateImageLayout layout;
    layout.spec           = spec;
    std::size_t offset    = 0;
    const auto add_part   = [&](StateImagePart::Kind kind, std::uint32_t index, std::size_t bytes) {
        if (bytes == 0) { throw std::invalid_argument("Glm5Next state image: a part has no bytes"); }
        layout.parts.push_back(StateImagePart{kind, index, offset, bytes});
        offset = align_up(offset + bytes);
    };
    add_part(StateImagePart::Kind::Header, 0, kStateImageHeaderBytes);
    std::uint32_t kda = 0, mla = 0;
    for (std::uint32_t layer = 0; layer < spec.layer_types.size(); ++layer) {
        layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
        if (spec.layer_types[layer] == MixerKind::Kda) {
            add_part(StateImagePart::Kind::KdaRecurrent, kda, spec.kda_recurrent_bytes);
            add_part(StateImagePart::Kind::KdaConv, kda, spec.kda_conv_bytes);
            ++kda;
        } else {
            add_part(StateImagePart::Kind::IndexKeys, mla, spec.index_bytes);
            add_part(StateImagePart::Kind::IndexGates, mla, spec.index_bytes);
            ++mla;
        }
    }
    layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
    for (std::uint32_t layer = 0; layer < spec.draft_ring.layers; ++layer) {
        add_part(StateImagePart::Kind::DraftRingK, layer, spec.draft_ring.k_bytes);
        add_part(StateImagePart::Kind::DraftRingV, layer, spec.draft_ring.v_bytes);
    }
    // The zero-token resume's saved head residual joins the last group: it is read only by the
    // resume hook, after the restore completes, so no forward pass waits on it.
    if (spec.head_saved_bytes != 0) { add_part(StateImagePart::Kind::HeadSaved, 0, spec.head_saved_bytes); }
    layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
    layout.image_bytes = offset;
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (const StateImagePart& part : layout.parts) {
        hash = fnv1a(hash, static_cast<std::uint64_t>(part.kind));
        hash = fnv1a(hash, part.index);
        hash = fnv1a(hash, part.offset);
        hash = fnv1a(hash, part.bytes);
    }
    layout.fingerprint = hash;
    return layout;
}

void write_state_image_header(const StateImageLayout& layout, const StateImageHeader& header,
                              std::byte* image_begin) {
    PackedHeader packed;
    packed.magic          = kHeaderMagic;
    packed.version        = kHeaderVersion;
    packed.flags          = (header.opener ? 1U : 0U) | (header.head_residual ? 2U : 0U);
    packed.fingerprint    = layout.fingerprint;
    packed.image_bytes    = layout.image_bytes;
    packed.frontier       = header.frontier;
    packed.draft_frontier = header.draft_frontier;
    std::memset(image_begin, 0, kStateImageHeaderBytes);
    std::memcpy(image_begin, &packed, sizeof(packed));
}

std::optional<StateImageHeader> read_state_image_header(const StateImageLayout& layout,
                                                        const std::byte* image_begin) {
    PackedHeader packed;
    std::memcpy(&packed, image_begin, sizeof(packed));
    if (packed.magic != kHeaderMagic || packed.version != kHeaderVersion ||
        packed.fingerprint != layout.fingerprint || packed.image_bytes != layout.image_bytes ||
        (packed.flags & ~3U) != 0) {
        return std::nullopt;
    }
    StateImageHeader header;
    header.frontier       = packed.frontier;
    header.opener         = (packed.flags & 1U) != 0;
    header.head_residual  = (packed.flags & 2U) != 0;
    header.draft_frontier = packed.draft_frontier;
    return header;
}

LaneStateImage::LaneStateImage(const StateImageLayout& layout, const LaneStateBuffers& buffers)
    : layout_(&layout), buffers_(buffers) {
    const StateImageSpec& spec = layout.spec;
    std::uint32_t kda_layers = 0, mla_layers = 0;
    for (const MixerKind kind : spec.layer_types) { ++(kind == MixerKind::Kda ? kda_layers : mla_layers); }
    if (buffers.lanes == 0 || buffers.index_ring == nullptr || buffers.index_plane_bytes == 0 ||
        (spec.head_saved_bytes != 0 && buffers.head_saved == nullptr)) {
        throw std::invalid_argument("Glm5Next state image: lane buffers are missing");
    }
    if (buffers.ring_k.size() != spec.draft_ring.layers || buffers.ring_v.size() != spec.draft_ring.layers ||
        std::any_of(buffers.ring_k.begin(), buffers.ring_k.end(), [](const std::byte* p) { return p == nullptr; }) ||
        std::any_of(buffers.ring_v.begin(), buffers.ring_v.end(), [](const std::byte* p) { return p == nullptr; })) {
        throw std::invalid_argument("Glm5Next state image: the DFlash2 ring does not match the layout");
    }
    if (kda_layers > 0) {
        if (buffers.kda == nullptr || buffers.kda->layer_count() != kda_layers ||
            buffers.kda->slot_count() < static_cast<std::int32_t>(buffers.lanes) ||
            buffers.kda->recurrent_slot(0, 0).bytes() != spec.kda_recurrent_bytes ||
            buffers.kda->conv_slot(0, 0).bytes() != spec.kda_conv_bytes) {
            throw std::invalid_argument("Glm5Next state image: the KDA state pool does not match the layout");
        }
    }
}

std::byte* LaneStateImage::device_part(const StateImagePart& part, std::uint32_t lane) const {
    check_lane(lane);
    const StateImageSpec& spec = layout_->spec;
    const auto slot = static_cast<std::int32_t>(lane);
    switch (part.kind) {
    case StateImagePart::Kind::Header: break;
    case StateImagePart::Kind::KdaRecurrent:
        return static_cast<std::byte*>(buffers_.kda->recurrent_slot(part.index, slot).data);
    case StateImagePart::Kind::KdaConv:
        return static_cast<std::byte*>(buffers_.kda->conv_slot(part.index, slot).data);
    case StateImagePart::Kind::IndexKeys:
        return buffers_.index_ring + static_cast<std::size_t>(part.index) * 2U * buffers_.index_plane_bytes +
               static_cast<std::size_t>(lane) * spec.index_bytes;
    case StateImagePart::Kind::IndexGates:
        return buffers_.index_ring +
               (static_cast<std::size_t>(part.index) * 2U + 1U) * buffers_.index_plane_bytes +
               static_cast<std::size_t>(lane) * spec.index_bytes;
    case StateImagePart::Kind::HeadSaved:
        return buffers_.head_saved + static_cast<std::size_t>(lane) * spec.head_saved_bytes;
    case StateImagePart::Kind::DraftRingK:
        return buffers_.ring_k.at(part.index) + static_cast<std::size_t>(lane) * spec.draft_ring.k_bytes;
    case StateImagePart::Kind::DraftRingV:
        return buffers_.ring_v.at(part.index) + static_cast<std::size_t>(lane) * spec.draft_ring.v_bytes;
    }
    throw std::logic_error("Glm5Next state image: the header has no Device bytes");
}

void LaneStateImage::check_lane(std::uint32_t lane) const {
    if (lane >= buffers_.lanes) { throw std::out_of_range("Glm5Next state image: lane out of range"); }
}

void LaneStateImage::check_image(std::size_t segments, std::size_t segment_bytes) const {
    if (segment_bytes == 0 || segments * segment_bytes < layout_->image_bytes) {
        throw std::invalid_argument("Glm5Next state image: the Host segments do not hold an image");
    }
}

void LaneStateImage::copy_lane_to_host(std::uint32_t lane, const HostImage& image, cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    for (std::size_t p = 1; p < layout_->parts.size(); ++p) {
        const StateImagePart& part = layout_->parts[p];
        const std::byte* source    = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, source + done, run,
                                                            cudaMemcpyDefault, stream));
                             });
    }
}

void LaneStateImage::copy_group_to_host(std::uint32_t group, std::uint32_t lane, const HostImage& image,
                                        cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    if (group >= layout_->groups()) { throw std::out_of_range("Glm5Next state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        const std::byte* source    = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, source + done, run,
                                                            cudaMemcpyDefault, stream));
                             });
    }
}

void LaneStateImage::copy_lane_to_packed(std::uint32_t lane, std::byte* packed, cudaStream_t stream) const {
    for (std::size_t p = 1; p < layout_->parts.size(); ++p) {
        const StateImagePart& part = layout_->parts[p];
        CUDA_CHECK(cudaMemcpyAsync(packed + part.offset, device_part(part, lane), part.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void LaneStateImage::copy_packed_to_host(const std::byte* packed, const HostImage& image, cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    for_each_segment_run(0, layout_->image_bytes, image.segment_bytes,
                         [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                             CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, packed + done, run,
                                                        cudaMemcpyDefault, stream));
                         });
}

void LaneStateImage::copy_group_from_host(std::uint32_t group, const HostImageConst& image, std::uint32_t lane,
                                          cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    if (group >= layout_->groups()) { throw std::out_of_range("Glm5Next state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        std::byte* destination     = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(destination + done, image.segments[segment] + within, run,
                                                            cudaMemcpyDefault, stream));
                             });
    }
}

void LaneStateImage::copy_group_from_packed(std::uint32_t group, const std::byte* packed, std::uint32_t lane,
                                            cudaStream_t stream) const {
    if (group >= layout_->groups()) { throw std::out_of_range("Glm5Next state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        CUDA_CHECK(cudaMemcpyAsync(device_part(part, lane), packed + part.offset, part.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

KvPageGeometry kv_page_geometry(const TextConfig& c, KvCacheStorage storage) {
    KvPageGeometry out;
    out.kv_layers = c.mla_layers;
    for (std::uint32_t l = 0; l < out.kv_layers; ++l) {
        if (storage == KvCacheStorage::Int8Group64) {
            out.geometry.planes.push_back({DType::I8, static_cast<std::int32_t>(c.mla.kv_lora), 1});
            out.geometry.planes.push_back({DType::FP16, static_cast<std::int32_t>(c.mla.kv_lora / 64), 1});
        } else {
            out.geometry.planes.push_back({DType::BF16, static_cast<std::int32_t>(c.mla.kv_lora), 1});
        }
        out.geometry.planes.push_back(
            {DType::BF16, static_cast<std::int32_t>(c.indexer.head_dim / c.indexer.pool), 1});
    }
    out.planes_per_layer = out.kv_layers == 0 ? 0U : static_cast<std::uint32_t>(out.geometry.planes.size()) / out.kv_layers;
    return out;
}

PlaneRange kv_layer_planes(const KvPageGeometry& geometry, std::uint32_t kv_layer) {
    if (kv_layer >= geometry.kv_layers) { throw std::out_of_range("Glm5Next KV layer out of range"); }
    const std::size_t begin = static_cast<std::size_t>(kv_layer) * geometry.planes_per_layer;
    return PlaneRange{begin, begin + geometry.planes_per_layer};
}

} // namespace infernix::models::glm5_next::prefix
