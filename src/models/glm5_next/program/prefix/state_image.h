#pragma once

// The packed state image of one Glm5Next lane at a frontier (docs/maintainer/glm5_3-flash-design.md
// §18): every byte of the lane's state that its KV pages do not hold — the KDA recurrence and
// convolution window of every KDA block, the DSA indexer's raw key and gate ring of every MLA
// block (the incomplete pool's inputs), and the DFlash2 drafter's context ring. Parts follow the
// forward pass, so a restore lands each decoder layer's state as one group the forward pass can
// wait for on its own:
//
//   header | layer 0 .. L-1: KDA (recurrent, conv) or MLA (indexer k ring, z ring) | drafter
//   group: the DFlash2 ring (K and V of each drafter layer), then the target head's saved
//   residual (the zero-token resume)
//
// Every part starts on a 256-byte boundary. All copies are byte copies; nothing is requantized. On
// the Host the packed byte o lives at segments[o / segment_bytes] + o % segment_bytes (the slabs of
// the prefix cache's pinned pool, not contiguous); a packed Device slot holds the same layout.

#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "models/glm5_next/config.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace infernix::models::glm5_next::prefix {

inline constexpr std::size_t kStateImageHeaderBytes = 256;
inline constexpr std::size_t kStateImageAlignment   = 256;

// A DFlash2 drafter's context ring (core/cyclic_kv_cache.h): per drafter layer one lane's K bytes
// and V bytes (each contiguous: the lane is the ring planes' outermost axis).
struct DraftRingGeometry {
    std::uint32_t layers = 0;
    std::size_t k_bytes  = 0;
    std::size_t v_bytes  = 0;
};

// Byte sizes of one lane's state, per layer where it is per layer.
struct StateImageSpec {
    std::vector<MixerKind> layer_types; // decoder layers in order
    std::size_t kda_recurrent_bytes = 0;
    std::size_t kda_conv_bytes      = 0;
    std::size_t index_bytes         = 0; // one lane's keys or gates of an MLA block's ring
    std::size_t head_saved_bytes    = 0; // the target head's saved residual (the zero-token resume)
    DraftRingGeometry draft_ring;        // DFlash2 (layers 0: none)
};

[[nodiscard]] StateImageSpec state_image_spec(const TextConfig& config, DraftRingGeometry draft_ring,
                                              std::size_t head_saved_bytes);

struct StateImagePart {
    enum class Kind : std::uint8_t {
        Header, KdaRecurrent, KdaConv, IndexKeys, IndexGates, DraftRingK, DraftRingV, HeadSaved
    };

    Kind kind           = Kind::Header;
    // The layer's index among KDA or MLA layers; for a ring plane, the drafter layer; 0 otherwise.
    std::uint32_t index = 0;
    std::size_t offset  = 0; // in the packed image
    std::size_t bytes   = 0;
};

// Group g < layers is decoder layer g; group `layers` is the drafter's group (empty without a
// drafter). The header (part 0) belongs to no group.
struct StateImageLayout {
    StateImageSpec spec;
    std::vector<StateImagePart> parts;
    std::vector<std::uint32_t> group_begin; // group g's parts are [group_begin[g], group_begin[g + 1])
    std::size_t image_bytes  = 0;
    std::uint64_t fingerprint = 0; // of every part's kind, index, offset and size

    [[nodiscard]] std::uint32_t groups() const noexcept {
        return static_cast<std::uint32_t>(group_begin.size()) - 1U;
    }
    [[nodiscard]] std::uint32_t drafter_group() const noexcept { return groups() - 1U; }
};

[[nodiscard]] StateImageLayout plan_state_image(const StateImageSpec& spec);

// Snapshot meta kept in the header part (written and read on the Host).
struct StateImageHeader {
    std::uint32_t frontier = 0;
    // The snapshot is the generation-opener tap (an exact split before the assistant turn).
    bool opener = false;
    // The HeadSaved part holds the target head's residual of this frontier's last token — valid
    // only on a prompt-end capture; taps carry a stale column and leave this clear.
    bool head_residual = false;
    // DFlash2: the ring holds the context of every position below this one (the frontier).
    std::uint32_t draft_frontier = 0;
};

void write_state_image_header(const StateImageLayout& layout, const StateImageHeader& header,
                              std::byte* image_begin);
// Absent when the bytes are not a header of this layout.
[[nodiscard]] std::optional<StateImageHeader> read_state_image_header(const StateImageLayout& layout,
                                                                      const std::byte* image_begin);

// Where the Program keeps every lane's state (program_impl.h).
struct LaneStateBuffers {
    const LinearAttentionStatePool* kda = nullptr;
    std::uint32_t lanes                 = 0;
    // The indexer's ring, per MLA block two planes (keys, then gates) of BF16
    // [head_dim, 64, lanes]: block k's keys of lane l at index_ring + (2k) * index_plane_bytes +
    // l * spec.index_bytes, its gates at index_ring + (2k + 1) * index_plane_bytes + l *
    // spec.index_bytes.
    std::byte* index_ring        = nullptr;
    std::size_t index_plane_bytes = 0;
    std::byte* head_saved         = nullptr; // lane l at head_saved + l * spec.head_saved_bytes
    // DFlash2's ring, per drafter layer: lane l's K at ring_k[layer] + l * k_bytes (V likewise).
    std::vector<std::byte*> ring_k, ring_v;
};

// Host placement of one packed image. The segments may also be device memory (a staging slot):
// the copies take either.
struct HostImage {
    std::span<std::byte* const> segments;
    std::size_t segment_bytes = 0;
};

struct HostImageConst {
    std::span<const std::byte* const> segments;
    std::size_t segment_bytes = 0;
};

// Copies between a lane's state, packed Device slots and Host images. Every copy is enqueued on the
// caller's stream; the caller orders it against the compute stream and records any events.
class LaneStateImage {
public:
    // Checks the buffers against the layout's sizes.
    LaneStateImage(const StateImageLayout& layout, const LaneStateBuffers& buffers);

    [[nodiscard]] const StateImageLayout& layout() const noexcept { return *layout_; }
    // The lane's Device bytes of a part (none for the header).
    [[nodiscard]] std::byte* device_part(const StateImagePart& part, std::uint32_t lane) const;

    // Capture: every part but the header, or one group's parts.
    void copy_lane_to_host(std::uint32_t lane, const HostImage& image, cudaStream_t stream) const;
    void copy_group_to_host(std::uint32_t group, std::uint32_t lane, const HostImage& image, cudaStream_t stream) const;
    void copy_lane_to_packed(std::uint32_t lane, std::byte* packed, cudaStream_t stream) const;
    // Write-through of a packed Device slot (header included, as the slot holds it).
    void copy_packed_to_host(const std::byte* packed, const HostImage& image, cudaStream_t stream) const;

    // Restore of one group's parts.
    void copy_group_from_host(std::uint32_t group, const HostImageConst& image, std::uint32_t lane,
                              cudaStream_t stream) const;
    void copy_group_from_packed(std::uint32_t group, const std::byte* packed, std::uint32_t lane,
                                cudaStream_t stream) const;

private:
    void check_lane(std::uint32_t lane) const;
    void check_image(std::size_t segments, std::size_t segment_bytes) const;

    const StateImageLayout* layout_ = nullptr;
    LaneStateBuffers buffers_;
};

// The Program's KV page group: for every MLA layer, the latent (BF16, or INT8 codes and their FP16
// group scales), then the indexer's pooled keys. A cached block's page records move per MLA layer,
// so a restore lands the planes each MLA layer reads in forward order.
struct KvPageGeometry {
    KVPageGeometry geometry;
    std::uint32_t kv_layers        = 0;
    std::uint32_t planes_per_layer = 0;
};

[[nodiscard]] KvPageGeometry kv_page_geometry(const TextConfig& config, KvCacheStorage storage);

struct PlaneRange {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// The planes of MLA layer `kv_layer`.
[[nodiscard]] PlaneRange kv_layer_planes(const KvPageGeometry& geometry, std::uint32_t kv_layer);

} // namespace infernix::models::glm5_next::prefix
