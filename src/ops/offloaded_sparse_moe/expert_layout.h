#pragma once

// The expert record layout and per-expert scalars shared by offloaded_sparse_moe's CPU and GPU
// routes (docs/maintainer/qwen3_8-flash-next-design.md §6.3).
//
// The record geometry is a parameter, not a constant: a `Geometry` names the hidden and
// intermediate sizes a record was built at (and its experts' SwiGLU limit), and every route that
// moves or computes records takes one. `kGeometries` registers the instances the engine serves;
// a caller outside them is refused. The first entry is the record every other bank format (W4A4,
// block FP8, ggml) is built at, and the layout constants below stay bound to it, so those banks
// and their routes keep their compiled code.

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_MOE_HD __host__ __device__
#else
#    define INFERNIX_MOE_HD
#endif

namespace infernix::ops::offloaded_moe {

inline constexpr std::size_t kUnitBytes =
    144; // one record unit: 16 weights x 16 rows, then their scales

// One served record geometry: hidden and intermediate sizes (whole 16-element blocks), and the
// SwiGLU limit its experts clamp gate and up with (0: no clamp). Derived record sizes are
// constexpr members, so host code sizes buffers from a Geometry and device code instantiates on
// a `Nvfp4Geometry<H, I>`.
struct Geometry {
    int hidden;
    int intermediate;
    float swiglu_limit; // 0: none

    INFERNIX_MOE_HD constexpr int gate_up_row_groups() const {
        return 2 * intermediate / 16;
    } // rows interleaved gate/up

    INFERNIX_MOE_HD constexpr int gate_up_blocks() const { return hidden / 16; }

    INFERNIX_MOE_HD constexpr int down_row_groups() const { return hidden / 16; }

    INFERNIX_MOE_HD constexpr int down_blocks() const { return intermediate / 16; }

    INFERNIX_MOE_HD constexpr int h_blocks() const {
        return intermediate / 16;
    } // units of 16 intermediates

    INFERNIX_MOE_HD constexpr std::size_t gate_up_bytes() const {
        return static_cast<std::size_t>(gate_up_row_groups()) * gate_up_blocks() * kUnitBytes;
    }

    INFERNIX_MOE_HD constexpr std::size_t down_bytes() const {
        return static_cast<std::size_t>(down_row_groups()) * down_blocks() * kUnitBytes;
    }

    INFERNIX_MOE_HD constexpr std::size_t record_bytes() const {
        return gate_up_bytes() + down_bytes();
    }
};

// The registered geometries. The first is the one every other bank format and the layout
// constants below are built at; the second is the served geometry with 16-bit activations and a
// SwiGLU limit of 10.
inline constexpr Geometry kGeometries[] = {
    {2560, 640, 0.0F},
    {4096, 2048, 10.0F},
};

// The geometry the layout constants describe (kGeometries[0]).
inline constexpr Geometry kDefaultGeometry = kGeometries[0];

// The record layout at geometry <H, I> (each matrix as row groups of 144-byte units: 16 weights
// x 16 rows, the unit's last 16 bytes the rows' E4M3 scales). A16 routes instantiate over it.
template <int H, int I>
struct Nvfp4Geometry {
    static constexpr int kHidden          = H;
    static constexpr int kIntermediate    = I;
    static constexpr int kGateUpRowGroups = 2 * I / 16; // rows interleaved gate/up
    static constexpr int kGateUpBlocks    = H / 16;
    static constexpr int kDownRowGroups   = H / 16;
    static constexpr int kDownBlocks      = I / 16;
    static constexpr int kHBlocks         = I / 16; // units of 16 intermediates
    static constexpr std::size_t kGateUpBytes =
        static_cast<std::size_t>(kGateUpRowGroups) * kGateUpBlocks * kUnitBytes;
    static constexpr std::size_t kDownBytes =
        static_cast<std::size_t>(kDownRowGroups) * kDownBlocks * kUnitBytes;
    static constexpr std::size_t kRecordBytes = kGateUpBytes + kDownBytes;
};

static_assert(Nvfp4Geometry<2560, 640>::kRecordBytes == 675 * 4096);
static_assert(Nvfp4Geometry<4096, 2048>::kRecordBytes == 9437184ULL + 4718592ULL);

// Record layout `nvfp4_expert_rg16_v1` at the default geometry (design §6.3). The W4A4,
// block-FP8 and ggml banks and their routes are built at this geometry only.
inline constexpr int kHidden       = kDefaultGeometry.hidden;
inline constexpr int kIntermediate = kDefaultGeometry.intermediate;
inline constexpr int kGateUpRowGroups =
    kDefaultGeometry.gate_up_row_groups(); // 80, rows interleaved gate/up
inline constexpr int kGateUpBlocks  = kDefaultGeometry.gate_up_blocks();  // 160
inline constexpr int kDownRowGroups = kDefaultGeometry.down_row_groups(); // 160
inline constexpr int kDownBlocks    = kDefaultGeometry.down_blocks();     // 40
inline constexpr int kHBlocks       = kDefaultGeometry.h_blocks(); // 40 units of 16 intermediates
inline constexpr std::size_t kGateUpBytes = kDefaultGeometry.gate_up_bytes(); // 1,843,200
inline constexpr std::size_t kDownBytes   = kDefaultGeometry.down_bytes();    // 921,600
inline constexpr std::size_t kRecordBytes = kDefaultGeometry.record_bytes();  // 2,764,800
inline constexpr int kMaxColumns          = 8; // narrow route: n <= 8

static_assert(kRecordBytes == 675 * 4096);

// Per-expert scalars from the checkpoint (design §6.3). alpha = fl32(weight_scale_2 * input_scale)
// is derived once by the loader, so every route reads the same words.
struct ExpertScales {
    float input_gate;
    float input_up;
    float input_down;
    float alpha_gate;
    float alpha_up;
    float alpha_down;
};

// The activation arithmetic of a layer's experts, fixed by the artifact: W4A4 with each matrix's
// stored input scale (§17.2), or W4A16 for experts stored without activation scales (§17.3,
// w4a16_expert.h), whose ExpertScales hold zero input scales and the multipliers in alpha_*.
enum class ExpertActivation : std::uint8_t { kA4, kA16 };

// The wide route of 16-bit-activation experts (design §13.4, §13.5): kExact, the exact limb
// arithmetic of §17.3; kA4Residual (NVFP4 W4A16 banks), a fast path that rounds each x and h row
// to NVFP4 terms (the row's NVFP4 and the NVFP4 of its remainder, one FP32 row scale) and runs
// them on the FP4 Tensor Cores, so its outputs depend on that rounding and on the tensor core's
// FP32 order, not only on the weights; kGgmlBf16 (GGUF banks, the ggml-bf16-prefill fast path),
// BF16 tiles over each expert's decoded planes (the default geometry) or its rows decoded in the K
// loop (the others), x and h as stored and FP32 sums (ops/ggml/ggml_bf16_tile.cuh). Every other
// bank ignores the value it does not name.
enum class WideA16 : std::uint8_t { kExact, kA4Residual, kGgmlBf16 };

// The SwiGLU limit registered for the record geometry <H, I> (0 when none is).
template <int H, int I>
inline INFERNIX_MOE_HD constexpr float geometry_swiglu_limit() {
    for (const Geometry& g : kGeometries) {
        if (g.hidden == H && g.intermediate == I) { return g.swiglu_limit; }
    }
    return 0.0F;
}

// The served geometry of a source or service, validated against kGeometries: nullptr when
// (hidden, intermediate, swiglu_limit) matches no registered instance.
inline constexpr const Geometry* registered_geometry(int hidden, int intermediate,
                                                     float swiglu_limit) {
    for (const Geometry& g : kGeometries) {
        if (g.hidden == hidden && g.intermediate == intermediate &&
            g.swiglu_limit == swiglu_limit) {
            return &g;
        }
    }
    return nullptr;
}

} // namespace infernix::ops::offloaded_moe
