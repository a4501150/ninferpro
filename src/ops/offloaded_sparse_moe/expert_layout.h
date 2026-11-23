#pragma once

// The expert record layout and per-expert scalars shared by offloaded_sparse_moe's CPU and GPU
// routes (docs/maintainer/qwen3_8-flash-next-design.md §6.3).

#include <cstddef>
#include <cstdint>

namespace infernix::ops::offloaded_moe {

// Record layout `nvfp4_expert_rg16_v1` (design §6.3).
inline constexpr int kHidden            = 2560;
inline constexpr int kIntermediate      = 640;
inline constexpr std::size_t kUnitBytes = 144;
inline constexpr int kGateUpRowGroups   = 2 * kIntermediate / 16; // 80, rows interleaved gate/up
inline constexpr int kGateUpBlocks      = kHidden / 16;           // 160
inline constexpr int kDownRowGroups     = kHidden / 16;           // 160
inline constexpr int kDownBlocks        = kIntermediate / 16;     // 40
inline constexpr int kHBlocks           = kIntermediate / 16;     // 40 units of 16 intermediates
inline constexpr std::size_t kGateUpBytes =
    static_cast<std::size_t>(kGateUpRowGroups) * kGateUpBlocks * kUnitBytes; // 1,843,200
inline constexpr std::size_t kDownBytes =
    static_cast<std::size_t>(kDownRowGroups) * kDownBlocks * kUnitBytes; // 921,600
inline constexpr std::size_t kRecordBytes = kGateUpBytes + kDownBytes;   // 2,764,800
inline constexpr int kMaxColumns          = 8;                           // narrow route: n <= 8

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
// arithmetic of §17.3; kA4Residual (NVFP4 W4A16 banks), a fast path that rounds each x and h
// row to two NVFP4 terms (the row's NVFP4 and the NVFP4 of its remainder, one FP32 row scale) and
// runs them on the FP4 Tensor Cores, so its outputs depend on that rounding and on the tensor
// core's FP32 order, not only on the weights; kGgmlBf16 (GGUF banks, the ggml-bf16-prefill fast
// path), BF16 tiles over each expert's decoded planes with x and h as stored and FP32 sums
// (ops/ggml/ggml_bf16_tile.cuh). Every other bank ignores the value it does not name.
enum class WideA16 : std::uint8_t { kExact, kA4Residual, kGgmlBf16 };

} // namespace infernix::ops::offloaded_moe
