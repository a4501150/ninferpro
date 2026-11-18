#pragma once

// The fast NVFP4 prompt kernel: the MXFP8 tiled kernel (mxfp8_tiled_mma.cuh) with NVFP4 keys. QK
// runs on block-scaled FP4 Tensor Cores straight from the stored K codes with a two-term NVFP4 Q,
// each 64-key V tile is decoded once per CTA into an FP16 arena, PV accumulates per tile on FP16
// Tensor Cores (or, with pv8, block-scaled E4M3), and the split partials merge with the inverse
// rotation. envelope.fast_prompt_pv8 selects the E4M3 PV form, and the key splits keep their FP32
// partials within envelope.prompt_split_workspace_bytes.

#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

#include <cstdint>

namespace infernix::ops::detail {

// Whether an NVFP4 prompt-route launch runs the fast kernel: over 768 visible keys. Forced at
// every visible count the sweep measured, picking the fast kernel above 768 keys is within a
// fraction of a percent of the per-shape best in both geometries (the sweep, and the cost of the
// older 2048 threshold it replaced, are in RESEARCH_NOTES.md, "NVFP4 fast-prompt threshold
// re-tuned to 768 visible keys").
inline bool nvfp4_fast_prompt_applies(std::uint32_t max_visible_keys) {
    return max_visible_keys > 768;
}

void nvfp4_kv_fast_prompt_attention(const CausalAttentionOperands&, Nvfp4KvReadView,
                                    CausalAttentionExecutionEnvelope, WorkspaceArena&,
                                    DeviceExecutionView);

} // namespace infernix::ops::detail
