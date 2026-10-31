#pragma once

// The Vision tower's parameter types, shared by Qwen3.5's and Qwen4Exp's execution: the tower's
// mathematics is one implementation, so both families prepare their weights in these types
// (qwen_common/execution/vision_tower.h).

#include "core/tensor.h"
#include "infernix/ops/weight_input.h"

#include <vector>

namespace infernix::models::qwen_common::execution {

using LinearParameters = ops::SingleProjectionWeight;

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

} // namespace infernix::models::qwen_common::execution
