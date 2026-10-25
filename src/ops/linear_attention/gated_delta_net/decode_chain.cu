// gated_delta_net_decode_chain (include/infernix/ops/gated_delta_net.h): a GDN layer's decode middle
// in one kernel - the causal convolution of the projection's q/k/v channels, the gated recurrent
// update and the gated RMSNorm of each head's output - with the bits of
// causal_conv1d_silu_snapshot_split, gated_delta_net_batch_update_gating and gated_rmsnorm.
//
// Building blocks composed inside one cluster: two CTAs per (value head, sequence), each owning half
// the head's state rows, run the head's three steps through shared memory - the convolution step
// writes q, k and the half's v, run_recurrent_sequence (the batch update's own loop) reads them and
// writes the half's output into the first half's shared memory (across the cluster), and one warp
// of the first half normalizes the whole row with rmsnorm_warp_row_bf16x2. The value heads that
// share a key head form one cluster: each CTA reads that key head's convolution state, and the
// cluster's first CTA updates it only after the cluster barrier, so the in-place update races no reader.

#include "infernix/ops/gated_delta_net.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/kernel/gdn_gate_values.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear_attention/gated_delta_net/decode_chain.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace infernix::ops {
namespace detail::gated_delta_net {
namespace {

template <RmsEpilogue Epilogue>
__global__ void __launch_bounds__(kChainThreads, 2) gdn_decode_chain_kernel(const ChainArgs args) {
    __shared__ ChainShared sm;
    const ChainPlace at = chain_place(args, static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y));
    ChainPreload p;
    chain_preload(args, at, p);
    pdl::enter_streaming();
    chain_finish<Epilogue>(args, at, p, sm);
}

} // namespace
} // namespace detail::gated_delta_net

bool gated_delta_net_decode_chain_supports(std::int32_t qk_heads, std::int32_t value_heads, std::int32_t conv_taps,
                                           std::int32_t head_dim) {
    return head_dim == detail::gated_delta_net::kStateDim && conv_taps == detail::gated_delta_net::kConvTaps &&
           qk_heads > 0 && value_heads % qk_heads == 0 && value_heads / qk_heads * detail::gated_delta_net::kHalves <= 8;
}

void gated_delta_net_decode_chain(const Tensor& qkv, const Tensor& conv_weight, Tensor& conv_states, const Tensor& a,
                                  const Tensor& b, const Tensor& A_log, const Tensor& dt_bias, float scale,
                                  Tensor& ssm_states, const Tensor& slots, const Tensor& norm_weight, const Tensor& z,
                                  RmsGate gate, float eps, std::int32_t qk_heads, Tensor& out, cudaStream_t stream) {
    namespace gdn = detail::gated_delta_net;
    const std::int32_t C = qkv.ne[0], B = qkv.ne[1], value_heads = A_log.ne[0];
    if (!gated_delta_net_decode_chain_supports(qk_heads, value_heads, conv_weight.ne[1], gdn::kStateDim) ||
        qkv.dtype != DType::BF16 || conv_weight.dtype != DType::BF16 || conv_states.dtype != DType::BF16 ||
        a.dtype != DType::BF16 || b.dtype != DType::BF16 || A_log.dtype != DType::FP32 ||
        dt_bias.dtype != DType::FP32 || ssm_states.dtype != DType::FP32 || slots.dtype != DType::I32 ||
        norm_weight.dtype != DType::BF16 || z.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        C != 2 * qk_heads * gdn::kStateDim + value_heads * gdn::kStateDim || conv_weight.ne[0] != C ||
        slots.numel() != B || norm_weight.numel() != gdn::kStateDim ||
        z.numel() != static_cast<std::int64_t>(value_heads) * gdn::kStateDim * B ||
        out.numel() != static_cast<std::int64_t>(value_heads) * gdn::kStateDim * B ||
        a.ne[0] != value_heads || b.ne[0] != value_heads || a.ne[1] != B || b.ne[1] != B) {
        throw std::invalid_argument("gated_delta_net_decode_chain: unsupported or disagreeing operands");
    }
    const auto column_stride = [&](const Tensor& t) {
        return t.ne[1] == 1 ? std::int64_t{value_heads} : t.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    };
    const gdn::ChainArgs args{static_cast<const __nv_bfloat16*>(qkv.data),
                              static_cast<const __nv_bfloat16*>(conv_weight.data),
                              static_cast<__nv_bfloat16*>(conv_states.data),
                              static_cast<std::int64_t>(conv_states.ne[0]) * conv_states.ne[1],
                              static_cast<const __nv_bfloat16*>(a.data),
                              static_cast<const __nv_bfloat16*>(b.data),
                              column_stride(a),
                              column_stride(b),
                              static_cast<const float*>(A_log.data),
                              static_cast<const float*>(dt_bias.data),
                              static_cast<float*>(ssm_states.data),
                              static_cast<std::int64_t>(gdn::kStateDim) * gdn::kStateDim * ssm_states.ne[2],
                              static_cast<const std::int32_t*>(slots.data),
                              static_cast<const __nv_bfloat16*>(norm_weight.data),
                              static_cast<const __nv_bfloat16*>(z.data),
                              static_cast<__nv_bfloat16*>(out.data),
                              gdn::head_map::of(qk_heads, value_heads),
                              C,
                              scale,
                              eps};
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = static_cast<unsigned>(value_heads / qk_heads * gdn::kHalves);
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    unsigned count                 = 1;
    if (pdl::graph_dependent(stream)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        count                                                    = 2;
    }
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(static_cast<unsigned>(value_heads * gdn::kHalves), static_cast<unsigned>(B));
    config.blockDim = dim3(gdn::kChainThreads);
    config.stream   = stream;
    config.attrs    = attributes;
    config.numAttrs = count;
    if (gate == RmsGate::Sigmoid) {
        CUDA_CHECK(cudaLaunchKernelEx(&config, gdn::gdn_decode_chain_kernel<RmsEpilogue::SigmoidGated>, args));
    } else {
        CUDA_CHECK(cudaLaunchKernelEx(&config, gdn::gdn_decode_chain_kernel<RmsEpilogue::Gated>, args));
    }
}

} // namespace infernix::ops
