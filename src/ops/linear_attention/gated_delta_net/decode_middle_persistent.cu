// Experiment (exp/gdn-chain): a GDN layer's decode input projection and its middle chain in one
// persistent launch, composed from the existing blocks with a grid barrier between the steps:
//   1. every CTA runs virtual CTAs of the q/k/v/z FP8 GEMV and the a/b BF16 GEMV (their blocks);
//   2. grid barrier;
//   3. the chain CTAs (two per value head, clusters of the heads sharing a key head) finish the
//      convolution, recurrence and gated norm - with the state tiles they loaded at kernel start,
//      while step 1 streamed.
// Same bits as the composed projection followed by gated_delta_net_decode_chain.

#include "infernix/ops/gated_delta_net.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_a16_gemv.cuh"
#include "ops/linear/bf16/bf16_decode_routes.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/common/row_parts.h"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_decode_routes.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear_attention/gated_delta_net/decode_chain.cuh"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace infernix::ops {
namespace {
int mega_clusters() {
    static const int n = [] { const char* e = std::getenv("INFERNIX_MEGA_CLUSTERS"); return e ? std::atoi(e) : 1 << 20; }();
    return n;
}
} // namespace
namespace detail::gated_delta_net {
namespace {

__device__ unsigned g_grid_barrier[2]; // arrivals, generation

__device__ __forceinline__ void grid_barrier() {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* generation = &g_grid_barrier[1];
        const unsigned seen           = *generation;
        __threadfence();
        if (atomicAdd(&g_grid_barrier[0], 1U) == gridDim.x - 1) {
            g_grid_barrier[0] = 0;
            __threadfence();
            atomicAdd(&g_grid_barrier[1], 1U);
        } else {
            while (*generation == seen) { __nanosleep(20); }
        }
        __threadfence();
    }
    __syncthreads();
}

template <RmsEpilogue Epilogue, class Fp8Block, class Bf16Block>
__global__ void __launch_bounds__(kChainThreads, 2)
    gdn_middle_persistent_kernel(const ChainArgs args, const Fp8Block fp8, const Bf16Block bf16, int fp8_ctas,
                                 int total, int chain_ctas) {
    static_assert(Fp8Block::kThreads <= kChainThreads && Bf16Block::kThreads <= kChainThreads);
    __shared__ ChainShared sm;
    const bool chains = static_cast<int>(blockIdx.x) < chain_ctas;
    ChainPlace at{};
    ChainPreload p{};
    if (chains) {
        at = chain_place(args, static_cast<int>(blockIdx.x), 0);
        chain_preload(args, at, p);
    }
    pdl::wait_for_dependencies();
    for (int v = static_cast<int>(blockIdx.x); v < total; v += static_cast<int>(gridDim.x)) {
        if (v < fp8_ctas) {
            if (static_cast<int>(threadIdx.x) < Fp8Block::kThreads) { fp8.run(v); }
        } else if (static_cast<int>(threadIdx.x) < Bf16Block::kThreads) {
            bf16.run(v - fp8_ctas);
        }
    }
    grid_barrier();
    if (chains) { chain_finish<Epilogue>(args, at, p, sm); }
}

} // namespace
} // namespace detail::gated_delta_net

bool gated_delta_net_decode_middle_persistent(const Tensor& x, const Weight& qkvz, Tensor& qkv, Tensor& z,
                                              const Weight& ab_weight, Tensor& ab, const Tensor& conv_weight,
                                              Tensor& conv_states, const Tensor& A_log, const Tensor& dt_bias,
                                              float scale, Tensor& ssm_states, const Tensor& slots,
                                              const Tensor& norm_weight, RmsGate gate, float eps,
                                              std::int32_t qk_heads, Tensor& out, cudaStream_t stream) {
    namespace gdn = detail::gated_delta_net;
    using namespace detail;
    const std::int32_t value_heads = A_log.ne[0];
    if (x.ne[1] != 1 || qkvz.qtype != QType::FP8_E4M3FN_ROW_BF16 || qkvz.n != 16384 || qkvz.k != 2560 ||
        ab_weight.qtype != QType::BF16 || ab_weight.n != 96 || ab_weight.k != 2560 || value_heads != 48 ||
        qk_heads != 16) {
        return false;
    }
    LinearBf16RowParts qkvz_parts{}, ab_parts{};
    qkvz_parts.data[0] = static_cast<__nv_bfloat16*>(qkv.data);
    qkvz_parts.end[0]  = qkv.ne[0];
    for (int i = 1; i < LinearBf16RowParts::kMaxParts; ++i) {
        qkvz_parts.data[i] = static_cast<__nv_bfloat16*>(z.data);
        qkvz_parts.end[i]  = qkv.ne[0] + z.ne[0];
    }
    for (int i = 0; i < LinearBf16RowParts::kMaxParts; ++i) {
        ab_parts.data[i] = static_cast<__nv_bfloat16*>(ab.data);
        ab_parts.end[i]  = ab.ne[0];
    }
    using Fp8Instance        = Fp8ScheduleInstance<Fp8N16384K2560Gemv, 2560>;
    const Fp8A16Operands f   = fp8_a16_operands(x, qkvz);
    validate_fp8_operands<Fp8Instance>(f);
    const Fp8A16GemvBlock<Fp8Instance, LinearBf16RowParts, LinearIdentityEpilogue, Fp8IdentityRows> fp8{
        f.x, f.codes, f.scales, qkvz_parts, {}, {}, f.rows, f.k};
    const Bf16A16Operands q = bf16_a16_operands(x, ab_weight);
    validate_bf16_operands<Bf16N96K2560Gemv>(q);
    const Bf16A16GemvBlock<Bf16N96K2560Gemv, LinearBf16RowParts, LinearIdentityEpilogue> bf16{
        q.x, q.weight, ab_parts, {}, q.rows, q.k};

    const Tensor a = ab.slice(0, 0, value_heads), b = ab.slice(0, value_heads, value_heads);
    const std::int64_t gate_ld = value_heads * 2;
    const gdn::ChainArgs args{static_cast<const __nv_bfloat16*>(qkv.data),
                              static_cast<const __nv_bfloat16*>(conv_weight.data),
                              static_cast<__nv_bfloat16*>(conv_states.data),
                              static_cast<std::int64_t>(conv_states.ne[0]) * conv_states.ne[1],
                              static_cast<const __nv_bfloat16*>(a.data),
                              static_cast<const __nv_bfloat16*>(b.data),
                              gate_ld,
                              gate_ld,
                              static_cast<const float*>(A_log.data),
                              static_cast<const float*>(dt_bias.data),
                              static_cast<float*>(ssm_states.data),
                              static_cast<std::int64_t>(gdn::kStateDim) * gdn::kStateDim * ssm_states.ne[2],
                              static_cast<const std::int32_t*>(slots.data),
                              static_cast<const __nv_bfloat16*>(norm_weight.data),
                              static_cast<const __nv_bfloat16*>(z.data),
                              static_cast<__nv_bfloat16*>(out.data),
                              gdn::head_map::of(qk_heads, value_heads),
                              qkv.ne[0],
                              scale,
                              eps};
    const int cluster    = value_heads / qk_heads * gdn::kHalves;
    const int chain_ctas = value_heads * gdn::kHalves;
    const int fp8_ctas   = fp8.ctas();
    const int total      = fp8_ctas + bf16.ctas();
    auto kernel = gate == RmsGate::Sigmoid
                      ? gdn::gdn_middle_persistent_kernel<RmsEpilogue::SigmoidGated, decltype(fp8), decltype(bf16)>
                      : gdn::gdn_middle_persistent_kernel<RmsEpilogue::Gated, decltype(fp8), decltype(bf16)>;
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = static_cast<unsigned>(cluster);
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.blockDim = dim3(gdn::kChainThreads);
    config.stream   = stream;
    config.attrs    = attributes;
    config.numAttrs = 1;
    static int clusters = [&] {
        int n = 0;
        config.gridDim = dim3(static_cast<unsigned>(cluster * 64));
        CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, kernel, &config));
        return n;
    }();
    if (clusters * cluster < chain_ctas) { return false; }
    config.gridDim = dim3(static_cast<unsigned>(std::min(clusters, mega_clusters()) * cluster));
    if (pdl::graph_dependent(stream)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        config.numAttrs                                          = 2;
    }
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args, fp8, bf16, fp8_ctas, total, chain_ctas));
    return true;
}

} // namespace infernix::ops
