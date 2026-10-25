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
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace infernix::ops {
namespace detail::gated_delta_net {
namespace {

namespace cg = cooperative_groups;

constexpr int kHalves       = 2;
constexpr int kHalfRows     = kStateDim / kHalves;
constexpr int kChainWarps   = kHalfRows / kDvPerWarp; // half the head's rows: 16 warps of 4 rows
constexpr int kChainThreads = kChainWarps * kWarpSize;
constexpr int kConvTaps     = 4;

struct ChainArgs {
    const __nv_bfloat16* qkv; // [C, B]: q | k | v channels of each sequence
    const __nv_bfloat16* conv_weight; // [taps][C]
    __nv_bfloat16* conv_states;       // slots of [taps - 1][C]
    std::int64_t conv_slot_stride;
    const __nv_bfloat16* gate_a; // [Hv] per sequence at gate_ld
    const __nv_bfloat16* gate_b;
    std::int64_t gate_a_ld, gate_b_ld;
    const float* a_log;
    const float* dt_bias;
    float* ssm_states; // slots of [Hv][128][128]
    std::int64_t state_slot_stride;
    const std::int32_t* slots; // each sequence's slot, read and written in place (both pools)
    const __nv_bfloat16* norm_weight; // [128]
    const __nv_bfloat16* z;           // [128] per (sequence, value head)
    __nv_bfloat16* out;               // [128] per (sequence, value head)
    head_map heads;
    int channels;
    float scale;
    float eps;
};

// run_recurrent_sequence's access to one head's q, k, v and output in shared memory.
struct ChainAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    __nv_bfloat16* o;
    __nv_bfloat16 a, b;
    float a_log, dt_bias;
    float scale;

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates&, std::int32_t) const { return k; }
    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates&, std::int32_t) const { return v; }
    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates&, std::int32_t) const { return q; }
    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates&, std::int32_t) const { return o; }
    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates&, std::int32_t) const {
        float g_value = 0.0f, beta_value = 0.0f;
        gdn_gate_values(a, b, a_log, dt_bias, g_value, beta_value);
        return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
    }
};

template <RmsEpilogue Epilogue>
__global__ void __launch_bounds__(kChainThreads, 2) gdn_decode_chain_kernel(const ChainArgs args) {
    __shared__ __align__(16) __nv_bfloat16 q_s[kStateDim];
    __shared__ __align__(16) __nv_bfloat16 k_s[kStateDim];
    __shared__ __align__(16) __nv_bfloat16 v_s[kStateDim];
    __shared__ __align__(16) __nv_bfloat16 o_s[kStateDim];
    __shared__ __align__(16) __nv_bfloat16 w_s[kStateDim]; // the norm's weight and the head's z row
    __shared__ __align__(16) __nv_bfloat16 z_s[kStateDim];
    const int t = static_cast<int>(threadIdx.x), lane = t % kWarpSize, warp = t / kWarpSize;
    const int head = static_cast<int>(blockIdx.x) / kHalves, half = static_cast<int>(blockIdx.x) % kHalves;
    const int batch = static_cast<int>(blockIdx.y);
    const int qk_head = args.heads.qk_head(head);
    const RecurrentCoordinates coord{lane,
                                     warp,
                                     batch,
                                     0,
                                     warp / kNumWarps,
                                     static_cast<std::uint32_t>(head),
                                     static_cast<std::uint32_t>(qk_head),
                                     static_cast<std::uint32_t>(half * kHalfRows + warp * kDvPerWarp),
                                     static_cast<std::uint32_t>(lane * kQkPerLane)};
    const std::int64_t slot = args.slots[batch];

    // Before the wait (both states were last written by an earlier round; the slot comes from the
    // round's ingress): the state tile, and for threads 0..319 one channel's taps and state - q and
    // k of the key head, then v of this half.
    float* state_base = args.ssm_states + slot * args.state_slot_stride +
                        static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, state_base, coord);
    const int part = t / kStateDim, i = part < 2 ? t % kStateDim : half * kHalfRows + t % kStateDim;
    const int key_width = args.heads.H_qk * kStateDim;
    const int c = part == 0 ? qk_head * kStateDim + i
                : part == 1 ? key_width + qk_head * kStateDim + i
                            : 2 * key_width + head * kStateDim + i;
    const bool convolves = t < 2 * kStateDim + kHalfRows;
    const std::int64_t C = args.channels;
    __nv_bfloat16* conv = args.conv_states + slot * args.conv_slot_stride;
    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
    __nv_bfloat16 s0{}, s1{}, s2{};
    if (convolves) {
        w0 = __bfloat162float(args.conv_weight[c]);
        w1 = __bfloat162float(args.conv_weight[C + c]);
        w2 = __bfloat162float(args.conv_weight[2 * C + c]);
        w3 = __bfloat162float(args.conv_weight[3 * C + c]);
        s0 = conv[c];
        s1 = conv[C + c];
        s2 = conv[2 * C + c];
    }
    const float a_log = args.a_log[head], dt_bias = args.dt_bias[head];
    const std::int64_t row = (static_cast<std::int64_t>(batch) * args.heads.H_v + head) * (kStateDim / 2);
    const bool norms = half == 0 && warp == 0;
    __nv_bfloat162 w_pair[2] = {}, z_pair[2] = {};
    if (norms) {
#pragma unroll
        for (int k = 0; k < 2; ++k) { w_pair[k] = reinterpret_cast<const __nv_bfloat162*>(args.norm_weight)[lane + 32 * k]; }
    }
    pdl::enter_streaming();

    if (norms) {
#pragma unroll
        for (int k = 0; k < 2; ++k) { z_pair[k] = reinterpret_cast<const __nv_bfloat162*>(args.z)[row + lane + 32 * k]; }
    }

    // 1. The convolution (causal_conv1d's snapshot decode arithmetic).
    __nv_bfloat16 x0{};
    if (convolves) {
        x0 = args.qkv[static_cast<std::int64_t>(batch) * C + c];
        float acc = 0.0f;
        acc += w0 * __bfloat162float(s0);
        acc += w1 * __bfloat162float(s1);
        acc += w2 * __bfloat162float(s2);
        acc += w3 * __bfloat162float(x0);
        __nv_bfloat16* destination = part == 0 ? q_s : part == 1 ? k_s : v_s;
        destination[i]             = __float2bfloat16_rn(silu(acc));
        if (part == 2) { // this half's own channels
            conv[c]         = s1;
            conv[C + c]     = s2;
            conv[2 * C + c] = x0;
        }
    }
    // Every CTA of the key head's cluster has read its q/k state (and this CTA's q, k, v are in
    // shared memory) before the first one updates that state.
    cg::this_cluster().sync();
    if (convolves && part < 2 && cg::this_cluster().block_rank() == 0) {
        conv[c]         = s1;
        conv[C + c]     = s2;
        conv[2 * C + c] = x0;
    }

    // 2. The recurrent update (gated_delta_net_batch_update_gating's loop) into the first half's o_s:
    // the second half writes its rows across the cluster.
    cg::cluster_group cluster = cg::this_cluster();
    __nv_bfloat16* row_out = half == 0 ? o_s : cluster.map_shared_rank(o_s, cluster.block_rank() - 1);
    const std::int64_t gate_column = batch;
    const ChainAccess access{q_s,
                             k_s,
                             v_s,
                             row_out,
                             args.gate_a[gate_column * args.gate_a_ld + head],
                             args.gate_b[gate_column * args.gate_b_ld + head],
                             a_log,
                             dt_bias,
                             args.scale};
    run_recurrent_sequence<true, OutputEffects>(state, access, coord, 1);
    store_state_tile(state, state_base, coord);
    cluster.sync(); // both halves' rows are in the first half's o_s

    // 3. The gated RMSNorm of the head's output row, by the first half with the second's rows.
    if (norms) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            reinterpret_cast<__nv_bfloat162*>(w_s)[lane + 32 * k] = w_pair[k];
            reinterpret_cast<__nv_bfloat162*>(z_s)[lane + 32 * k] = z_pair[k];
        }
        __syncwarp();
        rmsnorm_warp_row_bf16x2<Epilogue, true>(reinterpret_cast<const __nv_bfloat162*>(o_s),
                                                reinterpret_cast<const __nv_bfloat162*>(w_s),
                                                reinterpret_cast<const __nv_bfloat162*>(z_s),
                                                reinterpret_cast<__nv_bfloat162*>(args.out) + row, kStateDim,
                                                args.eps, lane);
    }
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
