#pragma once

// The Gated DeltaNet decode chain's steps (decode_chain.cu), as device functions shared by its own
// kernel and the persistent GDN-middle experiment.

#include "core/pdl.cuh"
#include "ops/kernel/gdn_gate_values.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace infernix::ops::detail::gated_delta_net {

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

struct ChainShared {
    __align__(16) __nv_bfloat16 q[kStateDim];
    __align__(16) __nv_bfloat16 k[kStateDim];
    __align__(16) __nv_bfloat16 v[kStateDim];
    __align__(16) __nv_bfloat16 o[kStateDim];
    __align__(16) __nv_bfloat16 w[kStateDim]; // the norm's weight and the head's z row
    __align__(16) __nv_bfloat16 z[kStateDim];
};

// What a chain CTA reads before the wait: its state tile and its channel's taps and state.
struct ChainPreload {
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    float w0, w1, w2, w3;
    __nv_bfloat16 s0, s1, s2;
    float a_log, dt_bias;
    __nv_bfloat162 w_pair[2];
};

struct ChainPlace {
    int t, lane, warp, head, half, batch, part, i, c;
    bool convolves, norms;
    std::int64_t slot, row;
    RecurrentCoordinates coord;
};

__device__ __forceinline__ ChainPlace chain_place(const ChainArgs& args, int cta, int batch) {
    ChainPlace at;
    at.t     = static_cast<int>(threadIdx.x);
    at.lane  = at.t % kWarpSize;
    at.warp  = at.t / kWarpSize;
    at.head  = cta / kHalves;
    at.half  = cta % kHalves;
    at.batch = batch;
    const int qk_head = args.heads.qk_head(at.head);
    at.coord = RecurrentCoordinates{at.lane,
                                    at.warp,
                                    batch,
                                    0,
                                    at.warp / kNumWarps,
                                    static_cast<std::uint32_t>(at.head),
                                    static_cast<std::uint32_t>(qk_head),
                                    static_cast<std::uint32_t>(at.half * kHalfRows + at.warp * kDvPerWarp),
                                    static_cast<std::uint32_t>(at.lane * kQkPerLane)};
    at.slot  = args.slots[batch];
    at.part  = at.t / kStateDim;
    at.i     = at.part < 2 ? at.t % kStateDim : at.half * kHalfRows + at.t % kStateDim;
    const int key_width = args.heads.H_qk * kStateDim;
    at.c = at.part == 0 ? qk_head * kStateDim + at.i
         : at.part == 1 ? key_width + qk_head * kStateDim + at.i
                        : 2 * key_width + at.head * kStateDim + at.i;
    at.convolves = at.t < 2 * kStateDim + kHalfRows;
    at.norms     = at.half == 0 && at.warp == 0;
    at.row       = (static_cast<std::int64_t>(batch) * args.heads.H_v + at.head) * (kStateDim / 2);
    return at;
}

__device__ __forceinline__ float* chain_state_base(const ChainArgs& args, const ChainPlace& at) {
    return args.ssm_states + at.slot * args.state_slot_stride + static_cast<std::int64_t>(at.head) * kStateDim * kStateDim;
}

// Before the wait (both states were last written by an earlier round; the slot comes from the
// round's ingress).
__device__ __forceinline__ void chain_preload(const ChainArgs& args, const ChainPlace& at, ChainPreload& p) {
    load_state_tile(p.state, chain_state_base(args, at), at.coord);
    const std::int64_t C = args.channels;
    const __nv_bfloat16* conv = args.conv_states + at.slot * args.conv_slot_stride;
    p.w0 = p.w1 = p.w2 = p.w3 = 0.0f;
    p.s0 = p.s1 = p.s2 = __nv_bfloat16{};
    if (at.convolves) {
        p.w0 = __bfloat162float(args.conv_weight[at.c]);
        p.w1 = __bfloat162float(args.conv_weight[C + at.c]);
        p.w2 = __bfloat162float(args.conv_weight[2 * C + at.c]);
        p.w3 = __bfloat162float(args.conv_weight[3 * C + at.c]);
        p.s0 = conv[at.c];
        p.s1 = conv[C + at.c];
        p.s2 = conv[2 * C + at.c];
    }
    p.a_log   = args.a_log[at.head];
    p.dt_bias = args.dt_bias[at.head];
    p.w_pair[0] = p.w_pair[1] = __nv_bfloat162{};
    if (at.norms) {
#pragma unroll
        for (int k = 0; k < 2; ++k) { p.w_pair[k] = reinterpret_cast<const __nv_bfloat162*>(args.norm_weight)[at.lane + 32 * k]; }
    }
}

// After the wait: the three steps.
template <RmsEpilogue Epilogue>
__device__ __forceinline__ void chain_finish(const ChainArgs& args, const ChainPlace& at, ChainPreload& p, ChainShared& sm) {
    __nv_bfloat162 z_pair[2] = {};
    if (at.norms) {
#pragma unroll
        for (int k = 0; k < 2; ++k) { z_pair[k] = reinterpret_cast<const __nv_bfloat162*>(args.z)[at.row + at.lane + 32 * k]; }
    }
    const std::int64_t C = args.channels;
    __nv_bfloat16* conv  = args.conv_states + at.slot * args.conv_slot_stride;

    // 1. The convolution (causal_conv1d's snapshot decode arithmetic).
    __nv_bfloat16 x0{};
    if (at.convolves) {
        x0 = args.qkv[static_cast<std::int64_t>(at.batch) * C + at.c];
        float acc = 0.0f;
        acc += p.w0 * __bfloat162float(p.s0);
        acc += p.w1 * __bfloat162float(p.s1);
        acc += p.w2 * __bfloat162float(p.s2);
        acc += p.w3 * __bfloat162float(x0);
        __nv_bfloat16* destination = at.part == 0 ? sm.q : at.part == 1 ? sm.k : sm.v;
        destination[at.i]          = __float2bfloat16_rn(silu(acc));
        if (at.part == 2) { // this half's own channels
            conv[at.c]         = p.s1;
            conv[C + at.c]     = p.s2;
            conv[2 * C + at.c] = x0;
        }
    }
    // Every CTA of the key head's cluster has read its q/k state (and this CTA's q, k, v are in
    // shared memory) before the first one updates that state.
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    if (at.convolves && at.part < 2 && cluster.block_rank() == 0) {
        conv[at.c]         = p.s1;
        conv[C + at.c]     = p.s2;
        conv[2 * C + at.c] = x0;
    }

    // 2. The recurrent update (gated_delta_net_batch_update_gating's loop) into the first half's o:
    // the second half writes its rows across the cluster.
    __nv_bfloat16* row_out = at.half == 0 ? sm.o : cluster.map_shared_rank(sm.o, cluster.block_rank() - 1);
    const std::int64_t gate_column = at.batch;
    const ChainAccess access{sm.q,
                             sm.k,
                             sm.v,
                             row_out,
                             args.gate_a[gate_column * args.gate_a_ld + at.head],
                             args.gate_b[gate_column * args.gate_b_ld + at.head],
                             p.a_log,
                             p.dt_bias,
                             args.scale};
    run_recurrent_sequence<true, OutputEffects>(p.state, access, at.coord, 1);
    store_state_tile(p.state, chain_state_base(args, at), at.coord);
    cluster.sync(); // both halves' rows are in the first half's o

    // 3. The gated RMSNorm of the head's output row, by the first half with the second's rows.
    if (at.norms) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            reinterpret_cast<__nv_bfloat162*>(sm.w)[at.lane + 32 * k] = p.w_pair[k];
            reinterpret_cast<__nv_bfloat162*>(sm.z)[at.lane + 32 * k] = z_pair[k];
        }
        __syncwarp();
        rmsnorm_warp_row_bf16x2<Epilogue, true>(reinterpret_cast<const __nv_bfloat162*>(sm.o),
                                                reinterpret_cast<const __nv_bfloat162*>(sm.w),
                                                reinterpret_cast<const __nv_bfloat162*>(sm.z),
                                                reinterpret_cast<__nv_bfloat162*>(args.out) + at.row, kStateDim,
                                                args.eps, at.lane);
    }
}


} // namespace infernix::ops::detail::gated_delta_net
