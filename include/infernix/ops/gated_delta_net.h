#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/speculative_tree.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Returns the transient arena capacity required by gated_delta_net for the given geometry. It is
 * zero when the private implementation requires no transient storage. The state/head dimension is
 * fixed at 128; `value_heads` must be at least `qk_heads` and divisible by it. The query covers
 * every T in the inclusive interval and throws for an invalid profile or interval.
 */
[[nodiscard]] std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                   std::int32_t value_heads,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

// The arithmetic of calls at or above the chunked route's minimum T (shorter calls run the recurrent
// route either way). Exact is the documented chunked route below. Recurrent runs the recurrent
// arithmetic at every T, token by token bit-identical to gated_delta_net_batch_update and
// gated_delta_net_replay_record from the same state, so outputs and the final state do not depend
// on how a sequence is split into calls (or into decode steps and verifications); it needs no
// workspace and is slower than the chunked route on long calls. Fp16Accumulate (the gdn-f16-accumulate fast
// path) requires normalize_qk: the private Q/K packet holds FP16 instead of BF16, and the chunk
// products of the state (the prediction K S^T, the readout Q S^T and the update K^T diag(suffix)
// Delta) take FP16 operands (S and the scaled Delta rounded to FP16) and accumulate each pair of
// k16 products (32 terms) in FP16 before adding it to an FP32 accumulator; the update's product is
// rounded once and added to the decayed FP32 state. The control products and the master state stay
// FP32. It runs the state products at the FP16-accumulate tensor rate. Compensated runs the chunked
// algorithm with every chunk product compensated: the normalized Q/K carry a second BF16 packet
// (each element minus its packet value), and the control dot products, the prediction K S^T, the
// readout Q S^T, both products with the control matrices and the update K^T diag(suffix) Delta
// split their FP32 operands into high/low pairs (BF16 pairs for the BF16 products, TF32 pairs for
// the TF32 ones) and sum hi*hi + hi*lo + lo*hi in FP32, about three times the exact route's tensor
// work. Its error against the FP64 oracle is the recurrent arithmetic's order, not the chunked
// route's, and its bits depend on the call's chunk grid like the exact route's.
enum class GdnChunkArithmetic : std::uint8_t { Exact, Fp16Accumulate, Recurrent, Compensated };

/**
 * Applies the Gated DeltaNet recurrence independently for each value head h. Let
 * G=value_heads/qk_heads; its Q/K head is qh=floor(h/G). Starting from S_h, for t in increasing
 * order:
 *
 *   alpha       = exp(g[h,t])
 *   delta       = beta[h,t] * (v[:,h,t] - alpha * S_h * k[:,qh,t])
 *   S_h         = alpha * S_h + outer(delta, k[:,qh,t])
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t].
 *
 * Shapes/dtypes are contiguous q/k BF16 [128,Hqk,T], v/out BF16 [128,Hv,T], g/beta FP32 [Hv,T],
 * and state FP32 [128,128,Hv], where Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale` is 1/sqrt(128). When
 * `normalize_qk` is true, the recurrent implementation consumes raw q/k and applies
 * x / sqrt(sum(x^2) + 1e-6) independently to every 128-element row before using it. When false,
 * q/k are consumed as supplied. The oracle evaluates the complete recurrence and `ideal` naively
 * in FP64 from the represented inputs and FP32 initial state. The BF16 out is promoted and
 * compared directly with that result; output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. The chunked route fuses normalization into preparation and stores
 * one private BF16 Q/K packet per Q/K head. FP32 control matrices and residuals feed native TF32
 * products; readout and state updates use BF16 MMA operands with FP32 accumulators. The master
 * state remains FP32 throughout. Scratch capacity is independent of normalize_qk.
 * `execution` supplies the stream and physical SM count for launch decomposition.
 * Inputs and out do not overlap state or one another. `ws` supplies transient storage reported by
 * gated_delta_net_workspace_capacity_bytes; scratch is scoped to the call. T may be any positive
 * value.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, DeviceExecutionView execution,
                     GdnChunkArithmetic arithmetic = GdnChunkArithmetic::Exact);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * arguments may overlap either state.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     DeviceExecutionView execution,
                     GdnChunkArithmetic arithmetic = GdnChunkArithmetic::Exact);

/**
 * One-token update for B independent state-pool slots. q/k are contiguous BF16 [128,Hqk,1,B],
 * v/out are BF16 [128,Hv,1,B], g/beta are FP32 [Hv,1,B], `ssm_states` is contiguous FP32
 * [128,128,Hv,Slots], and both selector tensors are contiguous device I32 [B]. B is in [1,8].
 * Row b reads source_state_slots[b] and publishes the transition to
 * destination_state_slots[b]; selectors may be equal for an in-place row. Destination slots are
 * distinct across active rows. This form uses no arena allocation.
 */
void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, Tensor& ssm_states,
                                  const Tensor& source_state_slots,
                                  const Tensor& destination_state_slots, Tensor& out,
                                  cudaStream_t stream);

/**
 * gated_delta_net_batch_update with the gates evaluated in the update from gdn_gating's inputs:
 * g = -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b), with the same bits as gdn_gating
 * produces. a and b are BF16 [Hv, B] whose columns are contiguous at any element stride of at least
 * Hv (row slices of one control projection qualify); A_log and dt_bias are contiguous FP32 [Hv].
 */
void gated_delta_net_batch_update_gating(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& a,
                                         const Tensor& b, const Tensor& A_log, const Tensor& dt_bias, float scale,
                                         bool normalize_qk, Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, Tensor& out, cudaStream_t stream);

/// Whether gated_delta_net_decode_chain serves this geometry: 128-wide heads, a 4-tap convolution
/// and at most eight value heads per key head.
[[nodiscard]] bool gated_delta_net_decode_chain_supports(std::int32_t qk_heads, std::int32_t value_heads,
                                                         std::int32_t conv_taps, std::int32_t head_dim);

/**
 * A GDN layer's decode middle in one launch, with the bits of the three calls it replaces:
 * causal_conv1d_silu_snapshot_split(qkv, conv_weight, conv_states, {}, slots, slots, q, k, v),
 * gated_delta_net_batch_update_gating(q, k, v, a, b, A_log, dt_bias, scale, true, ssm_states, slots,
 * slots, o) and gated_rmsnorm(o, norm_weight, z, gate, eps, out). qkv: BF16 [C, B] (q | k | v
 * channels); a, b: BF16 [Hv, B]; z, out: BF16 [128, Hv * B]. Both states update in place at
 * `slots`. q, k, v and o are not materialized.
 */
void gated_delta_net_decode_chain(const Tensor& qkv, const Tensor& conv_weight, Tensor& conv_states, const Tensor& a,
                                  const Tensor& b, const Tensor& A_log, const Tensor& dt_bias, float scale,
                                  Tensor& ssm_states, const Tensor& slots, const Tensor& norm_weight, const Tensor& z,
                                  RmsGate gate, float eps, std::int32_t qk_heads, Tensor& out, cudaStream_t stream);

/**
 * Op: gated_delta_net_replay_record
 *
 * Evaluates B independent normalized Gated DeltaNet recurrences from absolute state-pool slots
 * without modifying any state. q/k are BF16 [128,Hq,T,B], v/out are BF16 [128,Hv,T,B], g/beta
 * are FP32 [Hv,T,B], and ssm_states is FP32 [128,128,Hv,S]. The ReplaySSM execution domain is
 * B=1..8 and T=2..16, plus B=1 and T=17..64, with Hq=16 and Hv in {32,48}.
 * scale is 1/sqrt(128).
 *
 * valid_columns is empty for dense rows or device I32 [B], with every caller-supplied extent in
 * [1,T]. initial_state_slots is device I32 [B] containing absolute slots in [0,S). For each valid
 * transition, key_record BF16 [128,Hq,T,B], value_record BF16 [128,Hv,T,B], and gate_record FP32
 * [2,Hv,T,B] receive bit-preserving copies of raw k, v, and {g,beta}. The invalid record suffix is
 * unchanged, while the invalid out suffix is exact BF16 zero. Inputs, state, records, and out are
 * pairwise non-overlapping. Valid outputs must be bit-identical to the normalized recurrent
 * execution from the same FP32 initial state and represented inputs; no independent FP64
 * comparison is needed to establish this record/snapshot equivalence.
 */
void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream);

/**
 * Verification-tree form of gated_delta_net_replay_record. tree_rows is I32
 * [kSpeculativeTreeRowWords,B] (one SpeculativeTreeRow per row), the physical width T is at most
 * kSpeculativeTreeMaxNodes and valid_columns is required. In a tree row (tree != 0) the output of
 * column c is the chain replay of the root path to c, from the row's initial state, bit-identical
 * to the chain Op evaluated on that path's columns alone, and each column's records are the raw
 * copies of its own inputs. A chain row is the chain Op over its valid prefix. Outputs past a
 * row's live columns are zero. The execution domain is the chain Op's; every live column is
 * computed once.
 */
void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, const Tensor& tree_rows,
                                   Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                                   Tensor& out, cudaStream_t stream);

} // namespace infernix::ops
