#pragma once

#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/linear.h"

namespace infernix::models::qwen4_exp::execution {

// The fast paths the Forward runs (models/qwen4_exp/program/fast_paths.h selects them,
// FastPathSelection::forward). The default value is the width-routed exact route; a construction-exact
// configuration's exact route also sets gdn_arithmetic and column_invariant.
struct ForwardFastPaths {
    // The dense projections' fast-route flags: q8-sliced-k (GDN q/k/v/z, GDN and QSA output) and
    // fp8-a8-pair (recipe C's FP8 rows). Each flag reaches only the problems that register it.
    ops::LinearRoutes dense_routes = ops::LinearRoutes::Exact;
    // gdn-conv-fused: the GDN projection and its causal convolution in one kernel (FP32 current token).
    bool gdn_conv_fused = false;
    // qsa-prepare-fused: one QSA prepare kernel (norms, rotations, K/V and index-key appends).
    bool qsa_prepare_fused = false;
    // expert-a4-residual: W4A16 experts' wide calls take the two-term A4 route
    // (ops::offloaded_moe::WideA16::kA4Residual, design §19.3.23); no effect on W4A4 experts.
    bool expert_a4_residual = false;
    // The GDN prefill's arithmetic: gdn-f16-accumulate's FP16 chunked products; in a construction-exact
    // configuration the recurrent arithmetic at every width (a call's bits do not depend on where a
    // prompt's calls start, which decode, verification and resumes move), on the fast route too
    // (against the recurrent reference the chunked route's state error failed its joint decode gate's
    // tail, model-quality.md 6.8); the chunked route otherwise.
    ops::GdnChunkArithmetic gdn_arithmetic = ops::GdnChunkArithmetic::Exact;
    // A construction-exact configuration's exact route: every projection and mix that registers an
    // A16ColumnInvariant route runs it, so a column's bits do not depend on its call's width or its
    // place in it (the resume and decode equivalences the prefix cache relies on).
    bool column_invariant = false;
};

} // namespace infernix::models::qwen4_exp::execution
