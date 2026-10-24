// Width equality of Qwen3.8-Flash-Next's exact route (docs/maintainer/model-quality.md section 6.8).
//
// For every Op the exact route runs in the production configuration (recipe C: FP8-row dense
// projections, Q8 hyper-connection mixers / PLE projections / lm_head, NVFP4 W4A16 experts, INT8 KV),
// this checks whether a column's output bits depend on the call's width or on how a sequence is
// decomposed into calls:
//
//   width equality    — the same input columns at width W (W around every route threshold the
//                       kernels' sources show: 1, 2, 8, 9, 15, 16, 17, 32, 33, 64, 65, 255, 256,
//                       257, 512, and 1024..4096 for the two big prefill projections) against the
//                       same column alone at width 1, byte for byte;
//   decomposition     — a sequence of L columns in one call against calls split at off-grid points
//   equality            (7, 17, 1003), 16-aligned points (16, 256, 512, 1024) and the production
//                       tail shape (full 16-chunks + a <16 tail), carrying state, comparing the
//                       outputs and the final state bytes; the stateful ops' decode counterparts
//                       (gated_delta_net_batch_update, causal_conv1d_silu_snapshot_split) are
//                       measured as the "last k tokens one at a time" decomposition.
//
// --invariant runs the construction-exact route (A16ColumnInvariant linears and mixers, the
// recurrent GDN prefill); --strict turns every divergence into a failure, and CTest runs both. Without
// them the binary reports the default routes' divergences (the table the campaign started from).
//
// The inputs are synthetic at the real model shapes in the real stored formats (Flash-Next: H=2560,
// S=4 streams, rank 320, GDN Hqk=16/Hv=48/D=128, conv C=10240 K=4, QSA 24/2 heads D=256, MoE 512
// experts top-10, PLE 16x160 K=4 dilation 3, V=248320). The routed experts (W4A16 exact integer
// arithmetic) and the QSA attention/pooled-key path are covered by the Forward-level decomposition
// check on the real artifact (tests/models/qwen4_exp/test_forward_real.cpp --split-equality), not
// here.
//

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/causal_conv1d_silu.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/gdn_gating.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/sigmoid_mul.h"
#include "infernix/ops/silu_mul.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
namespace qw = infernix::test::quantized_weight;
namespace t  = infernix::test;

constexpr float kEps = 1.0e-6F;

// Flash-Next geometry (docs/maintainer/qwen3_8-flash-next-design.md, model table).
constexpr int kHidden = 2560, kStreams = 4, kRank = 320, kResidual = kStreams * kHidden;
constexpr int kGdnQkHeads = 16, kGdnValueHeads = 48, kGdnDim = 128;
constexpr int kConvChannels = 10240;             // 2*key_width (2*2048) + value_width (6144)
constexpr int kPleTaps = 4, kPleDilation = 3;   // span 9
constexpr int kVocab = 248320;

int g_failures = 0;
bool g_strict = false;
// --invariant: the exact route's column-invariant linear set (ForwardFastPaths::column_invariant).
ops::LinearPolicy g_policy = ops::LinearPolicy::A16Only;

// The GDN prefill arithmetic the policy's route runs (ForwardFastPaths::gdn_arithmetic).
ops::GdnChunkArithmetic gdn_arithmetic() {
    return g_policy == ops::LinearPolicy::A16ColumnInvariant ? ops::GdnChunkArithmetic::Recurrent
                                                             : ops::GdnChunkArithmetic::Exact;
}

// ---------------------------------------------------------------- reporting --

struct Diff {
    bool equal = true;
    std::size_t differing = 0;   // elements
    double max_abs = 0.0;        // promoted to double
    double max_ulp = 0.0;        // ordered bit distance
    std::string where;           // first differing element
};

// The ordered bit distance of two same-width floating words (sign-magnitude -> monotone order).
template <typename Word>
double ulp_distance(Word a, Word b) {
    const auto order = [](Word w) {
        using U = std::make_unsigned_t<Word>;
        const U bits = static_cast<U>(w);
        const U flip = static_cast<U>(1) << (sizeof(Word) * 8 - 1);
        const U body = bits & ~flip;
        return static_cast<double>((bits & flip) ? (flip - body) : (flip + body));
    };
    return std::abs(order(a) - order(b));
}

Diff diff_bf16(const std::uint16_t* a, const std::uint16_t* b, std::size_t count) {
    Diff out;
    for (std::size_t i = 0; i < count; ++i) {
        if (a[i] == b[i]) { continue; }
        if (out.equal) { out.where = "element " + std::to_string(i); }
        out.equal = false;
        ++out.differing;
        out.max_abs = std::max(out.max_abs, std::abs(double(bf16_to_f32(a[i])) - double(bf16_to_f32(b[i]))));
        out.max_ulp = std::max(out.max_ulp, ulp_distance(a[i], b[i]));
    }
    return out;
}

Diff diff_f32(const float* a, const float* b, std::size_t count) {
    Diff out;
    for (std::size_t i = 0; i < count; ++i) {
        if (std::memcmp(a + i, b + i, 4) == 0) { continue; }
        if (out.equal) { out.where = "element " + std::to_string(i); }
        out.equal = false;
        ++out.differing;
        out.max_abs = std::max(out.max_abs, std::abs(double(a[i]) - double(b[i])));
        std::uint32_t ua, ub;
        std::memcpy(&ua, a + i, 4);
        std::memcpy(&ub, b + i, 4);
        out.max_ulp = std::max(out.max_ulp, ulp_distance(ua, ub));
    }
    return out;
}

Diff diff_i32(const std::int32_t* a, const std::int32_t* b, std::size_t count) {
    Diff out;
    for (std::size_t i = 0; i < count; ++i) {
        if (a[i] == b[i]) { continue; }
        if (out.equal) { out.where = "element " + std::to_string(i); }
        out.equal = false;
        ++out.differing;
        out.max_abs = std::max(out.max_abs, std::abs(double(a[i]) - double(b[i])));
    }
    return out;
}

void record(const std::string& op, const std::string& detail, const Diff& d) {
    std::printf("%-46s %-30s %s differing=%zu max_abs=%.3e max_ulp=%.0f%s\n", op.c_str(), detail.c_str(),
                d.equal ? "byte-equal" : "DIFFERS  ", d.differing, d.max_abs, d.max_ulp,
                d.equal ? "" : (" first: " + d.where).c_str());
    if (!d.equal && g_strict) { ++g_failures; }
}

// ------------------------------------------------------------- input makers --

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed, float lo = -2.0F, float hi = 2.0F) {
    std::vector<float> v(count);
    fill_uniform(v, seed, lo, hi);
    std::vector<std::uint16_t> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = f32_to_bf16(v[i]); }
    return out;
}

// A deterministic BF16 column set: column c's values depend on c, so cross-column coupling shows.
std::vector<std::uint16_t> activation(std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * columns);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::uint32_t value = seed ^ static_cast<std::uint32_t>(i) * 0x9E3779B9U;
        value ^= value >> 16;
        value *= 0x7FEB352DU;
        value ^= value >> 15;
        out[i] = f32_to_bf16(static_cast<float>(static_cast<int>(value & 0x3FFU) - 512) / 256.0F);
    }
    return out;
}

DeviceBuffer device_copy(const void* host, std::size_t bytes) {
    DeviceBuffer out(bytes);
    out.copy_from_host(host, bytes);
    return out;
}

// ==================================================================== linear ==

void section_linear_width(const char* name, QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                          const std::vector<std::int32_t>& widths) {
    qw::PackedWeight packed = qw::make_patterned_weight(qtype, n, k, seed);
    DeviceBuffer wbuf = device_copy(packed.payload.data(), packed.payload.size());
    Weight w = packed.device_weight(wbuf.p);
    const std::int32_t widest = *std::max_element(widths.begin(), widths.end());
    const std::vector<std::uint16_t> x = activation(k, widest, seed ^ 0x5A5AU);
    DeviceBuffer xbuf = device_copy(x.data(), x.size() * 2);
    DeviceArena ws(1 << 24);
    // The width-1 reference of every column (the route decode runs).
    std::vector<std::uint16_t> ref(static_cast<std::size_t>(n) * widest);
    {
        DeviceBuffer col(static_cast<std::size_t>(k) * 2), out(static_cast<std::size_t>(n) * 2);
        Tensor tin(col.p, DType::BF16, {k, 1}), tout(out.p, DType::BF16, {n, 1});
        for (std::int32_t c = 0; c < widest; ++c) {
            col.copy_from_host(x.data() + static_cast<std::size_t>(c) * k, static_cast<std::size_t>(k) * 2);
            ops::linear(tin, w, tout, g_policy, ws, nullptr, ops::LinearRoutes::Exact);
            out.copy_to_host(ref.data() + static_cast<std::size_t>(c) * n, static_cast<std::size_t>(n) * 2);
        }
    }
    DeviceBuffer outbuf(static_cast<std::size_t>(n) * widest * 2);
    for (const std::int32_t width : widths) {
        Tensor xin(xbuf.p, DType::BF16, {k, width}), tout(outbuf.p, DType::BF16, {n, width});
        ops::linear(xin, w, tout, g_policy, ws, nullptr, ops::LinearRoutes::Exact);
        std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * width);
        outbuf.copy_to_host(got.data(), got.size() * 2);
        Diff d;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff column = diff_bf16(got.data() + static_cast<std::size_t>(c) * n,
                                          ref.data() + static_cast<std::size_t>(c) * n, n);
            if (!column.equal) {
                d = column;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record(name, "width " + std::to_string(width), d);
    }
}

// Cross-width equality at the 16-multiple widths production actually runs between a prefix-cache
// tap and the chunk: a resumed request's middle calls have the tap's widths, a fresh request's
// the chunk's, so byte equality across these widths is what off-grid resume exactness needs.
void section_linear_cross(const char* name, QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                          std::int32_t reference_width, const std::vector<std::int32_t>& widths) {
    qw::PackedWeight packed = qw::make_patterned_weight(qtype, n, k, seed ^ 0xC0DEU);
    DeviceBuffer wbuf = device_copy(packed.payload.data(), packed.payload.size());
    Weight w = packed.device_weight(wbuf.p);
    const std::int32_t widest = std::max(reference_width, *std::max_element(widths.begin(), widths.end()));
    const std::vector<std::uint16_t> x = activation(k, widest, seed ^ 0x5A5AU);
    DeviceBuffer xbuf = device_copy(x.data(), x.size() * 2);
    DeviceArena ws(1 << 24);
    const auto run_width = [&](std::int32_t width, std::vector<std::uint16_t>& out) {
        Tensor xin(xbuf.p, DType::BF16, {k, width});
        DeviceBuffer outbuf(static_cast<std::size_t>(n) * width * 2);
        Tensor tout(outbuf.p, DType::BF16, {n, width});
        ops::linear(xin, w, tout, g_policy, ws, nullptr, ops::LinearRoutes::Exact);
        out.resize(static_cast<std::size_t>(n) * width);
        outbuf.copy_to_host(out.data(), out.size() * 2);
    };
    std::vector<std::uint16_t> ref;
    run_width(reference_width, ref);
    for (const std::int32_t width : widths) {
        std::vector<std::uint16_t> got;
        run_width(width, got);
        Diff d;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff column =
                diff_bf16(got.data() + static_cast<std::size_t>(c) * n,
                          ref.data() + static_cast<std::size_t>(c) * n, n);
            if (!column.equal) {
                d = column;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record(name, "width " + std::to_string(width) + " vs width " + std::to_string(reference_width), d);
    }
}

// ======================================================== hyper-connection ==

struct MixProblem {
    qw::PackedWeight down, up;
    DeviceBuffer down_dev, up_dev;
    Weight down_w, up_w;
    std::vector<std::uint16_t> norm;
    DeviceBuffer norm_dev;
};

std::uint32_t next_rand(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

double uniform_rand(std::uint32_t& state) {
    return double(static_cast<std::int32_t>(next_rand(state) >> 8) - (1 << 23)) / double(1 << 23);
}

std::uint16_t fp16_bits(double value) {
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    const int e = exponent - 1;
    const auto fraction = static_cast<int>(std::lround((2.0 * mantissa - 1.0) * 1024.0));
    if (e < -14 || e > 15) { throw std::logic_error("scale outside FP16's normal range"); }
    return static_cast<std::uint16_t>(((e + 15) << 10) | std::min(fraction, 1023));
}

// Q8 weights of the model's scale (random codes and FP16 group scales as in
// test_hyper_connection.cpp's randomize_q8, so the gates are not saturated).
void randomize_q8(qw::PackedWeight& w, std::uint32_t seed) {
    const std::int32_t n = w.weight.n, k = w.weight.k, padded = w.weight.padded_shape[1];
    std::uint32_t state = seed;
    for (std::int32_t r = 0; r < n; ++r) {
        for (std::int32_t c = 0; c < padded; ++c) {
            const auto code = c < k ? static_cast<std::int32_t>(next_rand(state) % 255U) - 127 : 0;
            w.payload[static_cast<std::size_t>(r) * padded + c] =
                static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
        }
        for (std::int32_t g = 0; g < padded / 32; ++g) {
            const double gain = 2.0 + 4.0 * (next_rand(state) >> 8) / double(1 << 24);
            const auto bits = fp16_bits(gain / (127.0 * std::sqrt(double(k))));
            const std::size_t at = w.scale_plane_offset + (static_cast<std::size_t>(r) * (padded / 32) + g) * 2;
            w.payload[at] = static_cast<std::uint8_t>(bits & 0xFFU);
            w.payload[at + 1] = static_cast<std::uint8_t>(bits >> 8);
        }
    }
}

MixProblem make_mix_problem() {
    MixProblem p;
    p.down = t::linear::make_q8_g32_fp16_weight(kRank + kStreams, kResidual, 501U);
    p.up = t::linear::make_q8_g32_fp16_weight(kResidual, kRank, 509U);
    randomize_q8(p.down, 601U);
    randomize_q8(p.up, 607U);
    p.down_dev = device_copy(p.down.payload.data(), p.down.payload.size());
    p.up_dev = device_copy(p.up.payload.data(), p.up.payload.size());
    p.down_w = p.down.device_weight(p.down_dev.p);
    p.up_w = p.up.device_weight(p.up_dev.p);
    std::uint32_t state = 77U;
    p.norm.resize(kResidual);
    for (auto& v : p.norm) { v = f32_to_bf16(static_cast<float>(0.25 * uniform_rand(state))); }
    p.norm_dev = device_copy(p.norm.data(), kResidual * 2);
    return p;
}

// Residual columns: streams at different scales so the per-stream norms matter.
std::vector<std::uint16_t> mix_residual(std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint16_t> r(static_cast<std::size_t>(kResidual) * columns);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < r.size(); ++i) {
        const int s = static_cast<int>((i % kResidual) / kHidden);
        r[i] = f32_to_bf16(static_cast<float>(uniform_rand(state)) * float(1 << s));
    }
    return r;
}

struct MixRun {
    std::vector<std::uint16_t> x;  // [H, T]
    std::vector<float> inject;     // [S, T]
};

MixRun mix_run(MixProblem& p, const std::uint16_t* residual, std::int32_t width, WorkspaceArena& ws) {
    DeviceBuffer rd(static_cast<std::size_t>(kResidual) * width * 2),
        xd(static_cast<std::size_t>(kHidden) * width * 2), id(static_cast<std::size_t>(kStreams) * width * 4);
    rd.copy_from_host(residual, static_cast<std::size_t>(kResidual) * width * 2);
    Tensor R(rd.p, DType::BF16, {kResidual, width});
    Tensor norm(p.norm_dev.p, DType::BF16, {kResidual});
    Tensor x(xd.p, DType::BF16, {kHidden, width});
    Tensor inj(id.p, DType::FP32, {kStreams, width});
    ops::hyper_connection_mix(R, norm, p.down_w, p.up_w, g_policy, kStreams, kRank, kEps, x, &inj, ws,
                              nullptr);
    cuda_synchronize();
    MixRun out;
    out.x.resize(static_cast<std::size_t>(kHidden) * width);
    xd.copy_to_host(out.x.data(), out.x.size() * 2);
    out.inject.resize(static_cast<std::size_t>(kStreams) * width);
    id.copy_to_host(out.inject.data(), out.inject.size() * 4);
    return out;
}

void section_hyper_connection_mix() {
    const std::int32_t widths[] = {1, 2, 8, 9, 15, 16, 17, 28, 29, 32, 64, 96, 97, 128, 129, 255, 256, 257, 512};
    constexpr std::int32_t kWidest       = 512;
    constexpr std::int32_t kFusedColumns = 28; // the default route's fused mixer serves 1..28 columns
    MixProblem p = make_mix_problem();
    DeviceArena ws(ops::hyper_connection_mix_workspace_capacity_bytes(p.down_w, p.up_w, g_policy,
                                                                     kStreams, kRank, kWidest));
    const std::vector<std::uint16_t> residual = mix_residual(kWidest, 0xBEEFU);
    // Width-1 references (the fused route, as decode runs it), one per column.
    std::vector<MixRun> ref;
    ref.reserve(kWidest);
    for (std::int32_t c = 0; c < kWidest; ++c) {
        ref.push_back(mix_run(p, residual.data() + static_cast<std::size_t>(kResidual) * c, 1, ws));
    }
    // A composed-family reference: the width-512 run's own columns.
    const MixRun wide = mix_run(p, residual.data(), kWidest, ws);
    for (const std::int32_t width : widths) {
        const MixRun run = mix_run(p, residual.data(), width, ws);
        Diff vs_fused, vs_composed;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff xcol = diff_bf16(run.x.data() + static_cast<std::size_t>(c) * kHidden,
                                       ref[c].x.data(), kHidden);
            const Diff icol = diff_f32(run.inject.data() + static_cast<std::size_t>(c) * kStreams,
                                      ref[c].inject.data(), kStreams);
            if (!xcol.equal || !icol.equal) {
                if (vs_fused.equal) {
                    vs_fused = xcol.equal ? icol : xcol;
                    vs_fused.where += " (column " + std::to_string(c) + ")";
                }
                vs_fused.equal = false;
            }
            if (width > kFusedColumns) {
                const Diff wc = diff_bf16(run.x.data() + static_cast<std::size_t>(c) * kHidden,
                                         wide.x.data() + static_cast<std::size_t>(c) * kHidden, kHidden);
                if (!wc.equal) {
                    if (vs_composed.equal) {
                        vs_composed = wc;
                        vs_composed.where += " (column " + std::to_string(c) + ")";
                    }
                    vs_composed.equal = false;
                }
            }
        }
        record("hyper_connection_mix", "width " + std::to_string(width) + " vs width 1", vs_fused);
        if (width > kFusedColumns) {
            record("hyper_connection_mix", "width " + std::to_string(width) + " vs width 512", vs_composed);
        }
    }
}

// ================================================================== gdn ops ==

void section_gdn_gating() {
    const std::int32_t widths[] = {1, 2, 8, 16, 17, 64, 257, 512};
    constexpr std::int32_t kWidest = 512;
    const std::vector<std::uint16_t> a = make_bf16(kGdnValueHeads * kWidest, 31U, -3.0F, 3.0F);
    const std::vector<std::uint16_t> b = make_bf16(kGdnValueHeads * kWidest, 37U, -3.0F, 3.0F);
    std::vector<float> a_log(kGdnValueHeads), dt_bias(kGdnValueHeads);
    for (int h = 0; h < kGdnValueHeads; ++h) {
        a_log[h] = -0.7F + 0.3F * std::sin(float(h));
        dt_bias[h] = 0.25F;
    }
    DeviceBuffer a_dev = device_copy(a.data(), a.size() * 2), b_dev = device_copy(b.data(), b.size() * 2);
    DeviceBuffer al_dev = device_copy(a_log.data(), a_log.size() * 4);
    DeviceBuffer dt_dev = device_copy(dt_bias.data(), dt_bias.size() * 4);
    Tensor A(al_dev.p, DType::FP32, {kGdnValueHeads}), DT(dt_dev.p, DType::FP32, {kGdnValueHeads});
    std::vector<float> g_ref(kGdnValueHeads * kWidest), beta_ref(kGdnValueHeads * kWidest);
    {
        DeviceBuffer g(4 * kGdnValueHeads), beta(4 * kGdnValueHeads);
        Tensor go(g.p, DType::FP32, {kGdnValueHeads, 1}), bo(beta.p, DType::FP32, {kGdnValueHeads, 1});
        for (std::int32_t c = 0; c < kWidest; ++c) {
            a_dev.copy_from_host(a.data() + c * kGdnValueHeads, kGdnValueHeads * 2, 0);
            b_dev.copy_from_host(b.data() + c * kGdnValueHeads, kGdnValueHeads * 2, 0);
            Tensor ai(a_dev.p, DType::BF16, {kGdnValueHeads, 1}), bi(b_dev.p, DType::BF16, {kGdnValueHeads, 1});
            ops::gdn_gating(ai, bi, A, DT, go, bo, nullptr);
            g.copy_to_host(g_ref.data() + c * kGdnValueHeads, 4 * kGdnValueHeads);
            beta.copy_to_host(beta_ref.data() + c * kGdnValueHeads, 4 * kGdnValueHeads);
        }
    }
    DeviceBuffer g_dev(4 * kGdnValueHeads * kWidest), beta_dev(4 * kGdnValueHeads * kWidest);
    for (const std::int32_t width : widths) {
        a_dev.copy_from_host(a.data(), static_cast<std::size_t>(kGdnValueHeads) * width * 2);
        b_dev.copy_from_host(b.data(), static_cast<std::size_t>(kGdnValueHeads) * width * 2);
        Tensor ai(a_dev.p, DType::BF16, {kGdnValueHeads, width}), bi(b_dev.p, DType::BF16, {kGdnValueHeads, width});
        Tensor go(g_dev.p, DType::FP32, {kGdnValueHeads, width}), bo(beta_dev.p, DType::FP32, {kGdnValueHeads, width});
        ops::gdn_gating(ai, bi, A, DT, go, bo, nullptr);
        cuda_synchronize();
        std::vector<float> g(static_cast<std::size_t>(kGdnValueHeads) * width),
            beta(static_cast<std::size_t>(kGdnValueHeads) * width);
        g_dev.copy_to_host(g.data(), g.size() * 4);
        beta_dev.copy_to_host(beta.data(), beta.size() * 4);
        Diff d;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff gc = diff_f32(g.data() + c * kGdnValueHeads, g_ref.data() + c * kGdnValueHeads, kGdnValueHeads);
            const Diff bc = diff_f32(beta.data() + c * kGdnValueHeads, beta_ref.data() + c * kGdnValueHeads,
                                     kGdnValueHeads);
            if (!gc.equal || !bc.equal) {
                d = gc.equal ? bc : gc;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("gdn_gating", "width " + std::to_string(width), d);
    }
}

void section_gated_rmsnorm() {
    const std::int32_t columns[] = {1, 2, 8, 16, 17, 48, 64, 512, 1027};
    constexpr std::int32_t kRows = kGdnDim;
    const std::int32_t widest = *std::max_element(columns, columns + sizeof(columns) / sizeof(int));
    const std::vector<std::uint16_t> x = make_bf16(kRows * widest, 41U, -3.0F, 3.0F);
    const std::vector<std::uint16_t> z = make_bf16(kRows * widest, 43U, -3.0F, 3.0F);
    const std::vector<std::uint16_t> weight = make_bf16(kRows, 47U, -0.5F, 0.5F);
    DeviceBuffer x_dev = device_copy(x.data(), x.size() * 2), z_dev = device_copy(z.data(), z.size() * 2);
    DeviceBuffer w_dev = device_copy(weight.data(), weight.size() * 2);
    Tensor W(w_dev.p, DType::BF16, {kRows});
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const DeviceExecutionView execution{nullptr, sms};
    std::vector<std::uint16_t> ref(kRows * widest);
    {
        DeviceBuffer out(kRows * 2);
        Tensor o(out.p, DType::BF16, {kRows, 1});
        for (std::int32_t c = 0; c < widest; ++c) {
            x_dev.copy_from_host(x.data() + c * kRows, kRows * 2, 0);
            z_dev.copy_from_host(z.data() + c * kRows, kRows * 2, 0);
            Tensor xi(x_dev.p, DType::BF16, {kRows, 1}), zi(z_dev.p, DType::BF16, {kRows, 1});
            ops::gated_rmsnorm(xi, W, zi, ops::RmsGate::Sigmoid, kEps, o, execution);
            out.copy_to_host(ref.data() + c * kRows, kRows * 2);
        }
    }
    DeviceBuffer out_dev(static_cast<std::size_t>(kRows) * widest * 2);
    for (const std::int32_t count : columns) {
        x_dev.copy_from_host(x.data(), static_cast<std::size_t>(kRows) * count * 2);
        z_dev.copy_from_host(z.data(), static_cast<std::size_t>(kRows) * count * 2);
        Tensor xi(x_dev.p, DType::BF16, {kRows, count}), zi(z_dev.p, DType::BF16, {kRows, count});
        Tensor o(out_dev.p, DType::BF16, {kRows, count});
        ops::gated_rmsnorm(xi, W, zi, ops::RmsGate::Sigmoid, kEps, o, execution);
        cuda_synchronize();
        std::vector<std::uint16_t> got(static_cast<std::size_t>(kRows) * count);
        out_dev.copy_to_host(got.data(), got.size() * 2);
        Diff d;
        for (std::int32_t c = 0; c < count; ++c) {
            const Diff col = diff_bf16(got.data() + c * kRows, ref.data() + c * kRows, kRows);
            if (!col.equal) {
                d = col;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("gated_rmsnorm", "columns " + std::to_string(count), d);
    }
}

// The GDN prefill convolution (causal_conv1d_silu_split, C = 10240, row profile 2048/2048/6144).
void section_gdn_conv_decomposition() {
    constexpr std::int32_t kL = 1027;  // 64 full 16-chunks + a 3-column tail
    constexpr std::int32_t kRows0 = 2048, kRows1 = 2048, kRows2 = 6144;
    const std::int32_t splits[] = {7, 16, 17, 64, 256, 512, 1003, 1024};
    const std::vector<std::uint16_t> x =
        make_bf16(static_cast<std::size_t>(kConvChannels) * kL, 53U, -2.0F, 2.0F);
    const std::vector<std::uint16_t> weight = make_bf16(kConvChannels * 4, 59U, -0.8F, 0.8F);
    const std::vector<std::uint16_t> state0 = make_bf16(kConvChannels * 3, 61U, -1.0F, 1.0F);
    DeviceBuffer x_dev = device_copy(x.data(), x.size() * 2);
    DeviceBuffer w_dev = device_copy(weight.data(), weight.size() * 2);
    Tensor W(w_dev.p, DType::BF16, {kConvChannels, 4});
    // One call over columns [begin, begin+width): returns the three output planes, each
    // [rows, width], and updates the conv state in place.
    const auto run = [&](std::int32_t begin, std::int32_t width, DeviceBuffer& state) {
        x_dev.copy_from_host(x.data() + static_cast<std::size_t>(kConvChannels) * begin,
                             static_cast<std::size_t>(kConvChannels) * width * 2, 0);
        Tensor xi(x_dev.p, DType::BF16, {kConvChannels, width});
        Tensor si(state.p, DType::BF16, {kConvChannels, 3});
        DeviceBuffer d0(static_cast<std::size_t>(kRows0) * width * 2), d1(static_cast<std::size_t>(kRows1) * width * 2),
            d2(static_cast<std::size_t>(kRows2) * width * 2);
        Tensor o0(d0.p, DType::BF16, {kRows0, width}), o1(d1.p, DType::BF16, {kRows1, width}),
            o2(d2.p, DType::BF16, {kRows2, width});
        ops::causal_conv1d_silu_split(xi, W, si, si, o0, o1, o2, nullptr);
        cuda_synchronize();
        std::array<std::vector<std::uint16_t>, 3> out;
        const std::int32_t rows[3] = {kRows0, kRows1, kRows2};
        DeviceBuffer* dev[3] = {&d0, &d1, &d2};
        for (int p = 0; p < 3; ++p) {
            out[p].resize(static_cast<std::size_t>(rows[p]) * width);
            dev[p]->copy_to_host(out[p].data(), out[p].size() * 2);
        }
        return out;
    };
    const auto compare_planes = [&](const std::array<std::vector<std::uint16_t>, 3>& got,
                                    const std::array<std::vector<std::uint16_t>, 3>& ref,
                                    const std::string& detail) {
        const std::int32_t rows[3] = {kRows0, kRows1, kRows2};
        for (int p = 0; p < 3; ++p) {
            record("causal_conv1d_silu_split", detail + " plane " + std::to_string(p),
                   diff_bf16(got[p].data(), ref[p].data(), got[p].size()));
        }
    };
    // Reference: one call over all L columns.
    DeviceBuffer ref_state = device_copy(state0.data(), state0.size() * 2);
    const auto ref_planes = run(0, kL, ref_state);
    std::vector<std::uint16_t> ref_state_host(state0.size());
    ref_state.copy_to_host(ref_state_host.data(), ref_state.bytes);
    for (const std::int32_t split : splits) {
        DeviceBuffer state = device_copy(state0.data(), state0.size() * 2);
        const auto first = run(0, split, state);
        const auto second = run(split, kL - split, state);
        std::array<std::vector<std::uint16_t>, 3> got;
        for (int p = 0; p < 3; ++p) {
            got[p].insert(got[p].end(), first[p].begin(), first[p].end());
            got[p].insert(got[p].end(), second[p].begin(), second[p].end());
        }
        std::vector<std::uint16_t> got_state(state0.size());
        state.copy_to_host(got_state.data(), state.bytes);
        compare_planes(got, ref_planes, "split " + std::to_string(split));
        record("causal_conv1d_silu_split", "split " + std::to_string(split) + " final state",
               diff_bf16(got_state.data(), ref_state_host.data(), state0.size()));
    }
    // Decode tail: [0, 1024) chunked, then the last 3 tokens one at a time through the snapshot
    // form (the decode path), against the one-call reference.
    {
        constexpr std::int32_t kSlots = 8;
        DeviceBuffer states(static_cast<std::size_t>(kConvChannels) * 3 * kSlots * 2);
        states.copy_from_host(state0.data(), state0.size() * 2, 0);
        auto planes = run(0, 1024, states);  // updates slot 0 in place
        const std::int32_t rows[3] = {kRows0, kRows1, kRows2};
        for (int i = 0; i < 3; ++i) {
            x_dev.copy_from_host(x.data() + static_cast<std::size_t>(kConvChannels) * (1024 + i),
                                 static_cast<std::size_t>(kConvChannels) * 2, 0);
            Tensor xi(x_dev.p, DType::BF16, {kConvChannels, 1, 1});
            Tensor si(states.p, DType::BF16, {kConvChannels, 3, kSlots});
            const std::int32_t src = i == 0 ? 0 : i, dst = 1 + i;  // step 0 reads slot 0; the rest chain
            const std::int32_t sel[] = {src, dst};
            DeviceBuffer slots = device_copy(sel, 8);
            Tensor initial(slots.p, DType::I32, {1});
            Tensor base(reinterpret_cast<std::byte*>(slots.p) + 4, DType::I32, {1});
            std::vector<DeviceBuffer> out;
            for (int p = 0; p < 3; ++p) { out.emplace_back(static_cast<std::size_t>(rows[p]) * 2); }
            Tensor o0(out[0].p, DType::BF16, {kRows0, 1}), o1(out[1].p, DType::BF16, {kRows1, 1}),
                o2(out[2].p, DType::BF16, {kRows2, 1});
            ops::causal_conv1d_silu_snapshot_split(xi, W, si, Tensor{}, initial, base, o0, o1, o2, nullptr);
            cuda_synchronize();
            for (int p = 0; p < 3; ++p) {
                std::vector<std::uint16_t> column(static_cast<std::size_t>(rows[p]));
                out[p].copy_to_host(column.data(), column.size() * 2);
                planes[p].insert(planes[p].end(), column.begin(), column.end());
            }
        }
        std::vector<std::uint16_t> got_state(state0.size());
        states.copy_to_host(got_state.data(), state0.size() * 2,
                            static_cast<std::size_t>(kConvChannels) * 3 * 2 * 3);  // slot 3
        compare_planes(planes, ref_planes, "decode tail (1024 + 3 snapshot steps)");
        record("causal_conv1d_silu (decode tail)", "final state",
               diff_bf16(got_state.data(), ref_state_host.data(), state0.size()));
    }
}

// The GDN chunked recurrence (prefill) against split calls and the decode update.
void section_gdn_recurrence_decomposition() {
    constexpr std::int32_t kL = 1027;
    const std::int32_t splits[] = {7, 16, 17, 64, 256, 512, 1003, 1024};
    constexpr std::size_t qk_size = std::size_t(kGdnDim) * kGdnQkHeads * kL;
    constexpr std::size_t v_size = std::size_t(kGdnDim) * kGdnValueHeads * kL;
    constexpr std::size_t per_column = std::size_t(kGdnDim) * kGdnValueHeads;
    constexpr std::size_t state_size = std::size_t(kGdnDim) * kGdnDim * kGdnValueHeads;
    const std::vector<std::uint16_t> q = make_bf16(qk_size, 67U, -1.5F, 1.5F);
    const std::vector<std::uint16_t> k = make_bf16(qk_size, 71U, -1.5F, 1.5F);
    const std::vector<std::uint16_t> v = make_bf16(v_size, 73U, -1.5F, 1.5F);
    std::vector<float> g(kGdnValueHeads * kL), beta(kGdnValueHeads * kL);
    for (std::size_t i = 0; i < g.size(); ++i) {
        g[i] = -0.001F - 0.12F * (float(i % 97) / 97.0F);
        beta[i] = 0.05F + 0.9F * (float(i % 89) / 89.0F);
    }
    std::vector<float> state(state_size);
    fill_uniform(state, 83U, -0.05F, 0.05F);
    DeviceBuffer q_dev = device_copy(q.data(), q.size() * 2), k_dev = device_copy(k.data(), k.size() * 2);
    DeviceBuffer v_dev = device_copy(v.data(), v.size() * 2);
    DeviceBuffer g_dev = device_copy(g.data(), g.size() * 4), beta_dev = device_copy(beta.data(), beta.size() * 4);
    const float scale = 1.0F / std::sqrt(float(kGdnDim));
    DeviceArena ws(ops::gated_delta_net_workspace_capacity_bytes(kGdnQkHeads, kGdnValueHeads, 1, kL));
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const DeviceExecutionView execution{nullptr, sms};
    // Runs columns [begin, begin+width) from `state` in place; returns those columns' outputs.
    const auto run = [&](std::int32_t begin, std::int32_t width, DeviceBuffer& state) {
        Tensor qt(q_dev.p, DType::BF16, {kGdnDim, kGdnQkHeads, kL});
        Tensor kt(k_dev.p, DType::BF16, {kGdnDim, kGdnQkHeads, kL});
        Tensor vt(v_dev.p, DType::BF16, {kGdnDim, kGdnValueHeads, kL});
        Tensor gt(g_dev.p, DType::FP32, {kGdnValueHeads, kL});
        Tensor bt(beta_dev.p, DType::FP32, {kGdnValueHeads, kL});
        Tensor st(state.p, DType::FP32, {kGdnDim, kGdnDim, kGdnValueHeads});
        DeviceBuffer out(per_column * width * 2);
        Tensor ot(out.p, DType::BF16, {kGdnDim, kGdnValueHeads, width});
        ops::gated_delta_net(qt.slice(2, begin, width), kt.slice(2, begin, width), vt.slice(2, begin, width),
                             gt.slice(1, begin, width), bt.slice(1, begin, width), scale,
                             /*normalize_qk=*/true, ws, st, st, ot, execution, gdn_arithmetic());
        cuda_synchronize();
        std::vector<std::uint16_t> got(per_column * width);
        out.copy_to_host(got.data(), out.bytes);
        return got;
    };
    // Reference: one call.
    DeviceBuffer ref_state = device_copy(state.data(), state.size() * 4);
    const std::vector<std::uint16_t> ref_out = run(0, kL, ref_state);
    std::vector<float> ref_state_host(state_size);
    ref_state.copy_to_host(ref_state_host.data(), ref_state.bytes);
    for (const std::int32_t split : splits) {
        DeviceBuffer st = device_copy(state.data(), state.size() * 4);
        const std::vector<std::uint16_t> first = run(0, split, st);
        const std::vector<std::uint16_t> second = run(split, kL - split, st);
        std::vector<std::uint16_t> got(v_size);
        std::copy(first.begin(), first.end(), got.begin());
        std::copy(second.begin(), second.end(), got.begin() + per_column * split);
        std::vector<float> got_state(state_size);
        st.copy_to_host(got_state.data(), st.bytes);
        record("gated_delta_net (chunked)", "split " + std::to_string(split),
               diff_bf16(got.data(), ref_out.data(), v_size));
        record("gated_delta_net (chunked)", "split " + std::to_string(split) + " final state",
               diff_f32(got_state.data(), ref_state_host.data(), state_size));
    }
    // Decode tail: [0, 1024) chunked, then the last 3 tokens one at a time through
    // gated_delta_net_batch_update (the decode update).
    {
        DeviceBuffer states(state_size * 4 * 2);  // slot 0: chunked; slot 1: decode steps
        states.copy_from_host(state.data(), state.size() * 4, 0);
        Tensor qt(q_dev.p, DType::BF16, {kGdnDim, kGdnQkHeads, 1024});
        Tensor kt(k_dev.p, DType::BF16, {kGdnDim, kGdnQkHeads, 1024});
        Tensor vt(v_dev.p, DType::BF16, {kGdnDim, kGdnValueHeads, 1024});
        Tensor gt(g_dev.p, DType::FP32, {kGdnValueHeads, 1024});
        Tensor bt(beta_dev.p, DType::FP32, {kGdnValueHeads, 1024});
        Tensor s0(states.p, DType::FP32, {kGdnDim, kGdnDim, kGdnValueHeads});
        DeviceBuffer out(v_size * 2);
        Tensor ot(out.p, DType::BF16, {kGdnDim, kGdnValueHeads, 1024});
        ops::gated_delta_net(qt, kt, vt, gt, bt, scale, true, ws, s0, s0, ot, execution, gdn_arithmetic());
        cuda_synchronize();
        const std::size_t qk_plane = std::size_t(kGdnDim) * kGdnQkHeads;
        for (int i = 0; i < 3; ++i) {
            const std::int32_t pos = 1024 + i;
            Tensor qi(reinterpret_cast<std::byte*>(q_dev.p) + qk_plane * pos * 2, DType::BF16,
                      {kGdnDim, kGdnQkHeads, 1, 1});
            Tensor ki(reinterpret_cast<std::byte*>(k_dev.p) + qk_plane * pos * 2, DType::BF16,
                      {kGdnDim, kGdnQkHeads, 1, 1});
            Tensor vi(reinterpret_cast<std::byte*>(v_dev.p) + per_column * pos * 2, DType::BF16,
                      {kGdnDim, kGdnValueHeads, 1, 1});
            Tensor gi(reinterpret_cast<std::byte*>(g_dev.p) + kGdnValueHeads * pos * 4, DType::FP32,
                      {kGdnValueHeads, 1, 1});
            Tensor bi(reinterpret_cast<std::byte*>(beta_dev.p) + kGdnValueHeads * pos * 4, DType::FP32,
                      {kGdnValueHeads, 1, 1});
            Tensor s(states.p, DType::FP32, {kGdnDim, kGdnDim, kGdnValueHeads, 2});
            const std::int32_t zero = 0;  // in place on the chunked state in slot 0
            DeviceBuffer sel = device_copy(&zero, 4);
            Tensor selector(sel.p, DType::I32, {1});
            Tensor oi(reinterpret_cast<std::byte*>(out.p) + per_column * pos * 2, DType::BF16,
                      {kGdnDim, kGdnValueHeads, 1, 1});
            ops::gated_delta_net_batch_update(qi, ki, vi, gi, bi, scale, true, s, selector, selector, oi, nullptr);
            cuda_synchronize();
        }
        std::vector<std::uint16_t> got(v_size);
        out.copy_to_host(got.data(), out.bytes);
        std::vector<float> got_state(state_size);
        states.copy_to_host(got_state.data(), state_size * 4, 0);  // slot 0
        record("gdn decode (batch_update)", "1024 chunked + 3 steps out",
               diff_bf16(got.data(), ref_out.data(), v_size));
        record("gdn decode (batch_update)", "final state", diff_f32(got_state.data(), ref_state_host.data(),
                                                                    state_size));
    }
}

// ====================================================================== PLE ==

void section_ple_gate() {
    const std::int32_t widths[] = {1, 8, 16, 17, 64, 257};
    const std::int32_t widest = 257;
    const std::vector<std::uint16_t> key = make_bf16(kResidual * widest, 91U, -2.0F, 2.0F);
    const std::vector<std::uint16_t> value = make_bf16(kHidden * widest, 93U, -2.0F, 2.0F);
    const std::vector<std::uint16_t> residual = make_bf16(kResidual * widest, 95U, -4.0F, 4.0F);
    const std::vector<std::uint16_t> key_norm = make_bf16(kResidual, 97U, -0.5F, 0.5F);
    const std::vector<std::uint16_t> query_norm = make_bf16(kResidual, 99U, -0.5F, 0.5F);
    const std::vector<std::uint16_t> conv_norm = make_bf16(kResidual, 101U, -0.5F, 0.5F);
    DeviceBuffer key_dev = device_copy(key.data(), key.size() * 2);
    DeviceBuffer value_dev = device_copy(value.data(), value.size() * 2);
    DeviceBuffer res_dev = device_copy(residual.data(), residual.size() * 2);
    DeviceBuffer kn = device_copy(key_norm.data(), key_norm.size() * 2);
    DeviceBuffer qn = device_copy(query_norm.data(), query_norm.size() * 2);
    DeviceBuffer cn = device_copy(conv_norm.data(), conv_norm.size() * 2);
    Tensor KN(kn.p, DType::BF16, {kResidual}), QN(qn.p, DType::BF16, {kResidual}), CN(cn.p, DType::BF16, {kResidual});
    DeviceBuffer gated_dev(kResidual * widest * 2), norm_dev(kResidual * widest * 2);
    std::vector<std::uint16_t> gated_ref(kResidual * widest), norm_ref(kResidual * widest);
    {
        DeviceBuffer gd(kResidual * 2), nd(kResidual * 2);
        Tensor go(gd.p, DType::BF16, {kResidual, 1}), no(nd.p, DType::BF16, {kResidual, 1});
        for (std::int32_t c = 0; c < widest; ++c) {
            key_dev.copy_from_host(key.data() + static_cast<std::size_t>(c) * kResidual, kResidual * 2, 0);
            value_dev.copy_from_host(value.data() + static_cast<std::size_t>(c) * kHidden, kHidden * 2, 0);
            res_dev.copy_from_host(residual.data() + static_cast<std::size_t>(c) * kResidual, kResidual * 2, 0);
            Tensor ki(key_dev.p, DType::BF16, {kResidual, 1}), vi(value_dev.p, DType::BF16, {kHidden, 1});
            Tensor ri(res_dev.p, DType::BF16, {kResidual, 1});
            ops::ple_gate(ki, vi, ri, KN, QN, CN, kStreams, kEps, go, no, nullptr);
            gd.copy_to_host(gated_ref.data() + static_cast<std::size_t>(c) * kResidual, kResidual * 2);
            nd.copy_to_host(norm_ref.data() + static_cast<std::size_t>(c) * kResidual, kResidual * 2);
        }
    }
    for (const std::int32_t width : widths) {
        key_dev.copy_from_host(key.data(), static_cast<std::size_t>(kResidual) * width * 2);
        value_dev.copy_from_host(value.data(), static_cast<std::size_t>(kHidden) * width * 2);
        res_dev.copy_from_host(residual.data(), static_cast<std::size_t>(kResidual) * width * 2);
        Tensor ki(key_dev.p, DType::BF16, {kResidual, width}), vi(value_dev.p, DType::BF16, {kHidden, width});
        Tensor ri(res_dev.p, DType::BF16, {kResidual, width});
        Tensor go(gated_dev.p, DType::BF16, {kResidual, width}), no(norm_dev.p, DType::BF16, {kResidual, width});
        ops::ple_gate(ki, vi, ri, KN, QN, CN, kStreams, kEps, go, no, nullptr);
        cuda_synchronize();
        std::vector<std::uint16_t> gated(static_cast<std::size_t>(kResidual) * width),
            normed(static_cast<std::size_t>(kResidual) * width);
        gated_dev.copy_to_host(gated.data(), gated.size() * 2);
        norm_dev.copy_to_host(normed.data(), normed.size() * 2);
        Diff d;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff gd = diff_bf16(gated.data() + c * kResidual, gated_ref.data() + c * kResidual, kResidual);
            const Diff nd =
                diff_bf16(normed.data() + c * kResidual, norm_ref.data() + c * kResidual, kResidual);
            if (!gd.equal || !nd.equal) {
                d = gd.equal ? nd : gd;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("ple_gate", "width " + std::to_string(width), d);
    }
}

void section_ple_conv_decomposition() {
    constexpr std::int32_t kL = 1027;
    constexpr std::int32_t kSpan = (kPleTaps - 1) * kPleDilation;
    const std::int32_t splits[] = {7, 16, 17, 256, 512, 1003, 1024};
    const std::vector<std::uint16_t> gated =
        make_bf16(static_cast<std::size_t>(kConvChannels) * kL, 103U, -1.0F, 1.0F);
    const std::vector<std::uint16_t> normalized =
        make_bf16(static_cast<std::size_t>(kConvChannels) * kL, 107U, -2.0F, 2.0F);
    const std::vector<std::uint16_t> weight = make_bf16(kConvChannels * kPleTaps, 109U, -0.8F, 0.8F);
    const std::vector<std::uint16_t> state0 =
        make_bf16(static_cast<std::size_t>(kConvChannels) * kSpan, 113U, -2.0F, 2.0F);
    const std::vector<std::uint16_t> residual0 =
        make_bf16(static_cast<std::size_t>(kConvChannels) * kL, 127U, -4.0F, 4.0F);
    DeviceBuffer gated_dev = device_copy(gated.data(), gated.size() * 2);
    DeviceBuffer norm_dev = device_copy(normalized.data(), normalized.size() * 2);
    DeviceBuffer w_dev = device_copy(weight.data(), weight.size() * 2);
    Tensor W(w_dev.p, DType::BF16, {kConvChannels, kPleTaps});
    // Runs columns [begin, begin+width): the residual region carries the running stream state,
    // the conv history lives in `states` slot 0.
    const auto run = [&](std::int32_t begin, std::int32_t width, DeviceBuffer& states, DeviceBuffer& residual) {
        gated_dev.copy_from_host(gated.data() + static_cast<std::size_t>(kConvChannels) * begin,
                                 static_cast<std::size_t>(kConvChannels) * width * 2, 0);
        norm_dev.copy_from_host(normalized.data() + static_cast<std::size_t>(kConvChannels) * begin,
                                 static_cast<std::size_t>(kConvChannels) * width * 2, 0);
        residual.copy_from_host(residual0.data() + static_cast<std::size_t>(kConvChannels) * begin,
                                static_cast<std::size_t>(kConvChannels) * width * 2,
                                static_cast<std::size_t>(kConvChannels) * begin * 2);
        Tensor g(gated_dev.p, DType::BF16, {kConvChannels, width});
        Tensor n(norm_dev.p, DType::BF16, {kConvChannels, width});
        Tensor s(states.p, DType::BF16, {kConvChannels, kSpan, 1});
        const std::int32_t zero = 0, sel[] = {zero, zero};
        DeviceBuffer slots = device_copy(sel, 8);
        Tensor src(slots.p, DType::I32, {1}), dst(reinterpret_cast<std::byte*>(slots.p) + 4, DType::I32, {1});
        Tensor r(residual.p, DType::BF16, {kConvChannels, kL});
        Tensor rs = r.slice(1, begin, width);
        ops::ple_conv_inject(g, n, W, kPleDilation, s, src, dst, rs, nullptr);
        cuda_synchronize();
    };
    const std::size_t res_bytes = static_cast<std::size_t>(kConvChannels) * kL * 2;
    DeviceBuffer ref_states = device_copy(state0.data(), state0.size() * 2);
    DeviceBuffer ref_residual(res_bytes);
    run(0, kL, ref_states, ref_residual);
    std::vector<std::uint16_t> ref_res(kConvChannels * kL), ref_state(state0.size());
    ref_residual.copy_to_host(ref_res.data(), res_bytes);
    ref_states.copy_to_host(ref_state.data(), ref_states.bytes);
    for (const std::int32_t split : splits) {
        DeviceBuffer states = device_copy(state0.data(), state0.size() * 2);
        DeviceBuffer residual(res_bytes);
        run(0, split, states, residual);
        run(split, kL - split, states, residual);
        std::vector<std::uint16_t> got(kConvChannels * kL), got_state(state0.size());
        residual.copy_to_host(got.data(), res_bytes);
        states.copy_to_host(got_state.data(), states.bytes);
        record("ple_conv_inject", "split " + std::to_string(split),
               diff_bf16(got.data(), ref_res.data(), got.size()));
        record("ple_conv_inject", "split " + std::to_string(split) + " final state",
               diff_bf16(got_state.data(), ref_state.data(), got_state.size()));
    }
}

// ================================================== elementwise / per-column ==

void section_rmsnorm() {
    const std::int32_t columns[] = {1, 2, 8, 16, 17, 24, 64, 512};
    constexpr std::int32_t kRows = 256;  // the QSA head dimension
    const std::int32_t widest = *std::max_element(columns, columns + sizeof(columns) / sizeof(int));
    const std::vector<std::uint16_t> x = make_bf16(kRows * widest, 131U, -3.0F, 3.0F);
    const std::vector<std::uint16_t> weight = make_bf16(kRows, 137U, -0.5F, 0.5F);
    DeviceBuffer x_dev = device_copy(x.data(), x.size() * 2);
    DeviceBuffer w_dev = device_copy(weight.data(), weight.size() * 2);
    Tensor W(w_dev.p, DType::BF16, {kRows});
    std::vector<std::uint16_t> ref(kRows * widest);
    {
        DeviceBuffer out(kRows * 2);
        Tensor o(out.p, DType::BF16, {kRows, 1});
        for (std::int32_t c = 0; c < widest; ++c) {
            x_dev.copy_from_host(x.data() + c * kRows, kRows * 2, 0);
            Tensor xi(x_dev.p, DType::BF16, {kRows, 1});
            ops::rmsnorm(xi, W, kEps, true, o, nullptr);
            out.copy_to_host(ref.data() + c * kRows, kRows * 2);
        }
    }
    DeviceBuffer out_dev(static_cast<std::size_t>(kRows) * widest * 2);
    for (const std::int32_t count : columns) {
        x_dev.copy_from_host(x.data(), static_cast<std::size_t>(kRows) * count * 2);
        Tensor xi(x_dev.p, DType::BF16, {kRows, count});
        Tensor o(out_dev.p, DType::BF16, {kRows, count});
        ops::rmsnorm(xi, W, kEps, true, o, nullptr);
        cuda_synchronize();
        std::vector<std::uint16_t> got(static_cast<std::size_t>(kRows) * count);
        out_dev.copy_to_host(got.data(), got.size() * 2);
        Diff d;
        for (std::int32_t c = 0; c < count; ++c) {
            const Diff col = diff_bf16(got.data() + c * kRows, ref.data() + c * kRows, kRows);
            if (!col.equal) {
                d = col;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("rmsnorm (QSA head norm)", "columns " + std::to_string(count), d);
    }
}

void section_rope() {
    const std::int32_t widths[] = {1, 2, 8, 16, 17, 64, 257, 512};
    constexpr std::int32_t kQHeads = 24, kKvHeads = 2, kDim = 256, kRotary = 64;
    const std::int32_t widest = 512;
    const auto prepared = ops::prepare_rope(kRotary, 5.0e6F);
    const std::vector<std::uint16_t> q = make_bf16(kDim * kQHeads * widest, 139U, -2.0F, 2.0F);
    const std::vector<std::uint16_t> k = make_bf16(kDim * kKvHeads * widest, 149U, -2.0F, 2.0F);
    DeviceBuffer q_dev = device_copy(q.data(), q.size() * 2);
    DeviceBuffer k_dev = device_copy(k.data(), k.size() * 2);
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const DeviceExecutionView execution{nullptr, sms};
    std::vector<std::uint16_t> q_ref(q.size()), k_ref(k.size());
    {
        DeviceBuffer pos(12);
        Tensor pt(pos.p, DType::I32, {1, 3});
        for (std::int32_t c = 0; c < widest; ++c) {
            q_dev.copy_from_host(q.data() + static_cast<std::size_t>(c) * kDim * kQHeads, kDim * kQHeads * 2, 0);
            k_dev.copy_from_host(k.data() + static_cast<std::size_t>(c) * kDim * kKvHeads, kDim * kKvHeads * 2, 0);
            const std::int32_t single[3] = {c, c, c};
            pos.copy_from_host(single, 12, 0);
            Tensor qi(q_dev.p, DType::BF16, {kDim, kQHeads, 1}), ki(k_dev.p, DType::BF16, {kDim, kKvHeads, 1});
            ops::rope(pt, prepared, qi, ki, execution);
            // rope rotates in place.
            q_dev.copy_to_host(q_ref.data() + static_cast<std::size_t>(c) * kDim * kQHeads, kDim * kQHeads * 2, 0);
            k_dev.copy_to_host(k_ref.data() + static_cast<std::size_t>(c) * kDim * kKvHeads, kDim * kKvHeads * 2, 0);
        }
    }
    // Cross-width: every width against the 512-column run's own columns (do the wide prefill
    // widths share one bit class?).
    const auto stage_positions = [](std::int32_t w) {
        std::vector<std::int32_t> p(3 * static_cast<std::size_t>(w));
        for (std::int32_t t = 0; t < w; ++t) { p[t] = t; p[w + t] = t; p[2 * w + t] = t; }
        return device_copy(p.data(), p.size() * 4);
    };
    {
        q_dev.copy_from_host(q.data(), q.size() * 2);
        k_dev.copy_from_host(k.data(), k.size() * 2);
        Tensor qi(q_dev.p, DType::BF16, {kDim, kQHeads, widest}), ki(k_dev.p, DType::BF16, {kDim, kKvHeads, widest});
        DeviceBuffer pos_widest = stage_positions(widest);
        Tensor pt(pos_widest.p, DType::I32, {widest, 3});
        ops::rope(pt, prepared, qi, ki, execution);
        cuda_synchronize();
        std::vector<std::uint16_t> q512(q.size()), k512(k.size());
        q_dev.copy_to_host(q512.data(), q512.size() * 2);
        k_dev.copy_to_host(k512.data(), k512.size() * 2);
        for (const std::int32_t width : widths) {
            if (width == widest) { continue; }
            q_dev.copy_from_host(q.data(), static_cast<std::size_t>(kDim) * kQHeads * width * 2);
            k_dev.copy_from_host(k.data(), static_cast<std::size_t>(kDim) * kKvHeads * width * 2);
            Tensor qj(q_dev.p, DType::BF16, {kDim, kQHeads, width}), kj(k_dev.p, DType::BF16, {kDim, kKvHeads, width});
            DeviceBuffer pos_w = stage_positions(width);
            Tensor pj(pos_w.p, DType::I32, {width, 3});
            ops::rope(pj, prepared, qj, kj, execution);
            cuda_synchronize();
            std::vector<std::uint16_t> qg(static_cast<std::size_t>(kDim) * kQHeads * width),
                kg(static_cast<std::size_t>(kDim) * kKvHeads * width);
            q_dev.copy_to_host(qg.data(), qg.size() * 2);
            k_dev.copy_to_host(kg.data(), kg.size() * 2);
            Diff d;
            for (std::int32_t c = 0; c < width; ++c) {
                const Diff qd = diff_bf16(qg.data() + c * kDim * kQHeads,
                                          q512.data() + static_cast<std::size_t>(c) * kDim * kQHeads,
                                          kDim * kQHeads);
                const Diff kd = diff_bf16(kg.data() + c * kDim * kKvHeads,
                                          k512.data() + static_cast<std::size_t>(c) * kDim * kKvHeads,
                                          kDim * kKvHeads);
                if (!qd.equal || !kd.equal) {
                    d = qd.equal ? kd : qd;
                    d.where += " (column " + std::to_string(c) + ")";
                    break;
                }
            }
            record("rope (QSA q/k) cross", "width " + std::to_string(width) + " vs width 512", d);
        }
    }
    for (const std::int32_t width : widths) {
        q_dev.copy_from_host(q.data(), static_cast<std::size_t>(kDim) * kQHeads * width * 2);
        k_dev.copy_from_host(k.data(), static_cast<std::size_t>(kDim) * kKvHeads * width * 2);
        Tensor qi(q_dev.p, DType::BF16, {kDim, kQHeads, width}), ki(k_dev.p, DType::BF16, {kDim, kKvHeads, width});
        DeviceBuffer pos_w = stage_positions(width);
        Tensor pt(pos_w.p, DType::I32, {width, 3});
        ops::rope(pt, prepared, qi, ki, execution);
        cuda_synchronize();
        std::vector<std::uint16_t> qg(static_cast<std::size_t>(kDim) * kQHeads * width),
            kg(static_cast<std::size_t>(kDim) * kKvHeads * width);
        q_dev.copy_to_host(qg.data(), qg.size() * 2);
        k_dev.copy_to_host(kg.data(), kg.size() * 2);
        Diff d;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff qd = diff_bf16(qg.data() + c * kDim * kQHeads,
                                      q_ref.data() + static_cast<std::size_t>(c) * kDim * kQHeads,
                                      kDim * kQHeads);
            const Diff kd = diff_bf16(kg.data() + c * kDim * kKvHeads,
                                      k_ref.data() + static_cast<std::size_t>(c) * kDim * kKvHeads,
                                      kDim * kKvHeads);
            if (!qd.equal || !kd.equal) {
                d = qd.equal ? kd : qd;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("rope (QSA q/k)", "width " + std::to_string(width), d);
    }
}

void section_elementwise() {
    const std::int32_t widths[] = {1, 16, 17, 512};
    constexpr std::int32_t kI = 640;  // the shared expert's intermediate width
    const std::int32_t widest = 512;
    const std::vector<std::uint16_t> gate = make_bf16(kI * widest, 151U, -4.0F, 4.0F);
    const std::vector<std::uint16_t> up = make_bf16(kI * widest, 157U, -4.0F, 4.0F);
    DeviceBuffer g_dev = device_copy(gate.data(), gate.size() * 2);
    DeviceBuffer u_dev = device_copy(up.data(), up.size() * 2);
    DeviceBuffer out_dev(gate.size() * 2);
    std::vector<std::uint16_t> silu_ref(kI * widest), sig_ref(kI * widest);
    {
        DeviceBuffer out(kI * 2);
        Tensor o(out.p, DType::BF16, {kI, 1});
        for (std::int32_t c = 0; c < widest; ++c) {
            g_dev.copy_from_host(gate.data() + c * kI, kI * 2, 0);
            u_dev.copy_from_host(up.data() + c * kI, kI * 2, 0);
            Tensor gi(g_dev.p, DType::BF16, {kI, 1}), ui(u_dev.p, DType::BF16, {kI, 1});
            ops::silu_mul(gi, ui, o, nullptr);
            out.copy_to_host(silu_ref.data() + c * kI, kI * 2);
            Tensor xi(u_dev.p, DType::BF16, {kI, 1});
            ops::sigmoid_mul(gi, xi, nullptr);
            u_dev.copy_to_host(sig_ref.data() + c * kI, kI * 2);
        }
    }
    for (const std::int32_t width : widths) {
        g_dev.copy_from_host(gate.data(), static_cast<std::size_t>(kI) * width * 2);
        u_dev.copy_from_host(up.data(), static_cast<std::size_t>(kI) * width * 2);
        Tensor gi(g_dev.p, DType::BF16, {kI, width}), ui(u_dev.p, DType::BF16, {kI, width});
        Tensor o(out_dev.p, DType::BF16, {kI, width});
        ops::silu_mul(gi, ui, o, nullptr);
        cuda_synchronize();
        std::vector<std::uint16_t> got(static_cast<std::size_t>(kI) * width);
        out_dev.copy_to_host(got.data(), got.size() * 2);
        Diff d;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff col = diff_bf16(got.data() + c * kI, silu_ref.data() + c * kI, kI);
            if (!col.equal) {
                d = col;
                d.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("silu_mul (shared swiglu)", "width " + std::to_string(width), d);
        u_dev.copy_from_host(up.data(), static_cast<std::size_t>(kI) * width * 2);
        Tensor xi(u_dev.p, DType::BF16, {kI, width});
        ops::sigmoid_mul(gi, xi, nullptr);
        cuda_synchronize();
        std::vector<std::uint16_t> xg(static_cast<std::size_t>(kI) * width);
        u_dev.copy_to_host(xg.data(), xg.size() * 2);
        Diff sd;
        for (std::int32_t c = 0; c < width; ++c) {
            const Diff col = diff_bf16(xg.data() + c * kI, sig_ref.data() + c * kI, kI);
            if (!col.equal) {
                sd = col;
                sd.where += " (column " + std::to_string(c) + ")";
                break;
            }
        }
        record("sigmoid_mul (QSA output gate)", "width " + std::to_string(width), sd);
    }
}

// ========================================================= projection_fp32 ==

void section_projection_fp32() {
    // The MoE router: two BF16 weights ([K, 512] experts and [K, 1] shared score).
    {
        const std::int32_t widths[] = {1, 2, 8, 16, 17, 64, 255, 256, 257, 512};
        constexpr std::int32_t kRows = 513;
        const std::int32_t widest = 512;
        std::vector<std::uint16_t> router_w(kHidden * 512), shared_w(kHidden);
        {
            std::vector<float> r(kHidden * 512), s(kHidden);
            fill_uniform(r, 163U, -0.05F, 0.05F);
            fill_uniform(s, 167U, -0.05F, 0.05F);
            for (std::size_t i = 0; i < r.size(); ++i) { router_w[i] = f32_to_bf16(r[i]); }
            for (std::size_t i = 0; i < s.size(); ++i) { shared_w[i] = f32_to_bf16(s[i]); }
        }
        const std::vector<std::uint16_t> x = activation(kHidden, widest, 173U);
        DeviceBuffer rw = device_copy(router_w.data(), router_w.size() * 2);
        DeviceBuffer sw = device_copy(shared_w.data(), shared_w.size() * 2);
        DeviceBuffer x_dev = device_copy(x.data(), x.size() * 2);
        Tensor router(rw.p, DType::BF16, {kHidden, 512}), shared(sw.p, DType::BF16, {kHidden, 1});
        const Tensor* rows[] = {&router, &shared};
        std::vector<float> ref(kRows * widest);
        {
            DeviceBuffer out(kRows * 4), col(kHidden * 2);
            Tensor o(out.p, DType::FP32, {kRows, 1});
            for (std::int32_t c = 0; c < widest; ++c) {
                col.copy_from_host(x.data() + static_cast<std::size_t>(c) * kHidden, kHidden * 2);
                Tensor xi(col.p, DType::BF16, {kHidden, 1});
                ops::projection_fp32(xi, rows, o, nullptr);
                out.copy_to_host(ref.data() + static_cast<std::size_t>(c) * kRows, kRows * 4);
            }
        }
        DeviceBuffer out_dev(static_cast<std::size_t>(kRows) * widest * 4);
        for (const std::int32_t width : widths) {
            Tensor xi(x_dev.p, DType::BF16, {kHidden, width});
            Tensor o(out_dev.p, DType::FP32, {kRows, width});
            ops::projection_fp32(xi, rows, o, nullptr);
            cuda_synchronize();
            std::vector<float> got(static_cast<std::size_t>(kRows) * width);
            out_dev.copy_to_host(got.data(), got.size() * 4);
            Diff d;
            for (std::int32_t c = 0; c < width; ++c) {
                const Diff col =
                    diff_f32(got.data() + c * kRows, ref.data() + static_cast<std::size_t>(c) * kRows, kRows);
                if (!col.equal) {
                    d = col;
                    d.where += " (column " + std::to_string(c) + ")";
                    break;
                }
            }
            record("projection_fp32 (MoE router)", "width " + std::to_string(width), d);
        }
    }
    // The Q8 lm_head (the exact-route text head of this artifact).
    {
        const std::int32_t widths[] = {1, 2, 8, 9, 15, 16, 17, 32, 64, 257};
        const std::int32_t widest = 257;
        qw::PackedWeight packed = t::linear::make_q8_g32_fp16_weight(kVocab, kHidden, 181U);
        DeviceBuffer wbuf = device_copy(packed.payload.data(), packed.payload.size());
        Weight w = packed.device_weight(wbuf.p);
        const std::vector<std::uint16_t> x = activation(kHidden, widest, 191U);
        DeviceBuffer x_dev = device_copy(x.data(), x.size() * 2);
        std::vector<float> ref(static_cast<std::size_t>(kVocab) * widest);
        {
            DeviceBuffer col(kHidden * 2), out(static_cast<std::size_t>(kVocab) * 4);
            Tensor o(out.p, DType::FP32, {kVocab, 1});
            for (std::int32_t c = 0; c < widest; ++c) {
                col.copy_from_host(x.data() + static_cast<std::size_t>(c) * kHidden, kHidden * 2);
                Tensor xi(col.p, DType::BF16, {kHidden, 1});
                ops::projection_fp32(xi, w, o, nullptr);
                out.copy_to_host(ref.data() + static_cast<std::size_t>(c) * kVocab,
                                 static_cast<std::size_t>(kVocab) * 4);
            }
        }
        DeviceBuffer out_dev(static_cast<std::size_t>(kVocab) * widest * 4);
        for (const std::int32_t width : widths) {
            Tensor xi(x_dev.p, DType::BF16, {kHidden, width});
            Tensor o(out_dev.p, DType::FP32, {kVocab, width});
            ops::projection_fp32(xi, w, o, nullptr);
            cuda_synchronize();
            std::vector<float> got(static_cast<std::size_t>(kVocab) * width);
            out_dev.copy_to_host(got.data(), got.size() * 4);
            Diff d;
            for (std::int32_t c = 0; c < width; ++c) {
                const Diff col = diff_f32(got.data() + static_cast<std::size_t>(c) * kVocab,
                                          ref.data() + static_cast<std::size_t>(c) * kVocab, kVocab);
                if (!col.equal) {
                    d = col;
                    d.where += " (column " + std::to_string(c) + ")";
                    break;
                }
            }
            record("projection_fp32 (Q8 lm_head)", "width " + std::to_string(width), d);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--strict") { g_strict = true; }
        if (std::string(argv[i]) == "--invariant") { g_policy = ops::LinearPolicy::A16ColumnInvariant; }
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    std::printf("width-equality harness (Flash-Next exact route, recipe C shapes; divergences are findings)\n");
    try {
        const std::vector<std::int32_t> fp8_widths = {1, 2, 8, 9, 15, 16, 17, 32, 33, 64, 65, 255, 256, 257, 512};
        section_linear_width("fp8 linear [16384,2560] (GDN q/k/v/z)", QType::FP8_E4M3FN_ROW_BF16, 16384, 2560, 71U,
                             {1, 2, 8, 9, 15, 16, 17, 32, 33, 64, 65, 255, 256, 257, 512, 1024, 2048, 4096});
        section_linear_width("fp8 linear [13312,2560] (QSA input)", QType::FP8_E4M3FN_ROW_BF16, 13312, 2560, 73U,
                             {1, 2, 8, 9, 15, 16, 17, 32, 33, 64, 65, 255, 256, 257, 512, 1024, 2048, 4096});
        section_linear_width("fp8 linear [2560,6144] (mixer output)", QType::FP8_E4M3FN_ROW_BF16, 2560, 6144, 77U,
                             fp8_widths);
        section_linear_width("fp8 linear [1280,2560] (shared gate_up)", QType::FP8_E4M3FN_ROW_BF16, 1280, 2560, 79U,
                             fp8_widths);
        section_linear_width("fp8 linear [2560,640] (shared down)", QType::FP8_E4M3FN_ROW_BF16, 2560, 640, 81U,
                             fp8_widths);
        section_linear_width("q8 linear [12800,2560] (PLE projection)", QType::Q8_G32_FP16, 12800, 2560, 83U,
                             {1, 2, 8, 9, 15, 16, 17, 32, 33, 64, 65, 255, 256, 257, 512});
        // The MTP drafter's Q8 projections.
        const std::vector<std::int32_t> q8_widths = {1, 2, 4, 5, 8, 9, 15, 16, 17, 32, 33, 48, 49, 64, 65, 96, 97,
                                                     255, 256, 257, 512, 513};
        section_linear_width("q8 linear [13952,2560] (MTP attention group)", QType::Q8_G32_FP16, 13952, 2560, 85U,
                             q8_widths);
        section_linear_width("q8 linear [2560,2560] (MTP embedding/hidden)", QType::Q8_G32_FP16, 2560, 2560, 87U,
                             q8_widths);
        section_linear_width("q8 linear [1280,2560] (MTP shared gate_up)", QType::Q8_G32_FP16, 1280, 2560, 89U,
                             q8_widths);
        section_linear_width("q8 linear [2560,6144] (MTP attention output)", QType::Q8_G32_FP16, 2560, 6144, 91U,
                             q8_widths);
        section_linear_width("q8 linear [2560,640] (MTP shared down)", QType::Q8_G32_FP16, 2560, 640, 93U,
                             q8_widths);
        const std::vector<std::int32_t> cross_widths = {16, 32, 48, 64, 240, 256, 272, 512, 1024};
        section_linear_cross("fp8 cross [16384,2560] (GDN q/k/v/z)", QType::FP8_E4M3FN_ROW_BF16, 16384, 2560, 71U,
                             2048, cross_widths);
        section_linear_cross("fp8 cross [13312,2560] (QSA input)", QType::FP8_E4M3FN_ROW_BF16, 13312, 2560, 73U,
                             2048, cross_widths);
        section_linear_cross("fp8 cross [2560,6144] (mixer output)", QType::FP8_E4M3FN_ROW_BF16, 2560, 6144, 77U,
                             2048, cross_widths);
        section_linear_cross("q8 cross [12800,2560] (PLE projection)", QType::Q8_G32_FP16, 12800, 2560, 83U, 2048,
                             cross_widths);
        section_hyper_connection_mix();
        section_gdn_gating();
        section_gated_rmsnorm();
        section_gdn_conv_decomposition();
        section_gdn_recurrence_decomposition();
        section_ple_gate();
        section_ple_conv_decomposition();
        section_rmsnorm();
        section_rope();
        section_elementwise();
        section_projection_fp32();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(g_strict ? "%s (strict)\n" : "%s\n", g_failures == 0 ? "OK" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
