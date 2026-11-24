// mhc_mix, mhc_update and mhc_update_mix against the FP64 closed formula
// (include/infernix/ops/mhc.h), at the real geometry (S = 4 streams of H = 4096, 24 fn rows, 20
// Sinkhorn iterations, eps 1e-6, eps_norm 1e-5) and at S = 2 and S = 1. The oracle evaluates, per
// column, the unweighted RMS over all S*H values, f = fn . r, the pre/post sigmoids, the
// output-index softmax and sinkhorn_iters alternating Sinkhorn normalizations of comb, x = sum_s
// pre_s R[sH+d], and the residual update R'[sH+d] = post_s y[d] + sum_j comb[j][s] R[jH+d] — all in
// FP64 from the represented BF16 inputs.
//
// Cases: T in {1, 2, 3, 8, 16, 37, 512} at the real geometry (3 and 37 cut inside the mixer's
// column groups, 512 a prefill width) with realistic magnitudes (streams at 200/50/10/2, fn within
// +/-0.06, base/scale of order one) and the extremes: one dominant stream, a near-tie comb
// (scale[2] = 0, base jitter 1e-4), a sharp comb (scale[2] = 30, softmax underflow into the eps
// floor), saturated gates (base +/-(1..30) rows) and a tiny residual (values ~1e-3, the eps_norm
// floor). S = 2/H = 512 and S = 1/H = 256 at T in {1, 5, 16}.
//
// Criteria: x and the updated residual are BF16 outputs of FP32 arithmetic, so a pointwise bound
// |got - ref| <= 2^-8 |ref| + 2^-12 rms(ref) — one BF16 rounding plus an absolute floor where the
// streams cancel while the gate errors (of order 2^-23 log2(S*H) sum|fn.r| against f, through the
// sigmoid/softmax Lipschitz 1/4) stay proportional to the stream magnitudes. post and comb are FP32
// outputs of that same FP32 f: their bound is 2e-5 absolute + 1e-4 relative for post and 1e-4
// absolute + 1e-4 relative for comb — the sigmoid/softmax slope and the eps-symmetric Sinkhorn
// divisions keep their sensitivity to f's error at that scale, and the measured worst cases across
// this whole domain (realistic, dominant stream, tiny residual, near-tie, sharp and saturated
// comb, S = 1/2/4, T up to 512) sit at or under 0.008 and 0.016 of those bounds. Column
// invariance: a column's x, post, comb and updated residual are bit-identical alone, at any
// position of a 3/37/512-column call, and across the mixer's 8-column group boundary.
// mhc_update_mix equals mhc_update then mhc_mix bit for bit (with distinct and with aliased
// residual/post/comb buffers), a graph-captured call equals the eager call, mhc_update's residual
// meets the same BF16 bound against the FP64 update, and the workspace capacity covers every T.

#include "infernix/ops/mhc.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

float uniform(std::uint32_t& state) {
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) /
           static_cast<float>(1 << 23);
}

struct Geometry {
    int S;
    int H;
    int iters      = 20;
    float eps      = 1e-6F;
    float eps_norm = 1e-5F;

    int rows() const { return (2 + S) * S; }

    int width() const { return S * H; }
};

struct Device {
    void* p = nullptr;

    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes), "cudaMalloc"); }

    ~Device() { cudaFree(p); }

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

enum class ResidualKind { Realistic, Dominant, Tiny };

std::vector<std::uint16_t> make_residual(const Geometry& g, int T, std::uint32_t seed,
                                         ResidualKind kind) {
    std::vector<std::uint16_t> r(static_cast<std::size_t>(T) * g.width());
    std::uint32_t state = seed;
    for (int c = 0; c < T; ++c) {
        for (int s = 0; s < g.S; ++s) {
            float magnitude = 1.0F;
            if (kind == ResidualKind::Realistic) {
                static constexpr float kMagnitudes[4] = {200.0F, 50.0F, 10.0F, 2.0F};
                magnitude = kMagnitudes[s & 3] * (s < 4 ? 1.0F : static_cast<float>(1 << (s - 4)));
            } else if (kind == ResidualKind::Dominant) {
                magnitude = s == 0 ? 300.0F : 0.5F;
            } else {
                magnitude = 1e-3F;
            }
            for (int d = 0; d < g.H; ++d) {
                r[static_cast<std::size_t>(c) * g.width() + static_cast<std::size_t>(s) * g.H + d] =
                    t::f32_to_bf16(uniform(state) * magnitude);
            }
        }
    }
    return r;
}

std::vector<std::uint16_t> make_fn(const Geometry& g, std::uint32_t seed) {
    std::vector<std::uint16_t> f(static_cast<std::size_t>(g.rows()) * g.width());
    std::uint32_t state = seed;
    for (auto& v : f) { v = t::f32_to_bf16(uniform(state) * 0.06F); }
    return f;
}

// base of order one; the comb rows' spread shrinks by `comb_jitter` for near-ties, and `saturate`
// pushes the gates to ±30.
std::vector<float> make_base(const Geometry& g, std::uint32_t seed, float comb_jitter = 1.0F,
                             bool saturate = false) {
    std::vector<float> b(g.rows());
    std::uint32_t state = seed;
    for (int i = 0; i < g.rows(); ++i) {
        if (saturate) {
            b[i] = (i % 2) == 0 ? 30.0F : -30.0F;
        } else {
            const float spread = 2 * g.S <= i ? comb_jitter : 1.0F;
            b[i]               = 0.5F + uniform(state) * spread;
        }
    }
    return b;
}

std::vector<float> make_scale(std::uint32_t seed, float comb_scale = 1.0F) {
    std::uint32_t state = seed;
    return {0.5F + 0.5F * uniform(state), 0.5F + 0.5F * uniform(state), comb_scale};
}

std::vector<std::uint16_t> make_y(const Geometry& g, int T, std::uint32_t seed) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(T) * g.H);
    std::uint32_t state = seed;
    for (auto& v : y) { v = t::f32_to_bf16(uniform(state) * 2.0F); }
    return y;
}

struct Reference {
    std::vector<double> x;        // [T][H]
    std::vector<double> post;     // [T][S]
    std::vector<double> comb;     // [T][S*S]
    std::vector<double> residual; // [T][W], the update applied
};

Reference oracle(const Geometry& g, const std::vector<std::uint16_t>& residual,
                 const std::vector<std::uint16_t>& fn, const std::vector<float>& base,
                 const std::vector<float>& scale, const std::vector<std::uint16_t>& y, int T) {
    const int S = g.S, H = g.H, W = g.width(), rows = g.rows();
    Reference out;
    out.x.assign(static_cast<std::size_t>(T) * H, 0.0);
    out.post.assign(static_cast<std::size_t>(T) * S, 0.0);
    out.comb.assign(static_cast<std::size_t>(T) * S * S, 0.0);
    out.residual.assign(static_cast<std::size_t>(T) * W, 0.0);
    std::vector<double> f(rows), comb_matrix(static_cast<std::size_t>(S) * S);
    for (int c = 0; c < T; ++c) {
        const std::size_t col = static_cast<std::size_t>(c) * W;
        double squared        = 0.0;
        for (int m = 0; m < W; ++m) {
            const double v = t::bf16_to_f32(residual[col + m]);
            squared += v * v;
        }
        const double inv = 1.0 / std::sqrt(squared / W + g.eps_norm);
        for (int i = 0; i < rows; ++i) {
            double acc = 0.0;
            for (int m = 0; m < W; ++m) {
                acc +=
                    static_cast<double>(t::bf16_to_f32(fn[static_cast<std::size_t>(i) * W + m])) *
                    t::bf16_to_f32(residual[col + m]) * inv;
            }
            f[i] = acc;
        }
        for (int s = 0; s < S; ++s) {
            out.post[static_cast<std::size_t>(c) * S + s] =
                2.0 / (1.0 + std::exp(-(f[S + s] * scale[1] + base[S + s])));
        }
        for (int i = 0; i < S; ++i) {
            double logit[4] = {}, m = -INFINITY;
            for (int j = 0; j < S; ++j) {
                logit[j] = f[2 * S + i * S + j] * scale[2] + base[2 * S + i * S + j];
                m        = std::max(m, logit[j]);
            }
            double den = 0.0;
            for (int j = 0; j < S; ++j) {
                comb_matrix[static_cast<std::size_t>(i) * S + j] = std::exp(logit[j] - m);
                den += comb_matrix[static_cast<std::size_t>(i) * S + j];
            }
            for (int j = 0; j < S; ++j) {
                comb_matrix[static_cast<std::size_t>(i) * S + j] =
                    comb_matrix[static_cast<std::size_t>(i) * S + j] / den + g.eps;
            }
        }
        // First the column normalization, then (iters - 1) alternating row/column passes.
        for (int it = 0; it < g.iters; ++it) {
            if (it > 0) {
                for (int i = 0; i < S; ++i) {
                    double rs = 0.0;
                    for (int j = 0; j < S; ++j) {
                        rs += comb_matrix[static_cast<std::size_t>(i) * S + j];
                    }
                    for (int j = 0; j < S; ++j) {
                        comb_matrix[static_cast<std::size_t>(i) * S + j] /= (rs + g.eps);
                    }
                }
            }
            for (int j = 0; j < S; ++j) {
                double cs = 0.0;
                for (int i = 0; i < S; ++i) {
                    cs += comb_matrix[static_cast<std::size_t>(i) * S + j];
                }
                for (int i = 0; i < S; ++i) {
                    comb_matrix[static_cast<std::size_t>(i) * S + j] /= (cs + g.eps);
                }
            }
        }
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                // contract layout: element (input i, output j, t) at (t*S + j)*S + i
                out.comb[static_cast<std::size_t>(c) * S * S + static_cast<std::size_t>(j) * S +
                         i] = comb_matrix[static_cast<std::size_t>(i) * S + j];
            }
        }
        for (int s = 0; s < S; ++s) {
            const double pre = 1.0 / (1.0 + std::exp(-(f[s] * scale[0] + base[s]))) + g.eps;
            for (int d = 0; d < H; ++d) {
                out.x[static_cast<std::size_t>(c) * H + d] +=
                    pre * t::bf16_to_f32(residual[col + static_cast<std::size_t>(s) * H + d]);
            }
        }
        for (int s = 0; s < S; ++s) {
            const double post_v = out.post[static_cast<std::size_t>(c) * S + s];
            for (int d = 0; d < H; ++d) {
                double acc = post_v * t::bf16_to_f32(y[static_cast<std::size_t>(c) * H + d]);
                for (int j = 0; j < S; ++j) {
                    acc += comb_matrix[static_cast<std::size_t>(j) * S + s] *
                           t::bf16_to_f32(residual[col + static_cast<std::size_t>(j) * H + d]);
                }
                out.residual[static_cast<std::size_t>(c) * W + static_cast<std::size_t>(s) * H +
                             d] = acc;
            }
        }
    }
    return out;
}

double rms_of(const std::vector<double>& v) {
    double ss = 0.0;
    for (const double x : v) { ss += x * x; }
    return std::sqrt(ss / static_cast<double>(std::max<std::size_t>(v.size(), 1)));
}

void report(const std::string& label, const t::PointwiseStats& stats) {
    if (stats.non_finite_count != 0 || stats.first_violation >= 0) {
        std::fprintf(stderr,
                     "FAIL: %s outside its bound (worst %.3f of the limit, first violation %lld)\n",
                     label.c_str(), stats.maximum_criterion_ratio,
                     static_cast<long long>(stats.first_violation));
        ++failures;
        return;
    }
    std::printf("  %s: worst %.4f of the bound\n", label.c_str(), stats.maximum_criterion_ratio);
}

// BF16 outputs (x, the updated residual): one rounding plus an absolute floor where streams cancel.
void compare_bf16(const std::string& label, const std::vector<std::uint16_t>& got,
                  const std::vector<double>& ref) {
    const t::PointwiseCriterion criterion{/*absolute*/ std::ldexp(1.0, -12) * rms_of(ref),
                                          /*relative*/ std::ldexp(1.0, -8)};
    std::vector<double> got_double(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) { got_double[i] = t::bf16_to_f32(got[i]); }
    report(label, t::compute_pointwise_stats(got_double.data(), ref.data(),
                                             static_cast<std::int64_t>(ref.size()), criterion));
}

// FP32 gate outputs (post, comb): see the header for the stated bound.
void compare_fp32(const std::string& label, const std::vector<float>& got,
                  const std::vector<double>& ref, double abs_limit, double rel_limit) {
    const t::PointwiseCriterion criterion{abs_limit, rel_limit};
    std::vector<double> got_double(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) { got_double[i] = got[i]; }
    report(label, t::compute_pointwise_stats(got_double.data(), ref.data(),
                                             static_cast<std::int64_t>(ref.size()), criterion));
}

struct RunOut {
    std::vector<std::uint16_t> x;
    std::vector<float> post;
    std::vector<float> comb;
    std::vector<std::uint16_t> residual;
};

bool same_bits(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    return a == b;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void upload(void* destination, const void* source, std::size_t bytes, const char* what) {
    t::cuda_check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), what);
}

std::vector<std::uint16_t> download_bf16(const void* source, std::size_t count) {
    std::vector<std::uint16_t> out(count);
    t::cuda_check(cudaMemcpy(out.data(), source, count * 2, cudaMemcpyDeviceToHost),
                  "download bf16");
    return out;
}

std::vector<float> download_f32(const void* source, std::size_t count) {
    std::vector<float> out(count);
    t::cuda_check(cudaMemcpy(out.data(), source, count * 4, cudaMemcpyDeviceToHost),
                  "download f32");
    return out;
}

// One mixer call through the public contract. With `y` the call is the fused mhc_update_mix (with
// optionally aliased residual/post/comb buffers), without it plain mhc_mix. `graph` captures the
// call into one CUDA graph, as the decode path does.
RunOut call(const Geometry& g, const std::vector<std::uint16_t>& residual,
            const std::vector<std::uint16_t>& fn, const std::vector<float>& base,
            const std::vector<float>& scale, const std::vector<std::uint16_t>* y, int T,
            infernix::WorkspaceArena& workspace, cudaStream_t stream, bool graph = false,
            bool alias = false) {
    const int S = g.S, H = g.H, W = g.width();
    const std::size_t rbytes = static_cast<std::size_t>(W) * T * 2;
    Device rd(rbytes);
    Device fd(fn.size() * 2), bd(base.size() * 4), sd(scale.size() * 4);
    Device xd(static_cast<std::size_t>(H) * T * 2), pd(static_cast<std::size_t>(S) * T * 4),
        cd(static_cast<std::size_t>(S) * S * T * 4);
    Device yd(y != nullptr ? static_cast<std::size_t>(H) * T * 2 : 16);
    Device ro(rbytes), p2(static_cast<std::size_t>(S) * T * 4),
        c2(static_cast<std::size_t>(S) * S * T * 4);
    upload(rd.p, residual.data(), rbytes, "residual");
    upload(fd.p, fn.data(), fn.size() * 2, "fn");
    upload(bd.p, base.data(), base.size() * 4, "base");
    upload(sd.p, scale.data(), scale.size() * 4, "scale");
    if (y != nullptr) { upload(yd.p, y->data(), y->size() * 2, "y"); }

    void* const out_r = alias ? rd.p : ro.p;
    void* const out_p = alias ? pd.p : p2.p;
    void* const out_c = alias ? cd.p : c2.p;
    const Tensor R(rd.p, DType::BF16, {W, T});
    Tensor RO(out_r, DType::BF16, {W, T});
    const Tensor F(fd.p, DType::BF16, {W, g.rows()});
    const Tensor B(bd.p, DType::FP32, {g.rows()});
    const Tensor Sc(sd.p, DType::FP32, {3});
    const Tensor Y(yd.p, DType::BF16, {H, T});
    Tensor X(xd.p, DType::BF16, {H, T});
    Tensor P(pd.p, DType::FP32, {S, T});
    Tensor C(cd.p, DType::FP32, {S, S, T});
    Tensor P2(out_p, DType::FP32, {S, T});
    Tensor C2(out_c, DType::FP32, {S, S, T});
    const auto call_op = [&] {
        if (y != nullptr) {
            infernix::ops::mhc_update_mix(Y, P, C, R, RO, F, B, Sc, g.S, g.iters, g.eps_norm, g.eps,
                                          X, &P2, &C2, workspace, stream);
        } else {
            infernix::ops::mhc_mix(R, F, B, Sc, g.S, g.iters, g.eps_norm, g.eps, X, &P, &C,
                                   workspace, stream);
        }
    };
    if (y != nullptr) {
        // The update consumes the mixer's outputs for this residual: produce them with one mix
        // first (eager, so a captured call below still starts from the same defined state).
        infernix::ops::mhc_mix(R, F, B, Sc, g.S, g.iters, g.eps_norm, g.eps, X, &P, &C, workspace,
                               stream);
        t::cuda_check(cudaStreamSynchronize(stream), "seed mix");
    }
    if (graph) {
        cudaGraph_t gr       = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        call_op();
        t::cuda_check(cudaStreamEndCapture(stream, &gr), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, gr, 0), "instantiate");
        t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(gr);
    } else {
        call_op();
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
    }
    RunOut out;
    out.x = download_bf16(xd.p, static_cast<std::size_t>(H) * T);
    if (y != nullptr) {
        out.post     = download_f32(out_p, static_cast<std::size_t>(S) * T);
        out.comb     = download_f32(out_c, static_cast<std::size_t>(S) * S * T);
        out.residual = download_bf16(out_r, static_cast<std::size_t>(W) * T);
    } else {
        // Plain mix: post/comb land in their own outputs and the residual is read-only.
        out.post     = download_f32(pd.p, static_cast<std::size_t>(S) * T);
        out.comb     = download_f32(cd.p, static_cast<std::size_t>(S) * S * T);
        out.residual = download_bf16(rd.p, static_cast<std::size_t>(W) * T);
    }
    return out;
}

// mhc_update followed by mhc_mix: the two separate calls the fused op must match bit for bit.
RunOut call_separate(const Geometry& g, const std::vector<std::uint16_t>& residual,
                     const std::vector<std::uint16_t>& fn, const std::vector<float>& base,
                     const std::vector<float>& scale, const std::vector<std::uint16_t>& y, int T,
                     infernix::WorkspaceArena& workspace, cudaStream_t stream) {
    const int S = g.S, H = g.H, W = g.width();
    const std::size_t rbytes = static_cast<std::size_t>(W) * T * 2;
    Device rd(rbytes), rm(rbytes);
    Device fd(fn.size() * 2), bd(base.size() * 4), sd(scale.size() * 4);
    Device xd(static_cast<std::size_t>(H) * T * 2), pd(static_cast<std::size_t>(S) * T * 4),
        cd(static_cast<std::size_t>(S) * S * T * 4), yd(static_cast<std::size_t>(H) * T * 2);
    Device p2(static_cast<std::size_t>(S) * T * 4), c2(static_cast<std::size_t>(S) * S * T * 4);
    upload(rd.p, residual.data(), rbytes, "residual");
    upload(yd.p, y.data(), y.size() * 2, "y");
    upload(fd.p, fn.data(), fn.size() * 2, "fn");
    upload(bd.p, base.data(), base.size() * 4, "base");
    upload(sd.p, scale.data(), scale.size() * 4, "scale");
    const Tensor R(rd.p, DType::BF16, {W, T});
    Tensor RM(rm.p, DType::BF16, {W, T});
    const Tensor F(fd.p, DType::BF16, {W, g.rows()});
    const Tensor B(bd.p, DType::FP32, {g.rows()});
    const Tensor Sc(sd.p, DType::FP32, {3});
    const Tensor Y(yd.p, DType::BF16, {H, T});
    Tensor P(pd.p, DType::FP32, {S, T});
    Tensor C(cd.p, DType::FP32, {S, S, T});
    Tensor X(xd.p, DType::BF16, {H, T});
    Tensor P2(p2.p, DType::FP32, {S, T});
    Tensor C2(c2.p, DType::FP32, {S, S, T});
    infernix::ops::mhc_mix(R, F, B, Sc, g.S, g.iters, g.eps_norm, g.eps, X, &P, &C, workspace,
                           stream);
    t::cuda_check(cudaStreamSynchronize(stream), "seed mix");
    infernix::ops::mhc_update(Y, P, C, R, RM, stream);
    infernix::ops::mhc_mix(RM, F, B, Sc, g.S, g.iters, g.eps_norm, g.eps, X, &P2, &C2, workspace,
                           stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    RunOut out;
    out.x        = download_bf16(xd.p, static_cast<std::size_t>(H) * T);
    out.post     = download_f32(p2.p, static_cast<std::size_t>(S) * T);
    out.comb     = download_f32(c2.p, static_cast<std::size_t>(S) * S * T);
    out.residual = download_bf16(rm.p, static_cast<std::size_t>(W) * T);
    return out;
}

// One input variant of one geometry: its weights, shared by every T of the case.
struct Fixture {
    std::string name;
    std::vector<std::uint16_t> fn;
    std::vector<float> base, scale;
};

void compare_mix_outputs(const std::string& label, const RunOut& got, const Reference& ref) {
    compare_bf16(label + " x", got.x, ref.x);
    compare_fp32(label + " post", got.post, ref.post, 2e-5, 1e-4);
    compare_fp32(label + " comb", got.comb, ref.comb, 1e-4, 1e-4);
}

void qualify(const Geometry& g, const Fixture& fx, ResidualKind kind,
             const std::vector<int>& widths, infernix::WorkspaceArena& workspace,
             cudaStream_t stream) {
    for (const int T : widths) {
        const std::string label = fx.name + " T=" + std::to_string(T);
        const auto residual     = make_residual(g, T, 1000U + static_cast<std::uint32_t>(T), kind);
        const auto y            = make_y(g, T, 2000U + static_cast<std::uint32_t>(T));
        const Reference ref     = oracle(g, residual, fx.fn, fx.base, fx.scale, y, T);
        // The mixer alone (x, post, comb); it must not modify the residual.
        const RunOut mixed =
            call(g, residual, fx.fn, fx.base, fx.scale, nullptr, T, workspace, stream);
        check(mixed.residual == residual, label + ": mhc_mix leaves the residual untouched");
        compare_mix_outputs(label + " mix", mixed, ref);
        // The separate and the fused update+mix: the updated residual against the FP64 update, and
        // the second mixer's outputs against the FP64 mixer run on the represented (BF16) updated
        // residual the calls actually consumed.
        const RunOut separate =
            call_separate(g, residual, fx.fn, fx.base, fx.scale, y, T, workspace, stream);
        const RunOut fused = call(g, residual, fx.fn, fx.base, fx.scale, &y, T, workspace, stream);
        const RunOut fused_alias =
            call(g, residual, fx.fn, fx.base, fx.scale, &y, T, workspace, stream, false, true);
        compare_bf16(label + " separate residual", separate.residual, ref.residual);
        compare_bf16(label + " fused residual", fused.residual, ref.residual);
        std::vector<std::uint16_t> updated(separate.residual);
        const Reference ref2 = oracle(g, updated, fx.fn, fx.base, fx.scale, y, T);
        compare_mix_outputs(label + " separate", separate, ref2);
        compare_mix_outputs(label + " fused", fused, ref2);
        compare_mix_outputs(label + " fused alias", fused_alias, ref2);
        check(same_bits(fused.residual, separate.residual) && same_bits(fused.x, separate.x) &&
                  same_bits(fused.post, separate.post) && same_bits(fused.comb, separate.comb),
              label + ": mhc_update_mix equals update then mix bit for bit");
        check(same_bits(fused_alias.residual, separate.residual) &&
                  same_bits(fused_alias.x, separate.x) &&
                  same_bits(fused_alias.post, separate.post) &&
                  same_bits(fused_alias.comb, separate.comb),
              label + ": the aliased mhc_update_mix equals the separate calls bit for bit");
        check(workspace.peak_used() <= workspace.capacity(), label + ": workspace within capacity");
    }
}

// A column's outputs are bit-identical alone, at any position of a wider call, across the mixer's
// 8-column group boundary, and in a prefill-width call.
void check_column_invariance(const Geometry& g, const Fixture& fx,
                             infernix::WorkspaceArena& workspace, cudaStream_t stream) {
    const int W        = g.width();
    auto wide_residual = make_residual(g, 37, 7000U, ResidualKind::Realistic);
    auto y37           = make_y(g, 37, 7100U);
    const std::vector<std::uint16_t> vc(wide_residual.begin() + static_cast<std::ptrdiff_t>(5) * W,
                                        wide_residual.begin() + static_cast<std::ptrdiff_t>(6) * W);
    const std::vector<std::uint16_t> yv(y37.begin() + static_cast<std::ptrdiff_t>(5) * g.H,
                                        y37.begin() + static_cast<std::ptrdiff_t>(6) * g.H);
    for (const int at : {7, 8}) { // across the 8-column group boundary
        std::copy(vc.begin(), vc.end(),
                  wide_residual.begin() + static_cast<std::ptrdiff_t>(at) * W);
        std::copy(yv.begin(), yv.end(), y37.begin() + static_cast<std::ptrdiff_t>(at) * g.H);
    }
    const RunOut wide =
        call(g, wide_residual, fx.fn, fx.base, fx.scale, &y37, 37, workspace, stream);
    const RunOut alone = call(g, vc, fx.fn, fx.base, fx.scale, &yv, 1, workspace, stream);

    // The column outputs of one call, at position `at`, against the column alone.
    const auto column_matches = [&](const RunOut& run, int at) {
        const std::size_t xo = static_cast<std::size_t>(at) * g.H,
                          ro = static_cast<std::size_t>(at) * W;
        const std::size_t po = static_cast<std::size_t>(at) * g.S,
                          co = static_cast<std::size_t>(at) * g.S * g.S;
        return std::vector<std::uint16_t>(run.x.begin() + xo, run.x.begin() + xo + g.H) ==
                   alone.x &&
               std::vector<std::uint16_t>(run.residual.begin() + ro,
                                          run.residual.begin() + ro + W) == alone.residual &&
               same_bits(std::vector<float>(run.post.begin() + po, run.post.begin() + po + g.S),
                         alone.post) &&
               same_bits(
                   std::vector<float>(run.comb.begin() + co, run.comb.begin() + co + g.S * g.S),
                   alone.comb);
    };
    for (const int at : {7, 8}) {
        check(column_matches(wide, at),
              "mhc column " + std::to_string(at) + " of a 37-column call is bit-identical alone");
    }
    // A 3-column call whose first column is vc.
    auto r3 = make_residual(g, 3, 7200U, ResidualKind::Realistic);
    auto y3 = make_y(g, 3, 7300U);
    std::copy(vc.begin(), vc.end(), r3.begin());
    std::copy(yv.begin(), yv.end(), y3.begin());
    const RunOut three = call(g, r3, fx.fn, fx.base, fx.scale, &y3, 3, workspace, stream);
    check(column_matches(three, 0), "mhc column 0 of a 3-column call is bit-identical alone");
    // A prefill-width call with vc first and last.
    auto r512 = make_residual(g, 512, 7400U, ResidualKind::Realistic);
    auto y512 = make_y(g, 512, 7500U);
    for (const int at : {0, 511}) {
        std::copy(vc.begin(), vc.end(), r512.begin() + static_cast<std::ptrdiff_t>(at) * W);
        std::copy(yv.begin(), yv.end(), y512.begin() + static_cast<std::ptrdiff_t>(at) * g.H);
    }
    const RunOut prefill = call(g, r512, fx.fn, fx.base, fx.scale, &y512, 512, workspace, stream);
    for (const int at : {0, 511}) {
        check(column_matches(prefill, at),
              "mhc column " + std::to_string(at) + " of a 512-column call is bit-identical alone");
    }
}

void expect_throw(const std::string& what, const std::function<void()>& body) {
    bool threw = false;
    try {
        body();
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, what);
}

} // namespace

int main() {
    try {
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no usable CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        const Geometry real{4, 4096};
        {
            const std::size_t capacity =
                infernix::ops::mhc_mix_workspace_capacity_bytes(real.S, real.H, 1, 512);
            infernix::WorkspaceArena workspace(capacity);
            Fixture realistic;
            realistic.name  = "realistic";
            realistic.fn    = make_fn(real, 11U);
            realistic.base  = make_base(real, 12U);
            realistic.scale = make_scale(13U);
            qualify(real, realistic, ResidualKind::Realistic, {1, 2, 3, 8, 16, 37, 512}, workspace,
                    stream);
            check_column_invariance(real, realistic, workspace, stream);
            // A graph-captured fused call equals the eager call.
            const auto r5      = make_residual(real, 5, 8000U, ResidualKind::Realistic);
            const auto y5      = make_y(real, 5, 8100U);
            const RunOut eager = call(real, r5, realistic.fn, realistic.base, realistic.scale, &y5,
                                      5, workspace, stream);
            const RunOut captured = call(real, r5, realistic.fn, realistic.base, realistic.scale,
                                         &y5, 5, workspace, stream, true);
            check(same_bits(captured.residual, eager.residual) && same_bits(captured.x, eager.x) &&
                      same_bits(captured.post, eager.post) && same_bits(captured.comb, eager.comb),
                  "a graph-captured mhc_update_mix equals the eager call");
            Fixture dominant;
            dominant.name  = "dominant stream";
            dominant.fn    = make_fn(real, 21U);
            dominant.base  = make_base(real, 22U);
            dominant.scale = make_scale(23U);
            qualify(real, dominant, ResidualKind::Dominant, {1, 3, 16, 37, 512}, workspace, stream);
            Fixture tinycase;
            tinycase.name  = "tiny residual";
            tinycase.fn    = make_fn(real, 31U);
            tinycase.base  = make_base(real, 32U);
            tinycase.scale = make_scale(33U);
            qualify(real, tinycase, ResidualKind::Tiny, {1, 3, 16}, workspace, stream);
            Fixture tie;
            tie.name  = "near-tie comb";
            tie.fn    = make_fn(real, 41U);
            tie.base  = make_base(real, 42U, 1e-4F);
            tie.scale = make_scale(43U, 0.0F);
            qualify(real, tie, ResidualKind::Realistic, {1, 3, 16, 37}, workspace, stream);
            Fixture sharp;
            sharp.name  = "sharp comb";
            sharp.fn    = make_fn(real, 51U);
            sharp.base  = make_base(real, 52U);
            sharp.scale = make_scale(53U, 30.0F);
            qualify(real, sharp, ResidualKind::Realistic, {1, 3, 16, 37}, workspace, stream);
            Fixture saturated;
            saturated.name  = "saturated gates";
            saturated.fn    = make_fn(real, 61U);
            saturated.base  = make_base(real, 62U, 1.0F, true);
            saturated.scale = make_scale(63U);
            qualify(real, saturated, ResidualKind::Realistic, {1, 3, 16}, workspace, stream);
        }
        for (const Geometry& small : {Geometry{2, 512}, Geometry{1, 256}}) {
            const std::size_t capacity =
                infernix::ops::mhc_mix_workspace_capacity_bytes(small.S, small.H, 1, 16);
            infernix::WorkspaceArena workspace(capacity);
            Fixture fx;
            fx.name  = "S=" + std::to_string(small.S);
            fx.fn    = make_fn(small, 71U + static_cast<std::uint32_t>(small.S));
            fx.base  = make_base(small, 72U + static_cast<std::uint32_t>(small.S));
            fx.scale = make_scale(73U + static_cast<std::uint32_t>(small.S));
            qualify(small, fx, ResidualKind::Realistic, {1, 5, 16}, workspace, stream);
        }
        // Shapes and parameters that disagree are refused.
        {
            infernix::WorkspaceArena workspace(1 << 20);
            const auto r  = make_residual(real, 1, 9000U, ResidualKind::Realistic);
            const auto y  = make_y(real, 1, 9100U);
            const auto f  = make_fn(real, 9200U);
            const auto b  = make_base(real, 9300U);
            const auto sc = make_scale(9400U);
            Device rd(r.size() * 2), yd(y.size() * 2), fd(f.size() * 2), bd(b.size() * 4),
                sd(sc.size() * 4);
            Device xd(static_cast<std::size_t>(real.H) * 2),
                pd(static_cast<std::size_t>(real.S) * 4),
                cd(static_cast<std::size_t>(real.S) * real.S * 4);
            upload(rd.p, r.data(), r.size() * 2, "r");
            upload(yd.p, y.data(), y.size() * 2, "y");
            upload(fd.p, f.data(), f.size() * 2, "f");
            upload(bd.p, b.data(), b.size() * 4, "b");
            upload(sd.p, sc.data(), sc.size() * 4, "sc");
            const Tensor R(rd.p, DType::BF16, {real.width(), 1});
            const Tensor Y(yd.p, DType::BF16, {real.H, 1});
            const Tensor F(fd.p, DType::BF16, {real.width(), real.rows()});
            const Tensor B(bd.p, DType::FP32, {real.rows()});
            const Tensor Sc(sd.p, DType::FP32, {3});
            Tensor X(xd.p, DType::BF16, {real.H, 1});
            Tensor P(pd.p, DType::FP32, {real.S, 1});
            Tensor C(cd.p, DType::FP32, {real.S, real.S, 1});
            using infernix::ops::mhc_mix;
            expect_throw("streams = 3 is refused", [&] {
                mhc_mix(R, F, B, Sc, 3, 20, 1e-5F, 1e-6F, X, &P, &C, workspace, stream);
            });
            const Tensor f_bad(fd.p, DType::BF16, {real.width(), real.rows() - 1});
            expect_throw("an fn with the wrong row count is refused", [&] {
                mhc_mix(R, f_bad, B, Sc, real.S, 20, 1e-5F, 1e-6F, X, &P, &C, workspace, stream);
            });
            expect_throw("eps_norm = 0 is refused", [&] {
                mhc_mix(R, F, B, Sc, real.S, 20, 0.0F, 1e-6F, X, &P, &C, workspace, stream);
            });
            expect_throw("sinkhorn_iters = 0 is refused", [&] {
                mhc_mix(R, F, B, Sc, real.S, 0, 1e-5F, 1e-6F, X, &P, &C, workspace, stream);
            });
            const Tensor scale_bad(sd.p, DType::FP32, {2});
            expect_throw("a two-element scale is refused", [&] {
                mhc_mix(R, F, B, scale_bad, real.S, 20, 1e-5F, 1e-6F, X, &P, &C, workspace, stream);
            });
            Tensor x_alias(rd.p, DType::BF16, {real.H, 1}); // aliases the residual
            expect_throw("an x that aliases the residual is refused", [&] {
                mhc_mix(R, F, B, Sc, real.S, 20, 1e-5F, 1e-6F, x_alias, &P, &C, workspace, stream);
            });
            const Tensor comb_bad(cd.p, DType::FP32, {real.S, real.S, 2});
            Tensor R_out(rd.p, DType::BF16, {real.width(), 1}); // aliases the residual, as allowed
            expect_throw("a comb with the wrong T is refused",
                         [&] { infernix::ops::mhc_update(Y, P, comb_bad, R, R_out, stream); });
            expect_throw("a workspace query with streams = 3 is refused", [] {
                (void)infernix::ops::mhc_mix_workspace_capacity_bytes(3, 4096, 1, 4);
            });
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "mhc checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
