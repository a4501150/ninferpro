// The second served record geometry (offloaded_moe::kGeometries[1]: hidden 4096, intermediate
// 2048, SwiGLU limit 10) of offloaded_sparse_moe's exact NVFP4 A16 routes, and the noaux_tc
// routing variant (docs/maintainer/moe-offload-design.md §5, docs/maintainer/op-development.md).
//
// Oracles:
//   - the expert output against an FP64 evaluation on the decoded NVFP4 record (the stored E2M1
//     codes times their E4M3 block scales, the record's global multipliers in alpha_*) and the
//     unencoded BF16 activations, the clamp in the formula;
//   - the CPU W4A16 engine, the GPU dp4a narrow route, its tensor-core narrow route and the exact
//     wide route bit for bit against each other, at every record location (device frame, zero-copy
//     host bank, staged slots) and through the CPU miss service's mapped channel at this hidden;
//   - the noaux_tc routing ids exactly against the FP64 score formula's top-k with lower-id ties
//     (ties built by construction and a bias that flips a selection included), the weights within
//     the FP32 evaluation of the FP64 formula, shared_gate exactly 1.
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace moe   = infernix::ops::offloaded_moe;
namespace canon = infernix::ops::canon;
using infernix::DType;
using infernix::Tensor;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

const moe::Geometry kGeo{4096, 2048, 10.0F};
constexpr int kH = 4096;
constexpr int kI = 2048;

template <class T>
T* device_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

struct Expert {
    std::vector<std::uint8_t> record;
    moe::ExpertScales scales;
};

// Random codes; block scales in the realistic range. No input scales; the alpha_* words are the
// record's multipliers m = fl32(1 / weight global scale).
Expert random_a16_expert(std::mt19937& rng) {
    Expert e;
    e.record.resize(kGeo.record_bytes());
    std::uniform_int_distribution<int> byte(0, 255), scale(40, 80);
    for (auto& b : e.record) { b = static_cast<std::uint8_t>(byte(rng)); }
    auto fix_scales = [&](std::size_t begin, int units) {
        for (int u = 0; u < units; ++u) {
            std::uint8_t* s =
                &e.record[begin + static_cast<std::size_t>(u) * moe::kUnitBytes + 128];
            for (int r = 0; r < 16; ++r) { s[r] = static_cast<std::uint8_t>(scale(rng)); }
        }
    };
    fix_scales(0, kGeo.gate_up_row_groups() * kGeo.gate_up_blocks());
    fix_scales(kGeo.gate_up_bytes(), kGeo.down_row_groups() * kGeo.down_blocks());
    std::uniform_real_distribution<float> lg(-14.0F, -6.0F);
    e.scales = {0.0F, 0.0F, 0.0F, std::exp2(lg(rng)), std::exp2(lg(rng)), std::exp2(lg(rng))};
    return e;
}

// Most elements near 0.05 with outliers, as real activations have, plus wide-range columns.
std::vector<std::uint16_t> random_activations(std::mt19937& rng, int ncols, bool wide_range) {
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::uniform_real_distribution<float> lg(-30.0F, 12.0F);
    std::uniform_int_distribution<int> pos(0, kH - 1);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(ncols) * kH);
    for (auto& v : x) { v = canon::f32_to_bf16_rn(n(rng) * 0.05F); }
    for (int c = 0; c < ncols; ++c) {
        x[static_cast<std::size_t>(c) * kH + static_cast<std::size_t>(17 * (c + 1) % kH)] =
            canon::f32_to_bf16_rn(9.0F);
        if (wide_range) {
            for (int i = 0; i < 64; ++i) {
                const float v = std::exp2(lg(rng)) * (i % 2 ? -1.0F : 1.0F);
                x[static_cast<std::size_t>(c) * kH + static_cast<std::size_t>(pos(rng))] =
                    canon::f32_to_bf16_rn(v);
            }
        }
    }
    return x;
}

// Weight (doubled code times scaled block scale) of row at k in a matrix of `blocks` blocks.
std::int64_t weight_int(const std::uint8_t* matrix, int blocks, int row, int k) {
    const int rg = row / 16, r = row % 16, b = k / 16, kk = k % 16;
    const std::uint8_t* unit =
        matrix + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes;
    const std::uint8_t bytev = unit[32 * (kk / 4) + 4 * (r % 8) + kk % 4];
    const unsigned code      = r < 8 ? (bytev & 15U) : (bytev >> 4);
    return static_cast<std::int64_t>(canon::e2m1_x2(code)) * canon::e4m3_scaled(unit[128 + r]);
}

// The FP64 oracle of the clamped W4A16 expert: the decoded NVFP4 record against the unencoded
// activations, h = silu(min(g, limit)) * clamp(u, -limit, limit) between the two products.
void check_fp64(const Expert& e, const std::vector<std::uint16_t>& x,
                const std::vector<std::uint16_t>& got_all, int ncols, int token = 0) {
    const float limit      = kGeo.swiglu_limit;
    const std::uint8_t* gu = e.record.data();
    const std::uint8_t* dn = gu + kGeo.gate_up_bytes();
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * kH);
    std::vector<std::uint16_t> hq(kI);
    for (int c = 0; c < ncols; ++c) {
        for (int i = 0; i < kI; ++i) {
            double g = 0, u = 0;
            for (int k = 0; k < kH; ++k) {
                const double xv = canon::bf16_to_f32(
                    x[static_cast<std::size_t>(token + c) * kH + static_cast<std::size_t>(k)]);
                g += static_cast<double>(weight_int(gu, kGeo.gate_up_blocks(), 2 * i, k)) * xv;
                u += static_cast<double>(weight_int(gu, kGeo.gate_up_blocks(), 2 * i + 1, k)) * xv;
            }
            g *= std::ldexp(1.0, -10) * e.scales.alpha_gate;
            u *= std::ldexp(1.0, -10) * e.scales.alpha_up;
            const float gb  = canon::bf16_to_f32(canon::f32_to_bf16_rn(static_cast<float>(g)));
            const float ub  = canon::bf16_to_f32(canon::f32_to_bf16_rn(static_cast<float>(u)));
            const double gc = limit > 0.0F && gb > limit ? static_cast<double>(limit) : gb;
            const double uc =
                limit <= 0.0F ? ub : (ub > limit ? limit : (ub < -limit ? -limit : ub));
            const double h                  = gc / (1.0 + std::exp(-gc)) * uc;
            hq[static_cast<std::size_t>(i)] = canon::f32_to_bf16_rn(static_cast<float>(h));
        }
        for (int row = 0; row < kH; ++row) {
            double acc = 0, mag = 0;
            for (int k = 0; k < kI; ++k) {
                const double t = static_cast<double>(weight_int(dn, kGeo.down_blocks(), row, k)) *
                                 canon::bf16_to_f32(hq[static_cast<std::size_t>(k)]);
                acc += t;
                mag += std::fabs(t);
            }
            const double scale = std::ldexp(1.0, -10) * e.scales.alpha_down;
            const double ref   = acc * scale;
            const double got   = canon::bf16_to_f32(
                got_all[static_cast<std::size_t>(c) * kH + static_cast<std::size_t>(row)]);
            const double tol =
                std::ldexp(std::fabs(ref), -7) + mag * scale * std::ldexp(1.0, -7) + 1e-30;
            if (std::fabs(got - ref) > tol) {
                std::fprintf(
                    stderr, "oracle miss c=%d row=%d got=%04x ref=%.9g\n", c, row,
                    got_all[static_cast<std::size_t>(c) * kH + static_cast<std::size_t>(row)], ref);
                ++g_failures;
            }
        }
    }
}

std::vector<std::uint16_t> cpu_run(moe::CpuIsa isa, const Expert& e, const std::uint16_t* x,
                                   int ncols) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * kH);
    const std::uint16_t* xp[moe::kMaxColumns];
    std::uint16_t* yp[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = &x[static_cast<std::size_t>(c) * kH];
        yp[c] = &y[static_cast<std::size_t>(c) * kH];
    }
    moe::expert_forward_a16(isa, e.record.data(), e.scales, ncols, xp, yp,
                            moe::kDefaultPrefetchBytes, kGeo);
    return y;
}

// One column of the whole layer's GPU output for entry (t, slot): expert ids[slot, t] applied to
// x's column t. Everything resident, one pass, no channels: the plain exact routes.

void run_gpu(const std::vector<Expert>& bank, const std::vector<std::uint16_t>& x, int columns,
             int top_k, const std::vector<std::int32_t>& ids_host,
             std::vector<std::uint16_t>& out_host, std::int32_t staging_slots = 0,
             bool zero_copy = false, moe::WideA16 wide_a16 = moe::WideA16::kExact) {
    const int experts_n        = static_cast<int>(bank.size());
    const int entries          = top_k * columns;
    const std::uint64_t stride = kGeo.record_bytes();
    // The pinned host bank (zero-copy source; a resident-only call still names one).
    std::uint8_t* host_bank = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_bank),
                             static_cast<std::size_t>(experts_n) * stride, cudaHostAllocMapped),
               "cudaHostAlloc");
    for (int e = 0; e < experts_n; ++e) {
        std::memcpy(host_bank + static_cast<std::size_t>(e) * stride,
                    bank[static_cast<std::size_t>(e)].record.data(), stride);
    }
    const std::size_t frame_bytes = zero_copy ? 0 : static_cast<std::size_t>(experts_n) * stride;
    std::uint8_t* frame_base      = nullptr;
    std::vector<std::int32_t> frames(experts_n, -1);
    if (!zero_copy) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&frame_base), frame_bytes), "cudaMalloc");
        for (int e = 0; e < experts_n; ++e) {
            cuda_check(cudaMemcpy(frame_base + static_cast<std::size_t>(e) * stride,
                                  bank[static_cast<std::size_t>(e)].record.data(), stride,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy");
            frames[static_cast<std::size_t>(e)] = e;
        }
    }
    std::vector<float> scales_host(static_cast<std::size_t>(experts_n) * 6);
    for (int e = 0; e < experts_n; ++e) {
        const auto& s        = bank[static_cast<std::size_t>(e)].scales;
        const float words[6] = {s.input_gate, s.input_up, s.input_down,
                                s.alpha_gate, s.alpha_up, s.alpha_down};
        for (int i = 0; i < 6; ++i) {
            scales_host[static_cast<std::size_t>(e) * 6 + static_cast<std::size_t>(i)] = words[i];
        }
    }
    auto* d_frames = device_copy(frames);
    auto* d_scales = device_copy(scales_host);
    auto* d_x      = device_copy(x);
    auto* d_ids    = device_copy(ids_host);
    std::vector<float> weights_host(static_cast<std::size_t>(entries), 0.25F);
    std::vector<float> gate_host(static_cast<std::size_t>(columns), 1.0F);
    auto* d_weights                   = device_copy(weights_host);
    auto* d_gate                      = device_copy(gate_host);
    void* workspace                   = nullptr;
    const std::int32_t max_jobs       = experts_n;
    const std::size_t workspace_bytes = infernix::ops::moe_experts_workspace_bytes(
        max_jobs, entries, columns, moe::ExpertActivation::kA16, wide_a16, kGeo);
    cuda_check(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc");
    auto* d_outputs = static_cast<std::uint16_t*>(nullptr);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_outputs),
                          static_cast<std::size_t>(entries) * kH * 2),
               "cudaMalloc");

    const Tensor x_t{d_x, DType::BF16, {kH, columns}};
    Tensor outputs{d_outputs, DType::BF16, {kH, entries}};
    const infernix::ops::MoeRouting routing{Tensor{d_ids, DType::I32, {top_k, columns}},
                                            Tensor{d_weights, DType::FP32, {top_k, columns}},
                                            Tensor{d_gate, DType::FP32, {columns, 1, 1, 1}}};
    auto* dispatch_mem               = static_cast<std::int32_t*>(nullptr);
    const std::size_t dispatch_bytes = infernix::ops::moe_dispatch_bytes(experts_n, entries);
    cuda_check(cudaMalloc(&dispatch_mem, dispatch_bytes), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(dispatch_mem, experts_n, entries);

    infernix::ops::MoeExpertSource source{.frame_base    = frame_base,
                                          .frames        = d_frames,
                                          .host_records  = host_bank,
                                          .record_stride = stride,
                                          .scales =
                                              reinterpret_cast<const moe::ExpertScales*>(d_scales),
                                          .activation    = moe::ExpertActivation::kA16,
                                          .wide_a16      = wide_a16,
                                          .geometry      = kGeo,
                                          .staging_base  = nullptr,
                                          .staging_slots = staging_slots};
    if (staging_slots > 0) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&source.staging_base),
                              static_cast<std::size_t>(staging_slots) * stride),
                   "cudaMalloc");
    }
    infernix::ops::moe_dispatch(routing, experts_n, dispatch, nullptr, nullptr);
    infernix::ops::moe_experts(x_t, dispatch, source, top_k, max_jobs, workspace, outputs, nullptr,
                               true);
    out_host.resize(static_cast<std::size_t>(entries) * kH);
    cuda_check(cudaMemcpy(out_host.data(), d_outputs, out_host.size() * 2, cudaMemcpyDeviceToHost),
               "cudaMemcpy");

    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaFree(d_gate);
    cudaFree(d_outputs);
    cudaFree(d_frames);
    cudaFree(d_scales);
    cudaFree(dispatch_mem);
    cudaFree(workspace);
    if (source.staging_base != nullptr) { cudaFree(source.staging_base); }
    cudaFreeHost(host_bank);
    (void)zero_copy;
}

// The layer call's outputs against the CPU engine's, bit for bit, for every routed entry.
void compare_gpu_cpu(const std::vector<Expert>& experts, const std::vector<std::uint16_t>& x,
                     const std::vector<std::int32_t>& ids_host, int columns, int top_k,
                     const std::vector<std::uint16_t>& gpu_out) {
    for (int t = 0; t < columns; ++t) {
        for (int slot = 0; slot < top_k; ++slot) {
            const int expert = ids_host[static_cast<std::size_t>(t) * top_k + slot];
            const auto ref = cpu_run(moe::best_cpu_isa(), experts[static_cast<std::size_t>(expert)],
                                     x.data() + static_cast<std::size_t>(t) * kH, 1);
            const auto* got   = gpu_out.data() + static_cast<std::size_t>(t) * top_k * kH +
                                static_cast<std::size_t>(slot) * kH;
            const auto* ref_p = ref.data();
            check(std::equal(ref_p, ref_p + kH, got),
                  "GPU route equals the CPU engine bit for bit");
        }
    }
}

// --------------------------------------------------------------------------------- layer routes

// A deterministic routing: narrow jobs (a few columns each), one wide job (expert 0 accumulating
// past kMaxColumns columns at the wider widths), shared-input scales so jobs of more than
// kMaxColumns columns take the exact wide route.
std::vector<std::int32_t> craft_ids(int columns, int top_k) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(columns) * top_k, 0);
    for (int t = 0; t < columns; ++t) {
        // Top slot: expert 0 for most columns (the wide job); a spread over the first experts
        // early.
        ids[static_cast<std::size_t>(t) * top_k] = t < 8 ? t : 0;
        // The other slots cycle experts 1..20: at the wider widths they stay narrow (a few
        // columns).
        ids[static_cast<std::size_t>(t) * top_k + 1] = 1 + t % 20;
        ids[static_cast<std::size_t>(t) * top_k + 2] = 1 + (t * 7 + 3) % 20;
    }
    return ids;
}

void test_layer_routes() {
    std::mt19937 rng(7);
    std::vector<Expert> experts;
    for (int e = 0; e < 25; ++e) { experts.push_back(random_a16_expert(rng)); }
    for (int columns : {2, 40, 130, 300}) {
        const int top_k = 3;
        const auto ids  = craft_ids(columns, top_k);
        const auto x    = random_activations(rng, columns, columns > 8);
        std::vector<std::uint16_t> gpu_out;
        run_gpu(experts, x, columns, top_k, ids, gpu_out);
        compare_gpu_cpu(experts, x, ids, columns, top_k, gpu_out);
        // Routed entries' outputs against the FP64 oracle, decoded NVFP4 and clamp in the formula
        // (every entry of the narrow call; one entry per wider call, the scalar loop is the cost).
        if (columns <= 8) {
            for (int q = 0; q < columns * top_k; ++q) {
                check_fp64(experts[static_cast<std::size_t>(ids[static_cast<std::size_t>(q)])], x,
                           {gpu_out.begin() + static_cast<std::size_t>(q) * kH,
                            gpu_out.begin() + static_cast<std::size_t>(q + 1) * kH},
                           1, q / top_k);
            }
        } else {
            const int q = columns / 2 * top_k;
            check_fp64(experts[static_cast<std::size_t>(ids[static_cast<std::size_t>(q)])], x,
                       {gpu_out.begin() + static_cast<std::size_t>(q) * kH,
                        gpu_out.begin() + static_cast<std::size_t>(q + 1) * kH},
                       1, q / top_k);
        }
        std::printf("layer route columns=%d checked\n", columns);
    }
    check(g_failures == 0,
          "layer routes (dp4a rows, tensor-core narrow, exact wide) at the second geometry");
}

// All records in the pinned host bank, read zero-copy over PCIe (no staging): the routes' record
// reads at the second geometry, bits unchanged.
void test_zero_copy() {
    std::mt19937 rng(11);
    std::vector<Expert> experts;
    for (int e = 0; e < 10; ++e) { experts.push_back(random_a16_expert(rng)); }
    const int columns = 3, top_k = 2;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(columns) * top_k);
    for (int t = 0; t < columns; ++t) {
        ids[static_cast<std::size_t>(t) * top_k]     = t;
        ids[static_cast<std::size_t>(t) * top_k + 1] = 5 + t;
    }
    const auto x = random_activations(rng, columns, false);
    std::vector<std::uint16_t> gpu_out;
    run_gpu(experts, x, columns, top_k, ids, gpu_out, /*staging_slots=*/0, /*zero_copy=*/true);
    compare_gpu_cpu(experts, x, ids, columns, top_k, gpu_out);
}

// Records staged through device slots: the stage kernels copy 14 MB records at this geometry.
void test_staged() {
    std::mt19937 rng(13);
    std::vector<Expert> experts;
    for (int e = 0; e < 6; ++e) { experts.push_back(random_a16_expert(rng)); }
    const int columns = 2, top_k = 2;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(columns) * top_k);
    for (int t = 0; t < columns; ++t) {
        ids[static_cast<std::size_t>(t) * top_k]     = t;
        ids[static_cast<std::size_t>(t) * top_k + 1] = 3 + t;
    }
    const auto x = random_activations(rng, columns, false);
    std::vector<std::uint16_t> gpu_out;
    run_gpu(experts, x, columns, top_k, ids, gpu_out, /*staging_slots=*/2, /*zero_copy=*/true);
    compare_gpu_cpu(experts, x, ids, columns, top_k, gpu_out);
}

// The model's routing shape in a prefill call: top-8 of 288 experts over 140 columns, so most jobs
// have at most kMaxColumns columns and take the tensor-core narrow route (the SwiGLU clamp and
// this geometry's h rows in its epilogue), the rest the exact wide route; records resident and
// read zero-copy.
void test_wide_call_narrow_jobs() {
    std::mt19937 rng(19);
    std::vector<Expert> experts;
    for (int e = 0; e < 288; ++e) { experts.push_back(random_a16_expert(rng)); }
    const int columns = 140, top_k = 8;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(columns) * top_k);
    std::vector<std::int32_t> all(experts.size());
    for (int t = 0; t < columns; ++t) {
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::copy_n(all.begin(), top_k, ids.begin() + static_cast<std::ptrdiff_t>(t) * top_k);
    }
    const auto x = random_activations(rng, columns, true);
    for (const bool zero_copy : {false, true}) {
        std::vector<std::uint16_t> gpu_out;
        run_gpu(experts, x, columns, top_k, ids, gpu_out, 0, zero_copy);
        compare_gpu_cpu(experts, x, ids, columns, top_k, gpu_out);
    }
}

// CPU-served misses through the service's mapped channel: the mapped x/y planes are sized at this
// hidden, and the host engine's bits reach the device outputs unchanged.
void test_cpu_service() {
    std::mt19937 rng(17);
    std::vector<Expert> experts;
    for (int e = 0; e < 8; ++e) { experts.push_back(random_a16_expert(rng)); }
    const int experts_n        = static_cast<int>(experts.size());
    const std::uint64_t stride = kGeo.record_bytes();
    std::uint8_t* host_bank    = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_bank),
                             static_cast<std::size_t>(experts_n) * stride, cudaHostAllocMapped),
               "cudaHostAlloc");
    for (int e = 0; e < experts_n; ++e) {
        std::memcpy(host_bank + static_cast<std::size_t>(e) * stride,
                    experts[static_cast<std::size_t>(e)].record.data(), stride);
    }
    std::vector<float> scales_host(static_cast<std::size_t>(experts_n) * 6);
    for (int e = 0; e < experts_n; ++e) {
        const auto& s        = experts[static_cast<std::size_t>(e)].scales;
        const float words[6] = {s.input_gate, s.input_up, s.input_down,
                                s.alpha_gate, s.alpha_up, s.alpha_down};
        for (int i = 0; i < 6; ++i) {
            scales_host[static_cast<std::size_t>(e) * 6 + static_cast<std::size_t>(i)] = words[i];
        }
    }
    moe::CpuMissService service(
        {moe::CpuMissService::Layer{
            .records       = host_bank,
            .record_stride = stride,
            .scales        = reinterpret_cast<const moe::ExpertScales*>(scales_host.data()),
            .ggml          = {}}},
        moe::CpuMissService::Options{.workers     = 4,
                                     .max_jobs    = 4,
                                     .max_columns = 16,
                                     .activation  = moe::ExpertActivation::kA16,
                                     .geometry    = kGeo});
    const int columns = 2, top_k = 2;
    const auto x = random_activations(rng, columns, false);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(columns) * top_k);
    for (int t = 0; t < columns; ++t) {
        ids[static_cast<std::size_t>(t) * top_k]     = t;
        ids[static_cast<std::size_t>(t) * top_k + 1] = 4 + t;
    }
    const int entries = top_k * columns;
    std::vector<std::int32_t> frames(experts_n, -1);
    auto* d_frames = device_copy(frames);
    auto* d_scales = device_copy(scales_host);
    auto* d_x      = device_copy(x);
    auto* d_ids    = device_copy(ids);
    std::vector<float> gate_host(static_cast<std::size_t>(columns), 1.0F);
    auto* d_gate    = device_copy(gate_host);
    auto* d_outputs = static_cast<std::uint16_t*>(nullptr);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_outputs),
                          static_cast<std::size_t>(entries) * kH * 2),
               "cudaMalloc");
    auto* dispatch_mem = static_cast<std::int32_t*>(nullptr);
    cuda_check(cudaMalloc(&dispatch_mem, infernix::ops::moe_dispatch_bytes(experts_n, entries)),
               "cudaMalloc");
    auto dispatch   = infernix::ops::carve_moe_dispatch(dispatch_mem, experts_n, entries);
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, infernix::ops::moe_experts_workspace_bytes(
                                          experts_n, entries, columns, moe::ExpertActivation::kA16,
                                          moe::WideA16::kExact, kGeo)),
               "cudaMalloc");
    const Tensor x_t{d_x, DType::BF16, {kH, columns}};
    Tensor outputs{d_outputs, DType::BF16, {kH, entries}};
    const infernix::ops::MoeRouting routing{Tensor{d_ids, DType::I32, {top_k, columns}}, Tensor{},
                                            Tensor{}};
    auto* d_weights = static_cast<float*>(nullptr);
    cuda_check(cudaMalloc(&d_weights, static_cast<std::size_t>(entries) * 4), "cudaMalloc");
    infernix::ops::moe_dispatch(routing, experts_n, dispatch, nullptr, nullptr);
    infernix::ops::MoeExpertSource source{.frame_base    = nullptr,
                                          .frames        = d_frames,
                                          .host_records  = host_bank,
                                          .record_stride = stride,
                                          .scales =
                                              reinterpret_cast<const moe::ExpertScales*>(d_scales),
                                          .activation    = moe::ExpertActivation::kA16,
                                          .geometry      = kGeo,
                                          .staging_base  = nullptr,
                                          .staging_slots = 0,
                                          .cpu           = service.channel(0)};
    infernix::ops::moe_experts(x_t, dispatch, source, top_k, experts_n, workspace, outputs, nullptr,
                               true);
    std::vector<std::uint16_t> out_host(static_cast<std::size_t>(entries) * kH);
    cuda_check(cudaMemcpy(out_host.data(), d_outputs, out_host.size() * 2, cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    compare_gpu_cpu(experts, x, ids, columns, top_k, out_host);
    check(service.served_requests() > 0, "the CPU service served the call's misses");
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_frames);
    cudaFree(d_scales);
    cudaFree(d_gate);
    cudaFree(d_outputs);
    cudaFree(dispatch_mem);
    cudaFree(workspace);
    cudaFree(d_weights);
    cudaFreeHost(host_bank);
}

// ---------------------------------------------------------------------------------- noaux_tc

// The FP64 oracle of the noaux_tc route: the scores, the exact top-k ids with lower-id ties, and
// the FP64 weights; ids exact, weights within the FP32 evaluation's rounding.
void check_noaux(const std::vector<float>& logits, const std::vector<float>& bias, int experts,
                 int columns, int top_k, float routed_scale,
                 const std::vector<std::int32_t>& ids_host, const std::vector<float>& weights_host,
                 const std::vector<float>& gate_host) {
    for (int t = 0; t < columns; ++t) {
        std::vector<std::pair<double, int>> scores(static_cast<std::size_t>(experts));
        for (int e = 0; e < experts; ++e) {
            const double s = 1.0 / (1.0 + std::exp(-static_cast<double>(
                                              logits[static_cast<std::size_t>(t) * experts + e])));
            scores[static_cast<std::size_t>(e)] = {
                s + static_cast<double>(bias[static_cast<std::size_t>(e)]), e};
        }
        // Larger score first, the lower id on exact ties (a strict weak order by (score, -id)).
        std::stable_sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) { return a.first > b.first; }
            return a.second < b.second;
        });
        std::vector<double> selected(static_cast<std::size_t>(top_k));
        for (int k = 0; k < top_k; ++k) {
            const auto& [score, id] = scores[static_cast<std::size_t>(k)];
            check(ids_host[static_cast<std::size_t>(t) * top_k + k] == id,
                  "noaux_tc id equals the FP64 oracle's");
            selected[static_cast<std::size_t>(k)] =
                score - static_cast<double>(bias[static_cast<std::size_t>(id)]);
        }
        // The denominator sums the selected s in selection order, as the kernel does.
        double sum = 0;
        for (int k = 0; k < top_k; ++k) { sum += selected[static_cast<std::size_t>(k)]; }
        for (int k = 0; k < top_k; ++k) {
            const double w  = static_cast<double>(routed_scale) *
                              selected[static_cast<std::size_t>(k)] / (sum + 1e-20);
            const float got = weights_host[static_cast<std::size_t>(t) * top_k + k];
            if (std::fabs(static_cast<double>(got) - w) > std::fabs(w) * 1e-5 + 1e-9) {
                std::fprintf(stderr, "noaux weight t=%d k=%d got %.9g want %.9g\n", t, k,
                             static_cast<double>(got), w);
                ++g_failures;
            }
        }
        check(gate_host[static_cast<std::size_t>(t)] == 1.0F, "noaux_tc shared_gate is 1");
    }
}

// Random logits plus constructed exact ties and a bias that flips a selection; every entry point
// (route, fused dispatch, fused dispatch+plan) against the oracle and against each other.
void test_noaux_routing() {
    const int experts = 288, top_k = 8;
    const float routed_scale = 2.5F;
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> logit(-8.0F, 4.0F), bias(-0.3F, 0.3F);
    std::vector<float> logits_host, bias_host;
    for (int columns : {1, 9, 40}) {
        logits_host.assign(static_cast<std::size_t>(columns) * experts, 0.0F);
        for (auto& v : logits_host) { v = logit(rng); }
        bias_host.assign(static_cast<std::size_t>(experts), 0.0F);
        for (auto& v : bias_host) { v = bias(rng); }
        // Exact ties: columns 0 and 1 hold pairs of equal scores (equal logit and bias), so the
        // lower expert id must win; the bias flips one pair's order.
        for (int t = 0; t < std::min(columns, 2); ++t) {
            for (int i = 0; i < top_k + 2; ++i) {
                const float v = 2.0F + i * 0.5F;
                logits_host[static_cast<std::size_t>(t) * experts + static_cast<std::size_t>(i)] =
                    v;
                logits_host[static_cast<std::size_t>(t) * experts +
                            static_cast<std::size_t>(i + 1)] = v;
            }
            if (t == 1) { // the second pair's bias makes its later id win over an earlier one
                bias_host[2] = 0.3F;
                bias_host[3] = -0.3F;
            }
        }
        auto* d_logits  = device_copy(logits_host);
        auto* d_bias    = device_copy(bias_host);
        auto* d_ids     = static_cast<std::int32_t*>(nullptr);
        auto* d_weights = static_cast<float*>(nullptr);
        auto* d_gate    = static_cast<float*>(nullptr);
        cuda_check(cudaMalloc(&d_ids, static_cast<std::size_t>(top_k) * columns * 4), "cudaMalloc");
        cuda_check(cudaMalloc(&d_weights, static_cast<std::size_t>(top_k) * columns * 4),
                   "cudaMalloc");
        cuda_check(cudaMalloc(&d_gate, static_cast<std::size_t>(columns) * 4), "cudaMalloc");

        const Tensor logits_t{d_logits, DType::FP32, {experts, columns}};
        const Tensor bias_t{d_bias, DType::FP32, {experts, 1, 1, 1}};
        infernix::ops::MoeRouting routing{Tensor{d_ids, DType::I32, {top_k, columns}},
                                          Tensor{d_weights, DType::FP32, {top_k, columns}},
                                          Tensor{d_gate, DType::FP32, {columns, 1, 1, 1}}};
        infernix::ops::moe_route_noaux_tc(logits_t, bias_t, routed_scale, top_k, routing, nullptr);
        std::vector<std::int32_t> ids_host(static_cast<std::size_t>(top_k) * columns);
        std::vector<float> weights_host(static_cast<std::size_t>(top_k) * columns);
        std::vector<float> gate_host(static_cast<std::size_t>(columns));
        cuda_check(cudaMemcpy(ids_host.data(), d_ids, ids_host.size() * 4, cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cuda_check(cudaMemcpy(weights_host.data(), d_weights, weights_host.size() * 4,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cuda_check(
            cudaMemcpy(gate_host.data(), d_gate, gate_host.size() * 4, cudaMemcpyDeviceToHost),
            "cudaMemcpy");
        check_noaux(logits_host, bias_host, experts, columns, top_k, routed_scale, ids_host,
                    weights_host, gate_host);

        // The fused route+dispatch: identical ids and weights to the two calls.
        auto* dispatch_mem = static_cast<std::int32_t*>(nullptr);
        cuda_check(
            cudaMalloc(&dispatch_mem, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)),
            "cudaMalloc");
        auto dispatch = infernix::ops::carve_moe_dispatch(dispatch_mem, experts, top_k * columns);
        infernix::ops::moe_route_dispatch_noaux_tc(logits_t, bias_t, routed_scale, top_k, routing,
                                                   dispatch, nullptr, nullptr);
        std::vector<std::int32_t> ids2(ids_host.size());
        std::vector<float> weights2(weights_host.size());
        cuda_check(cudaMemcpy(ids2.data(), d_ids, ids2.size() * 4, cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cuda_check(
            cudaMemcpy(weights2.data(), d_weights, weights2.size() * 4, cudaMemcpyDeviceToHost),
            "cudaMemcpy");
        check(ids2 == ids_host && weights2 == weights_host,
              "moe_route_dispatch_noaux_tc equals moe_route_noaux_tc plus moe_dispatch");
        cudaFree(d_logits);
        cudaFree(d_bias);
        cudaFree(d_ids);
        cudaFree(d_weights);
        cudaFree(d_gate);
        cudaFree(dispatch_mem);
    }
}

// The fused route+plan of a noaux_tc call that stages nothing: every entry's output equals the
// separate calls' (the plan's kernel routes, dispatches, resolves the records and encodes x).
void test_noaux_plan() {
    std::mt19937 rng(23);
    std::vector<Expert> experts;
    for (int e = 0; e < 24; ++e) { experts.push_back(random_a16_expert(rng)); }
    const int experts_n = static_cast<int>(experts.size());
    const int columns = 6, top_k = 3;
    const int entries        = top_k * columns;
    const float routed_scale = 2.5F;
    // A shared expert plane of ones and weights 1/(sum) so the combine is the routed sum: any
    // routing change shows in the outputs.
    const auto x = random_activations(rng, columns, false);
    std::vector<float> logits_host(static_cast<std::size_t>(columns) * experts_n);
    std::uniform_real_distribution<float> logit(-2.0F, 3.0F);
    for (auto& v : logits_host) { v = logit(rng); }
    // Column 0: an exact tie between experts 4 and 5 (equal logits, equal bias).
    logits_host[static_cast<std::size_t>(4)] = logits_host[static_cast<std::size_t>(5)] = 3.0F;
    std::vector<float> bias_host(static_cast<std::size_t>(experts_n), 0.0F);

    const std::uint64_t stride = kGeo.record_bytes();
    std::uint8_t* host_bank    = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_bank),
                             static_cast<std::size_t>(experts_n) * stride, cudaHostAllocMapped),
               "cudaHostAlloc");
    for (int e = 0; e < experts_n; ++e) {
        std::memcpy(host_bank + static_cast<std::size_t>(e) * stride,
                    experts[static_cast<std::size_t>(e)].record.data(), stride);
    }
    std::vector<float> scales_host(static_cast<std::size_t>(experts_n) * 6);
    for (int e = 0; e < experts_n; ++e) {
        const auto& s        = experts[static_cast<std::size_t>(e)].scales;
        const float words[6] = {s.input_gate, s.input_up, s.input_down,
                                s.alpha_gate, s.alpha_up, s.alpha_down};
        for (int i = 0; i < 6; ++i) {
            scales_host[static_cast<std::size_t>(e) * 6 + static_cast<std::size_t>(i)] = words[i];
        }
    }
    std::vector<std::int32_t> frames(experts_n);
    std::uint8_t* frame_pool = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&frame_pool),
                          static_cast<std::size_t>(experts_n) * stride),
               "cudaMalloc");
    for (int e = 0; e < experts_n; ++e) {
        cuda_check(cudaMemcpy(frame_pool + static_cast<std::size_t>(e) * stride,
                              experts[static_cast<std::size_t>(e)].record.data(), stride,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        frames[static_cast<std::size_t>(e)] = e;
    }
    auto* d_frames  = device_copy(frames);
    auto* d_scales  = device_copy(scales_host);
    auto* d_x       = device_copy(x);
    auto* d_logits  = device_copy(logits_host);
    auto* d_bias    = device_copy(bias_host);
    auto* d_outputs = static_cast<std::uint16_t*>(nullptr);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_outputs),
                          static_cast<std::size_t>(entries) * kH * 2),
               "cudaMalloc");
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, infernix::ops::moe_experts_workspace_bytes(
                                          experts_n, entries, columns, moe::ExpertActivation::kA16,
                                          moe::WideA16::kExact, kGeo)),
               "cudaMalloc");
    const Tensor x_t{d_x, DType::BF16, {kH, columns}};
    Tensor outputs{d_outputs, DType::BF16, {kH, entries}};
    infernix::ops::MoeExpertSource source{.frame_base    = frame_pool,
                                          .frames        = d_frames,
                                          .host_records  = host_bank,
                                          .record_stride = stride,
                                          .scales =
                                              reinterpret_cast<const moe::ExpertScales*>(d_scales),
                                          .activation = moe::ExpertActivation::kA16,
                                          .geometry   = kGeo};
    infernix::ops::MoeRouting routing;
    infernix::ops::MoeDispatch dispatch;
    std::vector<std::int32_t> ids_host(static_cast<std::size_t>(top_k) * columns);
    std::vector<float> weights_host(static_cast<std::size_t>(top_k) * columns);
    std::vector<float> gate_host(static_cast<std::size_t>(columns));
    const auto separate = [&] {
        auto* d_ids     = static_cast<std::int32_t*>(nullptr);
        auto* d_weights = static_cast<float*>(nullptr);
        auto* d_gate    = static_cast<float*>(nullptr);
        cuda_check(cudaMalloc(&d_ids, ids_host.size() * 4), "cudaMalloc");
        cuda_check(cudaMalloc(&d_weights, weights_host.size() * 4), "cudaMalloc");
        cuda_check(cudaMalloc(&d_gate, gate_host.size() * 4), "cudaMalloc");
        routing = infernix::ops::MoeRouting{Tensor{d_ids, DType::I32, {top_k, columns}},
                                            Tensor{d_weights, DType::FP32, {top_k, columns}},
                                            Tensor{d_gate, DType::FP32, {columns, 1, 1, 1}}};
        infernix::ops::moe_route_noaux_tc(Tensor{d_logits, DType::FP32, {experts_n, columns}},
                                          Tensor{d_bias, DType::FP32, {experts_n, 1, 1, 1}},
                                          routed_scale, top_k, routing, nullptr);
        auto* dispatch_mem = static_cast<std::int32_t*>(nullptr);
        cuda_check(cudaMalloc(&dispatch_mem, infernix::ops::moe_dispatch_bytes(experts_n, entries)),
                   "cudaMalloc");
        dispatch = infernix::ops::carve_moe_dispatch(dispatch_mem, experts_n, entries);
        infernix::ops::moe_dispatch(routing, experts_n, dispatch, nullptr, nullptr);
        infernix::ops::moe_experts(x_t, dispatch, source, top_k, experts_n, workspace, outputs,
                                   nullptr, true);
        cuda_check(cudaMemcpy(ids_host.data(), d_ids, ids_host.size() * 4, cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cuda_check(cudaMemcpy(weights_host.data(), d_weights, weights_host.size() * 4,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cuda_check(
            cudaMemcpy(gate_host.data(), d_gate, gate_host.size() * 4, cudaMemcpyDeviceToHost),
            "cudaMemcpy");
        std::vector<std::uint16_t> separate_out(entries * kH);
        cuda_check(cudaMemcpy(separate_out.data(), d_outputs, separate_out.size() * 2,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cudaFree(d_ids);
        cudaFree(d_weights);
        cudaFree(d_gate);
        cudaFree(dispatch_mem);
        return separate_out;
    }();
    // The fused plan on fresh buffers.
    auto* d_outputs2 = static_cast<std::uint16_t*>(nullptr);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_outputs2),
                          static_cast<std::size_t>(entries) * kH * 2),
               "cudaMalloc");
    Tensor outputs2{d_outputs2, DType::BF16, {kH, entries}};
    infernix::ops::MoeRouting routing2;
    infernix::ops::MoeDispatch dispatch2;
    auto* d_ids2     = static_cast<std::int32_t*>(nullptr);
    auto* d_weights2 = static_cast<float*>(nullptr);
    auto* d_gate2    = static_cast<float*>(nullptr);
    cuda_check(cudaMalloc(&d_ids2, ids_host.size() * 4), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights2, weights_host.size() * 4), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate2, gate_host.size() * 4), "cudaMalloc");
    routing2 = infernix::ops::MoeRouting{Tensor{d_ids2, DType::I32, {top_k, columns}},
                                         Tensor{d_weights2, DType::FP32, {top_k, columns}},
                                         Tensor{d_gate2, DType::FP32, {columns, 1, 1, 1}}};
    auto* dispatch_mem2 = static_cast<std::int32_t*>(nullptr);
    cuda_check(cudaMalloc(&dispatch_mem2, infernix::ops::moe_dispatch_bytes(experts_n, entries)),
               "cudaMalloc");
    dispatch2       = infernix::ops::carve_moe_dispatch(dispatch_mem2, experts_n, entries);
    const auto plan = infernix::ops::moe_route_dispatch_plan_noaux_tc(
        Tensor{d_logits, DType::FP32, {experts_n, columns}},
        Tensor{d_bias, DType::FP32, {experts_n, 1, 1, 1}}, routed_scale, top_k, routing2, dispatch2,
        nullptr, x_t, source, experts_n, workspace, outputs2, nullptr);
    check(plan == infernix::ops::MoeRoutePlan::InPlace,
          "the noaux plan takes the in-place fused kernel");
    infernix::ops::moe_experts(x_t, dispatch2, source, top_k, experts_n, workspace, outputs2,
                               nullptr, true, infernix::ops::MoeRoutePlan::InPlace, true);
    std::vector<std::uint16_t> fused_out(static_cast<std::size_t>(entries) * kH);
    cuda_check(
        cudaMemcpy(fused_out.data(), d_outputs2, fused_out.size() * 2, cudaMemcpyDeviceToHost),
        "cudaMemcpy");
    check(fused_out == separate, "the noaux fused plan's outputs equal the separate calls'");
    std::vector<std::int32_t> ids2(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids2.data(), routing2.ids.data, ids2.size() * 4, cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    check(ids2 == ids_host, "the noaux fused plan's ids equal the route's (ties included)");
    cudaFree(d_frames);
    cudaFree(d_scales);
    cudaFree(d_x);
    cudaFree(d_logits);
    cudaFree(d_bias);
    cudaFree(d_outputs);
    cudaFree(d_outputs2);
    cudaFree(workspace);
    cudaFree(frame_pool);
    cudaFreeHost(host_bank);
}

} // namespace

// Every vector ISA against the scalar reference, bit for bit, at this geometry.
void test_cpu_isa_equality() {
    std::mt19937 rng(29);
    const Expert e = random_a16_expert(rng);
    for (int trial = 0; trial < 2; ++trial) {
        const auto x   = random_activations(rng, 3, trial == 1);
        const auto ref = cpu_run(moe::CpuIsa::kScalar, e, x.data(), 3);
        for (moe::CpuIsa isa :
             {moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
            if (!moe::cpu_isa_supported(isa)) { continue; }
            check(cpu_run(isa, e, x.data(), 3) == ref,
                  "every CPU ISA equals the scalar reference at the second geometry");
        }
    }
}

// Residual A4 fast path (WideA16::kA4Residual) at the second geometry (4096, 2048, clamp 10):
//   - columns = 40 (< 64): narrow jobs (<= 8 cols) stay bit-exact against CPU; wide jobs within
//     2^-6 worst / 2^-9 mean relative L2 against CPU exact (clamped SwiGLU).
//   - columns = 130 (>= 64, all-jobs in-place h terms): every job including narrow ones runs the
//     residual A4 tiles within 2^-6 worst / 2^-9 mean relative L2, a non-finite column produces
//     the canonical NaN, and resident / staged / zero-copy placements produce bit-identical output.
void test_a4_residual() {
    std::mt19937 rng(37);
    std::vector<Expert> experts;
    for (int e = 0; e < 25; ++e) { experts.push_back(random_a16_expert(rng)); }
    const int top_k = 3;
    for (int columns : {40, 130}) {
        const auto ids = craft_ids(columns, top_k);
        auto x         = random_activations(rng, columns, true);
        if (columns == 130) { x[5] = 0x7F80U; } // +Inf in column 0
        std::vector<std::uint16_t> gpu_res, gpu_staged, gpu_zc;
        run_gpu(experts, x, columns, top_k, ids, gpu_res, /*staging_slots=*/0, /*zero_copy=*/false,
                moe::WideA16::kA4Residual);
        run_gpu(experts, x, columns, top_k, ids, gpu_staged, /*staging_slots=*/4, /*zero_copy=*/true,
                moe::WideA16::kA4Residual);
        run_gpu(experts, x, columns, top_k, ids, gpu_zc, /*staging_slots=*/0, /*zero_copy=*/true,
                moe::WideA16::kA4Residual);
        check(gpu_res == gpu_staged && gpu_res == gpu_zc,
              "kA4Residual at second geometry is placement-invariant across resident, staged and zero-copy");

        std::vector<int> counts(experts.size(), 0);
        for (int id : ids) { ++counts[static_cast<std::size_t>(id)]; }
        const bool all_jobs = columns >= 64;
        double worst = 0.0, sum = 0.0;
        long measured = 0, narrow_bad = 0, nan_bad = 0;
        for (int t = 0; t < columns; ++t) {
            for (int slot = 0; slot < top_k; ++slot) {
                const int e      = ids[static_cast<std::size_t>(t) * top_k + slot];
                const auto* got  = gpu_res.data() + (static_cast<std::size_t>(t) * top_k + slot) * kH;
                if (columns == 130 && t == 0) {
                    for (int i = 0; i < kH; ++i) { nan_bad += got[i] != 0x7FC0U; }
                    continue;
                }
                const auto ref = cpu_run(moe::best_cpu_isa(), experts[static_cast<std::size_t>(e)],
                                         x.data() + static_cast<std::size_t>(t) * kH, 1);
                const bool residual = all_jobs || counts[static_cast<std::size_t>(e)] > moe::kMaxColumns;
                if (!residual) {
                    narrow_bad += !std::equal(ref.begin(), ref.end(), got);
                } else {
                    double err = 0.0, norm = 0.0;
                    for (int i = 0; i < kH; ++i) {
                        const double r = canon::bf16_to_f32(ref[static_cast<std::size_t>(i)]);
                        const double d = canon::bf16_to_f32(got[i]) - r;
                        err += d * d;
                        norm += r * r;
                    }
                    const double rel = norm > 0.0 ? std::sqrt(err / norm) : std::sqrt(err);
                    worst            = std::max(worst, rel);
                    sum += rel;
                    ++measured;
                }
            }
        }
        const double mean = measured > 0 ? sum / static_cast<double>(measured) : 0.0;
        std::printf("kA4Residual columns=%d (%s): mean rel L2 %.5f, worst %.5f (%ld entries), narrow_bad=%ld, nan_bad=%ld\n",
                    columns, all_jobs ? "all-jobs in-place" : "mixed", mean, worst, measured, narrow_bad, nan_bad);
        check(narrow_bad == 0, "kA4Residual narrow entries equal CPU bit for bit");
        check(nan_bad == 0, "kA4Residual non-finite column produces canonical NaN");
        check(worst <= std::ldexp(1.0, -6), "kA4Residual every residual entry within 2^-6 relative L2 of exact");
        check(mean <= std::ldexp(1.0, -9), "kA4Residual mean relative L2 within 2^-9 of exact");
    }
}

int main() {
    std::mt19937 rng(3);
    for (int trial = 0; trial < 2; ++trial) {
        const Expert e = random_a16_expert(rng);
        const auto x   = random_activations(rng, 2, trial == 1);
        check_fp64(e, x, cpu_run(moe::best_cpu_isa(), e, x.data(), 2), 2);
    }
    test_cpu_isa_equality();
    test_layer_routes();
    test_a4_residual();
    test_zero_copy();
    test_staged();
    test_wide_call_narrow_jobs();
    test_cpu_service();
    test_noaux_routing();
    test_noaux_plan();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe geometry checks passed\n");
    return 0;
}
