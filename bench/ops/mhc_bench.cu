// Benchmark of mhc_update_mix, the decode path's fused residual update and next-block mixer of the
// manifold-constrained hyper-connection residual, at its real geometry: S = 4 streams of H = 4096,
// 24 fn rows, 20 Sinkhorn iterations. One call moves about 885 KiB at T = 1, of which 768 KiB are
// the fn weights. Each point times one CUDA graph of --layers back-to-back calls at T columns, each
// with its own fn (layers x 768 KiB far exceeds the L2, so every call reads its fn from DRAM, as
// decode does; the graph, as decode's, keeps host launch time out), and reports the device time per
// call against the bytes one call must move. The FNV-1a hash of the last call's outputs identifies
// the result bits.
//
//   infernix_mhc_bench [--columns 1,4,8] [--layers 176] [--warmup 5] [--repeat 50] [--seed N]

#include "infernix/ops/mhc.h"

#include "core/arena.h"
#include "core/device.h"
#include "infernix_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

constexpr int kStreams = 4;
constexpr int kHidden  = 4096;
constexpr int kRows    = (2 + kStreams) * kStreams;
constexpr int kWidth   = kStreams * kHidden;

struct Options {
    std::vector<int> columns{1, 4, 8};
    int layers         = 176;
    int warmup         = 5;
    int repeat         = 50;
    std::uint32_t seed = 20261123U;
};

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        out.push_back(std::stoi(
            text.substr(begin, end == std::string::npos ? std::string::npos : end - begin)));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value            = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--columns") {
            options.columns = parse_list(value());
        } else if (arg == "--layers") {
            options.layers = std::stoi(value());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    if (options.layers < 1) { throw std::invalid_argument("--layers must be positive"); }
    return options;
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint8_t b : bytes) { h = (h ^ b) * 1099511628211ULL; }
    return h;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        int max_columns = 1;
        for (const int c : options.columns) { max_columns = std::max(max_columns, c); }
        // fn: kRows x kWidth BF16 per layer; 176 layers x 768 KiB = 132 MiB, beyond the L2.
        std::vector<DeviceBuffer> fns, bases, scales;
        for (int l = 0; l < options.layers; ++l) {
            fns.push_back(bench::make_bf16(static_cast<std::size_t>(kRows) * kWidth,
                                           options.seed + l, -0.06F, 0.06F));
            bases.push_back(bench::make_f32(kRows, options.seed + 1000 + l, -1.0F, 1.0F));
            scales.push_back(bench::make_f32(3, options.seed + 2000 + l, 0.5F, 1.5F));
        }
        WorkspaceArena workspace(std::max<std::size_t>(
            ops::mhc_mix_workspace_capacity_bytes(kStreams, kHidden, 1, max_columns), 1));
        std::printf(
            "# gpu=%s timed=mhc_update_mix (graph of %d calls, distinct fn) S=%d H=%d rows=%d "
            "fn=%.2f MiB per call\n",
            context.props.name, options.layers, kStreams, kHidden, kRows,
            static_cast<double>(fns.front().bytes) / 1048576.0);
        for (const int columns : options.columns) {
            const DeviceBuffer residual =
                bench::make_bf16(static_cast<std::size_t>(kWidth) * columns,
                                 options.seed + 10 + columns, -100.0F, 100.0F);
            const DeviceBuffer y = bench::make_bf16(static_cast<std::size_t>(kHidden) * columns,
                                                    options.seed + 20 + columns, -2.0F, 2.0F);
            DeviceBuffer post(sizeof(float) * kStreams * columns);
            DeviceBuffer comb(sizeof(float) * kStreams * kStreams * columns);
            DeviceBuffer x(sizeof(std::uint16_t) * kHidden * columns);
            CUDA_CHECK(bench::fixture::fill_values(static_cast<float*>(post.p), kStreams * columns,
                                                   options.seed + 30, 0.5F, 1.5F));
            CUDA_CHECK(bench::fixture::fill_values(static_cast<float*>(comb.p),
                                                   kStreams * kStreams * columns, options.seed + 40,
                                                   0.1F, 0.4F));
            CUDA_CHECK(cudaDeviceSynchronize());
            const Tensor ty(y.p, DType::BF16, {kHidden, columns});
            Tensor tr(residual.p, DType::BF16, {kWidth, columns});
            Tensor tp(post.p, DType::FP32, {kStreams, columns});
            Tensor tc(comb.p, DType::FP32, {kStreams, kStreams, columns});
            Tensor tx(x.p, DType::BF16, {kHidden, columns});
            bench::TimedGraph graph;
            graph.capture(context.stream, [&](cudaStream_t s) {
                for (int l = 0; l < options.layers; ++l) {
                    auto scope = workspace.scope();
                    const Tensor tf(fns[static_cast<std::size_t>(l)].p, DType::BF16,
                                    {kWidth, kRows});
                    const Tensor tb(bases[static_cast<std::size_t>(l)].p, DType::FP32, {kRows});
                    const Tensor ts(scales[static_cast<std::size_t>(l)].p, DType::FP32, {3});
                    ops::mhc_update_mix(ty, tp, tc, tr, tr, tf, tb, ts, kStreams, 20, 1e-5F, 1e-6F,
                                        tx, &tp, &tc, workspace, s);
                }
            });
            const auto timing =
                bench::measure_graph(graph, context.stream, options.warmup, options.repeat);
            const double per = 1.0 / options.layers;
            // Bytes one call must move: fn once; the residual read by the update, written back, and
            // read again by the mixer's x step; y read; x written; post/comb in and out; the
            // split-K partials written and read (64 features per chunk, the mixer's split-K width).
            const double chunks = (kHidden + 63) / 64;
            const double bytes  = static_cast<double>(fns.front().bytes) +
                                  3.0 * kWidth * columns * 2 + 2.0 * kHidden * columns * 2 +
                                  2.0 * (kStreams + kStreams * kStreams) * columns * 4 +
                                  2.0 * (kRows + 1) * columns * chunks * 4;
            std::vector<std::uint8_t> out_bytes(x.bytes + residual.bytes + post.bytes + comb.bytes);
            CUDA_CHECK(cudaMemcpy(out_bytes.data(), x.p, x.bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(out_bytes.data() + x.bytes, residual.p, residual.bytes,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(out_bytes.data() + x.bytes + residual.bytes, post.p, post.bytes,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(out_bytes.data() + x.bytes + residual.bytes + post.bytes, comb.p,
                                  comb.bytes, cudaMemcpyDeviceToHost));
            std::printf(
                "T=%d | per call median %7.2f us (min %7.2f, p95 %7.2f) | %6.0f GB/s of %7.1f KiB "
                "| output fnv1a %016llx\n",
                columns, timing.median_us * per, timing.min_us * per, timing.p95_us * per,
                bytes / (timing.median_us * per * 1e-6) / 1e9, bytes / 1024.0,
                static_cast<unsigned long long>(fnv1a(out_bytes)));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_mhc_bench: %s\n", error.what());
        return 1;
    }
}
