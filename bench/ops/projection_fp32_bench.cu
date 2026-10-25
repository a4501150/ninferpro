// infernix_projection_fp32_bench — the FP32-output BF16 projection (ops::projection_fp32) at
// Qwen3.8-Flash-Next's router shape: the [512, 2560] router and the [1, 2560] shared-expert score as
// two row segments of one call. Per width it times the production route (cold L2, graph) and checks
// that its bits equal the narrow mapping's, which every mapping of the op must reproduce.
#include "core/tensor.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix_bench_common.h"
#include "ops/projection_fp32/projection_fp32_route.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;

namespace {
std::uint64_t fnv1a(const DeviceBuffer& buffer, std::size_t bytes) {
    std::vector<unsigned char> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), buffer.p, bytes, cudaMemcpyDeviceToHost));
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : host) { hash = (hash ^ byte) * 1099511628211ULL; }
    return hash;
}

std::vector<int> parse_list(std::string_view raw) {
    std::vector<int> values;
    while (!raw.empty()) {
        const std::size_t comma = raw.find(',');
        values.push_back(std::stoi(std::string(raw.substr(0, comma))));
        raw = comma == std::string_view::npos ? std::string_view{} : raw.substr(comma + 1);
    }
    return values;
}
} // namespace

int main(int argc, char** argv) {
    std::vector<int> tokens{1, 4, 8, 12, 32, 48, 64, 65, 128, 512, 4096, 16384};
    int warmup = 5, repeat = 20, k = 2560, router_rows = 512, score_rows = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        const auto next          = [&]() -> std::string_view {
            if (++i == argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[i - 1]);
                std::exit(2);
            }
            return argv[i];
        };
        if (a == "--tokens") {
            tokens = parse_list(next());
        } else if (a == "--warmup") {
            warmup = std::stoi(std::string(next()));
        } else if (a == "--repeat") {
            repeat = std::stoi(std::string(next()));
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: infernix_projection_fp32_bench [--tokens T,...] [--warmup N] [--repeat N]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    int max_t = 1;
    for (const int t : tokens) { max_t = std::max(max_t, t); }
    const int rows = router_rows + score_rows;
    DeviceBuffer router = bench::make_bf16(static_cast<std::size_t>(router_rows) * k, 11, -0.05F, 0.05F);
    DeviceBuffer score  = bench::make_bf16(static_cast<std::size_t>(score_rows) * k, 13, -0.05F, 0.05F);
    DeviceBuffer x      = bench::make_bf16(static_cast<std::size_t>(k) * max_t, 17, -2.0F, 2.0F);
    DeviceBuffer out(static_cast<std::size_t>(rows) * max_t * sizeof(float));
    DeviceBuffer reference(static_cast<std::size_t>(rows) * max_t * sizeof(float));
    bench::L2FlushBuffer flush(256ULL << 20);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const Tensor router_tensor(router.p, DType::BF16, {k, router_rows});
    const Tensor score_tensor(score.p, DType::BF16, {k, score_rows});
    const Tensor* segments[] = {&router_tensor, &score_tensor};

    std::printf("# router [%d + %d, %d] on projection_fp32, cold L2, graph\n", router_rows, score_rows, k);
    std::printf("%8s %12s %10s %18s %s\n", "T", "median_us", "TFLOP/s", "hash", "bits vs narrow");
    for (const int t : tokens) {
        const Tensor xt(x.p, DType::BF16, {k, t});
        Tensor ot(out.p, DType::FP32, {rows, t});
        Tensor rt(reference.p, DType::FP32, {rows, t});
        const std::size_t bytes = static_cast<std::size_t>(rows) * t * sizeof(float);
        ops::detail::projection_fp32_route(xt, segments, rt, ops::detail::ProjectionRoute::Narrow, stream);
        ops::detail::projection_fp32_route(xt, segments, ot, ops::detail::ProjectionRoute::Automatic, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::uint64_t hash = fnv1a(out, bytes);
        const bool equal         = hash == fnv1a(reference, bytes);
        bench::TimedGraph graph;
        graph.capture(stream, [&](cudaStream_t s) {
            ops::detail::projection_fp32_route(xt, segments, ot, ops::detail::ProjectionRoute::Automatic, s);
        });
        const bench::ColdTiming timing = bench::measure_cold_graph(graph, flush, stream, warmup, repeat);
        const double tflops            = 2.0 * rows * static_cast<double>(k) * t / (timing.median_us * 1e6);
        std::printf("%8d %12.2f %10.2f %18llx %s\n", t, timing.median_us, tflops,
                    static_cast<unsigned long long>(hash), equal ? "equal" : "DIFFER");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
