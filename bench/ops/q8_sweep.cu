// Candidate sweep for one Q8 (q8_g32_fp16) linear problem [N,K]: times the existing Q8 kernel
// templates at every T of the list with cold-L2 CUDA graphs and proposes a selector body for a
// family's shape file (src/models/<family>/shapes/q8/n<N>_k<K>.cu).
//
// Decode widths (T <= 8): a request's output may not depend on how many requests share the call
// (the generic route's SIMT pair guarantees it), so a ladder takes its T <= 8 routes from one bit
// class: candidates whose column bits are the same at every width they serve. The sweep reports each
// class's best time per width against the DRAM floor and keeps the fastest class. Wider calls may
// take any screened candidate. Every candidate is first screened against the generic route at
// T = 1 (GEMV only), 4, 77 and 1000; one that fails is never selected.
//
// Usage:
//   infernix_q8_sweep --n N --k K [--t-list 1,2,...] [--max-t T] [--warmup N] [--repeat N]
//       [--tolerance-pct P] [--bulk-tolerance-pct P] [--dram-gbps G] [--csv PATH]
// The hot interval (T <= 128) merges routes within --tolerance-pct (2); wider calls within
// --bulk-tolerance-pct (5), so close tiles do not alternate across the sparse bulk anchors.
#include "core/device.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "infernix_bench_common.h"
#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::ops::detail;

namespace {

using Body = std::function<void(const Tensor&, const Weight&, Tensor&, cudaStream_t)>;

struct Candidate {
    std::string name;
    std::string code; // the selector's return expression
    Body launch;
    int max_t = INT_MAX; // GEMV: 1
};

template <class S>
void gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<S>(q8_linear_operands(x, w),
                          LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                          LinearIdentityEpilogue{}, stream);
}
template <class S>
void sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<S>(q8_linear_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                  LinearIdentityEpilogue{}, stream);
}

// The register-streamed route needs its K at compile time: one instance per K this sweep serves.
// Shared-x schedules exist only where the x tile fits shared memory.
template <int K, int RowsPerWarp, int KSplit, int Warps, int MaxCols, int ColsPerPass, Q8StreamX X>
constexpr bool stream_valid() {
    if (K % (64 * KSplit) != 0 || Warps % KSplit != 0) return false;
    const int x_bytes   = X == Q8StreamX::Shared ? ColsPerPass * K * 2 : 0;
    const int partial   = KSplit > 1 ? (Warps / KSplit) * KSplit * RowsPerWarp * ColsPerPass * 4 : 0;
    return x_bytes + partial <= 99 * 1024;
}

template <int K, int RowsPerWarp, int KSplit, int Warps, int MaxCols, int ColsPerPass, Q8StreamX X>
void add_stream(std::vector<Candidate>& list, int k) {
    if constexpr (stream_valid<K, RowsPerWarp, KSplit, Warps, MaxCols, ColsPerPass, X>()) {
        if (k != K) return;
        using S = Q8A16StreamSchedule<K, RowsPerWarp, KSplit, Warps, MaxCols, ColsPerPass, X, 2>;
        char name[96], code[192];
        std::snprintf(name, sizeof(name), "stream_r%dks%dw%d_c%dp%d_%s", RowsPerWarp, KSplit, Warps,
                      MaxCols, ColsPerPass, X == Q8StreamX::Shared ? "sx" : "gx");
        std::snprintf(code, sizeof(code),
                      "launch_q8_a16_stream_linear<Q8A16StreamSchedule<%d, %d, %d, %d, %d, %d, "
                      "Q8StreamX::%s, 2>>",
                      K, RowsPerWarp, KSplit, Warps, MaxCols, ColsPerPass,
                      X == Q8StreamX::Shared ? "Shared" : "Global");
        list.push_back({name, code, launch_q8_a16_stream_linear<S>});
    }
}

template <int K>
void add_streams(std::vector<Candidate>& list, int k) {
    add_stream<K, 2, 1, 8, 4, 4, Q8StreamX::Shared>(list, k);
    add_stream<K, 2, 1, 8, 8, 8, Q8StreamX::Shared>(list, k);
    add_stream<K, 2, 4, 8, 4, 4, Q8StreamX::Global>(list, k);
    add_stream<K, 2, 4, 8, 8, 8, Q8StreamX::Global>(list, k);
    add_stream<K, 2, 8, 8, 4, 4, Q8StreamX::Global>(list, k);
    add_stream<K, 2, 8, 8, 8, 8, Q8StreamX::Global>(list, k);
}

#define INFERNIX_FN(name) Candidate{#name, #name, name}
#define INFERNIX_GEMV(R, W, M)                                                                     \
    Candidate{"gemv_r" #R "w" #W "m" #M, "gemv<Q8A16GemvSchedule<" #R ", " #W ", " #M ">>",         \
              gemv<Q8A16GemvSchedule<R, W, M>>, 1}
#define INFERNIX_SLICED(BT, KW, S)                                                                 \
    Candidate{"sliced_t" #BT "kw" #KW "s" #S,                                                       \
              "sliced<Q8A16SlicedKMmaSchedule<" #BT ", " #KW ", " #S ", 1, Q8ScaleAccess::Shared, "  \
              "Cache::ca, Cache::cg, Q8ActivationStage::PaddedZero>>",                             \
              sliced<Q8A16SlicedKMmaSchedule<BT, KW, S, 1, Q8ScaleAccess::Shared, ops::Cache::ca,   \
                                             ops::Cache::cg, Q8ActivationStage::PaddedZero>>}

std::vector<Candidate> candidates(int k) {
    std::vector<Candidate> list{
        INFERNIX_GEMV(4, 1, 2), INFERNIX_GEMV(4, 2, 2), INFERNIX_GEMV(8, 1, 2), INFERNIX_GEMV(2, 2, 2),
        INFERNIX_GEMV(4, 4, 1),
        INFERNIX_FN(launch_q8_a16_simt_r8_t4), INFERNIX_FN(launch_q8_a16_simt_r8_t8),
        INFERNIX_FN(launch_q8_a16_simt_r1_t4_w8), INFERNIX_FN(launch_q8_a16_simt_r1_t8_w8),
        INFERNIX_FN(launch_q8_a16_simt_r4_t4_w2_g16_s2),
        INFERNIX_FN(launch_q8_a16_sliced_r16_t16_w8_s2), INFERNIX_FN(launch_q8_a16_sliced_r16_t16_w4_s2),
        INFERNIX_FN(launch_q8_a16_sliced_r16_t32_w4_s2),
        INFERNIX_SLICED(8, 8, 1), INFERNIX_SLICED(16, 8, 1), INFERNIX_SLICED(16, 4, 2),
        INFERNIX_SLICED(24, 4, 2), INFERNIX_SLICED(32, 4, 2), INFERNIX_SLICED(32, 8, 1),
        INFERNIX_SLICED(40, 4, 2), INFERNIX_SLICED(48, 4, 2), INFERNIX_SLICED(64, 4, 1),
        INFERNIX_FN(launch_q8_a16_mma_r64_t8_k128), INFERNIX_FN(launch_q8_a16_mma_r32_t64),
        INFERNIX_FN(launch_q8_a16_mma_r32_t96), INFERNIX_FN(launch_q8_a16_mma_r32_t128),
        INFERNIX_FN(launch_q8_a16_mma_r48_t64), INFERNIX_FN(launch_q8_a16_mma_r64_t96),
        INFERNIX_FN(launch_q8_a16_mma_r64_t128), INFERNIX_FN(launch_q8_a16_mma_r96_t96),
        INFERNIX_FN(launch_q8_a16_mma_r128_t64), INFERNIX_FN(launch_q8_a16_mma_r128_t80),
        INFERNIX_FN(launch_q8_a16_mma_r64x16_t48_k128_a1),
        INFERNIX_FN(launch_q8_a16_mma_r64x32_t64_k128_a1),
    };
    add_streams<1536>(list, k);
    add_streams<2048>(list, k);
    add_streams<4096>(list, k);
    add_streams<8192>(list, k);
    add_streams<12288>(list, k);
    add_streams<16384>(list, k);
    return list;
}

struct Options {
    int n = 0, k = 0, max_t = 4096;
    std::vector<int> ts;
    int warmup = 2, repeat = 21;
    double tolerance = 0.02, bulk_tolerance = 0.05, dram_gbps = 1792.0;
    std::string csv;
};

std::vector<int> parse_list(const std::string& value) {
    std::vector<int> result;
    std::size_t start = 0;
    while (start < value.size()) {
        const std::size_t end = value.find(',', start);
        result.push_back(std::atoi(value.substr(start, end == std::string::npos ? end : end - start).c_str()));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.empty() || result.front() < 1) throw std::invalid_argument("--t-list needs positive Ts");
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--n") o.n = std::atoi(next().c_str());
        else if (arg == "--k") o.k = std::atoi(next().c_str());
        else if (arg == "--max-t") o.max_t = std::atoi(next().c_str());
        else if (arg == "--t-list") o.ts = parse_list(next());
        else if (arg == "--warmup") o.warmup = std::atoi(next().c_str());
        else if (arg == "--repeat") o.repeat = std::atoi(next().c_str());
        else if (arg == "--tolerance-pct") o.tolerance = std::atof(next().c_str()) / 100.0;
        else if (arg == "--bulk-tolerance-pct") o.bulk_tolerance = std::atof(next().c_str()) / 100.0;
        else if (arg == "--csv") o.csv = next();
        else if (arg == "--dram-gbps") o.dram_gbps = std::atof(next().c_str());
        else throw std::invalid_argument("unknown argument " + arg);
    }
    if (o.n <= 0 || o.k <= 0 || o.k % 128 != 0) throw std::invalid_argument("--n and --k (a multiple of 128) are required");
    if (o.ts.empty()) {
        // linear-tuning.md: every T of the hot interval, then the bulk anchors and prefill chunks.
        for (int t = 1; t <= 128; ++t) o.ts.push_back(t);
        for (const int t : {160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096})
            if (t <= o.max_t) o.ts.push_back(t);
    }
    return o;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<std::uint16_t> download(const DeviceBuffer& buffer, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    CUDA_CHECK(cudaMemcpy(host.data(), buffer.p, elements * 2, cudaMemcpyDeviceToHost));
    return host;
}

bool close_to(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref) {
    double error = 0, norm = 0, worst = 0, largest = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = bf16_value(got[i]), r = bf16_value(ref[i]);
        if (!std::isfinite(a)) return false;
        error += (a - r) * (a - r);
        norm += r * r;
        worst   = std::max(worst, std::fabs(a - r));
        largest = std::max(largest, std::fabs(r));
    }
    const double l2 = norm > 0 ? std::sqrt(error / norm) : std::sqrt(error);
    return l2 <= 1.0 / 128.0 && (largest > 0 ? worst / largest : worst) <= 1.0 / 64.0;
}

struct Problem {
    int n, k;
    const Weight* weight;
    DeviceBuffer* x;
    DeviceBuffer* out;
    cudaStream_t stream;
    std::vector<std::uint16_t> run(const Body& body, int t) const {
        Tensor xt(x->p, DType::BF16, {k, t});
        Tensor ot(out->p, DType::BF16, {n, t});
        body(xt, *weight, ot, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return download(*out, static_cast<std::size_t>(n) * t);
    }
};

// The event timer ticks in about 2 us steps here, coarse against 10-20 us decode kernels: a median
// keeps that step, the mean of many cold samples does not (the slowest fifth is dropped as outliers).
double cold_mean_us(const bench::TimedGraph& graph, bench::L2FlushBuffer& flush, cudaStream_t stream,
                    int warmup, int repeat) {
    for (int i = 0; i < warmup; ++i) {
        bench::flush_l2(flush, stream);
        graph.launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> samples;
    for (int i = 0; i < repeat; ++i) {
        bench::flush_l2(flush, stream);
        samples.push_back(graph.launch_timed(stream));
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t kept = std::max<std::size_t>(1, samples.size() - samples.size() / 5);
    double sum = 0;
    for (std::size_t i = 0; i < kept; ++i) sum += samples[i];
    return sum / static_cast<double>(kept);
}

Body generic(int n, int k) {
    return [n, k](const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
        select_q8_launch(n, k, static_cast<std::int32_t>(x.ne[1]), ops::LinearPolicy::A16Only,
                         ops::LinearRoutes::Exact)(x, w, out, s);
    };
}

struct Span {
    int last_t;
    std::size_t candidate;
};

// us[i][c]: candidate c's time at ts[i] (NaN when not admitted). Greedy intervals, merged while one
// candidate stays within tolerance of the fastest at every T of both neighbours.
std::vector<Span> intervals(const std::vector<int>& ts, const std::vector<std::vector<double>>& us,
                            double tolerance, double bulk_tolerance) {
    const auto best_at = [&](std::size_t i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < us[i].size(); ++c)
            if (!std::isnan(us[i][c]) && (best == SIZE_MAX || us[i][c] < us[i][best])) best = c;
        if (best == SIZE_MAX) throw std::runtime_error("no admitted candidate at T=" + std::to_string(ts[i]));
        return best;
    };
    const auto within = [&](std::size_t c, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i <= last; ++i)
            if (std::isnan(us[i][c]) ||
                us[i][c] > (1.0 + (ts[i] <= 128 ? tolerance : bulk_tolerance)) * us[i][best_at(i)])
                return false;
        return true;
    };
    struct Run {
        std::size_t first, last, candidate;
    };
    std::vector<Run> runs;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        if (!runs.empty() && within(runs.back().candidate, i, i)) runs.back().last = i;
        else runs.push_back({i, i, best_at(i)});
    }
    for (bool merged = true; merged;) {
        merged = false;
        for (std::size_t r = 0; r + 1 < runs.size() && !merged; ++r) {
            for (const std::size_t c : {runs[r + 1].candidate, runs[r].candidate}) {
                if (within(c, runs[r].first, runs[r + 1].last)) {
                    runs[r] = {runs[r].first, runs[r + 1].last, c};
                    runs.erase(runs.begin() + static_cast<std::ptrdiff_t>(r) + 1);
                    merged = true;
                    break;
                }
            }
        }
    }
    std::vector<Span> spans;
    for (const Run& run : runs) spans.push_back({ts[run.last], run.candidate});
    return spans;
}

void sweep(const Options& o) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    bench::L2FlushBuffer flush(256ULL << 20);
    const int max_t = o.ts.back();
    auto packed     = bench::make_row_split_weight(QType::Q8_G32_FP16, o.n, o.k, o.k);
    DeviceBuffer x  = bench::make_bf16(static_cast<std::size_t>(o.k) * std::max(max_t, 1000), 101U);
    DeviceBuffer out(static_cast<std::size_t>(o.n) * std::max(max_t, 1000) * 2);
    const Problem p{o.n, o.k, &packed.weight, &x, &out, stream};
    const std::vector<Candidate> list = candidates(o.k);

    // Screen against the generic route.
    std::vector<bool> admitted(list.size(), true);
    std::map<int, std::vector<std::uint16_t>> reference;
    for (const int t : {1, 4, 77, 1000}) reference[t] = p.run(generic(o.n, o.k), t);
    for (std::size_t c = 0; c < list.size(); ++c) {
        try {
            for (const int t : {1, 4, 77, 1000}) {
                if (t > list[c].max_t) continue;
                if (!close_to(p.run(list[c].launch, t), reference[t])) {
                    admitted[c] = false;
                    std::printf("screen: %s differs from the generic route at T=%d\n", list[c].name.c_str(), t);
                    break;
                }
            }
        } catch (const std::exception& error) {
            admitted[c] = false;
            std::printf("screen: %s refused (%s)\n", list[c].name.c_str(), error.what());
        }
    }

    // Bit classes at T <= 8: column 0's bits must be one value at every width a candidate serves.
    std::vector<std::string> klass(list.size());
    for (std::size_t c = 0; c < list.size(); ++c) {
        if (!admitted[c]) continue;
        std::string key;
        for (int t = 1; t <= std::min(8, list[c].max_t); ++t) {
            const auto bits = p.run(list[c].launch, t);
            const std::string column(reinterpret_cast<const char*>(bits.data()), static_cast<std::size_t>(o.n) * 2);
            if (key.empty()) key = column;
            else if (key != column) {
                key.clear();
                break;
            }
        }
        klass[c] = key; // empty: not column-invariant up to 8
    }

    const double weight_bytes = static_cast<double>(o.n) * o.k * (1.0 + 2.0 / 32.0);
    const double floor_us     = weight_bytes / (o.dram_gbps * 1e3);
    std::vector<std::vector<double>> us(o.ts.size(), std::vector<double>(list.size(), NAN));
    for (std::size_t i = 0; i < o.ts.size(); ++i) {
        const int t = o.ts[i];
        for (std::size_t c = 0; c < list.size(); ++c) {
            if (!admitted[c] || t > list[c].max_t || (t <= 8 && klass[c].empty())) continue;
            Tensor xt(x.p, DType::BF16, {o.k, t});
            Tensor ot(out.p, DType::BF16, {o.n, t});
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t s) { list[c].launch(xt, packed.weight, ot, s); });
            us[i][c] = cold_mean_us(graph, flush, stream, o.warmup, o.repeat);
        }
    }

    // The fastest decode class: the smallest sum over T = 1..8 of its best member's time.
    std::map<std::string, double> class_cost;
    std::map<std::string, int> class_id;
    for (std::size_t c = 0; c < list.size(); ++c)
        if (!klass[c].empty() && !class_id.contains(klass[c])) class_id.emplace(klass[c], static_cast<int>(class_id.size()));
    std::printf("\n## Q8 [%d,%d] decode widths, floor %.2f us (weight bytes at %.0f GB/s)\n", o.n, o.k, floor_us,
                o.dram_gbps);
    std::printf("%-40s %5s", "candidate", "class");
    for (int t = 1; t <= 8; ++t) std::printf(" %7s", ("T=" + std::to_string(t)).c_str());
    std::printf("\n");
    for (std::size_t c = 0; c < list.size(); ++c) {
        if (klass[c].empty()) continue;
        std::printf("%-40s %5d", list[c].name.c_str(), class_id[klass[c]]);
        for (int t = 1; t <= 8; ++t) {
            const auto it = std::find(o.ts.begin(), o.ts.end(), t);
            const double v = it == o.ts.end() ? NAN : us[it - o.ts.begin()][c];
            if (std::isnan(v)) std::printf(" %7s", "-");
            else std::printf(" %7.1f", v);
        }
        std::printf("\n");
    }
    std::string best_class;
    for (const auto& [key, id] : class_id) {
        double cost = 0;
        bool complete = true;
        for (int t = 1; t <= 8; ++t) {
            const auto it = std::find(o.ts.begin(), o.ts.end(), t);
            if (it == o.ts.end()) continue;
            double best = NAN;
            for (std::size_t c = 0; c < list.size(); ++c) {
                const double v = us[it - o.ts.begin()][c];
                if (klass[c] == key && !std::isnan(v) && (std::isnan(best) || v < best)) best = v;
            }
            if (std::isnan(best)) complete = false;
            else cost += best;
        }
        if (complete && (best_class.empty() || cost < class_cost[best_class])) best_class = key;
        class_cost[key] = complete ? cost : INFINITY;
        std::printf("class %d: T=1..8 sum %.1f us%s\n", id, cost, complete ? "" : " (incomplete)");
    }
    if (best_class.empty()) throw std::runtime_error("no column-invariant class covers T = 1..8");
    for (std::size_t i = 0; i < o.ts.size(); ++i)
        if (o.ts[i] <= 8)
            for (std::size_t c = 0; c < list.size(); ++c)
                if (klass[c] != best_class) us[i][c] = NAN;

    std::printf("\n## Q8 [%d,%d] best per T (decode class %d)\n%6s %10s %8s  %s\n", o.n, o.k, class_id[best_class],
                "T", "best_us", "floor%", "candidate");
    for (std::size_t i = 0; i < o.ts.size(); ++i) {
        std::size_t best = SIZE_MAX;
        for (std::size_t c = 0; c < list.size(); ++c)
            if (!std::isnan(us[i][c]) && (best == SIZE_MAX || us[i][c] < us[i][best])) best = c;
        if (best == SIZE_MAX) continue;
        std::printf("%6d %10.2f %7.0f%%  %s\n", o.ts[i], us[i][best], 100.0 * floor_us / us[i][best],
                    list[best].name.c_str());
    }

    if (!o.csv.empty()) {
        std::FILE* f = std::fopen(o.csv.c_str(), "w");
        if (f == nullptr) throw std::runtime_error("cannot open " + o.csv);
        std::fprintf(f, "T,candidate,us\n");
        for (std::size_t i = 0; i < o.ts.size(); ++i)
            for (std::size_t c = 0; c < list.size(); ++c)
                if (!std::isnan(us[i][c])) std::fprintf(f, "%d,%s,%.3f\n", o.ts[i], list[c].name.c_str(), us[i][c]);
        std::fclose(f);
    }
    const auto spans = intervals(o.ts, us, o.tolerance, o.bulk_tolerance);
    std::printf("\n// Proposed selector (tolerance %.1f %% to T = 128, %.1f %% beyond):\n", o.tolerance * 100.0,
                o.bulk_tolerance * 100.0);
    std::printf("Q8Launch select_q8_n%d_k%d(std::int32_t tokens) {\n", o.n, o.k);
    for (std::size_t s = 0; s < spans.size(); ++s) {
        if (s + 1 == spans.size()) std::printf("    return %s;\n", list[spans[s].candidate].code.c_str());
        else std::printf("    if (tokens <= %d) return %s;\n", spans[s].last_t, list[spans[s].candidate].code.c_str());
    }
    std::printf("}\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    try {
        sweep(parse(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q8_sweep: %s\n", error.what());
        return 1;
    }
}
