// Public-Op benchmark for one sparse-MLA decode layer call (geometry 64 heads of 256 over a
// 512-wide shared latent; indexer 32 heads of 128, pool 4, top 512 pools). One case is one
// decode round of `batch` lanes (width 1, every lane at the context's last position): indexer
// update, indexer select, absorbed query projection and sparse attention, timed as one captured
// graph after an L2 flush (a decode round re-reads the layer's weights and the whole selected
// cache from DRAM at these sizes). The printed bytes are the unavoidable traffic each stage
// moves (weights once per call, per-column cache and workspace streams), so the rightmost column
// is the round's achieved bandwidth against its speed of light.

#include "core/device.h"
#include "infernix/ops/dsa_indexer.h"
#include "infernix/ops/mla_sparse_attention.h"
#include "infernix_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;

namespace {

constexpr int kLatent = 512, kHeadDim = 256, kHeads = 64; // latent attention
constexpr int kIndexDim = 128, kIndexHeads = 32;          // indexer
constexpr int kPool = 4, kTopPools = 512, kPage = 64;
constexpr float kScale       = 1.0F / 16.0F;         // 1/sqrt(256)
constexpr float kIndexScale  = 0.08838834764831845F; // 1/sqrt(128)
constexpr std::size_t kFlush = std::size_t{256} << 20;

const ops::MlaAttentionGeometry kAttentionGeometry{
    .heads = kHeads, .head_dim = kHeadDim, .latent_dim = kLatent};
const ops::DsaIndexerGeometry kIndexerGeometry{.heads     = kIndexHeads,
                                               .head_dim  = kIndexDim,
                                               .pool      = kPool,
                                               .top_pools = kTopPools,
                                               .eps       = 1.0e-6F};

struct Options {
    std::vector<int> contexts{4096, 32768, 131072};
    std::vector<int> batches{1, 4};
    int width          = 1;
    int warmup         = 5;
    int repeat         = 50;
    bool tensor_cores  = false;
    bool int8_kv       = false;
    bool attention_only = false;
};

std::vector<int> parse_ints(std::string_view s) {
    std::vector<int> out;
    while (!s.empty()) {
        const auto comma = s.find(',');
        out.push_back(std::atoi(std::string(s.substr(0, comma)).c_str()));
        s = comma == std::string_view::npos ? std::string_view{} : s.substr(comma + 1);
    }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + std::string(arg));
            }
            return argv[++i];
        };
        if (arg == "--contexts") {
            o.contexts = parse_ints(value());
        } else if (arg == "--batches") {
            o.batches = parse_ints(value());
        } else if (arg == "--width") {
            o.width = std::atoi(argv[++i]);
        } else if (arg == "--tensor-cores") {
            o.tensor_cores = true;
        } else if (arg == "--int8") {
            o.int8_kv = true;
        } else if (arg == "--attention-only") {
            o.attention_only = true;
        } else if (arg == "--warmup") {
            o.warmup = std::atoi(argv[++i]);
        } else if (arg == "--repeat") {
            o.repeat = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::printf("infernix_mla_sparse_bench [--contexts 4096,32768,131072] [--batches 1,4] "
                        "[--width N] [--tensor-cores] [--int8] [--attention-only] "
                        "[--warmup N] [--repeat N]\n");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + std::string(arg));
        }
    }
    return o;
}

struct Case {
    int batch, width, context, pages_per_row, pages;
    bool tensor_cores, int8_kv;
    std::size_t workspace_bytes = 0, index_workspace_bytes = 0;
    DeviceBuffer k, z, pooled, latents, latent_scales, ape, gamma, beta, raw_k, index_z, index_q,
        index_w, pools, counts;
    DeviceBuffer q, w_uk, w_uv, qa, out, tables, positions, rows, attn_workspace, index_workspace;

    Case(int batch_, int width_, int context_, bool tensor_cores_, bool int8_kv_)
        : batch(batch_), width(width_), context(context_), tensor_cores(tensor_cores_),
          int8_kv(int8_kv_) {
        pages_per_row  = (context + kPage - 1) / kPage;
        pages          = pages_per_row * batch;
        const int cols = batch * width;
        // Index cache planes (fixture BF16 in [0.002, 0.01]; the timing does not read values).
        k      = bench::make_bf16(static_cast<std::size_t>(kIndexDim) * kPage * batch, 11);
        z      = bench::make_bf16(static_cast<std::size_t>(kIndexDim) * kPage * batch, 12);
        pooled = bench::make_bf16(static_cast<std::size_t>(kIndexDim / kPool) * kPage * pages, 13);
        if (int8_kv) {
            latents       = DeviceBuffer(static_cast<std::size_t>(kLatent) * kPage * pages);
            latent_scales = bench::make_bf16(static_cast<std::size_t>(kLatent / 64) * kPage * pages, 14, 0.005F, 0.02F);
            CUDA_CHECK(cudaMemset(latents.p, 17, latents.bytes));
        } else {
            latents = bench::make_bf16(static_cast<std::size_t>(kLatent) * kPage * pages, 14);
        }
        ape     = bench::make_zeros(sizeof(float) * kPool * kIndexDim);
        gamma   = bench::make_bf16(kIndexDim, 15, 0.9F, 1.1F);
        beta    = bench::make_bf16(kIndexDim, 16, -0.1F, 0.1F);
        raw_k   = bench::make_bf16(static_cast<std::size_t>(kIndexDim) * cols, 17);
        index_z = bench::make_bf16(static_cast<std::size_t>(kIndexDim) * cols, 18);
        index_q = bench::make_bf16(static_cast<std::size_t>(kIndexDim) * kIndexHeads * cols, 19);
        index_w = bench::make_bf16(static_cast<std::size_t>(kIndexHeads) * cols, 20);
        pools   = DeviceBuffer(sizeof(std::int32_t) * kTopPools * cols);
        counts  = DeviceBuffer(sizeof(std::int32_t) * cols);
        // Latent attention operands.
        q    = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kHeads * cols, 21);
        w_uk = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kLatent * kHeads, 22);
        w_uv = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kLatent * kHeads, 23);
        qa   = bench::make_bf16(static_cast<std::size_t>(kLatent) * kHeads * cols, 24);
        out  = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kHeads * cols, 25);
        // Batch descriptors: lane b at the context's last `width` positions, disjoint pages per lane.
        std::vector<std::int32_t> table(static_cast<std::size_t>(pages_per_row) * batch),
            position(cols), rows_h(batch);
        for (int b = 0; b < batch; ++b) {
            rows_h[b] = b;
            for (int p = 0; p < pages_per_row; ++p) {
                table[b * pages_per_row + p] = b * pages_per_row + p;
            }
            for (int w = 0; w < width; ++w) {
                position[b * width + w] = context - width + w;
            }
        }
        const auto upload = [](const std::vector<std::int32_t>& h) {
            DeviceBuffer d(h.size() * sizeof(std::int32_t));
            CUDA_CHECK(cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice));
            return d;
        };
        tables    = upload(table);
        positions = upload(position);
        rows      = upload(rows_h);
        // Deterministic sparse pool selection (even stride across available pools) so attention
        // can run standalone without select() when --attention-only is set.
        std::vector<std::int32_t> pools_h(static_cast<std::size_t>(kTopPools) * cols, 0);
        std::vector<std::int32_t> counts_h(cols, 0);
        for (int c = 0; c < cols; ++c) {
            const int avail = (position[c] + 1) / kPool;
            const int cnt   = std::min(kTopPools, avail);
            counts_h[c]     = cnt;
            for (int i = 0; i < cnt; ++i) {
                pools_h[static_cast<std::size_t>(c) * kTopPools + i] =
                    static_cast<std::int32_t>((static_cast<std::int64_t>(i) * avail) / cnt);
            }
        }
        CUDA_CHECK(cudaMemcpy(pools.p, pools_h.data(), pools.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(counts.p, counts_h.data(), counts.bytes, cudaMemcpyHostToDevice));
        const ops::MlaSparseBatch batch_desc{
            .block_tables = Tensor(tables.p, DType::I32, {pages_per_row, batch}),
            .table_rows   = Tensor(rows.p, DType::I32, {batch}),
            .positions    = Tensor(positions.p, DType::I32, {cols}),
            .batch        = batch,
            .width        = width};
        index_workspace_bytes =
            ops::dsa_indexer_select_workspace_bytes(kIndexerGeometry, cols, context);
        index_workspace = DeviceBuffer(index_workspace_bytes);
        const ops::MlaAttentionExecutionEnvelope env{.tensor_cores = tensor_cores};
        workspace_bytes = ops::mla_sparse_attention_workspace_bytes(
            kAttentionGeometry, cols, context, env);
        attn_workspace = DeviceBuffer(workspace_bytes);
        batch_desc_    = batch_desc;
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ops::MlaSparseBatch batch_desc_{};

    ops::DsaIndexerCacheView index_cache() const {
        return {.k_recent = Tensor(k.p, DType::BF16, {kIndexDim, kPage, batch}),
                .z_recent = Tensor(z.p, DType::BF16, {kIndexDim, kPage, batch}),
                .pooled_pages =
                    Tensor(pooled.p, DType::BF16, {kIndexDim / kPool, kPage, 1, pages})};
    }

    void update(cudaStream_t s) const {
        const int cols = batch * width;
        ops::dsa_indexer_update(Tensor(raw_k.p, DType::BF16, {kIndexDim, cols}),
                                Tensor(index_z.p, DType::BF16, {kIndexDim, cols}),
                                Tensor(gamma.p, DType::BF16, {kIndexDim}),
                                Tensor(beta.p, DType::BF16, {kIndexDim}),
                                Tensor(ape.p, DType::FP32, {kIndexDim, kPool}), index_cache(),
                                batch_desc_, kIndexerGeometry, s);
    }

    void select(cudaStream_t s) const {
        const int cols = batch * width;
        Tensor pools_out(pools.p, DType::I32, {kTopPools, cols});
        Tensor counts_out(counts.p, DType::I32, {cols});
        ops::dsa_indexer_select(Tensor(index_q.p, DType::BF16, {kIndexDim, kIndexHeads, cols}),
                                Tensor(index_w.p, DType::BF16, {kIndexHeads, cols}), index_cache(),
                                batch_desc_, kIndexerGeometry, kIndexScale, context,
                                index_workspace.p, index_workspace_bytes, pools_out, counts_out, s);
    }

    void project(cudaStream_t s) const {
        const int cols = batch * width;
        Tensor qa_out(qa.p, DType::BF16, {kLatent, kHeads, cols});
        ops::mla_absorbed_q_projection(Tensor(q.p, DType::BF16, {kHeadDim, kHeads, cols}),
                                       w_uk_tensor(), qa_out, s);
    }

    void attention(cudaStream_t s) const {
        const int cols = batch * width;
        Tensor pools_out(pools.p, DType::I32, {kTopPools, cols});
        Tensor counts_out(counts.p, DType::I32, {cols});
        const ops::MlaSelectedPools selected{
            .pools = &pools_out, .counts = &counts_out, .pool = kPool};
        Tensor o(out.p, DType::BF16, {kHeadDim, kHeads, cols});
        ops::MlaLatentBatchView cache{};
        if (int8_kv) {
            cache.pages  = Tensor(latents.p, DType::I8, {kLatent, kPage, 1, pages});
            cache.scales = Tensor(latent_scales.p, DType::FP16, {kLatent / 64, kPage, 1, pages});
        } else {
            cache.pages = Tensor(latents.p, DType::BF16, {kLatent, kPage, 1, pages});
        }
        ops::mla_sparse_attention(
            Tensor(qa.p, DType::BF16, {kLatent, kHeads, cols}), w_uv_tensor(), cache, batch_desc_,
            selected, kAttentionGeometry, kScale, context,
            ops::MlaAttentionExecutionEnvelope{.tensor_cores = tensor_cores}, attn_workspace.p,
            workspace_bytes, o, s);
    }

    Tensor w_uk_tensor() const { return Tensor(w_uk.p, DType::BF16, {kHeadDim, kLatent, kHeads}); }

    Tensor w_uv_tensor() const { return Tensor(w_uv.p, DType::BF16, {kLatent, kHeadDim, kHeads}); }

    // The unavoidable DRAM traffic of each stage of one round (bytes), for the printed
    // speed-of-light column. Weights and per-column streams are cold: an L2 flush precedes
    // every sample and a decode round reads each of them once.
    void bytes(double& update_b, double& select_b, double& project_b, double& attention_b) const {
        const int cols             = batch;
        const int completing       = context % kPool == 0 ? cols : 0; // lanes completing a pool
        const long long pools_n    = static_cast<long long>(context) / kPool;
        const long long selected   = std::min(kTopPools, static_cast<int>(pools_n));
        const long long attended   = selected * kPool;
        const long long splits     = std::min(std::max((context + 31) / 32, 1), 65);
        const long long col_splits = std::min(splits, (attended + 31) / 32);
        const double index_token   = 2.0 * kIndexDim; // one BF16 token of k or z
        // update: the call's raw vectors in, the stored k/z tokens out; a pool completion reads
        // its pool's four stored k/z tokens and writes one 64-wide BF16 pooled key.
        update_b = cols * (2.0 * index_token + 2.0 * index_token) +
                   completing * (kPool * 2.0 * index_token + 2.0 * (kIndexDim / kPool));
        // select: the queries and per-head weights in, every complete pool's pooled key in,
        // the score rows and the histogram state through the workspace, the selection out.
        select_b = cols * (2.0 * kIndexDim * kIndexHeads + 2.0 * kIndexHeads) +
                   cols * pools_n * 2.0 * (kIndexDim / kPool) +
                   cols * (pools_n * 4.0 + (2048 + 1) * 4.0) + cols * (kTopPools * 4.0 + 4.0);
        // project: q in, the layer's W_uk in once, qa out.
        project_b = cols * 2.0 * kHeadDim * kHeads + 2.0 * kHeadDim * kLatent * kHeads +
                    cols * 2.0 * kLatent * kHeads;
        // attention: qa in, W_uv in once, the selected tokens' latents in, the split partials
        // written and read back, the merged sums through the projection, out.
        const double partial = col_splits * (kHeads / 16) * 16.0 * 514.0 * 4.0;
        attention_b = cols * 2.0 * kLatent * kHeads + 2.0 * kHeadDim * kLatent * kHeads +
                      cols * attended * 2.0 * kLatent + cols * 2.0 * partial +
                      cols * (2.0 * kLatent * kHeads * 4.0) + cols * 2.0 * kHeadDim * kHeads;
    }
};

bench::ColdTiming time(bench::L2FlushBuffer& flush, cudaStream_t stream, int warmup, int repeat,
                       const std::function<void(cudaStream_t)>& body) {
    body(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bench::TimedGraph graph;
    graph.capture(stream, body);
    return bench::measure_cold_graph(graph, flush, stream, warmup, repeat);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlush);
        std::printf("context batch   update_us  select_us project_us attention_us  total_us"
                    "     bytes_mb   gb_per_s  (median of %d, cold L2, graph)\n",
                    options.repeat);
        for (const int context : options.contexts) {
            for (const int batch : options.batches) {
                Case data(batch, options.width, context, options.tensor_cores, options.int8_kv);
                double update_b = 0, select_b = 0, project_b = 0, attention_b = 0;
                data.bytes(update_b, select_b, project_b, attention_b);
                const double bytes = update_b + select_b + project_b + attention_b;
                try {
                    const double update_us =
                        options.attention_only
                            ? 0.0
                            : time(flush, stream, options.warmup, options.repeat,
                                   [&](cudaStream_t s) { data.update(s); })
                                  .median_us;
                    const double select_us =
                        options.attention_only
                            ? 0.0
                            : time(flush, stream, options.warmup, options.repeat,
                                   [&](cudaStream_t s) { data.select(s); })
                                  .median_us;
                    const double project_us =
                        options.attention_only
                            ? 0.0
                            : time(flush, stream, options.warmup, options.repeat,
                                   [&](cudaStream_t s) { data.project(s); })
                                  .median_us;
                    const auto attention = time(flush, stream, options.warmup, options.repeat,
                                                [&](cudaStream_t s) { data.attention(s); });
                    const double total   = update_us + select_us + project_us + attention.median_us;
                    std::printf("%7d %5d %11.2f %10.2f %10.2f %12.2f %9.2f %12.2f %10.2f\n",
                                context, batch * options.width, update_us, select_us, project_us,
                                attention.median_us, total, bytes / 1.0e6, bytes / total / 1.0e3);
                } catch (const std::invalid_argument& e) {
                    std::printf("%7d %5d unsupported: %s\n", context, batch, e.what());
                }
                std::fflush(stdout);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_mla_sparse_bench: %s\n", error.what());
        return 1;
    }
}
