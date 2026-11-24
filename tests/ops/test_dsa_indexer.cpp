// The pooled-key indexer of sparse latent attention (include/infernix/ops/dsa_indexer.h) against
// independent FP64 oracles (docs/maintainer/op-development.md section 6), at the registered
// geometry (32 index heads, head_dim 128, pools of 4, a 512-pool budget):
//
//   - dsa_indexer_update: the LayerNorm'd index keys in the recent ring (FP64 layer norm, one
//     BF16 rounding), the gate vectors bit for bit, and the pooled keys (FP64 softmax over the
//     BF16 gates and the FP32 ape, FP64 channel sums, one BF16 rounding), across lanes, widths
//     wider than the ring, split calls and a rollback; untouched cache slots keep their bits.
//   - dsa_indexer_select: the selected pool set equals the exact top-512 of the FP64 scores
//     (score = sum_h w_h * relu(scale * <q_h, key_p>), ties to the lower pool id) whenever the
//     boundary gap exceeds the FP32 evaluation error; all-equal scores select pools 0..511;
//     counts = -1 at and below the dense boundary (P <= 512) and the pool slots stay untouched;
//     lanes with different contexts and verification widths in one call.
//   - column invariance: a column's selection bits do not depend on the call's other columns or
//     lanes.
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "infernix/ops/dsa_indexer.h"
#include "ops/host_parallel.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ops = infernix::ops;
namespace t   = infernix::test;
using infernix::DeviceBuffer;
using infernix::DType;
using infernix::Tensor;

constexpr int kHeads = 32, kHeadDim = 128, kPool = 4, kTopPools = 512, kPage = 64, kRecent = 64;
constexpr float kScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
constexpr float kEps   = 1.0e-6F;

const ops::DsaIndexerGeometry kGeometry{kHeads, kHeadDim, kPool, kTopPools, kEps};

// One BF16 rounding of an FP32 evaluation of the formula (layer norm or pooled key).
constexpr t::PointwiseCriterion kStoredCriterion{/*absolute*/ 1.0e-5, /*relative*/ 4.0e-3};

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

std::uint16_t to_bf16(float f) { return t::f32_to_bf16(f); }

float from_bf16(std::uint16_t h) { return t::bf16_to_f32(h); }

std::vector<std::uint16_t> bf16_inputs(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    t::fill_uniform(v, seed, lo, hi);
    t::round_to_bf16(v);
    std::vector<std::uint16_t> bits(n);
    for (std::size_t i = 0; i < n; ++i) { bits[i] = to_bf16(v[i]); }
    return bits;
}

DeviceBuffer upload(const std::vector<std::uint16_t>& bits) {
    DeviceBuffer d(bits.size() * 2);
    d.copy_from_host(bits.data(), d.bytes);
    return d;
}

DeviceBuffer upload_i32(const std::vector<std::int32_t>& v) {
    DeviceBuffer d(v.size() * sizeof(std::int32_t));
    d.copy_from_host(v.data(), d.bytes);
    return d;
}

DeviceBuffer upload_f32(const std::vector<float>& v) {
    DeviceBuffer d(v.size() * 4);
    d.copy_from_host(v.data(), d.bytes);
    return d;
}

/// One batched indexer fixture: lanes with their own contexts, scattered page tables, canary
/// cache planes and per-column query streams.
struct Fixture {
    int batch = 0, width = 0, pages_per_row = 0, plane_pages = 0, max_context = 0;
    std::vector<int> contexts;
    std::vector<std::int32_t> positions_host, tables_host, pools_host, counts_host;
    std::vector<std::uint16_t> raw_k_host, z_host, gamma_host, beta_host, q_host, w_host;
    std::vector<std::uint16_t> k_recent_host, z_recent_host, pooled_host;
    std::vector<float> ape_host;
    DeviceBuffer raw_k_buf, z_buf, gamma_buf, beta_buf, ape_buf, k_buf, z_buf2, pooled_buf,
        tables_buf, rows_buf, positions_buf, q_buf, w_buf, pools_buf, counts_buf, ws_buf;
    Tensor raw_k, z, gamma, beta, ape, q, w, pools, counts;
    ops::DsaIndexerCacheView cache;
    ops::MlaSparseBatch batch_desc;
    std::size_t workspace_bytes = 0;

    Fixture() = default;

    Fixture(int batch_, int width_, std::vector<int> contexts_, std::uint32_t seed) {
        batch             = batch_;
        width             = width_;
        contexts          = std::move(contexts_);
        max_context       = *std::max_element(contexts.begin(), contexts.end());
        pages_per_row     = (max_context + kPage - 1) / kPage;
        const int columns = batch * width;

        raw_k_host = bf16_inputs(static_cast<std::size_t>(kHeadDim) * columns, seed, -1.0F, 1.0F);
        z_host = bf16_inputs(static_cast<std::size_t>(kHeadDim) * columns, seed + 1, -2.0F, 2.0F);
        gamma_host = bf16_inputs(kHeadDim, seed + 2, 0.5F, 1.5F);
        beta_host  = bf16_inputs(kHeadDim, seed + 3, -0.25F, 0.25F);
        q_host = bf16_inputs(static_cast<std::size_t>(kHeadDim) * kHeads * columns, seed + 4, -1.0F,
                             1.0F);
        w_host = bf16_inputs(static_cast<std::size_t>(kHeads) * columns, seed + 5, -1.0F, 1.0F);
        ape_host.resize(static_cast<std::size_t>(kPool) * kHeadDim);
        {
            std::mt19937 rng(seed + 6);
            std::uniform_real_distribution<float> d(-0.5F, 0.5F);
            for (auto& a : ape_host) { a = d(rng); }
        }
        // The cache planes start from a canary pattern; the update must overwrite exactly the
        // call's slots and completed pools' keys.
        plane_pages = pages_per_row * batch;
        k_recent_host.assign(static_cast<std::size_t>(kHeadDim) * kRecent * batch,
                             static_cast<std::uint16_t>(0x3F80));
        z_recent_host = k_recent_host;
        pooled_host.assign(static_cast<std::size_t>(kHeadDim / kPool) * kPage * plane_pages,
                           static_cast<std::uint16_t>(0x3F80));

        std::vector<std::int32_t> tables(static_cast<std::size_t>(pages_per_row) * batch);
        for (int r = 0; r < batch; ++r) {
            for (int p = 0; p < pages_per_row; ++p) {
                tables[static_cast<std::size_t>(r) * pages_per_row + p] =
                    static_cast<std::int32_t>(p + r * pages_per_row); // disjoint pages per lane
            }
        }
        tables_host = tables;
        std::vector<std::int32_t> rows(batch);
        std::iota(rows.begin(), rows.end(), 0);
        std::vector<std::int32_t> positions(columns);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < width; ++j) { positions[b * width + j] = contexts[b] - width + j; }
        }
        positions_host = positions;
        pools_host.assign(static_cast<std::size_t>(kTopPools) * columns, -12345);
        counts_host.assign(columns, -6789);

        raw_k_buf     = upload(raw_k_host);
        z_buf         = upload(z_host);
        gamma_buf     = upload(gamma_host);
        beta_buf      = upload(beta_host);
        ape_buf       = upload_f32(ape_host);
        k_buf         = upload(k_recent_host);
        z_buf2        = upload(z_recent_host);
        pooled_buf    = upload(pooled_host);
        tables_buf    = upload_i32(tables);
        rows_buf      = upload_i32(rows);
        positions_buf = upload_i32(positions);
        q_buf         = upload(q_host);
        w_buf         = upload(w_host);
        pools_buf     = upload_i32(pools_host);
        counts_buf    = upload_i32(counts_host);

        raw_k  = Tensor(raw_k_buf.p, DType::BF16, {kHeadDim, columns});
        z      = Tensor(z_buf.p, DType::BF16, {kHeadDim, columns});
        gamma  = Tensor(gamma_buf.p, DType::BF16, {kHeadDim});
        beta   = Tensor(beta_buf.p, DType::BF16, {kHeadDim});
        ape    = Tensor(ape_buf.p, DType::FP32, {kHeadDim, kPool});
        q      = Tensor(q_buf.p, DType::BF16, {kHeadDim, kHeads, columns});
        w      = Tensor(w_buf.p, DType::BF16, {kHeads, columns});
        pools  = Tensor(pools_buf.p, DType::I32, {kTopPools, columns});
        counts = Tensor(counts_buf.p, DType::I32, {columns});
        cache  = ops::DsaIndexerCacheView{
            .k_recent = Tensor(k_buf.p, DType::BF16, {kHeadDim, kRecent, batch}),
            .z_recent = Tensor(z_buf2.p, DType::BF16, {kHeadDim, kRecent, batch}),
            .pooled_pages =
                Tensor(pooled_buf.p, DType::BF16, {kHeadDim / kPool, kPage, 1, plane_pages})};
        batch_desc = ops::MlaSparseBatch{
            .block_tables = Tensor(tables_buf.p, DType::I32, {pages_per_row, batch}),
            .table_rows   = Tensor(rows_buf.p, DType::I32, {batch}),
            .positions    = Tensor(positions_buf.p, DType::I32, {columns}),
            .batch        = batch,
            .width        = width};
        workspace_bytes = ops::dsa_indexer_select_workspace_bytes(kGeometry, columns, max_context);
        ws_buf          = DeviceBuffer(workspace_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // The select Op only reads the pooled plane; the select tests fill it with a random fixture
    // instead of deriving it from an update over the whole context.
    void fill_pooled_random(std::uint32_t seed) {
        pooled_host = bf16_inputs(pooled_host.size(), seed, 0.05F, 0.3F);
        pooled_buf.copy_from_host(pooled_host.data(), pooled_host.size() * 2);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_update(cudaStream_t stream) {
        ops::dsa_indexer_update(raw_k, z, gamma, beta, ape, cache, batch_desc, kGeometry, stream);
        t::cuda_synchronize(stream);
    }

    void run_select(cudaStream_t stream) {
        ops::dsa_indexer_select(q, w, cache, batch_desc, kGeometry, kScale, max_context, ws_buf.p,
                                workspace_bytes, pools, counts, stream);
        t::cuda_synchronize(stream);
    }

    // The stored cache planes after the call, from the device.
    void read_cache() {
        k_recent_host = t::from_device<std::uint16_t>(k_buf, k_recent_host.size());
        z_recent_host = t::from_device<std::uint16_t>(z_buf2, z_recent_host.size());
        pooled_host  = t::from_device<std::uint16_t>(pooled_buf, pooled_host.size());
        pools_host   = t::from_device<std::int32_t>(pools_buf, pools_host.size());
        counts_host  = t::from_device<std::int32_t>(counts_buf, counts_host.size());
    }

    // The page-table page of a position of sequence b.
    std::int32_t page_of(int b, int position) const {
        return tables_host[static_cast<std::size_t>(position / kPage) +
                           static_cast<std::size_t>(b) * pages_per_row];
    }

    // The recent ring's first element of sequence b's position.
    static std::size_t ring_base(int b, int position) {
        return (static_cast<std::size_t>(b) * kRecent + position % kRecent) * kHeadDim;
    }

    // The stored pooled key of pool p of sequence b (channels 0..127), from the cache plane.
    std::array<float, kHeadDim> pooled_key(int b, int pool) const {
        std::array<float, kHeadDim> key;
        const int slot_width = kHeadDim / kPool;
        for (int ch = 0; ch < kHeadDim; ++ch) {
            const int token         = pool * kPool + ch / slot_width;
            const std::int32_t page = page_of(b, token);
            const std::size_t at    = static_cast<std::size_t>(ch % slot_width) +
                                      static_cast<std::size_t>(token % kPage) * slot_width +
                                      static_cast<std::size_t>(page) * slot_width * kPage;
            key[ch]                 = from_bf16(pooled_host[at]);
        }
        return key;
    }
};

// --------------------------------------------------------------------------------------- oracles

// The FP64 layer norm of one column's raw key: what the stored BF16 key rounds.
std::vector<double> ln_reference(const Fixture& f, int column) {
    std::vector<double> x(kHeadDim);
    double mean = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
        x[d] = from_bf16(f.raw_k_host[d + static_cast<std::size_t>(column) * kHeadDim]);
        mean += x[d];
    }
    mean /= kHeadDim;
    double var = 0.0;
    for (int d = 0; d < kHeadDim; ++d) { var += (x[d] - mean) * (x[d] - mean); }
    var /= kHeadDim;
    std::vector<double> y(kHeadDim);
    for (int d = 0; d < kHeadDim; ++d) {
        y[d] = (x[d] - mean) / std::sqrt(var + kEps) * from_bf16(f.gamma_host[d]) +
               from_bf16(f.beta_host[d]);
    }
    return y;
}

// The FP64 pooled key of pool `pool` of sequence b: softmax over the BF16 gates and the FP32 ape,
// channel sums of the BF16 keys, before the stored key's own BF16 rounding. `token_kz` gives a
// pool token's (k, z) bits for a channel.
template <class TokenKz>
std::vector<double> pooled_reference(const Fixture& f, int pool, TokenKz token_kz) {
    std::vector<double> key(kHeadDim, 0.0);
    for (int d = 0; d < kHeadDim; ++d) {
        double e[kPool], k[kPool], total = 0.0;
        for (int i = 0; i < kPool; ++i) {
            const auto [kb, zb] = token_kz(pool * kPool + i, d);
            k[i]                = from_bf16(kb);
            e[i] = std::exp(from_bf16(zb) + f.ape_host[static_cast<std::size_t>(i) * kHeadDim + d]);
            total += e[i];
        }
        for (int i = 0; i < kPool; ++i) { key[d] += e[i] / total * k[i]; }
    }
    return key;
}

// The FP64 score of pool `pool` for column c: sum_h w_h * relu(scale * <q_h, key_p>) over the
// stored pooled key.
double score_reference(const Fixture& f, int column, int pool) {
    const std::array<float, kHeadDim> key = f.pooled_key(column / f.width, pool);
    double score                          = 0.0;
    for (int h = 0; h < kHeads; ++h) {
        double dot = 0.0;
        for (int d = 0; d < kHeadDim; ++d) {
            dot += from_bf16(f.q_host[d + static_cast<std::size_t>(h) * kHeadDim +
                                      static_cast<std::size_t>(column) * kHeadDim * kHeads]) *
                   key[d];
        }
        // relu(scale * dot) = scale * relu(dot) for a positive scale, bitwise in FP32.
        score += from_bf16(f.w_host[h + static_cast<std::size_t>(column) * kHeads]) *
                 std::fmax(dot, 0.0);
    }
    return score * kScale;
}

// ---------------------------------------------------------------------------------------- update

void test_update() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Fixture f(2, 5, {300, 130}, 11);
    f.run_update(stream);
    f.read_cache();
    const std::uint16_t canary = static_cast<std::uint16_t>(0x3F80);
    // Every column's stored key against the FP64 layer norm, its gate bit for bit, and every
    // completed pool's key against the FP64 formula; every other slot keeps the canary.
    std::vector<double> k_got, k_ref, key_got, key_ref;
    int z_bad = 0, untouched_bad = 0;
    std::vector<char> pool_completed(
        static_cast<std::size_t>(kPool) * ((f.max_context + kPool - 1) / kPool) + 1, 0);
    for (int b = 0; b < f.batch; ++b) {
        for (int j = 0; j < f.width; ++j) {
            const int column            = b * f.width + j;
            const int p                 = f.positions_host[column];
            const std::size_t slot      = Fixture::ring_base(b, p);
            const std::vector<double> y = ln_reference(f, column);
            for (int d = 0; d < kHeadDim; ++d) {
                k_got.push_back(from_bf16(f.k_recent_host[slot + d]));
                k_ref.push_back(y[d]);
                if (f.z_recent_host[slot + d] !=
                    f.z_host[d + static_cast<std::size_t>(column) * kHeadDim]) {
                    ++z_bad;
                }
            }
            if ((p + 1) % kPool == 0) {
                const int pool                = (p + 1) / kPool - 1;
                pool_completed[static_cast<std::size_t>(b) * ((f.max_context + kPool - 1) / kPool) +
                               pool]          = 1;
                // Every pool token's key and gate is in the ring: the call's (checked above) and
                // the fixture's earlier ones.
                const std::vector<double> ref = pooled_reference(f, pool, [&](int token, int d) {
                    const std::size_t at = Fixture::ring_base(b, token) + d;
                    return std::pair{f.k_recent_host[at], f.z_recent_host[at]};
                });
                const auto stored             = f.pooled_key(b, pool);
                for (int d = 0; d < kHeadDim; ++d) {
                    key_got.push_back(stored[d]);
                    key_ref.push_back(ref[d]);
                }
            }
        }
    }
    g_failures += t::verify_pointwise("stored index keys", k_got, k_ref, kStoredCriterion);
    g_failures += t::verify_pointwise("stored pooled keys", key_got, key_ref, kStoredCriterion);
    expect(z_bad == 0, "update: gate vectors stored bit for bit");
    // Untouched slots: every ring slot of a position outside the call, and every pool the call
    // did not complete.
    for (int b = 0; b < f.batch; ++b) {
        const int first = f.positions_host[b * f.width];
        for (int p = first + f.width; p < first + kRecent; ++p) {
            const std::size_t slot = Fixture::ring_base(b, p);
            for (int d = 0; d < kHeadDim; ++d) {
                if (f.k_recent_host[slot + d] != canary || f.z_recent_host[slot + d] != canary) {
                    ++untouched_bad;
                }
            }
        }
        const int pools_of_row = (f.contexts[b] + kPool - 1) / kPool;
        for (int pool = 0; pool < pools_of_row; ++pool) {
            if (pool_completed[static_cast<std::size_t>(b) * ((f.max_context + kPool - 1) / kPool) +
                               pool]) {
                continue;
            }
            const auto stored = f.pooled_key(b, pool);
            for (int d = 0; d < kHeadDim; ++d) {
                if (stored[d] != from_bf16(canary)) { ++untouched_bad; }
            }
        }
    }
    expect(untouched_bad == 0, "update: untouched cache slots keep their bits");

    // Splitting one sequence's update into calls of 5 + 3 columns stores the same bytes.
    Fixture one(1, 8, {308}, 11);
    one.run_update(stream);
    one.read_cache();
    Fixture split(1, 8, {308}, 11);
    const Tensor tables = split.batch_desc.block_tables, rows = split.batch_desc.table_rows;
    const ops::MlaSparseBatch call_a{.block_tables = tables,
                                     .table_rows   = rows,
                                     .positions    = split.batch_desc.positions.slice(0, 0, 5),
                                     .batch        = 1,
                                     .width        = 5};
    const ops::MlaSparseBatch call_b{.block_tables = tables,
                                     .table_rows   = rows,
                                     .positions    = split.batch_desc.positions.slice(0, 5, 3),
                                     .batch        = 1,
                                     .width        = 3};
    ops::dsa_indexer_update(split.raw_k.slice(1, 0, 5), split.z.slice(1, 0, 5), split.gamma,
                            split.beta, split.ape, split.cache, call_a, kGeometry, stream);
    t::cuda_synchronize(stream);
    ops::dsa_indexer_update(split.raw_k.slice(1, 5, 3), split.z.slice(1, 5, 3), split.gamma,
                            split.beta, split.ape, split.cache, call_b, kGeometry, stream);
    t::cuda_synchronize(stream);
    split.read_cache();
    expect(split.k_recent_host == one.k_recent_host && split.z_recent_host == one.z_recent_host &&
               split.pooled_host == one.pooled_host,
           "update: split calls store the same bytes");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// The call's columns [begin, begin + n) of fixture `inputs` appended to `cache_of`'s cache (same
// positions; `cache_of`'s LayerNorm weights and ape).
void append_columns(Fixture& cache_of, const Fixture& inputs, int begin, int n,
                    cudaStream_t stream) {
    const ops::MlaSparseBatch call{.block_tables = cache_of.batch_desc.block_tables,
                                   .table_rows   = cache_of.batch_desc.table_rows,
                                   .positions    = cache_of.batch_desc.positions.slice(0, begin, n),
                                   .batch        = 1,
                                   .width        = n};
    ops::dsa_indexer_update(inputs.raw_k.slice(1, begin, n), inputs.z.slice(1, begin, n),
                            cache_of.gamma, cache_of.beta, cache_of.ape, cache_of.cache, call,
                            kGeometry, stream);
    t::cuda_synchronize(stream);
}

// Calls wider than the ring: the pooled keys of a 150-column prefill (whose pools the kernel
// forms from the call's own inputs) against the FP64 formula, the same bytes in calls of 70 + 80
// and 5 + 145 columns, and a rollback: positions 0..99 appended, then 90..109 with other inputs,
// stores what 0..89 and then 90..109 store.
void test_wide_and_rollback() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr int kWide = 150;
    Fixture whole(1, kWide, {kWide}, 71);
    whole.run_update(stream);
    whole.read_cache();
    std::vector<double> key_got, key_ref;
    for (int pool = 0; pool < kWide / kPool; ++pool) {
        const auto ref    = pooled_reference(whole, pool, [&](int token, int d) {
            const std::vector<double> y = ln_reference(whole, token);
            return std::pair{to_bf16(static_cast<float>(y[d])),
                             whole.z_host[d + static_cast<std::size_t>(token) * kHeadDim]};
        });
        const auto stored = whole.pooled_key(0, pool);
        for (int d = 0; d < kHeadDim; ++d) {
            key_got.push_back(stored[d]);
            key_ref.push_back(ref[d]);
        }
    }
    g_failures += t::verify_pointwise("wide call pooled keys", key_got, key_ref, kStoredCriterion);
    for (const auto& [a, b] : {std::pair{70, 80}, std::pair{5, 145}}) {
        Fixture split(1, kWide, {kWide}, 71);
        append_columns(split, split, 0, a, stream);
        append_columns(split, split, a, b, stream);
        split.read_cache();
        expect(split.k_recent_host == whole.k_recent_host &&
                   split.z_recent_host == whole.z_recent_host &&
                   split.pooled_host == whole.pooled_host,
               "update: a wide sequence split into " + std::to_string(a) + " + " +
                   std::to_string(b) + " columns stores the same bytes");
    }
    Fixture rolled(1, 110, {110}, 81), redo(1, 110, {110}, 91), fresh(1, 110, {110}, 81);
    append_columns(rolled, rolled, 0, 100, stream);
    append_columns(rolled, redo, 90, 20, stream);
    rolled.read_cache();
    append_columns(fresh, fresh, 0, 90, stream);
    append_columns(fresh, redo, 90, 20, stream);
    fresh.read_cache();
    expect(rolled.k_recent_host == fresh.k_recent_host &&
               rolled.z_recent_host == fresh.z_recent_host &&
               rolled.pooled_host == fresh.pooled_host,
           "update: a rollback of 10 positions stores what the rewritten sequence stores");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------------------- select

// The exact top-`top` pool ids of one column's FP64 scores (ties to the lower pool id), ascending.
std::vector<int> top_pools(const std::vector<double>& scores, int top) {
    std::vector<std::pair<double, int>> order(scores.size());
    for (std::size_t b = 0; b < scores.size(); ++b) { order[b] = {scores[b], static_cast<int>(b)}; }
    std::stable_sort(order.begin(), order.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> chosen;
    for (int k = 0; k < top; ++k) { chosen.push_back(order[k].second); }
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

void test_select() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // The dense boundary: P = 512 at n = 2051 selects nothing and touches no pool slot.
    {
        Fixture f(1, 1, {2051}, 21);
        f.fill_pooled_random(21);
        f.run_select(stream);
        f.read_cache();
        expect(f.counts_host[0] == -1, "select: the dense boundary count is -1");
        expect(f.pools_host[0] == -12345,
               "select: the dense boundary leaves the pool slots untouched");
    }
    // Sparse lanes: n = 6000 (1500 pools), the sparse boundary n = 2052 (513 pools) and the dense
    // n = 2051, against the exact top-512 of the FP64 scores whenever the boundary gap exceeds
    // the FP32 evaluation error.
    Fixture f;
    for (std::uint32_t seed = 31; seed < 39; ++seed) {
        f = Fixture(3, 1, {6000, 2052, 2051}, seed);
        f.fill_pooled_random(seed);
        f.run_select(stream);
        f.read_cache();
        double worst_gap = 1.0e30;
        bool gaps_ok     = true;
        for (int c = 0; c < 3; ++c) {
            const int pools_n = (f.positions_host[c] + 1) / kPool;
            if (pools_n <= kTopPools) { continue; }
            std::vector<double> scores(pools_n);
            for (int b = 0; b < pools_n; ++b) { scores[b] = score_reference(f, c, b); }
            std::vector<double> sorted(scores);
            std::sort(sorted.begin(), sorted.end(), std::greater<double>());
            const double gap       = sorted[kTopPools - 1] - sorted[kTopPools];
            const double scale_ref = std::max(std::abs(sorted.front()), 1.0e-30);
            worst_gap              = std::min(worst_gap, gap / scale_ref);
            if (gap <= 1.0e-4 * scale_ref) { gaps_ok = false; }
        }
        if (gaps_ok) { break; }
        if (seed == 38) { f = Fixture(); } // no well-separated seed found; the checks below skip
    }
    if (f.batch != 0) {
        f.read_cache();
        expect(f.counts_host[0] == kTopPools && f.counts_host[1] == kTopPools &&
                   f.counts_host[2] == -1,
               "select: counts at and beyond the sparse boundary");
        for (int c = 0; c < 3; ++c) {
            const int pools_n = (f.positions_host[c] + 1) / kPool;
            if (pools_n <= kTopPools) {
                expect(f.pools_host[static_cast<std::size_t>(c) * kTopPools] == -12345,
                       "select: the dense column leaves the pool slots untouched");
                continue;
            }
            std::vector<double> scores(pools_n);
            for (int b = 0; b < pools_n; ++b) { scores[b] = score_reference(f, c, b); }
            const auto want = top_pools(scores, kTopPools);
            bool same       = true;
            for (int k = 0; k < kTopPools; ++k) {
                if (f.pools_host[static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(c) * kTopPools] != want[k]) {
                    if (same) {
                        const int got_pool = f.pools_host[k + c * kTopPools];
                        std::fprintf(
                            stderr,
                            "set mismatch col %d at %d: got %d want %d; scores got~%g want~%g\n", c,
                            k, got_pool, want[k], scores[got_pool], scores[want[k]]);
                    }
                    same = false;
                }
            }
            expect(same,
                   "select: column " + std::to_string(c) + " selects the exact top-512 pools");
        }
    } else {
        std::fprintf(stderr, "note: no well-separated sparse seed found; set equality skipped\n");
    }
    // All pooled keys equal: every score ties and the selection is pools 0..511.
    {
        Fixture f(1, 1, {6000}, 41);
        std::fill(f.pooled_host.begin(), f.pooled_host.end(), to_bf16(0.25F));
        f.pooled_buf.copy_from_host(f.pooled_host.data(), f.pooled_host.size() * 2);
        CUDA_CHECK(cudaDeviceSynchronize());
        f.run_select(stream);
        f.read_cache();
        expect(f.counts_host[0] == kTopPools, "select: all-tied scores still select the budget");
        bool same = true;
        for (int k = 0; k < kTopPools; ++k) {
            if (f.pools_host[k] != k) { same = false; }
        }
        expect(same, "select: equal scores select the lower pool ids");
    }
    // Verification widths and lanes in one call: lane 0's three columns at n = 6006..6008 (sparse),
    // lane 1 dense.
    Fixture v;
    for (std::uint32_t seed = 51; seed < 59; ++seed) {
        v = Fixture(2, 3, {6008, 300}, seed);
        v.fill_pooled_random(seed);
        v.run_select(stream);
        v.read_cache();
        bool gaps_ok = true;
        for (int c = 0; c < v.batch * v.width; ++c) {
            const int pools_n = (v.positions_host[c] + 1) / kPool;
            if (pools_n <= kTopPools) { continue; }
            std::vector<double> scores(pools_n);
            for (int b = 0; b < pools_n; ++b) { scores[b] = score_reference(v, c, b); }
            std::vector<double> sorted(scores);
            std::sort(sorted.begin(), sorted.end(), std::greater<double>());
            if (sorted[kTopPools - 1] - sorted[kTopPools] <=
                1.0e-4 * std::max(std::abs(sorted.front()), 1.0e-30)) {
                gaps_ok = false;
            }
        }
        if (gaps_ok) { break; }
        if (seed == 58) { v = Fixture(); }
    }
    if (v.batch != 0) {
        v.read_cache();
        bool same = true;
        for (int c = 0; c < v.batch * v.width; ++c) {
            const int pools_n = (v.positions_host[c] + 1) / kPool;
            if (pools_n <= kTopPools) {
                if (v.counts_host[c] != -1) { same = false; }
                continue;
            }
            if (v.counts_host[c] != kTopPools) { same = false; }
            std::vector<double> scores(pools_n);
            for (int b = 0; b < pools_n; ++b) { scores[b] = score_reference(v, c, b); }
            const auto want = top_pools(scores, kTopPools);
            for (int k = 0; k < kTopPools; ++k) {
                if (v.pools_host[static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(c) * kTopPools] != want[k]) {
                    same = false;
                }
            }
        }
        expect(same, "select: verification widths and lanes select the exact top-512 pools");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------- column invariance

void test_invariance() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // The shared column is flat column 0 of every fixture: same position, query stream, weights
    // and cache bytes, in a decode call, a two-column verification call and beside another lane.
    Fixture one(1, 1, {6000}, 61);
    one.fill_pooled_random(61);
    one.run_select(stream);
    one.read_cache();
    Fixture pair(1, 2, {6001, 6002}, 61);
    pair.fill_pooled_random(61);
    pair.run_select(stream);
    pair.read_cache();
    Fixture lanes(2, 1, {6000, 4000}, 61);
    lanes.fill_pooled_random(61);
    lanes.run_select(stream);
    lanes.read_cache();
    bool same =
        one.counts_host[0] == pair.counts_host[0] && one.counts_host[0] == lanes.counts_host[0];
    for (int k = 0; same && k < kTopPools; ++k) {
        same = one.pools_host[k] == pair.pools_host[k] && one.pools_host[k] == lanes.pools_host[k];
    }
    // The shared column's pooled keys are the same bytes too.
    const auto pk_one        = one.pooled_key(0, 1499);
    const auto pk_pair       = pair.pooled_key(0, 1499);
    const auto pk_lanes      = lanes.pooled_key(0, 1499);
    same                     = same && pk_one == pk_pair && pk_one == pk_lanes;
    expect(same,
           "select: a column's selection bits do not depend on the call's other columns or lanes");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main() {
    try {
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no usable CUDA device\n");
            return 77;
        }
        test_update();
        test_wide_and_rollback();
        test_select();
        test_invariance();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_dsa_indexer_test: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("infernix_dsa_indexer_test: all checks passed\n");
    return 0;
}
