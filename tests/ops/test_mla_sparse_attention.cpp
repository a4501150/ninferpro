// Sparse multi-head latent attention (include/infernix/ops/mla_sparse_attention.h) against
// independent FP64 oracles (docs/maintainer/op-development.md section 6), at the registered
// geometry (64 query heads, head_dim 256, latent_dim 512):
//
//   - mla_latent_append: the stored latent rows, bit for bit, at their paged positions, also when
//     a sequence is split into several calls; untouched pages stay untouched. The INT8-G64 cache
//     stores the exact oracle's codes and scales (a host encoder of the codec).
//   - mla_absorbed_q_projection and mla_latent_output_projection: FP64 dots of the BF16 inputs
//     with one BF16 output rounding, under a pointwise criterion; a column's bits do not depend
//     on the call's other columns.
//   - mla_sparse_attention, dense mode: FP64 stable softmax attention of every head over the
//     causal set 0..p, then the FP64 output projection, at decode, verification and multi-lane
//     widths, across page boundaries and the 2048-token budget.
//   - mla_sparse_attention, sparse mode: the same oracle over the call's selected token set (the
//     union of the test's selected pools and the visible incomplete tail), at n just above the
//     dense boundary and at long contexts, with automatic and fixed splits.
//   - column invariance: a column's output bits are the same alone, beside other lanes and inside
//     a wider verification call (both envelope modes).
//   - an INT8-G64 cache: the same oracles over the latents decoded with their stored scales.
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "infernix/ops/mla_sparse_attention.h"
#include "ops/host_parallel.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ops = infernix::ops;
namespace t   = infernix::test;
using infernix::DeviceBuffer;
using infernix::DType;
using infernix::Tensor;

constexpr int kHeads    = 64;
constexpr int kHeadDim  = 256;
constexpr int kLatent   = 512;
constexpr int kPage     = 64;
constexpr int kPool     = 4;
constexpr int kTopPools = 512;
constexpr float kScale  = 1.0F / 16.0F;

const ops::MlaAttentionGeometry kGeometry{kHeads, kHeadDim, kLatent};

// One BF16 rounding of an FP32-accumulated dot of BF16 inputs, plus FP32 reduction noise.
constexpr t::PointwiseCriterion kProjCriterion{/*absolute*/ 2.0e-3, /*relative*/ 3.0e-3};
// BF16 attention output over FP32 scores, softmax and latent sums (the softmax attention suite's
// bf16-decode profile).
constexpr t::ReductionCriterion kAttentionCriterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-2,
    /*gross_relative_to_max_reference*/ 5.0e-3,
};

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

std::uint16_t to_bf16(float f) { return t::f32_to_bf16(f); }

float from_bf16(std::uint16_t h) { return t::bf16_to_f32(h); }

// FP32 to FP16 bits, round to nearest even (normal, subnormal and overflow ranges).
std::uint16_t to_fp16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    const std::uint32_t exp  = (bits >> 23) & 0xffU;
    std::uint32_t mantissa   = bits & 0x007fffffU;
    const int half_exp       = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000U;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1U << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1U << shift) - 1U);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1U) != 0U)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffU;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000U || (tail == 0x1000U && (half_mantissa & 1U) != 0U)) {
        if (++half_mantissa == 0x400U) {
            half_mantissa = 0;
            if (++rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float from_fp16(std::uint16_t bits) {
    const int exp      = (bits >> 10) & 0x1f;
    const int mantissa = bits & 0x03ff;
    const float magnitude =
        exp == 0 ? std::ldexp(static_cast<float>(mantissa), -24)
                 : std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F, exp - 15);
    return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
}

constexpr int kGroup  = 64;
constexpr int kGroups = kLatent / kGroup;

// The INT8-G64 codec of one latent row (kv_cache_append.h): per group of 64, the FP16 scale of
// max |x| / 127 and the RNE codes of x times the FP32 reciprocal of the scale, clamped to 127.
void encode_int8_row(const std::uint16_t* row, std::int8_t* codes, std::uint16_t* scales) {
    for (int g = 0; g < kGroups; ++g) {
        float a = 0.0F;
        for (int i = 0; i < kGroup; ++i) {
            a = std::max(a, std::fabs(from_bf16(row[g * kGroup + i])));
        }
        scales[g]       = to_fp16(a / 127.0F);
        const float s   = from_fp16(scales[g]);
        const float inv = s == 0.0F ? 0.0F : 1.0F / s;
        for (int i = 0; i < kGroup; ++i) {
            const float x = from_bf16(row[g * kGroup + i]) * inv;
            codes[g * kGroup + i] =
                s == 0.0F
                    ? 0
                    : static_cast<std::int8_t>(std::clamp(std::nearbyint(x), -127.0F, 127.0F));
        }
    }
}

// Fills host floats, rounds them to the BF16 grid, and returns the BF16 bits.
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

/// One batched fixture: B lanes, per-lane contexts, scattered page tables, a materialized latent
/// plane (BF16, or its INT8-G64 encoding) and a hand-built selected pool set (or none: dense).
struct Fixture {
    int batch = 0, width = 0;
    bool int8 = false;
    std::vector<int> contexts;
    int max_context = 0, pages_per_row = 0;
    DeviceBuffer latents_buf, pages_buf, scales_buf, w_uk_buf, w_uv_buf, qa_buf, ol_buf, out_buf;
    DeviceBuffer tables_buf, rows_buf, positions_buf, pools_buf, counts_buf;
    ops::MlaSparseBatch batch_desc;
    ops::MlaLatentBatchView cache;
    ops::MlaSelectedPools selected{};
    Tensor pools_tensor, counts_tensor;

    Tensor latents_view() const {
        return Tensor(latents_buf.p, DType::BF16, {kLatent, width, batch});
    }

    std::vector<std::int32_t> positions_host, tables_host, pools_host, counts_host;
    std::vector<std::uint16_t> pages_host, w_uk_host, w_uv_host, latents_host;
    std::vector<std::int8_t> codes_host; // INT8-G64: the stored codes and FP16 scale bits
    std::vector<std::uint16_t> scales_host;
    Tensor q_tensor, qa_tensor, w_uk_tensor, w_uv_tensor, ol_tensor, out_tensor;

    /// The physical cache row of lane b's position.
    std::size_t row_of(int b, int position) const {
        return static_cast<std::size_t>(tables_host[position / kPage + b * pages_per_row]) * kPage +
               position % kPage;
    }

    /// Latent dimension d of a cache row as readers see it: BF16 as stored, INT8 decoded.
    double cached(std::size_t row, int d) const {
        const std::size_t i = row * kLatent + d;
        if (!int8) { return from_bf16(pages_host[i]); }
        return static_cast<double>(codes_host[i]) *
               from_fp16(scales_host[row * kGroups + d / kGroup]);
    }

    Fixture(int batch_, int width_, const std::vector<int>& contexts_, bool with_selection,
            std::uint32_t seed, bool int8_ = false) {
        batch             = batch_;
        width             = width_;
        int8              = int8_;
        contexts          = contexts_;
        max_context       = *std::max_element(contexts.begin(), contexts.end());
        pages_per_row     = (max_context + kPage - 1) / kPage;
        const int columns = batch * width;

        // The call's latent rows, BF16.
        latents_host =
            bf16_inputs(static_cast<std::size_t>(kLatent) * width * batch, seed, -1.0F, 1.0F);
        latents_buf = upload(latents_host);
        // The cache plane: fixture latents in [0.002, 0.01] (finite, well-scaled), scattered pages.
        pages_host = bf16_inputs(static_cast<std::size_t>(kLatent) * kPage * pages_per_row,
                                 seed + 1, 0.002F, 0.01F);
        if (int8) {
            const std::size_t rows = static_cast<std::size_t>(kPage) * pages_per_row;
            codes_host.resize(rows * kLatent);
            scales_host.resize(rows * kGroups);
            for (std::size_t r = 0; r < rows; ++r) {
                encode_int8_row(&pages_host[r * kLatent], &codes_host[r * kLatent],
                                &scales_host[r * kGroups]);
            }
            pages_buf = DeviceBuffer(codes_host.size());
            pages_buf.copy_from_host(codes_host.data(), pages_buf.bytes);
            scales_buf = upload(scales_host);
        } else {
            pages_buf = upload(pages_host);
        }
        w_uk_host  = bf16_inputs(static_cast<std::size_t>(kHeads) * kLatent * kHeadDim, seed + 2,
                                 -0.05F, 0.05F);
        w_uk_buf   = upload(w_uk_host);
        w_uv_host  = bf16_inputs(static_cast<std::size_t>(kHeads) * kHeadDim * kLatent, seed + 3,
                                 -0.05F, 0.05F);
        w_uv_buf   = upload(w_uv_host);

        std::vector<std::int32_t> tables(static_cast<std::size_t>(pages_per_row) * batch);
        for (int r = 0; r < batch; ++r) {
            for (int p = 0; p < pages_per_row; ++p) {
                tables[static_cast<std::size_t>(r) * pages_per_row + p] = static_cast<std::int32_t>(
                    (static_cast<std::int64_t>(p) * 7919 + 13) % pages_per_row);
            }
        }
        tables_host = tables;
        tables_buf  = upload_i32(tables);
        std::vector<std::int32_t> rows(batch);
        std::iota(rows.begin(), rows.end(), 0);
        rows_buf = upload_i32(rows);
        std::vector<std::int32_t> positions(columns);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < width; ++j) { positions[b * width + j] = contexts[b] - width + j; }
        }
        positions_host = positions;
        positions_buf  = upload_i32(positions);

        batch_desc = ops::MlaSparseBatch{
            .block_tables = Tensor(tables_buf.p, DType::I32, {pages_per_row, batch}),
            .table_rows   = Tensor(rows_buf.p, DType::I32, {batch}),
            .positions    = Tensor(positions_buf.p, DType::I32, {columns}),
            .batch        = batch,
            .width        = width};
        cache = ops::MlaLatentBatchView{.pages = Tensor(pages_buf.p, int8 ? DType::I8 : DType::BF16,
                                                        {kLatent, kPage, 1, pages_per_row})};
        if (int8) {
            cache.scales = Tensor(scales_buf.p, DType::FP16, {kGroups, kPage, 1, pages_per_row});
        }

        qa_buf      = DeviceBuffer(static_cast<std::size_t>(kLatent) * kHeads * columns * 2);
        ol_buf      = DeviceBuffer(static_cast<std::size_t>(kLatent) * kHeads * columns * 2);
        out_buf     = DeviceBuffer(static_cast<std::size_t>(kHeadDim) * kHeads * columns * 2);
        qa_tensor   = Tensor(qa_buf.p, DType::BF16, {kLatent, kHeads, columns});
        ol_tensor   = Tensor(ol_buf.p, DType::BF16, {kLatent, kHeads, columns});
        out_tensor  = Tensor(out_buf.p, DType::BF16, {kHeadDim, kHeads, columns});
        w_uk_tensor = Tensor(w_uk_buf.p, DType::BF16, {kHeadDim, kLatent, kHeads});
        w_uv_tensor = Tensor(w_uv_buf.p, DType::BF16, {kLatent, kHeadDim, kHeads});

        if (with_selection) {
            // Per column: the top kTopPools of random per-pool priorities, ties to the lower pool
            // id, ids ascending - the indexer's selection semantics without coupling the suites.
            const int total_columns = columns;
            std::vector<std::int32_t> pools(static_cast<std::size_t>(kTopPools) * total_columns);
            std::vector<std::int32_t> counts(total_columns);
            std::mt19937 rng(seed + 7);
            for (int c = 0; c < total_columns; ++c) {
                const int n              = positions[c] + 1;
                const int complete_pools = n / kPool;
                counts[c]                = std::min(kTopPools, complete_pools);
                std::vector<std::pair<float, int>> priority(complete_pools);
                for (int p = 0; p < complete_pools; ++p) {
                    priority[p] = {std::uniform_real_distribution<float>(0.0F, 1.0F)(rng), p};
                }
                std::stable_sort(priority.begin(), priority.end(),
                                 [](const auto& a, const auto& b) { return a.first > b.first; });
                std::vector<int> chosen;
                chosen.reserve(counts[c]);
                for (int k = 0; k < counts[c]; ++k) { chosen.push_back(priority[k].second); }
                std::sort(chosen.begin(), chosen.end());
                for (int k = 0; k < counts[c]; ++k) {
                    pools[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * kTopPools] =
                        chosen[k];
                }
            }
            pools_host    = pools;
            counts_host   = counts;
            pools_buf     = upload_i32(pools);
            counts_buf    = upload_i32(counts);
            pools_tensor  = Tensor(pools_buf.p, DType::I32, {kTopPools, total_columns});
            counts_tensor = Tensor(counts_buf.p, DType::I32, {total_columns});
            selected      = ops::MlaSelectedPools{
                .pools = &pools_tensor, .counts = &counts_tensor, .pool = kPool};
        }
    }
};

// --------------------------------------------------------------------------------------- append

// The fixture's cache as the device stores it: BF16 bits, or INT8 codes followed by scale bits.
std::vector<std::uint8_t> stored_cache(const Fixture& f) {
    std::vector<std::uint8_t> bytes(f.pages_buf.bytes + (f.int8 ? f.scales_buf.bytes : 0));
    f.pages_buf.copy_to_host(bytes.data(), f.pages_buf.bytes);
    if (f.int8) { f.scales_buf.copy_to_host(bytes.data() + f.pages_buf.bytes, f.scales_buf.bytes); }
    return bytes;
}

void test_append(bool int8) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const std::string label = int8 ? "append int8-g64: " : "append: ";
    // B = 2 lanes of different lengths: every written row holds its latent row (copied, or the
    // codec's codes and scales) at its position and every other byte keeps its fixture value.
    Fixture f(2, 5, {300, 130}, false, 101, int8);
    // One all-zero group (scale 0, codes 0) in the first column.
    std::fill_n(f.latents_host.begin(), kGroup, std::uint16_t{0});
    f.latents_buf.copy_from_host(f.latents_host.data(), f.latents_buf.bytes);
    auto want_pages  = f.pages_host;
    auto want_codes  = f.codes_host;
    auto want_scales = f.scales_host;
    for (int b = 0; b < f.batch; ++b) {
        for (int j = 0; j < f.width; ++j) {
            const std::size_t row = f.row_of(b, f.positions_host[b * f.width + j]);
            const std::uint16_t* src =
                &f.latents_host[static_cast<std::size_t>(j) * kLatent +
                                static_cast<std::size_t>(b) * kLatent * f.width];
            if (int8) {
                encode_int8_row(src, &want_codes[row * kLatent], &want_scales[row * kGroups]);
            } else {
                std::copy_n(src, kLatent, &want_pages[row * kLatent]);
            }
        }
    }
    ops::mla_latent_append(f.latents_view(), f.batch_desc, f.cache, stream);
    t::cuda_synchronize(stream);
    if (int8) {
        const auto codes  = t::from_device<std::int8_t>(f.pages_buf, want_codes.size());
        const auto scales = t::from_device<std::uint16_t>(f.scales_buf, want_scales.size());
        expect(codes == want_codes, label + "stored codes equal the codec's at their positions");
        expect(scales == want_scales, label + "stored scales equal the codec's at their positions");
    } else {
        const auto got = t::from_device<std::uint16_t>(f.pages_buf, want_pages.size());
        expect(got == want_pages, label + "stored latent rows at their paged positions");
    }
    // Splitting one sequence's append into calls of 5 + 3 columns stores the same bytes.
    Fixture one(1, 8, {308}, false, 101, int8);
    ops::mla_latent_append(one.latents_view(), one.batch_desc, one.cache, stream);
    t::cuda_synchronize(stream);
    Fixture split(1, 8, {308}, false, 101, int8);
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
    ops::mla_latent_append(split.latents_view().slice(1, 0, 5), call_a, split.cache, stream);
    ops::mla_latent_append(split.latents_view().slice(1, 5, 3), call_b, split.cache, stream);
    t::cuda_synchronize(stream);
    expect(stored_cache(split) == stored_cache(one), label + "split calls store the same bytes");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------------- projections

void test_projections() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int columns = 33;
    Fixture f(1, columns, {columns}, false, 202);
    const auto q_bits =
        bf16_inputs(static_cast<std::size_t>(kHeadDim) * kHeads * columns, 9, -1.0F, 1.0F);
    DeviceBuffer q_buf = upload(q_bits);
    const Tensor q_tensor(q_buf.p, DType::BF16, {kHeadDim, kHeads, columns});
    ops::mla_absorbed_q_projection(q_tensor, f.w_uk_tensor, f.qa_tensor, stream);
    t::cuda_synchronize(stream);
    // The output projection consumes the qa result as its stored BF16 latent sums.
    CUDA_CHECK(cudaMemcpy(f.ol_buf.p, f.qa_buf.p, f.ol_buf.bytes, cudaMemcpyDeviceToDevice));
    ops::mla_latent_output_projection(f.ol_tensor, f.w_uv_tensor, f.out_tensor, stream);
    t::cuda_synchronize(stream);

    // FP64 oracle of the absorbed query projection, parallel over (head, column).
    std::vector<double> qa_ref(static_cast<std::size_t>(kLatent) * kHeads * columns);
    t::parallel_ranges(
        static_cast<std::int64_t>(kHeads) * columns, t::host_thread_count(),
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t hc = begin; hc < end; ++hc) {
                const int h = static_cast<int>(hc / columns), c = static_cast<int>(hc % columns);
                for (int d = 0; d < kLatent; ++d) {
                    double acc = 0.0;
                    for (int d0 = 0; d0 < kHeadDim; ++d0) {
                        acc += from_bf16(
                                   f.w_uk_host[d0 + static_cast<std::size_t>(d) * kHeadDim +
                                               static_cast<std::size_t>(h) * kHeadDim * kLatent]) *
                               from_bf16(q_bits[d0 + static_cast<std::size_t>(h) * kHeadDim +
                                                static_cast<std::size_t>(c) * kHeadDim * kHeads]);
                    }
                    qa_ref[d + static_cast<std::size_t>(h) * kLatent +
                           static_cast<std::size_t>(c) * kLatent * kHeads] = acc;
                }
            }
        });
    g_failures +=
        t::verify_pointwise("absorbed q projection", t::from_device_bf16(f.qa_buf, qa_ref.size()),
                            qa_ref, kProjCriterion);

    // The output projection: the stored BF16 latent sums in, FP64 dots out.
    const auto ol_bits = t::from_device<std::uint16_t>(f.qa_buf, qa_ref.size());
    std::vector<double> o_ref(static_cast<std::size_t>(kHeadDim) * kHeads * columns);
    t::parallel_ranges(
        static_cast<std::int64_t>(kHeads) * columns, t::host_thread_count(),
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t hc = begin; hc < end; ++hc) {
                const int h = static_cast<int>(hc / columns), c = static_cast<int>(hc % columns);
                for (int e = 0; e < kHeadDim; ++e) {
                    double acc = 0.0;
                    for (int d = 0; d < kLatent; ++d) {
                        acc += from_bf16(
                                   f.w_uv_host[d + static_cast<std::size_t>(e) * kLatent +
                                               static_cast<std::size_t>(h) * kHeadDim * kLatent]) *
                               from_bf16(ol_bits[d + static_cast<std::size_t>(h) * kLatent +
                                                 static_cast<std::size_t>(c) * kLatent * kHeads]);
                    }
                    o_ref[e + static_cast<std::size_t>(h) * kHeadDim +
                          static_cast<std::size_t>(c) * kHeadDim * kHeads] = acc;
                }
            }
        });
    g_failures +=
        t::verify_pointwise("latent output projection",
                            t::from_device_bf16(f.out_buf, o_ref.size()), o_ref, kProjCriterion);

    // Column invariance: the same column's bits do not depend on the call's other columns.
    Fixture one(1, 1, {1}, false, 202);
    std::vector<std::uint16_t> q_one_bits(static_cast<std::size_t>(kHeadDim) * kHeads);
    for (int h = 0; h < kHeads; ++h) {
        for (int d = 0; d < kHeadDim; ++d) {
            q_one_bits[d + static_cast<std::size_t>(h) * kHeadDim] =
                q_bits[d + static_cast<std::size_t>(h) * kHeadDim + 7 * kHeadDim * kHeads];
        }
    }
    DeviceBuffer q_one = upload(q_one_bits);
    const Tensor q_one_tensor(q_one.p, DType::BF16, {kHeadDim, kHeads, 1});
    ops::mla_absorbed_q_projection(q_one_tensor, one.w_uk_tensor, one.qa_tensor, stream);
    t::cuda_synchronize(stream);
    const auto qa_wide = t::from_device<std::uint16_t>(f.qa_buf, qa_ref.size());
    const auto qa_one =
        t::from_device<std::uint16_t>(one.qa_buf, static_cast<std::size_t>(kLatent) * kHeads);
    bool same = true;
    for (int h = 0; h < kHeads && same; ++h) {
        for (int d = 0; d < kLatent; ++d) {
            if (qa_wide[d + static_cast<std::size_t>(h) * kLatent + 7 * kLatent * kHeads] !=
                qa_one[d + static_cast<std::size_t>(h) * kLatent]) {
                same = false;
            }
        }
    }
    expect(same, "absorbed q projection: column bits do not depend on the other columns");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ------------------------------------------------------------------------------------ attention

// The FP64 oracle of one (column, head): stable softmax over the column's attended token set, the
// FP64 latent sums, and the FP64 output projection. qa enters as its BF16 bits, the cached latents
// as readers decode them, the weights as stored.
struct ColumnOracle {
    // The column's attended token positions, ascending: the selected pools' tokens and the
    // visible incomplete tail, or the causal set 0..n-1 in dense mode.
    static std::vector<int> attended_set(const Fixture& f, int column) {
        const int n = f.positions_host[column] + 1;
        if (f.selected.pools == nullptr) {
            std::vector<int> set(n);
            std::iota(set.begin(), set.end(), 0);
            return set;
        }
        const int count     = f.counts_host[column];
        const int tail_base = n - n % kPool;
        std::vector<int> set;
        set.reserve(static_cast<std::size_t>(count) * kPool + n % kPool);
        for (int i = 0; i < count * kPool; ++i) {
            set.push_back(f.pools_host[i / kPool + static_cast<std::size_t>(column) * kTopPools] *
                              kPool +
                          i % kPool);
        }
        for (int p = tail_base; p < n; ++p) { set.push_back(p); }
        return set;
    }

    static std::vector<double> run_one(const Fixture& f, int column, int head,
                                       const std::vector<int>& set,
                                       const std::vector<std::uint16_t>& qa_bits) {
        const int n_set = static_cast<int>(set.size());
        // The attended latents as readers see them (BF16 bits, or INT8 codes times their scales),
        // through the page table.
        std::vector<double> latent(static_cast<std::size_t>(n_set) * kLatent);
        for (int j = 0; j < n_set; ++j) {
            const std::size_t row = f.row_of(column / f.width, set[j]);
            for (int d = 0; d < kLatent; ++d) {
                latent[static_cast<std::size_t>(j) * kLatent + d] = f.cached(row, d);
            }
        }
        // Scores, then a stable FP64 softmax over the set.
        std::vector<double> score(n_set);
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_set; ++j) {
            double dot = 0.0;
            for (int d = 0; d < kLatent; ++d) {
                dot += from_bf16(qa_bits[d + static_cast<std::size_t>(head) * kLatent +
                                         static_cast<std::size_t>(column) * kLatent * kHeads]) *
                       latent[static_cast<std::size_t>(j) * kLatent + d];
            }
            score[j] = kScale * dot;
            m        = std::max(m, score[j]);
        }
        double l = 0.0;
        for (int j = 0; j < n_set; ++j) { l += std::exp(score[j] - m); }
        std::vector<double> ol(kLatent, 0.0);
        for (int j = 0; j < n_set; ++j) {
            const double w = std::exp(score[j] - m) / l;
            for (int d = 0; d < kLatent; ++d) {
                ol[d] += w * latent[static_cast<std::size_t>(j) * kLatent + d];
            }
        }
        std::vector<double> out(kHeadDim);
        for (int e = 0; e < kHeadDim; ++e) {
            double acc = 0.0;
            for (int d = 0; d < kLatent; ++d) {
                acc += from_bf16(f.w_uv_host[d + static_cast<std::size_t>(e) * kLatent +
                                             static_cast<std::size_t>(head) * kHeadDim * kLatent]) *
                       ol[d];
            }
            out[e] = acc;
        }
        return out;
    }
};

// Per-column query streams, so the same column has the same queries in every batch composition.
std::vector<std::uint16_t> q_bits_for(int columns) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kHeadDim) * kHeads * columns);
    for (int c = 0; c < columns; ++c) {
        const auto one =
            bf16_inputs(static_cast<std::size_t>(kHeadDim) * kHeads, 9 + c, -1.0F, 1.0F);
        std::copy(one.begin(), one.end(),
                  bits.begin() + static_cast<std::ptrdiff_t>(c) * kHeadDim * kHeads);
    }
    return bits;
}

// Projects the fixture's queries with the Op and returns their BF16 bits.
std::vector<std::uint16_t> prepare_qa(Fixture& f, cudaStream_t stream) {
    const int columns  = f.batch * f.width;
    DeviceBuffer q_buf = upload(q_bits_for(columns));
    const Tensor q_tensor(q_buf.p, DType::BF16, {kHeadDim, kHeads, columns});
    ops::mla_absorbed_q_projection(q_tensor, f.w_uk_tensor, f.qa_tensor, stream);
    t::cuda_synchronize(stream);
    return t::from_device<std::uint16_t>(f.qa_buf,
                                         static_cast<std::size_t>(kLatent) * kHeads * columns);
}

// Runs the attention Op and returns the output bits.
std::vector<std::uint16_t> run_attention(Fixture& f, ops::MlaAttentionExecutionEnvelope envelope,
                                         cudaStream_t stream) {
    const int columns = f.batch * f.width;
    const std::size_t workspace_bytes =
        ops::mla_sparse_attention_workspace_bytes(kGeometry, columns, f.max_context, envelope);
    DeviceBuffer workspace(workspace_bytes);
    f.out_buf.fill(0x57);
    ops::mla_sparse_attention(f.qa_tensor, f.w_uv_tensor, f.cache, f.batch_desc, f.selected,
                              kGeometry, kScale, f.max_context, envelope, workspace.p,
                              workspace_bytes, f.out_tensor, stream);
    t::cuda_synchronize(stream);
    return t::from_device<std::uint16_t>(f.out_buf,
                                         static_cast<std::size_t>(kHeadDim) * kHeads * columns);
}

// Compares every column's output against the FP64 oracle over the oracle's own attended set.
void verify_attention(const Fixture& f, const std::vector<std::uint16_t>& qa_bits,
                      const std::vector<std::uint16_t>& got_bits, const char* what) {
    const int columns = f.batch * f.width;
    t::parallel_ranges(
        static_cast<std::int64_t>(columns) * kHeads, t::host_thread_count(),
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t hc = begin; hc < end; ++hc) {
                const int c = static_cast<int>(hc / kHeads), h = static_cast<int>(hc % kHeads);
                const std::vector<int> set    = ColumnOracle::attended_set(f, c);
                const std::vector<double> ref = ColumnOracle::run_one(f, c, h, set, qa_bits);
                std::vector<double> got(kHeadDim);
                for (int e = 0; e < kHeadDim; ++e) {
                    got[e] = from_bf16(got_bits[e + static_cast<std::size_t>(h) * kHeadDim +
                                                static_cast<std::size_t>(c) * kHeadDim * kHeads]);
                }
                const std::string label = std::string(what) + ": column " + std::to_string(c) +
                                          " head " + std::to_string(h);
                g_failures += t::verify_reduction(label, got, ref, kAttentionCriterion);
            }
        });
}

void test_attention_dense() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Decode at the dense boundary, a page-boundary verification call, and lanes with different
    // contexts. Every column attends 0..p.
    {
        Fixture f(1, 1, {2051}, false, 303);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {}, stream);
        verify_attention(f, qa_bits, got, "dense decode n=2051");
    }
    {
        Fixture f(1, 3, {2053}, false, 304);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {}, stream);
        verify_attention(f, qa_bits, got, "dense verification w=3");
    }
    {
        Fixture f(3, 1, {63, 64, 300}, false, 305);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {}, stream);
        verify_attention(f, qa_bits, got, "dense lanes");
    }
    {
        // Fixed splits (the model's prefill envelope) against the same oracle.
        Fixture f(1, 4, {512}, false, 306);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {.splits = 1}, stream);
        verify_attention(f, qa_bits, got, "dense fixed splits");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_attention_sparse() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // One lane at n = 6000 (1500 complete pools, 512 selected plus the tail), a sparse-boundary
    // lane beside it, and a fixed-splits call.
    {
        Fixture f(1, 1, {6000}, true, 307);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {}, stream);
        verify_attention(f, qa_bits, got, "sparse decode n=6000");
    }
    {
        Fixture f(2, 1, {6000, 2052}, true, 308);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {}, stream);
        verify_attention(f, qa_bits, got, "sparse lanes");
    }
    {
        Fixture f(1, 4, {6000}, true, 309);
        const auto qa_bits = prepare_qa(f, stream);
        const auto got     = run_attention(f, {.splits = 3}, stream);
        verify_attention(f, qa_bits, got, "sparse fixed splits");
    }
    // Column invariance: a column's bits are the same alone, inside a wider verification call and
    // beside another lane, in both envelope modes.
    for (const auto& envelope :
         std::vector<ops::MlaAttentionExecutionEnvelope>{{}, {.splits = 2}}) {
        // The shared column is flat column 0 of every fixture (per-column query streams), at the
        // same position 6000 with the same fixture weights and cache.
        Fixture one(1, 1, {6000}, true, 310);
        const auto qa_one        = prepare_qa(one, stream);
        const auto got_one       = run_attention(one, envelope, stream);
        const auto got_one_again = run_attention(one, envelope, stream);
        expect(got_one == got_one_again, "attention: the same call twice gives the same bits");
        Fixture pair(1, 2, {6001, 6002}, true, 310);
        const auto qa_pair  = prepare_qa(pair, stream);
        const auto got_pair = run_attention(pair, envelope, stream);
        Fixture lanes(2, 1, {6000, 4000}, true, 310);
        const auto qa_lanes          = prepare_qa(lanes, stream);
        const auto got_lanes         = run_attention(lanes, envelope, stream);
        const std::size_t per_column = static_cast<std::size_t>(kHeadDim) * kHeads;
        bool same     = got_one.size() == per_column && got_pair.size() == 2 * per_column &&
                        got_lanes.size() == 2 * per_column;
        int diff_pair = -1, diff_lanes = -1;
        for (std::size_t i = 0; i < per_column; ++i) {
            if (diff_pair < 0 && got_one[i] != got_pair[i]) { diff_pair = static_cast<int>(i); }
            if (diff_lanes < 0 && got_one[i] != got_lanes[i]) { diff_lanes = static_cast<int>(i); }
        }
        for (std::size_t i = 0; same && i < per_column; ++i) {
            if (got_one[i] != got_pair[i] || got_one[i] != got_lanes[i]) { same = false; }
        }
        expect(same,
               "attention: a column's bits do not depend on the call's other columns or lanes");
        expect(std::equal(qa_one.begin(), qa_one.end(), qa_pair.begin()) &&
                   std::equal(qa_one.begin(), qa_one.end(), qa_lanes.begin()),
               "column invariance: the shared column's queries match across calls");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// An INT8-G64 cache: the oracle over the decoded latents, dense and sparse, automatic and fixed
// splits, and a column's bits beside another lane.
void test_attention_int8() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        Fixture f(1, 3, {2053}, false, 311, true);
        const auto qa_bits = prepare_qa(f, stream);
        verify_attention(f, qa_bits, run_attention(f, {}, stream), "int8 dense verification w=3");
    }
    {
        Fixture f(3, 1, {63, 64, 300}, false, 312, true);
        const auto qa_bits = prepare_qa(f, stream);
        verify_attention(f, qa_bits, run_attention(f, {.splits = 1}, stream),
                         "int8 dense lanes, fixed splits");
    }
    {
        Fixture f(2, 1, {6000, 2052}, true, 313, true);
        const auto qa_bits = prepare_qa(f, stream);
        verify_attention(f, qa_bits, run_attention(f, {}, stream), "int8 sparse lanes");
    }
    {
        Fixture f(1, 4, {6000}, true, 314, true);
        const auto qa_bits = prepare_qa(f, stream);
        verify_attention(f, qa_bits, run_attention(f, {.splits = 3}, stream),
                         "int8 sparse fixed splits");
    }
    {
        Fixture one(1, 1, {6000}, true, 315, true);
        (void)prepare_qa(one, stream);
        const auto got_one = run_attention(one, {}, stream);
        Fixture lanes(2, 1, {6000, 4000}, true, 315, true);
        (void)prepare_qa(lanes, stream);
        const auto got_lanes = run_attention(lanes, {}, stream);
        expect(std::equal(got_one.begin(), got_one.end(), got_lanes.begin()),
               "int8 attention: a column's bits do not depend on the call's other lanes");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main() {
    try {
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no usable CUDA device\n");
            return 77;
        }
        test_append(false);
        test_append(true);
        test_projections();
        test_attention_dense();
        test_attention_sparse();
        test_attention_int8();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_mla_sparse_attention_test: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("infernix_mla_sparse_attention_test: all checks passed\n");
    return 0;
}
