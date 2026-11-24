// Glm5Next forward on the real converted artifact, against the FP64 reference forward
// (tools/glm5_next/reference.py) of the same checkpoint. INFERNIX_GLM_ARTIFACT names the artifact
// and INFERNIX_GLM_REFERENCE a directory holding two reference runs of one prompt: fp64/ (with
// --no-bf16-boundaries, the oracle) and bf16/ (the default, rounded like a BF16 runtime, which sets
// the floor), each the tool's --out directory; without both variables the test skips (77).
//
// One prefill of every token is compared with the oracle at every layer (residual relative L2
// error), in its routed experts and in its logits (relative L2 error and argmax), within the floor.
// A second lane then prefills all but the last few tokens and decodes those one at a time; each
// decode step's logits must agree with the single prefill's at the same position. The banks stay in the artifact
// (the whole bank, 159 GiB, does not fit beside the desktop in this machine's RAM): every expert the
// reference or either lane routes is read into VRAM frames, repeating both lanes until their
// routing stays inside the frames, and every other record is one pinned stand-in.

#include "artifact/reader.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "models/glm5_next/execution/forward.h"
#include "models/glm5_next/execution/parameters.h"
#include "models/glm5_next/load.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace infernix;
namespace glm  = infernix::models::glm5_next;
namespace exec = infernix::models::glm5_next::execution;

namespace {

int failures = 0;

void expect(bool ok, const std::string& message) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
}

template <typename T>
std::vector<T> read_raw(const std::string& path, std::size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path);
    std::vector<T> out(count);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T)) throw std::runtime_error(path + " is short");
    return out;
}

std::vector<std::int32_t> read_tokens(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path);
    std::vector<std::int32_t> out;
    std::int32_t id = 0;
    while (file >> id) out.push_back(id);
    return out;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// sqrt(sum (a - b)^2 / sum b^2) over n values.
double relative_l2(const float* a, const float* b, std::size_t n) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

std::size_t argmax(const float* v, std::size_t n) {
    return static_cast<std::size_t>(std::max_element(v, v + n) - v);
}

// One reference run's outputs (tools/glm5_next/reference.py --out DIR).
struct ReferenceRun {
    ReferenceRun(const std::string& dir, std::size_t T, std::size_t V, std::size_t residual_values,
                 std::size_t routed_values)
        : logits(read_raw<float>(dir + "/logits.f32", T * V)),
          residuals(read_raw<float>(dir + "/residuals.f32", residual_values)),
          routed(read_raw<std::int32_t>(dir + "/routed.i32", routed_values)) {}
    std::vector<float> logits;        // [T, V]
    std::vector<float> residuals;     // [layers, T, S*H]
    std::vector<std::int32_t> routed; // [MoE layers, T, k]
};

// Share of the `rows` routed sets of k experts that are equal in a and b.
double route_share(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b, std::size_t rows,
                   std::size_t k) {
    std::size_t equal = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        equal += std::set<std::int32_t>(a.begin() + r * k, a.begin() + (r + 1) * k) ==
                 std::set<std::int32_t>(b.begin() + r * k, b.begin() + (r + 1) * k);
    }
    return static_cast<double>(equal) / static_cast<double>(rows);
}

// Reads every record of `used[l]` (MoE layer l's experts) from the artifact into one frame pool per
// layer, and fills each layer's frame table (frame or -1).
struct ExpertFrames {
    ExpertFrames(const glm::Model& model, const std::vector<std::vector<std::int32_t>>& used) {
        const auto& store = *model.expert_store();
        const std::uint32_t E = store.experts();
        std::vector<std::ifstream> files;
        for (const auto& path : store.files()) {
            files.emplace_back(path, std::ios::binary);
            if (!files.back()) throw std::runtime_error("cannot open " + path.string());
        }
        tables = DeviceBuffer(sizeof(std::int32_t) * E * used.size());
        std::vector<std::byte> record;
        for (std::uint32_t l = 0; l < used.size(); ++l) {
            const std::uint64_t pitch = model.expert_banks()[l].planes.record_stride;
            if (pitch != store.record_bytes(l)) throw std::runtime_error("bank stride differs from the store's");
            pools.emplace_back(std::max<std::size_t>(1, used[l].size()) * pitch);
            std::vector<std::int32_t> table(E, -1);
            record.resize(pitch);
            for (std::size_t f = 0; f < used[l].size(); ++f) {
                const auto e = static_cast<std::uint32_t>(used[l][f]);
                for (const auto& segment : store.segments(l, e)) {
                    auto& file = files.at(segment.file);
                    file.seekg(static_cast<std::streamoff>(segment.offset));
                    file.read(reinterpret_cast<char*>(record.data()) + segment.at, segment.bytes);
                    if (!file) throw std::runtime_error("short expert record read");
                }
                pools.back().copy_from_host(record.data(), pitch, f * pitch);
                table[e] = static_cast<std::int32_t>(f);
            }
            tables.copy_from_host(table.data(), table.size() * sizeof(std::int32_t), l * E * sizeof(std::int32_t));
            pitches.push_back(pitch);
        }
        // Every record outside the frames points at one pinned stand-in (a copy of the last record
        // read): a routing that leaves the reference's set then computes a wrong expert, which the
        // route and residual comparisons report, instead of reading a null record.
        void* stand_in = nullptr;
        if (cudaHostAlloc(&stand_in, record.size(), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            throw std::runtime_error("cannot pin the stand-in record");
        }
        std::memcpy(stand_in, record.data(), record.size());
        stand_in_record = stand_in;
        void* device_stand_in = nullptr;
        if (cudaHostGetDevicePointer(&device_stand_in, stand_in, 0) != cudaSuccess) {
            throw std::runtime_error("cannot map the stand-in record");
        }
        const std::vector<const void*> pointers(static_cast<std::size_t>(E) * used.size(), device_stand_in);
        host_tables = DeviceBuffer(sizeof(const void*) * pointers.size());
        host_tables.copy_from_host(pointers.data(), sizeof(const void*) * pointers.size());
    }
    ~ExpertFrames() {
        if (stand_in_record != nullptr) { (void)cudaFreeHost(stand_in_record); }
    }
    ExpertFrames(const ExpertFrames&)            = delete;
    ExpertFrames& operator=(const ExpertFrames&) = delete;
    void* stand_in_record = nullptr;
    std::vector<DeviceBuffer> pools;
    std::vector<std::uint64_t> pitches;
    DeviceBuffer tables, host_tables;
};

// KDA state for `rows` slots, latent and indexer pages for `rows` sequences of up to `context`
// tokens, and the experts of `frames` resident.
struct Harness {
    Harness(const exec::Parameters& parameters, DeviceContext& device, const ExpertFrames& frames,
            std::int32_t context, std::int32_t columns, std::int32_t rows)
        : config(parameters.model.config().text) {
        const auto& c = config;
        LayoutBuilder builder;
        const LinearAttentionStatePoolSpec spec{
            .layers         = c.kda_layers,
            .conv_channels  = static_cast<std::int32_t>(3 * c.kda.width()),
            .conv_width     = static_cast<std::int32_t>(c.kda.conv_kernel - 1),
            .value_heads    = static_cast<std::int32_t>(c.kda.heads),
            .value_head_dim = static_cast<std::int32_t>(c.kda.head_dim),
            .key_head_dim   = static_cast<std::int32_t>(c.kda.head_dim),
            .slot_count     = rows,
        };
        const auto layout = plan_linear_attention_state_pool(builder, spec);
        state_backing     = DeviceBuffer(builder.finish(256));
        state_backing.fill(0);
        kda = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing.p, state_backing.bytes}, layout);

        // Per MLA layer: the latent and the indexer's pooled keys; per layer and row the indexer's
        // recent ring of keys and gates.
        const auto latent = static_cast<std::int32_t>(c.mla.kv_lora);
        const auto id     = static_cast<std::int32_t>(c.indexer.head_dim);
        const auto pooled = id / static_cast<std::int32_t>(c.indexer.pool);
        KVPageGeometry geometry;
        for (std::uint32_t l = 0; l < c.mla_layers; ++l) {
            geometry.planes.push_back({DType::BF16, latent, 1});
            geometry.planes.push_back({DType::BF16, pooled, 1});
        }
        ring_elements = static_cast<std::size_t>(id) * 64 * rows;
        ring_backing  = DeviceBuffer(2 * 2 * ring_elements * c.mla_layers);
        ring_backing.fill(0);
        pages_per_row = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        LayoutBuilder kv_builder;
        const auto pool_layout = plan_device_kv_page_pool(
            kv_builder, {.page_group_count = static_cast<std::uint32_t>(pages_per_row * rows), .geometry = geometry});
        const auto table_layout = plan_kv_execution_tables(
            kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(pages_per_row), .table_rows = rows});
        kv_backing = DeviceBuffer(kv_builder.finish(256));
        kv_backing.fill(0);
        pool   = std::make_unique<DeviceKVPagePool>(DeviceSpan{kv_backing.p, kv_backing.bytes}, pool_layout);
        tables = std::make_unique<KVExecutionTablePool>(DeviceSpan{kv_backing.p, kv_backing.bytes}, table_layout, *pool);
        for (std::int32_t row = 0; row < rows; ++row) {
            auto reservation = pool->reserve(static_cast<std::uint32_t>(pages_per_row));
            if (!reservation) throw std::runtime_error("KV pool too small");
            std::vector<DeviceKVPageLease> pages;
            pages.reserve(static_cast<std::size_t>(pages_per_row));
            pool->materialize(*reservation, static_cast<std::uint32_t>(pages_per_row), pages);
            row_leases.push_back(tables->acquire(row));
            tables->publish(row_leases.back().handle(), 0, pages, device.stream);
            page_leases.push_back(std::move(pages));
            reservations.push_back(std::move(*reservation));
        }
        device.synchronize();

        const std::uint32_t moe_layers = c.moe_layers();

        exec::ForwardState state{.kda = kda.get()};
        exec::ForwardKV kv;
        kv.block_tables   = tables->matrix();
        std::size_t plane = 0;
        for (std::uint32_t l = 0; l < c.mla_layers; ++l) {
            exec::MlaLayerCache layer;
            layer.latent.pages        = pool->plane(plane++);
            auto* ring                = static_cast<std::uint16_t*>(ring_backing.p);
            layer.index.k_recent      = Tensor(ring + (2 * l) * ring_elements, DType::BF16,
                                               {static_cast<std::int32_t>(c.indexer.head_dim), 64, rows});
            layer.index.z_recent      = Tensor(ring + (2 * l + 1) * ring_elements, DType::BF16,
                                               {static_cast<std::int32_t>(c.indexer.head_dim), 64, rows});
            layer.index.pooled_pages  = pool->plane(plane++);
            kv.layers.push_back(layer);
        }
        exec::ForwardExperts experts;
        route_stride = static_cast<std::size_t>(c.moe.top_k) * columns;
        route_log    = DeviceBuffer(sizeof(std::int32_t) * route_stride * moe_layers);
        experts.route_log    = static_cast<std::int32_t*>(route_log.p);
        experts.route_stride = route_stride;
        for (std::uint32_t l = 0; l < moe_layers; ++l) {
            experts.frame_bases.push_back(static_cast<const std::uint8_t*>(frames.pools[l].p));
            experts.frame_pitches.push_back(frames.pitches[l]);
            experts.frames.push_back(static_cast<const std::int32_t*>(frames.tables.p) + l * c.moe.experts);
            experts.host_tables.push_back(static_cast<const std::uint8_t* const*>(frames.host_tables.p) +
                                          l * c.moe.experts);
        }
        experts.resident = true;
        work    = std::make_unique<WorkspaceArena>(exec::Forward::workspace_bytes(c, columns, context));
        forward = std::make_unique<exec::Forward>(parameters, device, *work, state, std::move(kv), std::move(experts),
                                                  context);
    }

    const glm::TextConfig& config;
    DeviceBuffer state_backing, ring_backing, kv_backing, route_log;
    std::size_t ring_elements = 0;
    std::size_t route_stride = 0;
    std::unique_ptr<LinearAttentionStatePool> kda;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::vector<DeviceKVPageReservation> reservations;
    std::vector<std::vector<DeviceKVPageLease>> page_leases;
    std::vector<KVExecutionRowLease> row_leases;
    std::int32_t pages_per_row = 0;
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<exec::Forward> forward;
};

// Device copies of one call's inputs: one sequence's tokens [first, first + width) in `row`.
struct Call {
    DeviceBuffer buffer;
    exec::ForwardBatch batch;
    std::vector<std::int32_t> host_slots;
};

Call make_call(const std::vector<std::int32_t>& tokens, std::int32_t first, std::int32_t width, std::int32_t row,
               bool every_column) {
    std::vector<std::int32_t> ids(tokens.begin() + first, tokens.begin() + first + width), positions, logit;
    for (std::int32_t i = 0; i < width; ++i) positions.push_back(first + i);
    if (every_column) {
        for (std::int32_t i = 0; i < width; ++i) logit.push_back(i);
    } else {
        logit.push_back(width - 1);
    }
    const std::vector<std::int32_t> slots{row}, rows{row};
    const auto padded = [](std::size_t n) { return (n * 4 + 255) / 256 * 256; };
    Call call;
    call.buffer = DeviceBuffer(padded(ids.size()) + padded(positions.size()) + padded(1) * 2 + padded(logit.size()));
    std::size_t offset = 0;
    auto put           = [&](const std::vector<std::int32_t>& v) {
        call.buffer.copy_from_host(v.data(), v.size() * 4, offset);
        void* p = static_cast<std::byte*>(call.buffer.p) + offset;
        offset += padded(v.size());
        return p;
    };
    call.batch.ids           = Tensor(put(ids), DType::I32, {width});
    call.batch.positions     = Tensor(put(positions), DType::I32, {width});
    call.batch.slots         = Tensor(put(slots), DType::I32, {1});
    call.batch.table_rows    = Tensor(put(rows), DType::I32, {1});
    call.batch.logit_columns = Tensor(put(logit), DType::I32, {static_cast<std::int32_t>(logit.size())});
    call.host_slots          = slots;
    call.batch.host_slots    = call.host_slots;
    call.batch.batch         = 1;
    call.batch.width         = width;
    return call;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

int main() {
    const char* artifact  = std::getenv("INFERNIX_GLM_ARTIFACT");
    const char* reference = std::getenv("INFERNIX_GLM_REFERENCE");
    if (artifact == nullptr || reference == nullptr) {
        std::printf("SKIP: set INFERNIX_GLM_ARTIFACT and INFERNIX_GLM_REFERENCE\n");
        return 77;
    }
    try {
        const std::string dir = reference;
        const auto tokens     = read_tokens(dir + "/fp64/tokens.txt");
        const auto T          = static_cast<std::int32_t>(tokens.size());
        if (read_tokens(dir + "/bf16/tokens.txt") != tokens) {
            throw std::invalid_argument("the fp64 and bf16 reference runs have different prompts");
        }

        DeviceContext device(0);
        const auto t0 = std::chrono::steady_clock::now();
        artifact::Reader reader(artifact);
        if (!glm::is_glm5_next(reader)) throw std::invalid_argument("artifact is not Glm5Next");
        models::LoadOptions options;
        options.stream_experts = true;
        auto model = glm::materialize_model(glm::plan_load(reader, options), device);
        device.synchronize();
        std::printf("loaded in %.1f s\n", seconds_since(t0));
        const exec::Parameters parameters(*model);
        const auto& c = model->config().text;
        const std::size_t V = c.vocab_size, W = c.residual_width(), L = c.num_hidden_layers, K = c.moe.top_k;
        const std::size_t moe_layers = c.moe_layers();

        const ReferenceRun ref(dir + "/fp64", T, V, L * T * W, moe_layers * T * K);
        const ReferenceRun rounded(dir + "/bf16", T, V, L * T * W, moe_layers * T * K);

        // The floor: how far BF16 rounding alone (the rounded reference) moves this prompt from the
        // binary64 oracle, at the worst block, the worst position's logits and in routed sets.
        double floor_layer = 0, floor_logits = 0;
        for (std::size_t l = 0; l < L; ++l) {
            floor_layer = std::max(floor_layer, relative_l2(rounded.residuals.data() + l * T * W,
                                                            ref.residuals.data() + l * T * W, W * T));
        }
        for (std::int32_t t = 0; t < T; ++t) {
            floor_logits = std::max(floor_logits, relative_l2(rounded.logits.data() + t * V, ref.logits.data() + t * V, V));
        }
        const double floor_routes = route_share(rounded.routed, ref.routed, moe_layers * T, K);
        std::printf("BF16 floor: worst block %.3e, worst logits %.3e, routed sets equal %.2f%%\n", floor_layer,
                    floor_logits, 100.0 * floor_routes);

        std::vector<std::vector<std::int32_t>> used(moe_layers);
        for (std::size_t l = 0; l < moe_layers; ++l) {
            std::set<std::int32_t> ids(ref.routed.begin() + l * T * K, ref.routed.begin() + (l + 1) * T * K);
            ids.insert(rounded.routed.begin() + l * T * K, rounded.routed.begin() + (l + 1) * T * K);
            used[l].assign(ids.begin(), ids.end());
        }
        constexpr std::int32_t kContext = 64;
        // One prefill of every token, observed at every block. The frames start as the references'
        // routed experts and grow by every expert either lane routes to, until a pass routes only to
        // resident experts (the stand-in record then serves no job).
        std::vector<DeviceBuffer> residual_buffers, route_buffers;
        std::vector<Tensor> residual_taps, route_taps;
        for (std::size_t l = 0; l < L; ++l) {
            residual_buffers.emplace_back(W * T * 2);
            residual_taps.push_back(Tensor(residual_buffers.back().p, DType::BF16, {static_cast<std::int32_t>(W), T}));
        }
        for (std::size_t l = 0; l < moe_layers; ++l) {
            route_buffers.emplace_back(K * T * 4);
            route_taps.push_back(Tensor(route_buffers.back().p, DType::I32, {static_cast<std::int32_t>(K), T}));
        }
        exec::ForwardTap tap{.residuals = &residual_taps, .routes = &route_taps};
        DeviceBuffer logits_buffer(V * T * 4);
        Tensor logits(logits_buffer.p, DType::FP32, {static_cast<std::int32_t>(V), T});
        Call whole = make_call(tokens, 0, T, 0, true);
        // Lane 1 prefills all but the last kDecoded tokens, then decodes those one at a time; its
        // logits at positions [prefix - 1, T) are compared with the single prefill's.
        constexpr std::int32_t kDecoded = 5;
        const std::int32_t prefix       = T - kDecoded;
        DeviceBuffer step_buffer(V * 4);
        Tensor step_logits(step_buffer.p, DType::FP32, {static_cast<std::int32_t>(V), 1});
        std::vector<std::vector<float>> split_logits;
        std::unique_ptr<ExpertFrames> frames;
        std::unique_ptr<Harness> lane;
        for (int pass = 0;; ++pass) {
            lane.reset();
            frames.reset();
            const auto t_frames = std::chrono::steady_clock::now();
            frames = std::make_unique<ExpertFrames>(*model, used);
            std::size_t resident = 0;
            for (const auto& u : used) resident += u.size();
            std::printf("pass %d: %zu experts read into frames in %.1f s\n", pass, resident, seconds_since(t_frames));
            lane = std::make_unique<Harness>(parameters, device, *frames, kContext, T, 2);
            std::vector<std::set<std::int32_t>> routed(moe_layers);
            const auto log_routes = [&](std::int32_t width) {
                std::vector<std::int32_t> ids(K * static_cast<std::size_t>(width));
                for (std::size_t l = 0; l < moe_layers; ++l) {
                    lane->route_log.copy_to_host(ids.data(), ids.size() * 4, l * lane->route_stride * 4);
                    routed[l].insert(ids.begin(), ids.end());
                }
            };
            const auto t1 = std::chrono::steady_clock::now();
            lane->forward->run(whole.batch, logits, &tap);
            device.synchronize();
            std::printf("prefill of %d tokens: %.3f s\n", T, seconds_since(t1));
            log_routes(T);

            split_logits.clear();
            Call head = make_call(tokens, 0, prefix, 1, false);
            lane->forward->run(head.batch, step_logits, nullptr);
            for (std::int32_t p = prefix - 1; p < T; ++p) {
                if (p >= prefix) {
                    Call call = make_call(tokens, p, 1, 1, false);
                    lane->forward->run(call.batch, step_logits, nullptr);
                }
                device.synchronize();
                log_routes(p >= prefix ? 1 : prefix);
                split_logits.emplace_back(V);
                step_buffer.copy_to_host(split_logits.back().data(), V * 4);
            }

            std::size_t added = 0;
            for (std::size_t l = 0; l < moe_layers; ++l) {
                std::set<std::int32_t> merged(used[l].begin(), used[l].end());
                for (const std::int32_t id : routed[l]) added += merged.insert(id).second;
                used[l].assign(merged.begin(), merged.end());
            }
            if (added == 0) break;
            if (pass == 4) throw std::runtime_error("the engine's routing did not settle in five passes");
            std::printf("pass %d routed %zu experts outside the frames; adding them\n", pass, added);
        }

        std::vector<float> got_logits(V * T);
        logits_buffer.copy_to_host(got_logits.data(), got_logits.size() * 4);
        std::vector<std::uint16_t> bits(W * T);
        std::vector<float> residual(W * T);
        double worst_layer = 0;
        // INFERNIX_GLM_DUMP=DIR: every block's residual (FP32 [layers, T, S*H]) and routes (I32 [MoE
        // layers, T, k]) of the single prefill, for tools/glm5_next/block_check.py.
        const char* dump = std::getenv("INFERNIX_GLM_DUMP");
        std::ofstream dump_residuals, dump_routes;
        if (dump != nullptr) {
            dump_residuals.open(std::string(dump) + "/residuals.f32", std::ios::binary);
            dump_routes.open(std::string(dump) + "/routed.i32", std::ios::binary);
        }
        for (std::size_t l = 0; l < L; ++l) {
            residual_buffers[l].copy_to_host(bits.data(), bits.size() * 2);
            for (std::size_t i = 0; i < bits.size(); ++i) residual[i] = bf16_to_float(bits[i]);
            if (dump != nullptr) {
                dump_residuals.write(reinterpret_cast<const char*>(residual.data()),
                                     static_cast<std::streamsize>(residual.size() * 4));
            }
            const double error = relative_l2(residual.data(), ref.residuals.data() + l * T * W, W * T);
            const double floor = relative_l2(rounded.residuals.data() + l * T * W, ref.residuals.data() + l * T * W, W * T);
            worst_layer        = std::max(worst_layer, error);
            std::printf("layer %2zu %s residual rel L2 %.3e (BF16 floor %.3e)\n", l,
                        c.layer_types[l] == glm::MixerKind::Kda ? "KDA" : "MLA", error, floor);
        }
        std::vector<std::int32_t> got_routed(moe_layers * T * K);
        for (std::size_t l = 0; l < moe_layers; ++l) {
            route_buffers[l].copy_to_host(got_routed.data() + l * T * K, T * K * 4);
        }
        if (dump != nullptr) {
            dump_routes.write(reinterpret_cast<const char*>(got_routed.data()),
                              static_cast<std::streamsize>(got_routed.size() * 4));
        }
        const double routes = route_share(got_routed, ref.routed, moe_layers * T, K);
        std::printf("routed sets equal to the reference: %.2f%% (BF16 floor %.2f%%)\n", 100.0 * routes,
                    100.0 * floor_routes);
        std::int32_t argmax_off = 0;
        double worst_logits     = 0;
        for (std::int32_t t = 0; t < T; ++t) {
            const float* mine  = got_logits.data() + t * V;
            const double error = relative_l2(mine, ref.logits.data() + t * V, V);
            worst_logits       = std::max(worst_logits, error);
            const std::size_t top = argmax(mine, V), exact = argmax(ref.logits.data() + t * V, V);
            const std::size_t near = argmax(rounded.logits.data() + t * V, V);
            argmax_off += top != exact && top != near;
            std::printf("position %2d: logits rel L2 %.3e, argmax %zu (reference %zu, rounded reference %zu)\n", t,
                        error, top, exact, near);
        }
        // Every block's own error is BF16-sized (block_check.py), but the model amplifies it, so the
        // whole forward is held to the floor the rounded reference sets: within twice its worst
        // deviation, the argmax one of the two references' (they differ only at near-ties), and the
        // routed sets equal about as often.
        expect(worst_layer < 2 * floor_layer, "a block's residual deviates from the reference by " +
                                                  std::to_string(worst_layer));
        expect(worst_logits < 2 * floor_logits, "logits deviate from the reference by " + std::to_string(worst_logits));
        expect(argmax_off == 0, "argmax is neither reference's at " + std::to_string(argmax_off) + " positions");
        expect(routes >= floor_routes - 0.1, "the routed sets equal the reference's less often than the floor allows");

        // The split lane's arithmetic differs from the single prefill's in order only.
        double worst_decode = 0;
        for (std::int32_t p = prefix - 1; p < T; ++p) {
            const auto& step          = split_logits[static_cast<std::size_t>(p - (prefix - 1))];
            const float* whole_logits = got_logits.data() + static_cast<std::size_t>(p) * V;
            const double error        = relative_l2(step.data(), whole_logits, V);
            worst_decode              = std::max(worst_decode, error);
            const std::size_t top = argmax(step.data(), V), whole = argmax(whole_logits, V);
            std::printf("position %2d %s: logits rel L2 to the single prefill %.3e, argmax %zu (prefill %zu)\n", p,
                        p < prefix ? "prefill" : "decode ", error, top, whole);
            expect(top == whole || top == argmax(ref.logits.data() + p * V, V) ||
                       top == argmax(rounded.logits.data() + p * V, V),
                   "the split lane's argmax at position " + std::to_string(p) + " is no reference's or the prefill's");
        }
        expect(worst_decode < 2 * floor_logits, "the split lane's logits deviate from the single prefill by " +
                                                    std::to_string(worst_decode));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
