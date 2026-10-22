// The T1 round harness (AGENTS.md "Iteration tiers"): one Qwen4Exp generation round of a chosen
// shape through the production Forward, its CUDA Graph and its verification commit, without the
// Engine, the server or the expert tiers. It loads the dense weights to the device (the expert banks
// stay in the artifact), places every routed expert in a VRAM frame as `--experts vram=100%` does,
// prefills each row from the benchmark corpus, then replays each round shape and reports its device
// time, the commit time of a verification round, an optional per-section breakdown and a hash of
// the round's logits.
//
//   infernix_qwen4_exp_round_bench [--artifact PATH] [--ngram-volume PATH] [--shape RxW[,RxW...]]
//       [--context N] [--replays N] [--kv bf16|int8|fp8|nvfp4|k8v4] [--experts N|all]
//       [--accept K] [--sections] [--layers] [--exact] [--profile] [--check FILE | --record FILE]
//       [--determinism N] [--no-fork] [--no-shared-stream]
//
// A shape R x W is R rows of W columns: W = 1 a plain decode round, W >= 2 a verification round of
// W - 1 drafts per row followed by the commit of K columns per row (--accept, default W). Each row
// holds --context tokens (default 1024) before the round; all 8 lanes are prefilled, the round runs the first R. --experts N (default 64) reads the first N
// records of every layer from the artifact and fills the other frames with copies of them, so every
// expert still has its own frame (the same bytes moved per round) at a fraction of the read time;
// --experts all reads every record (about 68 GB). The logits hash depends on that choice, on the KV
// storage and on the fast paths (--exact runs the exact route), and is compared between builds with
// the same options. --sections times each section (embedding, every layer's mixer and MoE halves, the
// head) as its own graph; --layers prints them per layer. --profile brackets the timed replays with
// cudaProfilerStart/Stop for tools/flash_next/t1_kernels.py. --check FILE fails (exit 2) when a shape's
// hash differs from the one FILE records for the same artifact and options; --record FILE writes them.
// Every shape replays N times (--determinism, default 2) from the same state and fails unless all
// give the same bits. --no-fork and --no-shared-stream run the MoE's miss fork and the shared expert
// in order on the main stream (to tell which overlap a nondeterminism needs).
//
// Not included: the DFlash2 drafter's calls and catch-up, the acceptance kernels, sampling and the
// host's round work (the server's GPU is busy about 98% of a round, the drafter included).

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/qsa.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/expert_store.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/program/fast_paths.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/rope_positions.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace infernix;
namespace q4 = infernix::models::qwen4_exp;

namespace {

constexpr std::int32_t kPrefillChunk = 512;
constexpr double kWarmSeconds       = 0.5;
constexpr std::int32_t kLanes       = 8;
constexpr const char* kDefaultArtifact = "models/infernix/Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix.infernix";
constexpr const char* kDefaultNgram    = "models/infernix/Qwen3.8-Flash-Next-BF16.ngram";

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

KvCacheStorage parse_kv(const std::string& name) {
    if (name == "bf16") { return KvCacheStorage::BFloat16; }
    if (name == "int8") { return KvCacheStorage::Int8Group64; }
    if (name == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (name == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (name == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("unknown --kv " + name);
}

struct Shape {
    std::int32_t rows = 0, width = 0;
};

struct Options {
    std::filesystem::path artifact, ngram;
    std::vector<Shape> shapes{{6, 1}, {6, 2}};
    std::int32_t context = 1024, replays = 50, warmup = 5, accept = -1;
    std::int32_t determinism = 2; // replays from the same state whose hashes must agree
    std::int32_t sample_experts = 64; // 0: every record
    std::string kv_name = "fp8";
    bool sections = false, layers = false, exact = false, profile = false;
    std::filesystem::path check, record; // golden hashes: compare with, or write into
    bool no_fork = false, no_shared_stream = false; // run the MoE fork / the shared expert in order
};

std::vector<Shape> parse_shapes(const std::string& text) {
    std::vector<Shape> out;
    std::stringstream list(text);
    std::string item;
    while (std::getline(list, item, ',')) {
        const auto x = item.find('x');
        if (x == std::string::npos) { throw std::invalid_argument("--shape takes RxW, e.g. 6x2"); }
        const Shape s{std::stoi(item.substr(0, x)), std::stoi(item.substr(x + 1))};
        if (s.rows < 1 || s.rows > kLanes || s.width < 1 || s.width > 16) {
            throw std::invalid_argument("--shape " + item + ": rows 1..8, width 1..16");
        }
        out.push_back(s);
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    const char* home = std::getenv("HOME");
    const std::filesystem::path base = home != nullptr ? home : ".";
    o.artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT") != nullptr ? std::getenv("INFERNIX_QWEN4_ARTIFACT")
                                                                   : (base / kDefaultArtifact).string();
    o.ngram    = std::getenv("INFERNIX_QWEN4_NGRAM") != nullptr ? std::getenv("INFERNIX_QWEN4_NGRAM")
                                                                : (base / kDefaultNgram).string();
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--artifact") {
            o.artifact = value();
        } else if (arg == "--ngram-volume") {
            o.ngram = value();
        } else if (arg == "--shape") {
            o.shapes = parse_shapes(value());
        } else if (arg == "--context") {
            o.context = std::stoi(value());
        } else if (arg == "--replays") {
            o.replays = std::max(1, std::stoi(value()));
        } else if (arg == "--kv") {
            o.kv_name = value();
        } else if (arg == "--experts") {
            const std::string v = value();
            o.sample_experts    = v == "all" ? 0 : std::stoi(v);
            if (o.sample_experts < 0) { throw std::invalid_argument("--experts takes N >= 1 or all"); }
        } else if (arg == "--accept") {
            o.accept = std::stoi(value());
        } else if (arg == "--sections") {
            o.sections = true;
        } else if (arg == "--layers") {
            o.sections = o.layers = true;
        } else if (arg == "--exact") {
            o.exact = true;
        } else if (arg == "--profile") {
            o.profile = true;
        } else if (arg == "--determinism") {
            o.determinism = std::max(2, std::stoi(value()));
        } else if (arg == "--serial-streams") {
            o.no_fork = o.no_shared_stream = true;
        } else if (arg == "--no-fork") {
            o.no_fork = true;
        } else if (arg == "--no-shared-stream") {
            o.no_shared_stream = true;
        } else if (arg == "--check") {
            o.check = value();
        } else if (arg == "--record") {
            o.record = value();
        } else if (arg == "-h" || arg == "--help") {
            std::printf("see the header of bench/models/qwen4_exp/round_bench.cpp\n");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    if (o.context < 1) { throw std::invalid_argument("--context must be positive"); }
    return o;
}

std::vector<std::int32_t> read_corpus(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("cannot read " + path.string()); }
    std::vector<std::int32_t> out{std::istream_iterator<std::int32_t>(file), std::istream_iterator<std::int32_t>()};
    if (out.size() < 1024) { throw std::runtime_error(path.string() + " holds too few tokens"); }
    return out;
}

// Every routed expert in a VRAM frame: one pool per layer at its record pitch, frame e = expert e.
struct ExpertFrames {
    std::vector<DeviceBuffer> pools;
    DeviceBuffer tables; // I32 [E] per layer: the identity
    // Per layer [E] host record pointers, all null: the banks stay in the artifact and no expert
    // misses its frame, so nothing reads them (the MoE requires a host source).
    DeviceBuffer host_tables;

    ExpertFrames(const q4::execution::Parameters& parameters, const q4::ExpertStore& store, std::int32_t sample,
                 DeviceContext& device) {
        const auto& c = parameters.model.config().text;
        const auto E  = static_cast<std::int32_t>(c.moe.experts);
        const auto L  = static_cast<std::int32_t>(c.num_hidden_layers);
        const std::int32_t read = sample == 0 ? E : std::min(sample, E);
        std::vector<int> files;
        for (const auto& path : store.files()) {
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) { throw std::runtime_error("cannot open " + path.string()); }
            files.push_back(fd);
        }
        constexpr std::int32_t kBatch   = 32;
        constexpr std::int32_t kReaders = 8;
        const std::uint64_t pitch_max   = store.record_bytes();
        PinnedHostBuffer staging(static_cast<std::size_t>(kBatch) * pitch_max);
        auto* host = static_cast<std::byte*>(staging.data());
        std::uint64_t bytes_read = 0;
        for (std::int32_t l = 0; l < L; ++l) {
            const std::uint64_t pitch = parameters.layers[l].moe.bank->planes.record_stride;
            if (pitch != store.record_bytes(static_cast<std::uint32_t>(l))) {
                throw std::runtime_error("expert store pitch differs from the bank's record stride");
            }
            pools.emplace_back(static_cast<std::size_t>(E) * pitch);
            auto* pool = static_cast<std::byte*>(pools.back().p);
            for (std::int32_t first = 0; first < read; first += kBatch) {
                const std::int32_t n = std::min(kBatch, read - first);
                std::vector<std::thread> readers;
                for (std::int32_t r = 0; r < kReaders; ++r) {
                    readers.emplace_back([&, r] {
                        for (std::int32_t j = r; j < n; j += kReaders) {
                            for (const auto& seg : store.segments(static_cast<std::uint32_t>(l),
                                                                  static_cast<std::uint32_t>(first + j))) {
                                std::byte* to = host + static_cast<std::size_t>(j) * pitch + seg.at;
                                std::uint64_t done = 0;
                                while (done < seg.bytes) {
                                    const ssize_t got = ::pread(files[seg.file], to + done, seg.bytes - done,
                                                                static_cast<off_t>(seg.offset + done));
                                    if (got <= 0) { throw std::runtime_error("expert record read failed"); }
                                    done += static_cast<std::uint64_t>(got);
                                }
                            }
                        }
                    });
                }
                for (auto& t : readers) { t.join(); }
                check(cudaMemcpy(pool + static_cast<std::size_t>(first) * pitch, host, static_cast<std::size_t>(n) * pitch,
                                 cudaMemcpyHostToDevice),
                      "expert upload");
                bytes_read += static_cast<std::uint64_t>(n) * pitch;
            }
            for (std::int32_t e = read; e < E; e += read) {
                const std::int32_t n = std::min(read, E - e);
                check(cudaMemcpy(pool + static_cast<std::size_t>(e) * pitch, pool, static_cast<std::size_t>(n) * pitch,
                                 cudaMemcpyDeviceToDevice),
                      "expert frame copy");
            }
        }
        for (const int fd : files) { ::close(fd); }
        std::vector<std::int32_t> identity(static_cast<std::size_t>(E) * L);
        for (std::size_t i = 0; i < identity.size(); ++i) { identity[i] = static_cast<std::int32_t>(i % E); }
        tables = DeviceBuffer(identity.size() * sizeof(std::int32_t));
        tables.copy_from_host(identity.data(), tables.bytes);
        host_tables = DeviceBuffer(identity.size() * sizeof(void*));
        host_tables.fill(0);
        device.synchronize();
        std::printf("experts: %d per layer in VRAM frames, %d records read per layer (%.1f GiB)\n", E, read,
                    bytes_read / 1073741824.0);
    }
};

// The Program's per-lane state, KV, verification records and streams for `rows` lanes of up to
// `context` tokens.
struct Harness {
    Harness(const q4::execution::Parameters& parameters, DeviceContext& device, const ExpertFrames& frames,
            std::int32_t context, std::int32_t columns, std::int32_t rows, std::int32_t max_width, KvCacheStorage storage,
            q4::execution::ForwardFastPaths fast, bool no_fork, bool no_shared_stream)
        : config(parameters.model.config().text), rows(rows) {
        const auto& c = config;
        LayoutBuilder builder;
        const LinearAttentionStatePoolSpec gdn_spec{
            .layers         = c.gdn_layers,
            .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
            .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
            .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
            .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
            .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
            .slot_count     = rows,
        };
        const auto gdn_layout = plan_linear_attention_state_pool(builder, gdn_spec);
        state_backing         = DeviceBuffer(builder.finish(256));
        state_backing.fill(0);
        gdn = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing.p, state_backing.bytes}, gdn_layout);

        width = static_cast<std::int32_t>(c.residual_width());
        span  = static_cast<std::int32_t>(c.ple.conv_span());
        ple_backing = DeviceBuffer(static_cast<std::size_t>(width) * span * rows * 2);
        ple_backing.fill(0);
        di = static_cast<std::int32_t>(c.qsa.index_head_dim);
        r  = static_cast<std::int32_t>(c.qsa.compress_ratio);
        tails_backing = DeviceBuffer(static_cast<std::size_t>(c.attention_layers) * di * (r - 1) * rows * 2);
        tails_backing.fill(0);

        const auto layout = paged_kv_storage_layout(storage, static_cast<std::int32_t>(c.attention.head_dim));
        KVPageGeometry geometry;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
            geometry.planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
            if (layout.key.has_scale()) {
                geometry.planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
            if (layout.value.has_scale()) {
                geometry.planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({DType::BF16, di / r, 1});
        }
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
            if (!reservation) { throw std::runtime_error("KV pool too small"); }
            std::vector<DeviceKVPageLease> pages;
            pages.reserve(static_cast<std::size_t>(pages_per_row));
            pool->materialize(*reservation, static_cast<std::uint32_t>(pages_per_row), pages);
            row_leases.push_back(tables->acquire(row));
            tables->publish(row_leases.back().handle(), 0, pages, device.stream);
            page_leases.push_back(std::move(pages));
            reservations.push_back(std::move(*reservation));
        }
        device.synchronize();

        q4::execution::ForwardState state;
        state.gdn      = gdn.get();
        state.ple_conv = Tensor(ple_backing.p, DType::BF16, {width, span, rows});
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(tails_backing.p) +
                                                 static_cast<std::size_t>(l) * di * (r - 1) * rows * 2,
                                             DType::BF16, {di, r - 1, rows}));
        }
        window_geometry = q4::prefix::kv_window_geometry(c, storage, false, static_cast<std::uint32_t>(rows));
        if (window_geometry.present()) {
            window_backing = DeviceBuffer(window_geometry.bytes());
            window_backing.fill(0);
        }
        q4::execution::ForwardKV kv;
        kv.block_tables = tables->matrix();
        std::size_t plane   = 0;
        const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            ops::QsaKVLayer layer;
            layer.kv.storage      = storage;
            layer.kv.head_dim     = static_cast<std::int32_t>(c.attention.head_dim);
            layer.kv.num_kv_heads = kv_heads;
            layer.kv.k_pages      = pool->plane(plane++);
            if (layout.key.has_scale()) { layer.kv.k_scale_pages = pool->plane(plane++); }
            layer.kv.v_pages = pool->plane(plane++);
            if (layout.value.has_scale()) { layer.kv.v_scale_pages = pool->plane(plane++); }
            layer.pooled_pages = pool->plane(plane++);
            if (window_geometry.present()) { layer.kv.window = q4::prefix::kv_window_view(window_geometry, window_backing.p, l); }
            kv.layers.push_back(layer);
        }

        // Verification records (the Program's allocate_verification, at the largest width).
        if (max_width > 1) {
            LayoutBuilder records_builder;
            const auto plan = plan_gdn_replay_records(records_builder, {.layers          = static_cast<std::int32_t>(c.gdn_layers),
                                                                        .record_capacity = rows,
                                                                        .width           = max_width,
                                                                        .conv_channels   = static_cast<std::int32_t>(c.gdn.conv_channels()),
                                                                        .qk_heads        = static_cast<std::int32_t>(c.gdn.key_heads),
                                                                        .value_heads     = static_cast<std::int32_t>(c.gdn.value_heads),
                                                                        .key_dim         = static_cast<std::int32_t>(c.gdn.key_head_dim),
                                                                        .value_dim       = static_cast<std::int32_t>(c.gdn.value_head_dim)});
            records_backing = DeviceBuffer(records_builder.finish(256));
            records         = GdnReplayRecords(DeviceSpan{records_backing.p, records_backing.bytes}, plan);
            ple_records     = DeviceBuffer(2ULL * width * max_width * rows);
            qsa_records     = DeviceBuffer(2ULL * di * max_width * rows * c.attention_layers);
        }

        // Expert sources and streams as the Program sets them for a resident placement.
        q4::execution::ForwardExperts experts;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            experts.frame_bases.push_back(static_cast<const std::uint8_t*>(frames.pools[l].p));
            experts.frame_pitches.push_back(parameters.layers[l].moe.bank->planes.record_stride);
            experts.frames.push_back(static_cast<const std::int32_t*>(frames.tables.p) + l * c.moe.experts);
            experts.host_tables.push_back(static_cast<const std::uint8_t* const*>(frames.host_tables.p) + l * c.moe.experts);
        }
        check(cudaStreamCreateWithFlags(&overlap_stream, cudaStreamNonBlocking), "stream");
        for (auto& event : overlap_events) { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
        check(cudaStreamCreateWithFlags(&shared_stream, cudaStreamNonBlocking), "stream");
        for (auto& event : shared_events) { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
        if (!no_fork) {
            experts.overlap_stream = overlap_stream;
            experts.overlap_events = overlap_events;
        }
        if (!no_shared_stream) {
            experts.shared_stream = shared_stream;
            experts.shared_events = shared_events;
        }

        const auto wide = fast.expert_a4_residual ? ops::offloaded_moe::WideA16::kA4Residual : ops::offloaded_moe::WideA16::kExact;
        work = std::make_unique<WorkspaceArena>(q4::execution::Forward::workspace_bytes(c, columns, context, storage, wide));
        forward = std::make_unique<q4::execution::Forward>(parameters, device, *work, std::move(state), std::move(kv),
                                                           std::move(experts), context, fast);
        snapshot[0] = DeviceBuffer(state_backing.bytes);
        snapshot[1] = DeviceBuffer(ple_backing.bytes);
        snapshot[2] = DeviceBuffer(tails_backing.bytes);
    }

    ~Harness() {
        forward.reset();
        for (auto event : overlap_events) { cudaEventDestroy(event); }
        for (auto event : shared_events) { cudaEventDestroy(event); }
        cudaStreamDestroy(overlap_stream);
        cudaStreamDestroy(shared_stream);
    }

    // The recurrent state of every lane (GDN, PLE history, QSA tails): saved after the prefill and
    // restored before each replay, so every replay starts from the same state.
    void save(cudaStream_t s) { copy_state(true, s); }
    void restore(cudaStream_t s) { copy_state(false, s); }

    q4::execution::ForwardVerify verify_view(std::int32_t batch, std::int32_t w) const {
        q4::execution::ForwardVerify v;
        v.gdn        = records.narrowed(w);
        v.ple_inputs = Tensor(ple_records.p, DType::BF16, {width, w, batch});
        v.qsa_keys   = Tensor(qsa_records.p, DType::BF16, {di, w, batch, static_cast<std::int32_t>(config.attention_layers)});
        return v;
    }

    const q4::TextConfig& config;
    std::int32_t rows = 0, width = 0, span = 0, di = 0, r = 0, pages_per_row = 0;
    DeviceBuffer state_backing, ple_backing, tails_backing, kv_backing, window_backing;
    DeviceBuffer records_backing, ple_records, qsa_records;
    DeviceBuffer snapshot[3];
    GdnReplayRecords records;
    q4::prefix::KvWindowGeometry window_geometry;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::vector<DeviceKVPageReservation> reservations;
    std::vector<std::vector<DeviceKVPageLease>> page_leases;
    std::vector<KVExecutionRowLease> row_leases;
    cudaStream_t overlap_stream = nullptr, shared_stream = nullptr;
    std::array<cudaEvent_t, 5> overlap_events{};
    std::array<cudaEvent_t, 2> shared_events{};
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<q4::execution::Forward> forward;

private:
    void copy_state(bool save_state, cudaStream_t s) {
        const DeviceBuffer* live[3] = {&state_backing, &ple_backing, &tails_backing};
        for (int i = 0; i < 3; ++i) {
            void* to         = save_state ? snapshot[i].p : live[i]->p;
            const void* from = save_state ? live[i]->p : snapshot[i].p;
            check(cudaMemcpyAsync(to, from, live[i]->bytes, cudaMemcpyDeviceToDevice, s), "state copy");
        }
    }
};

// Device copies of one call's inputs: `batch` rows of `width` columns from position `first`.
struct Call {
    DeviceBuffer buffer;
    q4::execution::ForwardBatch batch;
    std::vector<std::int32_t> host_slots;
};

Call make_call(const q4::TextConfig& c, const q4::NgramVolume& volume, const std::vector<std::vector<std::int32_t>>& histories,
               std::span<const std::int32_t> slots, std::int32_t first, std::int32_t width, bool every_column) {
    const auto batch           = static_cast<std::int32_t>(slots.size());
    const std::int32_t columns = batch * width;
    const q4::NgramHash hash(c.ple.ngram, 0);
    const std::size_t row_bytes = volume.table().row_bytes;
    const std::size_t heads     = hash.heads();
    std::vector<std::int32_t> ids, positions, last;
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(columns)), block_rope(3 * static_cast<std::size_t>(batch));
    const q4::LaneRope text;
    std::vector<std::byte> ngram(columns * heads * row_bytes);
    for (std::int32_t b = 0; b < batch; ++b) {
        const auto& history = histories[static_cast<std::size_t>(slots[b])];
        std::vector<std::int32_t> padded(c.ple.ngram.ngram_size - 1, c.eos_token_id);
        padded.insert(padded.end(), history.begin(), history.begin() + first + width);
        std::vector<std::uint32_t> row_ids(static_cast<std::size_t>(width) * heads);
        hash.row_ids(padded, static_cast<std::size_t>(width), row_ids.data());
        volume.read_rows(row_ids, std::span<std::byte>(ngram).subspan(static_cast<std::size_t>(b) * width * heads * row_bytes,
                                                                    static_cast<std::size_t>(width) * heads * row_bytes));
        for (std::int32_t i = 0; i < width; ++i) {
            ids.push_back(history[static_cast<std::size_t>(first + i)]);
            positions.push_back(first + i);
            if (every_column || i == width - 1) { last.push_back(b * width + i); }
        }
        q4::stage_rope(text, static_cast<std::uint32_t>(first), width, rope, columns, b * width);
        q4::stage_rope_value(q4::block_start_rope(text, static_cast<std::uint32_t>(first), c.qsa.compress_ratio),
                             block_rope, batch, b);
    }
    Call call;
    call.host_slots.assign(slots.begin(), slots.end());
    const std::size_t i32 = sizeof(std::int32_t);
    const auto padded     = [](std::size_t n) { return (n + 255) / 256 * 256; };
    call.buffer = DeviceBuffer(padded(ids.size() * i32) * 2 + padded(rope.size() * i32) + padded(block_rope.size() * i32) +
                               padded(slots.size() * i32) * 2 + padded(last.size() * i32) + padded(ngram.size()));
    auto* base         = static_cast<std::byte*>(call.buffer.p);
    std::size_t offset = 0;
    const auto put     = [&](const void* data, std::size_t n) {
        call.buffer.copy_from_host(data, n, offset);
        void* p = base + offset;
        offset += padded(n);
        return p;
    };
    call.batch.ids              = Tensor(put(ids.data(), ids.size() * i32), DType::I32, {columns});
    call.batch.positions        = Tensor(put(positions.data(), positions.size() * i32), DType::I32, {columns});
    call.batch.rope_positions   = Tensor(put(rope.data(), rope.size() * i32), DType::I32, {columns, 3});
    call.batch.block_start_rope = Tensor(put(block_rope.data(), block_rope.size() * i32), DType::I32, {batch, 3});
    call.batch.slots            = Tensor(put(slots.data(), slots.size() * i32), DType::I32, {batch});
    call.batch.table_rows       = Tensor(put(slots.data(), slots.size() * i32), DType::I32, {batch});
    call.batch.logit_columns =
        Tensor(put(last.data(), last.size() * i32), DType::I32, {static_cast<std::int32_t>(last.size())});
    call.batch.ngram_rows   = Tensor(put(ngram.data(), ngram.size()), DType::U8,
                                     {static_cast<std::int32_t>(row_bytes), static_cast<std::int32_t>(heads), columns});
    call.batch.ngram_format = volume.table().format;
    call.batch.host_slots      = call.host_slots;
    call.batch.host_table_rows = call.host_slots;
    call.batch.batch           = batch;
    call.batch.width           = width;
    return call;
}

// A 64-bit hash of the bytes, a word at a time (FNV-1a over 64-bit words; the tail byte-wise).
std::uint64_t hash_bytes(const void* data, std::size_t bytes) {
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p   = static_cast<const unsigned char*>(data);
    std::size_t i   = 0;
    for (; i + 8 <= bytes; i += 8) {
        std::uint64_t word;
        std::memcpy(&word, p + i, 8);
        h = (h ^ word) * kPrime;
    }
    for (; i < bytes; ++i) { h = (h ^ p[i]) * kPrime; }
    return h ^ (h >> 29);
}

struct Stats {
    double median = 0, min = 0, mean = 0;
};

Stats summarize(std::vector<double> v) {
    if (v.empty()) { return {}; }
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size())};
}

class Timer {
public:
    explicit Timer(std::size_t marks) : events_(marks) {
        for (auto& e : events_) { check(cudaEventCreate(&e), "event"); }
    }
    ~Timer() {
        for (auto e : events_) { cudaEventDestroy(e); }
    }
    void mark(std::size_t i, cudaStream_t s) { check(cudaEventRecord(events_[i], s), "event record"); }
    [[nodiscard]] double us(std::size_t from, std::size_t to) const {
        float ms = 0;
        check(cudaEventElapsedTime(&ms, events_[from], events_[to]), "event time");
        return 1000.0 * ms;
    }

private:
    std::vector<cudaEvent_t> events_;
};

// One round shape: its graph, the commit of a verification round, timings, hashes and sections.
// Returns the hashes ("logits state", hex).
std::string run_shape(const Options& o, const Shape& shape, Harness& h, DeviceContext& device, const q4::NgramVolume& volume,
               const std::vector<std::vector<std::int32_t>>& histories, const q4::execution::Parameters& parameters) {
    const auto& c        = h.config;
    const cudaStream_t s = device.stream;
    const std::int32_t R = shape.rows, W = shape.width, T = R * W;
    const auto vocab     = static_cast<std::int32_t>(c.vocab_size);
    const bool verify    = W > 1;
    const std::int32_t accept = o.accept < 0 ? W : std::clamp(o.accept, 0, W);
    std::vector<std::int32_t> slots(static_cast<std::size_t>(R));
    std::iota(slots.begin(), slots.end(), 0);
    Call call = make_call(c, volume, histories, slots, o.context, W, verify);
    call.batch.decode_round = true;
    q4::execution::ForwardVerify view;
    if (verify) {
        view              = h.verify_view(R, W);
        call.batch.verify = &view;
    }
    DeviceBuffer logits_buffer(static_cast<std::size_t>(vocab) * T * sizeof(float));
    Tensor logits(logits_buffer.p, DType::FP32, {vocab, call.batch.logit_columns.ne[0]});

    // The commit of a verification round (the Program's commit_verified): GDN fold, QSA tails, PLE.
    std::optional<ops::GdnReplayFoldPlan> fold;
    DeviceBuffer commit_device;
    std::vector<ops::GdnReplayFoldRow> fold_rows;
    if (verify) {
        fold.emplace(h.records.narrowed(W), h.gdn->all_layers_view());
        std::vector<std::int32_t> commit(static_cast<std::size_t>(R), accept);
        commit_device = DeviceBuffer(commit.size() * sizeof(std::int32_t));
        commit_device.copy_from_host(commit.data(), commit_device.bytes);
        for (std::int32_t b = 0; b < R; ++b) { fold_rows.push_back({b, b, accept}); }
    }
    const auto commit = [&] {
        if (!verify) { return; }
        fold->execute(fold_rows, s);
        const Tensor columns(commit_device.p, DType::I32, {R});
        const auto layers = static_cast<std::int32_t>(c.attention_layers);
        Tensor tails(h.tails_backing.p, DType::BF16, {h.di, h.r - 1, h.rows, layers});
        ops::qsa_commit_tails(Tensor(h.qsa_records.p, DType::BF16, {h.di, W, R, layers}), call.batch.positions.view({W, R}),
                              columns, tails, call.batch.slots, q4::execution::qsa_geometry(c), s);
        Tensor ple_states(h.ple_backing.p, DType::BF16, {h.width, h.span, h.rows});
        ops::ple_conv_commit(Tensor(h.ple_records.p, DType::BF16, {h.width, W, R}), columns, ple_states, call.batch.slots, s);
    };

    DecodeGraphDefinition definition;
    definition.capture(s, [&] { h.forward->run(call.batch, logits); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    graph.upload(s);

    // Hashes: the logits and the lanes' recurrent state after the round (and its commit); every replay
    // from the same state must give the same bits.
    std::vector<float> host(static_cast<std::size_t>(vocab) * logits.ne[1]);
    std::vector<std::byte> state(h.state_backing.bytes + h.ple_backing.bytes + h.tails_backing.bytes);
    std::vector<std::array<std::uint64_t, 2>> hashes(static_cast<std::size_t>(o.determinism));
    std::vector<std::int32_t> top;
    for (auto& hash : hashes) {
        h.restore(s);
        graph.launch(s);
        commit();
        device.synchronize();
        logits_buffer.copy_to_host(host.data(), host.size() * sizeof(float));
        h.state_backing.copy_to_host(state.data(), h.state_backing.bytes);
        h.ple_backing.copy_to_host(state.data() + h.state_backing.bytes, h.ple_backing.bytes);
        h.tails_backing.copy_to_host(state.data() + h.state_backing.bytes + h.ple_backing.bytes, h.tails_backing.bytes);
        hash[0] = hash_bytes(host.data(), host.size() * sizeof(float));
        hash[1] = hash_bytes(state.data(), state.size());
    }
    std::int32_t differing = 0;
    for (const auto& hash : hashes) { differing += hash != hashes.front(); }
    for (std::int32_t col = 0; col < logits.ne[1]; ++col) {
        const float* row = host.data() + static_cast<std::size_t>(col) * vocab;
        top.push_back(static_cast<std::int32_t>(std::max_element(row, row + vocab) - row));
    }

    // Warm-up: at least `warmup` replays and kWarmSeconds of them, so the clocks have settled.
    Timer timer(3);
    std::vector<double> forward_us, commit_us;
    const auto warm_start = std::chrono::steady_clock::now();
    std::int32_t warm     = 0;
    for (std::int32_t timed_count = 0; timed_count < o.replays;) {
        const bool timed = warm >= o.warmup && seconds_since(warm_start) >= kWarmSeconds;
        h.restore(s);
        if (timed && timed_count == 0 && o.profile) { check(cudaProfilerStart(), "cudaProfilerStart"); }
        timer.mark(0, s);
        graph.launch(s);
        timer.mark(1, s);
        commit();
        timer.mark(2, s);
        check(cudaStreamSynchronize(s), "replay");
        if (!timed) {
            ++warm;
        } else {
            ++timed_count;
            forward_us.push_back(timer.us(0, 1));
            commit_us.push_back(timer.us(1, 2));
        }
    }
    if (o.profile) { check(cudaProfilerStop(), "cudaProfilerStop"); }
    const Stats f = summarize(forward_us), m = summarize(commit_us);
    std::printf("\nshape %dx%d (%d columns, context %d%s)\n", R, W, T, o.context,
                verify ? (", commit " + std::to_string(accept) + " per row").c_str() : "");
    std::printf("  forward graph  median %8.1f us   min %8.1f   mean %8.1f   (%d replays)\n", f.median, f.min, f.mean,
                o.replays);
    if (verify) { std::printf("  commit         median %8.1f us   min %8.1f   mean %8.1f\n", m.median, m.min, m.mean); }
    std::printf("  round          median %8.1f us\n", f.median + m.median);
    const bool stable = differing == 0;
    std::printf("  hash logits %016llx state %016llx   top-1:", static_cast<unsigned long long>(hashes[0][0]),
                static_cast<unsigned long long>(hashes[0][1]));
    for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 12); ++i) { std::printf(" %d", top[i]); }
    std::printf("\n");
    if (!stable) {
        std::printf("  NONDETERMINISTIC: %d of %d replays from the same state differ from the first:", differing,
                    o.determinism);
        for (const auto& hash : hashes) {
            if (hash != hashes.front()) {
                std::printf(" %016llx/%016llx", static_cast<unsigned long long>(hash[0]),
                            static_cast<unsigned long long>(hash[1]));
            }
        }
        std::printf("\n");
        throw std::runtime_error("replays from the same state gave different bits");
    }
    char value[40];
    std::snprintf(value, sizeof(value), "%016llx %016llx", static_cast<unsigned long long>(hashes[0][0]),
                  static_cast<unsigned long long>(hashes[0][1]));
    const std::string result = value;

    if (!o.sections) { return result; }
    // Sections: each its own graph, so the breakdown holds kernels only (no launch gaps).
    const auto L = static_cast<std::int32_t>(c.num_hidden_layers);
    const auto S = static_cast<std::int32_t>(c.hc.streams), H = static_cast<std::int32_t>(c.hidden_size);
    DeviceBuffer residual_buffer(static_cast<std::size_t>(c.residual_width()) * T * 2);
    DeviceBuffer xm_buffer(static_cast<std::size_t>(H) * T * 2), inject_buffer(static_cast<std::size_t>(S) * T * 4);
    Tensor residual(residual_buffer.p, DType::BF16, {static_cast<std::int32_t>(c.residual_width()), T});
    Tensor xm(xm_buffer.p, DType::BF16, {H, T});
    Tensor inject(inject_buffer.p, DType::FP32, {S, T});
    std::vector<DecodeGraphExecutable> sections;
    std::vector<std::string> names;
    const auto add = [&](std::string name, const std::function<void()>& body) {
        h.forward->begin_call(call.batch, &logits);
        DecodeGraphDefinition d;
        d.capture(s, body);
        sections.emplace_back();
        sections.back().instantiate(d);
        sections.back().upload(s);
        names.push_back(std::move(name));
    };
    add("embed", [&] { h.forward->embed(call.batch, residual); });
    for (std::int32_t l = 0; l < L; ++l) {
        const bool is_gdn = std::holds_alternative<q4::execution::GdnParameters>(parameters.layers[l].mixer);
        add((is_gdn ? "gdn " : "qsa ") + std::to_string(l),
            [&, l] { h.forward->layer_mixer(static_cast<std::uint32_t>(l), call.batch, residual, xm, inject); });
        add("moe " + std::to_string(l),
            [&, l] { h.forward->layer_moe(static_cast<std::uint32_t>(l), call.batch, xm, inject, residual, -1); });
    }
    add("head", [&] { h.forward->finish(call.batch, residual, &logits); });
    Timer marks(sections.size() + 1);
    std::vector<std::vector<double>> times(sections.size());
    for (std::int32_t i = 0; i < o.warmup + o.replays; ++i) {
        h.restore(s);
        marks.mark(0, s);
        for (std::size_t k = 0; k < sections.size(); ++k) {
            sections[k].launch(s);
            marks.mark(k + 1, s);
        }
        check(cudaStreamSynchronize(s), "sections");
        if (i < o.warmup) { continue; }
        for (std::size_t k = 0; k < sections.size(); ++k) { times[k].push_back(marks.us(k, k + 1)); }
    }
    struct Group {
        std::string name;
        double total = 0;
        int count    = 0;
    };
    std::vector<Group> groups{{"embed"}, {"gdn mixer"}, {"qsa mixer"}, {"moe"}, {"head"}};
    double sum = 0;
    for (std::size_t k = 0; k < sections.size(); ++k) {
        const double t = summarize(times[k]).median;
        sum += t;
        const std::string& n = names[k];
        Group& g = n == "embed" ? groups[0] : n.rfind("gdn", 0) == 0 ? groups[1] : n.rfind("qsa", 0) == 0 ? groups[2]
                                          : n.rfind("moe", 0) == 0   ? groups[3] : groups[4];
        g.total += t;
        ++g.count;
        if (o.layers) { std::printf("    %-10s %8.1f us\n", n.c_str(), t); }
    }
    std::printf("  sections (median of each, as separate graphs):\n");
    for (const Group& g : groups) {
        std::printf("    %-10s %8.1f us  %5.1f%%  (%d x %.1f)\n", g.name.c_str(), g.total, 100.0 * g.total / sum, g.count,
                    g.total / std::max(1, g.count));
    }
    std::printf("    %-10s %8.1f us  (the whole graph: %.1f)\n", "sum", sum, f.median);
    return result;
}

// The golden-hash file: one line per configuration, "key = logits-hash state-hash".
std::string hash_key(const Options& o, const Shape& shape) {
    std::ostringstream key;
    key << o.artifact.filename().string() << ' ' << shape.rows << 'x' << shape.width << " context=" << o.context
        << " kv=" << o.kv_name << " experts=" << (o.sample_experts == 0 ? std::string("all") : std::to_string(o.sample_experts))
        << ' ' << (o.exact ? "exact" : "fast") << " accept=" << (o.accept < 0 ? shape.width : std::clamp(o.accept, 0, shape.width));
    return key.str();
}

std::map<std::string, std::string> read_hashes(const std::filesystem::path& path) {
    std::map<std::string, std::string> out;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        const auto eq = line.find(" = ");
        if (line.empty() || line.front() == '#' || eq == std::string::npos) { continue; }
        out[line.substr(0, eq)] = line.substr(eq + 3);
    }
    return out;
}

void write_hashes(const std::filesystem::path& path, const std::map<std::string, std::string>& hashes) {
    std::ofstream file(path);
    file << "# T1 golden hashes (infernix_qwen4_exp_round_bench --record): logits and lane state after one round.\n";
    for (const auto& [key, value] : hashes) { file << key << " = " << value << '\n'; }
    if (!file) { throw std::runtime_error("cannot write " + path.string()); }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const KvCacheStorage storage = parse_kv(o.kv_name);
        // Always every lane the engine runs, prefilled, so the state hash covers the same lanes
        // whatever shapes one run selects.
        const std::int32_t rows = kLanes;
        std::int32_t max_width = 1, max_columns = 0;
        for (const Shape& s : o.shapes) {
            max_width   = std::max(max_width, s.width);
            max_columns = std::max(max_columns, s.rows * s.width);
        }
        const std::int32_t context = o.context + max_width + 1;

        const auto t0 = std::chrono::steady_clock::now();
        DeviceContext device(0);
        artifact::Reader reader(o.artifact);
        if (!q4::is_qwen4_exp(reader)) { throw std::invalid_argument(o.artifact.string() + " is not a Qwen4Exp artifact"); }
        models::LoadOptions load;
        load.stream_experts = true;
        auto model = q4::materialize_model(q4::plan_load(reader, load), device);
        device.synchronize();
        const q4::execution::Parameters parameters(*model);
        const auto& c = model->config().text;
        std::printf("dense weights loaded in %.1f s (%.2f GiB device)\n", seconds_since(t0),
                    model->storage_stats().device_capacity_bytes / 1073741824.0);

        EngineOptions engine;
        engine.fast_paths = !o.exact;
        engine.kv_cache   = storage;
        const q4::FastPathSelection selection = q4::select_fast_paths(parameters, engine);
        std::printf("fast paths:");
        for (const FastPath path : selection.paths) { std::printf(" %s", fast_path_name(path)); }
        std::printf("%s\n", selection.paths.empty() ? " none (exact route)" : "");

        const auto t1 = std::chrono::steady_clock::now();
        const ExpertFrames frames(parameters, *model->expert_store(), o.sample_experts, device);
        std::printf("expert frames filled in %.1f s\n", seconds_since(t1));

        const q4::NgramVolume volume(o.ngram, q4::resolve_ngram_volume(o.ngram, c.ple));
        Harness h(parameters, device, frames, context, std::max(max_columns, kPrefillChunk), rows, max_width, storage,
                  selection.forward(), o.no_fork, o.no_shared_stream);

        // Each row: its own slice of the corpus, prefilled in chunks.
        const auto corpus = read_corpus(INFERNIX_BENCH_CORPUS);
        std::vector<std::vector<std::int32_t>> histories(static_cast<std::size_t>(rows));
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t start = static_cast<std::size_t>(row) * 4099 % corpus.size();
            for (std::int32_t i = 0; i < context; ++i) {
                histories[static_cast<std::size_t>(row)].push_back(corpus[(start + i) % corpus.size()]);
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        DeviceBuffer prefill_logits(c.vocab_size * sizeof(float));
        Tensor last(prefill_logits.p, DType::FP32, {static_cast<std::int32_t>(c.vocab_size), 1});
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t slot[] = {row};
            for (std::int32_t first = 0; first < o.context; first += kPrefillChunk) {
                const std::int32_t n = std::min(kPrefillChunk, o.context - first);
                Call call = make_call(c, volume, histories, slot, first, n, false);
                h.forward->run(call.batch, last);
                device.synchronize();
            }
        }
        h.save(device.stream);
        device.synchronize();
        std::printf("prefilled %d rows x %d tokens in %.1f s\n", rows, o.context, seconds_since(t2));

        const auto golden = o.check.empty() ? std::map<std::string, std::string>{} : read_hashes(o.check);
        auto recorded     = o.record.empty() ? std::map<std::string, std::string>{} : read_hashes(o.record);
        int mismatches = 0, missing = 0;
        for (const Shape& shape : o.shapes) {
            const std::string hashes = run_shape(o, shape, h, device, volume, histories, parameters);
            const std::string key    = hash_key(o, shape);
            if (!o.record.empty()) { recorded[key] = hashes; }
            if (o.check.empty()) { continue; }
            const auto it = golden.find(key);
            if (it == golden.end()) {
                std::printf("  check: no golden hash for \"%s\"\n", key.c_str());
                ++missing;
            } else if (it->second != hashes) {
                std::printf("  check: DIFFERS from the golden %s\n", it->second.c_str());
                ++mismatches;
            } else {
                std::printf("  check: equals the golden hashes\n");
            }
        }
        if (!o.record.empty()) { write_hashes(o.record, recorded); }
        if (mismatches != 0 || missing != 0) {
            std::printf("\ncheck: %d shape(s) differ, %d without a golden hash\n", mismatches, missing);
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
