// The T1 round harness for GLM-5.3-Flash (AGENTS.md "Iteration tiers",
// docs/maintainer/round-bench.md): decode rounds of a chosen shape on the real artifact through
// the production Program and its CUDA Graphs, without the Engine or the server.
//
// GLM's routed experts (159.5 GiB) do not fit in VRAM, so unlike the Qwen3.8-Flash-Next round
// bench this one cannot hold every expert resident and time the Forward alone: it builds the
// family's instance with production's options (the expert placement of --experts over the VRAM,
// RAM and SSD tiers, CPU-served misses), prefills each row's context, then times whole rounds
// (Program::decode and its commit: the Forward graph, the host's expert misses and the sampling).
// A round's time therefore includes the placement's misses, and it reports their hit rate; there
// is no device speed of light (the bytes a round moves over the link and the host's DRAM are not
// a device read).
//
//   infernix_glm5_next_round_bench [--artifact PATH] [--draft PATH] [--shape RxW[,RxW...]]
//       [--context N] [--prefill-chunk N] [--rounds N] [--warm N] [--kv int8|bf16]
//       [--experts SPEC] [--check FILE | --record FILE] [--determinism N | --replays N]
//       [--report FILE] [--profile]
//
// R rows of W columns: W = 1 a plain decode round of R rows; W >= 2 a one-row DFlash2 round
// drafting W - 1 tokens (--draft names the drafter). Each row's context is --context tokens of a
// fixed text (rows start at different offsets), prefilled in calls of --prefill-chunk. --warm
// rounds run first (graph capture, promotions), then --rounds timed rounds. The hash is FNV-1a
// over every row's greedy tokens, from the prefill's first token through the last timed round;
// --check and --record compare with or write bench/models/glm5_next/round_hashes.txt (keyed by
// artifact, shape, context, chunk, KV, experts and rounds). --determinism N repeats each shape N
// times from a fresh binding and fails on differing tokens (--replays is its name in
// make t1-record). --profile brackets the timed rounds with cudaProfilerStart/Stop.

#include "core/device.h"
#include "infernix/types.h"
#include "models/glm5_next/instance.h"
#include "models/registry.h"
#include "runtime/engine/engine_options.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
namespace glm = infernix::models::glm5_next;

namespace {

struct Shape {
    std::uint32_t rows = 1, width = 1;

    [[nodiscard]] std::string name() const {
        return std::to_string(rows) + "x" + std::to_string(width);
    }
};

struct Settings {
    std::filesystem::path artifact =
        std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "") /
        "models/infernix/GLM-5.3-Flash-Uncensored-NVFP4-Infernix.infernix";
    std::filesystem::path draft;
    std::vector<Shape> shapes{{1, 1}, {4, 1}};
    std::uint32_t context = 1024, chunk = 4096, rounds = 64, warm = 16, determinism = 1;
    KvCacheStorage kv   = KvCacheStorage::Int8Group64;
    std::string experts = "vram=40%,ram=60%";
    std::filesystem::path check, record, report;
    bool profile = false;
};

std::vector<Shape> parse_shapes(const std::string& text) {
    std::vector<Shape> out;
    std::stringstream list(text);
    std::string item;
    while (std::getline(list, item, ',')) {
        const auto x = item.find('x');
        if (x == std::string::npos) { throw std::invalid_argument("a shape is RxW: " + item); }
        Shape s{static_cast<std::uint32_t>(std::stoul(item.substr(0, x))),
                static_cast<std::uint32_t>(std::stoul(item.substr(x + 1)))};
        if (s.rows < 1 || s.rows > 8 || s.width < 1 || s.width > 16 ||
            (s.width > 1 && s.rows != 1)) {
            throw std::invalid_argument("shape " + item +
                                        ": 1..8 rows of one column, or one row of 2..16");
        }
        out.push_back(s);
    }
    return out;
}

std::uint64_t fnv(std::uint64_t h, std::int32_t v) {
    for (int b = 0; b < 4; ++b) {
        h ^= static_cast<std::uint64_t>((static_cast<std::uint32_t>(v) >> (8 * b)) & 0xFFU);
        h *= 1099511628211ULL;
    }
    return h;
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

void write_hashes(const std::filesystem::path& path,
                  const std::map<std::string, std::string>& hashes) {
    std::ofstream file(path);
    file << "# T1 golden token hashes (infernix_glm5_next_round_bench --record): FNV-1a of every "
            "row's greedy tokens.\n";
    for (const auto& [key, value] : hashes) { file << key << " = " << value << '\n'; }
    if (!file) { throw std::runtime_error("cannot write " + path.string()); }
}

// The rows' context text: numbered lines of words, tokenized once.
std::vector<TokenId> context_text(const glm::Frontend& frontend) {
    static const char* kWords[] = {"lighthouse", "harbor", "keeper", "lens",   "beam",
                                   "storm",      "coast",  "rock",   "signal", "night",
                                   "ship",       "tide",   "wave",   "fog"};
    std::string text;
    for (int line = 0; line < 1600; ++line) {
        text += std::to_string(line) + ". ";
        for (int w = 0; w < 6; ++w) { text += std::string(kWords[(line * 7 + w * 3) % 14]) + " "; }
        text += "\n";
    }
    return frontend.tokenize_text(text);
}

runtime::ResolvedExecutionOptions greedy(std::uint32_t tokens) {
    runtime::ResolvedExecutionOptions out;
    out.requested_output_tokens = tokens;
    out.allow_prefix_reuse      = false;
    out.sampling.temperature    = 0.0F;
    return out;
}

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5)];
}

struct RunResult {
    std::uint64_t hash = 14695981039346656037ULL;
    std::vector<double> round_us;
    std::uint64_t committed = 0, hits = 0, routed = 0;
};

// Binds every row, prefills its context, runs the warm and timed rounds and finishes the rows.
RunResult run_shape(runtime::Glm5NextInstance& instance, const Settings& settings,
                    const Shape& shape, const std::vector<TokenId>& text) {
    glm::Program& program      = *instance.program;
    const std::uint32_t output = (settings.warm + settings.rounds) * shape.width + 2;
    RunResult out;
    std::vector<glm::SequenceHandle> rows;
    for (std::uint32_t r = 0; r < shape.rows; ++r) {
        std::vector<TokenId> prompt;
        const std::size_t start = static_cast<std::size_t>(r) * 4099 % text.size();
        for (std::uint32_t i = 0; i < settings.context; ++i) {
            prompt.push_back(text[(start + i) % text.size()]);
        }
        glm::RequestBasePlan base =
            program.plan_request(instance.frontend.prepare_tokens(prompt, false), greedy(output));
        const auto sources = program.hybrid_sources(base, UINT32_MAX);
        if (!program.start_binding(base, runtime::LaneId{r}, sources.front()).reserved) {
            throw std::runtime_error("the binding of row " + std::to_string(r) + " was refused");
        }
        const glm::ContextProgress bound = program.poll_context({});
        if (!bound.sequence) { throw std::runtime_error("a binding did not complete"); }
        for (;;) {
            glm::PrefillProgress progress = program.advance_prefill(*bound.sequence);
            if (!progress.complete) { continue; }
            out.hash                               = fnv(out.hash, progress.pending->tokens()[0]);
            const runtime::CommitDecision commit[] = {{.accepted_tokens = 1}};
            (void)program.commit(std::move(*progress.pending), commit);
            break;
        }
        rows.push_back(*bound.sequence);
    }
    std::vector<runtime::RoundBudget> budgets(shape.rows);
    std::uint32_t produced = 1;
    for (std::uint32_t round = 0; round < settings.warm + settings.rounds; ++round) {
        const bool timed = round >= settings.warm;
        if (timed && round == settings.warm && settings.profile) { (void)cudaProfilerStart(); }
        for (auto& b : budgets) { b = {output - produced}; }
        const auto t0             = std::chrono::steady_clock::now();
        glm::PendingBatch pending = program.decode(rows, budgets);
        std::vector<runtime::CommitDecision> commit(shape.rows);
        std::uint32_t most = 0;
        for (std::uint32_t r = 0; r < shape.rows; ++r) {
            // A plain round (one token a row) carries no row counts.
            const std::uint32_t n = pending.row_counts().empty()
                                        ? 1U
                                        : static_cast<std::uint32_t>(pending.row_counts()[r]);
            for (std::uint32_t i = 0; i < n; ++i) {
                out.hash = fnv(out.hash, pending.tokens()[r * pending.row_stride() + i]);
            }
            commit[r].accepted_tokens = n;
            commit[r].terminal        = round + 1 == settings.warm + settings.rounds;
            most                      = std::max(most, n);
            if (timed) { out.committed += n; }
        }
        (void)program.commit(std::move(pending), commit);
        if (timed) {
            out.round_us.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count());
        }
        produced += most;
    }
    if (settings.profile) { (void)cudaProfilerStop(); }
    for (const glm::SequenceHandle& row : rows) {
        const glm::FinishResult finished = program.finish(row);
        if (finished.expert_cache) {
            out.hits += finished.expert_cache->hits;
            out.routed += finished.expert_cache->routed;
        }
    }
    return out;
}

EngineOptions engine_options(const Settings& settings, std::uint32_t lanes, std::uint32_t drafts) {
    EngineOptions options;
    options.artifact_path   = settings.artifact;
    options.kv_cache        = settings.kv;
    options.max_concurrency = lanes;
    options.max_context =
        (settings.context + (settings.warm + settings.rounds) * 16 + 64 + 4095) / 4096 * 4096;
    options.kv_capacity           = KvCapacityPolicy::automatic();
    options.prefill_chunk         = settings.chunk;
    options.experts_spec          = settings.experts;
    options.context_cache.enabled = false;
    if (drafts > 0) {
        options.speculative.backend      = SpeculativeBackend::DFlash2;
        options.speculative.draft_model  = settings.draft;
        options.speculative.draft_tokens = drafts;
    }
    options.diagnostic_observer.callback = [](const Diagnostic& d) {
        if (d.level >= DiagnosticLevel::Warning || d.message.find("VRAM:") != std::string::npos ||
            d.message.find("expert placement:") != std::string::npos) {
            std::printf("  [engine] %s\n", d.message.c_str());
        }
    };
    const runtime::FamilyOptionRules& family =
        models::family_option_rules(models::Architecture::Glm5Next);
    runtime::validate_engine_options(options, family);
    return runtime::normalize_engine_options(std::move(options), family);
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Settings settings;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--profile") {
                settings.profile = true;
                continue;
            }
            if (arg == "--exact") {
                throw std::invalid_argument("GLM-5.3-Flash has no exact route");
            }
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            const std::string value = argv[++i];
            if (arg == "--artifact") {
                settings.artifact = value;
            } else if (arg == "--draft") {
                settings.draft = value;
            } else if (arg == "--shape") {
                settings.shapes = parse_shapes(value);
            } else if (arg == "--context") {
                settings.context = static_cast<std::uint32_t>(std::stoul(value));
            } else if (arg == "--prefill-chunk") {
                settings.chunk = static_cast<std::uint32_t>(std::stoul(value));
            } else if (arg == "--rounds") {
                settings.rounds = static_cast<std::uint32_t>(std::stoul(value));
            } else if (arg == "--warm") {
                settings.warm = static_cast<std::uint32_t>(std::stoul(value));
            } else if (arg == "--determinism" || arg == "--replays") {
                settings.determinism = std::max(1U, static_cast<std::uint32_t>(std::stoul(value)));
            } else if (arg == "--kv") {
                if (value != "int8" && value != "bf16") {
                    throw std::invalid_argument("--kv int8|bf16");
                }
                settings.kv =
                    value == "int8" ? KvCacheStorage::Int8Group64 : KvCacheStorage::BFloat16;
            } else if (arg == "--experts") {
                settings.experts = value;
            } else if (arg == "--check") {
                settings.check = value;
            } else if (arg == "--record") {
                settings.record = value;
            } else if (arg == "--report") {
                settings.report = value;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (settings.context < 1 || settings.rounds < 1) {
            throw std::invalid_argument("--context and --rounds must be at least 1");
        }
        for (const Shape& s : settings.shapes) {
            if (s.width > 1 && settings.draft.empty()) {
                throw std::invalid_argument("shape " + s.name() +
                                            " drafts: --draft names the drafter");
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    try {
        DeviceContext device(0);
        const auto golden = settings.check.empty() ? std::map<std::string, std::string>{}
                                                   : read_hashes(settings.check);
        auto recorded     = settings.record.empty() ? std::map<std::string, std::string>{}
                                                    : read_hashes(settings.record);
        int failures      = 0;
        // One instance per drafter setting: plain rounds share one with as many lanes as the
        // widest plain shape; each drafting width gets its own (the draft length is fixed at
        // startup).
        std::map<std::uint32_t, std::vector<Shape>> groups;
        for (const Shape& s : settings.shapes) { groups[s.width].push_back(s); }
        for (const auto& [width, shapes] : groups) {
            std::uint32_t lanes = 1;
            for (const Shape& s : shapes) { lanes = std::max(lanes, s.rows); }
            const auto t0 = std::chrono::steady_clock::now();
            auto instance = runtime::construct_glm5_next_instance(
                engine_options(settings, lanes, width > 1 ? width - 1 : 0), device);
            std::printf(
                "instance: %u lanes%s, started in %.1f s\n", lanes,
                width > 1 ? (", drafting " + std::to_string(width - 1)).c_str() : "",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            const auto text = context_text(instance->frontend);
            for (const Shape& shape : shapes) {
                std::vector<RunResult> runs;
                for (std::uint32_t d = 0; d < settings.determinism; ++d) {
                    runs.push_back(run_shape(*instance, settings, shape, text));
                }
                const RunResult& r  = runs.front();
                const double median = percentile(r.round_us, 0.5);
                const double accept = static_cast<double>(r.committed) /
                                      (static_cast<double>(settings.rounds) * shape.rows);
                std::printf("shape %s (context %u, chunk %u, %s KV, experts %s)\n",
                            shape.name().c_str(), settings.context, settings.chunk,
                            settings.kv == KvCacheStorage::Int8Group64 ? "int8" : "bf16",
                            settings.experts.c_str());
                std::printf("  round median %.1f us (p10 %.1f, p90 %.1f) over %u rounds\n", median,
                            percentile(r.round_us, 0.1), percentile(r.round_us, 0.9),
                            settings.rounds);
                std::printf("  %.2f tokens a row a round: %.1f tok/s a row, %.1f in aggregate\n",
                            accept, accept / (median * 1e-6),
                            accept * shape.rows / (median * 1e-6));
                std::printf("  expert cache: %.1f %% of routed experts in VRAM\n",
                            r.routed ? 100.0 * static_cast<double>(r.hits) /
                                           static_cast<double>(r.routed)
                                     : 0.0);
                char hash[32];
                std::snprintf(hash, sizeof hash, "%016llx",
                              static_cast<unsigned long long>(r.hash));
                std::printf("  hash %s\n", hash);
                for (std::size_t d = 1; d < runs.size(); ++d) {
                    if (runs[d].hash != r.hash) {
                        ++failures;
                        std::printf("  FAIL: replay %zu of %s generated different tokens\n", d,
                                    shape.name().c_str());
                    }
                }
                const std::string key =
                    settings.artifact.filename().string() + " shape=" + shape.name() +
                    " context=" + std::to_string(settings.context) +
                    " chunk=" + std::to_string(settings.chunk) +
                    " kv=" + (settings.kv == KvCacheStorage::Int8Group64 ? "int8" : "bf16") +
                    " experts=" + settings.experts + " rounds=" + std::to_string(settings.warm) +
                    "+" + std::to_string(settings.rounds);
                if (!settings.check.empty()) {
                    const auto it = golden.find(key);
                    const bool ok = it != golden.end() && it->second == hash;
                    failures += ok ? 0 : 1;
                    std::printf("  %s golden hash (%s)\n",
                                ok ? "matches its" : "FAIL: differs from its",
                                it == golden.end() ? "none recorded" : it->second.c_str());
                }
                if (!settings.record.empty()) { recorded[key] = hash; }
                if (!settings.report.empty()) {
                    std::ofstream file(settings.report, std::ios::app);
                    file << "{\"bench\":\"glm5_next\",\"shape\":\"" << shape.name()
                         << "\",\"rows\":" << shape.rows << ",\"width\":" << shape.width
                         << ",\"prefill\":0,\"contexts\":[" << settings.context << "],\"kv\":\""
                         << (settings.kv == KvCacheStorage::Int8Group64 ? "int8" : "bf16")
                         << "\",\"route\":\"default\",\"accept\":" << accept
                         << ",\"round_us\":" << median << ",\"expert_hit\":"
                         << (r.routed ? static_cast<double>(r.hits) / static_cast<double>(r.routed)
                                      : 0.0)
                         << "}\n";
                }
            }
        }
        if (!settings.record.empty()) {
            write_hashes(settings.record, recorded);
            std::printf("recorded %zu shapes into %s\n", settings.shapes.size(),
                        settings.record.c_str());
        }
        std::printf(failures == 0 ? "PASS\n" : "FAIL: %d checks\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
